"""
Encoder Pre-training Script for CQL

This script pre-trains the encoder of a CQL model using a custom loss function
on trajectories sampled from a combined MDP dataset. The pre-trained encoder
can later be loaded to train policies offline with a different dataset.

Usage:
    1. Define your custom loss function in `compute_encoder_loss`
    2. Run the pre-training loop
    3. Save the pre-trained encoder weights
    4. Load the encoder weights when training CQL on another dataset
"""

import os
import sys
from pathlib import Path
from typing import Callable, Optional, Dict, Any
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import d3rlpy
from d3rlpy.dataset import MDPDataset, TrajectoryMiniBatch
from d3rlpy.preprocessing import PixelObservationScaler, ClipRewardScaler
from d3rlpy.models.encoders import PixelEncoderFactory

# Add parent directory to path for imports
Path_root = Path(os.path.dirname(os.path.abspath(__file__))).parent
sys.path.insert(0, str(Path_root))

from afr_scripts.extended_dataset import CombinedMDPDataset, TrajectoryBatchWithSource


@dataclass
class EncoderPretrainConfig:
    """Configuration for encoder pre-training."""
    # Training
    learning_rate: float = 1e-4
    batch_size: int = 32
    trajectory_length: int = 10
    n_steps: int = 100000
    
    # Logging
    log_interval: int = 100
    save_interval: int = 10000
    log_dir: str = "encoder_pretrain_logs"
    
    # Device
    device: str = "cuda:0"


class EncoderPretrainer:
    """
    Pre-trains CQL encoders using a custom loss function on trajectories
    from a combined MDP dataset.
    
    The encoder is extracted from the CQL's Q-function network.
    For DiscreteCQL, the Q-function has an encoder that processes observations
    before the final Q-value head.
    
    Example:
        >>> pretrainer = EncoderPretrainer(
        ...     cql=cql,
        ...     combined_dataset=combined,
        ...     config=config,
        ...     loss_fn=my_custom_loss_fn,
        ... )
        >>> pretrainer.pretrain()
        >>> pretrainer.save_encoder_weights("encoder_weights.pt")
    """
    
    def __init__(
        self,
        cql: d3rlpy.algos.DiscreteCQL,
        combined_dataset: CombinedMDPDataset,
        config: EncoderPretrainConfig,
        loss_fn: Optional[Callable] = None,
    ):
        """
        Args:
            cql: A built DiscreteCQL model.
            combined_dataset: Combined MDP dataset for sampling trajectories.
            config: Pre-training configuration.
            loss_fn: Custom loss function. Should accept (encoder, trajectory_batch, source_ids)
                     and return a loss tensor. If None, uses a placeholder.
        """
        self.cql = cql
        self.combined_dataset = combined_dataset
        self.config = config
        self.device = config.device
        
        # Extract encoders from Q-functions
        # The CQL implementation stores Q-functions in _impl._modules.q_funcs
        self.encoders = self._extract_encoders()
        
        # Set up optimizer for encoder parameters
        encoder_params = []
        for encoder in self.encoders:
            encoder_params.extend(encoder.parameters())
        self.optimizer = optim.Adam(encoder_params, lr=config.learning_rate)
        
        # Loss function (user-defined or placeholder)
        self.loss_fn = loss_fn or self._placeholder_loss
        
        # Logging
        self.writer = SummaryWriter(config.log_dir)
        
    def _extract_encoders(self) -> list:
        """Extract encoder modules from CQL's Q-functions."""
        if self.cql._impl is None:
            raise RuntimeError("CQL must be built before extracting encoders. "
                             "Call cql.build_with_dataset(dataset) first.")
        
        q_funcs = self.cql._impl._modules.q_funcs
        encoders = []
        for q_func in q_funcs:
            # Each q_func has an encoder property
            encoders.append(q_func.encoder)
        return encoders
    
    def get_encoder(self, index: int = 0) -> nn.Module:
        """Get a specific encoder by index."""
        return self.encoders[index]
    
    def get_encoder_feature_size(self) -> int:
        """Get the output feature size of the encoder."""
        # Run a forward pass to determine output size
        encoder = self.encoders[0]
        with torch.no_grad():
            # Create dummy input matching observation shape
            obs_shape = self.cql._impl.observation_shape
            dummy_input = torch.zeros(1, *obs_shape, device=self.device)
            output = encoder(dummy_input)
        return output.shape[-1]
    
    def _placeholder_loss(
        self,
        encoder: nn.Module,
        trajectory_batch: TrajectoryMiniBatch,
        source_ids: np.ndarray,
    ) -> torch.Tensor:
        """
        Placeholder loss function. Replace with your custom loss.
        
        This example computes a simple reconstruction-style loss, but you should
        define your own loss based on your pre-training objective.
        
        Args:
            encoder: The encoder module to train.
            trajectory_batch: Batch of trajectories containing:
                - observations: (batch_size, trajectory_length, *obs_shape)
                - actions: (batch_size, trajectory_length)
                - rewards: (batch_size, trajectory_length)
                - returns_to_go: (batch_size, trajectory_length)
                - terminals: (batch_size, trajectory_length)
                - timesteps: (batch_size, trajectory_length)
            source_ids: Array indicating which dataset each trajectory came from.
        
        Returns:
            Loss tensor.
        """
        # Example: encode observations and compute a dummy contrastive loss
        # TODO: Replace with your actual pre-training loss
        
        observations = trajectory_batch.observations  # (batch, seq, *obs_shape)
        batch_size, seq_len = observations.shape[:2]
        obs_shape = observations.shape[2:]
        
        # Flatten batch and sequence dimensions for encoding
        flat_obs = observations.reshape(-1, *obs_shape)  # (batch * seq, *obs_shape)
        
        # Move to device and normalize if using pixel observations
        flat_obs = flat_obs.to(self.device).float()
        
        # Encode observations
        features = encoder(flat_obs)  # (batch * seq, feature_size)
        
        # Reshape back to (batch, seq, feature_size)
        features = features.reshape(batch_size, seq_len, -1)
        
        # Placeholder: use temporal consistency loss
        # Minimize distance between adjacent time steps
        if seq_len > 1:
            loss = torch.mean((features[:, 1:] - features[:, :-1]) ** 2)
        else:
            loss = torch.tensor(0.0, device=self.device)
        
        return loss
    
    def _prepare_trajectory_batch(
        self, batch_with_source: TrajectoryBatchWithSource
    ) -> tuple:
        """Prepare trajectory batch for training."""
        trajectory_batch = batch_with_source.batch
        source_ids = batch_with_source.source_ids
        return trajectory_batch, source_ids
    
    def pretrain_step(self) -> Dict[str, float]:
        """Execute a single pre-training step."""
        # Sample trajectory batch from combined dataset
        batch_with_source = self.combined_dataset.sample_trajectory_batch(
            batch_size=self.config.batch_size,
            length=self.config.trajectory_length,
        )
        trajectory_batch, source_ids = self._prepare_trajectory_batch(batch_with_source)
        
        # Compute loss for each encoder
        total_loss = 0.0
        for encoder in self.encoders:
            encoder.train()
            loss = self.loss_fn(encoder, trajectory_batch, source_ids)
            total_loss = total_loss + loss
        
        # Average over encoders
        avg_loss = total_loss / len(self.encoders)
        
        # Backward pass
        self.optimizer.zero_grad()
        avg_loss.backward()
        self.optimizer.step()
        
        return {"loss": avg_loss.item()}
    
    def pretrain(self) -> None:
        """Run the full pre-training loop."""
        print(f"Starting encoder pre-training for {self.config.n_steps} steps...")
        print(f"Batch size: {self.config.batch_size}, Trajectory length: {self.config.trajectory_length}")
        print(f"Number of encoders: {len(self.encoders)}")
        print(f"Encoder feature size: {self.get_encoder_feature_size()}")
        print(f"Logging to: {self.config.log_dir}")
        print("-" * 50)
        
        for step in range(1, self.config.n_steps + 1):
            metrics = self.pretrain_step()
            
            # Log metrics
            for name, value in metrics.items():
                self.writer.add_scalar(f"pretrain/{name}", value, step)
            
            # Print progress
            if step % self.config.log_interval == 0:
                loss_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
                print(f"Step {step}/{self.config.n_steps} - {loss_str}")
            
            # Save checkpoint
            if step % self.config.save_interval == 0:
                checkpoint_path = os.path.join(
                    self.config.log_dir, f"encoder_checkpoint_{step}.pt"
                )
                self.save_encoder_weights(checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
        
        print("Pre-training complete!")
        self.writer.close()
    
    def save_encoder_weights(self, path: str) -> None:
        """Save encoder weights to a file."""
        state_dict = {}
        for i, encoder in enumerate(self.encoders):
            state_dict[f"encoder_{i}"] = encoder.state_dict()
        torch.save(state_dict, path)
        print(f"Saved encoder weights to {path}")
    
    def load_encoder_weights(self, path: str) -> None:
        """Load encoder weights from a file."""
        state_dict = torch.load(path, map_location=self.device)
        for i, encoder in enumerate(self.encoders):
            encoder.load_state_dict(state_dict[f"encoder_{i}"])
        print(f"Loaded encoder weights from {path}")


def load_pretrained_encoder_to_cql(
    cql: d3rlpy.algos.DiscreteCQL,
    encoder_weights_path: str,
    device: str = "cuda:0",
) -> None:
    """
    Load pre-trained encoder weights into a CQL model.
    
    This function loads encoder weights saved during pre-training into
    a newly created CQL model, allowing you to train the policy with
    a pre-trained encoder.
    
    Args:
        cql: A built DiscreteCQL model.
        encoder_weights_path: Path to saved encoder weights.
        device: Device to load weights to.
    
    Example:
        >>> cql = d3rlpy.algos.DiscreteCQLConfig(...).create(device='cuda:0')
        >>> cql.build_with_dataset(new_dataset)
        >>> load_pretrained_encoder_to_cql(cql, "encoder_weights.pt")
        >>> cql.fit(new_dataset, n_steps=100000)
    """
    if cql._impl is None:
        raise RuntimeError("CQL must be built before loading encoder weights. "
                         "Call cql.build_with_dataset(dataset) first.")
    
    state_dict = torch.load(encoder_weights_path, map_location=device)
    q_funcs = cql._impl._modules.q_funcs
    
    for i, q_func in enumerate(q_funcs):
        encoder_key = f"encoder_{i}"
        if encoder_key in state_dict:
            q_func.encoder.load_state_dict(state_dict[encoder_key])
            print(f"Loaded weights for Q-function {i} encoder")
    
    # Also load into target Q-functions for consistency
    targ_q_funcs = cql._impl._modules.targ_q_funcs
    for i, targ_q_func in enumerate(targ_q_funcs):
        encoder_key = f"encoder_{i}"
        if encoder_key in state_dict:
            targ_q_func.encoder.load_state_dict(state_dict[encoder_key])
            print(f"Loaded weights for target Q-function {i} encoder")


# =============================================================================
# Example custom loss functions
# =============================================================================

def contrastive_temporal_loss(
    encoder: nn.Module,
    trajectory_batch: TrajectoryMiniBatch,
    source_ids: np.ndarray,
    device: str = "cuda:0",
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Example: Temporal contrastive loss.
    
    Encourages representations of adjacent timesteps to be similar,
    and representations from different trajectories to be different.
    """
    observations = trajectory_batch.observations
    batch_size, seq_len = observations.shape[:2]
    obs_shape = observations.shape[2:]
    
    # Flatten and encode
    flat_obs = observations.reshape(-1, *obs_shape).to(device).float()
    features = encoder(flat_obs)  # (batch * seq, feature_dim)
    features = features.reshape(batch_size, seq_len, -1)
    
    # Normalize features
    features = nn.functional.normalize(features, dim=-1)
    
    if seq_len < 2:
        return torch.tensor(0.0, device=device)
    
    # Positive pairs: adjacent timesteps
    anchors = features[:, :-1].reshape(-1, features.shape[-1])
    positives = features[:, 1:].reshape(-1, features.shape[-1])
    
    # Similarity matrix
    pos_sim = torch.sum(anchors * positives, dim=-1) / temperature
    
    # Negative samples: other trajectories at same timestep
    # For simplicity, use all other features as negatives
    all_features = features.reshape(-1, features.shape[-1])
    neg_sim = torch.mm(anchors, all_features.t()) / temperature
    
    # InfoNCE loss
    logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=device)
    loss = nn.functional.cross_entropy(logits, labels)
    
    return loss


def source_prediction_loss(
    encoder: nn.Module,
    trajectory_batch: TrajectoryMiniBatch,
    source_ids: np.ndarray,
    device: str = "cuda:0",
    num_sources: int = 2,
) -> torch.Tensor:
    """
    Example: Source dataset prediction loss.
    
    Trains the encoder to predict which dataset a trajectory came from.
    This can help learn representations that capture policy-specific features.
    """
    observations = trajectory_batch.observations
    batch_size, seq_len = observations.shape[:2]
    obs_shape = observations.shape[2:]
    
    # Encode first observation of each trajectory
    first_obs = observations[:, 0].to(device).float()
    features = encoder(first_obs)  # (batch, feature_dim)
    
    # Simple linear classifier for source prediction
    # In practice, you'd want to maintain this as a separate module
    if not hasattr(encoder, '_source_classifier'):
        encoder._source_classifier = nn.Linear(
            features.shape[-1], num_sources
        ).to(device)
    
    logits = encoder._source_classifier(features)
    labels = torch.tensor(source_ids, dtype=torch.long, device=device)
    
    loss = nn.functional.cross_entropy(logits, labels)
    return loss


# =============================================================================
# Main script
# =============================================================================

if __name__ == "__main__":
    import gymnasium as gym
    import ale_py
    from gymnasium.wrappers import AtariPreprocessing, ResizeObservation, FrameStackObservation
    
    def make_env(env_id: str):
        env = gym.make(env_id)
        env = AtariPreprocessing(env, screen_size=84, frame_skip=4, 
                                 terminal_on_life_loss=False, grayscale_obs=True, noop_max=30)
        env = ResizeObservation(env, (84, 84))
        env = FrameStackObservation(env, 4)
        return env
    
    # Configuration
    data_paths = [
        '/u/mrudolph/documents/rl-baselines3-zoo/atari_data/a2c_QbertNoFrameskip-v4_0_100000.pth',
        '/u/mrudolph/documents/rl-baselines3-zoo/atari_data/dqn_QbertNoFrameskip-v4_0_100000.pth',
    ]
    env_id = "QbertNoFrameskip-v4"
    log_dir = 'encoder_pretrain_qbert'
    device = 'cuda:0'
    
    # Create environment
    env = make_env(env_id)
    
    # Load datasets
    print("Loading datasets...")
    datasets = []
    for data_path in data_paths:
        if os.path.exists(data_path):
            data = torch.load(data_path, weights_only=False)
            dataset = d3rlpy.dataset.MDPDataset(
                observations=data['obs'],
                actions=data['action'],
                rewards=data['reward'],
                terminals=data['done'],
                action_space=d3rlpy.constants.ActionSpace.DISCRETE,
                action_size=env.action_space.n,
            )
            datasets.append(dataset)
            print(f"  Loaded {data_path}")
        else:
            print(f"  Warning: {data_path} not found, skipping...")
    
    if len(datasets) == 0:
        print("No datasets found! Exiting.")
        sys.exit(1)
    
    # Create combined dataset
    combined = CombinedMDPDataset(
        datasets=datasets,
        names=["a2c", "dqn"][:len(datasets)],
    )
    print(f"\n{combined}\n")
    
    # Create CQL model
    print("Creating CQL model...")
    cql = d3rlpy.algos.DiscreteCQLConfig(
        observation_scaler=PixelObservationScaler(),
        reward_scaler=ClipRewardScaler(-1.0, 1.0),
        compile_graph=False,
    ).create(device=device)
    
    # Build the model with one of the datasets
    cql.build_with_dataset(datasets[0])
    print("CQL model built.")
    
    # Create pre-trainer
    config = EncoderPretrainConfig(
        learning_rate=1e-4,
        batch_size=32,
        trajectory_length=10,
        n_steps=10000,  # Reduced for demonstration
        log_interval=100,
        save_interval=2000,
        log_dir=log_dir,
        device=device,
    )
    
    # Define your custom loss function here
    # For now, using the placeholder (temporal consistency)
    def my_custom_loss(encoder, trajectory_batch, source_ids):
        """
        TODO: Define your custom loss function here.
        
        This function receives:
        - encoder: The neural network encoder to train
        - trajectory_batch: TrajectoryMiniBatch with observations, actions, rewards, etc.
        - source_ids: Array indicating which dataset each trajectory came from
        
        Return a scalar loss tensor.
        """
        # Example: use contrastive temporal loss
        return contrastive_temporal_loss(
            encoder, trajectory_batch, source_ids, 
            device=config.device, temperature=0.1
        )
    
    pretrainer = EncoderPretrainer(
        cql=cql,
        combined_dataset=combined,
        config=config,
        loss_fn=my_custom_loss,
    )
    
    print(f"\nEncoder architecture:")
    print(pretrainer.get_encoder(0))
    print(f"\nEncoder feature size: {pretrainer.get_encoder_feature_size()}")
    
    # Run pre-training
    print("\n" + "=" * 50)
    print("Starting pre-training...")
    print("=" * 50 + "\n")
    
    pretrainer.pretrain()
    
    # Save final encoder weights
    final_weights_path = os.path.join(log_dir, "encoder_final.pt")
    pretrainer.save_encoder_weights(final_weights_path)
    
    print("\n" + "=" * 50)
    print("Pre-training complete!")
    print(f"Final weights saved to: {final_weights_path}")
    print("=" * 50)
    
    # Example: How to use pre-trained encoder for policy training
    print("\n--- Example: Loading pre-trained encoder for policy training ---")
    print("""
    # Create a new CQL for policy training with a different dataset
    policy_cql = d3rlpy.algos.DiscreteCQLConfig(
        observation_scaler=PixelObservationScaler(),
        reward_scaler=ClipRewardScaler(-1.0, 1.0),
        compile_graph=False,
    ).create(device='cuda:0')
    
    # Build with the target dataset
    policy_cql.build_with_dataset(target_dataset)
    
    # Load pre-trained encoder weights
    load_pretrained_encoder_to_cql(policy_cql, "encoder_final.pt")
    
    # Optional: Freeze encoder weights during policy training
    for q_func in policy_cql._impl._modules.q_funcs:
        for param in q_func.encoder.parameters():
            param.requires_grad = False
    
    # Train policy
    policy_cql.fit(target_dataset, n_steps=100000)
    """)
