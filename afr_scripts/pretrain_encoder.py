"""
CARDPOL: Contrastive Actor Recognition for Diverse POLicies

This script pre-trains the encoder of a CQL model using CARDPOL loss on
trajectories sampled from a combined MDP dataset containing data from
multiple source policies. The pre-trained encoder can later be loaded 
to train policies offline with a different dataset.

CARDPOL learns encoder representations by:
1. Sampling pairs of observations from each trajectory (t=0 and random t)
2. Encoding both observations
3. Using a classifier to predict which source policy generated the trajectory

This encourages the encoder to learn representations that capture
policy-specific temporal dynamics.

Usage:
    1. Load multiple datasets from different source policies
    2. Create a CombinedMDPDataset
    3. Run the CARDPOL pre-training loop
    4. Save and load the pre-trained encoder for offline policy training
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


# =============================================================================
# CARDPOL Classifier
# =============================================================================

class CARDPOLClassifier(nn.Module):
    """
    CARDPOL (Contrastive Actor Recognition for Diverse POLicies) classifier.
    
    A classifier network that takes two embeddings from the encoder and
    outputs logits representing source_id (policy) probabilities.
    
    The classifier can combine the embeddings in various ways:
    - concatenation: [emb1, emb2]
    - difference: emb1 - emb2
    - concatenation + difference: [emb1, emb2, emb1 - emb2]
    
    Args:
        embedding_size: Size of each embedding from the encoder.
        num_sources: Number of source datasets/policies to classify.
        hidden_sizes: List of hidden layer sizes.
        combine_mode: How to combine the two embeddings ('concat', 'diff', 'concat_diff').
    """
    
    def __init__(
        self,
        embedding_size: int,
        num_sources: int,
        hidden_sizes: list = None,
        combine_mode: str = "concat",
    ):
        super().__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [256, 128]
        
        self.combine_mode = combine_mode
        self.embedding_size = embedding_size
        self.num_sources = num_sources
        
        # Compute input size based on combine mode
        if combine_mode == "concat":
            input_size = embedding_size * 2
        elif combine_mode == "diff":
            input_size = embedding_size
        elif combine_mode == "concat_diff":
            input_size = embedding_size * 3
        else:
            raise ValueError(f"Unknown combine_mode: {combine_mode}")
        
        # Build MLP layers
        layers = []
        in_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            in_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(in_size, num_sources))
        
        self.network = nn.Sequential(*layers)
    
    def forward(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the classifier.
        
        Args:
            embedding1: First embedding, shape (batch_size, embedding_size)
            embedding2: Second embedding, shape (batch_size, embedding_size)
        
        Returns:
            Logits of shape (batch_size, num_sources)
        """
        if self.combine_mode == "concat":
            combined = torch.cat([embedding1, embedding2], dim=-1)
        elif self.combine_mode == "diff":
            combined = embedding1 - embedding2
        elif self.combine_mode == "concat_diff":
            combined = torch.cat([
                embedding1,
                embedding2,
                embedding1 - embedding2
            ], dim=-1)
        else:
            raise ValueError(f"Unknown combine_mode: {self.combine_mode}")
        
        return self.network(combined)
    
    def predict_proba(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor,
    ) -> torch.Tensor:
        """Get probability predictions."""
        logits = self.forward(embedding1, embedding2)
        return torch.softmax(logits, dim=-1)
    
    def predict(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor,
    ) -> torch.Tensor:
        """Get class predictions."""
        logits = self.forward(embedding1, embedding2)
        return torch.argmax(logits, dim=-1)


@dataclass
class EncoderPretrainConfig:
    """Configuration for encoder pre-training."""
    # Training
    learning_rate: float = 1e-4
    classifier_learning_rate: float = 1e-4
    batch_size: int = 32
    trajectory_length: int = 10
    n_steps: int = 100000
    
    # Classifier configuration
    num_sources: int = 2
    classifier_hidden_sizes: list = None  # Default: [256, 128]
    classifier_combine_mode: str = "concat"  # 'concat', 'diff', or 'concat_diff'
    
    # Logging
    log_interval: int = 100
    save_interval: int = 10000
    log_dir: str = "encoder_pretrain_logs"
    
    # Device
    device: str = "cuda:0"
    
    def __post_init__(self):
        if self.classifier_hidden_sizes is None:
            self.classifier_hidden_sizes = [256, 128]


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
        classifier: Optional[CARDPOLClassifier] = None,
    ):
        """
        Args:
            cql: A built DiscreteCQL model.
            combined_dataset: Combined MDP dataset for sampling trajectories.
            config: Pre-training configuration.
            loss_fn: Custom loss function. Should accept 
                     (encoder, classifier, trajectory_batch, source_ids, device)
                     and return a loss tensor. If None, uses cardpol_loss.
            classifier: Optional pre-initialized CARDPOLClassifier.
                       If None, one will be created based on config.
        """
        self.cql = cql
        self.combined_dataset = combined_dataset
        self.config = config
        self.device = config.device
        
        # Extract encoders from Q-functions
        # The CQL implementation stores Q-functions in _impl._modules.q_funcs
        self.encoders = self._extract_encoders()
        
        # Get encoder feature size
        encoder_feature_size = self._compute_encoder_feature_size()
        
        # Create or use provided classifier
        if classifier is not None:
            self.classifier = classifier.to(self.device)
        else:
            self.classifier = CARDPOLClassifier(
                embedding_size=encoder_feature_size,
                num_sources=config.num_sources,
                hidden_sizes=config.classifier_hidden_sizes,
                combine_mode=config.classifier_combine_mode,
            ).to(self.device)
        
        # Set up optimizer for encoder parameters
        encoder_params = []
        for encoder in self.encoders:
            encoder_params.extend(encoder.parameters())
        self.encoder_optimizer = optim.Adam(encoder_params, lr=config.learning_rate)
        
        # Set up optimizer for classifier parameters
        self.classifier_optimizer = optim.Adam(
            self.classifier.parameters(), 
            lr=config.classifier_learning_rate
        )
        
        # Loss function (user-defined or default pairwise source prediction)
        self.loss_fn = loss_fn or cardpol_loss
        
        # Logging
        self.writer = SummaryWriter(config.log_dir)
    
    def _compute_encoder_feature_size(self) -> int:
        """Compute the output feature size of the encoder."""
        encoder = self.encoders[0]
        with torch.no_grad():
            obs_shape = self.cql._impl.observation_shape
            dummy_input = torch.zeros(1, *obs_shape, device=self.device)
            output = encoder(dummy_input)
        return output.shape[-1]
        
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
        all_metrics = {}
        
        for encoder in self.encoders:
            encoder.train()
        self.classifier.train()
        
        # Use the first encoder for the loss (they share the same architecture)
        # You could also average across encoders if needed
        loss, metrics = self.loss_fn(
            encoder=self.encoders[0],
            classifier=self.classifier,
            trajectory_batch=trajectory_batch,
            source_ids=source_ids,
            device=self.device,
        )
        
        # If using multiple encoders, compute loss for each and average
        if len(self.encoders) > 1:
            for encoder in self.encoders[1:]:
                additional_loss, _ = self.loss_fn(
                    encoder=encoder,
                    classifier=self.classifier,
                    trajectory_batch=trajectory_batch,
                    source_ids=source_ids,
                    device=self.device,
                )
                loss = loss + additional_loss
            loss = loss / len(self.encoders)
        
        # Backward pass
        self.encoder_optimizer.zero_grad()
        self.classifier_optimizer.zero_grad()
        loss.backward()
        self.encoder_optimizer.step()
        self.classifier_optimizer.step()
        
        # Collect metrics
        all_metrics["loss"] = loss.item()
        all_metrics.update(metrics)
        
        return all_metrics
    
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
    
    def save_encoder_weights(self, path: str, include_classifier: bool = True) -> None:
        """
        Save encoder and classifier weights to a file.
        
        Args:
            path: Path to save the weights.
            include_classifier: Whether to include classifier weights.
        """
        state_dict = {}
        for i, encoder in enumerate(self.encoders):
            state_dict[f"encoder_{i}"] = encoder.state_dict()
        
        if include_classifier:
            state_dict["classifier"] = self.classifier.state_dict()
            state_dict["classifier_config"] = {
                "embedding_size": self.classifier.embedding_size,
                "num_sources": self.classifier.num_sources,
                "combine_mode": self.classifier.combine_mode,
            }
        
        torch.save(state_dict, path)
        print(f"Saved encoder weights to {path}")
    
    def load_encoder_weights(self, path: str, load_classifier: bool = True) -> None:
        """
        Load encoder and classifier weights from a file.
        
        Args:
            path: Path to load the weights from.
            load_classifier: Whether to load classifier weights.
        """
        state_dict = torch.load(path, map_location=self.device)
        for i, encoder in enumerate(self.encoders):
            encoder.load_state_dict(state_dict[f"encoder_{i}"])
        
        if load_classifier and "classifier" in state_dict:
            self.classifier.load_state_dict(state_dict["classifier"])
            print(f"Loaded encoder and classifier weights from {path}")
        else:
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
# Loss Functions
# =============================================================================

def cardpol_loss(
    encoder: nn.Module,
    classifier: CARDPOLClassifier,
    trajectory_batch: TrajectoryMiniBatch,
    source_ids: np.ndarray,
    device: str = "cuda:0",
) -> tuple:
    """
    CARDPOL (Contrastive Actor Recognition for Diverse POLicies) loss.
    
    For each trajectory:
    1. Take the observation at t=0
    2. Sample another observation from t in [1, trajectory_length)
    3. Encode both observations
    4. Use the classifier to predict which source policy they came from
    
    This loss encourages the encoder to learn representations that capture
    policy-specific temporal dynamics, enabling recognition of different
    behavior patterns across datasets.
    
    Args:
        encoder: The encoder network to train.
        classifier: The CARDPOLClassifier network.
        trajectory_batch: Batch of trajectories.
        source_ids: Array of source dataset/policy IDs for each trajectory.
        device: Device to run computations on.
    
    Returns:
        Tuple of (loss, metrics_dict)
    """
    observations = trajectory_batch.observations  # (batch, seq, *obs_shape)
    batch_size, seq_len = observations.shape[:2]
    obs_shape = observations.shape[2:]
    
    if seq_len < 2:
        # Need at least 2 timesteps
        return torch.tensor(0.0, device=device, requires_grad=True), {"accuracy": 0.0}
    
    # Get observation at t=0 for all trajectories
    obs_t0 = observations[:, 0]  # (batch, *obs_shape)
    
    # Sample a random timestep from [1, seq_len) for each trajectory
    random_t = torch.randint(1, seq_len, (batch_size,))
    
    # Get observation at random_t for each trajectory
    # We need to index each batch element with its corresponding random timestep
    obs_t_random = observations[torch.arange(batch_size), random_t]  # (batch, *obs_shape)
    
    # Move to device and convert to float
    obs_t0 = obs_t0.to(device).float()
    obs_t_random = obs_t_random.to(device).float()
    
    # Encode both observations
    embedding_t0 = encoder(obs_t0)  # (batch, feature_dim)
    embedding_t_random = encoder(obs_t_random)  # (batch, feature_dim)
    
    # Get source prediction from classifier
    logits = classifier(embedding_t0, embedding_t_random)  # (batch, num_sources)
    
    # Compute cross-entropy loss
    labels = torch.tensor(source_ids, dtype=torch.long, device=device)
    loss = nn.functional.cross_entropy(logits, labels)
    
    # Compute accuracy for logging
    with torch.no_grad():
        predictions = torch.argmax(logits, dim=-1)
        accuracy = (predictions == labels).float().mean().item()
    
    metrics = {
        "accuracy": accuracy,
        "avg_timestep": random_t.float().mean().item(),
    }
    
    return loss, metrics


def cardpol_loss_with_temporal_regularization(
    encoder: nn.Module,
    classifier: CARDPOLClassifier,
    trajectory_batch: TrajectoryMiniBatch,
    source_ids: np.ndarray,
    device: str = "cuda:0",
    temporal_weight: float = 0.1,
) -> tuple:
    """
    CARDPOL loss with temporal regularization.
    
    Same as cardpol_loss, but also encourages embeddings at different 
    timesteps within the same trajectory to be similar (temporal consistency).
    
    Args:
        encoder: The encoder network to train.
        classifier: The CARDPOLClassifier network.
        trajectory_batch: Batch of trajectories.
        source_ids: Array of source dataset IDs for each trajectory.
        device: Device to run computations on.
        temporal_weight: Weight for the temporal consistency loss.
    
    Returns:
        Tuple of (loss, metrics_dict)
    """
    observations = trajectory_batch.observations
    batch_size, seq_len = observations.shape[:2]
    obs_shape = observations.shape[2:]
    
    if seq_len < 2:
        return torch.tensor(0.0, device=device, requires_grad=True), {"accuracy": 0.0}
    
    # Get observation at t=0
    obs_t0 = observations[:, 0].to(device).float()
    
    # Sample random timestep from [1, seq_len)
    random_t = torch.randint(1, seq_len, (batch_size,))
    obs_t_random = observations[torch.arange(batch_size), random_t].to(device).float()
    
    # Encode both observations
    embedding_t0 = encoder(obs_t0)
    embedding_t_random = encoder(obs_t_random)
    
    # Source prediction loss
    logits = classifier(embedding_t0, embedding_t_random)
    labels = torch.tensor(source_ids, dtype=torch.long, device=device)
    classification_loss = nn.functional.cross_entropy(logits, labels)
    
    # Temporal consistency loss (embeddings should be similar within a trajectory)
    temporal_loss = torch.mean((embedding_t0 - embedding_t_random) ** 2)
    
    # Combined loss
    loss = classification_loss + temporal_weight * temporal_loss
    
    # Compute metrics
    with torch.no_grad():
        predictions = torch.argmax(logits, dim=-1)
        accuracy = (predictions == labels).float().mean().item()
    
    metrics = {
        "accuracy": accuracy,
        "classification_loss": classification_loss.item(),
        "temporal_loss": temporal_loss.item(),
        "avg_timestep": random_t.float().mean().item(),
    }
    
    return loss, metrics


def cardpol_contrastive_loss(
    encoder: nn.Module,
    classifier: CARDPOLClassifier,
    trajectory_batch: TrajectoryMiniBatch,
    source_ids: np.ndarray,
    device: str = "cuda:0",
    contrastive_weight: float = 0.5,
    temperature: float = 0.1,
) -> tuple:
    """
    CARDPOL loss with contrastive regularization.
    
    Combines CARDPOL source prediction with a contrastive objective that
    pulls together embeddings from the same source policy and pushes apart
    embeddings from different source policies.
    
    Args:
        encoder: The encoder network to train.
        classifier: The CARDPOLClassifier network.
        trajectory_batch: Batch of trajectories.
        source_ids: Array of source dataset IDs for each trajectory.
        device: Device to run computations on.
        contrastive_weight: Weight for the contrastive loss term.
        temperature: Temperature for contrastive loss.
    
    Returns:
        Tuple of (loss, metrics_dict)
    """
    observations = trajectory_batch.observations
    batch_size, seq_len = observations.shape[:2]
    
    if seq_len < 2:
        return torch.tensor(0.0, device=device, requires_grad=True), {"accuracy": 0.0}
    
    # Get observations at t=0 and random t
    obs_t0 = observations[:, 0].to(device).float()
    random_t = torch.randint(1, seq_len, (batch_size,))
    obs_t_random = observations[torch.arange(batch_size), random_t].to(device).float()
    
    # Encode observations
    embedding_t0 = encoder(obs_t0)
    embedding_t_random = encoder(obs_t_random)
    
    # Source prediction loss
    logits = classifier(embedding_t0, embedding_t_random)
    labels = torch.tensor(source_ids, dtype=torch.long, device=device)
    classification_loss = nn.functional.cross_entropy(logits, labels)
    
    # Contrastive loss: pull together same-source, push apart different-source
    # Concatenate embeddings for contrastive computation
    all_embeddings = torch.cat([embedding_t0, embedding_t_random], dim=0)  # (2*batch, feature_dim)
    all_embeddings = nn.functional.normalize(all_embeddings, dim=-1)
    all_labels = torch.cat([labels, labels], dim=0)  # (2*batch,)
    
    # Compute similarity matrix
    similarity = torch.mm(all_embeddings, all_embeddings.t()) / temperature  # (2*batch, 2*batch)
    
    # Create mask for positive pairs (same source)
    label_matrix = all_labels.unsqueeze(0) == all_labels.unsqueeze(1)  # (2*batch, 2*batch)
    
    # Remove diagonal (self-similarity)
    mask = ~torch.eye(similarity.shape[0], dtype=torch.bool, device=device)
    
    # Supervised contrastive loss
    positive_mask = label_matrix & mask
    negative_mask = ~label_matrix & mask
    
    if positive_mask.any():
        # For each anchor, compute loss over its positive and negative pairs
        exp_sim = torch.exp(similarity) * mask.float()
        
        # Sum of exp similarities with negatives + positives
        denominator = exp_sim.sum(dim=1, keepdim=True)
        
        # Log probability of positive pairs
        log_prob = similarity - torch.log(denominator + 1e-8)
        
        # Average over positive pairs
        contrastive_loss = -(log_prob * positive_mask.float()).sum() / positive_mask.float().sum()
    else:
        contrastive_loss = torch.tensor(0.0, device=device)
    
    # Combined loss
    loss = classification_loss + contrastive_weight * contrastive_loss
    
    # Compute metrics
    with torch.no_grad():
        predictions = torch.argmax(logits, dim=-1)
        accuracy = (predictions == labels).float().mean().item()
    
    metrics = {
        "accuracy": accuracy,
        "classification_loss": classification_loss.item(),
        "contrastive_loss": contrastive_loss.item() if isinstance(contrastive_loss, torch.Tensor) else contrastive_loss,
        "avg_timestep": random_t.float().mean().item(),
    }
    
    return loss, metrics


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
    
    # Create pre-trainer config
    config = EncoderPretrainConfig(
        learning_rate=1e-4,
        classifier_learning_rate=1e-4,
        batch_size=32,
        trajectory_length=10,
        n_steps=10000,  # Reduced for demonstration
        num_sources=len(datasets),
        classifier_hidden_sizes=[256, 128],
        classifier_combine_mode="concat",  # Options: 'concat', 'diff', 'concat_diff'
        log_interval=100,
        save_interval=2000,
        log_dir=log_dir,
        device=device,
    )
    
    # Create pre-trainer with the pairwise source prediction loss
    # The default loss is cardpol_loss which:
    # 1. Takes observation at t=0
    # 2. Samples another observation from t in [1, trajectory_length)
    # 3. Encodes both and uses the classifier to predict source_id
    pretrainer = EncoderPretrainer(
        cql=cql,
        combined_dataset=combined,
        config=config,
        loss_fn=cardpol_loss,  # Default loss
        # Alternative losses:
        # loss_fn=cardpol_loss_with_temporal_regularization,
        # loss_fn=cardpol_contrastive_loss,
    )
    
    print(f"\nEncoder architecture:")
    print(pretrainer.get_encoder(0))
    print(f"\nEncoder feature size: {pretrainer.get_encoder_feature_size()}")
    
    print(f"\nClassifier architecture:")
    print(pretrainer.classifier)
    print(f"Number of sources: {config.num_sources}")
    print(f"Combine mode: {config.classifier_combine_mode}")
    
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
    
    # Load pre-trained encoder weights (classifier weights are saved but not loaded into CQL)
    load_pretrained_encoder_to_cql(policy_cql, "encoder_final.pt")
    
    # Optional: Freeze encoder weights during policy training
    for q_func in policy_cql._impl._modules.q_funcs:
        for param in q_func.encoder.parameters():
            param.requires_grad = False
    
    # Train policy
    policy_cql.fit(target_dataset, n_steps=100000)
    
    # Note: The classifier is saved alongside encoder weights.
    # You can load it separately if needed for analysis:
    # 
    # checkpoint = torch.load("encoder_final.pt")
    # classifier_config = checkpoint["classifier_config"]
    # classifier = CARDPOLClassifier(
    #     embedding_size=classifier_config["embedding_size"],
    #     num_sources=classifier_config["num_sources"],
    #     combine_mode=classifier_config["combine_mode"],
    # )
    # classifier.load_state_dict(checkpoint["classifier"])
    """)
