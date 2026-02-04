"""
Loss functions for CARDPOL encoder pre-training.
"""

import numpy as np
import torch
import torch.nn as nn

from d3rlpy.dataset import TrajectoryMiniBatch

from afr_scripts.classifier import CARDPOLClassifier


def normalize_pixel_obs(obs: torch.Tensor) -> torch.Tensor:
    """
    Normalize pixel observations from [0, 255] to [-1, 1].

    Args:
        obs: Tensor with pixel values in range [0, 255].

    Returns:
        Normalized tensor with values in range [-1, 1].
    """
    return obs.float() / 127.5 - 1.0


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
        Tuple of (loss, metrics_dict).
    """
    observations = trajectory_batch.observations  # (batch, seq, *obs_shape)
    batch_size, seq_len = observations.shape[:2]

    if seq_len < 2:
        # Need at least 2 timesteps
        return torch.tensor(0.0, device=device, requires_grad=True), {"accuracy": 0.0}

    # Convert to tensor if numpy array
    if isinstance(observations, np.ndarray):
        observations = torch.from_numpy(observations)

    # Get observation at t=0 for all trajectories
    obs_t0 = observations[:, 0]  # (batch, *obs_shape)

    # Sample a random timestep from [1, seq_len) for each trajectory
    random_t = torch.randint(1, seq_len, (batch_size,))

    # Get observation at random_t for each trajectory
    obs_t_random = observations[torch.arange(batch_size), random_t]  # (batch, *obs_shape)

    # Move to device and normalize pixel observations from [0, 255] to [-1, 1]
    obs_t0 = normalize_pixel_obs(obs_t0.to(device))
    obs_t_random = normalize_pixel_obs(obs_t_random.to(device))

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
