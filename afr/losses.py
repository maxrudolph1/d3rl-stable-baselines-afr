"""
Loss functions for CARDPOL encoder pre-training.
"""

import numpy as np
import torch
import torch.nn as nn

from d3rlpy.dataset import TrajectoryMiniBatch

from afr.classifier import CARDPOLClassifier, BehaviorCloningHead, StateDecoderHead, StateClassifier


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


def bc_loss(
    encoder: nn.Module,
    bc_head: BehaviorCloningHead,
    trajectory_batch: TrajectoryMiniBatch,
    device: str = "cuda:0",
    detach_encoder: bool = True,
) -> tuple:
    """
    Behavior Cloning loss for action prediction from t0 embedding.

    Takes the observation at t=0, encodes it, and predicts the action taken.
    Uses cross-entropy loss for discrete actions.

    IMPORTANT: By default, gradients do NOT backpropagate through the encoder.
    The embedding is detached before being passed to the BC head, so only the
    BC head parameters are updated by this loss.

    Args:
        encoder: The encoder network (gradients will NOT flow through if detach_encoder=True).
        bc_head: The BehaviorCloningHead network to train.
        trajectory_batch: Batch of trajectories.
        device: Device to run computations on.
        detach_encoder: If True, detach the embedding to prevent gradients from
                       flowing back to the encoder. Defaults to True.

    Returns:
        Tuple of (loss, metrics_dict).
    """
    observations = trajectory_batch.observations  # (batch, seq, *obs_shape)
    actions = trajectory_batch.actions  # (batch, seq) for discrete actions
    batch_size, seq_len = observations.shape[:2]

    if seq_len < 1:
        return torch.tensor(0.0, device=device, requires_grad=True), {"bc_accuracy": 0.0}

    # Convert to tensor if numpy array
    if isinstance(observations, np.ndarray):
        observations = torch.from_numpy(observations)
    if isinstance(actions, np.ndarray):
        actions = torch.from_numpy(actions)

    # Get observation at t=0 for all trajectories
    obs_t0 = observations[:, 0]  # (batch, *obs_shape)

    # Get action at t=0 for all trajectories
    action_t0 = actions[:, 0, 0]  # (batch, horizon, 1) -> (batch,)

    # Move to device and normalize pixel observations from [0, 255] to [-1, 1]
    obs_t0 = normalize_pixel_obs(obs_t0.to(device))
    action_t0 = action_t0.long().to(device)

    # Encode the observation
    embedding_t0 = encoder(obs_t0)  # (batch, feature_dim)

    # IMPORTANT: Detach embedding to prevent gradients from flowing back to encoder
    if detach_encoder:
        embedding_t0 = embedding_t0.detach()

    # Get action prediction from BC head
    logits = bc_head(embedding_t0)  # (batch, num_actions)

    # Compute cross-entropy loss
    loss = nn.functional.cross_entropy(logits, action_t0)

    # Compute accuracy for logging
    with torch.no_grad():
        predictions = torch.argmax(logits, dim=-1)
        accuracy = (predictions == action_t0).float().mean().item()

    metrics = {
        "bc_accuracy": accuracy,
    }

    return loss, metrics


def state_decoder_loss(
    encoder: nn.Module,
    state_decoder: StateDecoderHead,
    trajectory_batch: TrajectoryMiniBatch,
    normalized_states: np.ndarray,
    device: str = "cuda:0",
    detach_encoder: bool = True,
) -> tuple:
    """
    State decoder loss: predict normalized state from encoder representation.

    For each observation in the trajectory batch, encode it, then decode the
    representation to normalized state. Uses MSE loss. Only valid timesteps
    (mask=1) are included in the loss.

    IMPORTANT: By default, gradients do NOT backpropagate through the encoder.
    The representation is detached before being passed to the state decoder.

    Args:
        encoder: The encoder network (gradients will NOT flow through if detach_encoder=True).
        state_decoder: The StateDecoderHead to train.
        trajectory_batch: Batch of trajectories (observations, masks).
        normalized_states: Ground-truth normalized states, shape (B, L, state_dim).
        device: Device to run on.
        detach_encoder: If True, detach the representation so gradients do not
                        flow back to the encoder. Defaults to True.

    Returns:
        Tuple of (loss, metrics_dict).
    """
    observations = trajectory_batch.observations  # (B, L, *obs_shape)
    masks = trajectory_batch.masks  # (B, L)
    batch_size, seq_len = observations.shape[:2]

    if isinstance(observations, np.ndarray):
        observations = torch.from_numpy(observations)
    if isinstance(masks, np.ndarray):
        masks = torch.from_numpy(masks)
    if isinstance(normalized_states, np.ndarray):
        normalized_states = torch.from_numpy(normalized_states)

    # Flatten to (B*L, *obs_shape)
    obs_flat = observations.reshape(-1, *observations.shape[2:])
    masks_flat = masks.reshape(-1)  # (B*L,)
    states_flat = normalized_states.reshape(-1, normalized_states.shape[-1]).to(device)  # (B*L, state_dim)

    obs_flat = normalize_pixel_obs(obs_flat.to(device))
    masks_flat = masks_flat.to(device)

    # Encode all observations
    embedding_flat = encoder(obs_flat)  # (B*L, embedding_size)
    if detach_encoder:
        embedding_flat = embedding_flat.detach()

    # Decode to state
    state_pred_flat = state_decoder(embedding_flat)  # (B*L, state_dim)

    # MSE only on valid timesteps
    diff = (state_pred_flat - states_flat) * masks_flat.unsqueeze(-1)
    loss = (diff ** 2).sum() / (masks_flat.sum() * state_pred_flat.shape[-1] + 1e-8)

    with torch.no_grad():
        mae = (diff.abs().sum(dim=-1) / (state_pred_flat.shape[-1] + 1e-8)) * masks_flat
        mae = mae.sum() / (masks_flat.sum() + 1e-8)

    metrics = {
        "state_decoder_mae": mae.item(),
    }
    return loss, metrics


def state_classifier_loss(
    state_classifier: StateClassifier,
    normalized_states: np.ndarray,
    source_ids: np.ndarray,
    device: str = "cuda:0",
) -> tuple:
    """
    State classifier loss: predict source_id from normalized state.

    Uses the state at t=0 for each trajectory. Cross-entropy loss
    against the trajectory's source_id.

    Args:
        state_classifier: The StateClassifier to train.
        normalized_states: Ground-truth normalized states, shape (B, L, state_dim).
        source_ids: Source ID per trajectory, shape (B,).
        device: Device to run on.

    Returns:
        Tuple of (loss, metrics_dict).
    """
    if isinstance(normalized_states, np.ndarray):
        normalized_states = torch.from_numpy(normalized_states)
    if isinstance(source_ids, np.ndarray):
        source_ids = torch.from_numpy(source_ids)

    # Use state at t=0: (B, state_dim)
    state_t0 = normalized_states[:, 0].to(device).float()
    labels = source_ids.long().to(device)

    logits = state_classifier(state_t0)  # (B, num_sources)
    loss = nn.functional.cross_entropy(logits, labels)

    with torch.no_grad():
        predictions = torch.argmax(logits, dim=-1)
        accuracy = (predictions == labels).float().mean().item()

    metrics = {
        "state_classifier_accuracy": accuracy,
    }
    return loss, metrics
