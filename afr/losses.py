"""
Loss functions for CARDPOL encoder pre-training.

Includes CARDPOL, CURL (Contrastive Unsupervised Representations for RL),
and CPC (Contrastive Predictive Coding) objectives.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from d3rlpy.dataset import TrajectoryMiniBatch

from afr.networks import (
    CARDPOLClassifier,
    BehaviorCloningHead,
    StateDecoderHead,
    StateClassifier,
    CNNImageDecoder,
    CPCContextEncoder,
)


def normalize_pixel_obs(obs: torch.Tensor) -> torch.Tensor:
    """
    Normalize pixel observations from [0, 255] to [-1, 1].

    Args:
        obs: Tensor with pixel values in range [0, 255].

    Returns:
        Normalized tensor with values in range [-1, 1].
    """
    return obs.float() / 127.5 - 1.0


def _curl_augment(obs: torch.Tensor, pad: int = 8) -> torch.Tensor:
    """
    Apply CURL-style augmentation: random crop.

    Pads the observation and randomly crops back to original size.
    Expects obs shape (B, C, H, W) with values in [-1, 1].

    Args:
        obs: Normalized observations.
        pad: Padding size on each side.

    Returns:
        Augmented observations of same shape.
    """
    if pad <= 0:
        return obs
    *batch_dims, c, h, w = obs.shape
    # Pad: (B, C, H, W) -> (B, C, H+2*pad, W+2*pad)
    obs_padded = F.pad(obs, [pad] * 4, mode="replicate")
    # Random crop: sample top-left (max_h, max_w) with max_h in [0, 2*pad], max_w in [0, 2*pad]
    max_h = 2 * pad
    max_w = 2 * pad
    top = torch.randint(0, max_h + 1, (1,), device=obs.device).item()
    left = torch.randint(0, max_w + 1, (1,), device=obs.device).item()
    return obs_padded[..., top : top + h, left : left + w]


def curl_loss(
    encoder: nn.Module,
    classifier: nn.Module,
    trajectory_batch: TrajectoryMiniBatch,
    source_ids: np.ndarray,
    device: str = "cuda:0",
    pad: int = 8,
    temperature: float = 0.1,
) -> tuple:
    """
    CURL (Contrastive Unsupervised Representations for RL) loss.

    Applies data augmentation to create two views of each observation, encodes
    both with the encoder, and uses an InfoNCE contrastive loss to pull
    together the two views of the same observation while pushing apart
    different observations.

    The classifier parameter is ignored (kept for API compatibility with
    EncoderPretrainer). Use curl_loss as the main encoder loss only.

    Args:
        encoder: The encoder network to train.
        classifier: Unused; kept for API compatibility.
        trajectory_batch: Batch of trajectories.
        source_ids: Unused; kept for API compatibility.
        device: Device to run computations on.
        pad: Padding for random crop augmentation (default 8 for 84x84 Atari).
        temperature: Temperature for InfoNCE loss (default 0.1).

    Returns:
        Tuple of (loss, metrics_dict).
    """
    observations = trajectory_batch.observations  # (batch, seq, *obs_shape)
    batch_size, seq_len = observations.shape[:2]

    if seq_len < 1:
        return torch.tensor(0.0, device=device, requires_grad=True), {"curl_acc": 0.0}

    if isinstance(observations, np.ndarray):
        observations = torch.from_numpy(observations)

    # Use all observations in the batch (flatten batch and seq)
    obs_flat = observations.reshape(-1, *observations.shape[2:])  # (B*L, C, H, W)
    obs_flat = normalize_pixel_obs(obs_flat.to(device))

    # Create two augmented views
    obs_q = _curl_augment(obs_flat, pad=pad)
    obs_k = _curl_augment(obs_flat, pad=pad)

    # Encode both views (both receive gradients)
    z_q = encoder(obs_q)  # (N, feature_dim)
    z_k = encoder(obs_k)  # (N, feature_dim)

    # L2 normalize for cosine similarity
    z_q = F.normalize(z_q, dim=-1)
    z_k = F.normalize(z_k, dim=-1)

    # InfoNCE: log(exp(sim(q,k+)/tau) / sum_j exp(sim(q,k_j)/tau))
    # Positive pairs: (z_q[i], z_k[i]). Negatives: (z_q[i], z_k[j]) for j != i
    logits = torch.matmul(z_q, z_k.T) / temperature  # (N, N)
    labels = torch.arange(z_q.shape[0], device=device)
    loss = F.cross_entropy(logits, labels)

    with torch.no_grad():
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == labels).float().mean().item()

    metrics = {"curl_acc": acc, "curl_loss": loss.item()}
    return loss, metrics


def cpc_loss(
    encoder: nn.Module,
    context_encoder: CPCContextEncoder,
    trajectory_batch: TrajectoryMiniBatch,
    source_ids: np.ndarray,
    device: str = "cuda:0",
    num_steps: int = 12,
    temperature: float = 0.1,
) -> tuple:
    """
    CPC (Contrastive Predictive Coding) loss.

    Learns representations by predicting future latent states from past context.
    For each position t, the context c_t (from GRU over z_1..z_t) is used to
    predict z_{t+k} among a set of candidates (positive + negatives) via
    InfoNCE. Uses in-batch negatives: positives are future latents from the
    same sequence; negatives are latents from other batch elements.

    Args:
        encoder: The encoder network to train.
        context_encoder: CPCContextEncoder (GRU) that aggregates past embeddings.
        trajectory_batch: Batch of trajectories.
        source_ids: Unused; kept for API compatibility.
        device: Device to run computations on.
        num_steps: Number of future steps to predict (default 12).
        temperature: Temperature for InfoNCE loss (default 0.1).

    Returns:
        Tuple of (loss, metrics_dict).
    """
    observations = trajectory_batch.observations  # (B, L, *obs_shape)
    masks = trajectory_batch.masks  # (B, L)
    batch_size, seq_len = observations.shape[:2]

    if seq_len < 2 or seq_len < num_steps + 1:
        return torch.tensor(0.0, device=device, requires_grad=True), {"cpc_acc": 0.0}

    if isinstance(observations, np.ndarray):
        observations = torch.from_numpy(observations)
    if isinstance(masks, np.ndarray):
        masks = torch.from_numpy(masks)

    obs_flat = observations.reshape(-1, *observations.shape[2:])
    obs_flat = normalize_pixel_obs(obs_flat.to(device))
    masks_flat = masks.reshape(-1).to(device)

    # Encode all observations
    z_all = encoder(obs_flat)  # (B*L, feature_dim)
    z_all = F.normalize(z_all, dim=-1)
    z_seq = z_all.reshape(batch_size, seq_len, -1)  # (B, L, D)
    masks_seq = masks.reshape(batch_size, seq_len)  # (B, L)

    # Get context for each timestep: c_t = GRU(z_1, ..., z_t)
    context = context_encoder(z_seq)  # (B, L, hidden_size)
    context = F.normalize(context, dim=-1)

    loss_total = 0.0
    n_predictions = 0

    # For each position t, predict z_{t+1}, z_{t+2}, ... z_{t+num_steps}
    for k in range(1, min(num_steps + 1, seq_len)):
        # Context at t, target at t+k
        # c: (B, L-k, hidden), z_future: (B, L-k, D)
        c_t = context[:, :-k]  # (B, L-k, hidden)
        z_future = z_seq[:, k:]  # (B, L-k, D)
        mask_valid = masks_seq[:, :-k] * masks_seq[:, k:]  # (B, L-k)

        # Flatten batch and time for contrastive loss
        c_flat = c_t.reshape(-1, c_t.shape[-1])  # (B*(L-k), hidden)
        z_flat = z_future.reshape(-1, z_future.shape[-1])  # (B*(L-k), D)
        mask_flat = mask_valid.reshape(-1)  # (B*(L-k),)

        valid_idx = mask_flat.nonzero(as_tuple=True)[0]
        if valid_idx.numel() == 0:
            continue

        c_valid = c_flat[valid_idx]  # (N_valid, hidden)
        z_valid = z_flat[valid_idx]  # (N_valid, D)

        # InfoNCE: positive pair (c_i, z_i), negatives from batch
        logits = torch.matmul(c_valid, z_valid.T) / temperature  # (N_valid, N_valid)
        labels = torch.arange(c_valid.shape[0], device=device)
        step_loss = F.cross_entropy(logits, labels)
        loss_total = loss_total + step_loss
        n_predictions += 1

    if n_predictions == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), {"cpc_acc": 0.0}

    loss = loss_total / n_predictions

    with torch.no_grad():
        # Compute accuracy for the last step
        logits_last = torch.matmul(c_valid, z_valid.T) / temperature
        preds = torch.argmax(logits_last, dim=-1)
        acc = (preds == labels).float().mean().item()

    metrics = {"cpc_acc": acc, "cpc_loss": loss.item(), "cpc_n_steps": n_predictions}
    return loss, metrics


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
        normalized_states: Ground-truth normalized states, shape (B, L, normalized_state_dim).
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
    states_flat = normalized_states.reshape(-1, normalized_states.shape[-1]).to(device)  # (B*L, normalized_state_dim)

    obs_flat = normalize_pixel_obs(obs_flat.to(device))
    masks_flat = masks_flat.to(device)

    # Encode all observations
    embedding_flat = encoder(obs_flat)  # (B*L, embedding_size)
    if detach_encoder:
        embedding_flat = embedding_flat.detach()

    # Decode to state
    state_pred_flat = state_decoder(embedding_flat)  # (B*L, normalized_state_dim)

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


def image_decoder_loss(
    encoder: nn.Module,
    image_decoder: CNNImageDecoder,
    trajectory_batch: TrajectoryMiniBatch,
    device: str = "cuda:0",
    detach_encoder: bool = True,
) -> tuple:
    """
    Image decoder loss: reconstruct first channel of observation from encoder representation.

    For each observation, encode it, then decode the representation to the first channel.
    Uses MSE loss. Predicts channel 0 only (e.g. for 4-channel stacked frames).

    IMPORTANT: By default, gradients do NOT backpropagate through the encoder.

    Args:
        encoder: The encoder network (gradients will NOT flow through if detach_encoder=True).
        image_decoder: The CNNImageDecoder to train.
        trajectory_batch: Batch of trajectories (observations, masks).
        device: Device to run on.
        detach_encoder: If True, detach the representation so gradients do not
                        flow back to the encoder. Defaults to True.

    Returns:
        Tuple of (loss, metrics_dict).
    """
    observations = trajectory_batch.observations  # (B, L, C, H, W)
    masks = trajectory_batch.masks  # (B, L)

    if isinstance(observations, np.ndarray):
        observations = torch.from_numpy(observations)
    if isinstance(masks, np.ndarray):
        masks = torch.from_numpy(masks)

    # Flatten to (B*L, C, H, W)
    obs_flat = observations.reshape(-1, *observations.shape[2:])
    masks_flat = masks.reshape(-1)  # (B*L,)

    obs_flat = normalize_pixel_obs(obs_flat.to(device))
    masks_flat = masks_flat.to(device)

    # Target: first channel only, shape (B*L, 1, H, W)
    target_first_channel = obs_flat[:, :1]  # (B*L, 1, H, W)

    # Encode all observations
    embedding_flat = encoder(obs_flat)
    if detach_encoder:
        embedding_flat = embedding_flat.detach()

    # Decode to image
    pred_first_channel = image_decoder(embedding_flat)  # (B*L, 1, H, W)

    # MSE only on valid timesteps
    diff = (pred_first_channel - target_first_channel) * masks_flat.view(-1, 1, 1, 1)
    n_pixels_per_frame = pred_first_channel.shape[1] * pred_first_channel.shape[2] * pred_first_channel.shape[3]
    n_valid_pixels = masks_flat.sum() * n_pixels_per_frame + 1e-8
    loss = (diff ** 2).sum() / n_valid_pixels

    with torch.no_grad():
        mae_per_frame = diff.abs().mean(dim=(1, 2, 3)) * masks_flat
        mae = mae_per_frame.sum() / (masks_flat.sum() + 1e-8)

    metrics = {
        "image_decoder_mae": mae.item(),
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
        normalized_states: Ground-truth normalized states, shape (B, L, normalized_state_dim).
        source_ids: Source ID per trajectory, shape (B,).
        device: Device to run on.

    Returns:
        Tuple of (loss, metrics_dict).
    """
    if isinstance(normalized_states, np.ndarray):
        normalized_states = torch.from_numpy(normalized_states)
    if isinstance(source_ids, np.ndarray):
        source_ids = torch.from_numpy(source_ids)

    # Use state at t=0: (B, normalized_state_dim)
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
