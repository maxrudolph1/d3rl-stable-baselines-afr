"""
Encoder pre-training for CARDPOL.

Provides EncoderPretrainConfig, EncoderPretrainer, and utilities for
loading pre-trained encoder weights into CQL models.
"""

import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from afr.config import EncoderPretrainConfig

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

import d3rlpy

from afr.networks import (
    CARDPOLClassifier,
    BehaviorCloningHead,
    StateDecoderHead,
    StateClassifier,
    CNNImageDecoder,
    VQVAEAuxiliary,
)
from afr.extended_dataset import CombinedMDPDataset, TrajectoryBatchWithSource
from afr.losses import (
    cardpol_loss,
    bc_loss,
    state_decoder_loss,
    state_classifier_loss,
    image_decoder_loss,
    vqvae_loss,
    normalize_pixel_obs,
)


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
        bc_head: Optional[BehaviorCloningHead] = None,
        state_decoder: Optional[StateDecoderHead] = None,
        state_classifier: Optional[StateClassifier] = None,
        image_decoder: Optional[CNNImageDecoder] = None,
        val_combined_dataset: Optional[CombinedMDPDataset] = None,
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
            bc_head: Optional pre-initialized BehaviorCloningHead.
                    If None and config.use_bc_head=True, one will be created.
            state_decoder: Optional pre-initialized StateDecoderHead.
                    If None and config.use_state_decoder=True, one will be created.
            state_classifier: Optional pre-initialized StateClassifier.
                    If None and config.use_state_classifier=True, one will be created.
            image_decoder: Optional pre-initialized CNNImageDecoder.
                    If None and config.use_image_decoder=True, one will be created.
            val_combined_dataset: Optional validation CombinedMDPDataset for
                                  evaluating classifier accuracy during training.
        """
        self.cql = cql
        self.combined_dataset = combined_dataset
        self.val_combined_dataset = val_combined_dataset
        self.config = config
        self.device = config.device

        # Extract encoders from Q-functions
        self.encoders = self._extract_encoders()

        # Get encoder feature size
        encoder_feature_size = self._compute_encoder_feature_size()

        self.use_vqvae = config.use_vqvae

        # Create or use provided classifier / VQVAE auxiliary
        if config.use_vqvae:
            obs_shape = tuple(self.cql._impl.observation_shape)
            if classifier is not None and isinstance(classifier, VQVAEAuxiliary):
                self.classifier = classifier.to(self.device)
            else:
                self.classifier = VQVAEAuxiliary(
                    embedding_dim=encoder_feature_size,
                    num_embeddings=config.vqvae_num_embeddings,
                    observation_shape=obs_shape,
                ).to(self.device)
        else:
            if classifier is not None:
                self.classifier = classifier.to(self.device)
            else:
                self.classifier = CARDPOLClassifier(
                    embedding_size=encoder_feature_size,
                    num_sources=config.num_sources,
                    hidden_sizes=config.classifier_hidden_sizes,
                    combine_mode=config.classifier_combine_mode,
                ).to(self.device)

        # Create or use provided BC head (if enabled)
        self.bc_head = None
        self.bc_optimizer = None
        if config.use_bc_head:
            if bc_head is not None:
                self.bc_head = bc_head.to(self.device)
            else:
                self.bc_head = BehaviorCloningHead(
                    embedding_size=encoder_feature_size,
                    num_actions=config.num_actions,
                    hidden_sizes=config.bc_hidden_sizes,
                ).to(self.device)
            # Set up optimizer for BC head parameters
            self.bc_optimizer = optim.Adam(
                self.bc_head.parameters(),
                lr=config.bc_learning_rate
            )
        # Create or use provided state decoder head (if enabled)
        self.state_decoder = None
        self.state_decoder_optimizer = None
        if config.use_state_decoder:
            if state_decoder is not None:
                self.state_decoder = state_decoder.to(self.device)
            else:
                self.state_decoder = StateDecoderHead(
                    embedding_size=encoder_feature_size,
                    normalized_state_dim=config.normalized_state_dim,
                    hidden_sizes=config.state_decoder_hidden_sizes,
                ).to(self.device)
            self.state_decoder_optimizer = optim.Adam(
                self.state_decoder.parameters(),
                lr=config.state_decoder_learning_rate,
            )

        # Create or use provided state classifier (if enabled)
        self.state_classifier = None
        self.state_classifier_optimizer = None
        if config.use_state_classifier:
            if state_classifier is not None:
                self.state_classifier = state_classifier.to(self.device)
            else:
                self.state_classifier = StateClassifier(
                    normalized_state_dim=config.normalized_state_dim,
                    num_sources=config.num_sources,
                    hidden_sizes=config.state_classifier_hidden_sizes,
                ).to(self.device)
            self.state_classifier_optimizer = optim.Adam(
                self.state_classifier.parameters(),
                lr=config.state_classifier_learning_rate,
            )

        # Create or use provided image decoder (if enabled; skipped when using VQVAE)
        self.image_decoder = None
        self.image_decoder_optimizer = None
        if config.use_image_decoder and not config.use_vqvae:
            obs_shape = tuple(self.cql._impl.observation_shape)
            if image_decoder is not None:
                self.image_decoder = image_decoder.to(self.device)
            else:
                self.image_decoder = CNNImageDecoder(
                    embedding_size=encoder_feature_size,
                    observation_shape=obs_shape,
                ).to(self.device)
            self.image_decoder_optimizer = optim.Adam(
                self.image_decoder.parameters(),
                lr=config.image_decoder_learning_rate,
            )

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

        # Loss function (user-defined or default based on method)
        if loss_fn is not None:
            self.loss_fn = loss_fn
        elif config.use_vqvae:
            self.loss_fn = lambda enc, aux, batch, ids, dev: vqvae_loss(
                enc, aux, batch, ids, dev,
                commitment_cost=config.vqvae_commitment_cost,
            )
        else:
            self.loss_fn = cardpol_loss

        # Logging
        self.writer = SummaryWriter(f"{config.log_dir}/{config.group}")

        # Initialize wandb if enabled
        self.use_wandb = config.use_wandb and WANDB_AVAILABLE
        if config.use_wandb and not WANDB_AVAILABLE:
            print("Warning: wandb requested but not installed. Skipping wandb logging.")
        if self.use_wandb:
            wandb_config = {
                "learning_rate": config.learning_rate,
                "classifier_learning_rate": config.classifier_learning_rate,
                "batch_size": config.batch_size,
                "trajectory_length": config.trajectory_length,
                "n_steps": config.n_steps,
                "num_sources": config.num_sources,
                "classifier_hidden_sizes": config.classifier_hidden_sizes,
                "classifier_combine_mode": config.classifier_combine_mode,
                "val_interval": config.val_interval,
                "val_batch_size": config.val_batch_size,
                "val_n_batches": config.val_n_batches,
                "encoder_feature_size": encoder_feature_size,
                "num_encoders": len(self.encoders),
                "device": config.device,
                # BC head config
                "use_bc_head": config.use_bc_head,
                "bc_learning_rate": config.bc_learning_rate,
                "bc_hidden_sizes": config.bc_hidden_sizes,
                "bc_loss_weight": config.bc_loss_weight,
                "num_actions": config.num_actions,
                # State decoder config
                "use_state_decoder": config.use_state_decoder,
                "normalized_state_dim": config.normalized_state_dim,
                "state_decoder_learning_rate": config.state_decoder_learning_rate,
                "state_decoder_loss_weight": config.state_decoder_loss_weight,
                "use_state_classifier": config.use_state_classifier,
                "state_classifier_learning_rate": config.state_classifier_learning_rate,
                "state_classifier_loss_weight": config.state_classifier_loss_weight,
                "use_image_decoder": config.use_image_decoder,
                "image_decoder_learning_rate": config.image_decoder_learning_rate,
                "image_decoder_loss_weight": config.image_decoder_loss_weight,
                "use_vqvae": config.use_vqvae,
                "vqvae_num_embeddings": config.vqvae_num_embeddings,
                "vqvae_commitment_cost": config.vqvae_commitment_cost,
                "group": config.group,
            }
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=config.wandb_run_name,
                tags=config.wandb_tags,
                config=wandb_config,
            )
            # Watch encoder and classifier models for gradient tracking
            wandb.watch(self.encoders[0], log="gradients", log_freq=config.log_interval)
            wandb.watch(self.classifier, log="gradients", log_freq=config.log_interval)
            if self.bc_head is not None:
                wandb.watch(self.bc_head, log="gradients", log_freq=config.log_interval)
            if self.state_decoder is not None:
                wandb.watch(self.state_decoder, log="gradients", log_freq=config.log_interval)
            if self.state_classifier is not None:
                wandb.watch(self.state_classifier, log="gradients", log_freq=config.log_interval)
            if self.image_decoder is not None:
                wandb.watch(self.image_decoder, log="gradients", log_freq=config.log_interval)

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
            encoders.append(q_func.encoder)
        return encoders

    def get_encoder(self, index: int = 0) -> nn.Module:
        """Get a specific encoder by index."""
        return self.encoders[index]

    def get_encoder_feature_size(self) -> int:
        """Get the output feature size of the encoder."""
        encoder = self.encoders[0]
        with torch.no_grad():
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
        all_metrics = {}

        for encoder in self.encoders:
            encoder.train()
        self.classifier.train()
        if self.bc_head is not None:
            self.bc_head.train()
        if self.state_decoder is not None:
            self.state_decoder.train()
        if self.state_classifier is not None:
            self.state_classifier.train()
        if self.image_decoder is not None:
            self.image_decoder.train()

        # Use the first encoder for the loss (they share the same architecture)
        cardpol_loss_val, metrics = self.loss_fn(
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
                cardpol_loss_val = cardpol_loss_val + additional_loss
            cardpol_loss_val = cardpol_loss_val / len(self.encoders)

        # Backward pass for CARDPOL loss (updates encoder + classifier)
        self.encoder_optimizer.zero_grad()
        self.classifier_optimizer.zero_grad()
        cardpol_loss_val.backward()
        self.encoder_optimizer.step()
        self.classifier_optimizer.step()

        # Collect metrics
        if self.use_vqvae:
            all_metrics["vqvae_loss"] = cardpol_loss_val.item()
        else:
            all_metrics["cardpol_loss"] = cardpol_loss_val.item()
        all_metrics.update(metrics)

        # Compute BC loss if BC head is enabled
        # Note: BC loss does NOT backpropagate to encoder (embedding is detached)
        if self.bc_head is not None:
            bc_loss_val, bc_metrics = bc_loss(
                encoder=self.encoders[0],
                bc_head=self.bc_head,
                trajectory_batch=trajectory_batch,
                device=self.device,
                detach_encoder=True,  # Prevents gradients flowing to encoder
            )

            # Backward pass for BC loss (only updates BC head)
            self.bc_optimizer.zero_grad()
            bc_loss_val.backward()
            self.bc_optimizer.step()

            all_metrics["bc_loss"] = bc_loss_val.item()
            all_metrics.update(bc_metrics)

        # State decoder loss: decode normalized_state from representation (no grad through encoder)
        if self.state_decoder is not None and batch_with_source.normalized_states is not None:
            state_decoder_loss_val, state_decoder_metrics = state_decoder_loss(
                encoder=self.encoders[0],
                state_decoder=self.state_decoder,
                trajectory_batch=trajectory_batch,
                normalized_states=batch_with_source.normalized_states,
                device=self.device,
                detach_encoder=True,
            )
            self.state_decoder_optimizer.zero_grad()
            state_decoder_loss_val.backward()
            self.state_decoder_optimizer.step()
            all_metrics["state_decoder_loss"] = state_decoder_loss_val.item()
            all_metrics.update(state_decoder_metrics)

        # State classifier loss: predict source_id from normalized_state (no encoder)
        if self.state_classifier is not None and batch_with_source.normalized_states is not None:
            state_classifier_loss_val, state_classifier_metrics = state_classifier_loss(
                state_classifier=self.state_classifier,
                normalized_states=batch_with_source.normalized_states,
                source_ids=source_ids,
                device=self.device,
            )
            self.state_classifier_optimizer.zero_grad()
            state_classifier_loss_val.backward()
            self.state_classifier_optimizer.step()
            all_metrics["state_classifier_loss"] = state_classifier_loss_val.item()
            all_metrics.update(state_classifier_metrics)

        # Image decoder loss: reconstruct first channel from representation (no grad through encoder)
        if self.image_decoder is not None:
            image_decoder_loss_val, image_decoder_metrics = image_decoder_loss(
                encoder=self.encoders[0],
                image_decoder=self.image_decoder,
                trajectory_batch=trajectory_batch,
                device=self.device,
                detach_encoder=True,
            )
            self.image_decoder_optimizer.zero_grad()
            image_decoder_loss_val.backward()
            self.image_decoder_optimizer.step()
            all_metrics["image_decoder_loss"] = image_decoder_loss_val.item()
            all_metrics.update(image_decoder_metrics)

        # Compute total loss for logging (weighted sum)
        total_loss = cardpol_loss_val.item()
        if self.bc_head is not None:
            total_loss += self.config.bc_loss_weight * bc_loss_val.item()
        if self.state_decoder is not None and "state_decoder_loss" in all_metrics:
            total_loss += self.config.state_decoder_loss_weight * all_metrics["state_decoder_loss"]
        if self.state_classifier is not None and "state_classifier_loss" in all_metrics:
            total_loss += self.config.state_classifier_loss_weight * all_metrics["state_classifier_loss"]
        if self.image_decoder is not None and "image_decoder_loss" in all_metrics:
            total_loss += self.config.image_decoder_loss_weight * all_metrics["image_decoder_loss"]
        all_metrics["loss"] = total_loss

        return all_metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Run validation on the validation dataset.

        Returns:
            Dictionary of validation metrics including accuracy and loss.
        """
        if self.val_combined_dataset is None:
            return {}

        # Set models to eval mode
        for encoder in self.encoders:
            encoder.eval()
        self.classifier.eval()
        if self.bc_head is not None:
            self.bc_head.eval()
        if self.state_decoder is not None:
            self.state_decoder.eval()
        if self.state_classifier is not None:
            self.state_classifier.eval()
        if self.image_decoder is not None:
            self.image_decoder.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_predictions = []
        all_labels = []

        # BC validation metrics
        bc_total_loss = 0.0
        bc_total_correct = 0
        bc_total_samples = 0

        # State decoder validation metrics
        state_decoder_total_mae = 0.0
        state_decoder_n_batches = 0

        # Image decoder validation metrics
        image_decoder_total_mae = 0.0
        image_decoder_n_batches = 0

        # State classifier validation metrics
        state_classifier_total_correct = 0
        state_classifier_total_samples = 0

        # VQVAE validation metrics
        vqvae_total_mae = 0.0
        vqvae_n_batches = 0

        for _ in range(self.config.val_n_batches):
            # Sample validation batch
            batch_with_source = self.val_combined_dataset.sample_trajectory_batch(
                batch_size=self.config.val_batch_size,
                length=self.config.trajectory_length,
            )
            trajectory_batch, source_ids = self._prepare_trajectory_batch(batch_with_source)

            # Get observations and actions
            observations = trajectory_batch.observations
            actions = trajectory_batch.actions
            batch_size, seq_len = observations.shape[:2]

            min_seq_len = 2 if not self.use_vqvae else 1
            if seq_len < min_seq_len:
                continue

            # Convert to tensor if numpy array
            if isinstance(observations, np.ndarray):
                observations = torch.from_numpy(observations)
            if isinstance(actions, np.ndarray):
                actions = torch.from_numpy(actions)

            # Get observation at t=0 for all trajectories and normalize
            obs_t0 = normalize_pixel_obs(observations[:, 0].to(self.device))

            # Encode for BC and CARDPOL (when not use_vqvae)
            embedding_t0 = self.encoders[0](obs_t0)

            if self.use_vqvae:
                # VQVAE validation: compute reconstruction metrics
                _, vqvae_metrics = vqvae_loss(
                    encoder=self.encoders[0],
                    vqvae_aux=self.classifier,
                    trajectory_batch=trajectory_batch,
                    source_ids=source_ids,
                    device=self.device,
                    commitment_cost=self.config.vqvae_commitment_cost,
                )
                vqvae_total_mae += vqvae_metrics["vqvae_mae"]
                vqvae_n_batches += 1
            else:
                # Sample random timesteps for CARDPOL
                random_t = torch.randint(1, seq_len, (batch_size,))
                obs_t_random = normalize_pixel_obs(
                    observations[torch.arange(batch_size), random_t].to(self.device)
                )
                embedding_t_random = self.encoders[0](obs_t_random)

                # Get CARDPOL predictions
                logits = self.classifier(embedding_t0, embedding_t_random)
                labels = torch.tensor(source_ids, dtype=torch.long, device=self.device)

                # Compute CARDPOL loss
                loss = nn.functional.cross_entropy(logits, labels)
                total_loss += loss.item() * batch_size

                # Compute CARDPOL accuracy
                predictions = torch.argmax(logits, dim=-1)
                total_correct += (predictions == labels).sum().item()
                total_samples += batch_size

                # Store for per-class metrics
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            # Compute BC validation metrics if BC head is enabled
            if self.bc_head is not None:
                action_t0 = actions[:, 0, 0].long().to(self.device)
                bc_logits = self.bc_head(embedding_t0.detach())
                bc_loss_val = nn.functional.cross_entropy(bc_logits, action_t0)
                bc_total_loss += bc_loss_val.item() * batch_size
                bc_predictions = torch.argmax(bc_logits, dim=-1)
                bc_total_correct += (bc_predictions == action_t0).sum().item()
                bc_total_samples += batch_size

            # State decoder validation when batch has normalized_states
            if (
                self.state_decoder is not None
                and batch_with_source.normalized_states is not None
            ):
                _, state_decoder_metrics = state_decoder_loss(
                    encoder=self.encoders[0],
                    state_decoder=self.state_decoder,
                    trajectory_batch=trajectory_batch,
                    normalized_states=batch_with_source.normalized_states,
                    device=self.device,
                    detach_encoder=True,
                )
                state_decoder_total_mae += state_decoder_metrics["state_decoder_mae"]
                state_decoder_n_batches += 1

            # Image decoder validation
            if self.image_decoder is not None:
                _, image_decoder_metrics = image_decoder_loss(
                    encoder=self.encoders[0],
                    image_decoder=self.image_decoder,
                    trajectory_batch=trajectory_batch,
                    device=self.device,
                    detach_encoder=True,
                )
                image_decoder_total_mae += image_decoder_metrics["image_decoder_mae"]
                image_decoder_n_batches += 1

            # State classifier validation when batch has normalized_states
            if (
                self.state_classifier is not None
                and batch_with_source.normalized_states is not None
            ):
                _, state_classifier_metrics = state_classifier_loss(
                    state_classifier=self.state_classifier,
                    normalized_states=batch_with_source.normalized_states,
                    source_ids=source_ids,
                    device=self.device,
                )
                # Recompute accuracy from loss batch size for val metric
                batch_size = batch_with_source.normalized_states.shape[0]
                state_classifier_total_correct += int(
                    state_classifier_metrics["state_classifier_accuracy"] * batch_size
                )
                state_classifier_total_samples += batch_size

        # Set models back to train mode
        for encoder in self.encoders:
            encoder.train()
        self.classifier.train()
        if self.bc_head is not None:
            self.bc_head.train()
        if self.state_decoder is not None:
            self.state_decoder.train()
        if self.state_classifier is not None:
            self.state_classifier.train()
        if self.image_decoder is not None:
            self.image_decoder.train()

        if total_samples == 0 and not self.use_vqvae:
            return {"val_accuracy": 0.0, "val_loss": 0.0}

        val_metrics = {}
        if self.use_vqvae:
            # VQVAE validation: use reconstruction MAE if we have any batches
            val_metrics["val_vqvae_mae"] = (
                vqvae_total_mae / vqvae_n_batches
                if vqvae_n_batches > 0
                else 0.0
            )
        else:
            val_metrics["val_accuracy"] = total_correct / total_samples
            val_metrics["val_loss"] = total_loss / total_samples

        # Compute per-class accuracy if we have enough data (CARDPOL only)
        if not self.use_vqvae and len(all_predictions) > 0:
            all_predictions = np.array(all_predictions)
            all_labels = np.array(all_labels)
            for source_id in range(self.config.num_sources):
                mask = all_labels == source_id
                if mask.sum() > 0:
                    class_acc = (all_predictions[mask] == source_id).mean()
                    val_metrics[f"val_accuracy_source_{source_id}"] = class_acc

        # Add BC validation metrics
        if self.bc_head is not None and bc_total_samples > 0:
            val_metrics["val_bc_accuracy"] = bc_total_correct / bc_total_samples
            val_metrics["val_bc_loss"] = bc_total_loss / bc_total_samples

        # Add state decoder validation metrics
        if self.state_decoder is not None and state_decoder_n_batches > 0:
            val_metrics["val_state_decoder_mae"] = (
                state_decoder_total_mae / state_decoder_n_batches
            )

        # Add image decoder validation metrics
        if self.image_decoder is not None and image_decoder_n_batches > 0:
            val_metrics["val_image_decoder_mae"] = (
                image_decoder_total_mae / image_decoder_n_batches
            )

        # Add state classifier validation metrics
        if self.state_classifier is not None and state_classifier_total_samples > 0:
            val_metrics["val_state_classifier_accuracy"] = (
                state_classifier_total_correct / state_classifier_total_samples
            )

        return val_metrics

    @torch.no_grad()
    def _log_image_decoder_samples(self, step: int, n_samples: int = 8) -> None:
        """Sample a batch, run encoder+decoder, and log input/output images to wandb."""
        if not WANDB_AVAILABLE or not self.use_wandb:
            return
        batch_with_source = self.combined_dataset.sample_trajectory_batch(
            batch_size=n_samples, length=1
        )
        trajectory_batch = batch_with_source.batch
        observations = trajectory_batch.observations  # (B, 1, C, H, W)
        if isinstance(observations, np.ndarray):
            observations = torch.from_numpy(observations)
        obs = observations[:, 0].to(self.device)  # (B, C, H, W)
        obs_norm = normalize_pixel_obs(obs)
        embedding = self.encoders[0](obs_norm)
        pred = self.image_decoder(embedding)  # (B, 1, H, W), values in [-1, 1]

        # Target: first channel of input (normalized)
        target = obs_norm[:, :1]  # (B, 1, H, W)

        # Convert to [0, 255] for wandb display: (x + 1) * 127.5
        def to_uint8(x: torch.Tensor) -> np.ndarray:
            return ((x.cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

        input_imgs = to_uint8(target)  # (B, 1, H, W)
        output_imgs = to_uint8(pred)  # (B, 1, H, W)
        # wandb.Image expects (H, W) or (H, W, C) for grayscale
        table = wandb.Table(columns=["Input (first channel)", "Output (reconstructed)"])
        for i in range(min(n_samples, input_imgs.shape[0])):
            inp = input_imgs[i, 0]  # (H, W)
            out = output_imgs[i, 0]  # (H, W)
            table.add_data(wandb.Image(inp), wandb.Image(out))
        wandb.log({"image_decoder/input_vs_output": table}, step=step)

    @torch.no_grad()
    def _log_vqvae_samples(self, step: int, n_samples: int = 8) -> None:
        """Sample a batch, run encoder+VQVAE, and log input/output images to wandb."""
        if not WANDB_AVAILABLE or not self.use_wandb or not self.use_vqvae:
            return
        batch_with_source = self.combined_dataset.sample_trajectory_batch(
            batch_size=n_samples, length=1
        )
        trajectory_batch = batch_with_source.batch
        observations = trajectory_batch.observations
        if isinstance(observations, np.ndarray):
            observations = torch.from_numpy(observations)
        obs = observations[:, 0].to(self.device)
        obs_norm = normalize_pixel_obs(obs)
        z = self.encoders[0](obs_norm)
        pred, _, _, _ = self.classifier(z)
        target = obs_norm[:, :1]

        def to_uint8(x: torch.Tensor) -> np.ndarray:
            return ((x.cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

        input_imgs = to_uint8(target)
        output_imgs = to_uint8(pred)
        table = wandb.Table(columns=["Input (first channel)", "VQVAE Reconstructed"])
        for i in range(min(n_samples, input_imgs.shape[0])):
            table.add_data(wandb.Image(input_imgs[i, 0]), wandb.Image(output_imgs[i, 0]))
        wandb.log({"vqvae/input_vs_output": table}, step=step)

    def pretrain(self) -> None:
        """Run the full pre-training loop."""
        print(f"Starting encoder pre-training for {self.config.n_steps} steps...")
        print(f"Batch size: {self.config.batch_size}, Trajectory length: {self.config.trajectory_length}")
        print(f"Number of encoders: {len(self.encoders)}")
        print(f"Encoder feature size: {self.get_encoder_feature_size()}")
        print(f"Logging to: {self.config.log_dir}")
        if self.use_wandb:
            print(f"Wandb logging enabled: {self.config.wandb_project}")
        if self.val_combined_dataset is not None:
            print(f"Validation enabled: every {self.config.val_interval} steps")
        else:
            print("Validation: disabled (no validation dataset provided)")
        if self.bc_head is not None:
            print(f"BC head enabled: {self.config.num_actions} actions, weight={self.config.bc_loss_weight}")
            print(f"  NOTE: BC gradients do NOT backpropagate to encoder")
        if self.state_decoder is not None:
            print(f"State decoder enabled: normalized_state_dim={self.config.normalized_state_dim}, weight={self.config.state_decoder_loss_weight}")
            print(f"  NOTE: State decoder gradients do NOT backpropagate to encoder")
        if self.state_classifier is not None:
            print(f"State classifier enabled: normalized_state_dim={self.config.normalized_state_dim}, weight={self.config.state_classifier_loss_weight}")
        if self.use_vqvae:
            print(f"VQVAE baseline: num_embeddings={self.config.vqvae_num_embeddings}, commitment_cost={self.config.vqvae_commitment_cost}")
        if self.image_decoder is not None:
            print(f"Image decoder enabled: predicts first channel, weight={self.config.image_decoder_loss_weight}, log every {self.config.image_decoder_log_interval} steps")
            print(f"  NOTE: Image decoder gradients do NOT backpropagate to encoder")
        print("-" * 50)

        best_val_accuracy = 0.0
        best_val_vqvae_mae = float("inf")
        for step in range(1, self.config.n_steps + 1):
            metrics = self.pretrain_step()

            # Log metrics to tensorboard
            for name, value in metrics.items():
                self.writer.add_scalar(f"pretrain/{name}", value, step)

            # Log metrics to wandb
            if self.use_wandb:
                wandb.log({f"train/{k}": v for k, v in metrics.items()}, step=step)

                # Log input/output images periodically
                if self.config.image_decoder_log_interval > 0 and step % self.config.image_decoder_log_interval == 0:
                    if self.use_vqvae:
                        self._log_vqvae_samples(step)
                    elif self.image_decoder is not None:
                        self._log_image_decoder_samples(step)

            # Run validation
            if (self.config.val_interval > 0 and
                self.val_combined_dataset is not None and
                step % self.config.val_interval == 0):
                val_metrics = self.validate()
                for name, value in val_metrics.items():
                    self.writer.add_scalar(f"pretrain/{name}", value, step)

                # Log validation metrics to wandb
                if self.use_wandb:
                    wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=step)

                    # Track best validation metric
                    if self.use_vqvae:
                        vqvae_mae = val_metrics.get("val_vqvae_mae", float("inf"))
                        if vqvae_mae < best_val_vqvae_mae:
                            best_val_vqvae_mae = vqvae_mae
                            wandb.run.summary["best_val_vqvae_mae"] = best_val_vqvae_mae
                            wandb.run.summary["best_val_step"] = step
                    elif val_metrics.get("val_accuracy", 0) > best_val_accuracy:
                        best_val_accuracy = val_metrics["val_accuracy"]
                        wandb.run.summary["best_val_accuracy"] = best_val_accuracy
                        wandb.run.summary["best_val_step"] = step

                # Print validation metrics
                val_str = ", ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items())
                print(f"Step {step}/{self.config.n_steps} - VALIDATION - {val_str}")

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

                # Log checkpoint artifact to wandb
                if self.use_wandb:
                    artifact = wandb.Artifact(
                        f"encoder_checkpoint_{step}",
                        type="model",
                        description=f"Encoder checkpoint at step {step}"
                    )
                    artifact.add_file(checkpoint_path)
                    wandb.log_artifact(artifact)

        # Run final validation
        if self.val_combined_dataset is not None:
            print("\nFinal validation:")
            val_metrics = self.validate()
            val_str = ", ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items())
            print(f"  {val_str}")

            # Log final validation metrics to wandb summary
            if self.use_wandb:
                for k, v in val_metrics.items():
                    wandb.run.summary[f"final_{k}"] = v

        print("Pre-training complete!")
        self.writer.close()

        # Finish wandb run
        if self.use_wandb:
            wandb.finish()

    def save_encoder_weights(
        self,
        path: str,
        include_classifier: bool = True,
        include_bc_head: bool = True,
        include_state_decoder: bool = True,
        include_state_classifier: bool = True,
        include_image_decoder: bool = True,
    ) -> None:
        """
        Save encoder, classifier, BC head, state decoder, and state classifier weights to a file.

        Args:
            path: Path to save the weights.
            include_classifier: Whether to include classifier weights.
            include_bc_head: Whether to include BC head weights.
            include_state_decoder: Whether to include state decoder weights.
            include_state_classifier: Whether to include state classifier weights.
            include_image_decoder: Whether to include image decoder weights.
        """
        state_dict = {}
        for i, encoder in enumerate(self.encoders):
            state_dict[f"encoder_{i}"] = encoder.state_dict()

        if include_classifier:
            state_dict["classifier"] = self.classifier.state_dict()
            if self.use_vqvae:
                state_dict["classifier_config"] = {
                    "use_vqvae": True,
                    "embedding_dim": self.classifier.quantizer.embedding_dim,
                    "num_embeddings": self.classifier.quantizer.num_embeddings,
                    "observation_shape": self.classifier.decoder.observation_shape,
                }
            else:
                state_dict["classifier_config"] = {
                    "embedding_size": self.classifier.embedding_size,
                    "num_sources": self.classifier.num_sources,
                    "combine_mode": self.classifier.combine_mode,
                }

        if include_bc_head and self.bc_head is not None:
            state_dict["bc_head"] = self.bc_head.state_dict()
            state_dict["bc_head_config"] = {
                "embedding_size": self.bc_head.embedding_size,
                "num_actions": self.bc_head.num_actions,
            }

        if include_state_decoder and self.state_decoder is not None:
            state_dict["state_decoder"] = self.state_decoder.state_dict()
            state_dict["state_decoder_config"] = {
                "embedding_size": self.state_decoder.embedding_size,
                "normalized_state_dim": self.state_decoder.normalized_state_dim,
            }

        if include_state_classifier and self.state_classifier is not None:
            state_dict["state_classifier"] = self.state_classifier.state_dict()
            state_dict["state_classifier_config"] = {
                "normalized_state_dim": self.state_classifier.normalized_state_dim,
                "num_sources": self.state_classifier.num_sources,
            }

        if include_image_decoder and self.image_decoder is not None:
            state_dict["image_decoder"] = self.image_decoder.state_dict()
            state_dict["image_decoder_config"] = {
                "embedding_size": self.image_decoder.embedding_size,
                "observation_shape": self.image_decoder.observation_shape,
            }

        torch.save(state_dict, path)
        print(f"Saved encoder weights to {path}")

    def load_encoder_weights(
        self,
        path: str,
        load_classifier: bool = True,
        load_bc_head: bool = True,
        load_state_decoder: bool = True,
        load_state_classifier: bool = True,
        load_image_decoder: bool = True,
    ) -> None:
        """
        Load encoder, classifier, BC head, state decoder, and state classifier weights from a file.

        Args:
            path: Path to load the weights from.
            load_classifier: Whether to load classifier weights.
            load_bc_head: Whether to load BC head weights.
            load_state_decoder: Whether to load state decoder weights.
            load_state_classifier: Whether to load state classifier weights.
            load_image_decoder: Whether to load image decoder weights.
        """
        state_dict = torch.load(path, map_location=self.device)
        for i, encoder in enumerate(self.encoders):
            encoder.load_state_dict(state_dict[f"encoder_{i}"])

        loaded_components = ["encoder"]

        if load_classifier and "classifier" in state_dict:
            self.classifier.load_state_dict(state_dict["classifier"])
            loaded_components.append("classifier")

        if load_bc_head and "bc_head" in state_dict and self.bc_head is not None:
            self.bc_head.load_state_dict(state_dict["bc_head"])
            loaded_components.append("bc_head")

        if load_state_decoder and "state_decoder" in state_dict and self.state_decoder is not None:
            self.state_decoder.load_state_dict(state_dict["state_decoder"])
            loaded_components.append("state_decoder")

        if load_state_classifier and "state_classifier" in state_dict and self.state_classifier is not None:
            self.state_classifier.load_state_dict(state_dict["state_classifier"])
            loaded_components.append("state_classifier")

        if load_image_decoder and "image_decoder" in state_dict and self.image_decoder is not None:
            self.image_decoder.load_state_dict(state_dict["image_decoder"])
            loaded_components.append("image_decoder")

        print(f"Loaded {', '.join(loaded_components)} weights from {path}")


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
