"""
CARDPOL Classifier for source policy prediction.

The classifier takes two embeddings (from different timesteps of the same
trajectory) and predicts which source policy generated the trajectory.
"""

import torch
import torch.nn as nn
import d3rlpy

def get_encoder_factory(output_size: int = 64):
    return d3rlpy.models.encoders.PixelEncoderFactory(
        filters=[[16, 8, 4], [32, 4, 2], [64, 3, 1]],  # fewer channels
        feature_size=output_size,
    )
    

class CARDPOLClassifier(nn.Module):
    """
    CARDPOL classifier network.

    Takes two embeddings from the encoder and outputs logits representing
    source_id (policy) probabilities.

    The classifier can combine the embeddings in various ways:
    - concatenation: [emb1, emb2]
    - difference: emb1 - emb2
    - concatenation + difference: [emb1, emb2, emb1 - emb2]

    Args:
        embedding_size: Size of each embedding from the encoder.
        num_sources: Number of source datasets/policies to classify.
        hidden_sizes: List of hidden layer sizes. Defaults to [256, 128].
        combine_mode: How to combine the two embeddings.
            Options: 'concat', 'diff', 'concat_diff'.
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
            embedding1: First embedding, shape (batch_size, embedding_size).
            embedding2: Second embedding, shape (batch_size, embedding_size).

        Returns:
            Logits of shape (batch_size, num_sources).
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


class BehaviorCloningHead(nn.Module):
    """
    Behavior Cloning head for action prediction.

    Takes a single embedding (e.g., from t0) and predicts the action.
    This is trained with cross-entropy loss for discrete actions.

    IMPORTANT: The embedding should be detached before being passed to this
    head to prevent gradients from backpropagating into the encoder.

    Args:
        embedding_size: Size of the embedding from the encoder.
        num_actions: Number of discrete actions to predict.
        hidden_sizes: List of hidden layer sizes. Defaults to [256, 128].
    """

    def __init__(
        self,
        embedding_size: int,
        num_actions: int,
        hidden_sizes: list = None,
    ):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [256, 128]

        self.embedding_size = embedding_size
        self.num_actions = num_actions

        # Build MLP layers
        layers = []
        in_size = embedding_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size

        # Output layer
        layers.append(nn.Linear(in_size, num_actions))

        self.network = nn.Sequential(*layers)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the BC head.

        Args:
            embedding: Embedding from encoder, shape (batch_size, embedding_size).
                      Should be detached if you want to prevent gradient flow
                      back to the encoder.

        Returns:
            Action logits of shape (batch_size, num_actions).
        """
        return self.network(embedding)

    def predict_proba(self, embedding: torch.Tensor) -> torch.Tensor:
        """Get action probability predictions."""
        logits = self.forward(embedding)
        return torch.softmax(logits, dim=-1)

    def predict(self, embedding: torch.Tensor) -> torch.Tensor:
        """Get action predictions."""
        logits = self.forward(embedding)
        return torch.argmax(logits, dim=-1)


class StateDecoderHead(nn.Module):
    """
    State decoder head: predicts normalized state from encoder representation.

    Takes a single embedding (e.g., from encoder) and predicts the d-dimensional
    normalized state. Trained with MSE loss. Used when datasets include
    "normalized_state" (e.g. RAM state that was normalized).

    IMPORTANT: The embedding should be detached before being passed to this
    head so that gradients do not backpropagate into the encoder.

    Args:
        embedding_size: Size of the embedding from the encoder.
        normalized_state_dim: Dimension of the normalized state to predict (from data YAML).
        hidden_sizes: List of hidden layer sizes. Defaults to [256, 128].
    """

    def __init__(
        self,
        embedding_size: int,
        normalized_state_dim: int,
        hidden_sizes: list = None,
    ):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [256, 128]

        self.embedding_size = embedding_size
        self.normalized_state_dim = normalized_state_dim

        layers = []
        in_size = embedding_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size

        layers.append(nn.Linear(in_size, normalized_state_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            embedding: Embedding from encoder, shape (batch_size, embedding_size).
                      Should be detached to prevent gradient flow back to encoder.

        Returns:
            Predicted normalized state, shape (batch_size, normalized_state_dim).
        """
        return self.network(embedding)


class CNNImageDecoder(nn.Module):
    """
    CNN decoder that reconstructs the first channel of an image from encoder representation.

    Takes a representation (embedding) from the encoder and outputs a single-channel image
    matching the spatial size of the first channel of the input. For 4-channel Atari inputs,
    this predicts channel 0 only.

    Mirrors the downsampling of the Nature DQN encoder: 84x84 -> 20x20 -> 9x9 -> 7x7.
    Uses transposed convolutions to upsample: 7x7 -> 14x14 -> 28x28 -> 84x84.

    Args:
        embedding_size: Size of the embedding from the encoder.
        observation_shape: (C, H, W) of the input observations. Output will be (1, H, W).
    """

    def __init__(
        self,
        embedding_size: int,
        observation_shape: tuple,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.observation_shape = observation_shape
        _, h, w = observation_shape

        # Encoder downsamples: 84->20->9->7 for default Nature DQN. Compute latent spatial size.
        latent_h, latent_w = 7, 7  # Nature DQN output
        latent_channels = 64

        self.linear = nn.Linear(embedding_size, latent_channels * latent_h * latent_w)

        # Transposed convs: 7->14->28->84 (for 84x84) or adapt for other sizes
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 64, 4, 2, 1),  # 7 -> 14
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 14 -> 28
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),  # 28 -> 56
        )
        # Interpolate to final spatial size (e.g. 84x84 for Atari)
        self._target_h, self._target_w = h, w
        self._needs_resize = h != 56 or w != 56

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            embedding: Embedding from encoder, shape (batch_size, embedding_size).

        Returns:
            Reconstructed first channel, shape (batch_size, 1, H, W). Values in [-1, 1].
        """
        x = self.linear(embedding)
        x = x.view(x.shape[0], 64, 7, 7)
        x = self.deconv(x)
        if self._needs_resize:
            x = torch.nn.functional.interpolate(
                x, size=(self._target_h, self._target_w),
                mode="bilinear", align_corners=False
            )
        return torch.tanh(x)


class VectorQuantizer(nn.Module):
    """
    Vector Quantizer for VQVAE.

    Maps continuous encoder outputs to discrete codes via nearest-neighbor lookup
    in a learnable codebook. Uses straight-through estimator for gradients.

    Args:
        num_embeddings: Number of codebook entries (K).
        embedding_dim: Dimension of each codebook entry (must match encoder output).
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize continuous latent z to nearest codebook entry.

        Args:
            z: Continuous latent from encoder, shape (batch, embedding_dim).

        Returns:
            Tuple of (z_q, indices, codebook_loss, commitment_loss) where:
            - z_q: Quantized latent (for decoder input), shape (batch, embedding_dim).
            - indices: Codebook indices, shape (batch,).
            - codebook_loss: ||sg[z] - z_q||^2 (codebook moves toward encoder).
            - commitment_loss: ||z - sg[z_q]||^2 (encoder commits to codebook).
        """
        # Flatten if needed: (B, D) -> (B, D)
        d = z.shape[-1]
        z_flat = z.view(-1, d)

        # L2 distance to codebook: (B, K)
        distances = (
            torch.sum(z_flat**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flat, self.embedding.weight.t())
        )

        # Find nearest codebook entry
        indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(indices)  # (B, D)

        # Codebook loss: ||sg[z] - z_q||^2 (gradients to codebook only)
        codebook_loss = torch.nn.functional.mse_loss(z_flat.detach(), z_q)

        # Commitment loss: ||z - sg[z_q]||^2 (gradients to encoder only)
        commitment_loss = torch.nn.functional.mse_loss(z_flat, z_q.detach())

        # Straight-through: copy gradients from z_q to z for reconstruction flow
        z_q = z_flat + (z_q - z_flat).detach()

        # Reshape z_q to match input shape
        z_q = z_q.view(*z.shape)
        indices = indices.view(*z.shape[:-1])

        return z_q, indices, codebook_loss, commitment_loss


class VQVAEDecoder(nn.Module):
    """
    CNN decoder for VQVAE that reconstructs the first channel from quantized latent.

    Same architecture as CNNImageDecoder: takes a flat latent vector and produces
    a single-channel image. Used in VQVAE to reconstruct observations.

    Args:
        embedding_size: Size of the quantized latent (matches encoder output).
        observation_shape: (C, H, W) of the input observations. Output will be (1, H, W).
    """

    def __init__(self, embedding_size: int, observation_shape: tuple):
        super().__init__()
        self.embedding_size = embedding_size
        self.observation_shape = observation_shape
        _, h, w = observation_shape

        latent_h, latent_w = 7, 7
        latent_channels = 64

        self.linear = nn.Linear(embedding_size, latent_channels * latent_h * latent_w)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
        )
        self._target_h, self._target_w = h, w
        self._needs_resize = h != 56 or w != 56

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Decode quantized latent to reconstructed image.

        Args:
            z_q: Quantized latent, shape (batch_size, embedding_size).

        Returns:
            Reconstructed first channel, shape (batch_size, 1, H, W). Values in [-1, 1].
        """
        x = self.linear(z_q)
        x = x.view(x.shape[0], 64, 7, 7)
        x = self.deconv(x)
        if self._needs_resize:
            x = torch.nn.functional.interpolate(
                x, size=(self._target_h, self._target_w),
                mode="bilinear", align_corners=False
            )
        return torch.tanh(x)


class VQVAEAuxiliary(nn.Module):
    """
    Auxiliary module for VQVAE baseline: wraps VectorQuantizer and VQVAEDecoder.

    Used as the 'classifier' argument in EncoderPretrainer when using VQVAE loss.
    Provides a unified interface for the quantizer and decoder.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: int,
        observation_shape: tuple,
    ):
        super().__init__()
        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
        )
        self.decoder = VQVAEDecoder(
            embedding_size=embedding_dim,
            observation_shape=observation_shape,
        )

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize and decode.

        Args:
            z: Encoder output, shape (batch, embedding_dim).

        Returns:
            Tuple of (reconstruction, indices, codebook_loss, commitment_loss).
        """
        z_q, indices, codebook_loss, commitment_loss = self.quantizer(z)
        recon = self.decoder(z_q)
        return recon, indices, codebook_loss, commitment_loss


class CPCContextEncoder(nn.Module):
    """
    Context encoder for Contrastive Predictive Coding (CPC).

    Aggregates a sequence of encoder embeddings into a context vector using a GRU.
    Used to predict future latent representations from past context.

    Args:
        embedding_size: Size of input embeddings from the encoder.
        hidden_size: Hidden size of the GRU. Defaults to embedding_size.
    """

    def __init__(
        self,
        embedding_size: int,
        hidden_size: int | None = None,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size or embedding_size

        self.gru = nn.GRU(
            input_size=embedding_size,
            hidden_size=self.hidden_size,
            batch_first=True,
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            embeddings: Sequence of embeddings, shape (batch, seq_len, embedding_size).

        Returns:
            Context vectors, shape (batch, seq_len, hidden_size).
        """
        return self.gru(embeddings)[0]


class StateClassifier(nn.Module):
    """
    Classifier that predicts source_id from normalized state.

    Takes d-dimensional normalized state (e.g. RAM state) and outputs
    logits over num_sources, matching the observation classifier's
    output space. Used when batches include normalized_states.

    Args:
        normalized_state_dim: Dimension of the normalized state input (from data YAML).
        num_sources: Number of source datasets/policies to classify.
        hidden_sizes: List of hidden layer sizes. Defaults to [256, 128].
    """

    def __init__(
        self,
        normalized_state_dim: int,
        num_sources: int,
        hidden_sizes: list = None,
    ):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [256, 128]

        self.normalized_state_dim = normalized_state_dim
        self.num_sources = num_sources

        layers = []
        in_size = normalized_state_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size

        layers.append(nn.Linear(in_size, num_sources))
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: Normalized state, shape (batch_size, normalized_state_dim).

        Returns:
            Logits of shape (batch_size, num_sources).
        """
        return self.network(state)

    def predict_proba(self, state: torch.Tensor) -> torch.Tensor:
        """Get probability predictions."""
        logits = self.forward(state)
        return torch.softmax(logits, dim=-1)

    def predict(self, state: torch.Tensor) -> torch.Tensor:
        """Get class predictions."""
        logits = self.forward(state)
        return torch.argmax(logits, dim=-1)
