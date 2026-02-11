"""
CARDPOL Classifier for source policy prediction.

The classifier takes two embeddings (from different timesteps of the same
trajectory) and predicts which source policy generated the trajectory.
"""

import torch
import torch.nn as nn


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
        state_dim: Dimension of the normalized state to predict.
        hidden_sizes: List of hidden layer sizes. Defaults to [256, 128].
    """

    def __init__(
        self,
        embedding_size: int,
        state_dim: int,
        hidden_sizes: list = None,
    ):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [256, 128]

        self.embedding_size = embedding_size
        self.state_dim = state_dim

        layers = []
        in_size = embedding_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size

        layers.append(nn.Linear(in_size, state_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            embedding: Embedding from encoder, shape (batch_size, embedding_size).
                      Should be detached to prevent gradient flow back to encoder.

        Returns:
            Predicted normalized state, shape (batch_size, state_dim).
        """
        return self.network(embedding)


class StateClassifier(nn.Module):
    """
    Classifier that predicts source_id from normalized state.

    Takes d-dimensional normalized state (e.g. RAM state) and outputs
    logits over num_sources, matching the observation classifier's
    output space. Used when batches include normalized_states.

    Args:
        state_dim: Dimension of the normalized state input.
        num_sources: Number of source datasets/policies to classify.
        hidden_sizes: List of hidden layer sizes. Defaults to [256, 128].
    """

    def __init__(
        self,
        state_dim: int,
        num_sources: int,
        hidden_sizes: list = None,
    ):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [256, 128]

        self.state_dim = state_dim
        self.num_sources = num_sources

        layers = []
        in_size = state_dim
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
            state: Normalized state, shape (batch_size, state_dim).

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
