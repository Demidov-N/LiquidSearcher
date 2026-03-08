# src/models/dual_encoder.py
"""Dual-Encoder model: BiMT-TCN + TabMixer for contrastive learning."""

import torch
import torch.nn as nn
import torch.nn.functional as functional

from src.models.tabmixer import TabMixer
from src.models.temporal_encoder import TemporalEncoder


class DualEncoder(nn.Module):
    """Dual-encoder model for stock substitute recommendation.

    Architecture:
    - Temporal Encoder: BiMT-TCN (TCN + Transformer) → 128-dim
    - Tabular Encoder: TabMixer → 128-dim

    Training: Compute dot product similarity between temporal and tabular
    Inference: Concatenate [temporal||tabular] → 256-dim joint embedding
    """

    def __init__(
        self,
        temporal_input_dim: int = 13,
        tabular_continuous_dim: int = 15,
        tabular_categorical_dims: list[int] | None = None,
        tabular_embedding_dims: list[int] | None = None,
        temporal_output_dim: int = 128,
        tabular_output_dim: int = 128,
    ) -> None:
        """Initialize dual-encoder model.

        Note: Temperature is owned by InfoNCELoss, not the model.
        """
        super().__init__()

        self.temporal_output_dim = temporal_output_dim
        self.tabular_output_dim = tabular_output_dim

        # Temporal encoder: BiMT-TCN
        self.temporal_encoder = TemporalEncoder(
            input_dim=temporal_input_dim, output_dim=temporal_output_dim
        )

        # Tabular encoder: TabMixer
        cat_dims = tabular_categorical_dims or []
        emb_dims = tabular_embedding_dims or []

        # Auto-generate embedding dims if not provided but categorical dims are
        if cat_dims and not emb_dims:
            # Default: use min(cardinality // 2, 50) for each categorical feature
            emb_dims = [min(dim // 2, 50) for dim in cat_dims]

        self.tabular_encoder = TabMixer(
            continuous_dim=tabular_continuous_dim,
            categorical_dims=cat_dims,
            embedding_dims=emb_dims,
            output_dim=tabular_output_dim,
        )

    def forward(
        self,
        x_temporal: torch.Tensor,
        x_tabular_continuous: torch.Tensor,
        x_tabular_categorical: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through both encoders.

        Returns:
            temporal_emb: (batch, temporal_output_dim=128)
            tabular_emb: (batch, tabular_output_dim=128)
        """
        temporal_emb = self.temporal_encoder(x_temporal)
        tabular_emb = self.tabular_encoder(x_tabular_continuous, x_tabular_categorical)

        return temporal_emb, tabular_emb

    def compute_pairwise_similarity(
        self,
        x_temporal: torch.Tensor,
        x_tabular_continuous: torch.Tensor,
        x_tabular_categorical: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute element-wise cosine similarity between temporal and tabular embeddings.

        NOTE: This is an inference diagnostic, not the training loss signal.
        InfoNCE training uses the full (batch x batch) similarity matrix internally.

        Returns:
            similarity: (batch,) cosine similarity per sample, in range [-1, 1]
        """
        temporal_emb, tabular_emb = self.forward(
            x_temporal, x_tabular_continuous, x_tabular_categorical
        )

        return functional.cosine_similarity(temporal_emb, tabular_emb, dim=1)

    def get_joint_embedding(
        self,
        x_temporal: torch.Tensor,
        x_tabular_continuous: torch.Tensor,
        x_tabular_categorical: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Get concatenated joint embedding for inference.

        Used for nearest-neighbor search during inference.
        Returns: Joint embedding (batch, 256) = [temporal||tabular]
        """
        temporal_emb, tabular_emb = self.forward(
            x_temporal, x_tabular_continuous, x_tabular_categorical
        )

        # Concatenate for joint representation
        joint_emb = torch.cat([temporal_emb, tabular_emb], dim=1)

        return joint_emb

    def encode_temporal_only(self, x_temporal: torch.Tensor) -> torch.Tensor:
        """Encode temporal features only (for fast inference)."""
        result = self.temporal_encoder(x_temporal)
        assert isinstance(result, torch.Tensor)
        return result

    def encode_tabular_only(
        self, x_tabular_continuous: torch.Tensor, x_tabular_categorical: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Encode tabular features only (for fast inference)."""
        result = self.tabular_encoder(x_tabular_continuous, x_tabular_categorical)
        assert isinstance(result, torch.Tensor)
        return result
