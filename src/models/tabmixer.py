# src/models/tabmixer.py
"""TabMixer: MLP-Mixer style encoder for tabular data."""

import torch
import torch.nn as nn

from src.models.base import BaseEncoder


class TabMixer(BaseEncoder):
    """TabMixer encoder for tabular data.

    Stacked residual MLP blocks (inspired by MLP-Mixer) for tabular encoding.
    Purpose-built for tabular data, handles missing values natively.

    Each ResMLPBlock applies two sequential MLP residual sub-layers (pre-norm),
    which is the tabular analogue of the token-mixing / channel-mixing structure
    from the original MLP-Mixer paper, adapted to the non-token input setting.

    From 2025 research: <0.01% FLOPs of FT-Transformer, handles missing data.
    """

    def __init__(
        self,
        continuous_dim: int,
        categorical_dims: list[int] | None = None,
        embedding_dims: list[int] | None = None,
        mixer_layers: int = 4,
        hidden_dim: int = 128,
        expansion_factor: int = 4,
        output_dim: int = 128,
        dropout: float = 0.1,
        handle_missing: bool = True,
    ) -> None:
        """Initialize TabMixer."""
        super().__init__(continuous_dim, output_dim)

        self.continuous_dim = continuous_dim
        self.categorical_dims = categorical_dims or []
        self.embedding_dims = embedding_dims or []
        self.handle_missing = handle_missing

        # Categorical embeddings
        if self.categorical_dims:
            if len(self.categorical_dims) != len(self.embedding_dims):
                raise ValueError("categorical_dims and embedding_dims must match")

            self.embeddings = nn.ModuleList(
                [
                    nn.Embedding(num_embeddings=card, embedding_dim=dim)
                    for card, dim in zip(self.categorical_dims, self.embedding_dims, strict=True)
                ]
            )
            total_emb_dim = sum(self.embedding_dims)
        else:
            total_emb_dim = 0
            self.embeddings = nn.ModuleList()

        # Total input dimension after embeddings
        self.input_features = continuous_dim + total_emb_dim

        # Initial linear projection
        self.input_proj = nn.Linear(self.input_features, hidden_dim)

        # Residual MLP blocks
        self.mixer_blocks = nn.ModuleList(
            [ResMLPBlock(hidden_dim, expansion_factor, dropout) for _ in range(mixer_layers)]
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, output_dim), nn.LayerNorm(output_dim)
        )

        # Missing value handling
        if handle_missing:
            self.missing_mask = nn.Parameter(torch.zeros(continuous_dim))

    def forward(
        self, x_continuous: torch.Tensor, x_categorical: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass through TabMixer."""
        # Handle missing values in continuous features
        if self.handle_missing and torch.isnan(x_continuous).any():
            mask = torch.isnan(x_continuous)
            x_continuous = torch.where(
                mask, self.missing_mask.expand_as(x_continuous), x_continuous
            )

        # Embed categorical features
        if x_categorical is not None and len(self.embeddings) > 0:
            embeddings = []
            for i, emb_layer in enumerate(self.embeddings):
                emb = emb_layer(x_categorical[:, i])
                embeddings.append(emb)
            x_embedded = torch.cat(embeddings, dim=1)

            # Concatenate continuous and categorical
            x = torch.cat([x_continuous, x_embedded], dim=1)
        else:
            x = x_continuous

        # Initial projection
        x = self.input_proj(x)  # (batch, hidden_dim)

        # Mixer blocks
        for block in self.mixer_blocks:
            x = block(x)

        # Output projection
        out: torch.Tensor = self.output_proj(x)

        return out


class ResMLPBlock(nn.Module):
    """Residual MLP block with two sequential pre-norm sub-layers.

    Applies two stacked MLP residual connections, analogous to the
    token-mixing and channel-mixing branches in MLP-Mixer but adapted
    for flat tabular vectors (no token axis).
    """

    def __init__(self, hidden_dim: int, expansion_factor: int = 4, dropout: float = 0.1) -> None:
        super().__init__()

        # First sub-layer
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.mlp1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * expansion_factor, hidden_dim),
            nn.Dropout(dropout),
        )

        # Second sub-layer
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * expansion_factor, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through residual MLP block."""
        x = x + self.mlp1(self.norm1(x))
        x = x + self.mlp2(self.norm2(x))
        return x
