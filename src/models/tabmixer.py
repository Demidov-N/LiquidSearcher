# src/models/tabmixer.py
"""TabMixer: MLP-Mixer style encoder for tabular data."""

import torch
import torch.nn as nn

from src.models.base import BaseEncoder


class TabMixer(BaseEncoder):
    """TabMixer encoder for tabular data.

    MLP-Mixer architecture: channel-wise mixing + instance-wise mixing
    Purpose-built for tabular data, handles missing values natively.

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

        # Mixer blocks
        self.mixer_blocks = nn.ModuleList(
            [MixerBlock(hidden_dim, expansion_factor, dropout) for _ in range(mixer_layers)]
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


class MixerBlock(nn.Module):
    """Single Mixer block: token mixing + channel mixing."""

    def __init__(self, hidden_dim: int, expansion_factor: int = 4, dropout: float = 0.1) -> None:
        super().__init__()

        # Token mixing (mix across features)
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * expansion_factor, hidden_dim),
            nn.Dropout(dropout),
        )

        # Channel mixing (mix within features)
        self.channel_norm = nn.LayerNorm(hidden_dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * expansion_factor, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through mixer block."""
        # Token mixing
        residual = x
        x = self.token_norm(x)
        x = self.token_mlp(x)
        x = x + residual

        # Channel mixing
        residual = x
        x = self.channel_norm(x)
        x = self.channel_mlp(x)
        x = x + residual

        return x
