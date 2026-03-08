"""BiMT-TCN: TCN + lightweight Transformer for temporal encoding."""

from typing import cast

import torch
import torch.nn as nn

from src.models.base import BaseEncoder
from src.models.tcn import TemporalConvNet


class TemporalEncoder(BaseEncoder):
    """Temporal encoder: BiMT-TCN architecture.

    Combines:
    - TCN (3 layers): Local multi-scale pattern detection
    - Transformer (2 layers, 4 heads): Global cross-timestep dependencies

    Architecture from 2025 research showing Transformer outperforms TCN alone.
    """

    def __init__(
        self,
        input_dim: int = 13,
        tcn_hidden: int = 64,
        tcn_layers: int = 3,
        tcn_kernel_size: int = 3,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        transformer_dim: int = 128,
        output_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        """Initialize BiMT-TCN encoder."""
        super().__init__(input_dim, output_dim)

        # TCN for local patterns
        self.tcn = TemporalConvNet(
            input_dim=input_dim,
            hidden_dim=tcn_hidden,
            output_dim=transformer_dim,
            num_layers=tcn_layers,
            kernel_size=tcn_kernel_size,
            dropout=dropout,
        )

        # Positional encoding for transformer
        self.pos_encoder = PositionalEncoding(transformer_dim, dropout)

        # Lightweight Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # Global average pooling over time
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Final projection to output_dim
        self.projection = nn.Linear(transformer_dim, output_dim)

        # Layer normalization for stability
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through BiMT-TCN."""
        batch_size, seq_len, _ = x.shape

        # TCN: Local patterns (batch, seq_len, transformer_dim)
        out = self.tcn(x)

        # Positional encoding
        out = self.pos_encoder(out)

        # Transformer: Global dependencies (batch, seq_len, transformer_dim)
        out = self.transformer(out)

        # Global average pooling over time (batch, transformer_dim)
        out = out.transpose(1, 2)  # (batch, transformer_dim, seq_len)
        out = self.pool(out).squeeze(-1)  # (batch, transformer_dim)

        # Project to output dimension (batch, output_dim)
        out = self.projection(out)

        # Final normalization
        out = self.norm(out)

        return cast(torch.Tensor, out)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    pe: torch.Tensor

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding."""
        pe_tensor = cast(torch.Tensor, self.pe)
        x = x + pe_tensor[: x.size(1), :].transpose(0, 1)
        return cast(torch.Tensor, self.dropout(x))
