"""Dual encoder model for embedding price and fundamental data."""

import torch
import torch.nn as nn


class DualEncoder(nn.Module):
    """Dual encoder for price series and fundamental data."""

    def __init__(
        self,
        temporal_dim: int = 128,
        fundamental_dim: int = 128,
        joint_dim: int = 256,
        price_seq_len: int = 60,
        num_fundamental_features: int = 20,
    ) -> None:
        """Initialize dual encoder model.

        Args:
            temporal_dim: Dimension of temporal encoder output
            fundamental_dim: Dimension of fundamental encoder output
            joint_dim: Dimension of joint embedding space
            price_seq_len: Length of price sequence input
            num_fundamental_features: Number of fundamental features
        """
        super().__init__()
        self.temporal_dim = temporal_dim
        self.fundamental_dim = fundamental_dim
        self.joint_dim = joint_dim
        self.price_seq_len = price_seq_len
        self.num_fundamental_features = num_fundamental_features

        self._build_encoders()

    def _build_encoders(self) -> None:
        """Build encoder sub-networks."""
        self.price_encoder = nn.Sequential(
            nn.Linear(self.price_seq_len, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.temporal_dim),
        )

        self.fundamental_encoder = nn.Sequential(
            nn.Linear(self.num_fundamental_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, self.fundamental_dim),
        )

        self.price_projection = nn.Linear(self.temporal_dim, self.joint_dim)
        self.fundamental_projection = nn.Linear(self.fundamental_dim, self.joint_dim)

    def encode_price(self, price_features: torch.Tensor) -> torch.Tensor:
        """Encode price/time-series features to embedding.

        Args:
            price_features: Price feature tensor of shape (batch, seq_len)

        Returns:
            Price embedding tensor of shape (batch, joint_dim)
        """
        hidden = self.price_encoder(price_features)
        return self.price_projection(hidden)

    def encode_fundamental(self, fundamental_features: torch.Tensor) -> torch.Tensor:
        """Encode fundamental features to embedding.

        Args:
            fundamental_features: Fundamental feature tensor
                of shape (batch, num_features)

        Returns:
            Fundamental embedding tensor of shape (batch, joint_dim)
        """
        hidden = self.fundamental_encoder(fundamental_features)
        return self.fundamental_projection(hidden)

    def forward(
        self,
        price_features: torch.Tensor,
        fundamental_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass computing both price and fundamental embeddings.

        Args:
            price_features: Price feature tensor
            fundamental_features: Fundamental feature tensor

        Returns:
            Tuple of (price_embedding, fundamental_embedding)
        """
        price_emb = self.encode_price(price_features)
        fund_emb = self.encode_fundamental(fundamental_features)
        return price_emb, fund_emb
