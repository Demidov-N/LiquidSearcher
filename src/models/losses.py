# src/models/losses.py
"""Loss functions for contrastive learning."""

import torch
import torch.nn as nn
import torch.nn.functional as functional


class InfoNCELoss(nn.Module):
    """InfoNCE loss for contrastive learning.

    Standard contrastive loss that maximizes agreement between positive pairs
    and minimizes agreement with negative pairs (in-batch negatives).

    Formula: L = -log[ exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ) ]

    Where:
    - z_i = temporal embedding
    - z_j = tabular embedding (positive pair)
    - z_k = all other embeddings in batch (negatives)
    - τ = temperature
    """

    def __init__(self, temperature: float = 0.07):
        """Initialize InfoNCE loss.

        Args:
            temperature: Temperature parameter (default 0.07)
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        temporal_emb: torch.Tensor,
        tabular_emb: torch.Tensor,
        hard_negatives_temporal: torch.Tensor | None = None,
        hard_negatives_tabular: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute InfoNCE loss.

        Args:
            temporal_emb: Temporal embeddings (batch, dim)
            tabular_emb: Tabular embeddings (batch, dim)
            hard_negatives_temporal: Optional hard negative temporal embeddings
            hard_negatives_tabular: Optional hard negative tabular embeddings

        Returns:
            Scalar loss value
        """
        batch_size = temporal_emb.size(0)

        # Normalize embeddings
        temporal_emb = functional.normalize(temporal_emb, dim=1)
        tabular_emb = functional.normalize(tabular_emb, dim=1)

        # Compute similarity matrix: (batch, batch)
        # sim[i,j] = similarity between temporal[i] and tabular[j]
        similarity = torch.matmul(temporal_emb, tabular_emb.t()) / self.temperature

        # Labels: positive pairs are on diagonal (i matches i)
        labels = torch.arange(batch_size, device=similarity.device)

        # InfoNCE loss: cross entropy with diagonal as positives
        loss = functional.cross_entropy(similarity, labels)

        return loss


class RankSCLLoss(nn.Module):
    """Rank-supervised contrastive learning loss (placeholder for future).

    Captures ordinal similarity: not just "are these similar?" but
    "HOW similar are these?" (preserves ranking order).

    To be implemented when ranking data available.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, temporal_emb: torch.Tensor, tabular_emb: torch.Tensor) -> torch.Tensor:
        """Placeholder - returns InfoNCE for now."""
        # TODO: Implement proper RankSCL with ordinal constraints
        infonce = InfoNCELoss(self.temperature)
        return infonce.forward(temporal_emb, tabular_emb)
