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
        """Compute symmetric InfoNCE loss.

        Args:
            temporal_emb: Temporal embeddings (batch, dim)
            tabular_emb: Tabular embeddings (batch, dim)
            hard_negatives_temporal: Optional hard negative temporal embeddings (n_hard, dim)
            hard_negatives_tabular: Optional hard negative tabular embeddings (n_hard, dim)

        Returns:
            Scalar loss value
        """
        batch_size = temporal_emb.size(0)

        # Normalize embeddings
        temporal_norm = functional.normalize(temporal_emb, dim=1)
        tabular_norm = functional.normalize(tabular_emb, dim=1)

        # Extend denominator with hard negatives if provided
        if hard_negatives_tabular is not None:
            hard_tab_norm = functional.normalize(hard_negatives_tabular, dim=1)
            all_tabular = torch.cat([tabular_norm, hard_tab_norm], dim=0)
        else:
            all_tabular = tabular_norm

        if hard_negatives_temporal is not None:
            hard_temp_norm = functional.normalize(hard_negatives_temporal, dim=1)
            all_temporal = torch.cat([temporal_norm, hard_temp_norm], dim=0)
        else:
            all_temporal = temporal_norm

        # Similarity matrices: temporal→tabular and tabular→temporal
        # sim_t2tab[i,j] = similarity between temporal[i] and tabular[j]
        sim_t2tab = torch.matmul(temporal_norm, all_tabular.t()) / self.temperature
        sim_tab2t = torch.matmul(tabular_norm, all_temporal.t()) / self.temperature

        # Labels: positive pairs are on diagonal (i matches i)
        labels = torch.arange(batch_size, device=temporal_norm.device)

        # Symmetric InfoNCE: average loss in both directions (CLIP-style)
        loss = 0.5 * (
            functional.cross_entropy(sim_t2tab, labels)
            + functional.cross_entropy(sim_tab2t, labels)
        )

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
        # Reuse InfoNCELoss instance to avoid reconstruction on every forward call
        self._infonce = InfoNCELoss(temperature)

    def forward(self, temporal_emb: torch.Tensor, tabular_emb: torch.Tensor) -> torch.Tensor:
        """Placeholder - returns InfoNCE for now."""
        # TODO: Implement proper RankSCL with ordinal constraints
        return self._infonce(temporal_emb, tabular_emb)
