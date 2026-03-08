# src/models/sampler.py
"""Hard negative sampling strategies for contrastive learning."""

from typing import Any, cast

import pandas as pd
import torch


class GICSHardNegativeSampler:
    """GICS-structured hard negative sampler.

    Uses GICS hierarchy for three-tier negative sampling:
    - Easy: Different gsector (exclude - too easy)
    - Hard: Same gsector, different ggroup
    - Hardest: Same ggroup (industry group), different beta/vol

    Critical for within-sector discrimination in stock substitutes.
    """

    def __init__(
        self, n_hard: int = 8, beta_threshold: float = 0.3, use_sector_fallback: bool = True
    ) -> None:
        """Initialize GICS hard negative sampler.

        Args:
            n_hard: Number of hard negatives to sample
            beta_threshold: Beta difference threshold for "different risk profile"
            use_sector_fallback: Whether to fall back to sector-level if no group matches
        """
        self.n_hard = n_hard
        self.beta_threshold = beta_threshold
        self.use_sector_fallback = use_sector_fallback

    def sample(
        self,
        target: pd.Series | dict[str, Any],
        candidates: pd.DataFrame,
        embeddings: torch.Tensor | None = None,
    ) -> list[str]:
        """Sample hard negatives for target stock.

        Args:
            target: Target stock with 'symbol', 'ggroup', 'gsector', 'beta'
            candidates: DataFrame of candidate stocks with same columns
            embeddings: Optional pre-computed embeddings for similarity-based sampling

        Returns:
            List of hard negative symbols
        """
        target_group = target["ggroup"]
        target_sector = target["gsector"]
        target_beta = target["beta"]

        # Level 1: Same ggroup (industry group), different beta
        same_group_diff_beta = candidates[
            (candidates["ggroup"] == target_group)
            & (candidates["symbol"] != target["symbol"])
            & (abs(candidates["beta"] - target_beta) > self.beta_threshold)
        ]

        if len(same_group_diff_beta) >= self.n_hard // 2:
            # Sample from same group with different beta
            group_samples = same_group_diff_beta.sample(
                n=min(self.n_hard // 2, len(same_group_diff_beta)), replace=False
            )["symbol"].tolist()
        elif len(same_group_diff_beta) > 0:
            # Take all available
            group_samples = same_group_diff_beta["symbol"].tolist()
        else:
            group_samples = []

        # Level 2: Same gsector, different ggroup (if need more)
        remaining = self.n_hard - len(group_samples)
        if remaining > 0 and self.use_sector_fallback:
            same_sector_diff_group = candidates[
                (candidates["gsector"] == target_sector)
                & (candidates["ggroup"] != target_group)
                & (~candidates["symbol"].isin(group_samples + [target["symbol"]]))
            ]

            if len(same_sector_diff_group) > 0:
                sector_samples = same_sector_diff_group.sample(
                    n=min(remaining, len(same_sector_diff_group)), replace=False
                )["symbol"].tolist()
                group_samples.extend(sector_samples)

        # If still not enough, fill with random from same sector
        remaining = self.n_hard - len(group_samples)
        if remaining > 0:
            same_sector = candidates[
                (candidates["gsector"] == target_sector)
                & (~candidates["symbol"].isin(group_samples + [target["symbol"]]))
            ]

            if len(same_sector) > 0:
                random_samples = same_sector.sample(
                    n=min(remaining, len(same_sector)), replace=False
                )["symbol"].tolist()
                group_samples.extend(random_samples)

        return cast(list[str], group_samples)

    def create_hard_negative_batch(
        self, batch_df: pd.DataFrame, temporal_data: torch.Tensor, tabular_data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create batch with hard negatives inserted.

        Args:
            batch_df: DataFrame with stock metadata including GICS
            temporal_data: Temporal embeddings (batch, seq, features)
            tabular_data: Tabular embeddings (batch, features)

        Returns:
            temporal_with_negs, tabular_with_negs with hard negatives added
        """
        # This is a placeholder - full implementation would sample negatives
        # and concatenate them to the batch for contrastive training
        # TODO: Implement when training pipeline ready
        return temporal_data, tabular_data
