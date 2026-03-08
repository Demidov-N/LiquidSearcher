"""Custom DataLoader with GICS-structured hard negative sampling."""

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.models.sampler import GICSHardNegativeSampler
from src.training.dataset import FeatureDataset


class StockDataLoader(DataLoader):
    """DataLoader that adds GICS-structured hard negatives to each batch.

    Attributes:
        dataset: The FeatureDataset being loaded
        n_hard: Number of hard negatives per batch
        hard_neg_sampler: GICSHardNegativeSampler instance
    """

    dataset: FeatureDataset

    def __init__(
        self,
        dataset: FeatureDataset,
        batch_size: int,
        sampler: GICSHardNegativeSampler,
        feature_dir: str,
        n_hard: int = 8,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> None:
        """Initialize dataloader with hard negative sampling.

        Args:
            dataset: FeatureDataset with pre-computed features
            batch_size: Number of positive samples per batch
            sampler: GICSHardNegativeSampler for finding hard negatives
            feature_dir: Path to feature files (for fetching hard neg features)
            n_hard: Number of hard negatives to add per batch
            shuffle: Whether to shuffle samples each epoch
            num_workers: Number of worker processes for loading
        """
        self.n_hard = n_hard
        self.hard_neg_sampler = sampler
        self.feature_dir = feature_dir

        # Initialize parent DataLoader
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_with_hard_negatives,
        )

    def _collate_with_hard_negatives(self, batch_samples: list[dict]) -> dict:
        """Collate function that adds hard negatives.

        Args:
            batch_samples: List of sample dicts from Dataset.__getitem__

        Returns:
            Batched dict with hard negatives concatenated
        """
        batch_size = len(batch_samples)

        # Extract positive samples
        temporal_pos = torch.stack([s["temporal"] for s in batch_samples])
        tabular_cont_pos = torch.stack([s["tabular_cont"] for s in batch_samples])
        tabular_cat_pos = torch.stack([s["tabular_cat"] for s in batch_samples])

        # Build candidate dataframe for hard negative sampling
        candidates = pd.DataFrame(
            [
                {
                    "symbol": s["symbol"],
                    "gsector": s["gsector"],
                    "ggroup": s["ggroup"],
                    "beta": s["beta"],
                }
                for s in batch_samples
            ]
        )

        # Sample hard negatives for each stock in batch
        hard_negatives_temporal = []
        hard_negatives_tabular_cont = []
        hard_negatives_tabular_cat = []

        for sample in batch_samples:
            target = {
                "symbol": sample["symbol"],
                "gsector": sample["gsector"],
                "ggroup": sample["ggroup"],
                "beta": sample["beta"],
            }

            # Get hard negative symbols
            neg_symbols = self.hard_neg_sampler.sample(target, candidates)

            # Fetch features for hard negatives
            for neg_symbol in neg_symbols[: self.n_hard]:
                neg_features = self.dataset.get_symbol_features(neg_symbol)

                if neg_features is None:
                    continue

                # Get features at same date as positive sample
                date = sample["date"]
                if date not in neg_features.index:
                    continue

                # Extract temporal window
                window_start = date - pd.Timedelta(days=self.dataset.window_size - 1)
                temporal_window = neg_features.loc[window_start:date, self.dataset.TEMPORAL_COLS]

                if len(temporal_window) < self.dataset.window_size:
                    continue

                # Extract tabular snapshot
                tabular_cont_neg = neg_features.loc[date, self.dataset.TABULAR_CONT_COLS]
                tabular_cat_neg = neg_features.loc[date, self.dataset.TABULAR_CAT_COLS]

                hard_negatives_temporal.append(
                    torch.tensor(temporal_window.values, dtype=torch.float32)
                )
                hard_negatives_tabular_cont.append(
                    torch.tensor(tabular_cont_neg.values, dtype=torch.float32)
                )
                hard_negatives_tabular_cat.append(
                    torch.tensor(tabular_cat_neg.values, dtype=torch.long)
                )

        # Concatenate hard negatives if any found
        if len(hard_negatives_temporal) > 0:
            temporal_hard = torch.stack(hard_negatives_temporal)
            tabular_cont_hard = torch.stack(hard_negatives_tabular_cont)
            tabular_cat_hard = torch.stack(hard_negatives_tabular_cat)

            # Concatenate to positives
            temporal = torch.cat([temporal_pos, temporal_hard], dim=0)
            tabular_cont = torch.cat([tabular_cont_pos, tabular_cont_hard], dim=0)
            tabular_cat = torch.cat([tabular_cat_pos, tabular_cat_hard], dim=0)

            n_hard_actual = len(hard_negatives_temporal)
        else:
            # No hard negatives found, use positives only
            temporal = temporal_pos
            tabular_cont = tabular_cont_pos
            tabular_cat = tabular_cat_pos
            n_hard_actual = 0

        # Return batched structure with metadata
        batch = {
            "temporal": temporal,
            "tabular_cont": tabular_cont,
            "tabular_cat": tabular_cat,
            "batch_size": batch_size,
            "n_hard": n_hard_actual,
        }

        return batch
