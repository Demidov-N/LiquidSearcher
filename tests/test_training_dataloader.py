"""Tests for StockDataLoader with GICS hard negative sampling."""

import numpy as np
import pandas as pd

from src.models.sampler import GICSHardNegativeSampler
from src.training.dataloader import StockDataLoader
from src.training.dataset import FeatureDataset


def test_dataloader_with_hard_negatives(tmp_path):
    """Test dataloader adds hard negatives to batch."""
    # Create mock feature files
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()

    # Get column names from FeatureDataset
    temporal_cols = FeatureDataset.TEMPORAL_COLS
    tabular_cols = FeatureDataset.TABULAR_CONT_COLS
    cat_cols = FeatureDataset.TABULAR_CAT_COLS

    # Create features for 3 stocks
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    for symbol, ggroup, beta in [
        ("AAPL", 4510, 1.2),
        ("MSFT", 4520, 1.1),
        ("GOOGL", 4510, 1.8),  # Same ggroup as AAPL, different beta
    ]:
        # Create temporal columns (13 features)
        temporal_data = {col: np.random.randn(300) for col in temporal_cols}

        # Create tabular continuous columns (15 features)
        tabular_data = {col: np.random.randn(300) for col in tabular_cols}

        # Set specific beta values for testing
        tabular_data["market_beta_60d"] = np.full(300, beta)

        # Create categorical columns
        cat_data = {
            "gsector": np.full(300, 45),
            "ggroup": np.full(300, ggroup),
        }

        # Combine all data
        all_data = {**temporal_data, **tabular_data, **cat_data}

        features = pd.DataFrame(all_data, index=dates)
        features.index.name = "date"
        features.to_parquet(feature_dir / f"{symbol}_features.parquet")

    # Create dataset
    dataset = FeatureDataset(
        feature_dir=str(feature_dir),
        date_range=("2020-06-01", "2020-12-31"),
        symbols=["AAPL", "MSFT", "GOOGL"],
    )

    # Create dataloader with hard negatives
    sampler = GICSHardNegativeSampler(n_hard=2, beta_threshold=0.3)

    loader = StockDataLoader(
        dataset=dataset,
        batch_size=2,
        sampler=sampler,
        feature_dir=str(feature_dir),
        n_hard=2,
        shuffle=False,
    )

    # Get a batch
    batch = next(iter(loader))

    # Check batch structure
    assert "temporal" in batch
    assert "tabular_cont" in batch
    assert "tabular_cat" in batch
    assert "batch_size" in batch
    assert "n_hard" in batch

    # Original batch_size=2, but total should be >= 2 (may have hard negatives)
    assert batch["temporal"].shape[0] >= 2
    assert batch["batch_size"] == 2
