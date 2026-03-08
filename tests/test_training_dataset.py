"""Tests for FeatureDataset PyTorch Dataset."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import torch

from src.training.dataset import FeatureDataset


def test_dataset_initialization(tmp_path):
    """Test dataset can be initialized with feature directory."""
    # Create mock feature file
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()

    # Create sample features for AAPL
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    features = pd.DataFrame(
        {
            "z_score_20d": np.random.randn(300),
            "market_beta_60d": np.random.randn(300),
            "gsector": np.full(300, 45),
            "ggroup": np.full(300, 4510),
        },
        index=dates,
    )
    features.index.name = "date"
    features.to_parquet(feature_dir / "AAPL_features.parquet")

    dataset = FeatureDataset(
        feature_dir=str(feature_dir),
        date_range=("2020-06-01", "2020-12-31"),
    )

    assert len(dataset) > 0


def test_dataset_getitem(tmp_path):
    """Test dataset returns correct structure."""
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()

    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    features = pd.DataFrame(
        {
            "z_score_20d": np.random.randn(300),
            "z_score_60d": np.random.randn(300),
            "ma_ratio_20_60": np.random.randn(300),
            "volume_trend": np.random.randn(300),
            "volatility_20d": np.random.randn(300),
            "momentum_20d": np.random.randn(300),
            "momentum_60d": np.random.randn(300),
            "momentum_252d": np.random.randn(300),
            "market_beta_60d": np.random.randn(300),
            "downside_beta_60d": np.random.randn(300),
            "realized_vol_20d": np.random.randn(300),
            "realized_vol_60d": np.random.randn(300),
            "pe_ratio": np.random.randn(300),
            "pb_ratio": np.random.randn(300),
            "roe": np.random.randn(300),
            "market_cap_log": np.random.randn(300),
            "parkinson_vol": np.random.randn(300),
            "garch_vol": np.random.randn(300),
            "momentum_ratio": np.random.randn(300),
            "reversal_5d": np.random.randn(300),
            "price_trend": np.random.randn(300),
            "gsector": np.full(300, 45),
            "ggroup": np.full(300, 4510),
        },
        index=dates,
    )
    features.index.name = "date"
    features.to_parquet(feature_dir / "AAPL_features.parquet")

    dataset = FeatureDataset(
        feature_dir=str(feature_dir),
        date_range=("2020-06-01", "2020-12-31"),
        window_size=60,
    )

    sample = dataset[0]

    assert "symbol" in sample
    assert "date" in sample
    assert "temporal" in sample
    assert "tabular_cont" in sample
    assert "tabular_cat" in sample
    assert "beta" in sample
    assert "gsector" in sample
    assert "ggroup" in sample

    # Check shapes
    assert sample["temporal"].shape == (60, 13)
    assert sample["tabular_cont"].shape == (15,)
    assert sample["tabular_cat"].shape == (2,)
