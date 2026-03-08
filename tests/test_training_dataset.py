"""Tests for FeatureDataset PyTorch Dataset."""

import numpy as np
import pandas as pd
import torch

from src.training.dataset import FeatureDataset


def test_dataset_initialization(tmp_path):
    """Test dataset can be initialized with feature directory."""
    # Create mock feature file
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()

    # Create sample features for AAPL with correct column names
    dates = pd.date_range("2020-01-01", periods=300, freq="B")

    # Use exact column names from FeatureDataset
    temporal_cols = FeatureDataset.TEMPORAL_COLS
    tabular_cols = FeatureDataset.TABULAR_CONT_COLS
    cat_cols = FeatureDataset.TABULAR_CAT_COLS

    # Create temporal columns (13 features)
    temporal_data = {col: np.random.randn(300) for col in temporal_cols}

    # Create tabular continuous columns (15 features)
    tabular_data = {col: np.random.randn(300) for col in tabular_cols}

    # Create categorical columns
    cat_data = {
        "gsector": np.full(300, 45),
        "ggroup": np.full(300, 4510),
    }

    # Combine all data
    all_data = {**temporal_data, **tabular_data, **cat_data}

    features = pd.DataFrame(all_data, index=dates)
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

    # Use exact column names from FeatureDataset
    temporal_cols = FeatureDataset.TEMPORAL_COLS
    tabular_cols = FeatureDataset.TABULAR_CONT_COLS
    cat_cols = FeatureDataset.TABULAR_CAT_COLS

    # Create temporal columns (13 features)
    temporal_data = {col: np.random.randn(300) for col in temporal_cols}

    # Create tabular continuous columns (15 features)
    tabular_data = {col: np.random.randn(300) for col in tabular_cols}

    # Create categorical columns
    cat_data = {
        "gsector": np.full(300, 45),
        "ggroup": np.full(300, 4510),
    }

    # Combine all data
    all_data = {**temporal_data, **tabular_data, **cat_data}

    features = pd.DataFrame(all_data, index=dates)
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
