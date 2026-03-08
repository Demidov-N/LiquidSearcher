"""Tests for FeatureDataset PyTorch Dataset."""

import numpy as np
import pandas as pd
import torch

from src.training.dataset import FeatureDataset


def create_unified_features(feature_dir, symbols, dates):
    """Create unified parquet file with multiple symbols.

    Args:
        feature_dir: Directory to save parquet file
        symbols: List of stock symbols
        dates: Date range for each symbol
    """
    temporal_cols = FeatureDataset.TEMPORAL_COLS
    tabular_cols = FeatureDataset.TABULAR_CONT_COLS
    cat_cols = FeatureDataset.TABULAR_CAT_COLS

    all_data = []

    for symbol in symbols:
        # Create temporal columns (13 features)
        temporal_data = {col: np.random.randn(len(dates)) for col in temporal_cols}

        # Create tabular continuous columns (15 features)
        tabular_data = {col: np.random.randn(len(dates)) for col in tabular_cols}

        # Create categorical columns with symbol-specific values
        cat_data = {
            "gsector": np.full(len(dates), 40 + hash(symbol) % 10),  # Different sectors
            "ggroup": np.full(len(dates), 4500 + hash(symbol) % 50),  # Different groups
        }

        # Combine all data
        symbol_data = {**temporal_data, **tabular_data, **cat_data}
        df = pd.DataFrame(symbol_data, index=dates)
        df.index.name = "date"
        df["symbol"] = symbol
        df = df.reset_index()
        all_data.append(df)

    # Concatenate all symbols and save as unified file
    unified_df = pd.concat(all_data, ignore_index=True)
    unified_df.to_parquet(feature_dir / "all_features.parquet", index=False)
    return unified_df


def test_dataset_initialization(tmp_path):
    """Test dataset can be initialized with unified parquet file."""
    # Create mock feature directory
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()

    # Create unified parquet with multiple symbols
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    create_unified_features(feature_dir, ["AAPL", "MSFT", "GOOGL"], dates)

    dataset = FeatureDataset(
        feature_dir=str(feature_dir),
        date_range=("2020-06-01", "2020-12-31"),
    )

    assert len(dataset) > 0
    assert len(dataset.symbols) == 3


def test_dataset_getitem(tmp_path):
    """Test dataset returns correct structure from unified parquet."""
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()

    # Create unified parquet with multiple symbols
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    create_unified_features(feature_dir, ["AAPL", "MSFT"], dates)

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


def test_dataset_filter_by_symbol(tmp_path):
    """Test dataset can filter and load specific symbols."""
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()

    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    create_unified_features(feature_dir, ["AAPL", "MSFT", "GOOGL", "TSLA"], dates)

    # Load only specific symbols
    dataset = FeatureDataset(
        feature_dir=str(feature_dir),
        date_range=("2020-06-01", "2020-12-31"),
        symbols=["AAPL", "MSFT"],
        window_size=60,
    )

    assert len(dataset.symbols) == 2
    assert "AAPL" in dataset.symbols
    assert "MSFT" in dataset.symbols
    assert "GOOGL" not in dataset.symbols
    assert "TSLA" not in dataset.symbols

    # Verify samples contain only selected symbols
    for i in range(min(10, len(dataset))):
        sample = dataset[i]
        assert sample["symbol"] in ["AAPL", "MSFT"]


def test_dataset_column_requirements(tmp_path):
    """Test that unified parquet has all required columns."""
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()

    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    df = create_unified_features(feature_dir, ["AAPL"], dates)

    # Verify all required columns exist
    required_cols = (
        FeatureDataset.TEMPORAL_COLS
        + FeatureDataset.TABULAR_CONT_COLS
        + FeatureDataset.TABULAR_CAT_COLS
        + ["symbol", "date"]
    )

    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"

    # Verify column counts
    assert len(FeatureDataset.TEMPORAL_COLS) == 13
    assert len(FeatureDataset.TABULAR_CONT_COLS) == 15
    assert len(FeatureDataset.TABULAR_CAT_COLS) == 2


def test_get_symbol_features(tmp_path):
    """Test loading features for a specific symbol."""
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()

    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    create_unified_features(feature_dir, ["AAPL", "MSFT"], dates)

    dataset = FeatureDataset(
        feature_dir=str(feature_dir),
        date_range=("2020-06-01", "2020-12-31"),
        window_size=60,
    )

    # Test get_symbol_features
    aapl_features = dataset.get_symbol_features("AAPL")
    assert aapl_features is not None
    assert "symbol" in aapl_features.columns
    assert all(aapl_features["symbol"] == "AAPL")

    msft_features = dataset.get_symbol_features("MSFT")
    assert msft_features is not None
    assert all(msft_features["symbol"] == "MSFT")

    # Test non-existent symbol
    tsla_features = dataset.get_symbol_features("TSLA")
    assert tsla_features is None
