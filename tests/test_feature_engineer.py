"""Test unified feature engineer with configurable feature groups."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.cache_manager import CacheManager
from src.features.engineer import FeatureEngineer


def test_engineer_initialization():
    """Test that feature groups are registered on init."""
    engineer = FeatureEngineer()

    # Check all groups are registered by default
    groups = engineer.get_feature_names()

    assert "market_risk" in groups
    assert "volatility" in groups
    assert "momentum" in groups
    assert "valuation" in groups
    assert "technical" in groups
    assert "sector" in groups


def test_engineer_with_selected_groups():
    """Test engineer with only selected feature groups."""
    engineer = FeatureEngineer(enabled_groups=["market_risk", "momentum", "sector"])

    groups = engineer.get_feature_names()
    assert "market_risk" in groups
    assert "momentum" in groups
    assert "sector" in groups
    assert "volatility" not in groups
    assert "valuation" not in groups
    assert "technical" not in groups


def test_engineer_list_available_groups():
    """Test listing available feature groups."""
    engineer = FeatureEngineer()

    available = engineer.list_available_groups()
    assert "market_risk" in available
    assert "volatility" in available
    assert "momentum" in available
    assert "valuation" in available
    assert "technical" in available
    assert "sector" in available


def test_engineer_add_remove_groups():
    """Test adding and removing feature groups."""
    engineer = FeatureEngineer(enabled_groups=["market_risk"])

    # Add a group
    engineer.add_group("momentum")
    assert "momentum" in engineer.list_enabled_groups()

    # Remove a group
    engineer.remove_group("market_risk")
    assert "market_risk" not in engineer.list_enabled_groups()


def test_engineer_compute_features():
    """Test computing features for all enabled groups."""
    engineer = FeatureEngineer()

    # Create test data with required columns for all groups
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    symbols = ["AAPL", "MSFT"]

    data = []
    for symbol in symbols:
        for i, date in enumerate(dates):
            data.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "return": np.random.randn() * 0.02,
                    "market_return": np.random.randn() * 0.01,
                    "close": 100 + i + np.random.randn() * 5,
                    "high": 102 + i + np.random.randn() * 5,
                    "low": 98 + i + np.random.randn() * 5,
                    "volume": 1000000 + np.random.randint(-100000, 100000),
                    "price": 100 + i,
                    "shares_outstanding": 1000000000,
                    "eps": 5.0,
                    "book_value_per_share": 50.0,
                    "net_income": 5000000000,
                    "equity": 50000000000,
                    "gics_sector_str": "Technology",
                    "gics_industry_group_str": "Software",
                    "smb_factor": np.random.randn() * 0.01,
                    "hml_factor": np.random.randn() * 0.01,
                    "mom_factor": np.random.randn() * 0.01,
                    "rmw_factor": np.random.randn() * 0.01,
                    "cma_factor": np.random.randn() * 0.01,
                }
            )

    df = pd.DataFrame(data)

    # Compute all features
    result = engineer.compute_features(df)

    # Verify all groups added features
    market_risk_features = [
        "market_beta_60d_zscore",
        "downside_beta_60d_zscore",
    ]

    for col in market_risk_features:
        assert col in result.columns, f"Market risk feature {col} should be present"

    volatility_features = [
        "realized_vol_20d_log_zscore",
    ]

    for col in volatility_features:
        assert col in result.columns, f"Volatility feature {col} should be present"


def test_engineer_caching_works():
    """Test that parquet caching works correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CacheManager(cache_dir=Path(tmpdir))
        engineer = FeatureEngineer(cache_manager=cache)

        # Create minimal test data
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        symbols = ["AAPL", "MSFT"]

        data = []
        for symbol in symbols:
            for i, date in enumerate(dates):
                data.append(
                    {
                        "symbol": symbol,
                        "date": date,
                        "return": np.random.randn() * 0.02,
                        "market_return": np.random.randn() * 0.01,
                        "close": 100 + i,
                        "high": 102 + i,
                        "low": 98 + i,
                        "volume": 1000000,
                        "price": 100 + i,
                        "shares_outstanding": 1000000000,
                        "eps": 5.0,
                        "book_value_per_share": 50.0,
                        "net_income": 5000000000,
                        "equity": 50000000000,
                        "gics_sector_str": "Technology",
                        "gics_industry_group_str": "Software",
                        "smb_factor": np.random.randn() * 0.01,
                        "hml_factor": np.random.randn() * 0.01,
                        "mom_factor": np.random.randn() * 0.01,
                        "rmw_factor": np.random.randn() * 0.01,
                        "cma_factor": np.random.randn() * 0.01,
                    }
                )

        df = pd.DataFrame(data)
        cache_key = "test_features_v1"

        # First computation should cache
        result1 = engineer.compute_features(df, cache_key=cache_key)

        # Second computation should use cache
        result2 = engineer.compute_features(df, cache_key=cache_key)

        # Verify results are the same
        pd.testing.assert_frame_equal(result1, result2)

        # Verify cache file exists
        assert cache.exists(cache_key), "Cache file should exist"
