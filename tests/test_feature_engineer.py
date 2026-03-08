"""Test unified feature engineer orchestrating G1-G6 computation."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.cache_manager import CacheManager
from src.features.engineer import FeatureEngineer


def test_engineer_initialization():
    """Test that all 6 feature groups are registered on init."""
    engineer = FeatureEngineer()

    # Check all 6 groups are registered
    groups = engineer.get_feature_names()

    # Count features per group
    assert "G1" in str(groups), "G1 features should be present"
    assert "G2" in str(groups), "G2 features should be present"
    assert "G3" in str(groups), "G3 features should be present"
    assert "G4" in str(groups), "G4 features should be present"
    assert "G5" in str(groups), "G5 features should be present"
    assert "G6" in str(groups), "G6 features should be present"

    # Verify total feature count (at minimum, should have all 6 groups)
    all_features = engineer.get_all_feature_names()
    assert len(all_features) > 0, "Should have features registered"


def test_engineer_compute_features():
    """Test computing features for all G1-G6 groups."""
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

    # Verify all 6 groups added features
    g1_features = [
        "market_beta_60d_zscore",
        "downside_beta_60d_zscore",
        "smb_loading_zscore",
        "hml_loading_zscore",
        "mom_loading_zscore",
        "rmw_loading_zscore",
        "cma_loading_zscore",
    ]

    for col in g1_features:
        assert col in result.columns, f"G1 feature {col} should be present"

    g2_features = [
        "realized_vol_20d_log_zscore",
        "realized_vol_60d_log_zscore",
        "idiosyncratic_vol_log_zscore",
        "vol_of_vol_log_zscore",
    ]

    for col in g2_features:
        assert col in result.columns, f"G2 feature {col} should be present"

    g3_features = [
        "mom_1m_rank",
        "mom_3m_rank",
        "mom_6m_rank",
        "mom_12_1m_rank",
        "macd_rank",
    ]

    for col in g3_features:
        assert col in result.columns, f"G3 feature {col} should be present"

    g4_features = [
        "log_mktcap",
        "pe_ratio",
        "pb_ratio",
        "roe",
    ]

    for col in g4_features:
        assert col in result.columns, f"G4 feature {col} should be present"

    g5_features = [
        "z_close_5d",
        "z_close_10d",
        "z_close_20d",
        "z_high",
        "z_low",
        "z_volume_5d",
        "z_volume_10d",
        "z_volume_20d",
        "ma_ratio_5",
        "ma_ratio_10",
        "ma_ratio_15",
        "ma_ratio_20",
        "ma_ratio_25",
    ]

    for col in g5_features:
        assert col in result.columns, f"G5 feature {col} should be present"

    g6_features = [
        "gics_sector",
        "gics_industry_group",
        "gics_sector_str",
        "gics_industry_group_str",
    ]

    for col in g6_features:
        assert col in result.columns, f"G6 feature {col} should be present"


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
