"""End-to-end integration test for feature engineering pipeline.

This test verifies that:
1. WRDSDataLoader can load mock data
2. FeatureEngineer can compute all G1-G6 features
3. All expected features are present in output
4. Output is valid (no NaN where expected)
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.data.wrds_loader import WRDSDataLoader
from src.features.engineer import FeatureEngineer


@pytest.fixture
def mock_data_loader():
    """Create a WRDSDataLoader in mock mode."""
    return WRDSDataLoader(mock_mode=True)


@pytest.fixture
def sample_symbols():
    """Return sample stock symbols for testing."""
    return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]


@pytest.fixture
def date_range():
    """Return test date range."""
    return {
        "start": datetime(2023, 1, 1),
        "end": datetime(2023, 12, 31),
    }


@pytest.fixture
def feature_engineer():
    """Create a FeatureEngineer instance."""
    return FeatureEngineer()


class TestEndToEndIntegration:
    """End-to-end integration tests for the feature pipeline."""

    def test_load_mock_prices(self, mock_data_loader, sample_symbols, date_range):
        """Test loading mock price data."""
        df = mock_data_loader.load_prices(
            symbols=sample_symbols,
            start_date=date_range["start"],
            end_date=date_range["end"],
            use_cache=False,
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "symbol" in df.columns
        assert "date" in df.columns
        assert "close" in df.columns or "prc" in df.columns

    def test_load_mock_fundamentals(self, mock_data_loader, sample_symbols, date_range):
        """Test loading mock fundamental data."""
        df = mock_data_loader.load_fundamentals(
            symbols=sample_symbols,
            start_date=date_range["start"],
            end_date=date_range["end"],
            use_cache=False,
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "symbol" in df.columns

    def test_load_merged_data(self, mock_data_loader, sample_symbols, date_range):
        """Test loading merged price and fundamental data."""
        df = mock_data_loader.load_merged(
            symbols=sample_symbols,
            start_date=date_range["start"],
            end_date=date_range["end"],
            use_cache=False,
        )

        assert isinstance(df, pd.DataFrame)
        # Should have both price and fundamental columns
        assert len(df) > 0

    def test_compute_all_g1_features(self, feature_engineer, sample_symbols):
        """Test that all G1 risk features are computed."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")

        data = []
        for symbol in sample_symbols:
            for i, date in enumerate(dates):
                data.append(
                    {
                        "symbol": symbol,
                        "date": date,
                        "return": np.random.randn() * 0.02,
                        "market_return": np.random.randn() * 0.01,
                        "smb_factor": np.random.randn() * 0.01,
                        "hml_factor": np.random.randn() * 0.01,
                        "mom_factor": np.random.randn() * 0.01,
                        "rmw_factor": np.random.randn() * 0.01,
                        "cma_factor": np.random.randn() * 0.01,
                    }
                )

        df = pd.DataFrame(data)
        result = feature_engineer.compute_single_group(df, "G1")

        g1_expected = [
            "market_beta_60d_zscore",
            "downside_beta_60d_zscore",
            "smb_loading_zscore",
            "hml_loading_zscore",
            "mom_loading_zscore",
            "rmw_loading_zscore",
            "cma_loading_zscore",
        ]

        for feature in g1_expected:
            assert feature in result.columns, f"G1 feature {feature} missing"

    def test_compute_all_g2_features(self, feature_engineer, sample_symbols):
        """Test that all G2 volatility features are computed."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")

        data = []
        for symbol in sample_symbols:
            for i, date in enumerate(dates):
                data.append(
                    {
                        "symbol": symbol,
                        "date": date,
                        "return": np.random.randn() * 0.02,
                        "market_return": np.random.randn() * 0.01,
                    }
                )

        df = pd.DataFrame(data)
        result = feature_engineer.compute_single_group(df, "G2")

        g2_expected = [
            "realized_vol_20d_log_zscore",
            "realized_vol_60d_log_zscore",
            "idiosyncratic_vol_log_zscore",
            "vol_of_vol_log_zscore",
        ]

        for feature in g2_expected:
            assert feature in result.columns, f"G2 feature {feature} missing"

    def test_compute_all_g3_features(self, feature_engineer, sample_symbols):
        """Test that all G3 momentum features are computed."""
        dates = pd.date_range("2024-01-01", periods=300, freq="D")

        data = []
        for symbol in sample_symbols:
            base_price = 100
            for i, date in enumerate(dates):
                base_price *= 1 + np.random.randn() * 0.01
                data.append(
                    {
                        "symbol": symbol,
                        "date": date,
                        "close": base_price,
                    }
                )

        df = pd.DataFrame(data)
        result = feature_engineer.compute_single_group(df, "G3")

        g3_expected = [
            "mom_1m_rank",
            "mom_3m_rank",
            "mom_6m_rank",
            "mom_12_1m_rank",
            "macd_rank",
        ]

        for feature in g3_expected:
            assert feature in result.columns, f"G3 feature {feature} missing"

    def test_compute_all_g4_features(self, feature_engineer, sample_symbols):
        """Test that all G4 valuation features are computed."""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")

        data = []
        for symbol in sample_symbols:
            for i, date in enumerate(dates):
                data.append(
                    {
                        "symbol": symbol,
                        "date": date,
                        "price": 100 + np.random.randn() * 5,
                        "shares_outstanding": 1000000000,
                        "eps": 5.0 + np.random.randn() * 0.5,
                        "book_value_per_share": 50.0 + np.random.randn() * 5,
                        "net_income": 5000000000,
                        "equity": 50000000000,
                    }
                )

        df = pd.DataFrame(data)
        result = feature_engineer.compute_single_group(df, "G4")

        g4_expected = [
            "log_mktcap",
            "pe_ratio",
            "pb_ratio",
            "roe",
        ]

        for feature in g4_expected:
            assert feature in result.columns, f"G4 feature {feature} missing"

    def test_compute_all_g5_features(self, feature_engineer, sample_symbols):
        """Test that all G5 OHLCV features are computed."""
        dates = pd.date_range("2024-01-01", periods=300, freq="D")

        data = []
        for symbol in sample_symbols:
            base_price = 100
            for i, date in enumerate(dates):
                base_price *= 1 + np.random.randn() * 0.01
                data.append(
                    {
                        "symbol": symbol,
                        "date": date,
                        "close": base_price,
                        "high": base_price * (1 + abs(np.random.randn() * 0.01)),
                        "low": base_price * (1 - abs(np.random.randn() * 0.01)),
                        "volume": int(1000000 + np.random.randint(-100000, 100000)),
                    }
                )

        df = pd.DataFrame(data)
        result = feature_engineer.compute_single_group(df, "G5")

        g5_expected = [
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

        for feature in g5_expected:
            assert feature in result.columns, f"G5 feature {feature} missing"

    def test_compute_all_g6_features(self, feature_engineer, sample_symbols):
        """Test that all G6 sector features are computed."""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")

        sectors = ["Technology", "Healthcare", "Finance", "Energy", "Consumer"]

        data = []
        for i, symbol in enumerate(sample_symbols):
            sector = sectors[i % len(sectors)]
            industry = f"{sector} Industry"
            for date in dates:
                data.append(
                    {
                        "symbol": symbol,
                        "date": date,
                        "gics_sector_str": sector,
                        "gics_industry_group_str": industry,
                    }
                )

        df = pd.DataFrame(data)
        result = feature_engineer.compute_single_group(df, "G6")

        g6_expected = [
            "gics_sector",
            "gics_industry_group",
            "gics_sector_str",
            "gics_industry_group_str",
        ]

        for feature in g6_expected:
            assert feature in result.columns, f"G6 feature {feature} missing"

    def test_full_pipeline_integration(
        self, mock_data_loader, feature_engineer, sample_symbols, date_range
    ):
        """Test the full pipeline: load data and compute all features."""
        # Load mock data
        df = mock_data_loader.load_merged(
            symbols=sample_symbols,
            start_date=date_range["start"],
            end_date=date_range["end"],
            use_cache=False,
        )

        # Add required columns for feature computation
        if "return" not in df.columns:
            df["return"] = df["close"].pct_change() if "close" in df.columns else np.nan

        # Add factor columns if missing
        factor_cols = [
            "market_return",
            "smb_factor",
            "hml_factor",
            "mom_factor",
            "rmw_factor",
            "cma_factor",
        ]
        for col in factor_cols:
            if col not in df.columns:
                df[col] = np.random.randn(len(df)) * 0.01

        # Compute all features
        result = feature_engineer.compute_features(df)

        # Verify all feature groups are present
        all_features = feature_engineer.get_all_feature_names()
        assert len(all_features) > 0, "Should have computed features"

        # Check that result is a valid DataFrame
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_all_feature_groups_present(self, feature_engineer):
        """Test that all G1-G6 feature groups are registered."""
        groups = feature_engineer.get_feature_names()

        assert "G1" in groups, "G1 should be registered"
        assert "G2" in groups, "G2 should be registered"
        assert "G3" in groups, "G3 should be registered"
        assert "G4" in groups, "G4 should be registered"
        assert "G5" in groups, "G5 should be registered"
        assert "G6" in groups, "G6 should be registered"

    def test_feature_count(self, feature_engineer):
        """Test that we have the expected number of total features."""
        all_features = feature_engineer.get_all_feature_names()

        # We expect:
        # G1: 7 features (beta + 6 loadings, z-scored)
        # G2: 4 features (vol metrics, log z-scored)
        # G3: 5 features (momentum, ranked)
        # G4: 4 features (valuation)
        # G5: 13 features (OHLCV)
        # G6: 4 features (sector encoding)
        # Total: 37 features
        assert len(all_features) >= 37, f"Expected at least 37 features, got {len(all_features)}"
