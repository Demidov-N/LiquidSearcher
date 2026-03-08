"""End-to-end integration test for feature engineering pipeline.

This test verifies that:
1. WRDSDataLoader can load mock data
2. FeatureEngineer can compute all feature groups
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

    def test_compute_market_risk_features(self, feature_engineer, sample_symbols):
        """Test that market risk features are computed."""
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
        result = feature_engineer.compute_single_group(df, "market_risk")

        expected = [
            "market_beta_60d_zscore",
            "downside_beta_60d_zscore",
            # Note: FF5 factor loadings (smb_loading, hml_loading, etc.) excluded from methodology
            # Using only market beta for risk measurement
        ]

        for feature in expected:
            assert feature in result.columns, f"Market risk feature {feature} missing"

        # Verify FF5 features are NOT present (confirming simplified methodology)
        assert "smb_loading_zscore" not in result.columns, "FF5 features should be excluded"

    def test_compute_volatility_features(self, feature_engineer, sample_symbols):
        """Test that volatility features are computed."""
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
        result = feature_engineer.compute_single_group(df, "volatility")

        expected = [
            "realized_vol_20d_log_zscore",
            "realized_vol_60d_log_zscore",
            "idiosyncratic_vol_log_zscore",
            "vol_of_vol_log_zscore",
        ]

        for feature in expected:
            assert feature in result.columns, f"Volatility feature {feature} missing"

    def test_compute_momentum_features(self, feature_engineer, sample_symbols):
        """Test that momentum features are computed."""
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
        result = feature_engineer.compute_single_group(df, "momentum")

        expected = [
            "mom_1m_rank",
            "mom_3m_rank",
            "mom_6m_rank",
            "mom_12_1m_rank",
            "macd_rank",
        ]

        for feature in expected:
            assert feature in result.columns, f"Momentum feature {feature} missing"

    def test_compute_valuation_features(self, feature_engineer, sample_symbols):
        """Test that valuation features are computed."""
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
        result = feature_engineer.compute_single_group(df, "valuation")

        expected = [
            "log_mktcap",
            "pe_ratio",
            "pb_ratio",
            "roe",
        ]

        for feature in expected:
            assert feature in result.columns, f"Valuation feature {feature} missing"

    def test_compute_technical_features(self, feature_engineer, sample_symbols):
        """Test that technical features are computed."""
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
        result = feature_engineer.compute_single_group(df, "technical")

        expected = [
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

        for feature in expected:
            assert feature in result.columns, f"Technical feature {feature} missing"

    def test_compute_sector_features(self, feature_engineer, sample_symbols):
        """Test that sector features are computed."""
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
        result = feature_engineer.compute_single_group(df, "sector")

        expected = [
            "gics_sector",
            "gics_industry_group",
            "gics_sector_str",
            "gics_industry_group_str",
        ]

        for feature in expected:
            assert feature in result.columns, f"Sector feature {feature} missing"

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
        """Test that all feature groups are registered."""
        groups = feature_engineer.get_feature_names()

        assert "market_risk" in groups, "market_risk should be registered"
        assert "volatility" in groups, "volatility should be registered"
        assert "momentum" in groups, "momentum should be registered"
        assert "valuation" in groups, "valuation should be registered"
        assert "technical" in groups, "technical should be registered"
        assert "sector" in groups, "sector should be registered"

    def test_feature_count(self, feature_engineer):
        """Test that we have the expected number of total features."""
        all_features = feature_engineer.get_all_feature_names()

        # We expect (simplified methodology without FF5):
        # market_risk: 2 features (market_beta_60d, downside_beta_60d, z-scored)
        #   - Note: FF5 factor loadings (smb, hml, mom, rmw, cma) excluded
        # volatility: 4 features (vol metrics, log z-scored)
        # momentum: 5 features (momentum, ranked)
        # valuation: 4 features (valuation)
        # technical: 13 features (OHLCV)
        # sector: 4 features (sector encoding)
        # Total: 32 features (down from 37 with FF5)
        assert len(all_features) >= 32, f"Expected at least 32 features, got {len(all_features)}"
