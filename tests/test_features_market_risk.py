"""Test market risk features."""

import numpy as np
import pandas as pd
import pytest

from src.features.market_risk import MarketRiskFeatures


def test_market_risk_initialization():
    """Test market risk feature group initialization."""
    features = MarketRiskFeatures()
    assert features.name == "market_risk"
    assert "market_beta_60d" in features.get_feature_names()
    assert "downside_beta_60d" in features.get_feature_names()
    # FF5 factor loadings excluded from methodology
    assert "smb_loading" not in features.get_feature_names()


def test_beta_computation():
    """Test beta computation."""
    # Create synthetic data with known beta
    np.random.seed(42)
    n_days = 100

    # Market returns
    market_ret = np.random.normal(0.001, 0.02, n_days)

    # Stock returns with beta = 1.5
    beta_true = 1.5
    stock_ret = beta_true * market_ret + np.random.normal(0, 0.01, n_days)

    df = pd.DataFrame(
        {
            "symbol": ["TEST"] * n_days,
            "date": pd.date_range("2020-01-01", periods=n_days, freq="B"),
            "return": stock_ret,
            "market_return": market_ret,
        }
    )

    features = MarketRiskFeatures()
    result = features.compute(df)

    assert "market_beta_60d" in result.columns
    assert "downside_beta_60d" in result.columns
    # Beta should be approximately 1.5 for the last rows (after 60-day window)
    last_beta = result["market_beta_60d"].iloc[-1]
    assert not pd.isna(last_beta)
    assert 1.0 < last_beta < 2.0  # Should be close to 1.5


def test_downside_beta():
    """Test downside beta computation."""
    np.random.seed(42)
    n_days = 100

    df = pd.DataFrame(
        {
            "symbol": ["TEST"] * n_days,
            "date": pd.date_range("2020-01-01", periods=n_days, freq="B"),
            "return": np.random.normal(0.001, 0.02, n_days),
            "market_return": np.random.normal(0.001, 0.01, n_days),
        }
    )

    features = MarketRiskFeatures()
    result = features.compute(df)

    # Check downside beta column exists
    assert "downside_beta_60d" in result.columns
    # Note: FF5 factor loadings removed - using only market beta
    assert "smb_loading" not in result.columns
