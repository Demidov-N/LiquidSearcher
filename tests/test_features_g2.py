"""Test G2 volatility features."""

import numpy as np
import pandas as pd

from src.features.g2_volatility import G2VolatilityFeatures


def test_g2_initialization():
    """Test G2 feature group initialization."""
    g2 = G2VolatilityFeatures()
    assert g2.name == "G2_volatility"
    assert "realized_vol_20d" in g2.get_feature_names()


def test_volatility_computation():
    """Test volatility feature computation."""
    np.random.seed(42)
    n_days = 100

    # Create synthetic returns with known volatility
    target_vol = 0.30  # 30% annualized
    daily_vol = target_vol / np.sqrt(252)
    returns = np.random.normal(0.001, daily_vol, n_days)

    df = pd.DataFrame(
        {
            "symbol": ["TEST"] * n_days,
            "date": pd.date_range("2020-01-01", periods=n_days, freq="B"),
            "return": returns,
        }
    )

    g2 = G2VolatilityFeatures()
    result = g2.compute(df)

    assert "realized_vol_20d" in result.columns
    assert "realized_vol_60d" in result.columns

    # Check vol is roughly in expected range (annualized)
    last_vol = result["realized_vol_20d"].iloc[-1]
    assert not pd.isna(last_vol)
    assert 0.10 < last_vol < 0.50  # Should be close to 30%


def test_idiosyncratic_vol():
    """Test idiosyncratic volatility computation."""
    np.random.seed(42)
    n_days = 100

    df = pd.DataFrame(
        {
            "symbol": ["TEST"] * n_days,
            "date": pd.date_range("2020-01-01", periods=n_days, freq="B"),
            "return": np.random.normal(0.001, 0.02, n_days),
            "market_return": np.random.normal(0.001, 0.015, n_days),
        }
    )

    g2 = G2VolatilityFeatures()
    result = g2.compute(df)

    assert "idiosyncratic_vol" in result.columns
    assert result["idiosyncratic_vol"].iloc[-1] >= 0
