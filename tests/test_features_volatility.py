"""Test volatility features."""

import numpy as np
import pandas as pd
import pytest

from src.features.volatility import VolatilityFeatures


def test_volatility_initialization():
    """Test volatility feature group initialization."""
    features = VolatilityFeatures()
    assert features.name == "volatility"
    assert "realized_vol_20d" in features.get_feature_names()
    assert "realized_vol_60d" in features.get_feature_names()


def test_realized_volatility():
    """Test realized volatility computation."""
    np.random.seed(42)
    n_days = 100

    # Generate returns with known volatility
    vol_target = 0.20  # 20% annualized volatility
    daily_vol = vol_target / np.sqrt(252)
    returns = np.random.normal(0, daily_vol, n_days)

    df = pd.DataFrame(
        {
            "symbol": ["TEST"] * n_days,
            "date": pd.date_range("2020-01-01", periods=n_days, freq="B"),
            "return": returns,
        }
    )

    features = VolatilityFeatures()
    result = features.compute(df)

    # Check volatility columns exist
    assert "realized_vol_20d" in result.columns
    assert "realized_vol_60d" in result.columns

    # Last values should not be NaN (after window fills)
    assert not pd.isna(result["realized_vol_20d"].iloc[-1])


def test_idiosyncratic_volatility():
    """Test idiosyncratic volatility computation."""
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

    features = VolatilityFeatures()
    result = features.compute(df)

    # Check idiosyncratic volatility column exists
    assert "idiosyncratic_vol" in result.columns
