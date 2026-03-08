"""Test G3 momentum features."""

import numpy as np
import pandas as pd
import pytest

from src.features.g3_momentum import G3MomentumFeatures


def test_g3_initialization():
    """Test G3 momentum feature group initialization."""
    g3 = G3MomentumFeatures()
    assert g3.name == "G3_momentum"
    feature_names = g3.get_feature_names()
    assert "mom_1m" in feature_names
    assert "mom_3m" in feature_names
    assert "mom_6m" in feature_names
    assert "mom_12_1m" in feature_names
    assert "macd" in feature_names


def test_momentum_computation():
    """Test momentum computation with upward trending prices."""
    np.random.seed(42)
    n_days = 300  # Need at least 252 days for 12-1m momentum

    # Create upward trending price series
    price = 100.0
    prices = []
    for i in range(n_days):
        price = price * (1 + np.random.normal(0.0005, 0.02))
        prices.append(price)

    df = pd.DataFrame(
        {
            "symbol": ["TEST"] * n_days,
            "date": pd.date_range("2020-01-01", periods=n_days, freq="B"),
            "close": prices,
        }
    )

    g3 = G3MomentumFeatures()
    result = g3.compute(df)

    # Check all momentum features exist
    assert "mom_1m" in result.columns
    assert "mom_3m" in result.columns
    assert "mom_6m" in result.columns
    assert "mom_12_1m" in result.columns

    # Check ranked versions exist (normalized to [0, 1])
    assert "mom_1m_rank" in result.columns
    assert "mom_3m_rank" in result.columns
    assert "mom_6m_rank" in result.columns
    assert "mom_12_1m_rank" in result.columns

    # For upward trending prices, momentum should be positive (after sufficient data)
    # Get the last valid values
    last_1m = result["mom_1m"].dropna().iloc[-1]
    last_3m = result["mom_3m"].dropna().iloc[-1]
    last_6m = result["mom_6m"].dropna().iloc[-1]

    assert last_1m > 0, "1-month momentum should be positive for upward trend"
    assert last_3m > 0, "3-month momentum should be positive for upward trend"
    assert last_6m > 0, "6-month momentum should be positive for upward trend"


def test_macd_computation():
    """Test MACD computation."""
    np.random.seed(42)
    n_days = 100

    # Create price series with some trend
    price = 100.0
    prices = []
    for i in range(n_days):
        price = price * (1 + np.random.normal(0.0005, 0.02))
        prices.append(price)

    df = pd.DataFrame(
        {
            "symbol": ["TEST"] * n_days,
            "date": pd.date_range("2020-01-01", periods=n_days, freq="B"),
            "close": prices,
        }
    )

    g3 = G3MomentumFeatures()
    result = g3.compute(df)

    # Check MACD feature exists
    assert "macd" in result.columns
    assert "macd_rank" in result.columns

    # MACD values should exist (may be NaN at first due to EMA warmup)
    assert "macd" in result.columns
