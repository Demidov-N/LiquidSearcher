"""Test momentum features."""

import numpy as np
import pandas as pd
import pytest

from src.features.momentum import MomentumFeatures


def test_momentum_initialization():
    """Test momentum feature group initialization."""
    features = MomentumFeatures()
    assert features.name == "momentum"
    assert "mom_1m" in features.get_feature_names()
    assert "macd" in features.get_feature_names()


def test_momentum_computation():
    """Test momentum computation."""
    n_days = 300

    # Generate trending prices
    base_price = 100
    prices = [base_price]
    for _ in range(1, n_days):
        prices.append(prices[-1] * (1 + np.random.normal(0.001, 0.01)))

    df = pd.DataFrame(
        {
            "symbol": ["TEST"] * n_days,
            "date": pd.date_range("2020-01-01", periods=n_days, freq="B"),
            "close": prices,
        }
    )

    features = MomentumFeatures()
    result = features.compute(df)

    # Check momentum columns exist
    assert "mom_1m" in result.columns
    assert "mom_3m" in result.columns
    assert "mom_6m" in result.columns
    assert "mom_12_1m" in result.columns
    assert "macd" in result.columns


def test_momentum_12_1m():
    """Test 12-1 month momentum computation."""
    n_days = 300
    base_price = 100
    prices = [base_price]
    for _ in range(1, n_days):
        prices.append(prices[-1] * (1 + np.random.normal(0.001, 0.01)))

    df = pd.DataFrame(
        {
            "symbol": ["TEST"] * n_days,
            "date": pd.date_range("2020-01-01", periods=n_days, freq="B"),
            "close": prices,
        }
    )

    features = MomentumFeatures()
    result = features.compute(df)

    # mom_12_1m should exist
    assert "mom_12_1m" in result.columns
