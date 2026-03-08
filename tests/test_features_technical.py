"""Test technical/price pattern features."""

import numpy as np
import pandas as pd
import pytest

from src.features.technical import TechnicalFeatures


def test_technical_initialization():
    """Test technical feature group initialization."""
    features = TechnicalFeatures()
    assert features.name == "technical"
    assert "z_close_5d" in features.get_feature_names()
    assert "ma_ratio_5" in features.get_feature_names()


def test_ma_ratios():
    """Test moving average ratio computation."""
    n_days = 50
    base_price = 100
    prices = [base_price]
    for _ in range(1, n_days):
        prices.append(prices[-1] * (1 + np.random.normal(0, 0.01)))

    df = pd.DataFrame(
        {
            "symbol": ["TEST"] * n_days,
            "date": pd.date_range("2020-01-01", periods=n_days, freq="B"),
            "close": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "volume": [1000000] * n_days,
        }
    )

    features = TechnicalFeatures()
    result = features.compute(df)

    # Check MA ratio columns exist
    assert "ma_ratio_5" in result.columns
    assert "ma_ratio_10" in result.columns
    assert "ma_ratio_20" in result.columns


def test_price_zscores():
    """Test price z-score computation."""
    n_days = 50
    prices = [100 + i for i in range(n_days)]

    df = pd.DataFrame(
        {
            "symbol": ["TEST"] * n_days,
            "date": pd.date_range("2020-01-01", periods=n_days, freq="B"),
            "close": prices,
            "high": [p + 1 for p in prices],
            "low": [p - 1 for p in prices],
            "volume": [1000000] * n_days,
        }
    )

    features = TechnicalFeatures()
    result = features.compute(df)

    # Check z-score columns exist
    assert "z_close_5d" in result.columns
    assert "z_close_10d" in result.columns
    assert "z_close_20d" in result.columns


def test_volume_zscores():
    """Test volume z-score computation."""
    n_days = 50
    volumes = [1000000 + i * 1000 for i in range(n_days)]

    df = pd.DataFrame(
        {
            "symbol": ["TEST"] * n_days,
            "date": pd.date_range("2020-01-01", periods=n_days, freq="B"),
            "close": [100.0] * n_days,
            "high": [101.0] * n_days,
            "low": [99.0] * n_days,
            "volume": volumes,
        }
    )

    features = TechnicalFeatures()
    result = features.compute(df)

    # Check volume z-score columns exist
    assert "z_volume_5d" in result.columns
    assert "z_volume_10d" in result.columns
    assert "z_volume_20d" in result.columns
