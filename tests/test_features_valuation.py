"""Test valuation features."""

import numpy as np
import pandas as pd
import pytest

from src.features.valuation import ValuationFeatures


def test_valuation_initialization():
    """Test valuation feature group initialization."""
    features = ValuationFeatures()
    assert features.name == "valuation"
    assert "pe_ratio" in features.get_feature_names()
    assert "pb_ratio" in features.get_feature_names()


def test_market_cap():
    """Test market cap computation."""
    df = pd.DataFrame(
        {
            "symbol": ["TEST"],
            "date": pd.to_datetime(["2024-01-01"]),
            "price": [100.0],
            "shares_outstanding": [1000000],
        }
    )

    features = ValuationFeatures()
    result = features.compute(df)

    assert "log_mktcap" in result.columns


def test_pe_ratio():
    """Test P/E ratio computation."""
    df = pd.DataFrame(
        {
            "symbol": ["TEST"],
            "date": pd.to_datetime(["2024-01-01"]),
            "price": [100.0],
            "eps": [5.0],
        }
    )

    features = ValuationFeatures()
    result = features.compute(df)

    assert "pe_ratio" in result.columns
    # P/E = 100 / 5 = 20
    assert result["pe_ratio_raw"].iloc[0] == 20.0


def test_pb_ratio():
    """Test P/B ratio computation."""
    df = pd.DataFrame(
        {
            "symbol": ["TEST"],
            "date": pd.to_datetime(["2024-01-01"]),
            "price": [100.0],
            "book_value_per_share": [50.0],
        }
    )

    features = ValuationFeatures()
    result = features.compute(df)

    assert "pb_ratio" in result.columns
    # P/B = 100 / 50 = 2.0
    assert result["pb_ratio_raw"].iloc[0] == np.log(2.0)


def test_roe():
    """Test ROE computation."""
    df = pd.DataFrame(
        {
            "symbol": ["TEST"],
            "date": pd.to_datetime(["2024-01-01"]),
            "net_income": [1000000.0],
            "equity": [10000000.0],
        }
    )

    features = ValuationFeatures()
    result = features.compute(df)

    assert "roe" in result.columns
    # ROE = 1M / 10M = 0.10 (10%)
    assert result["roe_raw"].iloc[0] == 0.10
