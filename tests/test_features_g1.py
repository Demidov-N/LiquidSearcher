"""Test G1 systematic risk features."""

import numpy as np
import pandas as pd
import pytest

from src.features.g1_risk import G1RiskFeatures


def test_g1_initialization():
    """Test G1 feature group initialization."""
    g1 = G1RiskFeatures()
    assert g1.name == "G1_risk"
    assert "market_beta_60d" in g1.get_feature_names()


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

    g1 = G1RiskFeatures()
    result = g1.compute(df)

    assert "market_beta_60d" in result.columns
    # Beta should be approximately 1.5 for the last rows (after 60-day window)
    last_beta = result["market_beta_60d"].iloc[-1]
    assert not pd.isna(last_beta)
    assert 1.0 < last_beta < 2.0  # Should be close to 1.5


def test_factor_loadings():
    """Test factor loading computations."""
    np.random.seed(42)
    n_days = 100

    df = pd.DataFrame(
        {
            "symbol": ["TEST"] * n_days,
            "date": pd.date_range("2020-01-01", periods=n_days, freq="B"),
            "return": np.random.normal(0.001, 0.02, n_days),
            "smb_factor": np.random.normal(0, 0.01, n_days),
            "hml_factor": np.random.normal(0, 0.01, n_days),
            "mom_factor": np.random.normal(0, 0.01, n_days),
            "rmw_factor": np.random.normal(0, 0.01, n_days),
            "cma_factor": np.random.normal(0, 0.01, n_days),
        }
    )

    g1 = G1RiskFeatures()
    result = g1.compute(df)

    # Check all factor loading columns exist
    assert "smb_loading" in result.columns
    assert "hml_loading" in result.columns
    assert "mom_loading" in result.columns
    assert "rmw_loading" in result.columns
    assert "cma_loading" in result.columns
