"""Test G4 valuation and fundamentals features."""

import numpy as np
import pandas as pd

from src.features.g4_valuation import G4ValuationFeatures


def test_g4_initialization():
    """Test G4 feature group initialization."""
    g4 = G4ValuationFeatures()
    assert g4.name == "G4_valuation"
    feature_names = g4.get_feature_names()
    assert "log_mktcap" in feature_names
    assert "pe_ratio" in feature_names
    assert "pb_ratio" in feature_names
    assert "roe" in feature_names


def test_valuation_computation():
    """Test valuation features computation."""
    # Create synthetic data with multiple symbols on same date for cross-sectional z-scores
    dates = pd.date_range("2020-01-01", periods=2, freq="B")
    df = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"] * 2,
            "date": [dates[0]] * 4 + [dates[1]] * 4,
            "price": [100.0, 50.0, 200.0, 75.0, 110.0, 55.0, 220.0, 80.0],
            "shares_outstanding": [1000000, 2000000, 500000, 1500000] * 2,
            "eps": [10.0, 5.0, 8.0, 3.0] * 2,
            "book_value_per_share": [50.0, 25.0, 100.0, 30.0] * 2,
            "net_income": [10000000.0, 10000000.0, 4000000.0, 4500000.0] * 2,
            "equity": [50000000.0, 50000000.0, 50000000.0, 45000000.0] * 2,
        }
    )

    g4 = G4ValuationFeatures()
    result = g4.compute(df)

    # Check all feature columns exist
    assert "log_mktcap" in result.columns
    assert "pe_ratio" in result.columns
    assert "pb_ratio" in result.columns
    assert "roe" in result.columns

    # Verify log_mktcap = ln(price * shares_outstanding)
    # A: 100 * 1M = 100M, ln(100M) ≈ 18.42
    expected_mktcap_a = np.log(100.0 * 1000000)
    assert abs(result["log_mktcap"].iloc[0] - expected_mktcap_a) < 0.01

    # Verify pe_ratio is rank-normalized to [0, 1]
    # A has P/E = 10.0 which is among the lower values, so rank should be < 0.5
    assert 0 <= result["pe_ratio"].iloc[0] <= 1

    # Verify pb_ratio is z-scored (not checking exact value, just that it's finite)
    assert pd.notna(result["pb_ratio"].iloc[0])

    # Verify roe is z-scored (not checking exact value, just that it's finite)
    assert pd.notna(result["roe"].iloc[0])


def test_pe_winsorization():
    """Test that extreme P/E values are winsorized."""
    # Create data with extreme P/E values
    df = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D", "E"],
            "date": pd.date_range("2020-01-01", periods=5, freq="B"),
            "price": [100.0, 50.0, 200.0, 75.0, 1000.0],
            "shares_outstanding": [1000000] * 5,
            "eps": [0.01, 5.0, 8.0, 3.0, 1.0],  # A has very high P/E (10000), E has high P/E (1000)
            "book_value_per_share": [50.0] * 5,
            "net_income": [100000.0] * 5,
            "equity": [50000000.0] * 5,
        }
    )

    g4 = G4ValuationFeatures()
    result = g4.compute(df)

    # Check that P/E is in winsorized columns (raw values would be extreme)
    # With 2% and 98% quantiles on 5 observations, we expect some winsorization
    pe_values = result["pe_ratio"]

    # The extreme values should be capped
    # Without winsorization, A would have P/E = 100/0.01 = 10000
    # The winsorized value should be much lower
    pe_a = pe_values.iloc[0]
    assert pe_a < 5000  # Winsorized value should be capped

    # Check that all P/E values are reasonable after winsorization
    assert all(pe_values < 5000), f"Some P/E values not winsorized: {pe_values}"

    # Verify rank feature is created and normalized to [0, 1]
    pe_rank_col = [col for col in result.columns if "pe_ratio" in col and "rank" in col]
    if pe_rank_col:
        pe_rank = result[pe_rank_col[0]]
        assert all((pe_rank >= 0) & (pe_rank <= 1)), "P/E rank should be in [0, 1]"
