"""Test G6 sector categorical features."""

import numpy as np
import pandas as pd
import pytest

from src.features.g6_sector import G6SectorFeatures


def test_g6_initialization():
    """Test G6 feature group initialization."""
    g6 = G6SectorFeatures()
    assert g6.name == "G6_sector"
    assert "gics_sector" in g6.get_feature_names()
    assert "gics_industry_group" in g6.get_feature_names()
    assert "gics_sector_str" in g6.get_feature_names()
    assert "gics_industry_group_str" in g6.get_feature_names()


def test_sector_encoding():
    """Test sector and industry encoding produces integers."""
    df = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "GOOGL", "JPM", "BAC"],
            "date": pd.to_datetime(["2020-01-01"] * 5),
            "gics_sector_str": [
                "Technology",
                "Technology",
                "Technology",
                "Financials",
                "Financials",
            ],
            "gics_industry_group_str": ["Software", "Software", "Internet", "Banks", "Banks"],
        }
    )

    g6 = G6SectorFeatures()
    result = g6.compute(df)

    # Check integer columns exist
    assert "gics_sector" in result.columns
    assert "gics_industry_group" in result.columns

    # Check original string columns preserved
    assert "gics_sector_str" in result.columns
    assert "gics_industry_group_str" in result.columns

    # Verify encoding produces integers
    assert result["gics_sector"].dtype in [np.int64, np.int32, int]
    assert result["gics_industry_group"].dtype in [np.int64, np.int32, int]

    # Technology should have same code (0)
    assert result["gics_sector"].iloc[0] == result["gics_sector"].iloc[1]
    assert result["gics_sector"].iloc[0] == result["gics_sector"].iloc[2]

    # Financials should have same code (different from Technology)
    assert result["gics_sector"].iloc[3] == result["gics_sector"].iloc[4]
    assert result["gics_sector"].iloc[0] != result["gics_sector"].iloc[3]


def test_consistency_across_calls():
    """Test that same sectors get same codes across multiple calls."""
    df1 = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "date": pd.to_datetime(["2020-01-01"]),
            "gics_sector_str": ["Technology"],
            "gics_industry_group_str": ["Software"],
        }
    )

    df2 = pd.DataFrame(
        {
            "symbol": ["MSFT"],
            "date": pd.to_datetime(["2020-01-01"]),
            "gics_sector_str": ["Technology"],
            "gics_industry_group_str": ["Software"],
        }
    )

    g6 = G6SectorFeatures()

    result1 = g6.compute(df1)
    result2 = g6.compute(df2)

    # Same sector should get same code
    assert result1["gics_sector"].iloc[0] == result2["gics_sector"].iloc[0]
    assert result1["gics_industry_group"].iloc[0] == result2["gics_industry_group"].iloc[0]
