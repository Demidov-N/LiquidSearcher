"""Test sector features."""

import pandas as pd
import pytest

from src.features.sector import SectorFeatures


def test_sector_initialization():
    """Test sector feature group initialization."""
    features = SectorFeatures()
    assert features.name == "sector"
    assert "gics_sector" in features.get_feature_names()


def test_sector_encoding():
    """Test sector integer encoding."""
    df = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "GOOGL"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-01"]),
            "gics_sector_str": ["Technology", "Technology", "Technology"],
            "gics_industry_group_str": ["Software", "Software", "Internet"],
        }
    )

    features = SectorFeatures()
    result = features.compute(df)

    # Check sector columns exist
    assert "gics_sector" in result.columns
    assert "gics_industry_group" in result.columns

    # Same sector should have same code
    assert result["gics_sector"].iloc[0] == result["gics_sector"].iloc[1]


def test_different_sectors():
    """Test encoding of different sectors."""
    df = pd.DataFrame(
        {
            "symbol": ["AAPL", "JPM", "XOM"],
            "date": pd.to_datetime(["2024-01-01"] * 3),
            "gics_sector_str": ["Technology", "Finance", "Energy"],
            "gics_industry_group_str": ["Software", "Banks", "Oil & Gas"],
        }
    )

    features = SectorFeatures()
    result = features.compute(df)

    # Different sectors should have different codes
    codes = result["gics_sector"].tolist()
    assert len(set(codes)) == 3  # All 3 sectors are different


def test_missing_sector_data():
    """Test handling of missing sector data."""
    df = pd.DataFrame(
        {
            "symbol": ["TEST"],
            "date": pd.to_datetime(["2024-01-01"]),
        }
    )

    features = SectorFeatures()
    result = features.compute(df)

    # Should handle missing columns gracefully
    assert "gics_sector" in result.columns
    assert result["gics_sector"].iloc[0] == -1
