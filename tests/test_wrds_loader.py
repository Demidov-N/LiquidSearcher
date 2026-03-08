"""Test unified WRDS data loader."""

import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.data.wrds_loader import WRDSDataLoader


def test_loader_initialization():
    """Test loader initialization with mock mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = WRDSDataLoader(
            wrds_username="mock",
            wrds_password="mock",
            cache_dir=Path(tmpdir),
            mock_mode=True,
        )
        assert loader._mock_mode is True
        assert loader.cache is not None


def test_load_prices_mock():
    """Test loading mock price data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = WRDSDataLoader(
            wrds_username="mock",
            wrds_password="mock",
            cache_dir=Path(tmpdir),
            mock_mode=True,
        )

        df = loader.load_prices(
            symbols=["AAPL", "MSFT"],
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 1, 10),
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "symbol" in df.columns
        assert "date" in df.columns
        assert "close" in df.columns or "prc" in df.columns


def test_load_fundamentals_mock():
    """Test loading mock fundamental data with GICS."""
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = WRDSDataLoader(
            wrds_username="mock",
            wrds_password="mock",
            cache_dir=Path(tmpdir),
            mock_mode=True,
        )

        df = loader.load_fundamentals(
            symbols=["AAPL", "MSFT"],
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "gsector" in df.columns
        assert "ggroup" in df.columns
        assert "symbol" in df.columns


def test_load_merged_mock():
    """Test loading merged data with automatic CCM linking."""
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = WRDSDataLoader(
            wrds_username="mock",
            wrds_password="mock",
            cache_dir=Path(tmpdir),
            mock_mode=True,
        )

        df = loader.load_merged(
            symbols=["AAPL", "MSFT"],
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        # Should have both price and fundamental columns
        assert "close" in df.columns or "prc" in df.columns
        assert "gsector" in df.columns
        assert "ggroup" in df.columns


def test_caching_works():
    """Test that data is cached and retrievable."""
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = WRDSDataLoader(
            wrds_username="mock",
            wrds_password="mock",
            cache_dir=Path(tmpdir),
            mock_mode=True,
        )

        # Load data (should cache)
        df1 = loader.load_prices(
            symbols=["AAPL"],
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 1, 5),
        )

        # Load again (should use cache)
        df2 = loader.load_prices(
            symbols=["AAPL"],
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 1, 5),
        )

        # Data should be identical
        pd.testing.assert_frame_equal(df1, df2)
