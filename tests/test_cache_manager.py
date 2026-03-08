"""Test cache manager with parquet format."""

import tempfile
from pathlib import Path

import pandas as pd

from src.data.cache_manager import CacheManager


def test_cache_save_and_load_parquet():
    """Test saving and loading DataFrame to cache as parquet."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CacheManager(cache_dir=Path(tmpdir))

        # Create test DataFrame with various types
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=10),
                "value": range(10),
                "float_val": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5],
                "string_val": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
            }
        )

        # Save to cache
        cache_key = "test_data_2020"
        cache.set(cache_key, df)

        # Load from cache
        loaded_df = cache.get(cache_key)

        assert loaded_df is not None
        assert len(loaded_df) == 10
        assert list(loaded_df.columns) == ["date", "value", "float_val", "string_val"]
        # Verify parquet preserved types (datetime may be microsecond precision)
        assert str(loaded_df["date"].dtype).startswith("datetime64")
        assert loaded_df["value"].dtype == "int64"


def test_cache_miss_returns_none():
    """Test that missing cache key returns None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CacheManager(cache_dir=Path(tmpdir))

        result = cache.get("nonexistent_key")
        assert result is None


def test_cache_exists_check():
    """Test cache exists method."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CacheManager(cache_dir=Path(tmpdir))

        df = pd.DataFrame({"a": [1, 2, 3]})
        cache.set("my_key", df)

        assert cache.exists("my_key") is True
        assert cache.exists("other_key") is False


def test_cache_notebook_compatibility():
    """Test that cache files can be read directly in notebooks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CacheManager(cache_dir=Path(tmpdir))

        df = pd.DataFrame(
            {
                "symbol": ["AAPL", "MSFT"],
                "price": [150.0, 250.0],
                "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            }
        )

        cache.set("notebook_test", df)

        # Simulate notebook reading the file directly
        cache_file = Path(tmpdir) / "notebook_test.parquet"
        notebook_df = pd.read_parquet(cache_file)

        assert len(notebook_df) == 2
        assert list(notebook_df.columns) == ["symbol", "price", "date"]


def test_cache_polars_to_pandas():
    """Test that polars DataFrames are converted to pandas on save."""
    import polars as pl

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = CacheManager(cache_dir=Path(tmpdir))

        # Create polars DataFrame
        df = pl.DataFrame(
            {
                "a": [1, 2, 3],
                "b": ["x", "y", "z"],
            }
        )

        cache.set("polars_test", df)

        # Load back - should be pandas
        loaded = cache.get("polars_test")
        assert isinstance(loaded, pd.DataFrame)
        assert list(loaded.columns) == ["a", "b"]
