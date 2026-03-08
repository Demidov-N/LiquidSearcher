"""Pytest configuration and fixtures."""

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    dates = pl.date_range(
        start="2020-01-01",
        end="2020-12-31",
        interval="1d",
        eager=True
    )
    n_days = len(dates)
    
    return pl.DataFrame({
        "date": dates,
        "permno": [1] * n_days,
        "prc": np.random.randn(n_days).cumsum() + 100,
        "vol": np.random.randint(1000, 100000, n_days),
        "bidlo": np.random.randn(n_days).cumsum() + 99,
        "askhi": np.random.randn(n_days).cumsum() + 101,
    })


@pytest.fixture
def sample_fundamental_data():
    """Generate sample fundamental data for testing."""
    return pl.DataFrame({
        "permno": [1, 2, 3],
        "year": [2020, 2020, 2020],
        "at": [1000.0, 2000.0, 1500.0],
        "seq": [500.0, 1000.0, 750.0],
        "ni": [50.0, 100.0, 75.0],
        "csho": [100.0, 200.0, 150.0],
        "prcc_f": [10.0, 20.0, 15.0],
    })