"""Test script for notebook compatibility verification."""

from datetime import datetime
from src.data import WRDSDataLoader

# Load mock data
loader = WRDSDataLoader(mock_mode=True)
df = loader.load_merged(["AAPL", "MSFT"], datetime(2020, 1, 1), datetime(2020, 1, 10))

# Show first few rows
print(df.head())
print(f"\nColumns: {list(df.columns)}")
print(f"\nCache file location: data/cache/")

# Verify you can read cache in notebook
import pandas as pd
# In your notebook, you can do:
# df = pd.read_parquet("data/cache/merged_AAPL_MSFT_20200101_20200110.parquet")
