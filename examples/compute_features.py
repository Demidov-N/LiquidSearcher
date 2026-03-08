"""Example script demonstrating feature engineering pipeline usage.

This script shows how to:
1. Load mock data using WRDSDataLoader
2. Compute all G1-G6 features using FeatureEngineer
3. Inspect the results

Usage:
    python examples/compute_features.py
"""

from datetime import datetime

import numpy as np
import pandas as pd

from src.data.wrds_loader import WRDSDataLoader
from src.features.engineer import FeatureEngineer


def main():
    """Run the feature computation example."""
    print("=" * 60)
    print("Feature Engineering Pipeline Example")
    print("=" * 60)

    # Configuration
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)

    print(f"\n1. Loading data for {len(symbols)} symbols...")
    print(f"   Date range: {start_date.date()} to {end_date.date()}")

    # Initialize data loader in mock mode (no WRDS credentials needed)
    data_loader = WRDSDataLoader(mock_mode=True)

    # Load merged price and fundamental data
    df = data_loader.load_merged(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        use_cache=True,
    )

    print(f"   Loaded {len(df):,} rows of data")
    print(f"   Columns: {list(df.columns)}")

    # Add required columns for feature computation if missing
    if "return" not in df.columns and "close" in df.columns:
        df["return"] = df.groupby("symbol")["close"].pct_change()

    # Add factor columns (in real scenario, these come from WRDS)
    factor_cols = {
        "market_return": 0.0,
        "smb_factor": 0.0,
        "hml_factor": 0.0,
        "mom_factor": 0.0,
        "rmw_factor": 0.0,
        "cma_factor": 0.0,
    }

    for col, default in factor_cols.items():
        if col not in df.columns:
            # Generate random factor returns
            df[col] = np.random.randn(len(df)) * 0.01 + default

    print("\n2. Computing features...")

    # Initialize feature engineer
    engineer = FeatureEngineer()

    # Show registered feature groups
    groups = engineer.get_feature_names()
    print(f"   Registered groups: {list(groups.keys())}")

    # Compute all features
    result = engineer.compute_features(df)

    print(f"   Computed {len(engineer.get_all_feature_names())} features")

    # Display feature counts by group
    print("\n3. Feature counts by group:")
    for group_name, features in groups.items():
        print(f"   {group_name}: {len(features)} features")

    # Show sample of features
    print("\n4. Sample features (last 5 rows):")
    feature_cols = engineer.get_all_feature_names()
    sample_cols = feature_cols[:10] if len(feature_cols) > 10 else feature_cols
    print(result[["symbol", "date"] + sample_cols].tail())

    # Show statistics
    print("\n5. Feature statistics:")
    print("   Total rows:", len(result))
    print("   Total columns:", len(result.columns))
    print("   Feature columns:", len(feature_cols))

    # Check for missing values
    missing_counts = result[feature_cols].isna().sum()
    features_with_missing = missing_counts[missing_counts > 0]
    if len(features_with_missing) > 0:
        print(f"\n   Features with missing values: {len(features_with_missing)}")
        print("   (Expected for features requiring rolling windows at start of series)")
    else:
        print("\n   No missing values in features")

    # Show specific feature examples
    print("\n6. Example feature values for AAPL (latest date):")
    aapl_data = result[result["symbol"] == "AAPL"].tail(1)
    if not aapl_data.empty:
        for group_name, features in groups.items():
            print(f"\n   {group_name}:")
            for feat in features[:3]:  # Show first 3 from each group
                if feat in aapl_data.columns:
                    value = aapl_data[feat].values[0]
                    print(f"     {feat}: {value:.4f}")

    print("\n" + "=" * 60)
    print("Feature computation complete!")
    print("=" * 60)

    return result


if __name__ == "__main__":
    result = main()
