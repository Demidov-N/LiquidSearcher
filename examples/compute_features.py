"""Example script demonstrating feature engineering pipeline usage.

This script shows how to:
1. Load mock data using WRDSDataLoader
2. Compute all features using FeatureEngineer
3. Use only selected feature groups
4. Inspect the results

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

    print("\n2. Computing features with ALL groups...")

    # Initialize feature engineer with all groups
    engineer_all = FeatureEngineer()

    # Show available feature groups
    available_groups = engineer_all.list_available_groups()
    print(f"   Available groups: {available_groups}")

    # Show registered feature groups
    groups = engineer_all.get_feature_names()
    print(f"   Enabled groups: {list(groups.keys())}")

    # Compute all features
    result_all = engineer_all.compute_features(df)

    print(f"   Computed {len(engineer_all.get_all_feature_names())} features")

    # Display feature counts by group
    print("\n3. Feature counts by group (all groups):")
    for group_name, features in groups.items():
        print(f"   {group_name}: {len(features)} features")

    print("\n4. Computing features with SELECTED groups only...")

    # Initialize feature engineer with only selected groups
    engineer_selected = FeatureEngineer(enabled_groups=["market_risk", "momentum", "sector"])

    # Show enabled groups
    enabled_groups = engineer_selected.list_enabled_groups()
    print(f"   Selected groups: {enabled_groups}")

    # Compute only selected features
    result_selected = engineer_selected.compute_features(df)

    selected_features = engineer_selected.get_all_feature_names()
    print(f"   Computed {len(selected_features)} features (from selected groups only)")

    # Show sample of features
    print("\n5. Sample features (last 5 rows, selected groups):")
    sample_cols = selected_features[:10] if len(selected_features) > 10 else selected_features
    print(result_selected[["symbol", "date"] + sample_cols].tail())

    # Show statistics
    print("\n6. Feature statistics:")
    print("   Total rows:", len(result_selected))
    print("   Total columns:", len(result_selected.columns))
    print("   Feature columns:", len(selected_features))

    # Check for missing values
    missing_counts = result_selected[selected_features].isna().sum()
    features_with_missing = missing_counts[missing_counts > 0]
    if len(features_with_missing) > 0:
        print(f"\n   Features with missing values: {len(features_with_missing)}")
        print("   (Expected for features requiring rolling windows at start of series)")
    else:
        print("\n   No missing values in features")

    # Show specific feature examples
    print("\n7. Example feature values for AAPL (latest date, selected groups):")
    aapl_data = result_selected[result_selected["symbol"] == "AAPL"].tail(1)
    if not aapl_data.empty:
        for group_name, features in engineer_selected.get_feature_names().items():
            print(f"\n   {group_name}:")
            for feat in features[:3]:  # Show first 3 from each group
                if feat in aapl_data.columns:
                    value = aapl_data[feat].values[0]
                    print(f"     {feat}: {value:.4f}")

    print("\n" + "=" * 60)
    print("Feature computation complete!")
    print("=" * 60)

    return result_selected


if __name__ == "__main__":
    result = main()
