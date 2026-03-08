"""One-time feature pre-computation for all stocks.

This script loads raw WRDS data and computes all 32 features
for every stock, saving to parquet files for efficient training.
"""

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.data.cache_manager import CacheManager
from src.data.credentials import validate_and_exit
from src.data.wrds_loader import (
    load_fundamental,
    load_ohlcv,
)
from src.features.engineer import FeatureEngineer


def main():
    parser = argparse.ArgumentParser(description="Pre-compute features for all stocks")
    parser.add_argument(
        "--start-date", type=str, default="2010-01-01", help="Start date for feature computation"
    )
    parser.add_argument(
        "--end-date", type=str, default="2024-12-31", help="End date for feature computation"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/features",
        help="Directory to save parquet feature files",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=None,
        help="Specific symbols to process",
    )
    parser.add_argument(
        "--use-mock",
        action="store_true",
        help="Use mock data instead of WRDS (for testing only)",
    )

    args = parser.parse_args()

    # Validate WRDS credentials unless using mock data
    validate_and_exit(use_mock=args.use_mock)

    # Initialize components
    cache_manager = CacheManager(cache_dir=args.output_dir)
    engineer = FeatureEngineer(cache_manager=cache_manager)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse dates
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    # Get symbols to process
    if args.symbols is None:
        print("Error: No symbols provided. Use --symbols to specify stocks.")
        return

    symbols = args.symbols
    print(f"Processing {len(symbols)} stocks...")

    # Process each stock
    for symbol in tqdm(symbols, desc="Computing features"):
        try:
            # Load OHLCV data
            prices = load_ohlcv(
                symbols=[symbol],
                start_date=start_date,
                end_date=end_date,
                use_mock=args.use_mock,
            )

            # Load fundamental data
            fundamentals = load_fundamental(
                symbols=[symbol],
                start_date=start_date,
                end_date=end_date,
                use_mock=args.use_mock,
            )

            # Combine into single dataframe
            if isinstance(prices, pd.DataFrame) and isinstance(fundamentals, pd.DataFrame):
                # Merge fundamentals into prices (forward fill for daily data)
                if "date" in fundamentals.columns:
                    fundamentals = fundamentals.rename(columns={"date": "datadate"})
                combined = prices.merge(fundamentals, on="symbol", how="left")
                features = engineer.compute_features(combined, cache_key=symbol)
            else:
                # If one is None or different format, use just prices
                features = engineer.compute_features(prices, cache_key=symbol)

            # Ensure required columns are present for FeatureDataset
            # Preserve raw OHLCV columns if they were dropped during feature computation
            raw_ohlcv = ["open", "high", "low", "close", "volume", "return"]

            # Check if we have the raw columns from prices dataframe
            if isinstance(prices, pd.DataFrame):
                for col in raw_ohlcv:
                    if col not in features.columns and col in prices.columns:
                        # Merge raw OHLCV columns back into features
                        if "date" in prices.columns:
                            # Merge on date
                            features = features.merge(
                                prices[["date", "symbol", col]], on=["date", "symbol"], how="left"
                            )

            # Add return column if not present (compute from close)
            if "return" not in features.columns and "close" in features.columns:
                features["return"] = features["close"].pct_change()

            # Ensure date column is present for proper indexing
            if "date" not in features.columns:
                # Try to infer from index or datadate
                if "datadate" in features.columns:
                    features["date"] = features["datadate"]
                elif hasattr(features.index, "name") and features.index.name == "date":
                    features["date"] = features.index
                elif hasattr(features.index, "name") and features.index.name is not None:
                    features["date"] = features.index

            # Check for missing columns and warn
            missing = [col for col in raw_ohlcv if col not in features.columns]
            if missing:
                print(f"Warning: {symbol} missing temporal columns: {missing}")

            output_file = output_dir / f"{symbol}_features.parquet"
            features.to_parquet(output_file)

        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            continue

    print(f"\nDone! Features saved to {output_dir}")
    print(f"Total stocks: {len(list(output_dir.glob('*_features.parquet')))}")


if __name__ == "__main__":
    main()
