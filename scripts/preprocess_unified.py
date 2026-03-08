"""Unified feature pre-computation for all stocks into single parquet file.

This script loads raw WRDS data and computes all 32 features
for every stock, saving to a single unified parquet file for efficient training.

Usage:
    # With real WRDS data (requires credentials)
    python -m scripts.preprocess_unified \
        --start-date 2023-01-01 \
        --end-date 2023-12-31 \
        --symbols AAPL MSFT GOOGL

    # With mock data (for testing)
    python -m scripts.preprocess_unified \
        --start-date 2023-01-01 \
        --end-date 2023-12-31 \
        --symbols AAPL MSFT GOOGL \
        --use-mock
"""

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.data.credentials import validate_and_exit
from src.data.wrds_loader import load_fundamental, load_ohlcv
from src.features.engineer import FeatureEngineer


def process_symbol_batch(
    symbols: list[str],
    start_date: datetime,
    end_date: datetime,
    output_dir: Path,
    use_mock: bool = False,
) -> pd.DataFrame | None:
    """Process a batch of symbols and return combined dataframe.

    Args:
        symbols: List of ticker symbols to process
        start_date: Start date for data retrieval
        end_date: End date for data retrieval
        output_dir: Directory for intermediate outputs
        use_mock: Whether to use mock data instead of WRDS

    Returns:
        DataFrame with features for all symbols in batch, or None if all fail
    """
    batch_results = []

    for symbol in symbols:
        try:
            # Load OHLCV data
            prices = load_ohlcv(
                symbols=[symbol],
                start_date=start_date,
                end_date=end_date,
                use_mock=use_mock,
            )

            # Load fundamental data
            fundamentals = load_fundamental(
                symbols=[symbol],
                start_date=start_date,
                end_date=end_date,
                use_mock=use_mock,
            )

            # Combine into single dataframe
            if isinstance(prices, pd.DataFrame) and isinstance(fundamentals, pd.DataFrame):
                # Merge fundamentals into prices (forward fill for daily data)
                if "date" in fundamentals.columns:
                    fundamentals = fundamentals.rename(columns={"date": "datadate"})

                # Use merge_asof for efficient forward-fill merging
                prices = prices.sort_values(["symbol", "date"])
                if "datadate" in fundamentals.columns:
                    fundamentals = fundamentals.sort_values(["symbol", "datadate"])
                    fundamentals["date"] = fundamentals["datadate"]

                # Perform merge_asof for each symbol
                merged_dfs = []
                for sym in prices["symbol"].unique():
                    sym_prices = prices[prices["symbol"] == sym]
                    sym_fund = fundamentals[fundamentals["symbol"] == sym]

                    if sym_fund.empty:
                        merged_dfs.append(sym_prices)
                    else:
                        # Use nearest direction to handle sparse fundamental data
                        # (e.g., annual reports available only at year-end)
                        merged = pd.merge_asof(
                            sym_prices,
                            sym_fund,
                            on="date",
                            direction="nearest",
                            suffixes=("", "_fund"),
                        )
                        merged_dfs.append(merged)

                combined = pd.concat(merged_dfs, ignore_index=True) if merged_dfs else prices
            else:
                combined = prices

            # Copy fundamental columns from _fund suffix to regular names
            # This allows feature engineering to use pre-computed values
            fund_col_map = {
                "pe_ratio_fund": "pe_ratio",
                "pb_ratio_fund": "pb_ratio",
                "roe_fund": "roe",
                "market_cap_fund": "market_cap",
                "gsector_fund": "gsector",
                "ggroup_fund": "ggroup",
            }
            for fund_col, regular_col in fund_col_map.items():
                if fund_col in combined.columns:
                    combined[regular_col] = combined[fund_col]

            # Compute features
            engineer = FeatureEngineer()
            features = engineer.compute_features(combined)

            # Ensure required columns are present
            raw_ohlcv = ["open", "high", "low", "close", "volume", "return"]

            # Preserve raw OHLCV columns
            if isinstance(prices, pd.DataFrame):
                for col in raw_ohlcv:
                    if col not in features.columns and col in prices.columns:
                        features = features.merge(
                            prices[["date", "symbol", col]],
                            on=["date", "symbol"],
                            how="left",
                        )

            # Add return column if not present
            if "return" not in features.columns and "close" in features.columns:
                features["return"] = features.groupby("symbol")["close"].pct_change()

            # Ensure date column is present
            if "date" not in features.columns:
                if "datadate" in features.columns:
                    features["date"] = features["datadate"]
                elif hasattr(features.index, "name") and features.index.name == "date":
                    features["date"] = features.index

            batch_results.append(features)

        except Exception as e:
            print(f"  Warning: Error processing {symbol}: {e}")
            continue

    if not batch_results:
        return None

    return pd.concat(batch_results, ignore_index=True)


def process_unified(
    symbols: list[str],
    start_date: datetime,
    end_date: datetime,
    output_dir: Path,
    batch_size: int = 100,
    use_mock: bool = False,
) -> Path:
    """Process all symbols and save to unified parquet file.

    Args:
        symbols: List of ticker symbols to process
        start_date: Start date for data retrieval
        end_date: End date for data retrieval
        output_dir: Directory for output file
        batch_size: Number of stocks to process per batch
        use_mock: Whether to use mock data instead of WRDS

    Returns:
        Path to the unified output file
    """
    output_file = output_dir / "all_features.parquet"

    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(symbols)} stocks to unified parquet...")
    print(f"Output: {output_file}")
    print(f"Batch size: {batch_size}")

    # Process in batches
    all_results = []
    num_batches = (len(symbols) + batch_size - 1) // batch_size

    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]
        batch_num = i // batch_size + 1

        # Use tqdm for first batch to show overall progress
        if i == 0:
            pbar = tqdm(total=len(symbols), desc="Computing features")

        batch_df = process_symbol_batch(batch, start_date, end_date, output_dir, use_mock=use_mock)

        if batch_df is not None:
            all_results.append(batch_df)
            pbar.update(len(batch))

        # For subsequent batches, just print status
        if i > 0:
            print(f"  Batch {batch_num}/{num_batches}: {len(batch)} stocks processed")

    if i == 0:
        pbar.close()

    # Combine all results
    if all_results:
        unified_df = pd.concat(all_results, ignore_index=True)

        # Ensure proper column ordering
        first_cols = ["date", "symbol", "open", "high", "low", "close", "volume", "return"]
        other_cols = [c for c in unified_df.columns if c not in first_cols]
        unified_df = unified_df[first_cols + other_cols]

        # Save to parquet with snappy compression
        unified_df.to_parquet(output_file, compression="snappy")

        # Print summary
        print(f"\n✅ Done! Unified features saved to {output_file}")
        print(f"Total rows: {len(unified_df):,}")
        print(f"Symbols: {unified_df['symbol'].nunique()}")
        print(f"Date range: {unified_df['date'].min()} to {unified_df['date'].max()}")
        print(f"Columns: {len(unified_df.columns)}")

        return output_file
    else:
        raise ValueError("No data was processed successfully")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute unified features for all stocks into single parquet file"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2010-01-01",
        help="Start date for feature computation (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-12-31",
        help="End date for feature computation (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/features",
        help="Directory to save unified parquet file",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        required=True,
        help="Specific symbols to process (space-separated)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of stocks to process per batch (default: 100)",
    )
    parser.add_argument(
        "--use-mock",
        action="store_true",
        help="Use mock data instead of WRDS (for testing only)",
    )

    args = parser.parse_args()

    # Validate WRDS credentials unless using mock data
    validate_and_exit(use_mock=args.use_mock)

    # Parse dates
    try:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    except ValueError as e:
        print("Error: Invalid date format. Use YYYY-MM-DD format.")
        raise SystemExit(1) from e

    # Validate symbols
    if not args.symbols:
        print("Error: No symbols provided. Use --symbols to specify stocks.")
        raise SystemExit(1)

    output_dir = Path(args.output_dir)

    # Process all symbols
    try:
        process_unified(
            symbols=args.symbols,
            start_date=start_date,
            end_date=end_date,
            output_dir=output_dir,
            batch_size=args.batch_size,
            use_mock=args.use_mock,
        )
    except Exception as e:
        print(f"Error during processing: {e}")
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
