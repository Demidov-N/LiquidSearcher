"""One-time feature pre-computation for all stocks.

This script loads raw WRDS data and computes all 32 features
for every stock, saving to parquet files for efficient training.
"""

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.data.cache_manager import CacheManager
from src.data.wrds_loader import WRDSLoader
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
        help="Specific symbols to process (default: all Russell 2000 + S&P 400)",
    )

    args = parser.parse_args()

    # Initialize components
    loader = WRDSLoader()
    engineer = FeatureEngineer(cache_manager=CacheManager(cache_dir=args.output_dir))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get universe of stocks
    if args.symbols is None:
        print("Loading Russell 2000 + S&P 400 universe...")
        symbols = loader.get_universe_symbols()
    else:
        symbols = args.symbols

    print(f"Processing {len(symbols)} stocks...")

    # Process each stock
    for symbol in tqdm(symbols, desc="Computing features"):
        try:
            prices = loader.load_prices(
                symbols=[symbol],
                start_date=args.start_date,
                end_date=args.end_date,
            )

            fundamentals = loader.load_fundamentals(
                symbols=[symbol],
                start_date=args.start_date,
                end_date=args.end_date,
            )

            features = engineer.compute_all_features(prices, fundamentals)

            output_file = output_dir / f"{symbol}_features.parquet"
            features.to_parquet(output_file)

        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            continue

    print(f"\nDone! Features saved to {output_dir}")
    print(f"Total stocks: {len(list(output_dir.glob('*_features.parquet')))}")


if __name__ == "__main__":
    main()
