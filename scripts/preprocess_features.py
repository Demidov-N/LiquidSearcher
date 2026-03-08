"""Preprocessing script for data collection and feature computation.

This script:
1. Loads symbols in batches (500-1000 per batch)
2. Fetches data from WRDS with progress tracking (tqdm)
3. Uses pre-computed betas from WRDS Beta Suite
4. Computes features locally with Polars
5. Writes incrementally to single parquet file
6. Applies two-pass normalization

Usage:
    python -m scripts.preprocess_features --start-date 2010-01-01 --end-date 2024-12-31
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

from src.config.settings import get_settings
from src.data.credentials import validate_and_exit
from src.data.universe import SymbolUniverse
from src.data.wrds_loader import WRDSDataLoader
from src.features.processor import FeatureProcessor
from src.utils.memory import get_available_memory_mb, get_recommended_batch_size, print_memory_status

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_symbol_batch(
    symbols: List[str],
    start_date: str,
    end_date: str,
    loader: WRDSDataLoader,
    processor: FeatureProcessor,
    skip_betas: bool = False,
) -> pd.DataFrame:
    """Process a single batch of symbols.
    
    Args:
        symbols: List of ticker symbols to process
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        loader: WRDS data loader instance
        processor: Feature processor instance
        
    Returns:
        DataFrame with computed features for all symbols
    """
    logger.info(f"Processing batch of {len(symbols)} symbols")
    
    # Fetch data from WRDS
    logger.info("  Fetching prices...")
    prices_df = loader.fetch_prices_batch(symbols, start_date, end_date)
    
    if prices_df.empty:
        logger.warning(f"  No price data for symbols: {symbols}")
        return pd.DataFrame()
    
    logger.info(f"  Got {len(prices_df)} price rows")
    
    # Fetch pre-computed betas (if enabled and not skipped)
    betas_df = None
    if not skip_betas:
        logger.info("  Fetching pre-computed betas...")
        try:
            betas_df = loader.fetch_precomputed_betas_batch(
                symbols, start_date, end_date, window=60
            )
            logger.info(f"  Got {len(betas_df)} beta rows" if not betas_df.empty else "  No beta data")
        except Exception as e:
            logger.warning(f"  Could not fetch betas: {e}")
            logger.warning("  Continuing without pre-computed betas...")
            betas_df = None
    
    # Fetch fundamentals
    logger.info("  Fetching fundamentals...")
    fundamentals_df = loader.fetch_fundamentals_batch(symbols, start_date, end_date)
    logger.info(f"  Got {len(fundamentals_df)} fundamental rows" if not fundamentals_df.empty else "  No fundamental data")
    
    # Fetch GICS codes
    logger.info("  Fetching GICS codes...")
    gics_df = loader.fetch_gics_codes(symbols)
    logger.info(f"  Got {len(gics_df)} GICS rows" if not gics_df.empty else "  No GICS data")
    
    # Process features
    logger.info("  Computing features...")
    features_df = processor.process_batch(
        prices_df=prices_df,
        betas_df=betas_df,
        fundamentals_df=fundamentals_df,
        gics_df=gics_df,
    )
    
    logger.info(f"  Computed {len(features_df.columns)} features for {len(features_df)} rows")
    
    return features_df


def write_batch_to_parquet(
    df: pd.DataFrame,
    output_path: Path,
    is_first_batch: bool = False
) -> None:
    """Write batch to parquet file incrementally.
    
    Args:
        df: DataFrame to write
        output_path: Path to output parquet file
        is_first_batch: Whether this is the first batch (create vs append)
    """
    if df.empty:
        return
    
    if is_first_batch:
        # First batch: create new file
        df.to_parquet(output_path, index=False, compression='snappy')
        logger.info(f"Created parquet file: {output_path}")
    else:
        # Subsequent batches: read, concat, write
        existing_df = pd.read_parquet(output_path)
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        combined_df.to_parquet(output_path, index=False, compression='snappy')
        logger.info(f"Appended to parquet file: {len(df)} rows (total: {len(combined_df)})")


def get_universe_symbols(settings) -> List[str]:
    """Get list of symbols in universe.
    
    For now, returns a hardcoded list. In production, this would:
    - Load from Russell 2000 + S&P 400 constituents
    - Filter by date range
    - Apply liquidity screens
    """
    # Placeholder: return a sample of large-cap stocks
    # In production, load from: crsp.dsenames filtered by exchange and dates
    symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM',
        'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'BAC', 'ABBV', 'PFE',
        'KO', 'AVGO', 'PEP', 'TMO', 'COST', 'DIS', 'ABT', 'ADBE',
        'WMT', 'MRK', 'CSCO', 'ACN', 'VZ', 'NKE', 'TXN', 'CMCSA',
    ]
    
    logger.info(f"Using {len(symbols)} symbols in universe")
    return symbols


def main():
    """Main entry point for preprocessing script."""
    parser = argparse.ArgumentParser(
        description="Preprocess features for all stocks in batches"
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default='2010-01-01',
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default='2024-12-31',
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Number of symbols per batch (auto-detected based on available RAM if not specified)'
    )
    parser.add_argument(
        '--auto-memory',
        action='store_true',
        default=True,
        help='Auto-detect available memory and adjust batch size (default: True)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed/all_features.parquet',
        help='Output parquet file path'
    )
    parser.add_argument(
        '--use-mock',
        action='store_true',
        help='Use mock data instead of WRDS (for testing)'
    )
    parser.add_argument(
        '--skip-betas',
        action='store_true',
        help='Skip fetching pre-computed betas'
    )
    
    args = parser.parse_args()
    
    # Print memory status and auto-detect batch size
    print("\n" + "="*60)
    print("DATA PREPROCESSING PIPELINE")
    print("="*60)
    print_memory_status()
    
    # Determine batch size
    if args.batch_size is not None:
        # User explicitly specified batch size
        batch_size = args.batch_size
        print(f"Using user-specified batch size: {batch_size}")
    elif args.auto_memory:
        # Auto-detect based on memory
        batch_size = get_recommended_batch_size(safety_factor=0.5)  # 50% for extra safety
        print(f"Auto-detected batch size: {batch_size} (based on available memory)")
    else:
        # Default conservative value
        batch_size = 200
        print(f"Using default batch size: {batch_size}")
    
    print("="*60 + "\n")
    
    # Validate WRDS credentials unless using mock data
    if not args.use_mock:
        validate_and_exit()
    
    # Setup
    settings = get_settings()
    settings.batch_size = batch_size
    if args.skip_betas:
        settings.use_precomputed_betas = False
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get universe symbols
    symbols = get_universe_symbols(settings)
    universe = SymbolUniverse(symbols, batch_size=batch_size)
    
    logger.info(f"Processing {len(universe)} symbols in batches of {batch_size}")
    logger.info(f"Date range: {args.start_date} to {args.end_date}")
    logger.info(f"Output: {output_path}")
    
    # Initialize processor
    processor = FeatureProcessor()
    
    # Process batches with progress tracking
    is_first_batch = True
    total_batches = (len(universe) + batch_size - 1) // batch_size
    
    with WRDSDataLoader() as loader:
        for batch_num, symbol_batch in enumerate(universe.batches(desc="Processing batches"), 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Batch {batch_num}/{total_batches}: {len(symbol_batch)} symbols")
            logger.info(f"{'='*60}")
            
            try:
                # Process this batch
                features_df = process_symbol_batch(
                    symbols=symbol_batch,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    loader=loader,
                    processor=processor,
                    skip_betas=args.skip_betas,
                )
                
                # Write incrementally
                if not features_df.empty:
                    write_batch_to_parquet(
                        features_df,
                        output_path,
                        is_first_batch=is_first_batch
                    )
                    is_first_batch = False
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {e}")
                logger.error(f"Symbols in failed batch: {symbol_batch}")
                # Continue with next batch (don't stop entire pipeline)
                continue
    
    logger.info(f"\n{'='*60}")
    logger.info("Processing complete!")
    logger.info(f"Output saved to: {output_path}")
    
    # Log file stats
    if output_path.exists():
        final_df = pd.read_parquet(output_path)
        logger.info(f"Total rows: {len(final_df)}")
        logger.info(f"Total columns: {len(final_df.columns)}")
        logger.info(f"Symbols: {final_df['symbol'].nunique()}")
        logger.info(f"Date range: {final_df['date'].min()} to {final_df['date'].max()}")
        logger.info(f"File size: {output_path.stat().st_size / (1024*1024):.1f} MB")


if __name__ == '__main__':
    main()
