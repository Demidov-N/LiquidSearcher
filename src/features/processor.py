"""Feature processor with Polars for efficient computation."""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

from src.features.normalization import two_pass_normalization

logger = logging.getLogger(__name__)


class FeatureProcessor:
    """Process raw data into features using Polars for efficiency.
    
    Features computed:
    - G1: Market risk (use pre-computed betas when available)
    - G2: Volatility (realized vol, idiosyncratic vol)
    - G3: Momentum (1m, 3m, 6m, 12_1m returns)
    - G4: Valuation (P/E, P/B, ROE from fundamentals)
    - G5: OHLCV technicals (z-scores, MA ratios)
    - G6: Sector (GICS codes)
    
    All computations include tqdm progress tracking.
    """
    
    def __init__(self):
        """Initialize feature processor."""
        self.temporal_features: List[str] = []
        self.tabular_features: List[str] = []
    
    def compute_ohlcv_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute OHLCV features from price data.
        
        Args:
            df: DataFrame with columns: symbol, date, prc, vol, ret
            
        Returns:
            DataFrame with OHLCV features added
        """
        pl_df = pl.from_pandas(df)
        result_pl = self._compute_ohlcv_features_polars(pl_df)
        return result_pl.to_pandas()
    
    def compute_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute momentum features from returns.
        
        Args:
            df: DataFrame with columns: symbol, date, ret (returns)
            
        Returns:
            DataFrame with momentum features added
        """
        pl_df = pl.from_pandas(df)
        result_pl = self._compute_momentum_features_polars(pl_df)
        return result_pl.to_pandas()
    
    def process_batch(
        self,
        prices_df: pd.DataFrame,
        betas_df: Optional[pd.DataFrame] = None,
        fundamentals_df: Optional[pd.DataFrame] = None,
        gics_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Process a batch of data into features.
        
        Args:
            prices_df: OHLCV price data
            betas_df: Pre-computed betas (optional)
            fundamentals_df: Fundamental data (optional)
            gics_df: GICS sector codes (optional)
            
        Returns:
            DataFrame with all computed features
        """
        logger.info(f"Processing batch with {len(prices_df)} price rows")
        
        # Convert to Polars for efficient computation
        prices_pl = pl.from_pandas(prices_df)
        
        # G5: Compute OHLCV features
        logger.info("Computing OHLCV features...")
        features_pl = self._compute_ohlcv_features_polars(prices_pl)
        
        # G3: Compute momentum features
        logger.info("Computing momentum features...")
        features_pl = self._compute_momentum_features_polars(features_pl)
        
        # G2: Compute volatility features
        logger.info("Computing volatility features...")
        features_pl = self._compute_volatility_features_polars(features_pl)
        
        # Convert back to pandas for merging
        features_df = features_pl.to_pandas()
        
        # G1: Add pre-computed betas if available
        if betas_df is not None and not betas_df.empty:
            logger.info("Merging pre-computed betas...")
            features_df = self._merge_betas(features_df, betas_df)
        else:
            logger.info("No beta data to merge, skipping...")
        
        # G4: Add fundamentals
        if fundamentals_df is not None and not fundamentals_df.empty:
            logger.info("Merging fundamentals...")
            features_df = self._merge_fundamentals(features_df, fundamentals_df)
        else:
            logger.info("No fundamental data to merge, skipping...")
        
        # G6: Add GICS codes
        if gics_df is not None and not gics_df.empty:
            logger.info("Merging GICS codes...")
            features_df = self._merge_gics(features_df, gics_df)
        else:
            logger.info("No GICS data to merge, skipping...")
        
        # G7: Compute local betas if WRDS Beta Suite not available
        if betas_df is None or betas_df.empty:
            logger.info("Computing betas locally from price data...")
            # Re-convert to Polars for beta computation
            features_pl = pl.from_pandas(features_df)
            features_pl = self._compute_betas_polars(features_pl, window=60)
            features_df = features_pl.to_pandas()
        
        return features_df
    
    def _compute_ohlcv_features_polars(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute OHLCV features using Polars.
        
        Features: z_close, z_volume, MA ratios
        """
        # Sort by symbol and date
        df = df.sort(['symbol', 'date'])
        
        # Compute z-scores for price changes (time-series)
        df = df.with_columns([
            (pl.col('prc') / pl.col('prc').shift(1) - 1).over('symbol').alias('ret_1d'),
        ])
        
        # Compute rolling z-score (252-day window)
        df = df.with_columns([
            pl.col('ret_1d')
            .rolling_mean(window_size=252, min_samples=60)
            .over('symbol')
            .alias('ret_mean_252d'),
            
            pl.col('ret_1d')
            .rolling_std(window_size=252, min_samples=60)
            .over('symbol')
            .alias('ret_std_252d'),
        ])
        
        # z_close: time-series z-score of returns
        df = df.with_columns([
            ((pl.col('ret_1d') - pl.col('ret_mean_252d')) / pl.col('ret_std_252d'))
            .alias('z_close')
        ])
        
        # z_volume: time-series z-score of volume changes
        df = df.with_columns([
            (pl.col('vol') / pl.col('vol').shift(1) - 1).over('symbol').alias('vol_change'),
        ])
        
        df = df.with_columns([
            pl.col('vol_change')
            .rolling_mean(window_size=252, min_samples=60)
            .over('symbol')
            .alias('vol_mean_252d'),
            
            pl.col('vol_change')
            .rolling_std(window_size=252, min_samples=60)
            .over('symbol')
            .alias('vol_std_252d'),
        ])
        
        df = df.with_columns([
            ((pl.col('vol_change') - pl.col('vol_mean_252d')) / pl.col('vol_std_252d'))
            .alias('z_volume')
        ])
        
        # MA ratios
        for window in [5, 10, 20]:
            df = df.with_columns([
                pl.col('prc')
                .rolling_mean(window_size=window, min_samples=window//2)
                .over('symbol')
                .alias(f'ma_{window}d')
            ])
        
        df = df.with_columns([
            (pl.col('prc') / pl.col('ma_5d') - 1).alias('ma_ratio_5d'),
            (pl.col('prc') / pl.col('ma_10d') - 1).alias('ma_ratio_10d'),
            (pl.col('prc') / pl.col('ma_20d') - 1).alias('ma_ratio_20d'),
        ])
        
        return df
    
    def _compute_momentum_features_polars(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute momentum features using Polars.
        
        Features: mom_1m, mom_3m, mom_6m, mom_12_1m
        """
        # Compute cumulative returns
        df = df.with_columns([
            (1 + pl.col('ret_1d')).log().alias('log_ret')
        ])
        
        # Momentum windows (in trading days)
        windows = {
            'mom_1m': 21,
            'mom_3m': 63,
            'mom_6m': 126,
            'mom_12m': 252,
        }
        
        for col_name, window in windows.items():
            df = df.with_columns([
                pl.col('log_ret')
                .rolling_sum(window_size=window, min_samples=window//2)
                .over('symbol')
                .alias(f'log_ret_{col_name}')
            ])
            
            df = df.with_columns([
                (pl.col(f'log_ret_{col_name}').exp() - 1).alias(col_name)
            ])
        
        # 12_1m momentum: 12-month skip last 1-month
        df = df.with_columns([
            ((1 + pl.col('mom_12m')) / (1 + pl.col('mom_1m')) - 1).alias('mom_12_1m')
        ])
        
        return df
    
    def _compute_betas_polars(self, df: pl.DataFrame, window: int = 60) -> pl.DataFrame:
        """Compute rolling market betas using OLS regression.
        
        Uses market excess return (VWRETD from CRSP) as market return proxy.
        For individual stocks, computes beta = cov(stock_ret, market_ret) / var(market_ret)
        
        Args:
            df: DataFrame with 'ret_1d' column and market returns
            window: Rolling window in days (default 60)
        
        Returns:
            DataFrame with 'beta' and 'idiosyncratic_vol' columns
        """
        # Compute rolling covariance and variance for beta calculation
        # Beta = Cov(stock_ret, market_ret) / Var(market_ret)
        # Since we don't have individual market returns per stock, we use:
        # - Cross-sectional market return as proxy (mean return across all stocks)
        
        # Compute market return as cross-sectional mean
        df = df.with_columns([
            pl.col('ret_1d').mean().over('date').alias('market_ret_1d')
        ])
        
        # Compute rolling means for covariance calculation
        df = df.with_columns([
            pl.col('ret_1d').rolling_mean(window_size=window, min_samples=window//2).over('symbol').alias('ret_mean'),
            pl.col('market_ret_1d').rolling_mean(window_size=window, min_samples=window//2).over('symbol').alias('market_ret_mean')
        ])
        
        # Compute deviations from mean
        df = df.with_columns([
            (pl.col('ret_1d') - pl.col('ret_mean')).alias('ret_dev'),
            (pl.col('market_ret_1d') - pl.col('market_ret_mean')).alias('market_ret_dev')
        ])
        
        # Compute rolling covariance and variance
        df = df.with_columns([
            (pl.col('ret_dev') * pl.col('market_ret_dev')).rolling_mean(window_size=window, min_samples=window//2).over('symbol').alias('cov_stock_market'),
            (pl.col('market_ret_dev') * pl.col('market_ret_dev')).rolling_mean(window_size=window, min_samples=window//2).over('symbol').alias('var_market')
        ])
        
        # Compute beta
        df = df.with_columns([
            (pl.col('cov_stock_market') / pl.col('var_market')).alias('beta_60d')
        ])
        
        # Compute idiosyncratic volatility (residual volatility)
        # Predicted return = beta * market_ret
        df = df.with_columns([
            (pl.col('beta_60d') * pl.col('market_ret_1d')).alias('predicted_ret')
        ])
        
        # Residual = actual - predicted
        df = df.with_columns([
            (pl.col('ret_1d') - pl.col('predicted_ret')).alias('residual')
        ])
        
        # Idiosyncratic volatility = std of residuals
        df = df.with_columns([
            pl.col('residual').rolling_std(window_size=window, min_samples=window//2).over('symbol').alias('idiosyncratic_vol')
        ])
        
        # Annualize idiosyncratic volatility
        df = df.with_columns([
            (pl.col('idiosyncratic_vol') * np.sqrt(252)).alias('idiosyncratic_vol')
        ])
        
        # Rename beta column to match expected naming
        df = df.rename({'beta_60d': 'beta'})
        
        # Clean up intermediate columns
        df = df.drop(['ret_mean', 'market_ret_mean', 'ret_dev', 'market_ret_dev', 
                      'cov_stock_market', 'var_market', 'market_ret_1d',
                      'predicted_ret', 'residual'])
        
        return df
    
    def _compute_volatility_features_polars(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute volatility features using Polars.
        
        Features: realized_vol_20d, realized_vol_60d
        """
        # Realized volatility (annualized)
        for window, label in [(20, '20d'), (60, '60d')]:
            df = df.with_columns([
                pl.col('ret_1d')
                .rolling_std(window_size=window, min_samples=window//2)
                .over('symbol')
                .alias(f'ret_std_{label}')
            ])
            
            # Annualize: std * sqrt(252)
            df = df.with_columns([
                (pl.col(f'ret_std_{label}') * np.sqrt(252)).alias(f'realized_vol_{label}')
            ])
        
        return df
    
    def _merge_betas(
        self,
        features_df: pd.DataFrame,
        betas_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge pre-computed betas into features."""
        # Check if betas_df has the required columns
        required_cols = ['symbol', 'date', 'beta', 'idiosyncratic_vol']
        available_cols = [col for col in required_cols if col in betas_df.columns]
        
        if not available_cols:
            logger.warning("Beta data missing required columns, skipping merge")
            return features_df
        
        # Merge on symbol and date
        merged = features_df.merge(
            betas_df[available_cols],
            on=['symbol', 'date'],
            how='left'
        )
        return merged
    
    def _merge_fundamentals(
        self,
        features_df: pd.DataFrame,
        fundamentals_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge fundamentals with merge_asof (forward fill)."""
        # Convert dates to datetime
        features_df['date'] = pd.to_datetime(features_df['date'])
        fundamentals_df['rdq'] = pd.to_datetime(fundamentals_df['rdq'])
        
        # Merge using merge_asof for forward-fill behavior
        merged = pd.merge_asof(
            features_df.sort_values('date'),
            fundamentals_df.sort_values('rdq'),
            left_on='date',
            right_on='rdq',
            by='symbol',
            direction='backward'  # Use most recent fundamental data
        )
        
        return merged
    
    def _merge_gics(
        self,
        features_df: pd.DataFrame,
        gics_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge GICS sector codes."""
        merged = features_df.merge(
            gics_df[['symbol', 'gsector', 'ggroup']],
            on='symbol',
            how='left'
        )
        return merged
    
    def apply_normalization(
        self,
        df: pd.DataFrame,
        feature_groups: dict[str, list[str]]
    ) -> pd.DataFrame:
        """Apply two-pass normalization to feature groups.
        
        Args:
            df: DataFrame with raw features
            feature_groups: Dict of {group_name: [feature_cols]}
            
        Returns:
            DataFrame with normalized features
        """
        result = df.copy()
        
        for group_name, feature_cols in feature_groups.items():
            logger.info(f"Normalizing {group_name} features...")
            
            # Filter to columns that exist
            existing_cols = [c for c in feature_cols if c in result.columns]
            
            if existing_cols:
                result = two_pass_normalization(
                    result,
                    feature_cols=existing_cols,
                    date_col='date'
                )
        
        return result
