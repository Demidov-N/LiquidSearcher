"""Feature normalization utilities with two-pass support."""

import numpy as np
import pandas as pd
from scipy import stats


def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Winsorize series at percentiles.
    
    Args:
        series: Input data
        lower: Lower percentile (default 1%)
        upper: Upper percentile (default 99%)
        
    Returns:
        Winsorized series
    """
    lower_val = series.quantile(lower)
    upper_val = series.quantile(upper)
    return series.clip(lower=lower_val, upper=upper_val)


def cross_sectional_zscore(
    df: pd.DataFrame,
    feature_col: str,
    date_col: str = 'date'
) -> pd.Series:
    """Compute cross-sectional z-score per date.
    
    Args:
        df: DataFrame with feature and date columns
        feature_col: Name of feature column to normalize
        date_col: Name of date column
        
    Returns:
        Series of z-scores
    """
    def zscore_transform(x):
        mean = x.mean()
        std = x.std()
        if std == 0 or pd.isna(std):
            return pd.Series(0.0, index=x.index)
        return (x - mean) / std
    
    result: pd.Series = df.groupby(date_col)[feature_col].transform(zscore_transform)
    return result


def rank_normalize(series: pd.Series) -> pd.Series:
    """Convert series to ranks normalized to [0, 1].
    
    Args:
        series: Input data
        
    Returns:
        Normalized ranks
    """
    ranks = series.rank(method='average')
    return (ranks - 1) / (len(ranks) - 1)


def rolling_zscore(
    series: pd.Series,
    window: int = 252,
    min_periods: int = 60
) -> pd.Series:
    """Compute rolling time-series z-score.
    
    Args:
        series: Input data
        window: Rolling window size
        min_periods: Minimum observations required
        
    Returns:
        Rolling z-scores
    """
    rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=window, min_periods=min_periods).std()
    
    return (series - rolling_mean) / rolling_std


def two_pass_normalization(
    df: pd.DataFrame,
    feature_cols: list[str],
    date_col: str = 'date',
    winsorize_limits: tuple = (0.01, 0.99)
) -> pd.DataFrame:
    """Apply two-pass normalization pipeline.
    
    Pass 1: Winsorize extreme values
    Pass 2: Cross-sectional z-score normalization
    
    Args:
        df: DataFrame with raw features
        feature_cols: List of feature columns to normalize
        date_col: Name of date column
        winsorize_limits: (lower, upper) percentiles for winsorization
        
    Returns:
        DataFrame with normalized features (new columns with _zscore suffix)
    """
    result = df.copy()
    
    for col in feature_cols:
        # Pass 1: Winsorize
        lower, upper = winsorize_limits
        result[f'{col}_winsorized'] = winsorize(result[col], lower, upper)
        
        # Pass 2: Cross-sectional z-score
        result[f'{col}_zscore'] = cross_sectional_zscore(
            result, f'{col}_winsorized', date_col
        )
    
    return result
