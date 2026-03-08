"""G3 momentum features (price momentum and trend indicators)."""

import numpy as np
import pandas as pd

from src.features.base import FeatureGroup


class G3MomentumFeatures(FeatureGroup):
    """G3 feature group: momentum indicators.

    Computes classic momentum measures including:
    - mom_1m: 1-month price momentum (21 trading days)
    - mom_3m: 3-month price momentum (63 trading days)
    - mom_6m: 6-month price momentum (126 trading days)
    - mom_12_1m: 12-month minus 1-month momentum (Jegadeesh-Titman)
    - macd: Moving Average Convergence Divergence

    All features are normalized using cross-sectional rank [0, 1].
    """

    def __init__(self) -> None:
        """Initialize G3 momentum features."""
        self.name = "G3_momentum"
        self._feature_names = [
            "mom_1m",
            "mom_3m",
            "mom_6m",
            "mom_12_1m",
            "macd",
        ]

    def get_feature_names(self) -> list[str]:
        """Return list of feature names produced by this group."""
        return self._feature_names.copy()

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute G3 momentum features.

        Args:
            df: Input dataframe with columns:
                - symbol: Stock identifier
                - date: Date column
                - close: Closing price

        Returns:
            DataFrame with G3 momentum features added.
        """
        result = df.copy()

        # Compute momentum features
        result = self._compute_momentum(result, periods=21, feature_name="mom_1m")
        result = self._compute_momentum(result, periods=63, feature_name="mom_3m")
        result = self._compute_momentum(result, periods=126, feature_name="mom_6m")
        result = self._compute_momentum_12_1m(result)
        result = self._compute_macd(result)

        # Normalize features using cross-sectional rank [0, 1]
        raw_features = ["mom_1m", "mom_3m", "mom_6m", "mom_12_1m", "macd"]
        result = self._cross_sectional_rank(result, raw_features, group_by="date")

        return result

    def _compute_momentum(self, df: pd.DataFrame, periods: int, feature_name: str) -> pd.DataFrame:
        """Compute price momentum: (P_t / P_{t-n}) - 1.

        Args:
            df: Input dataframe
            periods: Number of periods to look back
            feature_name: Name for the output feature column

        Returns:
            DataFrame with momentum feature added
        """
        result = df.copy()

        if "close" not in result.columns:
            result[feature_name] = np.nan
            return result

        # Compute momentum per symbol using transform
        def calc_momentum(prices: pd.Series) -> pd.Series:
            """Calculate momentum for a price series."""
            shifted = prices.shift(periods)
            momentum = (prices / shifted) - 1
            return momentum

        # Apply per symbol
        result[feature_name] = result.groupby("symbol")["close"].transform(calc_momentum)

        return result

    def _compute_momentum_12_1m(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute 12-month skip 1-month momentum (Jegadeesh-Titman).

        This is: (P_t / P_{t-252}) - (P_t / P_{t-21})
        Or equivalently: mom_12m - mom_1m

        Args:
            df: Input dataframe

        Returns:
            DataFrame with mom_12_1m feature added
        """
        result = df.copy()

        if "close" not in result.columns:
            result["mom_12_1m"] = np.nan
            return result

        # Compute 12-1m momentum per symbol using transform
        def calc_momentum_12_1m(prices: pd.Series) -> pd.Series:
            """Calculate 12-1m momentum for a price series."""
            prices_shifted_12m = prices.shift(252)  # ~12 months
            prices_shifted_1m = prices.shift(21)  # ~1 month

            mom_12m = (prices / prices_shifted_12m) - 1
            mom_1m = (prices / prices_shifted_1m) - 1

            return mom_12m - mom_1m

        # Apply per symbol
        result["mom_12_1m"] = result.groupby("symbol")["close"].transform(calc_momentum_12_1m)

        return result

    def _compute_macd(
        self, df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26
    ) -> pd.DataFrame:
        """Compute MACD (Moving Average Convergence Divergence).

        MACD = EMA(12) - EMA(26)

        Args:
            df: Input dataframe
            fast_period: Fast EMA period (default 12)
            slow_period: Slow EMA period (default 26)

        Returns:
            DataFrame with MACD feature added
        """
        result = df.copy()

        if "close" not in result.columns:
            result["macd"] = np.nan
            return result

        # Compute MACD per symbol using transform
        def calc_macd(prices: pd.Series) -> pd.Series:
            """Calculate MACD for a price series."""
            ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
            ema_slow = prices.ewm(span=slow_period, adjust=False).mean()
            return ema_fast - ema_slow

        # Apply per symbol
        result["macd"] = result.groupby("symbol")["close"].transform(calc_macd)

        return result
