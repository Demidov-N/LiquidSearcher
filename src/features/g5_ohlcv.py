"""G5 OHLCV behavior features (price behavior)."""

import pandas as pd

from src.features.base import FeatureGroup


class G5OHLCVFeatures(FeatureGroup):
    """G5 feature group: OHLCV behavior and technical features.

    Computes z-scores of returns, high/low relative to close,
    z-scores of volume changes, and moving average ratios.
    All features use rolling time-series z-score normalization
    per stock's own 252-day history.
    """

    def __init__(self) -> None:
        """Initialize G5 OHLCV features."""
        self.name = "G5_ohlcv"
        self._feature_names = [
            "z_close_5d",
            "z_close_10d",
            "z_close_20d",
            "z_high",
            "z_low",
            "z_volume_5d",
            "z_volume_10d",
            "z_volume_20d",
            "ma_ratio_5",
            "ma_ratio_10",
            "ma_ratio_15",
            "ma_ratio_20",
            "ma_ratio_25",
        ]

    def get_feature_names(self) -> list[str]:
        """Return list of feature names produced by this group."""
        return self._feature_names.copy()

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute G5 OHLCV features.

        Args:
            df: Input dataframe with columns:
                - symbol: Stock identifier
                - date: Date column
                - close: Close price
                - high: High price
                - low: Low price
                - volume: Trading volume

        Returns:
            DataFrame with G5 OHLCV features added.
        """
        result = df.copy()

        # Compute returns
        result = self._compute_returns(result)

        # Compute moving average ratios
        result = self._compute_ma_ratios(result)

        # Compute high/low z-scores
        result = self._compute_high_low_zscores(result)

        # Compute volume change z-scores
        result = self._compute_volume_zscores(result)

        # Normalize: rolling z-score per stock
        # This normalizes all the raw features using each stock's own history
        raw_features = [
            "ret_5d",  # 5-day return over MA
            "ret_10d",  # 10-day return over MA
            "ret_20d",  # 20-day return over MA
            "high_vs_close",
            "low_vs_close",
            "volume_change_5d",
            "volume_change_10d",
            "volume_change_20d",
            "ma_ratio_5",
            "ma_ratio_10",
            "ma_ratio_15",
            "ma_ratio_20",
            "ma_ratio_25",
        ]

        # Apply rolling z-score normalization per stock
        result = self._compute_rolling_zscores(result, raw_features)

        return result

    def _compute_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute returns over different windows."""
        result = df.copy()

        # Ensure data is sorted by symbol and date
        result = result.sort_values(["symbol", "date"]).reset_index(drop=True)

        # Compute close returns over different MA windows
        for window in [5, 10, 20]:
            # Compute moving average
            ma_col = f"ma_{window}"
            result[ma_col] = (
                result.groupby("symbol")["close"]
                .rolling(window=window, min_periods=window)
                .mean()
                .reset_index(level=0, drop=True)
            )

            # Return relative to MA: (close / MA) - 1
            result[f"ret_{window}d"] = (result["close"] / result[ma_col]) - 1

        return result

    def _compute_ma_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute moving average ratios: (close / MA_n) - 1."""
        result = df.copy()

        # Ensure data is sorted
        result = result.sort_values(["symbol", "date"]).reset_index(drop=True)

        for window in [5, 10, 15, 20, 25]:
            # Compute moving average
            ma = (
                result.groupby("symbol")["close"]
                .rolling(window=window, min_periods=window)
                .mean()
                .reset_index(level=0, drop=True)
            )

            # MA ratio: (close / MA) - 1
            result[f"ma_ratio_{window}"] = (result["close"] / ma) - 1

        return result

    def _compute_high_low_zscores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute high and low relative to close."""
        result = df.copy()

        # High relative to close: (high - close) / close
        result["high_vs_close"] = (result["high"] - result["close"]) / result["close"]

        # Low relative to close: (low - close) / close (negative)
        result["low_vs_close"] = (result["low"] - result["close"]) / result["close"]

        return result

    def _compute_volume_zscores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute volume changes over different windows."""
        result = df.copy()

        # Ensure data is sorted
        result = result.sort_values(["symbol", "date"]).reset_index(drop=True)

        # Compute volume changes: (volume / volume_MA) - 1
        for window in [5, 10, 20]:
            vol_ma = (
                result.groupby("symbol")["volume"]
                .rolling(window=window, min_periods=window)
                .mean()
                .reset_index(level=0, drop=True)
            )

            result[f"volume_change_{window}d"] = (result["volume"] / vol_ma) - 1

        return result

    def _compute_rolling_zscores(
        self,
        df: pd.DataFrame,
        columns: list[str],
        window: int = 252,
        min_periods: int = 63,
    ) -> pd.DataFrame:
        """Compute rolling z-scores and assign to final feature names."""
        result = df.copy()

        # Ensure data is sorted
        result = result.sort_values(["symbol", "date"]).reset_index(drop=True)

        # Mapping from raw columns to final feature names
        feature_mapping = {
            "ret_5d": "z_close_5d",
            "ret_10d": "z_close_10d",
            "ret_20d": "z_close_20d",
            "high_vs_close": "z_high",
            "low_vs_close": "z_low",
            "volume_change_5d": "z_volume_5d",
            "volume_change_10d": "z_volume_10d",
            "volume_change_20d": "z_volume_20d",
            "ma_ratio_5": "ma_ratio_5",
            "ma_ratio_10": "ma_ratio_10",
            "ma_ratio_15": "ma_ratio_15",
            "ma_ratio_20": "ma_ratio_20",
            "ma_ratio_25": "ma_ratio_25",
        }

        for col in columns:
            if col not in result.columns:
                continue

            # Compute rolling mean per stock
            rolling_mean = (
                result.groupby("symbol")[col]
                .rolling(window=window, min_periods=min_periods)
                .mean()
                .reset_index(level=0, drop=True)
            )

            # Compute rolling std per stock
            rolling_std = (
                result.groupby("symbol")[col]
                .rolling(window=window, min_periods=min_periods)
                .std()
                .reset_index(level=0, drop=True)
            )

            # Compute z-score
            zscore = (result[col] - rolling_mean) / rolling_std.replace(0, 1e-8)

            # Assign to final feature name
            final_name = feature_mapping.get(col, col)
            result[final_name] = zscore

        return result
