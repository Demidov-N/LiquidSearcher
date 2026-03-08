"""G2 volatility features (realized and idiosyncratic volatility)."""

import numpy as np
import pandas as pd
from scipy import stats

from src.features.base import FeatureGroup


class G2VolatilityFeatures(FeatureGroup):
    """G2 feature group: volatility metrics.

    Computes realized volatility (20d and 60d), idiosyncratic volatility
    (residual after beta regression), and volatility of volatility.
    """

    def __init__(self) -> None:
        """Initialize G2 volatility features."""
        self.name = "G2_volatility"
        self._feature_names = [
            "realized_vol_20d",
            "realized_vol_60d",
            "idiosyncratic_vol",
            "vol_of_vol",
        ]

    def get_feature_names(self) -> list[str]:
        """Return list of feature names produced by this group."""
        return self._feature_names.copy()

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute G2 volatility features.

        Args:
            df: Input dataframe with columns:
                - symbol: Stock identifier
                - date: Date column
                - return: Stock returns
                - market_return: Market returns (for idiosyncratic vol)

        Returns:
            DataFrame with G2 volatility features added.
        """
        result = df.copy()

        # Compute realized volatility
        result = self._compute_realized_vol(result, window=20, col_name="realized_vol_20d")
        result = self._compute_realized_vol(result, window=60, col_name="realized_vol_60d")

        # Compute idiosyncratic volatility
        result = self._compute_idiosyncratic_vol(result, window=60)

        # Compute volatility of volatility
        result = self._compute_vol_of_vol(result, vol_col="realized_vol_20d", window=20)

        # Normalize: log transform then cross-sectional z-score
        raw_features = [
            "realized_vol_20d",
            "realized_vol_60d",
            "idiosyncratic_vol",
            "vol_of_vol",
        ]

        # Log transform (add small constant to avoid log(0))
        for col in raw_features:
            if col in result.columns:
                result[f"{col}_log"] = np.log(result[col] + 1e-8)

        # Cross-sectional z-score on log-transformed values
        log_features = [f"{col}_log" for col in raw_features]
        result = self._cross_sectional_zscore(result, log_features, group_by="date")

        return result

    def _compute_realized_vol(
        self,
        df: pd.DataFrame,
        window: int = 20,
        min_periods: int = 10,
        col_name: str = "realized_vol",
    ) -> pd.DataFrame:
        """Compute rolling realized volatility (annualized)."""
        result = df.copy()

        def rolling_vol(group: pd.DataFrame) -> pd.Series:
            """Calculate rolling volatility for a single symbol."""
            returns = group["return"].values
            vols = []

            for i in range(len(group)):
                if i < min_periods - 1:
                    vols.append(np.nan)
                    continue

                start_idx = max(0, i - window + 1)
                window_returns = returns[start_idx : i + 1]

                # Remove NaN values
                valid_returns = window_returns[~np.isnan(window_returns)]

                if len(valid_returns) < min_periods:
                    vols.append(np.nan)
                    continue

                # Annualized volatility: std * sqrt(252)
                vol = np.std(valid_returns) * np.sqrt(252)
                vols.append(vol)

            return pd.Series(vols, index=group.index)

        # Apply rolling vol computation per symbol
        if len(result) > 0 and "return" in result.columns:
            vol_list = []
            for _symbol, group in result.groupby("symbol", group_keys=False):
                vol_values = rolling_vol(group)
                vol_list.append(pd.Series(vol_values, index=group.index))

            if vol_list:
                vol_series = pd.concat(vol_list)
                result[col_name] = vol_series.reindex(result.index)
        else:
            result[col_name] = np.nan

        return result

    def _compute_idiosyncratic_vol(
        self, df: pd.DataFrame, window: int = 60, min_periods: int = 30
    ) -> pd.DataFrame:
        """Compute idiosyncratic volatility (residual after beta regression)."""
        result = df.copy()

        if "market_return" not in result.columns:
            result["idiosyncratic_vol"] = np.nan
            return result

        def rolling_idio_vol(group: pd.DataFrame) -> pd.Series:
            """Calculate rolling idiosyncratic volatility for a single symbol."""
            returns = group["return"].values
            market_returns = group["market_return"].values
            idio_vols = []

            for i in range(len(group)):
                if i < min_periods - 1:
                    idio_vols.append(np.nan)
                    continue

                start_idx = max(0, i - window + 1)
                stock_ret = returns[start_idx : i + 1]
                mkt_ret = market_returns[start_idx : i + 1]

                # Remove NaN values
                mask = ~(np.isnan(stock_ret) | np.isnan(mkt_ret))
                if mask.sum() < min_periods:
                    idio_vols.append(np.nan)
                    continue

                stock_ret_clean = stock_ret[mask]
                mkt_ret_clean = mkt_ret[mask]

                # OLS regression to get beta
                try:
                    slope, intercept, _, _, _ = stats.linregress(mkt_ret_clean, stock_ret_clean)

                    # Compute residuals
                    predicted = intercept + slope * mkt_ret_clean
                    residuals = stock_ret_clean - predicted

                    # Idiosyncratic volatility (annualized)
                    idio_vol = np.std(residuals) * np.sqrt(252)
                    idio_vols.append(max(0, idio_vol))  # Ensure non-negative

                except Exception:
                    idio_vols.append(np.nan)

            return pd.Series(idio_vols, index=group.index)

        # Apply rolling idiosyncratic vol computation per symbol
        if len(result) > 0:
            idio_vol_list = []
            for _symbol, group in result.groupby("symbol", group_keys=False):
                idio_vol_values = rolling_idio_vol(group)
                idio_vol_list.append(pd.Series(idio_vol_values, index=group.index))

            if idio_vol_list:
                idio_vol_series = pd.concat(idio_vol_list)
                result["idiosyncratic_vol"] = idio_vol_series.reindex(result.index)
        else:
            result["idiosyncratic_vol"] = np.nan

        return result

    def _compute_vol_of_vol(
        self,
        df: pd.DataFrame,
        vol_col: str = "realized_vol_20d",
        window: int = 20,
        min_periods: int = 10,
    ) -> pd.DataFrame:
        """Compute volatility of volatility (std of rolling vol estimates)."""
        result = df.copy()

        if vol_col not in result.columns:
            result["vol_of_vol"] = np.nan
            return result

        def rolling_vovol(group: pd.DataFrame) -> pd.Series:
            """Calculate rolling volatility of volatility for a single symbol."""
            vols = group[vol_col].values
            vovols = []

            for i in range(len(group)):
                if i < min_periods - 1:
                    vovols.append(np.nan)
                    continue

                start_idx = max(0, i - window + 1)
                window_vols = vols[start_idx : i + 1]

                # Remove NaN values
                valid_vols = window_vols[~np.isnan(window_vols)]

                if len(valid_vols) < min_periods:
                    vovols.append(np.nan)
                    continue

                # Vol of vol: std of volatility estimates
                vovol = np.std(valid_vols)
                vovols.append(vovol)

            return pd.Series(vovols, index=group.index)

        # Apply rolling vol of vol computation per symbol
        if len(result) > 0:
            vovol_list = []
            for _symbol, group in result.groupby("symbol", group_keys=False):
                vovol_values = rolling_vovol(group)
                vovol_list.append(pd.Series(vovol_values, index=group.index))

            if vovol_list:
                vovol_series = pd.concat(vovol_list)
                result["vol_of_vol"] = vovol_series.reindex(result.index)
        else:
            result["vol_of_vol"] = np.nan

        return result
