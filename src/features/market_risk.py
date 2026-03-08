"""Market risk features (beta computations)."""

import numpy as np
import pandas as pd
from scipy import stats

from src.features.base import FeatureGroup


class MarketRiskFeatures(FeatureGroup):
    """Market risk feature group: systematic risk exposure.

    Computes market beta and downside beta (beta during negative market returns).
    Note: FF5 factor loadings excluded - using only market beta for methodology.
    """

    def __init__(self) -> None:
        """Initialize market risk features."""
        self.name = "market_risk"
        self._feature_names = [
            "market_beta_60d",
            "downside_beta_60d",
        ]

    def get_feature_names(self) -> list[str]:
        """Return list of feature names produced by this group."""
        return self._feature_names.copy()

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute market risk features.

        Args:
            df: Input dataframe with columns:
                - symbol: Stock identifier
                - date: Date column
                - return: Stock returns
                - market_return: Market returns (for beta computation)

        Returns:
            DataFrame with market risk features added.
        """
        result = df.copy()

        # Compute market beta (60-day rolling OLS)
        result = self._compute_market_beta(result, window=60)

        # Compute downside beta
        result = self._compute_downside_beta(result, window=60)

        # Normalize features: winsorize then cross-sectional z-score
        raw_features = [
            "market_beta_60d",
            "downside_beta_60d",
        ]

        # Winsorize at 1% and 99%
        result = self._winsorize(result, raw_features, 0.01, 0.99)

        # Cross-sectional z-score
        result = self._cross_sectional_zscore(result, raw_features, group_by="date")

        return result

    def _compute_market_beta(
        self, df: pd.DataFrame, window: int = 60, min_periods: int = 30
    ) -> pd.DataFrame:
        """Compute rolling market beta via OLS regression."""
        result = df.copy()

        if "market_return" not in result.columns:
            result["market_beta_60d"] = np.nan
            return result

        def rolling_beta(group: pd.DataFrame) -> pd.Series:
            """Calculate rolling beta for a single symbol."""
            betas = []
            returns = group["return"].values
            market_returns = group["market_return"].values

            for i in range(len(group)):
                if i < min_periods - 1:
                    betas.append(np.nan)
                    continue

                start_idx = max(0, i - window + 1)
                stock_ret = returns[start_idx : i + 1]
                mkt_ret = market_returns[start_idx : i + 1]

                # Remove NaN values
                mask = ~(np.isnan(stock_ret) | np.isnan(mkt_ret))
                if mask.sum() < min_periods:
                    betas.append(np.nan)
                    continue

                stock_ret_clean = stock_ret[mask]
                mkt_ret_clean = mkt_ret[mask]

                # OLS: beta = Cov(r_stock, r_market) / Var(r_market)
                # Using scipy.stats.linregress for simplicity
                if len(stock_ret_clean) >= min_periods:
                    try:
                        slope, _, _, _, _ = stats.linregress(mkt_ret_clean, stock_ret_clean)
                        betas.append(slope)
                    except Exception:
                        betas.append(np.nan)
                else:
                    betas.append(np.nan)

            return pd.Series(betas, index=group.index)

        # Apply rolling beta computation per symbol
        if len(result) > 0:
            beta_list = []
            for _symbol, group in result.groupby("symbol", group_keys=False):
                beta_values = rolling_beta(group)
                beta_list.append(pd.Series(beta_values, index=group.index))

            if beta_list:
                beta_series = pd.concat(beta_list)
                result["market_beta_60d"] = beta_series.reindex(result.index)
        else:
            result["market_beta_60d"] = np.nan

        return result

    def _compute_downside_beta(
        self, df: pd.DataFrame, window: int = 60, min_periods: int = 20
    ) -> pd.DataFrame:
        """Compute downside beta (beta on days with negative market returns)."""
        result = df.copy()

        if "market_return" not in result.columns:
            result["downside_beta_60d"] = np.nan
            return result

        def rolling_downside_beta(group: pd.DataFrame) -> pd.Series:
            """Calculate rolling downside beta for a single symbol."""
            betas = []
            returns = group["return"].values
            market_returns = group["market_return"].values

            for i in range(len(group)):
                if i < min_periods - 1:
                    betas.append(np.nan)
                    continue

                start_idx = max(0, i - window + 1)
                stock_ret = returns[start_idx : i + 1]
                mkt_ret = market_returns[start_idx : i + 1]

                # Filter to days with negative market returns
                downside_mask = mkt_ret < 0

                if downside_mask.sum() < min_periods:
                    betas.append(np.nan)
                    continue

                stock_ret_down = stock_ret[downside_mask]
                mkt_ret_down = mkt_ret[downside_mask]

                # Remove NaN values
                mask = ~(np.isnan(stock_ret_down) | np.isnan(mkt_ret_down))
                if mask.sum() < min_periods:
                    betas.append(np.nan)
                    continue

                stock_ret_clean = stock_ret_down[mask]
                mkt_ret_clean = mkt_ret_down[mask]

                # OLS: beta = Cov(r_stock, r_market) / Var(r_market)
                if len(stock_ret_clean) >= min_periods:
                    try:
                        slope, _, _, _, _ = stats.linregress(mkt_ret_clean, stock_ret_clean)
                        betas.append(slope)
                    except Exception:
                        betas.append(np.nan)
                else:
                    betas.append(np.nan)

            return pd.Series(betas, index=group.index)

        # Apply rolling downside beta computation per symbol
        if len(result) > 0:
            beta_list = []
            for _symbol, group in result.groupby("symbol", group_keys=False):
                beta_values = rolling_downside_beta(group)
                beta_list.append(pd.Series(beta_values, index=group.index))

            if beta_list:
                beta_series = pd.concat(beta_list)
                result["downside_beta_60d"] = beta_series.reindex(result.index)
        else:
            result["downside_beta_60d"] = np.nan

        return result
