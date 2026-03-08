"""G1 systematic risk features (beta and factor loadings)."""

import numpy as np
import pandas as pd
from scipy import stats

from src.features.base import FeatureGroup


class G1RiskFeatures(FeatureGroup):
    """G1 feature group: systematic risk exposure.

    Computes market beta, downside beta, and factor loadings
    for Fama-French 5 factors.
    """

    def __init__(self) -> None:
        """Initialize G1 risk features."""
        self.name = "G1_risk"
        self._feature_names = [
            "market_beta_60d",
            "downside_beta_60d",
            "smb_loading",
            "hml_loading",
            "mom_loading",
            "rmw_loading",
            "cma_loading",
        ]

    def get_feature_names(self) -> list[str]:
        """Return list of feature names produced by this group."""
        return self._feature_names.copy()

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute G1 risk features.

        Args:
            df: Input dataframe with columns:
                - symbol: Stock identifier
                - date: Date column
                - return: Stock returns
                - market_return: Market returns (for beta)
                - smb_factor, hml_factor, mom_factor, rmw_factor, cma_factor: FF5 factors

        Returns:
            DataFrame with G1 risk features added.
        """
        result = df.copy()

        # Compute market beta (60-day rolling OLS)
        result = self._compute_market_beta(result, window=60)

        # Compute downside beta
        result = self._compute_downside_beta(result, window=60)

        # Compute factor loadings (252-day window)
        result = self._compute_factor_loadings(result, window=252)

        # Normalize features: winsorize then cross-sectional z-score
        raw_features = [
            "market_beta_60d",
            "downside_beta_60d",
            "smb_loading",
            "hml_loading",
            "mom_loading",
            "rmw_loading",
            "cma_loading",
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

    def _compute_factor_loadings(
        self, df: pd.DataFrame, window: int = 252, min_periods: int = 126
    ) -> pd.DataFrame:
        """Compute Fama-French 5 factor loadings via rolling OLS."""
        result = df.copy()

        factor_columns = ["smb_factor", "hml_factor", "mom_factor", "rmw_factor", "cma_factor"]
        factor_names = ["smb_loading", "hml_loading", "mom_loading", "rmw_loading", "cma_loading"]

        # Check if all factor columns are present
        missing_factors = [col for col in factor_columns if col not in result.columns]
        if missing_factors:
            # If factors are missing, set all loadings to NaN
            for name in factor_names:
                result[name] = np.nan
            return result

        def rolling_factor_loadings(group: pd.DataFrame) -> pd.DataFrame:
            """Calculate rolling factor loadings for a single symbol."""
            n = len(group)
            loadings = {name: [] for name in factor_names}

            returns = group["return"].values
            factors = {col: group[col].values for col in factor_columns}

            for i in range(n):
                if i < min_periods - 1:
                    for name in factor_names:
                        loadings[name].append(np.nan)
                    continue

                start_idx = max(0, i - window + 1)
                stock_ret = returns[start_idx : i + 1]

                # Build factor matrix
                factor_matrix = np.column_stack(
                    [factors[col][start_idx : i + 1] for col in factor_columns]
                )

                # Remove rows with any NaN
                valid_mask = ~(np.isnan(stock_ret) | np.isnan(factor_matrix).any(axis=1))

                if valid_mask.sum() < min_periods:
                    for name in factor_names:
                        loadings[name].append(np.nan)
                    continue

                stock_ret_clean = stock_ret[valid_mask]
                factor_matrix_clean = factor_matrix[valid_mask]

                # Multiple regression via least squares
                try:
                    # Add intercept
                    x_matrix = np.column_stack(
                        [np.ones(len(factor_matrix_clean)), factor_matrix_clean]
                    )

                    # Solve: beta = (X'X)^(-1) X'y
                    coeffs, _, _, _ = np.linalg.lstsq(x_matrix, stock_ret_clean, rcond=None)

                    # coeffs[0] is intercept, coeffs[1:] are factor loadings
                    for idx, name in enumerate(factor_names):
                        loadings[name].append(coeffs[idx + 1])

                except Exception:
                    for name in factor_names:
                        loadings[name].append(np.nan)

            return pd.DataFrame(loadings, index=group.index)

        # Apply rolling factor loadings computation per symbol
        loadings_df = result.groupby("symbol", group_keys=False).apply(rolling_factor_loadings)

        # Merge loadings back into result
        for name in factor_names:
            result[name] = loadings_df[name].values

        return result
