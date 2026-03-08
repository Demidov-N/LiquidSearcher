"""Valuation and fundamentals features (P/E, P/B, ROE, market cap)."""

import numpy as np
import pandas as pd

from src.features.base import FeatureGroup


class ValuationFeatures(FeatureGroup):
    """Valuation feature group: valuation and fundamental metrics.

    Computes market capitalization, P/E ratio, P/B ratio, and ROE
    with appropriate normalization.
    """

    def __init__(self) -> None:
        """Initialize valuation features."""
        self.name = "valuation"
        self._feature_names = [
            "log_mktcap",
            "pe_ratio",
            "pb_ratio",
            "roe",
        ]

    def get_feature_names(self) -> list[str]:
        """Return list of feature names produced by this group."""
        return self._feature_names.copy()

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute valuation features.

        Args:
            df: Input dataframe with columns:
                - symbol: Stock identifier
                - date: Date column
                - price: Stock price (or close)
                - shares_outstanding: Number of shares outstanding
                - eps: Earnings per share
                - book_value_per_share: Book value per share
                - net_income: Net income
                - equity: Shareholder equity
                OR pre-computed values:
                - market_cap: Market capitalization
                - pe_ratio: P/E ratio
                - pb_ratio: P/B ratio
                - roe: Return on equity

        Returns:
            DataFrame with valuation features added.
        """
        result = df.copy()

        # Compute market cap and log transform
        result = self._compute_market_cap(result)

        # Compute P/E ratio (winsorize [2%, 98%] → rank [0,1])
        result = self._compute_pe_ratio(result)

        # Compute P/B ratio (log → z-score)
        result = self._compute_pb_ratio(result)

        # Compute ROE (winsorize → z-score)
        result = self._compute_roe(result)

        return result

    def _compute_market_cap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute log market capitalization: ln(price × shares_outstanding)."""
        result = df.copy()

        # If market_cap already exists (from fundamentals), use it directly
        if "market_cap" in result.columns and not result["market_cap"].isna().all():
            mktcap = result["market_cap"].replace(0, np.nan)
            result["log_mktcap"] = np.log(mktcap)
            return result

        # Check if we can compute from price and shares_outstanding
        # Use 'close' as price if 'price' column doesn't exist
        price_col = "price" if "price" in result.columns else "close"

        if price_col not in result.columns or "shares_outstanding" not in result.columns:
            result["log_mktcap"] = np.nan
            return result

        # Market cap = price × shares_outstanding
        mktcap = result[price_col] * result["shares_outstanding"]

        # Log transform (handle negative or zero values)
        mktcap = mktcap.replace(0, np.nan)
        result["log_mktcap"] = np.log(mktcap)

        return result

    def _compute_pe_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute P/E ratio: Price / EPS (winsorize [2%, 98%] → rank [0,1])."""
        result = df.copy()

        # If pe_ratio already exists (from fundamentals), preserve it
        if "pe_ratio" in result.columns and not result["pe_ratio"].isna().all():
            # Already have valid pe_ratio values, normalize them
            result["pe_ratio_raw"] = result["pe_ratio"]
        elif "price" not in result.columns and "close" not in result.columns:
            # Can't compute without price
            result["pe_ratio"] = np.nan
            return result
        elif "eps" not in result.columns:
            # Can't compute without EPS
            result["pe_ratio"] = np.nan
            return result
        else:
            # Use 'close' as price if 'price' column doesn't exist
            price_col = "price" if "price" in result.columns else "close"

            # Compute raw P/E ratio
            pe = result[price_col] / result["eps"]

            # Handle division by zero and negative values
            pe = pe.replace([np.inf, -np.inf], np.nan)
            pe = pe.where(pe > 0, np.nan)

            result["pe_ratio_raw"] = pe

        # Winsorize at 2% and 98% quantiles
        result = self._winsorize(result, ["pe_ratio_raw"], 0.02, 0.98)

        # Cross-sectional rank to [0, 1]
        result["pe_ratio"] = result.groupby("date")["pe_ratio_raw"].rank(pct=True)

        return result

    def _compute_pb_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute P/B ratio: Price / Book_value (log → z-score)."""
        result = df.copy()

        # If pb_ratio already exists (from fundamentals), use it directly
        if "pb_ratio" in result.columns and not result["pb_ratio"].isna().all():
            # Already have valid pb_ratio values, just return them
            return result

        # Check if we can compute from components
        has_price = "price" in result.columns or "close" in result.columns
        has_book = "book_value_per_share" in result.columns

        if not has_price or not has_book:
            result["pb_ratio"] = np.nan
            return result

        # Compute from components
        price_col = "price" if "price" in result.columns else "close"

        # Compute raw P/B ratio
        pb = result[price_col] / result["book_value_per_share"]

        # Handle division by zero and negative values
        pb = pb.replace([np.inf, -np.inf], np.nan)
        pb = pb.where(pb > 0, np.nan)

        # Log transform
        pb = pb.replace(0, np.nan)
        result["pb_ratio_raw"] = np.log(pb)

        # Cross-sectional z-score
        result["pb_ratio"] = (
            result["pb_ratio_raw"] - result.groupby("date")["pb_ratio_raw"].transform("mean")
        ) / result.groupby("date")["pb_ratio_raw"].transform("std").replace(0, 1e-8)

        return result

    def _compute_roe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute ROE: Net_income / Equity (winsorize → z-score)."""
        result = df.copy()

        # If roe already exists (from fundamentals), use it directly
        if "roe" in result.columns and not result["roe"].isna().all():
            # Already have valid roe values, just return them
            return result

        # Check if we can compute from components
        if "net_income" not in result.columns or "equity" not in result.columns:
            result["roe"] = np.nan
            return result

        # Compute from components
        roe = result["net_income"] / result["equity"]

        # Handle division by zero
        roe = roe.replace([np.inf, -np.inf], np.nan)

        result["roe_raw"] = roe

        # Winsorize at 1% and 99% (default from base class)
        result = self._winsorize(result, ["roe_raw"], 0.01, 0.99)

        # Cross-sectional z-score
        result["roe"] = (
            result["roe_raw"] - result.groupby("date")["roe_raw"].transform("mean")
        ) / result.groupby("date")["roe_raw"].transform("std").replace(0, 1e-8)

        return result
