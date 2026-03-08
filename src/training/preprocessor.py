"""Standalone preprocessing pipeline for inference on new stocks."""

import pandas as pd
import torch

from src.features.engineer import FeatureEngineer


class InferencePreprocessor:
    """Preprocesses raw stock data into model-ready features for inference.

    This class provides a standalone pipeline to compute all 32 features
    for any stock, enabling real-time inference on new securities.

    Usage:
        preprocessor = InferencePreprocessor()
        features = preprocessor.compute_features(prices, fundamentals)
        temporal, tabular = preprocessor.to_tensors(features)
    """

    MIN_HISTORY_DAYS = 252  # Minimum days needed for feature computation

    # Column definitions for feature groups
    TEMPORAL_COLS = [
        "z_score_20d",
        "z_score_60d",
        "ma_ratio_20_60",
        "volume_trend",
        "volatility_20d",
        "momentum_20d",
        "momentum_60d",
        "momentum_252d",
        "momentum_ratio",
        "reversal_5d",
        "parkinson_vol",
        "garch_vol",
        "price_trend",
    ]

    TABULAR_CONT_COLS = [
        "market_beta_60d",
        "downside_beta_60d",
        "realized_vol_20d",
        "realized_vol_60d",
        "garch_vol",
        "parkinson_vol",
        "momentum_20d",
        "momentum_60d",
        "momentum_252d",
        "momentum_ratio",
        "reversal_5d",
        "pe_ratio",
        "pb_ratio",
        "roe",
        "market_cap_log",
    ]

    TABULAR_CAT_COLS = ["gsector", "ggroup"]

    def __init__(self) -> None:
        """Initialize preprocessor with feature engineer."""
        self.engineer = FeatureEngineer()

    def compute_features(
        self,
        prices: pd.DataFrame,
        fundamentals: pd.DataFrame | None = None,
        min_history: int = MIN_HISTORY_DAYS,
    ) -> pd.DataFrame:
        """Compute all 32 features from raw price and fundamental data.

        Args:
            prices: DataFrame with OHLCV data (index=date)
            fundamentals: DataFrame with fundamental data (optional)
            min_history: Minimum required history in days

        Returns:
            DataFrame with all 32 features computed

        Raises:
            ValueError: If insufficient history for feature computation
        """
        if len(prices) < min_history:
            raise ValueError(
                f"Insufficient history: got {len(prices)} days, minimum {min_history} days required"
            )

        # Compute all feature groups using existing engineer
        features = self.engineer.compute_features(prices)

        return features

    def extract_latest_window(
        self,
        features: pd.DataFrame,
        window_size: int = 60,
    ) -> dict[str, pd.DataFrame | pd.Series]:
        """Extract the latest window for model input.

        Args:
            features: DataFrame with all computed features
            window_size: Number of days for temporal window (default 60)

        Returns:
            Dictionary with temporal window and tabular snapshot
        """
        # Extract last window_size days for temporal features
        temporal_window = features.iloc[-window_size:][self.TEMPORAL_COLS]

        # Extract latest snapshot for tabular features
        latest = features.iloc[-1]
        tabular_cont = latest[self.TABULAR_CONT_COLS]
        tabular_cat = latest[self.TABULAR_CAT_COLS]

        return {
            "temporal": temporal_window,
            "tabular_cont": tabular_cont,
            "tabular_cat": tabular_cat,
        }

    def to_tensors(
        self,
        window_dict: dict[str, pd.DataFrame | pd.Series],
        normalize: bool = False,
        stats: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        """Convert pandas data to PyTorch tensors.

        Args:
            window_dict: Output from extract_latest_window()
            normalize: Whether to normalize using training statistics
            stats: Training statistics (mean, std) for normalization

        Returns:
            Dictionary with tensors ready for model input
        """
        temporal = torch.tensor(
            window_dict["temporal"].values,
            dtype=torch.float32,
        )  # (60, 13)

        tabular_cont = torch.tensor(
            window_dict["tabular_cont"].values,
            dtype=torch.float32,
        )  # (15,)

        tabular_cat = torch.tensor(
            window_dict["tabular_cat"].values,
            dtype=torch.long,
        )  # (2,)

        if normalize and stats is not None:
            # Apply training statistics
            temporal = (temporal - stats["temporal_mean"]) / stats["temporal_std"]
            tabular_cont = (tabular_cont - stats["tabular_mean"]) / stats["tabular_std"]

        return {
            "temporal": temporal,
            "tabular_cont": tabular_cont,
            "tabular_cat": tabular_cat,
        }

    def preprocess_stock(
        self,
        prices: pd.DataFrame,
        fundamentals: pd.DataFrame | None = None,
        stats: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        """End-to-end preprocessing: raw data → model tensors.

        This is the main entry point for inference preprocessing.

        Args:
            prices: Raw OHLCV price data
            fundamentals: Raw fundamental data (optional)
            stats: Training normalization statistics

        Returns:
            Dictionary with tensors ready for model.forward()
        """
        # Step 1: Compute all features
        features = self.compute_features(prices, fundamentals)

        # Step 2: Extract latest window
        window_dict = self.extract_latest_window(features)

        # Step 3: Convert to tensors
        tensors = self.to_tensors(
            window_dict,
            normalize=(stats is not None),
            stats=stats,
        )

        # Add metadata for reference
        tensors["date"] = features.index[-1]
        tensors["symbol"] = prices.attrs.get("symbol", "unknown")

        return tensors
