"""Unified feature engineer orchestrating G1-G6 computation."""

import pandas as pd

from src.data.cache_manager import CacheManager
from src.features.base import FeatureRegistry
from src.features.g1_risk import G1RiskFeatures
from src.features.g2_volatility import G2VolatilityFeatures
from src.features.g3_momentum import G3MomentumFeatures
from src.features.g4_valuation import G4ValuationFeatures
from src.features.g5_ohlcv import G5OHLCVFeatures
from src.features.g6_sector import G6SectorFeatures


class FeatureEngineer:
    """Orchestrates computation of all G1-G6 feature groups.

    This class provides a unified interface for computing all feature groups
    with caching support for efficient repeated computation.

    Attributes:
        registry: FeatureRegistry containing all registered feature groups
        cache_manager: Optional CacheManager for parquet caching
    """

    def __init__(self, cache_manager: CacheManager | None = None) -> None:
        """Initialize feature engineer with all G1-G6 groups.

        Args:
            cache_manager: Optional cache manager for parquet caching
        """
        self.registry = FeatureRegistry()
        self.cache_manager = cache_manager

        # Register all G1-G6 feature groups
        self.registry.register("G1", G1RiskFeatures())
        self.registry.register("G2", G2VolatilityFeatures())
        self.registry.register("G3", G3MomentumFeatures())
        self.registry.register("G4", G4ValuationFeatures())
        self.registry.register("G5", G5OHLCVFeatures())
        self.registry.register("G6", G6SectorFeatures())

    def compute_features(self, df: pd.DataFrame, cache_key: str | None = None) -> pd.DataFrame:
        """Compute all registered feature groups.

        Args:
            df: Input dataframe with required columns for feature computation
            cache_key: Optional cache key for storing/retrieving results

        Returns:
            DataFrame with all feature columns added
        """
        # Check cache first if cache_key provided
        if cache_key and self.cache_manager:
            cached = self.cache_manager.get(cache_key)
            if cached is not None:
                return cached

        # Compute all features
        result = self.registry.compute_all(df)

        # Cache result if cache_key provided
        if cache_key and self.cache_manager:
            self.cache_manager.set(cache_key, result)

        return result

    def compute_single_group(self, df: pd.DataFrame, group_name: str) -> pd.DataFrame:
        """Compute a single feature group by name (for debugging).

        Args:
            df: Input dataframe
            group_name: Name of the feature group (G1-G6)

        Returns:
            DataFrame with only that group's features added

        Raises:
            ValueError: If group_name is not registered
        """
        group = self.registry.get(group_name)
        if group is None:
            raise ValueError(
                f"Feature group '{group_name}' not found. Available: {self.registry.list_groups()}"
            )

        return group.compute(df)

    def get_feature_names(self) -> dict[str, list[str]]:
        """Return feature names organized by group.

        Returns:
            Dictionary mapping group names to lists of feature names
        """
        return {name: group.get_feature_names() for name, group in self.registry._groups.items()}

    def get_all_feature_names(self) -> list[str]:
        """Return all feature names from all groups.

        Returns:
            Flattened list of all feature names
        """
        all_names: list[str] = []
        for group in self.registry._groups.values():
            all_names.extend(group.get_feature_names())
        return all_names
