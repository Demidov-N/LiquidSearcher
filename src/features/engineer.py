"""Unified feature engineer with configurable feature groups."""

import pandas as pd

from src.data.cache_manager import CacheManager
from src.features.base import FeatureRegistry, FeatureGroup
from src.features.market_risk import MarketRiskFeatures
from src.features.momentum import MomentumFeatures
from src.features.sector import SectorFeatures
from src.features.technical import TechnicalFeatures
from src.features.valuation import ValuationFeatures
from src.features.volatility import VolatilityFeatures


class FeatureEngineer:
    """Orchestrates computation of feature groups with flexible configuration.

    This class provides a unified interface for computing feature groups
    with caching support for efficient repeated computation.

    Attributes:
        registry: FeatureRegistry containing registered feature groups
        cache_manager: Optional CacheManager for parquet caching
        enabled_groups: List of enabled feature group names
    """

    # Available feature group classes with their default names
    AVAILABLE_GROUPS: dict[str, type[FeatureGroup]] = {
        "market_risk": MarketRiskFeatures,
        "volatility": VolatilityFeatures,
        "momentum": MomentumFeatures,
        "valuation": ValuationFeatures,
        "technical": TechnicalFeatures,
        "sector": SectorFeatures,
    }

    def __init__(
        self,
        cache_manager: CacheManager | None = None,
        enabled_groups: list[str] | None = None,
    ) -> None:
        """Initialize feature engineer with configurable feature groups.

        Args:
            cache_manager: Optional cache manager for parquet caching
            enabled_groups: Optional list of group names to enable. If None,
                          all groups are enabled by default.

        Example:
            # Use all groups (default)
            engineer = FeatureEngineer()

            # Use only specific groups
            engineer = FeatureEngineer(enabled_groups=["market_risk", "momentum", "sector"])
        """
        self.registry = FeatureRegistry()
        self.cache_manager = cache_manager
        self.enabled_groups: list[str] = enabled_groups or list(self.AVAILABLE_GROUPS.keys())

        # Register enabled feature groups
        for name in self.enabled_groups:
            if name in self.AVAILABLE_GROUPS:
                self.registry.register(name, self.AVAILABLE_GROUPS[name]())

    def compute_features(self, df: pd.DataFrame, cache_key: str | None = None) -> pd.DataFrame:
        """Compute all enabled feature groups.

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
            group_name: Name of the feature group (e.g., "market_risk", "momentum")

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
        """Return all feature names from all enabled groups.

        Returns:
            Flattened list of all feature names
        """
        all_names: list[str] = []
        for group in self.registry._groups.values():
            all_names.extend(group.get_feature_names())
        return all_names

    def list_available_groups(self) -> list[str]:
        """Return list of all available feature group names.

        Returns:
            List of available group names
        """
        return list(self.AVAILABLE_GROUPS.keys())

    def list_enabled_groups(self) -> list[str]:
        """Return list of currently enabled feature group names.

        Returns:
            List of enabled group names
        """
        return self.enabled_groups.copy()

    def add_group(self, name: str) -> None:
        """Add a feature group to the enabled set.

        Args:
            name: Name of the group to add (must be in AVAILABLE_GROUPS)

        Raises:
            ValueError: If group name is not available
        """
        if name not in self.AVAILABLE_GROUPS:
            raise ValueError(
                f"Group '{name}' not available. Available: {self.list_available_groups()}"
            )
        if name not in self.enabled_groups:
            self.enabled_groups.append(name)
            self.registry.register(name, self.AVAILABLE_GROUPS[name]())

    def remove_group(self, name: str) -> None:
        """Remove a feature group from the enabled set.

        Args:
            name: Name of the group to remove

        Raises:
            ValueError: If group is not currently enabled
        """
        if name not in self.enabled_groups:
            raise ValueError(f"Group '{name}' is not enabled")
        self.enabled_groups.remove(name)
        if name in self.registry._groups:
            del self.registry._groups[name]
