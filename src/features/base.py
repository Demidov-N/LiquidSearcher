"""Feature engineering base infrastructure."""

from abc import ABC, abstractmethod
from enum import Enum

import pandas as pd


class NormalizationMethod(Enum):
    """Normalization methods for features."""

    Z_SCORE = "z_score"
    LOG_Z_SCORE = "log_z_score"
    RANK = "rank"
    ROLLING_Z_SCORE = "rolling_z_score"
    NONE = "none"


class FeatureGroup(ABC):
    """Abstract base class for feature groups."""

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute features and add them to the dataframe.

        Args:
            df: Input dataframe with at least 'symbol' and 'date' columns.

        Returns:
            DataFrame with new feature columns added.
        """
        ...

    @abstractmethod
    def get_feature_names(self) -> list[str]:
        """Return list of feature names produced by this group."""
        ...

    def _winsorize(
        self,
        df: pd.DataFrame,
        columns: list[str],
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
    ) -> pd.DataFrame:
        """Winsorize values at specified quantiles to reduce outliers."""
        result = df.copy()
        for col in columns:
            if col in result.columns:
                lower = result[col].quantile(lower_quantile)
                upper = result[col].quantile(upper_quantile)
                result[col] = result[col].clip(lower=lower, upper=upper)
        return result

    def _cross_sectional_zscore(
        self,
        df: pd.DataFrame,
        columns: list[str],
        group_by: str = "date",
    ) -> pd.DataFrame:
        """Compute cross-sectional z-scores within each group."""
        result = df.copy()
        for col in columns:
            if col in result.columns:
                mean = result.groupby(group_by)[col].transform("mean")
                std = result.groupby(group_by)[col].transform("std")
                result[f"{col}_zscore"] = (result[col] - mean) / std.replace(0, 1e-8)
        return result

    def _cross_sectional_rank(
        self,
        df: pd.DataFrame,
        columns: list[str],
        group_by: str = "date",
    ) -> pd.DataFrame:
        """Compute cross-sectional ranks within each group (normalized to [0, 1])."""
        result = df.copy()
        for col in columns:
            if col in result.columns:
                result[f"{col}_rank"] = result.groupby(group_by)[col].rank(pct=True)
        return result

    def _rolling_zscore(
        self,
        df: pd.DataFrame,
        columns: list[str],
        window: int = 63,
        min_periods: int = 20,
        group_by: str = "symbol",
    ) -> pd.DataFrame:
        """Compute rolling z-scores within each group."""
        result = df.copy()
        for col in columns:
            if col in result.columns:
                rolling_mean = (
                    result.groupby(group_by)[col]
                    .rolling(window=window, min_periods=min_periods)
                    .mean()
                    .reset_index(level=0, drop=True)
                )
                rolling_std = (
                    result.groupby(group_by)[col]
                    .rolling(window=window, min_periods=min_periods)
                    .std()
                    .reset_index(level=0, drop=True)
                )
                result[f"{col}_rolling_zscore"] = (
                    result[col] - rolling_mean
                ) / rolling_std.replace(0, 1e-8)
        return result


class FeatureRegistry:
    """Registry for managing feature groups."""

    def __init__(self) -> None:
        self._groups: dict[str, FeatureGroup] = {}

    def register(self, name: str, group: FeatureGroup) -> None:
        """Register a feature group."""
        self._groups[name] = group

    def get(self, name: str) -> FeatureGroup | None:
        """Get a feature group by name."""
        return self._groups.get(name)

    def list_groups(self) -> list[str]:
        """List all registered group names."""
        return list(self._groups.keys())

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all registered feature groups."""
        result = df.copy()
        for group in self._groups.values():
            result = group.compute(result)
        return result
