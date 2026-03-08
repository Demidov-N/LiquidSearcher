"""Feature pipeline for building and processing features."""

from collections.abc import Callable
from dataclasses import dataclass, field

import polars as pl


@dataclass
class FeatureConfig:
    """Configuration for a single feature."""

    name: str
    func: Callable[[pl.DataFrame], pl.Series]
    params: dict = field(default_factory=dict)


class FeaturePipeline:
    """Pipeline for building and processing features."""

    def __init__(self) -> None:
        """Initialize empty feature pipeline."""
        self._features: list[FeatureConfig] = []
        self._fitted = False

    def add_feature(
        self,
        name: str,
        func: Callable[[pl.DataFrame], pl.Series],
        **params: object,
    ) -> "FeaturePipeline":
        """Add a feature to the pipeline.

        Args:
            name: Feature name
            func: Function to compute feature from DataFrame
            params: Additional parameters for the feature

        Returns:
            Self for method chaining
        """
        config = FeatureConfig(name=name, func=func, params=params)
        self._features.append(config)
        self._fitted = False
        return self

    def process(self, data: pl.DataFrame) -> pl.DataFrame:
        """Process data through the feature pipeline.

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with added feature columns
        """
        raise NotImplementedError("Feature processing not implemented")

    def get_features(self) -> list[str]:
        """Get list of feature names in pipeline.

        Returns:
            List of feature names
        """
        return [f.name for f in self._features]

    def fit(self, data: pl.DataFrame) -> "FeaturePipeline":
        """Fit pipeline on training data.

        Args:
            data: Training DataFrame

        Returns:
            Self for method chaining
        """
        self._fitted = True
        return self

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        """Transform data using fitted pipeline.

        Args:
            data: Input DataFrame

        Returns:
            Transformed DataFrame
        """
        return self.process(data)

    def fit_transform(self, data: pl.DataFrame) -> pl.DataFrame:
        """Fit and transform data in one step.

        Args:
            data: Input DataFrame

        Returns:
            Transformed DataFrame
        """
        self.fit(data)
        return self.transform(data)
