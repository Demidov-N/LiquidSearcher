"""Feature normalization module for standardizing input features."""

from abc import ABC, abstractmethod
from enum import Enum, auto

import numpy as np


class NormalizerType(Enum):
    """Types of normalization strategies."""

    STANDARD = auto()
    MINMAX = auto()
    ROBUST = auto()


class Normalizer(ABC):
    """Abstract base class for feature normalizers."""

    @abstractmethod
    def fit(self, data: np.ndarray) -> "Normalizer":
        """Fit normalizer to training data.

        Args:
            data: Training data array

        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted parameters.

        Args:
            data: Data to transform

        Returns:
            Normalized data
        """
        pass

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform data in one step.

        Args:
            data: Training data

        Returns:
            Normalized data
        """
        self.fit(data)
        return self.transform(data)


class StandardNormalizer(Normalizer):
    """Z-score normalization (zero mean, unit variance)."""

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, data: np.ndarray) -> "StandardNormalizer":
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Normalizer must be fitted before transform")
        return (data - self.mean) / self.std


class MinMaxNormalizer(Normalizer):
    """Min-max scaling to [0, 1] range."""

    def __init__(self) -> None:
        self.min: np.ndarray | None = None
        self.max: np.ndarray | None = None

    def fit(self, data: np.ndarray) -> "MinMaxNormalizer":
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.min is None or self.max is None:
            raise RuntimeError("Normalizer must be fitted before transform")
        return (data - self.min) / (self.max - self.min)


class RobustNormalizer(Normalizer):
    """Robust scaling using median and IQR."""

    def __init__(self) -> None:
        self.median: np.ndarray | None = None
        self.iqr: np.ndarray | None = None

    def fit(self, data: np.ndarray) -> "RobustNormalizer":
        self.median = np.median(data, axis=0)
        q75 = np.percentile(data, 75, axis=0)
        q25 = np.percentile(data, 25, axis=0)
        self.iqr = q75 - q25
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.median is None or self.iqr is None:
            raise RuntimeError("Normalizer must be fitted before transform")
        return (data - self.median) / self.iqr


def create_normalizer(norm_type: NormalizerType) -> Normalizer:
    """Factory function to create normalizer by type.

    Args:
        norm_type: Type of normalization

    Returns:
        Normalizer instance
    """
    normalizers = {
        NormalizerType.STANDARD: StandardNormalizer,
        NormalizerType.MINMAX: MinMaxNormalizer,
        NormalizerType.ROBUST: RobustNormalizer,
    }
    return normalizers[norm_type]()
