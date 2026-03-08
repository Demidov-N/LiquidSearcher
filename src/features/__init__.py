"""Feature engineering pipeline with selectable feature groups."""

from src.features.base import (
    FeatureGroup,
    FeatureRegistry,
    NormalizationMethod,
)
from src.features.engineer import FeatureEngineer
from src.features.market_risk import MarketRiskFeatures
from src.features.momentum import MomentumFeatures
from src.features.sector import SectorFeatures
from src.features.technical import TechnicalFeatures
from src.features.valuation import ValuationFeatures
from src.features.volatility import VolatilityFeatures

__all__ = [
    "FeatureGroup",
    "FeatureRegistry",
    "NormalizationMethod",
    "FeatureEngineer",
    "MarketRiskFeatures",
    "MomentumFeatures",
    "SectorFeatures",
    "TechnicalFeatures",
    "ValuationFeatures",
    "VolatilityFeatures",
]
