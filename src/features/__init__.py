"""Feature engineering pipeline (G1-G6)."""

from src.features.base import (
    FeatureGroup,
    FeatureRegistry,
    NormalizationMethod,
)
from src.features.engineer import FeatureEngineer
from src.features.g1_risk import G1RiskFeatures
from src.features.g2_volatility import G2VolatilityFeatures
from src.features.g3_momentum import G3MomentumFeatures
from src.features.g4_valuation import G4ValuationFeatures
from src.features.g5_ohlcv import G5OHLCVFeatures
from src.features.g6_sector import G6SectorFeatures

__all__ = [
    "FeatureGroup",
    "FeatureRegistry",
    "NormalizationMethod",
    "FeatureEngineer",
    "G1RiskFeatures",
    "G2VolatilityFeatures",
    "G3MomentumFeatures",
    "G4ValuationFeatures",
    "G5OHLCVFeatures",
    "G6SectorFeatures",
]
