"""
LiquidSearcher: Liquidity-aware stock substitute recommendation system.

A dual-encoder contrastive learning system for financial similarity learning,
designed for production deployment with MLOps best practices.
"""

__version__ = "0.1.0"
__author__ = "Developer"

# Core model exports
from .models.dual_encoder import DualEncoder
from .models.temporal_encoder import TemporalEncoder
from .models.tabular_encoder import TabularEncoder

# Data loading utilities
from .data.wrds_loader import WRDSLoader
from .data.universe import SymbolUniverse

# Feature engineering
from .features.processor import FeatureProcessor
from .features.normalization import winsorize, cross_sectional_zscore, rank_normalize

__all__ = [
    "DualEncoder",
    "TemporalEncoder",
    "TabularEncoder",
    "WRDSLoader",
    "SymbolUniverse",
    "FeatureProcessor",
    "winsorize",
    "cross_sectional_zscore",
    "rank_normalize",
]
