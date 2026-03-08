"""Stock recommender pipeline for finding liquidity substitutes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from src.types.common import RecommendationResult

if TYPE_CHECKING:
    import torch


@dataclass
class RecommenderConfig:
    """Configuration for the recommender system."""

    top_k: int = 10
    similarity_threshold: float = 0.5
    use_cached_embeddings: bool = True


class StockRecommender:
    """Recommender system for finding liquid stock substitutes."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: RecommenderConfig | None = None,
    ) -> None:
        """Initialize stock recommender.

        Args:
            model: Dual encoder model for embedding generation
            config: Recommender configuration
        """
        self.model = model
        self.config = config or RecommenderConfig()
        self._embeddings_cache: dict[str, np.ndarray] = {}
        self._fitted = False

    def fit(self, data: pl.DataFrame) -> StockRecommender:
        """Fit recommender on training data.

        Args:
            data: DataFrame with price and fundamental features

        Returns:
            Self for method chaining
        """
        raise NotImplementedError("Recommender fit not implemented")

    def find_substitutes(
        self,
        symbol: str,
        top_k: int | None = None,
    ) -> list[RecommendationResult]:
        """Find liquid substitutes for a given stock.

        Args:
            symbol: Target stock symbol
            top_k: Number of substitutes to return (overrides config)

        Returns:
            List of recommended substitute stocks with scores
        """
        raise NotImplementedError("Find substitutes not implemented")

    def get_recommendations(
        self,
        symbols: list[str],
        top_k: int | None = None,
    ) -> dict[str, list[RecommendationResult]]:
        """Get recommendations for multiple symbols.

        Args:
            symbols: List of target stock symbols
            top_k: Number of substitutes per symbol

        Returns:
            Dictionary mapping symbols to their recommendations
        """
        raise NotImplementedError("Get recommendations not implemented")

    def compute_similarity(
        self,
        symbol_a: str,
        symbol_b: str,
    ) -> float:
        """Compute similarity score between two stocks.

        Args:
            symbol_a: First stock symbol
            symbol_b: Second stock symbol

        Returns:
            Similarity score in [0, 1]
        """
        raise NotImplementedError("Similarity computation not implemented")

    def _load_embeddings(self, symbol: str) -> np.ndarray:
        """Load embeddings from cache or compute if not cached.

        Args:
            symbol: Stock symbol

        Returns:
            Embedding vector
        """
        if symbol in self._embeddings_cache:
            return self._embeddings_cache[symbol]
        raise KeyError(f"Embedding for {symbol} not found in cache")

    def _save_embedding(self, symbol: str, embedding: np.ndarray) -> None:
        """Save embedding to cache.

        Args:
            symbol: Stock symbol
            embedding: Embedding vector
        """
        self._embeddings_cache[symbol] = embedding

    def clear_cache(self) -> None:
        """Clear embeddings cache."""
        self._embeddings_cache.clear()
