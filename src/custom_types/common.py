from typing import TypeAlias

import polars as pl

OHLCVData: TypeAlias = pl.DataFrame
FundamentalData: TypeAlias = pl.DataFrame
FeatureVector: TypeAlias = list[float]
FeatureMatrix: TypeAlias = list[FeatureVector]
Embedding: TypeAlias = list[float]
Score: TypeAlias = float


class RecommendationResult:
    symbol: str
    score: Score
    metadata: dict

    def __init__(self, symbol: str, score: Score, metadata: dict | None = None) -> None:
        self.symbol = symbol
        self.score = score
        self.metadata = metadata or {}


class LiquidityMetrics:
    ADVANCE_DECLINE_RATIO: str = "advance_decline_ratio"
    AMIHUD_ILLIQUIDITY: str = "amihud_illiquidity"
    TRADING_VOLUME_RATIO: str = "trading_volume_ratio"
    TURNOVER_RATE: str = "turnover_rate"
    BID_ASK_SPREAD: str = "bid_ask_spread"
    EFFECTIVE_SPRID: str = "effective_spread"
    REALIZED_SPREAD: str = "realized_spread"
    VOLUME_WEIGHTED_PRICE: str = "vwap"
    LIQUIDITY_SCORE: str = "liquidity_score"
