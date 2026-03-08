MIN_LIQUIDITY_SCORE = 0.0
MAX_LIQUIDITY_SCORE = 1.0

DEFAULT_LIQUIDITY_THRESHOLD = 0.5

MIN_AVG_VOLUME = 10000.0

MIN_MARKET_CAP = 1_000_000_000.0

MIN_DAYS_TRADING = 252

VOLATILITY_WINDOW = 20
VOLATILITY_ANNUALIZATION_FACTOR = 16.0

LIQUIDITY_METRICS = (
    "advance_decline_ratio",
    "amihud_illiquidity",
    "trading_volume_ratio",
    "turnover_rate",
    "bid_ask_spread",
    "liquidity_score",
)
