OHLCV_COLUMNS = ("date", "symbol", "open", "high", "low", "close", "volume")

FUNDAMENTAL_COLUMNS = (
    "symbol",
    "date",
    "pe_ratio",
    "pb_ratio",
    "market_cap",
    "volume_avg",
    "dividend_yield",
)

DERIVED_FEATURES = (
    "returns",
    "log_returns",
    "volatility",
    " momentum",
    "volume_change",
)

NORMALIZATION_METHOD_STANDARD = "standard"
NORMALIZATION_METHOD_MINMAX = "minmax"
NORMALIZATION_METHOD_ROBUST = "robust"

DEFAULT_NORMALIZATION = NORMALIZATION_METHOD_STANDARD

FEATURE_DTYPE = "float64"
