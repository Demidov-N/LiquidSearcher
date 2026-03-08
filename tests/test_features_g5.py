"""Test G5 OHLCV behavior features."""

import numpy as np
import pandas as pd

from src.features.g5_ohlcv import G5OHLCVFeatures


def test_g5_initialization():
    """Test G5 feature group initialization."""
    g5 = G5OHLCVFeatures()
    assert g5.name == "G5_ohlcv"

    # Verify all expected features exist
    feature_names = g5.get_feature_names()
    expected_features = [
        "z_close_5d",
        "z_close_10d",
        "z_close_20d",
        "z_high",
        "z_low",
        "z_volume_5d",
        "z_volume_10d",
        "z_volume_20d",
        "ma_ratio_5",
        "ma_ratio_10",
        "ma_ratio_15",
        "ma_ratio_20",
        "ma_ratio_25",
    ]

    for feature in expected_features:
        assert feature in feature_names, f"Missing feature: {feature}"


def test_ohlcv_computation():
    """Test OHLCV feature computation."""
    np.random.seed(42)
    n_days = 300  # Need enough for 252-day window + calculations
    n_stocks = 3

    # Create synthetic OHLCV data for multiple stocks
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")

    data = []
    for stock_idx in range(n_stocks):
        symbol = f"STOCK{stock_idx}"
        # Generate random walk for price
        returns = np.random.normal(0.001, 0.02, n_days)
        close = 100 * np.exp(np.cumsum(returns))

        # Generate high and low based on close
        high = close * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
        low = close * (1 - np.abs(np.random.normal(0, 0.01, n_days)))

        # Generate volume
        volume = np.random.lognormal(15, 0.5, n_days)

        for i, date in enumerate(dates):
            data.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "close": close[i],
                    "high": high[i],
                    "low": low[i],
                    "volume": volume[i],
                }
            )

    df = pd.DataFrame(data)

    g5 = G5OHLCVFeatures()
    result = g5.compute(df)

    # Check all feature columns exist
    for feature in g5.get_feature_names():
        assert feature in result.columns, f"Missing column: {feature}"

    # Check MA ratio features are computed (price/MA - 1)
    for window in [5, 10, 15, 20, 25]:
        col_name = f"ma_ratio_{window}"
        assert col_name in result.columns
        # MA ratio should be finite (not all NaN after initial periods)
        valid_values = result[col_name].dropna()
        assert len(valid_values) > 0

    # Check z-score features exist
    z_features = [
        "z_close_5d",
        "z_close_10d",
        "z_close_20d",
        "z_high",
        "z_low",
        "z_volume_5d",
        "z_volume_10d",
        "z_volume_20d",
    ]
    for feature in z_features:
        assert feature in result.columns


def test_rolling_zscore_per_stock():
    """Test that z-scores are centered around 0 per stock."""
    np.random.seed(42)
    n_days = 300

    # Create data with different volatilities for different stocks
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")

    data = []
    for symbol, vol in [("HIGH_VOL", 0.04), ("LOW_VOL", 0.01)]:
        returns = np.random.normal(0.001, vol, n_days)
        close = 100 * np.exp(np.cumsum(returns))
        high = close * (1 + np.abs(np.random.normal(0, vol * 0.5, n_days)))
        low = close * (1 - np.abs(np.random.normal(0, vol * 0.5, n_days)))
        volume = np.random.lognormal(15, 0.5, n_days)

        for i, date in enumerate(dates):
            data.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "close": close[i],
                    "high": high[i],
                    "low": low[i],
                    "volume": volume[i],
                }
            )

    df = pd.DataFrame(data)

    g5 = G5OHLCVFeatures()
    result = g5.compute(df)

    # Check that z-scores are roughly centered at 0 per stock
    for symbol in ["HIGH_VOL", "LOW_VOL"]:
        symbol_data = result[result["symbol"] == symbol]

        # Skip initial NaN periods
        for feature in ["z_close_5d", "z_volume_5d"]:
            valid_zscores = symbol_data[feature].dropna()
            if len(valid_zscores) > 50:  # Need enough data
                mean_zscore = valid_zscores.mean()
                # Z-scores should be centered around 0 (within reasonable tolerance)
                assert abs(mean_zscore) < 0.5, (
                    f"Z-score mean for {symbol} {feature} is {mean_zscore}, should be near 0"
                )
