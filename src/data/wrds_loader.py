"""WRDS data loading module for accessing Wharton Research Data Services."""

import os
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl

from src.data.cache_manager import CacheManager


@dataclass
class WRDSConfig:
    """Configuration for WRDS connection.

    Credentials can be provided directly or via environment variables:
    - WRDS_USERNAME
    - WRDS_PASSWORD
    """

    username: str | None = None
    password: str | None = None
    host: str = "wrds-cloud.wharton.upenn.edu"
    port: int = 9737

    def get_username(self) -> str:
        """Get username from config or environment."""
        return self.username or os.getenv("WRDS_USERNAME", "")

    def get_password(self) -> str:
        """Get password from config or environment."""
        return self.password or os.getenv("WRDS_PASSWORD", "")

    def has_credentials(self) -> bool:
        """Check if credentials are available."""
        return bool(self.get_username() and self.get_password())


class WRDSConnection:
    """Wrapper for WRDS database connection with context manager support."""

    def __init__(self, config: WRDSConfig | None = None) -> None:
        """Initialize WRDS connection with config.

        Args:
            config: WRDS configuration. If None, uses environment variables.
        """
        self.config = config or WRDSConfig()
        self._connection = None
        self._mock_mode = False

    def connect(self) -> None:
        """Establish connection to WRDS.

        Raises:
            ImportError: If wrds library not installed
            ConnectionError: If connection fails
            RuntimeError: If no credentials available
        """
        if not self.config.has_credentials():
            # No credentials - use mock mode
            self._mock_mode = True
            return

        try:
            import wrds

            self._connection = wrds.Connection(
                wrds_username=self.config.get_username(),
                wrds_password=self.config.get_password(),
            )
        except ImportError as err:
            raise ImportError("wrds library not installed. Run: pip install wrds") from err
        except Exception as err:
            raise ConnectionError(f"Failed to connect to WRDS: {err}") from err

    def disconnect(self) -> None:
        """Close WRDS connection."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        self._mock_mode = False

    def is_connected(self) -> bool:
        """Check if connection is active (real or mock)."""
        return self._connection is not None or self._mock_mode

    def is_mock_mode(self) -> bool:
        """Check if running in mock mode."""
        return self._mock_mode

    def get_connection(self):
        """Get raw WRDS connection object.

        Returns:
            WRDS connection object or None if mock mode

        Raises:
            RuntimeError: If not connected
        """
        if self._mock_mode:
            return None
        if not self.is_connected():
            raise RuntimeError("Not connected to WRDS")
        return self._connection

    def __enter__(self) -> "WRDSConnection":
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Context manager exit."""
        self.disconnect()


def load_ohlcv(
    symbols: list[str],
    start_date: datetime,
    end_date: datetime,
    frequency: str = "D",
    use_cache: bool = True,
    use_mock: bool = False,
    output_format: str = "pandas",
) -> pd.DataFrame | pl.DataFrame:
    """Load OHLCV data from WRDS or generate sample data.

    Args:
        symbols: List of ticker symbols
        start_date: Start date for data retrieval
        end_date: End date for data retrieval
        frequency: Data frequency (D=Daily, W=Weekly, M=Monthly)
        use_cache: Whether to use cached data if available
        use_mock: Force mock data generation (bypasses WRDS)
        output_format: "pandas" (default) or "polars"

    Returns:
        DataFrame with OHLCV data in requested format
    """
    # Check for cached data
    if use_cache:
        cache = CacheManager()
        cache_key = f"ohlcv_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_{len(symbols)}symbols"
        cached = cache.get(cache_key)
        if cached is not None:
            return _convert_format(cached, output_format)

    # Determine if we use mock data
    config = WRDSConfig()
    should_use_mock = use_mock or not config.has_credentials()

    if frequency != "D":
        raise NotImplementedError("Only daily frequency supported")

    if should_use_mock:
        df = _generate_mock_ohlcv(symbols, start_date, end_date)
    else:
        df = _fetch_ohlcv_from_wrds(symbols, start_date, end_date)

    # Cache the result
    if use_cache:
        cache = CacheManager()
        cache.set(cache_key, df)

    return _convert_format(df, output_format)


def _generate_mock_ohlcv(
    symbols: list[str],
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame:
    """Generate mock OHLCV data for testing/development."""
    # Generate business days
    dates = pd.date_range(start=start_date, end=end_date, freq="B")

    data_rows = []
    for symbol in symbols:
        base_price = np.random.uniform(10, 500)
        returns = np.random.normal(0, 0.02, len(dates))
        prices = base_price * np.exp(np.cumsum(returns))

        for i, date in enumerate(dates):
            data_rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": prices[i] * (1 + np.random.uniform(-0.01, 0.01)),
                    "high": prices[i] * (1 + np.random.uniform(0, 0.03)),
                    "low": prices[i] * (1 + np.random.uniform(-0.03, 0)),
                    "close": prices[i],
                    "volume": int(np.random.lognormal(10, 1.5)),
                    "adj_close": prices[i],
                    "return": returns[i],
                }
            )

    return pd.DataFrame(data_rows)


def _fetch_ohlcv_from_wrds(
    symbols: list[str],
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame:
    """Fetch OHLCV data from WRDS."""
    config = WRDSConfig()

    with WRDSConnection(config) as conn:
        wrds_conn = conn.get_connection()

        # Build SQL query
        tickers_str = ", ".join([f"'{s.upper()}'" for s in symbols])

        query = f"""
            SELECT
                a.permno,
                a.date,
                a.prc,
                a.ret,
                a.vol,
                a.shrout,
                a.bidlo,
                a.askhi,
                a.cfacpr,
                b.ticker
            FROM crsp.dsf AS a
            INNER JOIN crsp.dsenames AS b ON a.permno = b.permno
            WHERE b.ticker IN ({tickers_str})
            AND a.date BETWEEN '{start_date.strftime("%Y-%m-%d")}' AND '{end_date.strftime("%Y-%m-%d")}'
            AND a.date BETWEEN b.namedt AND b.nameenddt
            ORDER BY b.ticker, a.date
        """

        df = wrds_conn.raw_sql(query)
        return df


def load_fundamental(
    symbols: list[str],
    start_date: datetime,
    end_date: datetime,
    metrics: list[str] | None = None,
    use_cache: bool = True,
    use_mock: bool = False,
    output_format: str = "pandas",
) -> pd.DataFrame | pl.DataFrame:
    """Load fundamental data from WRDS or generate sample data.

    Args:
        symbols: List of ticker symbols
        start_date: Start date for data retrieval
        end_date: End date for data retrieval
        metrics: List of fundamental metrics to retrieve
        use_cache: Whether to use cached data if available
        use_mock: Force mock data generation
        output_format: "pandas" (default) or "polars"

    Returns:
        DataFrame with fundamental data in requested format
    """
    if metrics is None:
        metrics = ["pe_ratio", "pb_ratio", "market_cap", "roe", "debt_equity", "gsector", "ggroup"]

    # Check cache
    if use_cache:
        cache = CacheManager()
        cache_key = f"fundamental_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_{len(symbols)}symbols"
        cached = cache.get(cache_key)
        if cached is not None:
            return _convert_format(cached, output_format)

    # Determine if we use mock data
    config = WRDSConfig()
    should_use_mock = use_mock or not config.has_credentials()

    if should_use_mock:
        df = _generate_mock_fundamental(symbols, start_date, end_date, metrics)
    else:
        df = _fetch_fundamental_from_wrds(symbols, start_date, end_date, metrics)

    # Cache result
    if use_cache:
        cache = CacheManager()
        cache.set(cache_key, df)

    return _convert_format(df, output_format)


def _generate_mock_fundamental(
    symbols: list[str],
    start_date: datetime,
    end_date: datetime,
    metrics: list[str],
) -> pd.DataFrame:
    """Generate mock fundamental data."""
    # Generate fiscal year-ends
    years = range(start_date.year, end_date.year + 1)

    gics_sectors = ["10", "15", "20", "25", "30", "35", "40", "45", "50", "55", "60"]
    gics_groups = ["1010", "1510", "2010", "2510", "3010", "3510"]

    data_rows = []
    for symbol in symbols:
        for year in years:
            row = {
                "symbol": symbol,
                "datadate": pd.Timestamp(f"{year}-12-31"),
                "fiscal_year": year,
            }

            if "pe_ratio" in metrics:
                row["pe_ratio"] = np.random.uniform(10, 50)
            if "pb_ratio" in metrics:
                row["pb_ratio"] = np.random.uniform(0.5, 10)
            if "market_cap" in metrics:
                row["market_cap"] = np.random.lognormal(20, 2)
            if "roe" in metrics:
                row["roe"] = np.random.uniform(-0.2, 0.3)
            if "debt_equity" in metrics:
                row["debt_equity"] = np.random.uniform(0, 3)
            if "dividend_yield" in metrics:
                row["dividend_yield"] = np.random.uniform(0, 0.05)
            if "gsector" in metrics:
                row["gsector"] = np.random.choice(gics_sectors)
            if "ggroup" in metrics:
                row["ggroup"] = np.random.choice(gics_groups)

            data_rows.append(row)

    return pd.DataFrame(data_rows)


def _fetch_fundamental_from_wrds(
    symbols: list[str],
    start_date: datetime,
    end_date: datetime,
    metrics: list[str],
) -> pd.DataFrame:
    """Fetch fundamental data from WRDS."""
    config = WRDSConfig()

    with WRDSConnection(config) as conn:
        wrds_conn = conn.get_connection()

        tickers_str = ", ".join([f"'{s.upper()}'" for s in symbols])

        # Map metrics to Compustat fields
        field_map = {
            "pe_ratio": "epspx",
            "pb_ratio": "book_val",
            "market_cap": "mkvalt",
            "roe": "ni",
            "debt_equity": "dt",
            "dividend_yield": "dv",
            "gsector": "gsector",
            "ggroup": "ggroup",
        }

        fields = ["gvkey", "datadate", "tic", "at", "seq", "ni", "csho", "prcc_f"]
        for m in metrics:
            if m in field_map and field_map[m] not in fields:
                fields.append(field_map[m])

        fields_str = ", ".join(fields)

        query = f"""
            SELECT {fields_str}
            FROM comp.funda
            WHERE tic IN ({tickers_str})
            AND datadate BETWEEN '{start_date.strftime("%Y-%m-%d")}' AND '{end_date.strftime("%Y-%m-%d")}'
            AND indfmt = 'INDL'
            AND datafmt = 'STD'
            AND popsrc = 'D'
            AND consol = 'C'
            ORDER BY tic, datadate
        """

        return wrds_conn.raw_sql(query)


def _convert_format(
    df: pd.DataFrame | pl.DataFrame,
    output_format: str,
) -> pd.DataFrame | pl.DataFrame:
    """Convert DataFrame to requested format."""
    if output_format == "polars" and isinstance(df, pd.DataFrame):
        return pl.from_pandas(df)
    elif output_format == "pandas" and isinstance(df, pl.DataFrame):
        return df.to_pandas()
    return df


def stream_ohlcv_batches(
    symbols: list[str],
    start_date: datetime,
    end_date: datetime,
    batch_size: int = 1000,
    output_format: str = "pandas",
) -> Iterator[pd.DataFrame | pl.DataFrame]:
    """Stream OHLCV data in batches for memory efficiency.

    Args:
        symbols: List of ticker symbols
        start_date: Start date for data retrieval
        end_date: End date for data retrieval
        batch_size: Number of rows per batch
        output_format: "pandas" or "polars"

    Yields:
        DataFrame batches in requested format
    """
    data = load_ohlcv(symbols, start_date, end_date, output_format=output_format)

    n_rows = len(data)
    for i in range(0, n_rows, batch_size):
        if output_format == "polars":
            yield data.slice(i, batch_size)
        else:
            yield data.iloc[i : i + batch_size]
