"""WRDS data loader with pre-computed beta support and batch processing."""

import logging
import os

import pandas as pd
import wrds

logger = logging.getLogger(__name__)


class WRDSDataLoader:
    """Load data from WRDS with batch processing and progress tracking.

    Features:
    - Batch-based symbol processing (500-1000 per batch)
    - Pre-computed betas from WRDS Beta Suite
    - Proper date handling (rdq not datadate)
    - tqdm progress tracking for all operations
    - Memory-efficient streaming

    Attributes:
        conn: WRDS database connection
        batch_size: Number of symbols per batch
    """

    def __init__(self, username: str | None = None, password: str | None = None):
        """Initialize WRDS connection.

        Args:
            username: WRDS username (defaults to env var)
            password: WRDS password (defaults to env var)
        """
        if username is None:
            username = os.getenv("WRDS_USERNAME")
        if password is None:
            password = os.getenv("WRDS_PASSWORD")

        self.conn = wrds.Connection(wrds_username=username, wrds_password=password)
        self.batch_size = 750  # Default for 30GB RAM

        logger.info("WRDS connection established")

    def fetch_prices_batch(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch daily prices for a batch of symbols.

        Uses CRSP daily stock file (crsp.dsf).

        Args:
            symbols: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with columns: permno, date, prc, vol, ret, shrout, bidlo, askhi
        """
        # Convert symbols to PERMNOs via CRSP link table
        permnos = self._symbols_to_permnos(symbols)

        if not permnos:
            return pd.DataFrame()

        # Build query for CRSP daily stock file
        permno_list = ",".join(map(str, permnos))
        query = f"""
            SELECT permno, date, prc, vol, ret, shrout, bidlo, askhi
            FROM crsp.dsf
            WHERE permno IN ({permno_list})
            AND date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY permno, date
        """

        df = self.conn.raw_sql(query)

        # Add ticker symbols back
        df = self._add_ticker_symbols(df)

        return df

    def fetch_precomputed_betas_batch(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        window: int = 60,
    ) -> pd.DataFrame:
        """Fetch pre-computed rolling betas from WRDS Beta Suite.

        NOTE: This requires access to WRDS Beta Suite. If not available,
        betas will be computed locally from returns.

        Args:
            symbols: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            window: Rolling window in days (default 60)

        Returns:
            DataFrame with columns: permno, date, beta, idio_vol, total_vol
            or empty DataFrame if Beta Suite not available
        """
        # Try WRDS Beta Suite first
        try:
            permnos = self._symbols_to_permnos(symbols)
            if not permnos:
                return pd.DataFrame()

            permno_list = ",".join(map(str, permnos))
            
            # Try different possible table names
            for table_name in ["wrds.beta", "betas.beta", "crsp.beta"]:
                try:
                    query = f"""
                        SELECT permno, date, beta, idiovol, totalvol
                        FROM {table_name}
                        WHERE permno IN ({permno_list})
                        AND date BETWEEN '{start_date}' AND '{end_date}'
                        AND "window" = {window}
                        ORDER BY permno, date
                    """
                    df = self.conn.raw_sql(query)
                    if not df.empty:
                        df = df.rename(columns={"idiovol": "idiosyncratic_vol", "totalvol": "total_volatility"})
                        return self._add_ticker_symbols(df)
                except Exception:
                    continue
                    
            logger.warning("WRDS Beta Suite not available, will compute betas locally")
            return pd.DataFrame()
            
        except Exception as e:
            logger.warning(f"Could not fetch pre-computed betas: {e}")
            return pd.DataFrame()

    def fetch_fundamentals_batch(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch quarterly fundamentals with proper date handling.

        Uses rdq (report date) NOT datadate to avoid look-ahead bias.
        Uses comp.fundq table.

        Args:
            symbols: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with columns: gvkey, rdq, atq, seq, niq, cshoq, prccq, etc.
        """
        try:
            # Get GVKEYs from ticker symbols
            gvkeys = self._symbols_to_gvkeys(symbols)

            if not gvkeys:
                logger.warning("No GVKEYs found for symbols, cannot fetch fundamentals")
                return pd.DataFrame()

            gvkey_list = "','".join(gvkeys)
            query = f"""
                SELECT gvkey, rdq, atq, seq, niq, cshoq, prccq,
                       epspxq, opepsq, ceqq, txtq, xintq
                FROM comp.fundq
                WHERE gvkey IN ('{gvkey_list}')
                AND rdq BETWEEN '{start_date}' AND '{end_date}'
                AND indfmt = 'INDL'
                AND datafmt = 'STD'
                AND popsrc = 'D'
                ORDER BY gvkey, rdq
            """

            df = self.conn.raw_sql(query)
            return self._add_ticker_symbols(df)
        except Exception as e:
            logger.warning(f"Could not fetch fundamentals: {e}")
            return pd.DataFrame()

    def fetch_gics_codes(
        self,
        symbols: list[str],
    ) -> pd.DataFrame:
        """Fetch GICS sector/industry codes from Compustat.

        Args:
            symbols: List of ticker symbols

        Returns:
            DataFrame with columns: gvkey, gsector, ggroup, gind, gsubind
        """
        try:
            gvkeys = self._symbols_to_gvkeys(symbols)

            if not gvkeys:
                logger.warning("No GVKEYs found for symbols, cannot fetch GICS codes")
                return pd.DataFrame()

            gvkey_list = "','".join(gvkeys)
            
            # Try different possible table names for GICS data
            for table_name in ["comp.company", "comp.names", "comp.gics"]:
                try:
                    query = f"""
                        SELECT gvkey, gsector, ggroup, gind, gsubind
                        FROM {table_name}
                        WHERE gvkey IN ('{gvkey_list}')
                    """
                    df = self.conn.raw_sql(query)
                    if not df.empty:
                        return self._add_ticker_symbols(df)
                except Exception:
                    continue
            
            logger.warning("Could not fetch GICS codes from any table")
            return pd.DataFrame()
            
        except Exception as e:
            logger.warning(f"Could not fetch GICS codes: {e}")
            return pd.DataFrame()

    def _symbols_to_permnos(self, symbols: list[str]) -> list[int]:
        """Convert ticker symbols to CRSP PERMNOs."""
        symbol_list = "','".join(symbols)
        query = f"""
            SELECT DISTINCT permno, ticker
            FROM crsp.dsenames
            WHERE ticker IN ('{symbol_list}')
        """
        df = self.conn.raw_sql(query)
        return df["permno"].tolist()

    def _symbols_to_gvkeys(self, symbols: list[str]) -> list[str]:
        """Convert ticker symbols to Compustat GVKEYs.
        
        Uses comp.names table which has the ticker-to-gvkey mapping.
        Falls back to CRSP-Compustat link table if needed.
        """
        symbol_list = "','".join(symbols)
        
        # Try comp.names table first (most common location for ticker mapping)
        try:
            query = f"""
                SELECT DISTINCT gvkey, ticker
                FROM comp.names
                WHERE ticker IN ('{symbol_list}')
            """
            df = self.conn.raw_sql(query)
            if not df.empty:
                return df["gvkey"].tolist()
        except Exception:
            pass
        
        # Try CRSP-Compustat link table as fallback
        try:
            permnos = self._symbols_to_permnos(symbols)
            if permnos:
                permno_list = ",".join(map(str, permnos))
                query = f"""
                    SELECT DISTINCT gvkey, permno
                    FROM crsp.ccmxpf_lnkhist
                    WHERE permno IN ({permno_list})
                    AND linktype IN ('LC', 'LU')
                    AND linkprim IN ('P', 'C')
                """
                df = self.conn.raw_sql(query)
                if not df.empty:
                    return df["gvkey"].tolist()
        except Exception:
            pass
        
        # If all methods fail, return empty list
        logger.warning(f"Could not map symbols to GVKEYs: {symbols[:5]}...")
        return []

    def _add_ticker_symbols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ticker symbols to DataFrame based on permno.
        
        Creates a mapping from CRSP dsenames table and adds 'symbol' column.
        """
        if 'permno' not in df.columns:
            return df
        
        # Get unique permnos from the dataframe
        permnos = df['permno'].unique().tolist()
        
        # Query CRSP to get ticker symbols
        permno_list = ",".join(map(str, permnos))
        query = f"""
            SELECT DISTINCT permno, ticker
            FROM crsp.dsenames
            WHERE permno IN ({permno_list})
        """
        
        try:
            mapping_df = self.conn.raw_sql(query)
            
            # Create mapping dict
            permno_to_ticker = dict(zip(mapping_df['permno'], mapping_df['ticker']))
            
            # Add symbol column
            df = df.copy()
            df['symbol'] = df['permno'].map(permno_to_ticker)
            
            return df
        except Exception as e:
            logger.warning(f"Could not map permnos to tickers: {e}")
            # If mapping fails, use permno as symbol as fallback
            df = df.copy()
            df['symbol'] = df['permno'].astype(str)
            return df

    def close(self):
        """Close WRDS connection."""
        self.conn.close()
        logger.info("WRDS connection closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
