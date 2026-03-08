"""PyTorch Dataset for pre-computed stock features."""

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


class FeatureDataset(Dataset):
    """Dataset for loading pre-computed stock features.

    Loads features from parquet files and samples random windows
    for training. Each sample contains both temporal sequences
    and tabular snapshots for dual-encoder training.

    Attributes:
        feature_dir: Directory containing parquet feature files
        date_range: (start_date, end_date) tuple for valid samples
        symbols: List of stock symbols to include
        window_size: Number of days in temporal window (default 60)
    """

    # Column definitions for feature groups (matching actual computed features)
    TEMPORAL_COLS = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "return",
        "ma_ratio_5",
        "ma_ratio_10",
        "ma_ratio_15",
        "ma_ratio_20",
        "z_close_5d",
        "z_close_10d",
        "z_close_20d",
    ]

    TABULAR_CONT_COLS = [
        "market_beta_60d",
        "downside_beta_60d",
        "realized_vol_20d",
        "realized_vol_60d",
        "idiosyncratic_vol",
        "vol_of_vol",
        "mom_1m",
        "mom_3m",
        "mom_6m",
        # "mom_12_1m",  # Temporarily excluded - requires 273 days, 2023 only has 252
        "macd",
        "log_mktcap",
        "pe_ratio",
        "pb_ratio",
        "roe",
    ]

    TABULAR_CAT_COLS = ["gsector", "ggroup"]

    def __init__(
        self,
        feature_dir: str,
        date_range: tuple[str, str],
        symbols: list[str] | None = None,
        window_size: int = 60,
        feature_file: str = "all_features.parquet",
    ) -> None:
        """Initialize feature dataset.

        Args:
            feature_dir: Path to directory with parquet feature files
            date_range: (start_date, end_date) for valid samples
            symbols: List of symbols to include (None = all in parquet file)
            window_size: Days in temporal window (default 60)
            feature_file: Name of the unified parquet file (default: "all_features.parquet")
        """
        self.feature_dir = Path(feature_dir)
        self.feature_file = feature_file
        self.date_range = (pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]))
        self.window_size = window_size

        # Load unified parquet file
        self._load_unified_data()

        # Set symbols (filter to requested subset if provided)
        if symbols is None:
            self.symbols = sorted(self._all_symbols)
        else:
            # Validate requested symbols exist in data
            available = set(self._all_symbols)
            requested = set(symbols)
            valid = requested & available
            if len(valid) != len(requested):
                missing = requested - available
                print(f"Warning: {len(missing)} symbols not found in data: {list(missing)[:5]}...")
            self.symbols = sorted(valid)

        # Build index of valid (symbol, date) pairs
        self.samples = self._build_index()

        # Build categorical mappings from data
        self.categorical_mappings = self._build_categorical_mappings()
        self.categorical_dims = self._get_categorical_dims()

        print(f"Dataset initialized: {len(self.symbols)} symbols, {len(self.samples)} samples")
        print(f"Categorical dimensions: {self.categorical_dims}")

    def _load_unified_data(self) -> None:
        """Load the unified parquet file containing all symbols.

        Loads data into memory for efficient filtering by symbol and date.
        Sets _all_symbols and _unified_df attributes.
        """
        filepath = self.feature_dir / self.feature_file

        if not filepath.exists():
            raise FileNotFoundError(f"Unified feature file not found: {filepath}")

        # Load the entire parquet file
        df = pd.read_parquet(filepath)

        # Ensure required columns exist
        if "symbol" not in df.columns:
            raise ValueError(
                f"Unified parquet must have 'symbol' column. Found: {list(df.columns)}"
            )
        if "date" not in df.columns:
            raise ValueError(f"Unified parquet must have 'date' column. Found: {list(df.columns)}")

        # Convert date column to datetime
        df["date"] = pd.to_datetime(df["date"])

        # Store unified dataframe and extract all symbols
        self._unified_df = df
        self._all_symbols = sorted(df["symbol"].unique().tolist())

    def _build_index(self) -> list[tuple[str, pd.Timestamp]]:
        """Build index of valid (symbol, date) samples.

        A sample is valid if:
        - Symbol exists in unified parquet
        - Date is within date_range
        - Has at least window_size days of history before date
        - Has no NaN values in required feature columns

        Returns:
            List of (symbol, date) tuples
        """
        samples = []
        df = self._unified_df

        # Filter to requested symbols and date range with sufficient history
        min_date = self.date_range[0] + pd.Timedelta(days=self.window_size)
        mask = (
            (df["symbol"].isin(self.symbols))
            & (df["date"] >= min_date)
            & (df["date"] <= self.date_range[1])
        )

        # Filter out rows with NaN in required columns
        required_cols = self.TEMPORAL_COLS + self.TABULAR_CONT_COLS + self.TABULAR_CAT_COLS
        # Only check columns that exist in the dataframe
        available_required = [col for col in required_cols if col in df.columns]

        # Debug: show which columns have NaN
        in_range_df = df[mask]
        if len(in_range_df) > 0:
            nan_cols = []
            for col in available_required:
                nan_count = in_range_df[col].isna().sum()
                if nan_count > 0:
                    nan_cols.append(f"{col}({nan_count})")
            if nan_cols:
                print(
                    f"  Columns with NaN: {', '.join(nan_cols[:5])}{'...' if len(nan_cols) > 5 else ''}"
                )

        valid_mask = mask & df[available_required].notna().all(axis=1)

        valid_rows = df[valid_mask][["symbol", "date"]].copy()

        # Report filtering stats
        total_in_range = mask.sum()
        valid_count = len(valid_rows)
        filtered_count = total_in_range - valid_count
        if filtered_count > 0:
            print(
                f"  Filtered {filtered_count} samples with NaN values ({filtered_count / total_in_range * 100:.1f}%)"
            )

        for _, row in valid_rows.iterrows():
            samples.append((str(row["symbol"]), pd.Timestamp(row["date"])))

        return samples

    def _build_categorical_mappings(self) -> dict[str, dict[str | int, int]]:
        """Build mappings from categorical values to indices.

        Scans unified dataframe to find unique values for each categorical column,
        then creates mappings to 0-indexed values suitable for embeddings.

        Returns:
            Dictionary mapping column name to {value: index} dict
        """
        unique_values: dict[str, set[str | int]] = {col: set() for col in self.TABULAR_CAT_COLS}

        # Filter to requested symbols
        df = self._unified_df[self._unified_df["symbol"].isin(self.symbols)]

        # Extract unique values from categorical columns
        for col in self.TABULAR_CAT_COLS:
            unique_values[col].update(df[col].unique())

        # Create mappings (sorted for consistency)
        mappings: dict[str, dict[str | int, int]] = {}
        for col in self.TABULAR_CAT_COLS:
            sorted_values = sorted(unique_values[col])
            mappings[col] = {val: idx for idx, val in enumerate(sorted_values)}

        return mappings

    def _get_categorical_dims(self) -> list[int]:
        """Get embedding dimensions for each categorical column.

        Returns:
            List of cardinalities for each categorical column
        """
        return [len(self.categorical_mappings[col]) for col in self.TABULAR_CAT_COLS]

    def __len__(self) -> int:
        """Return number of valid samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """Get sample by index.

        Returns:
            Dictionary with:
                - symbol: Stock symbol (str)
                - date: Sample date (datetime)
                - temporal: Tensor (window_size, 13)
                - tabular_cont: Tensor (15,)
                - tabular_cat: Tensor (2,)
                - beta: Current beta value (float)
                - gsector: Sector code (int)
                - ggroup: Group code (int)
        """
        symbol, date = self.samples[idx]

        # Filter unified dataframe to this symbol
        symbol_df = self._unified_df[self._unified_df["symbol"] == symbol].copy()
        symbol_df = symbol_df.sort_values("date").reset_index(drop=True)

        # Find the index of the target date
        date_mask = symbol_df["date"] == date
        if not date_mask.any():
            raise ValueError(f"Date {date} not found for symbol {symbol}")

        date_idx = symbol_df[date_mask].index[0]

        # Extract temporal window: last window_size rows ending at date
        start_idx = max(0, date_idx - self.window_size + 1)
        temporal_window = symbol_df.loc[start_idx:date_idx, self.TEMPORAL_COLS]

        # Pad if necessary to ensure consistent window size
        if len(temporal_window) < self.window_size:
            n_pad = self.window_size - len(temporal_window)
            padding = pd.DataFrame(0, index=range(n_pad), columns=self.TEMPORAL_COLS)
            temporal_window = pd.concat([padding, temporal_window], ignore_index=True)

        # Extract tabular snapshot at date
        date_row = symbol_df[symbol_df["date"] == date].iloc[0]
        tabular_cont = date_row[self.TABULAR_CONT_COLS].astype(float)
        tabular_cat = date_row[self.TABULAR_CAT_COLS]
        beta = float(date_row["market_beta_60d"])

        gsector_val = int(tabular_cat["gsector"])
        ggroup_val = int(tabular_cat["ggroup"])
        gsector_idx = self.categorical_mappings["gsector"].get(gsector_val, 0)
        ggroup_idx = self.categorical_mappings["ggroup"].get(ggroup_val, 0)

        # Convert to tensors
        # Ensure temporal data is float64 (not pandas nullable Float64 which converts to object)
        temporal_values = temporal_window.values
        if temporal_values.dtype == object:
            temporal_values = temporal_window.astype(float).values

        sample = {
            "symbol": symbol,
            "date": date,
            "temporal": torch.tensor(temporal_values, dtype=torch.float32),
            "tabular_cont": torch.tensor(tabular_cont.values, dtype=torch.float32),
            "tabular_cat": torch.tensor([gsector_idx, ggroup_idx], dtype=torch.long),
            "beta": beta,
            "gsector": gsector_val,
            "ggroup": ggroup_val,
        }

        return sample

    def get_symbol_features(self, symbol: str) -> pd.DataFrame | None:
        """Load all features for a specific symbol.

        Useful for hard negative sampling to fetch
        features of negative examples.

        Args:
            symbol: Stock symbol to load

        Returns:
            DataFrame with all features or None if not found
        """
        # Check if symbol exists in unified data
        if symbol not in self._all_symbols:
            return None

        # Filter unified dataframe to this symbol
        symbol_df = self._unified_df[self._unified_df["symbol"] == symbol].copy()
        return symbol_df
