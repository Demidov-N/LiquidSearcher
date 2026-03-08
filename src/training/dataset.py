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
        "mom_12_1m",  # Jegadeesh-Titman momentum
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
    ) -> None:
        """Initialize feature dataset.

        Args:
            feature_dir: Path to directory with parquet feature files
            date_range: (start_date, end_date) for valid samples
            symbols: List of symbols to include (None = all in directory)
            window_size: Days in temporal window (default 60)
        """
        self.feature_dir = Path(feature_dir)
        self.date_range = (pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]))
        self.window_size = window_size

        # Discover available symbols
        if symbols is None:
            self.symbols = [
                f.stem.replace("_features", "") for f in self.feature_dir.glob("*_features.parquet")
            ]
        else:
            self.symbols = symbols

        # Build index of valid (symbol, date) pairs
        self.samples = self._build_index()

        # Build categorical mappings from data
        self.categorical_mappings = self._build_categorical_mappings()
        self.categorical_dims = self._get_categorical_dims()

        print(f"Dataset initialized: {len(self.symbols)} symbols, {len(self.samples)} samples")
        print(f"Categorical dimensions: {self.categorical_dims}")

    def _build_index(self) -> list[tuple[str, pd.Timestamp]]:
        """Build index of valid (symbol, date) samples.

        A sample is valid if:
        - Symbol has parquet file
        - Date is within date_range
        - Has at least window_size days of history before date

        Returns:
            List of (symbol, date) tuples
        """
        samples = []

        for symbol in self.symbols:
            filepath = self.feature_dir / f"{symbol}_features.parquet"

            if not filepath.exists():
                continue

            # Load parquet and get dates (handle both column and index)
            df = pd.read_parquet(filepath)
            if "date" in df.columns:
                dates = pd.to_datetime(df["date"])
            elif "date" in df.index.names:
                dates = pd.to_datetime(df.index)
            else:
                # Try first column as date fallback
                dates = pd.to_datetime(df.iloc[:, 0])

            # Filter to date range with sufficient history
            min_date = self.date_range[0] + pd.Timedelta(days=self.window_size)
            valid_dates = dates[(dates >= min_date) & (dates <= self.date_range[1])]

            for date in valid_dates:
                samples.append((symbol, date))

        return samples

    def _build_categorical_mappings(self) -> dict[str, dict[str | int, int]]:
        """Build mappings from categorical values to indices.

        Scans all symbols to find unique values for each categorical column,
        then creates mappings to 0-indexed values suitable for embeddings.

        Returns:
            Dictionary mapping column name to {value: index} dict
        """
        unique_values: dict[str, set[str | int]] = {col: set() for col in self.TABULAR_CAT_COLS}

        for symbol in self.symbols:
            filepath = self.feature_dir / f"{symbol}_features.parquet"
            if not filepath.exists():
                continue

            # Load categorical columns
            df = pd.read_parquet(filepath, columns=self.TABULAR_CAT_COLS)
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

        # Load features for this symbol
        filepath = self.feature_dir / f"{symbol}_features.parquet"
        features = pd.read_parquet(filepath)

        # Handle date as column or index
        if "date" in features.columns:
            features["date"] = pd.to_datetime(features["date"])
            features = features.set_index("date")
        elif "date" in features.index.names:
            features.index = pd.to_datetime(features.index)

        # Extract temporal window: last window_size rows ending at date
        # Use position-based indexing to get exactly window_size rows
        date_idx_raw = features.index.get_loc(date)
        if not isinstance(date_idx_raw, int):
            # get_loc can return slice or ndarray for duplicate indices
            # but we expect unique dates, so this shouldn't happen
            raise TypeError(f"Expected int from get_loc, got {type(date_idx_raw)}")
        date_idx = date_idx_raw
        start_idx = max(0, date_idx - self.window_size + 1)
        temporal_window = features.iloc[start_idx : date_idx + 1][self.TEMPORAL_COLS]

        # Pad if necessary to ensure consistent window size
        if len(temporal_window) < self.window_size:
            n_pad = self.window_size - len(temporal_window)
            padding = pd.DataFrame(0, index=range(n_pad), columns=self.TEMPORAL_COLS)
            temporal_window = pd.concat([padding, temporal_window], ignore_index=True)

        # Extract tabular snapshot at date
        tabular_cont_raw = features.loc[date, self.TABULAR_CONT_COLS]
        tabular_cat_raw = features.loc[date, self.TABULAR_CAT_COLS]
        beta_raw = features.loc[date, "market_beta_60d"]

        if not isinstance(tabular_cont_raw, pd.Series):
            raise TypeError(f"Expected Series for tabular_cont, got {type(tabular_cont_raw)}")
        if not isinstance(tabular_cat_raw, pd.Series):
            raise TypeError(f"Expected Series for tabular_cat, got {type(tabular_cat_raw)}")
        if not isinstance(beta_raw, (int, float)):
            raise TypeError(f"Expected scalar for beta, got {type(beta_raw)}")

        tabular_cont = tabular_cont_raw
        tabular_cat = tabular_cat_raw
        beta = beta_raw

        # Map categorical values to indices for embeddings
        gsector_val = tabular_cat["gsector"]
        ggroup_val = tabular_cat["ggroup"]
        gsector_idx = self.categorical_mappings["gsector"].get(gsector_val, 0)
        ggroup_idx = self.categorical_mappings["ggroup"].get(ggroup_val, 0)

        # Convert to tensors
        sample = {
            "symbol": symbol,
            "date": date,
            "temporal": torch.tensor(temporal_window.values, dtype=torch.float32),
            "tabular_cont": torch.tensor(tabular_cont.values, dtype=torch.float32),
            "tabular_cat": torch.tensor([gsector_idx, ggroup_idx], dtype=torch.long),
            "beta": float(beta),
            "gsector": int(gsector_val),
            "ggroup": int(ggroup_val),
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
        filepath = self.feature_dir / f"{symbol}_features.parquet"

        if not filepath.exists():
            return None

        return pd.read_parquet(filepath)
