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

    # Column definitions for feature groups
    TEMPORAL_COLS = [
        "z_score_20d",
        "z_score_60d",
        "ma_ratio_20_60",
        "volume_trend",
        "volatility_20d",
        "momentum_20d",
        "momentum_60d",
        "momentum_252d",
        "momentum_ratio",
        "reversal_5d",
        "parkinson_vol",
        "garch_vol",
        "price_trend",
    ]

    TABULAR_CONT_COLS = [
        "market_beta_60d",
        "downside_beta_60d",
        "realized_vol_20d",
        "realized_vol_60d",
        "garch_vol",
        "parkinson_vol",
        "momentum_20d",
        "momentum_60d",
        "momentum_252d",
        "momentum_ratio",
        "reversal_5d",
        "pe_ratio",
        "pb_ratio",
        "roe",
        "market_cap_log",
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

        print(f"Dataset initialized: {len(self.symbols)} symbols, {len(self.samples)} samples")

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

            # Load just the index to check dates (fast)
            df = pd.read_parquet(filepath, columns=[])
            dates = df.index

            # Filter to date range with sufficient history
            min_date = self.date_range[0] + pd.Timedelta(days=self.window_size)
            valid_dates = dates[(dates >= min_date) & (dates <= self.date_range[1])]

            for date in valid_dates:
                samples.append((symbol, date))

        return samples

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

        # Convert to tensors
        sample = {
            "symbol": symbol,
            "date": date,
            "temporal": torch.tensor(temporal_window.values, dtype=torch.float32),
            "tabular_cont": torch.tensor(tabular_cont.values, dtype=torch.float32),
            "tabular_cat": torch.tensor(tabular_cat.values, dtype=torch.long),
            "beta": float(beta),
            "gsector": int(tabular_cat["gsector"]),
            "ggroup": int(tabular_cat["ggroup"]),
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
