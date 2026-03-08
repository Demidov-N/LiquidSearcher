"""PyTorch Lightning DataModule with temporal splits for financial data."""

from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class StockDataset(Dataset):
    """Dataset for stock features with random window sampling."""

    def __init__(
        self,
        feature_dir: str,
        date_range: tuple[str, str],
        symbols: Optional[List[str]] = None,
        window_size: int = 60,
    ):
        self.feature_dir = Path(feature_dir)
        self.window_size = window_size
        self.start_date = pd.Timestamp(date_range[0])
        self.end_date = pd.Timestamp(date_range[1])
        self.symbols = symbols or self._discover_symbols()
        self.samples = self._build_sample_index()

    def _discover_symbols(self) -> List[str]:
        """Discover available symbols from feature directory."""
        symbols = []
        for f in self.feature_dir.glob("*_features.parquet"):
            symbol = f.stem.replace("_features", "")
            symbols.append(symbol)
        return sorted(symbols)

    def _build_sample_index(self) -> List[Dict[str, Any]]:
        """Build index of valid (symbol, date) samples."""
        samples = []
        for symbol in self.symbols:
            file_path = self.feature_dir / f"{symbol}_features.parquet"
            if not file_path.exists():
                continue
            try:
                df = pd.read_parquet(file_path, columns=["date"])
                df["date"] = pd.to_datetime(df["date"])
                mask = (df["date"] >= self.start_date) & (df["date"] <= self.end_date)
                valid_dates = df.loc[mask, "date"].tolist()
                for date in valid_dates:
                    samples.append(
                        {
                            "symbol": symbol,
                            "date": date,
                            "file_path": str(file_path),
                        }
                    )
            except Exception as e:
                print(f"Warning: Could not load {symbol}: {e}")
                continue
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get sample by index."""
        sample = self.samples[idx]
        symbol = sample["symbol"]
        end_date = sample["date"]
        file_path = sample["file_path"]

        df = pd.read_parquet(file_path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        end_idx = df[df["date"] == end_date].index
        if len(end_idx) == 0:
            end_idx = df[df["date"] <= end_date].index[-1:]
        end_idx = end_idx[0]

        start_idx = max(0, end_idx - self.window_size + 1)
        temporal_df = df.iloc[start_idx : end_idx + 1]

        # Extract temporal features (skip date column at index 0)
        # Temporal features are at indices 1-14 (13 features)
        temporal_cols = temporal_df.values[:, 1:14].astype(np.float32)

        # Pad if necessary
        if len(temporal_df) < self.window_size:
            padding = self.window_size - len(temporal_df)
            temporal_data = np.zeros((self.window_size, 13), dtype=np.float32)
            temporal_data[padding:] = temporal_cols
        else:
            temporal_data = temporal_cols

        # Tabular features at end_date
        # Tabular continuous: indices 14-29 (15 features)
        # Tabular categorical: indices 29-31 (2 features)
        tabular_row = df.iloc[end_idx]
        tabular_cont = tabular_row.values[14:29].astype(np.float32)  # 15 continuous
        tabular_cat = tabular_row.values[29:31].astype(int)  # 2 categorical

        # Handle missing values
        tabular_cont = np.nan_to_num(tabular_cont, nan=0.0)
        tabular_cat = np.nan_to_num(tabular_cat, nan=0).astype(int)

        return {
            "symbol": symbol,
            "date": str(end_date),
            "temporal": torch.tensor(temporal_data, dtype=torch.float32),
            "tabular_cont": torch.tensor(tabular_cont, dtype=torch.float32),
            "tabular_cat": torch.tensor(tabular_cat, dtype=torch.long),
            "beta": float(tabular_cont[0] if len(tabular_cont) > 0 else 1.0),
            "gsector": int(tabular_cat[0]) if len(tabular_cat) > 0 else 0,
            "ggroup": int(tabular_cat[1]) if len(tabular_cat) > 1 else 0,
        }


class StockDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule with temporal train/val/test splits."""

    def __init__(
        self,
        feature_dir: str,
        train_start: str,
        train_end: str,
        val_start: str,
        val_end: str,
        test_start: Optional[str] = None,
        test_end: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        purge_days: int = 252,
        embargo_days: int = 63,
        window_size: int = 60,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.feature_dir = feature_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.window_size = window_size

        self.train_start = pd.Timestamp(train_start)
        self.train_end = pd.Timestamp(train_end)
        self.val_start = pd.Timestamp(val_start)
        self.val_end = pd.Timestamp(val_end)

        # Apply purge and embargo
        self.effective_train_end = self.train_end - timedelta(days=purge_days)
        self.effective_val_start = self.val_start + timedelta(days=embargo_days)

        self.symbols = symbols
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Setup datasets."""
        if stage == "fit" or stage is None:
            self.train_dataset = StockDataset(
                feature_dir=self.feature_dir,
                date_range=(str(self.train_start), str(self.effective_train_end)),
                symbols=self.symbols,
                window_size=self.window_size,
            )
            print(f"Train dataset: {len(self.train_dataset)} samples")

            self.val_dataset = StockDataset(
                feature_dir=self.feature_dir,
                date_range=(str(self.effective_val_start), str(self.val_end)),
                symbols=self.symbols,
                window_size=self.window_size,
            )
            print(f"Val dataset: {len(self.val_dataset)} samples")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
