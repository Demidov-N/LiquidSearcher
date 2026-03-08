"""Application settings and constants from MVP.md."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    """Data configuration."""

    train_start: str = "2010-01-01"
    train_end: str = "2022-12-31"
    val_start: str = "2023-01-01"
    val_end: str = "2023-12-31"
    test_start: str = "2024-01-01"
    test_end: str = "2024-12-31"

    universe: list[str] = None

    def __post_init__(self):
        if self.universe is None:
            object.__setattr__(self, "universe", ["Russell 2000", "S&P 400"])


@dataclass(frozen=True)
class FeatureConfig:
    """Feature engineering configuration."""

    beta_window: int = 60
    vol_windows: list[int] = None
    momentum_windows: list[int] = None
    rolling_z_window: int = 252

    winsorize_lower: float = 0.01
    winsorize_upper: float = 0.99

    def __post_init__(self):
        if self.vol_windows is None:
            object.__setattr__(self, "vol_windows", [20, 60])
        if self.momentum_windows is None:
            object.__setattr__(self, "momentum_windows", [21, 63, 126])


@dataclass(frozen=True)
class ModelConfig:
    """Model architecture configuration."""

    temporal_dim: int = 128
    fundamental_dim: int = 128
    joint_dim: int = 256
    temperature: float = 0.07
    batch_size: int = 256


@dataclass(frozen=True)
class LiquidityConfig:
    """Liquidity filter thresholds."""

    amihud_percentile: float = 0.30
    illiq_zscore_threshold: float = 2.0
    spread_bps_threshold: float = 50.0
    spread_vol_threshold: float = 20.0
    zero_return_threshold: float = 0.05


@dataclass(frozen=True)
class Settings:
    """Application settings."""

    data: DataConfig = None
    features: FeatureConfig = None
    model: ModelConfig = None
    liquidity: LiquidityConfig = None

    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    cache_dir: Path = Path("data/cache")

    def __post_init__(self):
        if self.data is None:
            object.__setattr__(self, "data", DataConfig())
        if self.features is None:
            object.__setattr__(self, "features", FeatureConfig())
        if self.model is None:
            object.__setattr__(self, "model", ModelConfig())
        if self.liquidity is None:
            object.__setattr__(self, "liquidity", LiquidityConfig())


_settings: Settings | None = None


def get_settings() -> Settings:
    """Get application settings (singleton)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
