"""Data caching with parquet format for fast persistence and notebook compatibility."""

import hashlib
import logging
from pathlib import Path
from typing import Literal

import pandas as pd
import polars as pl

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


ParquetCompression = Literal["snappy", "zstd", "lz4", "gzip"] | None


class CacheManager:
    """Manage local data cache in parquet format (notebook-friendly)."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        compression: ParquetCompression | str = "snappy",
    ):
        """Initialize cache manager.

        Args:
            cache_dir: Directory for cache files. Uses settings default if None.
            compression: Compression algorithm (snappy, zstd, lz4, gzip, none)
        """
        if cache_dir is None:
            settings = get_settings()
            cache_dir = settings.cache_dir

        self.cache_dir = Path(cache_dir)
        # Convert "none" string to None for pandas compatibility
        self.compression: ParquetCompression = (
            None if compression == "none" else compression  # type: ignore[assignment]
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, key: str) -> Path:
        """Generate safe file path for cache key."""
        safe_key = key.replace("/", "_").replace(":", "_").replace(" ", "_").replace("\\", "_")
        if len(safe_key) > 100:
            safe_key = safe_key[:50] + "_" + hashlib.md5(key.encode()).hexdigest()[:16]
        return self.cache_dir / f"{safe_key}.parquet"

    def get(self, key: str) -> pd.DataFrame | None:
        """Load DataFrame from cache if exists.

        Args:
            key: Cache key

        Returns:
            Cached DataFrame or None if not found
        """
        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            return None

        try:
            # Always return pandas DataFrame (primary format per AGENTS.md)
            return pd.read_parquet(cache_path)
        except Exception as e:
            logger.warning(f"Failed to read cache file {cache_path}: {e}")
            return None

    def set(
        self,
        key: str,
        df: pd.DataFrame | pl.DataFrame,
    ) -> None:
        """Save DataFrame to cache.

        Args:
            key: Cache key
            df: DataFrame to cache (pandas or polars)
        """
        cache_path = self._get_cache_path(key)

        try:
            if isinstance(df, pd.DataFrame):
                df.to_parquet(cache_path, compression=self.compression)
            elif isinstance(df, pl.DataFrame):
                # Convert to pandas for consistent format
                df.to_pandas().to_parquet(cache_path, compression=self.compression)
            else:
                raise TypeError(f"Unsupported DataFrame type: {type(df)}")
            logger.info(f"Cached data to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to write cache file {cache_path}: {e}")

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        return self._get_cache_path(key).exists()

    def delete(self, key: str) -> None:
        """Delete cached item."""
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            cache_path.unlink()

    def clear(self) -> None:
        """Clear all cached data."""
        for f in self.cache_dir.glob("*.parquet"):
            f.unlink(missing_ok=True)

    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern.

        Args:
            pattern: Glob pattern for matching keys

        Returns:
            Number of keys invalidated
        """
        count = 0
        # Validate pattern to prevent path traversal
        if ".." in pattern or "/" in pattern or "\\" in pattern or "\x00" in pattern:
            logger.warning(f"Invalid pattern contains path traversal characters: {pattern}")
            return 0
        for path in self.cache_dir.glob(f"{pattern}*.parquet"):
            path.unlink()
            count += 1
        return count
