"""Sector categorical features."""

import pandas as pd

from src.features.base import FeatureGroup


class SectorFeatures(FeatureGroup):
    """Sector feature group: sector categorical encoding.

    Integer encodes GICS sector and industry group for use
    with embedding layers. Maintains consistent mapping
    across calls.
    """

    def __init__(self) -> None:
        """Initialize sector features."""
        self.name = "sector"
        self._feature_names = [
            "gics_sector",
            "gics_industry_group",
            "gics_sector_str",
            "gics_industry_group_str",
        ]
        # Mapping dictionaries for consistent encoding
        self._sector_mapping: dict[str, int] = {}
        self._industry_mapping: dict[str, int] = {}
        self._next_sector_code = 0
        self._next_industry_code = 0

    def get_feature_names(self) -> list[str]:
        """Return list of feature names produced by this group."""
        return self._feature_names.copy()

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute sector categorical features.

        Args:
            df: Input dataframe with columns:
                - symbol: Stock identifier
                - date: Date column
                - gics_sector_str: GICS sector name (string)
                - gics_industry_group_str: GICS industry group name (string)

        Returns:
            DataFrame with sector features added:
                - gics_sector: Integer encoding of sector
                - gics_industry_group: Integer encoding of industry group
                - gics_sector_str: Original sector string (preserved)
                - gics_industry_group_str: Original industry string (preserved)
        """
        result = df.copy()

        # Encode sectors
        if "gics_sector_str" in result.columns:
            result["gics_sector"] = result["gics_sector_str"].apply(self._get_sector_code)
        else:
            result["gics_sector"] = -1

        # Encode industry groups
        if "gics_industry_group_str" in result.columns:
            result["gics_industry_group"] = result["gics_industry_group_str"].apply(
                self._get_industry_code
            )
        else:
            result["gics_industry_group"] = -1

        # Ensure string columns exist (for consistency)
        if "gics_sector_str" not in result.columns:
            result["gics_sector_str"] = ""
        if "gics_industry_group_str" not in result.columns:
            result["gics_industry_group_str"] = ""

        # No normalization for categorical features (used with embeddings)
        # Recommended embedding dims: 8 for sector, 16 for industry

        return result

    def _get_sector_code(self, sector: str) -> int:
        """Get integer code for a sector, creating if new."""
        if pd.isna(sector) or sector == "":
            return -1

        sector_str = str(sector)
        if sector_str not in self._sector_mapping:
            self._sector_mapping[sector_str] = self._next_sector_code
            self._next_sector_code += 1

        return self._sector_mapping[sector_str]

    def _get_industry_code(self, industry: str) -> int:
        """Get integer code for an industry group, creating if new."""
        if pd.isna(industry) or industry == "":
            return -1

        industry_str = str(industry)
        if industry_str not in self._industry_mapping:
            self._industry_mapping[industry_str] = self._next_industry_code
            self._next_industry_code += 1

        return self._industry_mapping[industry_str]
