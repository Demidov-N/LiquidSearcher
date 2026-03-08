"""Test feature engineering base classes."""

import pandas as pd
import pytest

from src.features.base import FeatureGroup, FeatureRegistry, NormalizationMethod


def test_normalization_method_enum():
    """Test normalization method enum."""
    assert NormalizationMethod.Z_SCORE.value == "z_score"
    assert NormalizationMethod.LOG_Z_SCORE.value == "log_z_score"
    assert NormalizationMethod.RANK.value == "rank"
    assert NormalizationMethod.ROLLING_Z_SCORE.value == "rolling_z_score"
    assert NormalizationMethod.NONE.value == "none"


def test_feature_group_base_class():
    """Test FeatureGroup base class interface."""

    class TestFeatureGroup(FeatureGroup):
        def compute(self, df: pd.DataFrame) -> pd.DataFrame:
            return df.assign(test_feature=1.0)

        def get_feature_names(self) -> list[str]:
            return ["test_feature"]

    group = TestFeatureGroup()
    assert group.get_feature_names() == ["test_feature"]

    test_df = pd.DataFrame({"symbol": ["AAPL"], "date": pd.to_datetime(["2020-01-01"])})
    result = group.compute(test_df)
    assert "test_feature" in result.columns


def test_feature_registry():
    """Test feature registry for managing feature groups."""
    registry = FeatureRegistry()

    class TestGroup(FeatureGroup):
        def compute(self, df: pd.DataFrame) -> pd.DataFrame:
            return df

        def get_feature_names(self) -> list[str]:
            return ["test"]

    registry.register("test", TestGroup())
    assert "test" in registry.list_groups()
    assert registry.get("test") is not None
