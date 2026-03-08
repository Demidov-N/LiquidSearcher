"""Tests for the InferencePreprocessor."""

import numpy as np
import pandas as pd
import pytest

from src.training.preprocessor import InferencePreprocessor


def test_preprocessor_initialization():
    """Test preprocessor can be initialized with feature engineer."""
    preprocessor = InferencePreprocessor()
    assert preprocessor is not None
    assert hasattr(preprocessor, "engineer")


def test_preprocessor_window_requirement():
    """Test preprocessor requires minimum window size."""
    # Create minimal data with only 30 days (insufficient)
    prices = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=30),
            "close": np.random.randn(30),
            "volume": np.random.randint(1000, 10000, 30),
        }
    ).set_index("date")

    preprocessor = InferencePreprocessor()

    # Should raise error for insufficient history
    with pytest.raises(ValueError, match="minimum 252 days"):
        preprocessor.compute_features(prices, min_history=252)
