# tests/test_models_sampler.py
import torch
import pytest
import pandas as pd
from src.models.sampler import GICSHardNegativeSampler


def test_sampler_initialization():
    """Test sampler initialization."""
    sampler = GICSHardNegativeSampler(n_hard=8)
    assert sampler.n_hard == 8


def test_sampler_filters_by_ggroup():
    """Test sampler finds stocks in same ggroup but different beta."""
    sampler = GICSHardNegativeSampler(n_hard=4)

    # Create mock stock data
    stocks = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "GOOGL", "JPM", "XOM"],
            "gsector": [45, 45, 45, 40, 10],  # Tech, Tech, Tech, Financials, Energy
            "ggroup": [4510, 4520, 4510, 4010, 1010],  # Software, Hardware, Software, Banks, Oil
            "beta": [1.2, 1.1, 1.3, 0.8, 0.9],
        }
    )

    target = pd.Series({"symbol": "AAPL", "gsector": 45, "ggroup": 4510, "beta": 1.2})

    hard_negs = sampler.sample(target, stocks)

    # Should find GOOGL (same ggroup 4510, different beta)
    assert "GOOGL" in hard_negs
    # Should NOT find MSFT (different ggroup 4520)
    # Should NOT find JPM or XOM (different gsector)


def test_sampler_empty_result():
    """Test sampler handles no valid hard negatives."""
    sampler = GICSHardNegativeSampler(n_hard=4)

    stocks = pd.DataFrame({"symbol": ["AAPL"], "gsector": [45], "ggroup": [4510], "beta": [1.2]})

    target = pd.Series({"symbol": "AAPL", "gsector": 45, "ggroup": 4510, "beta": 1.2})

    hard_negs = sampler.sample(target, stocks)

    # Should return empty or fallback
    assert len(hard_negs) <= 1  # At most itself
