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
    """Test sampler finds stocks in same ggroup with sufficiently different beta."""
    sampler = GICSHardNegativeSampler(n_hard=4, beta_threshold=0.3)

    # Create mock stock data
    # GOOGL has beta=2.0 so |2.0 - 1.2| = 0.8 > 0.3 → qualifies as group-level hard negative
    # MSFT is in different ggroup (4520) → should NOT appear as a group-level negative
    stocks = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "GOOGL", "JPM", "XOM"],
            "gsector": [45, 45, 45, 40, 10],  # Tech, Tech, Tech, Financials, Energy
            "ggroup": [4510, 4520, 4510, 4010, 1010],  # Software, Hardware, Software, Banks, Oil
            "beta": [1.2, 1.1, 2.0, 0.8, 0.9],
        }
    )

    target = pd.Series({"symbol": "AAPL", "gsector": 45, "ggroup": 4510, "beta": 1.2})

    hard_negs = sampler.sample(target, stocks)

    # GOOGL: same ggroup 4510, beta diff = 0.8 > threshold → must be in results
    assert "GOOGL" in hard_negs
    # MSFT: different ggroup 4520 → must not appear as a group-level negative
    # (it may appear as a sector-level fallback, but ggroup-level negatives come first)
    # JPM and XOM are different gsector entirely → never sampled
    assert "JPM" not in hard_negs
    assert "XOM" not in hard_negs


def test_sampler_empty_result():
    """Test sampler handles no valid hard negatives."""
    sampler = GICSHardNegativeSampler(n_hard=4)

    stocks = pd.DataFrame({"symbol": ["AAPL"], "gsector": [45], "ggroup": [4510], "beta": [1.2]})

    target = pd.Series({"symbol": "AAPL", "gsector": 45, "ggroup": 4510, "beta": 1.2})

    hard_negs = sampler.sample(target, stocks)

    # Should return empty or fallback
    assert len(hard_negs) <= 1  # At most itself
