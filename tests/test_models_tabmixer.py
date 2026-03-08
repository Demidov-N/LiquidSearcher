# tests/test_models_tabmixer.py
"""Test suite for TabMixer model."""

import pytest
import torch

from src.models.tabmixer import TabMixer


def test_tabmixer_basic():
    """Test TabMixer with continuous features only."""
    batch_size = 32
    n_continuous = 15

    model = TabMixer(continuous_dim=n_continuous, categorical_dims=[], output_dim=128)

    x_cont = torch.randn(batch_size, n_continuous)
    out = model(x_cont)

    assert out.shape == (batch_size, 128)


def test_tabmixer_with_categorical():
    """Test TabMixer with categorical embeddings."""
    batch_size = 32
    n_continuous = 15

    model = TabMixer(
        continuous_dim=n_continuous,
        categorical_dims=[11, 25],  # GICS: sector, industry group
        embedding_dims=[8, 16],
        output_dim=128,
    )

    x_cont = torch.randn(batch_size, n_continuous)
    x_cat = torch.randint(0, 5, (batch_size, 2))  # 2 categorical features

    out = model(x_cont, x_cat)

    assert out.shape == (batch_size, 128)


def test_tabmixer_missing_values():
    """Test TabMixer handles missing values."""
    model = TabMixer(continuous_dim=15, handle_missing=True)

    x_cont = torch.randn(32, 15)
    x_cont[0, 3] = float("nan")  # Missing value

    out = model(x_cont)

    assert not torch.isnan(out).any()
    assert out.shape == (32, 128)
