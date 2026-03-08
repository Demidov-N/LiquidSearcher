# tests/test_models_integration.py
"""Integration tests for model components."""

import torch
import pandas as pd
import pytest

from src.models import (
    TemporalEncoder,
    TabMixer,
    DualEncoder,
    InfoNCELoss,
    GICSHardNegativeSampler,
)


def test_all_models_importable():
    """Test all model components can be imported."""
    assert TemporalEncoder is not None
    assert TabMixer is not None
    assert DualEncoder is not None
    assert InfoNCELoss is not None
    assert GICSHardNegativeSampler is not None


def test_end_to_end_forward():
    """Test full forward pass through dual encoder."""
    batch_size = 16

    # Initialize model
    model = DualEncoder(
        temporal_input_dim=13,
        tabular_continuous_dim=15,
        tabular_categorical_dims=[11, 25],
        tabular_embedding_dims=[8, 16],
    )

    # Create sample data
    x_temporal = torch.randn(batch_size, 60, 13)
    x_tabular_cont = torch.randn(batch_size, 15)
    x_tabular_cat = torch.randint(0, 5, (batch_size, 2))

    # Forward pass
    temporal_emb, tabular_emb = model(x_temporal, x_tabular_cont, x_tabular_cat)

    # Check shapes
    assert temporal_emb.shape == (batch_size, 128)
    assert tabular_emb.shape == (batch_size, 128)

    # Compute pairwise similarity (inference diagnostic)
    similarity = model.compute_pairwise_similarity(x_temporal, x_tabular_cont, x_tabular_cat)
    assert similarity.shape == (batch_size,)

    # Get joint embedding
    joint_emb = model.get_joint_embedding(x_temporal, x_tabular_cont, x_tabular_cat)
    assert joint_emb.shape == (batch_size, 256)


def test_loss_computation():
    """Test InfoNCE loss with dual encoder outputs."""
    model = DualEncoder(temporal_input_dim=13, tabular_continuous_dim=15)
    loss_fn = InfoNCELoss(temperature=0.07)

    x_temp = torch.randn(32, 60, 13)
    x_tab_cont = torch.randn(32, 15)
    x_tab_cat = torch.randint(0, 5, (32, 2))

    temporal_emb, tabular_emb = model(x_temp, x_tab_cont, x_tab_cat)
    loss = loss_fn(temporal_emb, tabular_emb)

    assert loss.item() > 0
    assert not torch.isnan(loss)
