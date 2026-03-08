# tests/test_models_losses.py
import torch
import pytest
from src.models.losses import InfoNCELoss


def test_infonce_loss_shape():
    """Test InfoNCE loss returns scalar."""
    batch_size = 32
    embedding_dim = 128

    loss_fn = InfoNCELoss(temperature=0.07)

    # Positive pairs: same stock, temporal vs tabular
    temporal_emb = torch.randn(batch_size, embedding_dim)
    tabular_emb = torch.randn(batch_size, embedding_dim)

    loss = loss_fn(temporal_emb, tabular_emb)

    assert loss.shape == ()
    assert loss.item() > 0


def test_infonce_loss_positive_pairs():
    """Test loss is low when embeddings are similar."""
    loss_fn = InfoNCELoss(temperature=0.07)

    # Identical embeddings (perfect match)
    emb = torch.randn(16, 128)
    loss = loss_fn(emb, emb.clone())

    # Loss should be near minimum for identical pairs
    assert loss.item() < 1.0


def test_infonce_loss_negative_pairs():
    """Test loss is high when embeddings are different."""
    loss_fn = InfoNCELoss(temperature=0.07)

    # Random unrelated embeddings
    temporal = torch.randn(16, 128)
    tabular = torch.randn(16, 128)

    loss = loss_fn(temporal, tabular)

    # Loss should be higher for random pairs
    assert loss.item() > 0.5
