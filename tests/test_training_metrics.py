"""Tests for training metrics module."""

import numpy as np
import pytest
import torch

from src.models import DualEncoder
from src.training.metrics import (
    compute_alignment_score,
    compute_hard_negative_similarity,
    compute_sector_silhouette,
)


def test_alignment_score():
    """Test alignment score computation."""
    # Perfect alignment: identical embeddings
    temporal_emb = torch.randn(16, 128)
    tabular_emb = temporal_emb.clone()

    alignment = compute_alignment_score(temporal_emb, tabular_emb)
    assert alignment > 0.99  # Should be near 1.0

    # Random alignment: should be near 0
    tabular_emb_random = torch.randn(16, 128)
    alignment_random = compute_alignment_score(temporal_emb, tabular_emb_random)
    assert -0.5 < alignment_random < 0.5  # Should be near 0


def test_hard_negative_similarity():
    """Test hard negative similarity metric."""
    batch_size = 8
    n_hard = 4

    # Create embeddings
    temporal_emb = torch.randn(batch_size + n_hard, 128)
    tabular_emb = torch.randn(batch_size + n_hard, 128)

    similarity = compute_hard_negative_similarity(temporal_emb, tabular_emb, batch_size, n_hard)

    # Should return scalar float
    assert isinstance(similarity, float)


def test_sector_silhouette_score():
    """Test sector silhouette computation."""
    model = DualEncoder(temporal_input_dim=13, tabular_continuous_dim=15)
    model.eval()

    # Create mock validation samples from 2 sectors
    val_samples = []
    for i in range(20):
        val_samples.append(
            {
                "temporal": torch.randn(60, 13),
                "tabular_cont": torch.randn(15),
                "tabular_cat": torch.tensor([45, 4510], dtype=torch.long),
                "gsector": 45,
            }
        )

    # Add some from different sector
    for i in range(10):
        val_samples.append(
            {
                "temporal": torch.randn(60, 13),
                "tabular_cont": torch.randn(15),
                "tabular_cat": torch.tensor([10, 1010], dtype=torch.long),
                "gsector": 10,
            }
        )

    silhouette = compute_sector_silhouette(model, val_samples)

    # Should return float in range [-1, 1]
    assert -1.0 <= silhouette <= 1.0
