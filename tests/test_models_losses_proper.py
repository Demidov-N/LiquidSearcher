"""Tests for proper InfoNCE loss computation."""

import torch
import pytest
from src.models.losses import InfoNCELoss


class TestInfoNCEProperComputation:
    """Test that InfoNCE computes correct loss values."""

    def test_symmetric_loss_computation(self):
        """Test that symmetric InfoNCE computes both directions."""
        loss_fn = InfoNCELoss(temperature=0.07)

        # Create embeddings: batch_size=4, dim=128
        temporal_emb = torch.randn(4, 128)
        tabular_emb = torch.randn(4, 128)

        # Compute InfoNCE loss
        loss = loss_fn(temporal_emb, tabular_emb)

        # Assert loss is scalar, positive, not NaN, not inf
        assert loss.dim() == 0, "Loss should be a scalar (0-dim tensor)"
        assert loss.item() > 0, "Loss should be positive"
        assert not torch.isnan(loss).item(), "Loss should not be NaN"
        assert not torch.isinf(loss).item(), "Loss should not be inf"

    def test_loss_with_hard_negatives(self):
        """Test InfoNCE loss with hard negatives."""
        loss_fn = InfoNCELoss(temperature=0.07)

        # Create positive embeddings: batch_size=4, dim=128
        temporal_emb = torch.randn(4, 128)
        tabular_emb = torch.randn(4, 128)

        # Create hard negatives: n_hard=2
        hard_negatives_temporal = torch.randn(2, 128)
        hard_negatives_tabular = torch.randn(2, 128)

        # Compute loss with hard negatives
        loss = loss_fn(
            temporal_emb,
            tabular_emb,
            hard_negatives_temporal=hard_negatives_temporal,
            hard_negatives_tabular=hard_negatives_tabular,
        )

        # Assert loss is valid scalar
        assert loss.dim() == 0, "Loss should be a scalar"
        assert loss.item() > 0, "Loss should be positive"
        assert not torch.isnan(loss).item(), "Loss should not be NaN"
        assert not torch.isinf(loss).item(), "Loss should not be inf"

    def test_loss_decreases_with_similarity(self):
        """Test that loss decreases when embeddings are more similar."""
        loss_fn = InfoNCELoss(temperature=0.07)

        # Create dissimilar embeddings: two random tensors
        temporal_dissimilar = torch.randn(4, 128)
        tabular_dissimilar = torch.randn(4, 128)

        # Create similar embeddings: one random tensor and its clone
        base_temporal = torch.randn(4, 128)
        temporal_similar = base_temporal.clone()
        tabular_similar = base_temporal.clone() + torch.randn(4, 128) * 0.01  # Small noise

        # Compute loss for both
        loss_dissimilar = loss_fn(temporal_dissimilar, tabular_dissimilar)
        loss_similar = loss_fn(temporal_similar, tabular_similar)

        # Assert loss for similar < loss for dissimilar
        assert loss_similar.item() < loss_dissimilar.item(), (
            f"Loss for similar embeddings ({loss_similar.item():.4f}) "
            f"should be less than loss for dissimilar embeddings ({loss_dissimilar.item():.4f})"
        )

    def test_gradient_flow(self):
        """Test that gradients flow properly through InfoNCE loss."""
        loss_fn = InfoNCELoss(temperature=0.07)

        # Create embeddings with requires_grad=True
        temporal_emb = torch.randn(4, 128, requires_grad=True)
        tabular_emb = torch.randn(4, 128, requires_grad=True)

        # Compute loss and backward
        loss = loss_fn(temporal_emb, tabular_emb)
        loss.backward()

        # Assert gradients exist and are not NaN
        assert temporal_emb.grad is not None, "Gradients should exist for temporal_emb"
        assert tabular_emb.grad is not None, "Gradients should exist for tabular_emb"
        assert not torch.isnan(temporal_emb.grad).any().item(), (
            "Temporal gradients should not contain NaN"
        )
        assert not torch.isnan(tabular_emb.grad).any().item(), (
            "Tabular gradients should not contain NaN"
        )
