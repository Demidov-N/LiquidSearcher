"""Tests for temporal encoder module."""

import torch
import pytest
from src.models.temporal_encoder import TemporalEncoder


def test_temporal_encoder_output():
    """Test temporal encoder produces 128-dim output."""
    batch_size = 32
    seq_len = 60
    input_dim = 13  # Technical features

    encoder = TemporalEncoder(
        input_dim=input_dim,
        tcn_hidden=64,
        tcn_layers=3,
        transformer_heads=4,
        transformer_layers=2,
        output_dim=128,
    )

    x = torch.randn(batch_size, seq_len, input_dim)
    out = encoder(x)

    assert out.shape == (batch_size, 128)


def test_temporal_encoder_preserves_batch():
    """Test different batch sizes work."""
    encoder = TemporalEncoder(input_dim=13)

    for batch_size in [1, 8, 32, 64]:
        x = torch.randn(batch_size, 60, 13)
        out = encoder(x)
        assert out.shape[0] == batch_size
        assert out.shape[1] == 128
