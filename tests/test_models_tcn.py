"""Tests for TCN (Temporal Convolutional Network) module."""

import torch

from src.models.tcn import TemporalConvNet


def test_tcn_output_shape():
    """Test TCN produces correct output shape."""
    batch_size = 32
    seq_len = 60
    input_dim = 13
    hidden_dim = 64
    output_dim = 64

    tcn = TemporalConvNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=3,
    )

    x = torch.randn(batch_size, seq_len, input_dim)
    out = tcn(x)

    assert out.shape == (batch_size, seq_len, output_dim)


def test_tcn_causal_property():
    """Test TCN maintains causality (no future info leakage)."""
    tcn = TemporalConvNet(input_dim=5, hidden_dim=16, output_dim=16, num_layers=2)
    tcn.eval()  # Set to eval mode to disable dropout for deterministic testing

    # Create input where last timestep is different
    x1 = torch.randn(1, 10, 5)
    x2 = x1.clone()
    x2[:, -1, :] = torch.randn(5)  # Change last timestep

    with torch.no_grad():
        out1 = tcn(x1)
        out2 = tcn(x2)

    # First 9 timesteps should be identical (causal)
    assert torch.allclose(out1[:, :9, :], out2[:, :9, :], atol=1e-5)


def test_tcn_single_layer():
    """Test TCN with single layer."""
    tcn = TemporalConvNet(input_dim=10, hidden_dim=32, output_dim=32, num_layers=1)

    x = torch.randn(8, 30, 10)
    out = tcn(x)

    assert out.shape == (8, 30, 32)


def test_tcn_batch_independence():
    """Test that batch items are processed independently."""
    tcn = TemporalConvNet(input_dim=8, hidden_dim=16, output_dim=16, num_layers=2)
    tcn.eval()  # Set to eval mode for deterministic behavior

    x1 = torch.randn(1, 20, 8)
    x2 = torch.randn(1, 20, 8)
    x_batch = torch.cat([x1, x2], dim=0)

    with torch.no_grad():
        out1 = tcn(x1)
        out2 = tcn(x2)
        out_batch = tcn(x_batch)

    assert torch.allclose(out1, out_batch[0:1], atol=1e-5)
    assert torch.allclose(out2, out_batch[1:2], atol=1e-5)


def test_tcn_different_input_output_dims():
    """Test TCN when input and output dimensions differ."""
    tcn = TemporalConvNet(input_dim=10, hidden_dim=32, output_dim=64, num_layers=3)

    x = torch.randn(4, 25, 10)
    out = tcn(x)

    assert out.shape == (4, 25, 64)
