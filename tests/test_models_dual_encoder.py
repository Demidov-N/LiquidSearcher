# tests/test_models_dual_encoder.py
import torch
import pytest
from src.models.dual_encoder import DualEncoder


def test_dual_encoder_forward():
    """Test dual encoder produces correct outputs."""
    batch_size = 32
    seq_len = 60

    model = DualEncoder(
        temporal_input_dim=13,
        tabular_continuous_dim=15,
        tabular_categorical_dims=[11, 25],
        temporal_output_dim=128,
        tabular_output_dim=128,
    )

    # Temporal input
    x_temporal = torch.randn(batch_size, seq_len, 13)

    # Tabular input
    x_tabular_cont = torch.randn(batch_size, 15)
    x_tabular_cat = torch.randint(0, 5, (batch_size, 2))

    # Forward pass
    temporal_emb, tabular_emb = model(x_temporal, x_tabular_cont, x_tabular_cat)

    assert temporal_emb.shape == (batch_size, 128)
    assert tabular_emb.shape == (batch_size, 128)


def test_dual_encoder_joint_embedding():
    """Test joint embedding concatenation for inference."""
    model = DualEncoder(temporal_input_dim=13, tabular_continuous_dim=15)

    x_temp = torch.randn(16, 60, 13)
    x_tab_cont = torch.randn(16, 15)
    x_tab_cat = torch.randint(0, 5, (16, 2))

    # Inference mode: get joint embedding
    joint_emb = model.get_joint_embedding(x_temp, x_tab_cont, x_tab_cat)

    assert joint_emb.shape == (16, 256)  # 128 + 128


def test_dual_encoder_pairwise_similarity():
    """Test pairwise similarity diagnostic (inference utility, not training signal)."""
    model = DualEncoder(temporal_input_dim=13, tabular_continuous_dim=15)
    model.eval()

    x_temp = torch.randn(16, 60, 13)
    x_tab_cont = torch.randn(16, 15)
    x_tab_cat = torch.randint(0, 5, (16, 2))

    with torch.no_grad():
        similarity = model.compute_pairwise_similarity(x_temp, x_tab_cont, x_tab_cat)

    assert similarity.shape == (16,)  # One similarity per batch element
    assert torch.all(similarity >= -1) and torch.all(similarity <= 1)  # Cosine similarity range
