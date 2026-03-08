# tests/test_training.py
import torch
import pytest
from src.models import DualEncoder, InfoNCELoss
from src.training.trainer import ContrastiveTrainer


def test_trainer_initialization():
    """Test trainer can be initialized."""
    model = DualEncoder(temporal_input_dim=13, tabular_continuous_dim=15)
    loss_fn = InfoNCELoss()

    trainer = ContrastiveTrainer(model, loss_fn, lr=1e-4)

    assert trainer.model is not None
    assert trainer.loss_fn is not None


def test_trainer_training_step():
    """Test training step produces loss."""
    model = DualEncoder(temporal_input_dim=13, tabular_continuous_dim=15)
    loss_fn = InfoNCELoss()
    trainer = ContrastiveTrainer(model, loss_fn, lr=1e-4)

    # Create fake batch
    batch = {
        "temporal": torch.randn(16, 60, 13),
        "tabular_cont": torch.randn(16, 15),
        "tabular_cat": torch.randint(0, 5, (16, 2)),
    }

    loss = trainer.train_step(batch)

    assert loss > 0
    assert not torch.isnan(torch.tensor(loss))
