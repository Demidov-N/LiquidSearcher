"""Training loop for contrastive learning."""

import torch
import torch.nn as nn
from torch.optim import AdamW


class ContrastiveTrainer:
    """Trainer for dual-encoder contrastive learning."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        """Initialize trainer.

        Args:
            model: DualEncoder model
            loss_fn: InfoNCE or RankSCL loss
            lr: Learning rate
            weight_decay: Weight decay for AdamW
            device: Device to train on
        """
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.device = device

        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_step(self, batch: dict) -> float:
        """Single training step.

        Args:
            batch: Dict with 'temporal', 'tabular_cont', 'tabular_cat'

        Returns:
            Loss value
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Move to device
        x_temp = batch["temporal"].to(self.device)
        x_tab_cont = batch["tabular_cont"].to(self.device)
        x_tab_cat = batch.get("tabular_cat")
        if x_tab_cat is not None:
            x_tab_cat = x_tab_cat.to(self.device)

        # Forward pass
        temporal_emb, tabular_emb = self.model(x_temp, x_tab_cont, x_tab_cat)

        # Compute loss
        loss = self.loss_fn(temporal_emb, tabular_emb)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        return loss.item()
