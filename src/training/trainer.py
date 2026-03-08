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
        max_grad_norm: float = 1.0,
        device: str | None = None,
    ) -> None:
        """Initialize trainer.

        Args:
            model: DualEncoder model
            loss_fn: InfoNCE or RankSCL loss
            lr: Learning rate
            weight_decay: Weight decay for AdamW
            max_grad_norm: Max gradient norm for clipping (transformer stability)
            device: Device to train on; auto-detected if None
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.device = device
        self.max_grad_norm = max_grad_norm

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

        # Backward pass with gradient clipping (critical for transformer stability)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return loss.item()

    def eval_step(self, batch: dict) -> float:
        """Single evaluation step (no gradient update).

        Args:
            batch: Dict with 'temporal', 'tabular_cont', 'tabular_cat'

        Returns:
            Loss value
        """
        self.model.eval()
        with torch.no_grad():
            x_temp = batch["temporal"].to(self.device)
            x_tab_cont = batch["tabular_cont"].to(self.device)
            x_tab_cat = batch.get("tabular_cat")
            if x_tab_cat is not None:
                x_tab_cat = x_tab_cat.to(self.device)

            temporal_emb, tabular_emb = self.model(x_temp, x_tab_cont, x_tab_cat)
            loss = self.loss_fn(temporal_emb, tabular_emb)

        return loss.item()
