"""PyTorch Lightning Module for dual-encoder contrastive learning."""

from typing import Any, Dict, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from info_nce import InfoNCE

from src.models.dual_encoder import DualEncoder


class DualEncoderModule(pl.LightningModule):
    """LightningModule for dual-encoder contrastive training."""

    def __init__(
        self,
        temporal_input_dim: int = 13,
        tabular_continuous_dim: int = 15,
        tabular_categorical_dims: Optional[List[int]] = None,
        tabular_embedding_dims: Optional[List[int]] = None,
        embedding_dim: int = 128,
        temperature: float = 0.07,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_epochs: int = 5,
        max_epochs: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Model
        self.model = DualEncoder(
            temporal_input_dim=temporal_input_dim,
            tabular_continuous_dim=tabular_continuous_dim,
            tabular_categorical_dims=tabular_categorical_dims or [11, 25],
            tabular_embedding_dims=tabular_embedding_dims or [8, 16],
            embedding_dim=embedding_dim,
            temperature=temperature,
        )

        # Loss function
        self.loss_fn = InfoNCE(temperature=temperature)

        # Training hyperparameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

    def forward(self, batch: Dict[str, torch.Tensor], mode: str = "train"):
        """Forward pass."""
        return self.model(
            batch["temporal"],
            batch["tabular_cont"],
            batch["tabular_cat"],
            mode=mode,
        )

    def encode(self, batch: Dict[str, torch.Tensor]):
        """Get embeddings without computing loss."""
        temporal_emb = self.model.temporal_encoder(batch["temporal"])
        tabular_emb = self.model.tabular_encoder(
            batch["tabular_cont"],
            batch["tabular_cat"],
        )
        return temporal_emb, tabular_emb

    def _compute_loss(self, temporal_emb: torch.Tensor, tabular_emb: torch.Tensor):
        """Compute InfoNCE loss and metrics."""
        loss = self.loss_fn(temporal_emb, tabular_emb)

        with torch.no_grad():
            sim_matrix = torch.matmul(temporal_emb, tabular_emb.t())
            pos_sim = torch.diag(sim_matrix).mean().item()
            mask = ~torch.eye(len(sim_matrix), dtype=torch.bool, device=sim_matrix.device)
            neg_sim = sim_matrix[mask].mean().item()

        return loss, {"alignment": pos_sim, "neg_sim": neg_sim}

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Training step."""
        temporal_emb, tabular_emb = self.encode(batch)
        loss, metrics = self._compute_loss(temporal_emb, tabular_emb)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/alignment", metrics["alignment"], on_step=False, on_epoch=True)
        self.log("train/neg_sim", metrics["neg_sim"], on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Validation step."""
        temporal_emb, tabular_emb = self.encode(batch)
        loss, metrics = self._compute_loss(temporal_emb, tabular_emb)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/alignment", metrics["alignment"], on_step=False, on_epoch=True)
        self.log("val/neg_sim", metrics["neg_sim"], on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.warmup_epochs,
                T_mult=2,
            ),
            "interval": "epoch",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    def get_joint_embeddings(self, batch: Dict[str, torch.Tensor]):
        """Get joint embeddings for inference."""
        self.eval()
        with torch.no_grad():
            temporal_emb, tabular_emb = self.encode(batch)
            joint_emb = torch.cat([temporal_emb, tabular_emb], dim=-1)
        return joint_emb
