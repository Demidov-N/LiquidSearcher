"""Loss functions for dual encoder training."""

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class ContrastiveLoss(nn.Module):
    """Pairwise contrastive loss for learned similarity."""

    def __init__(self, margin: float = 1.0) -> None:
        """Initialize contrastive loss.

        Args:
            margin: Margin for triplet loss
        """
        super().__init__()
        self.margin = margin

    def forward(
        self,
        embeddings_a: torch.Tensor,
        embeddings_b: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute contrastive loss between two embedding sets.

        Args:
            embeddings_a: First set of embeddings (batch, dim)
            embeddings_b: Second set of embeddings (batch, dim)
            labels: Binary labels indicating positive/negative pairs (batch,)

        Returns:
            Scalar loss value
        """
        distances = F.pairwise_distance(embeddings_a, embeddings_b)
        loss = labels * distances.pow(2) + (1 - labels) * F.relu(self.margin - distances).pow(2)
        return loss.mean()


class InfoNCELoss(nn.Module):
    """InfoNCE loss for contrastive learning."""

    def __init__(self, temperature: float = 0.07) -> None:
        """Initialize InfoNCE loss.

        Args:
            temperature: Temperature parameter for softmax
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings_pos: torch.Tensor,
        embeddings_neg: torch.Tensor,
    ) -> torch.Tensor:
        """Compute InfoNCE loss.

        Args:
            embeddings_pos: Positive pair embeddings (batch, dim)
            embeddings_neg: Negative sample embeddings (batch, num_neg, dim)

        Returns:
            Scalar loss value
        """
        batch_size = embeddings_pos.shape[0]

        pos_sim = torch.sum(embeddings_pos * embeddings_pos, dim=1) / self.temperature

        neg_sim = torch.matmul(embeddings_pos, embeddings_neg.transpose(1, 2)) / self.temperature

        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=embeddings_pos.device)

        return F.cross_entropy(logits, labels)
