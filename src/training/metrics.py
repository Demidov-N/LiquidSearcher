"""Training and validation metrics for dual-encoder model."""

import numpy as np
import torch
import torch.nn.functional as functional
from sklearn.metrics import silhouette_score

from src.models import DualEncoder


def compute_alignment_score(
    temporal_emb: torch.Tensor,
    tabular_emb: torch.Tensor,
) -> float:
    """Compute average cosine similarity of positive pairs.

    This measures how well temporal and tabular views of the SAME stock align.
    High alignment = model is learning consistent embeddings across modalities.

    Args:
        temporal_emb: Temporal embeddings (batch_size, dim)
        tabular_emb: Tabular embeddings (batch_size, dim)

    Returns:
        Average cosine similarity of positive pairs (range: [-1, 1])
    """
    # L2 normalize
    temporal_norm = functional.normalize(temporal_emb, dim=1)
    tabular_norm = functional.normalize(tabular_emb, dim=1)

    # Cosine similarity for each positive pair (diagonal)
    similarities = functional.cosine_similarity(temporal_norm, tabular_norm, dim=1)

    return float(similarities.mean().item())


def compute_hard_negative_similarity(
    temporal_emb: torch.Tensor,
    tabular_emb: torch.Tensor,
    batch_size: int,
    n_hard: int,
) -> float:
    """Compute average similarity between positives and their hard negatives.

    This measures whether hard negatives are actually different from positives.
    Low similarity = model is learning to discriminate hard negatives.

    Args:
        temporal_emb: All embeddings (batch_size + n_hard, dim)
        tabular_emb: All embeddings (batch_size + n_hard, dim)
        batch_size: Number of positive samples
        n_hard: Number of hard negative samples

    Returns:
        Average similarity of positive-to-hard-negative pairs
    """
    if n_hard == 0:
        return 0.0  # No hard negatives to evaluate

    # L2 normalize
    temporal_norm = functional.normalize(temporal_emb, dim=1)
    tabular_norm = functional.normalize(tabular_emb, dim=1)

    # Positive temporal embeddings
    temporal_pos = temporal_norm[:batch_size]

    # Hard negative tabular embeddings
    tabular_hard = tabular_norm[batch_size : batch_size + n_hard]

    # Compute similarity between each positive and each hard negative
    similarity_matrix = torch.matmul(temporal_pos, tabular_hard.t())

    return float(similarity_matrix.mean().item())


def compute_sector_silhouette(
    model: DualEncoder,
    val_samples: list[dict],
) -> float:
    """Compute silhouette score based on GICS sector clustering.

    This measures whether embeddings naturally cluster by sector.
    High silhouette (>0.1) = embeddings have financial meaning.

    Args:
        model: Trained DualEncoder model
        val_samples: List of validation samples with 'gsector' labels

    Returns:
        Silhouette score in range [-1, 1]
    """
    model.eval()

    embeddings_list = []
    labels_list = []

    with torch.no_grad():
        for sample in val_samples:
            # Get joint embedding
            temporal = sample["temporal"].unsqueeze(0)
            tabular_cont = sample["tabular_cont"].unsqueeze(0)
            tabular_cat = sample["tabular_cat"].unsqueeze(0)

            joint_emb = model.get_joint_embedding(temporal, tabular_cont, tabular_cat)
            embeddings_list.append(joint_emb.squeeze(0).cpu().numpy())
            labels_list.append(sample["gsector"])

    # Stack embeddings
    embeddings = np.stack(embeddings_list)
    labels = np.array(labels_list)

    # Compute silhouette score
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0  # Cannot compute silhouette with only 1 cluster

    # Check each label has at least 2 samples
    for label in unique_labels:
        if np.sum(labels == label) < 2:
            return 0.0  # Cannot compute silhouette

    score = silhouette_score(embeddings, labels)

    return float(score)


def compute_all_metrics(
    model: DualEncoder,
    temporal_emb: torch.Tensor,
    tabular_emb: torch.Tensor,
    batch_size: int,
    n_hard: int,
    val_samples: list[dict] | None = None,
) -> dict[str, float]:
    """Compute all training metrics in one call.

    Args:
        model: DualEncoder model
        temporal_emb: Temporal embeddings
        tabular_emb: Tabular embeddings
        batch_size: Number of positive samples
        n_hard: Number of hard negatives
        val_samples: Optional validation samples for silhouette

    Returns:
        Dictionary of metric_name: value
    """
    metrics = {
        "alignment": compute_alignment_score(temporal_emb[:batch_size], tabular_emb[:batch_size]),
        "hard_neg_similarity": compute_hard_negative_similarity(
            temporal_emb, tabular_emb, batch_size, n_hard
        ),
    }

    if val_samples is not None:
        metrics["sector_silhouette"] = compute_sector_silhouette(model, val_samples)

    return metrics
