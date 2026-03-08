"""Main training script for dual-encoder model.

Trains across three validation folds with:
- GICS-structured hard negative sampling
- Symmetric InfoNCE loss
- Multi-level metrics (batch, epoch, fold)
- Cross-regime validation
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models import DualEncoder, InfoNCELoss
from src.models.sampler import GICSHardNegativeSampler
from src.training.dataloader import StockDataLoader
from src.training.dataset import FeatureDataset
from src.training.metrics import compute_all_metrics
from src.training.trainer import ContrastiveTrainer
from src.training.validator import CrossRegimeValidator


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    trainer: ContrastiveTrainer,
    epoch: int,
    log_interval: int = 10,
) -> dict:
    """Train for one epoch and return metrics."""
    model.train()

    epoch_metrics = {
        "loss": [],
        "alignment": [],
        "hard_neg_similarity": [],
    }

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for _batch_idx, batch in enumerate(pbar):
        loss = trainer.train_step(batch)

        batch_size = batch["batch_size"]
        n_hard = batch["n_hard"]

        with torch.no_grad():
            temporal_emb, tabular_emb = model(
                batch["temporal"],
                batch["tabular_cont"],
                batch["tabular_cat"],
            )

        metrics = compute_all_metrics(
            model, temporal_emb, tabular_emb, batch_size, n_hard, val_samples=None
        )

        epoch_metrics["loss"].append(loss)
        epoch_metrics["alignment"].append(metrics["alignment"])
        epoch_metrics["hard_neg_similarity"].append(metrics["hard_neg_similarity"])

        pbar.set_postfix(
            {
                "loss": f"{loss:.4f}",
                "align": f"{metrics['alignment']:.3f}",
            }
        )

    return {k: sum(v) / len(v) if v else 0.0 for k, v in epoch_metrics.items()}


def validate_fold(
    model: nn.Module,
    val_dataset: FeatureDataset,
    trainer: ContrastiveTrainer,
    n_val_samples: int = 100,
) -> dict:
    """Run validation on a fold and return metrics."""
    model.eval()

    val_samples = [val_dataset[i] for i in range(min(n_val_samples, len(val_dataset)))]

    from src.training.metrics import compute_sector_silhouette

    silhouette = compute_sector_silhouette(model, val_samples)

    return {"sector_silhouette": silhouette}


def main():
    parser = argparse.ArgumentParser(description="Train dual-encoder model")
    parser.add_argument("--fold", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n-hard", type=int, default=8)
    parser.add_argument("--feature-dir", type=str, default="data/processed/features")
    parser.add_argument("--output-dir", type=str, default="models/checkpoints")

    args = parser.parse_args()

    # Use CrossRegimeValidator for proper date ranges
    validator = CrossRegimeValidator()

    # Check if we have data for the full date ranges or just test data
    feature_dir = Path(args.feature_dir)
    available_files = list(feature_dir.glob("*_features.parquet"))

    if available_files:
        # Load first file to check date range
        sample_df = pd.read_parquet(available_files[0], columns=["date"])
        min_date = sample_df["date"].min()
        max_date = sample_df["date"].max()

        # If data only covers 2023 (test mode), adjust validation fold dates
        if min_date >= pd.Timestamp("2023-01-01") and max_date <= pd.Timestamp("2023-12-31"):
            # Test mode: split 2023 data (Jan-Sep train, Oct-Dec val)
            train_start = pd.Timestamp("2023-01-01")
            train_end = pd.Timestamp("2023-09-30")
            val_start = pd.Timestamp("2023-10-01")
            val_end = pd.Timestamp("2023-12-31")
        else:
            # Full mode: use CrossRegimeValidator dates
            val_fold = validator.get_val_fold(args.fold)
            train_start = pd.Timestamp(validator.train_start)
            train_end = pd.Timestamp(validator.train_end)
            val_start = pd.Timestamp(val_fold.start)
            val_end = pd.Timestamp(val_fold.end)
    else:
        raise ValueError(f"No feature files found in {args.feature_dir}")

    print(f"\nTraining: {train_start} → {train_end}")
    print(f"Validation: {val_start} → {val_end}")

    train_dataset = FeatureDataset(
        feature_dir=args.feature_dir,
        date_range=(train_start.strftime("%Y-%m-%d"), train_end.strftime("%Y-%m-%d")),
    )

    val_dataset = FeatureDataset(
        feature_dir=args.feature_dir,
        date_range=(val_start.strftime("%Y-%m-%d"), val_end.strftime("%Y-%m-%d")),
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    sampler = GICSHardNegativeSampler(n_hard=args.n_hard)

    train_loader = StockDataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        feature_dir=args.feature_dir,
        n_hard=args.n_hard,
        shuffle=True,
    )

    # Get actual dimensions from dataset instance
    n_temporal = len(FeatureDataset.TEMPORAL_COLS)
    n_tabular_cont = len(FeatureDataset.TABULAR_CONT_COLS)
    n_tabular_cat = train_dataset.categorical_dims  # Use actual dimensions from data

    print(
        f"Model dimensions: temporal={n_temporal}, tabular_cont={n_tabular_cont}, tabular_cat={n_tabular_cat}"
    )

    model = DualEncoder(
        temporal_input_dim=n_temporal,
        tabular_continuous_dim=n_tabular_cont,
        tabular_categorical_dims=n_tabular_cat,
    )

    loss_fn = InfoNCELoss(temperature=0.07)
    trainer = ContrastiveTrainer(model, loss_fn, lr=args.lr)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_silhouette = -1.0

    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, trainer, epoch)

        print(
            f"\nEpoch {epoch} - Loss: {train_metrics['loss']:.4f}, "
            f"Align: {train_metrics['alignment']:.3f}"
        )

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            val_metrics = validate_fold(model, val_dataset, trainer)
            print(f"Val Silhouette: {val_metrics['sector_silhouette']:.3f}")

            if val_metrics["sector_silhouette"] > best_silhouette:
                best_silhouette = val_metrics["sector_silhouette"]
                checkpoint_path = output_dir / f"fold{args.fold}_best.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "silhouette": best_silhouette,
                    },
                    checkpoint_path,
                )

    final_path = output_dir / f"fold{args.fold}_final.pt"
    torch.save({"epoch": args.epochs, "model_state_dict": model.state_dict()}, final_path)

    print(f"\nTraining complete! Best silhouette: {best_silhouette:.3f}")


if __name__ == "__main__":
    main()
