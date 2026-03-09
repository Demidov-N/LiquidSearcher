"""Training script for dual-encoder model using PyTorch Lightning."""

import argparse
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from src.training.data_module import StockDataModule
from src.training.module import DualEncoderModule


def parse_args():
    parser = argparse.ArgumentParser(description="Train dual-encoder model")
    parser.add_argument("--feature-dir", type=str, default="data/processed/features")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--train-start", type=str, default="2010-01-01")
    parser.add_argument("--train-end", type=str, default="2018-12-31")
    parser.add_argument("--val-start", type=str, default="2020-01-01")
    parser.add_argument("--val-end", type=str, default="2020-12-31")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"dual_encoder_{timestamp}"
    
    print(f"Experiment: {experiment_name}")
    print(f"Training: {args.train_start} to {args.train_end}")
    print(f"Validation: {args.val_start} to {args.val_end}")
    
    data_module = StockDataModule(
        feature_dir=args.feature_dir,
        train_start=args.train_start,
        train_end=args.train_end,
        val_start=args.val_start,
        val_end=args.val_end,
        symbols=args.symbols,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    model_module = DualEncoderModule(
        temporal_input_dim=13,
        tabular_continuous_dim=15,
        embedding_dim=128,
        lr=args.lr,
    )
    
    callbacks = [
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename=f"{experiment_name}" + "-{epoch:02d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
        ),
        EarlyStopping(monitor="val/loss", patience=15, mode="min"),
    ]
    
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=callbacks,
        logger=False,
        accelerator="auto",
        devices="auto",
        gradient_clip_val=1.0,
        log_every_n_steps=50,
    )
    
    print("\nStarting training...")
    trainer.fit(model_module, data_module)
    
    best_ckpt = trainer.checkpoint_callback.best_model_path
    if best_ckpt:
        print(f"\nBest checkpoint: {best_ckpt}")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()