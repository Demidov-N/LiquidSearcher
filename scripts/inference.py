"""Simple inference script for stock substitute recommendation."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.dual_encoder import DualEncoder


def load_model(checkpoint_path: str, device: str = "cpu") -> DualEncoder:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = DualEncoder(
        temporal_input_dim=13,
        tabular_continuous_dim=14,
        tabular_categorical_dims=[1, 1],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def compute_stock_embedding(
    model: DualEncoder,
    feature_df: pd.DataFrame,
    symbol: str,
    date: str,
    device: str = "cpu",
) -> dict:
    """Compute embedding for a specific stock on a specific date.

    Args:
        model: Trained DualEncoder model
        feature_df: DataFrame with all features
        symbol: Stock symbol
        date: Date string (YYYY-MM-DD)
        device: Device to use

    Returns:
        Dict with embeddings and metadata
    """
    # Find the row for this symbol/date
    row = feature_df[(feature_df["symbol"] == symbol) & (feature_df["date"] == date)]

    if len(row) == 0:
        raise ValueError(f"No data found for {symbol} on {date}")

    # Get historical window (60 days)
    row_idx = row.index[0]
    symbol_df = (
        feature_df[feature_df["symbol"] == symbol].sort_values("date").reset_index(drop=True)
    )

    # Find position in symbol's data
    pos = symbol_df[symbol_df["date"] == date].index[0]

    # Get 60-day window ending at this date
    start_pos = max(0, pos - 59)
    window = symbol_df.iloc[start_pos : pos + 1]

    # Extract temporal features (13 columns × 60 days)
    temporal_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "return",
        "ma_ratio_5",
        "ma_ratio_10",
        "ma_ratio_15",
        "ma_ratio_20",
        "z_close_5d",
        "z_close_10d",
        "z_close_20d",
    ]

    temporal_data = window[temporal_cols].astype(float).values  # Shape: (n_days, 13)

    # Pad if less than 60 days
    if temporal_data.shape[0] < 60:
        pad = np.zeros((60 - temporal_data.shape[0], 13))
        temporal_data = np.vstack([pad, temporal_data])

    # Extract tabular features (current date only)
    tabular_cont_cols = [
        "market_beta_60d",
        "downside_beta_60d",
        "realized_vol_20d",
        "realized_vol_60d",
        "idiosyncratic_vol",
        "vol_of_vol",
        "mom_1m",
        "mom_3m",
        "mom_6m",
        "macd",
        "log_mktcap",
        "pe_ratio",
        "pb_ratio",
        "roe",
    ]

    # Extract tabular features - keep NaN for TabMixer's native masking
    # Convert pandas NA/NaN to numpy NaN, then let TabMixer handle masking
    tabular_cont_raw = symbol_df.iloc[pos][tabular_cont_cols]
    # Replace pandas NA with numpy NaN, then convert to float
    tabular_cont = pd.to_numeric(tabular_cont_raw, errors="coerce").values
    # Ensure categorical indices are non-negative (embedding lookup requires >= 0)
    gsector_val = max(0, int(symbol_df.iloc[pos]["gsector"]))
    ggroup_val = max(0, int(symbol_df.iloc[pos]["ggroup"]))
    tabular_cat = np.array([gsector_val, ggroup_val])

    # Convert to tensors
    temporal_tensor = (
        torch.tensor(temporal_data, dtype=torch.float32).unsqueeze(0).to(device)
    )  # (1, 60, 13)
    tabular_cont_tensor = (
        torch.tensor(tabular_cont, dtype=torch.float32).unsqueeze(0).to(device)
    )  # (1, 14)
    tabular_cat_tensor = (
        torch.tensor(tabular_cat, dtype=torch.long).unsqueeze(0).to(device)
    )  # (1, 2)

    # Forward pass
    with torch.no_grad():
        temporal_emb, tabular_emb = model(temporal_tensor, tabular_cont_tensor, tabular_cat_tensor)
        joint_emb = torch.cat([temporal_emb, tabular_emb], dim=1)

    return {
        "symbol": symbol,
        "date": date,
        "temporal_emb": temporal_emb[0].cpu().numpy(),
        "tabular_emb": tabular_emb[0].cpu().numpy(),
        "joint_emb": joint_emb[0].cpu().numpy(),
        "gsector": int(symbol_df.iloc[pos]["gsector"]),
        "ggroup": int(symbol_df.iloc[pos]["ggroup"]),
    }


def find_similar_stocks(
    query_emb: np.ndarray,
    query_sector: int,
    query_group: int,
    all_embeddings: list,
    n_neighbors: int = 5,
    same_sector_only: bool = False,
) -> list:
    """Find similar stocks based on embeddings."""
    similarities = []

    for emb in all_embeddings:
        if same_sector_only and emb["gsector"] != query_sector:
            continue

        # Cosine similarity
        sim = np.dot(query_emb, emb["joint_emb"]) / (
            np.linalg.norm(query_emb) * np.linalg.norm(emb["joint_emb"])
        )

        similarities.append(
            {
                "symbol": emb["symbol"],
                "date": emb["date"],
                "similarity": float(sim),
                "gsector": emb["gsector"],
                "ggroup": emb["ggroup"],
                "same_sector": emb["gsector"] == query_sector,
                "same_group": emb["ggroup"] == query_group,
            }
        )

    # Sort by similarity
    similarities.sort(key=lambda x: x["similarity"], reverse=True)

    return similarities[:n_neighbors]


def main():
    parser = argparse.ArgumentParser(description="Find similar stocks")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/fold0_best.pt")
    parser.add_argument(
        "--features", type=str, default="data/processed/features/all_features.parquet"
    )
    parser.add_argument("--symbol", type=str, default="AAPL")
    parser.add_argument("--date", type=str, default="2023-12-20")
    parser.add_argument("--n-neighbors", type=int, default=5)
    parser.add_argument("--same-sector", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, args.device)

    # Load features
    print(f"Loading features from {args.features}")
    df = pd.read_parquet(args.features)

    # Get all unique symbols and dates
    symbols = df["symbol"].unique()
    print(f"Available symbols: {symbols.tolist()}")

    # Compute embeddings for all stocks on the query date
    print(f"\nComputing embeddings for {args.date}...")
    all_embeddings = []

    for symbol in symbols:
        try:
            emb = compute_stock_embedding(model, df, symbol, args.date, args.device)
            all_embeddings.append(emb)
            print(f"  ✓ {symbol}")
        except Exception as e:
            print(f"  ✗ {symbol}: {e}")

    # Find query embedding
    query_emb = None
    query_sector = None
    query_group = None

    for emb in all_embeddings:
        if emb["symbol"] == args.symbol:
            query_emb = emb["joint_emb"]
            query_sector = emb["gsector"]
            query_group = emb["ggroup"]
            break

    if query_emb is None:
        print(f"ERROR: Could not find {args.symbol} on {args.date}")
        return

    # Find similar stocks
    print(f"\nFinding stocks similar to {args.symbol}...")
    similar = find_similar_stocks(
        query_emb, query_sector, query_group, all_embeddings, args.n_neighbors, args.same_sector
    )

    # Display results
    print("\n" + "=" * 70)
    print(f"TOP {args.n_neighbors} STOCKS SIMILAR TO {args.symbol} ON {args.date}")
    print("=" * 70)
    print(
        f"{'Rank':<6}{'Symbol':<10}{'Similarity':<15}{'Sector':<8}{'Group':<12}{'Same Sector':<12}"
    )
    print("-" * 70)

    for i, stock in enumerate(similar, 1):
        same_sector = "✓" if stock["same_sector"] else "✗"
        print(
            f"{i:<6}{stock['symbol']:<10}{stock['similarity']:.6f}     {stock['gsector']:<8}{stock['ggroup']:<12}{same_sector:<12}"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()
