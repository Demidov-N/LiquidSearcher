# Liquidity Risk Management System

Stock substitute recommendation system for liquidity-constrained portfolio management.

## Problem
When small-cap positions become illiquid (spread spikes, volume drops), portfolio managers need liquid substitutes that preserve risk/return characteristics.

## Solution
Dual-encoder contrastive learning model:
- **Temporal Encoder**: TCN/Transformer on OHLCV price behavior (G5 features)
- **Tabular Encoder**: FT-Transformer on fundamentals + risk factors (G1-G4 + G6)
- **RankSCL**: Ordinal similarity ranking
- **Liquidity Filters**: 7 hard constraints (Amihud, DArLiQ, spread, volume)

## Quick Start

```bash
# Install dependencies
uv pip install -e ".[dev]"

# Or with pip
pip install -e ".[dev]"
```

## Project Structure
- `src/`: Source code
  - `data/`: Data loading and WRDS integration
  - `features/`: Feature engineering (G1-G6)
  - `models/`: Dual-encoder architecture
  - `liquidity/`: Liquidity metrics and filters
  - `pipeline/`: End-to-end inference pipeline
- `tests/`: Test files
- `notebooks/`: Analysis notebooks
- `data/`: Data storage (not committed)
- `docs/`: Documentation and plans

## Data Requirements
- WRDS access (CRSP, Compustat, IBES, TAQ)
- Universe: Russell 2000 + S&P 400 (~2,400 stocks)
- Period: 2010–2024
- Fallback: ML-estimated spreads if TAQ unavailable

## License
MIT