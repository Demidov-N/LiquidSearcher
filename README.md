![GitHub stars](https://img.shields.io/github/stars/Demidov-N/LiquidSearcher?style=social)
![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)

# LiquidSearcher

A dual-encoder contrastive learning system for stock substitute recommendation. This project implements a neural network architecture that learns joint embeddings from temporal market data and tabular fundamental features to identify similar stocks within the same sector.

## Architecture

The system uses a **dual-encoder architecture** inspired by CLIP:

```
┌─────────────────────────────────────────────────────────────────┐
│                     DUAL-ENCODER MODEL                          │
├────────────────────────────┬────────────────────────────────────┤
│   TEMPORAL ENCODER         │   TABULAR ENCODER                  │
│   (BiMT-TCN)               │   (TabMixer)                       │
├────────────────────────────┼────────────────────────────────────┤
│   • TCN (3 layers)         │   • Continuous features (15)       │
│   • Transformer (2 layers) │   • Categorical:                   │
│   • Local + global         │     - gsector → 8-dim              │
│     patterns               │     - ggroup → 16-dim              │
├────────────────────────────┴────────────────────────────────────┤
│                        128-dim each                              │
└────────────────────────────┬─────────────────────────────────────┘
                             │
         Training: Dot product (cosine similarity)
         Inference: Concatenate → 256-dim joint embedding
```

### Key Components

- **BiMT-TCN Temporal Encoder**: Combines TCN for local multi-scale pattern detection with Transformer for global cross-timestep dependencies
- **TabMixer Tabular Encoder**: MLP-Mixer architecture for fundamental features with native missing value handling
- **GICS Hard Negative Sampling**: Three-tier sampling strategy using GICS hierarchy (sector/industry group)
- **InfoNCE Loss**: Contrastive learning with in-batch negatives and temperature scaling

## Installation

```bash
# Clone the repository
git clone https://github.com/Demidov-N/LiquidSearcher.git
cd LiquidSearcher

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Load Data from WRDS

```python
from src.data.wrds_loader import WRDSLoader

# Initialize loader
loader = WRDSLoader()

# Load price data
prices = loader.load_prices(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    start_date='2020-01-01',
    end_date='2023-12-31'
)

# Load fundamentals
fundamentals = loader.load_fundamentals(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    start_date='2020-01-01',
    end_date='2023-12-31'
)
```

### 2. Engineer Features

```python
from src.features import FeatureEngineer

# Create feature engineer
engineer = FeatureEngineer()

# Compute all feature groups
features = engineer.compute_all_features(prices, fundamentals)

# Features include:
# - market_risk: Beta (60d), Downside beta
# - volatility: Realized volatility (20d, 60d), GARCH, Parkinson
# - momentum: Returns (20d, 60d, 252d), Momentum ratio, Reversal
# - valuation: P/E, P/B, ROE, Market cap
# - technical: Z-scores, MA ratios, Volume trends
# - sector: GICS codes (gsector, ggroup, gind, gsubind)
```

### 3. Train Dual-Encoder Model

```python
import torch
from src.models import DualEncoder, InfoNCELoss
from src.training import ContrastiveTrainer

# Initialize model
model = DualEncoder(
    temporal_input_dim=13,          # Technical features
    tabular_continuous_dim=15,      # Fundamental features
    tabular_categorical_dims=[11, 25],  # GICS: sector, group
    tabular_embedding_dims=[8, 16],     # Embedding dimensions
)

# Initialize loss and trainer
loss_fn = InfoNCELoss(temperature=0.07)
trainer = ContrastiveTrainer(model, loss_fn, lr=1e-4)

# Training loop
for batch in dataloader:
    loss = trainer.train_step(batch)
    print(f"Loss: {loss:.4f}")
```

### 4. Inference: Find Similar Stocks

```python
# Get joint embeddings (256-dim)
model.eval()
with torch.no_grad():
    joint_emb = model.get_joint_embedding(
        temporal_features,      # (batch, 60, 13)
        tabular_continuous,   # (batch, 15)
        tabular_categorical   # (batch, 2)
    )

# joint_emb shape: (batch, 256)
# Use for nearest-neighbor search to find similar stocks
```

## Project Structure

```
LiquidSearcher/
├── src/
│   ├── data/
│   │   └── wrds_loader.py          # WRDS data loading
│   ├── features/
│   │   ├── __init__.py
│   │   ├── engineer.py              # Feature orchestration
│   │   ├── market_risk.py           # G1: Beta features
│   │   ├── volatility.py            # G2: Volatility features
│   │   ├── momentum.py              # G3: Momentum features
│   │   ├── valuation.py             # G4: Valuation features
│   │   ├── technical.py             # G5: OHLCV features
│   │   └── sector.py                # G6: GICS features
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                  # Abstract encoder base
│   │   ├── tcn.py                   # Temporal convolutions
│   │   ├── temporal_encoder.py      # BiMT-TCN
│   │   ├── tabmixer.py              # Tabular encoder
│   │   ├── dual_encoder.py          # Combined model
│   │   ├── losses.py                # InfoNCE, RankSCL
│   │   └── sampler.py               # GICS hard negatives
│   └── training/
│       ├── __init__.py
│       └── trainer.py               # Contrastive trainer
├── tests/                           # Comprehensive tests
├── docs/
│   ├── architecture-analysis.md       # Architecture decisions
│   └── plans/                       # Implementation plans
├── examples/                        # Usage examples
├── pyproject.toml                   # Project config
└── requirements.txt                 # Dependencies
```

## Model Details

### Feature Groups

| Group | Features | Count | Description |
|-------|----------|-------|-------------|
| **market_risk** | Beta 60d, Downside beta | 2 | Market risk exposure |
| **volatility** | Realized vol, GARCH, Parkinson | 4 | Volatility measures |
| **momentum** | Returns (20d, 60d, 252d), Ratios | 5 | Price momentum |
| **valuation** | P/E, P/B, ROE, Market cap | 4 | Fundamental ratios |
| **technical** | Z-scores, MA ratios, Volume | 13 | Technical indicators |
| **sector** | GICS codes | 4 | Sector classification |

**Total: 32 features**

### Dual-Encoder Architecture

**Temporal Encoder (BiMT-TCN)**
- Input: (batch, seq_len=60, 13) - technical features over 60 days
- TCN: 3 layers with dilated convolutions (kernel=3, dilation=1,2,4)
- Transformer: 2 layers, 4 heads, hidden=128
- Output: (batch, 128) temporal embedding

**Tabular Encoder (TabMixer)**
- Input: (batch, 15) continuous + (batch, 2) categorical
- Categorical embeddings: gsector→8-dim, ggroup→16-dim
- TabMixer: 4 mixer blocks with token/channel mixing
- Output: (batch, 128) tabular embedding

### Training

- **Loss**: InfoNCE (CLIP-style contrastive learning)
- **Negative Sampling**: GICS-structured hard negatives
  - Level 1: Same ggroup, different beta
  - Level 2: Same gsector, different ggroup
- **Optimizer**: AdamW (lr=1e-4, weight_decay=0.01)

## Testing

Run all tests:

```bash
python -m pytest tests/ -v
```

Run specific test file:

```bash
python -m pytest tests/test_models_dual_encoder.py -v
```

## Development

### Code Quality

The project uses:
- **Ruff**: Linting and formatting
- **mypy**: Type checking
- **pytest**: Testing with coverage

Check code quality:

```bash
# Linting
python -m ruff check src/

# Type checking
python -m mypy src/

# Formatting
python -m ruff format src/
```

### Adding New Features

1. Create feature module in `src/features/`
2. Add tests in `tests/test_features_*.py`
3. Register in `src/features/engineer.py`
4. Run tests and linting
5. Commit with conventional commits

## Documentation

- [Architecture Analysis](docs/architecture-analysis.md) - Detailed architecture decisions
- [Implementation Plan](docs/plans/2026-03-08-dual-encoder-implementation.md) - Development roadmap
- [MVP Specification](MVP.md) - Project requirements

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- BiMT-TCN architecture inspired by recent temporal modeling research
- TabMixer based on MLP-Mixer architecture for tabular data
- GICS classification from S&P Global Market Intelligence
- WRDS data from Wharton Research Data Services

## Citation

If you use this code in your research, please cite:

```bibtex
@software{liquidity_2024,
  title = {LiquidSearcher: Dual-Encoder Stock Substitute Recommendation},
  author = {Demidov, Nikolai},
  year = {2024},
  url = {https://github.com/Demidov-N/LiquidSearcher}
}
```

## Contact

For questions or issues, please open a GitHub issue or contact the maintainer.
