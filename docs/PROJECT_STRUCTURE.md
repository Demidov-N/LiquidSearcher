# LiquidSearcher MLOps Project Structure

This document describes the new MLOps-standard project structure for LiquidSearcher after the restructuring.

## Overview

The project has been restructured to follow modern MLOps best practices, separating concerns into distinct directories optimized for production deployment, research, and collaboration.

## Directory Structure

```
liquid-searcher/
├── src/
│   └── liquid_searcher/           # Production-ready Python package
│       ├── __init__.py           # Package exports and version
│       ├── models/               # Model definitions
│       │   ├── __init__.py
│       │   ├── base.py          # Abstract base encoder
│       │   ├── dual_encoder.py  # Main dual encoder
│       │   ├── temporal_encoder.py  # BiMT-TCN
│       │   ├── tabular_encoder.py   # TabMixer
│       │   ├── mixer.py         # TabMixer implementation
│       │   ├── tcn.py           # Temporal convolution
│       │   └── positional_encoding.py
│       ├── data/                # Data loading and processing
│       │   ├── __init__.py
│       │   ├── wrds_loader.py   # WRDS data integration
│       │   ├── universe.py      # Symbol universe management
│       │   └── credentials.py   # WRDS credentials
│       ├── features/            # Feature engineering
│       │   ├── __init__.py
│       │   ├── processor.py     # Feature computation
│       │   └── normalization.py # Data normalization
│       ├── training/            # Training infrastructure
│       │   ├── __init__.py
│       │   ├── module.py        # Lightning module
│       │   └── data_module.py   # Lightning data module
│       ├── inference/           # Inference and serving (future)
│       │   └── __init__.py
│       ├── config/              # Configuration management
│       │   ├── __init__.py
│       │   └── settings.py
│       └── utils/               # Utilities
│           ├── __init__.py
│           └── memory.py
├── ml_pipeline/                 # ML pipeline scripts
│   ├── __init__.py
│   ├── data_ingestion/         # Data loading scripts
│   │   ├── __init__.py
│   │   └── shard_by_symbol.py
│   ├── feature_engineering/    # Feature computation scripts
│   │   ├── __init__.py
│   │   └── preprocess_features.py
│   ├── training/               # Training scripts
│   │   ├── __init__.py
│   │   └── train.py
│   └── evaluation/             # Evaluation scripts
│       ├── __init__.py
│       └── validate.py
├── configs/                    # Configuration files
│   └── params.yaml            # Central configuration
├── tests/                     # Test suite
│   ├── test_models_*.py
│   ├── test_training_*.py
│   └── test_*.py
├── scripts/                   # Utility scripts (remaining)
│   ├── __init__.py
│   └── [other scripts]
├── docs/                      # Documentation
├── notebooks/                 # Jupyter notebooks
├── data/                      # Data files (gitignored)
│   ├── raw/                  # Raw WRDS data
│   └── processed/            # Feature files
├── checkpoints/              # Model checkpoints
├── experiments/              # Experiment tracking logs
├── results/                  # Evaluation results
├── deployment/               # Deployment configs (future)
├── dvc.yaml                  # DVC pipeline definition
├── pyproject.toml           # Package configuration
├── .gitignore               # Git ignores
└── README.md                # Project documentation
```

## Key Changes from Original Structure

### 1. **Production Package Structure**
- **Before**: `src/` with loose module organization
- **After**: `src/liquid_searcher/` - proper Python package with clear API

### 2. **Separated ML Pipeline**
- **Before**: `scripts/` for all operational code
- **After**: `ml_pipeline/` with organized subdirectories for each stage

### 3. **Centralized Configuration**
- **Before**: Command-line arguments and scattered configs
- **After**: `configs/params.yaml` with hierarchical configuration

### 4. **MLOps Integration**
- **Added**: DVC pipeline for reproducible workflows
- **Added**: Experiment tracking setup
- **Added**: Structured evaluation framework

## Package Usage

### Installation
```bash
# Install in development mode
uv pip install -e .

# Install from PyPI (when published)
pip install liquid-searcher
```

### Import Structure
```python
# Core models
from liquid_searcher import DualEncoder, TemporalEncoder, TabularEncoder

# Data utilities  
from liquid_searcher.data import WRDSLoader, SymbolUniverse

# Feature engineering
from liquid_searcher.features import FeatureProcessor, winsorize

# Training (for research/development)
from liquid_searcher.training import DualEncoderModule
```

## Running the ML Pipeline

### DVC Pipeline (Recommended)
```bash
# Run entire pipeline
dvc repro

# Run specific stage
dvc repro training

# Compare experiments
dvc exp show
```

### Manual Pipeline Execution
```bash
# Data ingestion
python -m ml_pipeline.data_ingestion.shard_by_symbol --config configs/params.yaml

# Feature engineering
python -m ml_pipeline.feature_engineering.preprocess_features --config configs/params.yaml

# Training
python -m ml_pipeline.training.train --config configs/params.yaml

# Evaluation
python -m ml_pipeline.evaluation.validate --config configs/params.yaml
```

## Configuration Management

All parameters are centralized in `configs/params.yaml`:

```yaml
data:
  symbols: ["AAPL", "MSFT", "GOOGL"]
  start_date: "2020-01-01"
  end_date: "2023-12-31"

model:
  temporal:
    input_channels: 13
    output_dim: 128
  tabular:
    continuous_dim: 15
    output_dim: 128

training:
  max_epochs: 100
  batch_size: 32
  learning_rate: 1e-4
```

## Testing

```bash
# Run all tests
python -m pytest

# Run specific test category
python -m pytest tests/test_models_*

# Run with coverage
python -m pytest --cov=liquid_searcher
```

## Development Workflow

1. **Feature Development**: Work in the `liquid_searcher` package
2. **Pipeline Development**: Create scripts in `ml_pipeline/`
3. **Configuration**: Update `configs/params.yaml`
4. **Testing**: Add tests in `tests/`
5. **Experimentation**: Use DVC pipeline with experiment tracking

## Benefits of New Structure

### 🎯 **Production Readiness**
- Clean package structure for deployment
- Separated inference code from training
- Proper Python packaging standards

### 🔄 **MLOps Integration**
- DVC for reproducible pipelines
- Centralized configuration management
- Experiment tracking built-in

### 🧪 **Better Development Experience**
- Clear separation of concerns
- Easier testing and debugging
- Standardized import patterns

### 🚀 **Scalability**
- Modular pipeline stages
- Easy to add new features
- Deployment-ready structure

## Migration Notes

- **Import Changes**: All `from src.*` imports are now `from liquid_searcher.*`
- **Script Locations**: Training scripts moved to `ml_pipeline/`
- **Configuration**: Use `configs/params.yaml` instead of command-line args
- **Package Name**: Changed from `liquidity-risk-system` to `liquid-searcher`

This structure provides a solid foundation for production deployment while maintaining all existing model functionality.