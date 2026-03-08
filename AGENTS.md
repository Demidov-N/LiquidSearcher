# Agent Instructions

> **⚠️ NOTE: This is a PROTOTYPE.** Focus on getting things functional first. Clean up, documentation, and edge cases can be handled later. Speed > perfection.
> 
> **BASED ON:** This project is derived from the liquidity project structure and workflow patterns.

## Package Manager
Use **uv** (preferred) or **pip**: `uv pip install -r requirements.txt`, `pip install -e .`

## Running Scripts
All scripts must be run as modules from the project root:

```bash
# Preprocessing
python -m scripts.preprocess_features --start-date 2023-01-01 --end-date 2023-12-31 --symbols AAPL MSFT

# Training
python -m scripts.train --fold 0 --epochs 5 --batch-size 8
```

**Why `python -m`:** Handles imports with `src.` prefix, avoids `sys.path` manipulation.

**Requirements:**
- Must have `scripts/__init__.py`
- Run from project root
- Imports use `from src.X import Y` (not relative)

## File-Scoped Commands
| Task | Command |
|------|---------|
| Typecheck | `python -m mypy src/file.py` |
| Lint | `python -m ruff check src/file.py` |
| Format | `python -m ruff format src/file.py` |
| Test | `python -m pytest tests/test_file.py::TestClass::test_method` |

## Testing
- Run single test: `python -m pytest tests/test_file.py::TestClass::test_method`
- Run pattern: `python -m pytest -k "test_name"`
- All tests: `python -m pytest`

## Commit Attribution
AI commits MUST include:
```
Co-Authored-By: Claude Code <noreply@anthropic.com>
```

## Project Structure
```
src/                    # Source code
  data/                 # WRDS loading, caching
  features/             # Feature engineering (market_risk, momentum, etc.)
  models/               # Dual-encoder model, losses, samplers
  training/             # Trainer, dataset, dataloader, metrics, validator
tests/                  # Test files (mirror src structure)
scripts/                # Runnable preprocessing and training scripts
docs/                   # Documentation and plans
  plans/                # Implementation plans
notebooks/              # Jupyter notebooks (outputs cleared before commit)
data/                   # Data files (gitignored)
  raw/                  # Raw WRDS data
  processed/            # Computed features
  cache/                # Cache files
checkpoints/            # Model checkpoints
```

## Data Handling
- Prefer **polars** for large datasets (>100k rows), pandas for small data
- Never mutate inputs in transformation functions - return new objects
- Use `polars.LazyFrame` for chained transformations

## Dependencies
Add to `pyproject.toml` [project] dependencies section, not requirements.txt.

## Data Files
- Raw data in `data/raw/`, processed in `data/processed/`
- Add to `.gitignore`: `*.parquet`, `*.csv`, `data/`
- Never commit large data files (>1MB)

## Notebooks
- Keep `.ipynb` in `notebooks/` directory
- Clear all outputs before committing
