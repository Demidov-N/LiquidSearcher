# Agent Instructions

> **⚠️ NOTE: This is a PROTOTYPE.** Focus on getting things functional first. Clean up, documentation, and edge cases can be handled later. Speed > perfection.

## Package Manager
Use **uv** (preferred) or **pip**: `uv pip install -r requirements.txt`, `pip install -e .`

## File-Scoped Commands
| Task | Command |
|------|---------|
| Typecheck | `python -m mypy src/file.py` |
| Lint | `python -m ruff check src/file.py` |
| Format | `python -m ruff format src/file.py` |
| Test | `python -m pytest tests/test_file.py` |

## Testing
- Run single test: `python -m pytest tests/test_file.py::TestClass::test_method`
- Run tests matching pattern: `python -m pytest -k "test_name"`


## Code Style
- Follow [PEP 8](https://peps.python.org/pep-0008/)
- Use Ruff for linting and formatting (config in `pyproject.toml` or `ruff.toml`)
- Imports: stdlib → third-party → local, sorted alphabetically within groups
- Naming: `snake_case` for functions/variables, `PascalCase` for classes, `SCREAMING_SNAKE_CASE` for constants

## Type Hints
- Use type hints for all function signatures and return types
- Prefer `X | None` over `Optional[X]`
- Prefer `list[X]` over `List[X]` (Python 3.9+)

## Error Handling
- Use custom exceptions for domain-specific errors
- Catch specific exceptions, not broad `Exception`
- Avoid bare `except:` clauses

## Project Structure
```
src/              # Source code
tests/            # Test files (mirror src structure)
pyproject.toml    # Project config (Ruff, mypy, pytest settings)
requirements.txt  # Dependencies
```

## Dependencies
Add to `pyproject.toml` [project] dependencies section, not requirements.txt.

## Data Handling
- Prefer **polars** for large datasets (>100k rows), pandas for small data
- Never mutate inputs in transformation functions - return new objects
- Use `polars.LazyFrame` for chained transformations

## Notebooks
- Keep `.ipynb` in `notebooks/` directory
- Clear all outputs before committing

## Data Files
- Raw data in `data/raw/`, processed in `data/processed/`
- Add to `.gitignore`: `*.parquet`, `*.csv`, `data/`
- Never commit large data files (>1MB)
