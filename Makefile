.PHONY: lint format typecheck test all-checks install clean

lint:
	python -m ruff check src/ tests/

format:
	python -m ruff format src/ tests/

typecheck:
	python -m mypy src/

test:
	python -m pytest tests/

all-checks: lint format typecheck test

install:
	pip install -e .

clean:
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true