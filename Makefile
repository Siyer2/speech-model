.PHONY: install train test lint format typecheck quality pre-commit

install:
	uv sync

train:
	./scripts/train.sh

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uvx ty check src/

quality: lint typecheck
	@echo "✓ All quality checks passed"

pre-commit:
	uv run --with pre-commit pre-commit install
	@echo "✓ Pre-commit hooks installed"

