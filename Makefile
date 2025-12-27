.PHONY: install train test lint format

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

