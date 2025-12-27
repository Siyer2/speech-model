.PHONY: install train test

install:
	uv sync

train:
	./scripts/train.sh

test:
	uv run pytest

