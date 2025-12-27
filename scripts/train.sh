#!/bin/bash

set -e

echo "Starting training..."
uv run python -m speech_model.train

