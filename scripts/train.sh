#!/bin/bash

set -e

echo "Starting training..."
if [ -n "$NOTE" ]; then
    echo "Note: $NOTE"
fi
if [ -n "$NAME" ]; then
    echo "Run name: $NAME"
fi
EXPERIMENT_NOTE="$NOTE" EXPERIMENT_NAME="$NAME" uv run python -m speech_model.train

