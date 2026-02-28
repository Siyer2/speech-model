#!/bin/bash

set -e

echo "Starting training..."
if [ -n "$NOTE" ]; then
    echo "Note: $NOTE"
fi
if [ -n "$NAME" ]; then
    echo "Run name: $NAME"
fi
PYTORCH_ENABLE_MPS_FALLBACK=1 EVAL_ONLY="$EVAL_ONLY" EXPERIMENT_NOTE="$NOTE" EXPERIMENT_NAME="$NAME" uv run python -m speech_model.train

