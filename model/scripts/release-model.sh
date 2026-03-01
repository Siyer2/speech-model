#!/usr/bin/env bash

set -euo pipefail

REPO="siyer2/speech-model"
MODEL_PATH="model/checkpoints/model-int8.onnx"
REMOTE_NAME="model.onnx"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "Error: HF_TOKEN is not set"
  exit 1
fi

if [ ! -f "${MODEL_PATH}" ]; then
  echo "Error: ${MODEL_PATH} not found"
  exit 1
fi

echo "Uploading ${MODEL_PATH} to huggingface.co/${REPO}..."

model/.venv/bin/python -c "
from huggingface_hub import HfApi
api = HfApi(token='${HF_TOKEN}')
api.create_repo(repo_id='${REPO}', repo_type='model', exist_ok=True)
api.upload_file(
    path_or_fileobj='${MODEL_PATH}',
    path_in_repo='${REMOTE_NAME}',
    repo_id='${REPO}',
    repo_type='model',
)
"

echo "Done. Model available at https://huggingface.co/${REPO}/resolve/main/${REMOTE_NAME}"
