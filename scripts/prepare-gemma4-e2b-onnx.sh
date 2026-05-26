#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv-models"
MODEL_DIR="${ROOT_DIR}/models/gemma4-e2b-it-onnx"
PUBLIC_LINK="${ROOT_DIR}/public/models/gemma4-e2b-it-onnx"
REPO_ID="${GEMMA4_E2B_ONNX_REPO_ID:-onnx-community/gemma-4-E2B-it-ONNX}"

mkdir -p "${MODEL_DIR}" "$(dirname "${PUBLIC_LINK}")"

if [ ! -d "${VENV_DIR}" ]; then
  python3 -m venv "${VENV_DIR}"
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
python -m pip install "huggingface_hub[hf_xet]>=0.36.0"

python - <<PY
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="${REPO_ID}",
    local_dir="${MODEL_DIR}",
    local_dir_use_symlinks=False,
    repo_type="model",
    allow_patterns=[
        "*.json",
        "*.jinja",
        "tokenizer.json",
        "onnx/embed_tokens_q4f16.onnx*",
        "onnx/decoder_model_merged_q4f16.onnx*",
    ],
)
PY

rm -rf "${PUBLIC_LINK}"
ln -s "../../models/gemma4-e2b-it-onnx" "${PUBLIC_LINK}"

cat <<EOF
Gemma4-E2B Transformers.js ONNX artifact ready:
  ${MODEL_DIR}

Local dev symlink:
  ${PUBLIC_LINK} -> ../../models/gemma4-e2b-it-onnx

Use these env values for local/offline serving:
  NEXT_PUBLIC_LLM_RUNTIME=transformers
  NEXT_PUBLIC_LLM_MODEL_ID=gemma-4-E2B-it-ONNX
  NEXT_PUBLIC_LLM_MODEL_URL=/models/gemma4-e2b-it-onnx

If the app can download from Hugging Face at first run, no local model URL is required:
  NEXT_PUBLIC_LLM_RUNTIME=transformers
  NEXT_PUBLIC_LLM_MODEL_URL=onnx-community/gemma-4-E2B-it-ONNX
EOF
