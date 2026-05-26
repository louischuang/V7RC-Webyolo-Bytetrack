#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv-models"
MODEL_DIR="${ROOT_DIR}/models/gemma4-e2b-it"
PUBLIC_LINK="${ROOT_DIR}/public/models/gemma4-e2b-it"
REPO_ID="${GEMMA4_E2B_REPO_ID:-welcoma/gemma-4-E2B-it-q4f16_1-MLC}"

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
)
PY

rm -rf "${PUBLIC_LINK}"
ln -s "../../models/gemma4-e2b-it" "${PUBLIC_LINK}"

cat <<EOF
Gemma4-E2B WebLLM artifact ready:
  ${MODEL_DIR}

Local dev symlink:
  ${PUBLIC_LINK} -> ../../models/gemma4-e2b-it

Use these env values for local/offline serving:
  NEXT_PUBLIC_LLM_MODEL_URL=/models/gemma4-e2b-it
  NEXT_PUBLIC_LLM_MODEL_LIB_URL=/models/gemma4-e2b-it/libs/gemma-4-E2B-it-q4f16_1-MLC-webgpu.wasm
EOF
