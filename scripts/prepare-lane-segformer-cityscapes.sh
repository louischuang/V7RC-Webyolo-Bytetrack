#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${ROOT_DIR}/models/lane/segformer-b0-cityscapes"
REPO_ID="${LANE_SEGFORMER_REPO_ID:-Xenova/segformer-b0-finetuned-cityscapes-640-1280}"
BASE_URL="https://huggingface.co/${REPO_ID}/resolve/main"

mkdir -p "${MODEL_DIR}"

curl -L "${BASE_URL}/config.json" -o "${MODEL_DIR}/config.json"
curl -L "${BASE_URL}/preprocessor_config.json" -o "${MODEL_DIR}/preprocessor_config.json"
curl -L "${BASE_URL}/onnx/model_quantized.onnx" -o "${MODEL_DIR}/model_quantized.onnx"

cat <<MSG
Lane segmentation model ready:
  ${MODEL_DIR}/model_quantized.onnx

Docker Compose serves it at:
  /models/lane/segformer-b0-cityscapes/model_quantized.onnx

Recommended env:
  NEXT_PUBLIC_LANE_MODEL_URL=/models/lane/segformer-b0-cityscapes/model_quantized.onnx
  NEXT_PUBLIC_LANE_MODEL_INPUT_SIZE=224
  NEXT_PUBLIC_LANE_MODEL_PROVIDER=webgpu,wasm
  NEXT_PUBLIC_LANE_MODEL_TARGET_CHANNEL=0
  NEXT_PUBLIC_LANE_MODEL_THRESHOLD=0.5
MSG
