#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv-yolo"
MODEL_DIR="${ROOT_DIR}/public/models/yolo"

mkdir -p "${MODEL_DIR}"

if [ ! -d "${VENV_DIR}" ]; then
  python3 -m venv "${VENV_DIR}"
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
python -m pip install "ultralytics>=8.3.0" onnx onnxslim

WORK_DIR="${ROOT_DIR}/.model-export"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

yolo export model=yolo11n.pt format=onnx imgsz=640 dynamic=false simplify=true opset=17 nms=false
cp yolo11n.onnx "${MODEL_DIR}/yolo11n.onnx"

echo "YOLO11n ONNX model ready: ${MODEL_DIR}/yolo11n.onnx"
