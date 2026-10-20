#!/usr/bin/env bash
# CPU-only Raspberry Pi launcher. Configuration is read from the environment.
set -euo pipefail

: "${PROJECT_DIR:?Set PROJECT_DIR in /etc/gesture-recognition.env}"
: "${MODEL_PATH:?Set MODEL_PATH in /etc/gesture-recognition.env}"
: "${CAMERA_SOURCE:=0}"
: "${MODEL:=mobilenetv2}"
: "${MODEL_DEPTH:=10}"
: "${WIDTH_MULT:=0.5}"
: "${CLASSES:?Set CLASSES to the checkpoint output-class count}"
: "${DURATION:=8}"
: "${INPUT_SIZE:=80}"
: "${INFERENCE_STRIDE:=2}"
: "${CPU_THREADS:=4}"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model checkpoint does not exist: $MODEL_PATH" >&2
  exit 1
fi

cd "$PROJECT_DIR"
source .venv/bin/activate

args=(--checkpoint "$MODEL_PATH" --source "$CAMERA_SOURCE" --model "$MODEL"
      --model-depth "$MODEL_DEPTH" --width-mult "$WIDTH_MULT" --classes "$CLASSES"
      --duration "$DURATION" --size "$INPUT_SIZE" --stride "$INFERENCE_STRIDE"
      --threads "$CPU_THREADS")
if [[ -n "${LABELS_PATH:-}" ]]; then
  args+=(--labels "$LABELS_PATH")
fi
exec python3 pi_inference.py "${args[@]}"
