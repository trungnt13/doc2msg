#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORT_LIB_DIR="${ORT_LIB_DIR:-}"

if [[ -z "$ORT_LIB_DIR" ]]; then
  shared_lib="$(find "${HOME}/.cache/ort.pyke.io" \
    -type f \
    -name 'libonnxruntime_providers_shared.so' \
    -print \
    -quit 2>/dev/null || true)"
  if [[ -n "$shared_lib" ]]; then
    ORT_LIB_DIR="$(dirname "$shared_lib")"
  fi
fi

if [[ -n "$ORT_LIB_DIR" ]]; then
  export LD_LIBRARY_PATH="$ORT_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

export RUST_LOG=info

cargo run --release --features tensorrt-ep -- \
  --model-path "$ROOT_DIR/models/rec_model.onnx" \
  --dict-path "$ROOT_DIR/models/ppocr_keys_v1.txt" \
  --host 0.0.0.0 \
  --port 8080 \
  --session-pool-size 4 \
  --max-batch 32 \
  --intra-threads 1 \
  --inter-threads 1 \
  --device-id 0
