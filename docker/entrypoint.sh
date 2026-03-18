#!/usr/bin/env bash
set -euo pipefail

append_library_path() {
  local path="$1"
  if [[ -d "$path" ]]; then
    export LD_LIBRARY_PATH="$path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
}

if [[ -z "${DOC2AGENT_DET_MODEL:-}" && -f /models/det_model.onnx ]]; then
  export DOC2AGENT_DET_MODEL=/models/det_model.onnx
fi

if [[ -z "${DOC2AGENT_MODEL_PATH:-}" && -f /models/rec_model.onnx ]]; then
  export DOC2AGENT_MODEL_PATH=/models/rec_model.onnx
fi

if [[ -z "${DOC2AGENT_DICT_PATH:-}" && -f /models/ppocr_keys_v1.txt ]]; then
  export DOC2AGENT_DICT_PATH=/models/ppocr_keys_v1.txt
fi

if [[ -z "${ORT_LIB_DIR:-}" && -f /opt/onnxruntime/libonnxruntime.so ]]; then
  export ORT_LIB_DIR=/opt/onnxruntime
fi

if [[ -n "${ORT_LIB_DIR:-}" ]]; then
  append_library_path "$ORT_LIB_DIR"
fi

if [[ -z "${DOC2AGENT_PDFIUM_LIB_PATH:-}" && -f /opt/pdfium/libpdfium.so ]]; then
  export DOC2AGENT_PDFIUM_LIB_PATH=/opt/pdfium/libpdfium.so
fi

if [[ -n "${DOC2AGENT_PDFIUM_LIB_PATH:-}" ]]; then
  pdfium_dir="$DOC2AGENT_PDFIUM_LIB_PATH"
  if [[ -f "$pdfium_dir" ]]; then
    pdfium_dir="$(dirname "$pdfium_dir")"
  fi
  append_library_path "$pdfium_dir"
fi

if [[ $# -eq 0 ]]; then
  set -- doc2agent
fi

exec "$@"

