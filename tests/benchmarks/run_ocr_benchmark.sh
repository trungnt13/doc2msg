#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-38080}"
BASE_URL="${BASE_URL:-http://${HOST}:${PORT}}"
RUNTIME_LABEL="${RUNTIME_LABEL:-cpu}"
CARGO_FEATURES="${CARGO_FEATURES:-}"
ORT_LIB_DIR="${ORT_LIB_DIR:-}"
DEVICE_ID="${DEVICE_ID:--1}"
SESSION_POOL_SIZES="${SESSION_POOL_SIZES:-2,4,8}"
BATCH_SIZES="${BATCH_SIZES:-1,4,8,16,32}"
REQUESTS_PER_WORKER="${REQUESTS_PER_WORKER:-10}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-3}"
IMAGE_WIDTH="${IMAGE_WIDTH:-320}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-48}"
MODEL_PATH="${MODEL_PATH:-$ROOT_DIR/models/rec_model.onnx}"
DICT_PATH="${DICT_PATH:-$ROOT_DIR/models/ppocr_keys_v1.txt}"
SERVER_BIN="${SERVER_BIN:-$ROOT_DIR/target/release/doc2agent}"
BENCH_BIN="${BENCH_BIN:-$ROOT_DIR/target/release/ocr_benchmark}"
OUTPUT_JSON="${OUTPUT_JSON:-$ROOT_DIR/tests/benchmarks/ocr_benchmark_results.json}"
OUTPUT_TXT="${OUTPUT_TXT:-$ROOT_DIR/tests/benchmarks/ocr_benchmark_results.txt}"
SERVER_LOG="${SERVER_LOG:-$ROOT_DIR/tests/benchmarks/ocr_benchmark_server.log}"
RUST_LOG="${RUST_LOG:-doc2agent=warn}"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "missing OCR model at $MODEL_PATH" >&2
  exit 1
fi

if [[ ! -f "$DICT_PATH" ]]; then
  echo "missing OCR dictionary at $DICT_PATH" >&2
  exit 1
fi

IFS=',' read -r -a SESSION_POOLS <<< "$SESSION_POOL_SIZES"
IFS=',' read -r -a BATCHES <<< "$BATCH_SIZES"

MAX_BATCH=0
for batch in "${BATCHES[@]}"; do
  if (( batch > MAX_BATCH )); then
    MAX_BATCH="$batch"
  fi
done

mkdir -p "$(dirname "$OUTPUT_JSON")"
TMP_JSONL="$(mktemp)"
SERVER_PID=""

cleanup() {
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
  rm -f "$TMP_JSONL"
}
trap cleanup EXIT

build_args=(build --release --bin doc2agent --bin ocr_benchmark)
if [[ -n "$CARGO_FEATURES" ]]; then
  build_args+=(--features "$CARGO_FEATURES")
fi
cargo "${build_args[@]}"

discover_ort_lib_dir() {
  if [[ -n "$ORT_LIB_DIR" ]]; then
    printf '%s\n' "$ORT_LIB_DIR"
    return 0
  fi

  local shared_lib
  shared_lib="$(find "${HOME}/.cache/ort.pyke.io" \
    -type f \
    -name 'libonnxruntime_providers_shared.so' \
    -print \
    -quit 2>/dev/null || true)"
  if [[ -n "$shared_lib" ]]; then
    dirname "$shared_lib"
  fi
}

ORT_LIB_DIR="$(discover_ort_lib_dir)"
if [[ -n "$ORT_LIB_DIR" ]]; then
  export LD_LIBRARY_PATH="$ORT_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

wait_for_health() {
  local url="$1"
  for _ in $(seq 1 60); do
    if curl --silent --fail "$url/health" >/dev/null; then
      return 0
    fi
    sleep 1
  done

  echo "server failed to become healthy at $url" >&2
  if [[ -f "$SERVER_LOG" ]]; then
    tail -n 40 "$SERVER_LOG" >&2 || true
  fi
  return 1
}

gpu_details_for_pid() {
  local pid="$1"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi

  nvidia-smi --query-compute-apps=pid,used_gpu_memory \
    --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' -v pid="$pid" '
        {
          gsub(/ /, "", $1);
          gsub(/ /, "", $2);
          if ($1 == pid) {
            print $2;
            exit;
          }
        }
      '
}

start_gpu_sampler() {
  local pid="$1"
  local output="$2"

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi

  (
    while kill -0 "$pid" 2>/dev/null; do
      gpu_details_for_pid "$pid" >>"$output" || true
      sleep 0.2
    done
  ) >/dev/null 2>&1 &
  echo "$!"
}

stop_gpu_sampler() {
  local sampler_pid="${1:-}"
  if [[ -n "$sampler_pid" ]]; then
    kill "$sampler_pid" 2>/dev/null || true
    wait "$sampler_pid" 2>/dev/null || true
  fi
}

max_gpu_memory_from_samples() {
  local sample_file="$1"
  awk '
    NF {
      value = $1 + 0;
      if (value > max) {
        max = value;
      }
    }
    END {
      if (max > 0) {
        print max;
      }
    }
  ' "$sample_file"
}

append_run_json() {
  local base_json="$1"
  local gpu_memory_mib="$2"
  python3 - "$base_json" "$gpu_memory_mib" <<'PY' >> "$TMP_JSONL"
import json
import sys

payload = json.loads(sys.argv[1])
gpu_memory = sys.argv[2].strip()
payload["gpu_process_detected"] = bool(gpu_memory)
payload["gpu_memory_mib"] = int(gpu_memory) if gpu_memory else None
print(json.dumps(payload, sort_keys=True))
PY
}

for pool_size in "${SESSION_POOLS[@]}"; do
  : >"$SERVER_LOG"
  RUST_LOG="$RUST_LOG" \
    "$SERVER_BIN" \
      --host "$HOST" \
      --port "$PORT" \
      --model-path "$MODEL_PATH" \
      --dict-path "$DICT_PATH" \
      --session-pool-size "$pool_size" \
      --ocr-concurrency "$pool_size" \
      --max-batch "$MAX_BATCH" \
      --device-id="$DEVICE_ID" \
      >"$SERVER_LOG" 2>&1 &
  SERVER_PID="$!"

  wait_for_health "$BASE_URL"

  for batch_size in "${BATCHES[@]}"; do
    gpu_sample_file="$(mktemp)"
    gpu_sampler_pid="$(start_gpu_sampler "$SERVER_PID" "$gpu_sample_file")"
    run_json="$("$BENCH_BIN" \
      --base-url "$BASE_URL" \
      --runtime-label "$RUNTIME_LABEL" \
      --session-pool-size "$pool_size" \
      --batch-size "$batch_size" \
      --concurrency "$pool_size" \
      --requests-per-worker "$REQUESTS_PER_WORKER" \
      --warmup-requests "$WARMUP_REQUESTS" \
      --image-width "$IMAGE_WIDTH" \
      --image-height "$IMAGE_HEIGHT")"
    stop_gpu_sampler "$gpu_sampler_pid"
    gpu_memory_mib="$(max_gpu_memory_from_samples "$gpu_sample_file" || true)"
    rm -f "$gpu_sample_file"
    append_run_json "$run_json" "$gpu_memory_mib"
  done

  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=""
done

python3 - "$TMP_JSONL" "$OUTPUT_JSON" "$OUTPUT_TXT" "$HOST" "$PORT" "$RUNTIME_LABEL" "$CARGO_FEATURES" "$DEVICE_ID" "$REQUESTS_PER_WORKER" "$WARMUP_REQUESTS" "$IMAGE_WIDTH" "$IMAGE_HEIGHT" "$ORT_LIB_DIR" <<'PY'
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

jsonl_path = Path(sys.argv[1])
output_json = Path(sys.argv[2])
output_txt = Path(sys.argv[3])
host = sys.argv[4]
port = int(sys.argv[5])
runtime_label = sys.argv[6]
cargo_features = sys.argv[7]
device_id = int(sys.argv[8])
requests_per_worker = int(sys.argv[9])
warmup_requests = int(sys.argv[10])
image_width = int(sys.argv[11])
image_height = int(sys.argv[12])
ort_lib_dir = sys.argv[13]

runs = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
runs.sort(key=lambda item: (item["session_pool_size"], item["batch_size"]))

def command_output(command):
    try:
        return subprocess.check_output(command, text=True).strip()
    except Exception:
        return None

metadata = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "host": host,
    "port": port,
    "runtime_label": runtime_label,
    "cargo_features": cargo_features,
    "device_id": device_id,
    "requests_per_worker": requests_per_worker,
    "warmup_requests": warmup_requests,
    "image_width": image_width,
    "image_height": image_height,
    "platform": platform.platform(),
    "gpu": command_output([
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader",
    ]),
    "ort_lib_dir": ort_lib_dir or None,
}

payload = {
    "metadata": metadata,
    "runs": runs,
}
output_json.write_text(json.dumps(payload, indent=2) + "\n")

lines = []
lines.append("Doc2Agent OCR benchmark results")
lines.append(f"Generated: {metadata['generated_at']}")
lines.append(f"Runtime: {runtime_label}")
lines.append(f"Cargo features: {cargo_features or '(none)'}")
lines.append(f"Device ID: {device_id}")
lines.append(f"Benchmark target: http://{host}:{port}/v1/ocr")
if metadata["gpu"]:
    lines.append(f"GPU: {metadata['gpu']}")
if metadata["ort_lib_dir"]:
    lines.append(f"ORT runtime libs: {metadata['ort_lib_dir']}")
lines.append(
    f"Workload: {requests_per_worker} timed requests/worker, {warmup_requests} warmups, synthetic {image_width}x{image_height} PNG crops"
)
lines.append("")
lines.append(
    "pool  batch  conc  reqs  images  obs_mean_ms  obs_p95_ms  srv_mean_ms  imgs_per_s  gpu"
)
lines.append(
    "----  -----  ----  ----  ------  -----------  ----------  -----------  ----------  ---"
)

for run in runs:
    lines.append(
        f"{run['session_pool_size']:>4}  "
        f"{run['batch_size']:>5}  "
        f"{run['concurrency']:>4}  "
        f"{run['total_requests']:>4}  "
        f"{run['total_images']:>6}  "
        f"{run['observed_latency_ms']['mean']:>11.2f}  "
        f"{run['observed_latency_ms']['p95']:>10.2f}  "
        f"{run['server_latency_ms']['mean']:>11.2f}  "
        f"{run['image_throughput_per_sec']:>10.2f}  "
        f"{'yes' if run['gpu_process_detected'] else 'no'}"
    )

output_txt.write_text("\n".join(lines) + "\n")
PY

echo "wrote $OUTPUT_JSON"
echo "wrote $OUTPUT_TXT"
