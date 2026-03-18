# Doc2Agent

Ultra-fast document → agent-friendly output. A Rust microservice that converts any document (web pages, PDFs, Markdown, images) into clean, chunked text streams optimized for consumption by LLM agent CLIs (Codex CLI, Copilot CLI, Claude Code).

## Quick Start

```bash
# Build (CPU-only, no GPU dependencies)
cargo build --release

# Build with GPU OCR support
cargo build --release --features cuda-ep
cargo build --release --features tensorrt-ep

# Run
cargo run --release -- --host 0.0.0.0 --port 8080

# Optional: enable pdfium-backed PDF text extraction / rendering helpers
DOC2AGENT_PDFIUM_ENABLED=1 \
DOC2AGENT_PDFIUM_LIB_PATH=/path/to/libpdfium.so \
cargo run --release -- --host 0.0.0.0 --port 8080

# Optional: point ONNX Runtime provider libraries at a non-standard cache dir
ORT_LIB_DIR=/path/to/onnxruntime/lib \
cargo run --release --features tensorrt-ep -- \
  --model-path ./models/rec_model.onnx \
  --dict-path ./models/ppocr_keys_v1.txt \
  --host 0.0.0.0 \
  --port 8080

# Test
cargo test

# Lint
cargo clippy -- -D warnings
```

## Docker Deployment

The repository now includes a GPU-capable multi-stage `Dockerfile` and a simple
`docker-compose.yml` for production-style deployment on NVIDIA hosts.

Prerequisites:

- Docker Engine with the NVIDIA Container Toolkit installed on the target host
- OCR assets in `./models/` when you want image/PDF OCR:
  - `det_model.onnx`
  - `rec_model.onnx`
  - `ppocr_keys_v1.txt`

Run the service with Docker Compose:

```bash
docker compose up --build -d
curl http://localhost:8080/health
```

Deployment notes:

- The container builds with `cuda-ep` by default on
  `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`.
- Models are mounted read-only at `/models`; the container entrypoint maps them
  to `DOC2AGENT_DET_MODEL`, `DOC2AGENT_MODEL_PATH`, and `DOC2AGENT_DICT_PATH`.
- Override the published port with `DOC2AGENT_PUBLISHED_PORT=3000`.
- Override build features with `DOC2AGENT_BUILD_FEATURES=cuda-ep`.
- To force CPU fallback inside the same image, set `DOC2AGENT_DEVICE_ID=-1`.
- To enable pdfium, mount `libpdfium.so` into `/opt/pdfium/` and set
  `DOC2AGENT_PDFIUM_ENABLED=true`.

## Usage

```bash
# Extract text from a web page
curl -s http://localhost:8080/v1/extract/url \
  -H 'Content-Type: application/json' \
  -d '{"url": "https://arxiv.org/abs/2603.08938", "stream": true}'

# Direct OCR (GPU required)
curl -s http://localhost:8080/v1/ocr \
  -H 'Content-Type: application/json' \
  -d '{"images": ["base64-encoded-image..."]}'

# Health check
curl http://localhost:8080/health

# Prometheus metrics
curl http://localhost:8080/metrics
```

## OCR Benchmarking

```bash
# CPU path
tests/benchmarks/run_ocr_benchmark.sh

# CUDA path
RUNTIME_LABEL=cuda-ep \
CARGO_FEATURES=cuda-ep \
DEVICE_ID=0 \
tests/benchmarks/run_ocr_benchmark.sh

# TensorRT path
RUNTIME_LABEL=tensorrt-ep \
CARGO_FEATURES=tensorrt-ep \
DEVICE_ID=0 \
tests/benchmarks/run_ocr_benchmark.sh
```

The benchmark harness builds `doc2agent` plus `ocr_benchmark`, exercises `/v1/ocr`
across session pool sizes `2,4,8` and batch sizes `1,4,8,16,32`, then writes:

- `tests/benchmarks/ocr_benchmark_results.json`
- `tests/benchmarks/ocr_benchmark_results.txt`

When GPU features are enabled, the harness auto-discovers
`libonnxruntime_providers_shared.so` under `~/.cache/ort.pyke.io` and prepends its
directory to `LD_LIBRARY_PATH`. Override that location with `ORT_LIB_DIR=/path/to/onnxruntime/lib`
if your ONNX Runtime GPU artifacts live elsewhere.

## Architecture

See [ai-docs/design-spec.md](ai-docs/design-spec.md) for the design specification and [ai-docs/execution-record.md](ai-docs/execution-record.md) for the execution record.

**Design principle: Progressive Escalation** — fetch → classify → cheap extraction → quality check → rich render/OCR only if needed → normalize → chunk → stream.

## Optional Pdfium Runtime

Phase 4 pdfium support is loaded dynamically at runtime; no binaries are bundled in this repository.

- Set `DOC2AGENT_PDFIUM_ENABLED=1` to allow binding to a system-installed pdfium library.
- Set `DOC2AGENT_PDFIUM_LIB_PATH` to either the `libpdfium` shared library itself or the directory containing it.
- Pdfium-specific tests in `tests/test_pdf.rs` automatically skip when no runtime binding is available.

## License

TBD
