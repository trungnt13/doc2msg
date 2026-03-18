---
status: completed
goal: Design the architecture and phased implementation roadmap for Doc2Msg, a Rust microservice converting documents to chunked Markdown for LLM agent CLIs.
prompt: Research best practices and design doc2msg implementation plan covering web, PDF, OCR, and production hardening phases.
created: 2026-03-17T00:00:00Z
finished: 2026-03-17T13:00:00Z
---

# Doc2Msg: Ultra-Fast Document → Agent-Friendly Output

## Design Specification

> A Rust microservice that converts any document (web pages, PDFs, Markdown, images) into
> clean, chunked text streams optimized for consumption by LLM agent CLIs
> (Codex CLI, Copilot CLI, Claude Code).

---

## 1. Problem Statement

Agent CLIs (GitHub Copilot CLI, OpenAI Codex CLI, Claude Code) cannot reliably ingest
raw PDFs, complex HTML, or scanned images. When a user says
`summary https://arxiv.org/abs/2603.08938`, something must:

1. Fetch the resource (resolve redirects, upgrade landing pages to direct assets)
2. Extract text + figures from any document type
3. OCR scanned/image-only content at extreme speed on GPU
4. Normalize everything to clean Markdown + chunked output
5. Stream results back so the agent sees first content in < 1 second

No MCP needed — a standalone HTTP microservice that any CLI can `curl`.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        doc2msg service                            │
│                                                                     │
│  ┌─────────┐   ┌────────────┐   ┌──────────────┐   ┌────────────┐ │
│  │ Ingress  │──▶│ Resolver / │──▶│  Extraction   │──▶│ Normalizer │ │
│  │ (axum)   │   │ Classifier │   │  Pipelines    │   │ / Chunker  │ │
│  └─────────┘   └────────────┘   └──────────────┘   └────────────┘ │
│       │                              │                      │       │
│       │                    ┌─────────┴──────────┐           │       │
│       │                    │                    │           │       │
│       │              ┌─────┴─────┐       ┌─────┴─────┐     │       │
│       │              │ Web lane  │       │ PDF lane  │     │       │
│       │              │ readabil. │       │ pdf-extr. │     │       │
│       │              │ html2md   │       │ pdfium    │     │       │
│       │              └───────────┘       └─────┬─────┘     │       │
│       │                                        │           │       │
│       │                                  ┌─────┴─────┐     │       │
│       │                                  │ OCR lane  │     │       │
│       │                                  │ GPU ONNX  │     │       │
│       │                                  │ OpenOCR   │     │       │
│       │                                  └───────────┘     │       │
│       │                                                    │       │
│       ◀────────────────────────────────────────────────────┘       │
│   NDJSON / SSE stream response                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Design principle: Progressive Escalation

```
fetch → classify → cheap extraction → quality check → rich render/OCR only if needed → normalize → chunk → stream
```

Do **not** OCR everything. Do **not** render every PDF page. Start cheap, escalate only when quality is insufficient.

---

## 3. Component Breakdown

### 3.1 Service Shell — `axum`

| Item | Detail |
|------|--------|
| Framework | `axum 0.8` with `tower` middleware |
| Transport | HTTP/1.1 (upgrade to HTTP/2 later) |
| Streaming | SSE or NDJSON for progressive output |
| Middleware | Timeout, request size limits, tracing, compression |
| State | Shared `AppState` with pooled clients, OCR session pool, cache |

### 3.2 HTTP Fetch — `reqwest`

- **One global `reqwest::Client`** — internally pooled, `Arc`-based, reused across all requests
- Connection pool: idle timeout 90s, configurable max-idle-per-host
- TCP keepalive, connect/read/global timeouts
- Redirect following with limit
- Decompression (gzip, brotli)

### 3.3 Resolver / Classifier

Determine the optimal extraction path before doing work:

| Input | Resolution Strategy |
|-------|-------------------|
| `arxiv.org/abs/<id>` | Upgrade to `arxiv.org/pdf/<id>.pdf` |
| URL with `Content-Type: application/pdf` | Route to PDF lane |
| `.md`, `.txt`, raw content URLs | Skip readability, direct normalize |
| HTML pages | Readability → html2md |
| Image URLs (`.png`, `.jpg`, `.webp`) | Direct to image/OCR lane |
| Uploaded bytes | Magic-byte sniff → route |

Output descriptor:

```rust
struct SourceDescriptor {
    canonical_url: Option<String>,
    source_kind: SourceKind,      // Web, Pdf, Image, Markdown, PlainText
    mime: String,
    filename: Option<String>,
    raw_bytes: Bytes,
}
```

### 3.4 Web Pipeline

```
HTML bytes → readability::extract() → html2md::parse_html() → heading-aware chunking
```

| Step | Crate | Notes |
|------|-------|-------|
| Main-content extraction | `readability` | Use `extract()`, NOT the blocking `scrape()` helper |
| HTML → Markdown | `html2md` | Handles paragraphs, headers, code, tables, images, links |
| Chunking | Custom | Heading-aware, 1200-2200 chars, 100-200 overlap |

### 3.5 PDF Pipeline

#### Fast path (text-native PDFs)

```
bytes → pdf-extract::extract_text_from_mem_by_pages() → quality score → emit chunks
```

- In-memory, no temp files
- Per-page extraction
- Quality scoring: char density, control-char noise, replacement char rate

#### Rich path (layout-heavy / scanned PDFs)

Escalate **only** when fast path quality is below threshold:

```
bytes → pdfium-render → page.text() + page.render_into_bitmap_with_config() → OCR selected pages
```

| Trigger | Action |
|---------|--------|
| Median chars/page < 50 | Escalate to pdfium |
| > 30% empty pages | Escalate to pdfium |
| High Unicode replacement rate | Escalate to pdfium |
| No text layer detected | Full OCR pipeline |
| `include_images=true` | Render page images |

**Critical perf note:** Reuse `PdfBitmap` allocations across pages. The pdfium-render docs explicitly warn that `render_with_config()` allocates a new bitmap per call.

### 3.6 Image Pipeline

```
image bytes → decode → fast_image_resize (SIMD) → OCR if needed → text
```

### 3.7 GPU OCR Engine — OpenOCR RepSVTR on RTX 3090

This is the high-performance core for scanned documents and images.

#### Model Choice

| Candidate | Verdict | Reason |
|-----------|---------|--------|
| **OpenOCR / RepSVTR Mobile** | ✅ **Selected** | CTC decoder = fast inference, ONNX export supported, accuracy + efficiency focus |
| GOT-OCR 2.0 | ❌ Too heavy | Multimodal VLM, low QPS on single 3090 |
| DeepSeek-OCR | ❌ Too heavy | Large memory footprint, accuracy-first not throughput-first |
| PaddleOCR-VL | ❌ Too heavy | Same trade-off as DeepSeek |

#### Runtime Stack

```
ONNX model (exported via OpenOCR tools/toonnx.py)
    ↓
ort (ONNX Runtime Rust bindings, v2.0.0-rc.10)
    ↓
TensorRT EP (first priority) → CUDA EP (fallback) → CPU (last resort)
    ↓
Session pool (N sessions, round-robin dispatch)
```

#### Recognition Preprocessing (SIMD-accelerated)

```rust
// OpenOCR-style preprocessing
// 1. Aspect-preserving resize to height=48, max width=320
// 2. Right-pad to width 320 with zeros
// 3. Normalize: (pixel / 255.0 - 0.5) / 0.5

// SIMD via fast_image_resize
// Auto-detect: AVX2 → SSE4.1 → Neon → scalar fallback
```

| Parameter | Value | Source |
|-----------|-------|--------|
| Input shape | `[batch, 3, 48, 320]` | `configs/rec/svtrv2/repsvtr_ch.yml` |
| Resize | `RecTVResize` (aspect-preserving) | OpenOCR config |
| Dictionary | `ppocr_keys_v1.txt` | OpenOCR config |
| Decoder | Greedy CTC (blank=0, duplicate collapse) | OpenOCR CTC config |

#### Detection (Phase 2 — not in initial prototype)

| Parameter | Value | Source |
|-----------|-------|--------|
| Model | RepViT DB Mobile | `configs/det/dbnet/repvit_db.yml` |
| Resize | `limit_side_len: 960` | OpenOCR config |
| Post-processing | DBPostProcess | `box_thresh: 0.6, thresh: 0.3, unclip_ratio: 1.5` |

#### Concurrency Model

```
Request → round-robin select from SessionPool[N] → lock session → run inference → unlock
```

- Each session is an independent ONNX Runtime instance
- Mutex-protected per-session to prevent concurrent access on same GPU context
- `N` = 2-8, tuned per-GPU for optimal throughput/latency
- Benchmark recommended: start at 4, measure at 2/4/8

### 3.8 Normalizer / Chunker

All pipelines converge to a common output:

```rust
struct DocumentOutput {
    title: Option<String>,
    canonical_url: Option<String>,
    markdown: String,
    chunks: Vec<Chunk>,
    metadata: DocumentMetadata,
    diagnostics: PipelineDiagnostics,
    image_manifest: Option<Vec<ImageRef>>,
}

struct Chunk {
    id: String,           // e.g. "p03-c02"
    text: String,
    section: Option<String>,
    page_start: Option<u32>,
    page_end: Option<u32>,
    char_count: usize,
    token_estimate: usize,
}

struct PipelineDiagnostics {
    pipeline_used: String,   // "web", "pdf-fast", "pdf-rich", "ocr"
    ocr_used: bool,
    render_used: bool,
    latency_ms: u128,
}
```

Chunking defaults:
- 1200–2200 chars per chunk
- 100–200 chars overlap
- Preserve page boundaries (PDF)
- Preserve heading boundaries (HTML/Markdown)

---

## 4. API Design

### `POST /v1/extract/url`

```json
{
  "url": "https://arxiv.org/abs/2603.08938",
  "mode": "auto",           // "auto" | "fast" | "rich" | "ocr"
  "output": "markdown",     // "markdown" | "plain" | "chunks"
  "include_images": false,
  "stream": true,
  "max_pages": null          // null = all
}
```

### `POST /v1/extract/bytes`

Multipart upload: `file` field + optional `filename`, `mime` fields.

### `POST /v1/ocr`

Direct OCR endpoint (recognition-only):

```json
{
  "images": ["data:image/png;base64,...", "base64-raw..."]
}
```

Response:

```json
{
  "model": "openocr-repsvtr",
  "batch_size": 2,
  "latency_ms": 12,
  "items": [
    { "text": "recognized line 1" },
    { "text": "recognized line 2" }
  ]
}
```

### `GET /health`

```json
{
  "status": "ok",
  "pipeline": "doc2msg",
  "gpu": true,
  "ocr_model": "openocr-repsvtr",
  "session_pool": 4
}
```

### Streaming Response (NDJSON)

Events are emitted in order — the agent can start processing before the full document is ready:

```
{"event": "metadata", "title": "...", "url": "...", "pages": 12}
{"event": "chunk", "id": "p01-c01", "text": "...", "section": "Abstract"}
{"event": "chunk", "id": "p01-c02", "text": "...", "section": "Introduction"}
...
{"event": "image_manifest", "images": [...]}
{"event": "done", "diagnostics": {...}}
```

---

## 5. Dependency Matrix

| Crate | Version | Purpose | Feature Flags |
|-------|---------|---------|---------------|
| `axum` | 0.8 | HTTP service | `json`, `tokio`, `http1` |
| `reqwest` | 0.12 | HTTP client | `json`, `gzip`, `brotli`, `stream` |
| `readability` | latest | HTML main-content extraction | — |
| `html2md` | latest | HTML → Markdown | — |
| `pdf-extract` | latest | Fast PDF text extraction | — |
| `pdfium-render` | latest | Rich PDF text + page rendering | — |
| `fast_image_resize` | 6 | SIMD image resize | `image`, `rayon` |
| `image` | =0.25.8 | Image decode/encode | `jpeg`, `png`, `webp` |
| `ort` | =2.0.0-rc.10 | ONNX Runtime bindings | `ndarray`, `cuda` (opt), `tensorrt` (opt) |
| `ndarray` | 0.16 | Tensor operations | — |
| `base64` | 0.22 | Base64 decode for image input | — |
| `clap` | 4.5 | CLI argument parsing | `derive`, `env` |
| `serde` / `serde_json` | 1 | JSON serialization | `derive` |
| `tokio` | 1 | Async runtime | `macros`, `rt-multi-thread`, `signal`, `sync` |
| `tracing` | 0.1 | Structured logging | — |
| `tracing-subscriber` | 0.3 | Log output | `env-filter` |
| `sha2` | 0.10 | Content-addressed cache keys | — |
| `bytes` | 1 | Zero-copy byte buffers | — |
| `mime_guess` | 2 | MIME type detection | — |

### Rust Toolchain

- Minimum: `rustc 1.85+` (pinned deps are compatible with 1.87)
- Target: `x86_64-unknown-linux-gnu` for RTX 3090 deployment

---

## 6. Project Structure

```
doc2msg/
├── Cargo.toml
├── Cargo.lock
├── run.sh                        # Example launch script
├── README.md
├── src/
│   ├── main.rs                   # CLI args, server bootstrap
│   ├── config.rs                 # RuntimeConfig, feature toggles
│   ├── server.rs                 # Axum router, middleware, state
│   ├── resolver.rs               # URL resolution, MIME classification
│   ├── pipeline/
│   │   ├── mod.rs                # Pipeline trait + dispatcher
│   │   ├── web.rs                # HTML → readability → html2md
│   │   ├── pdf.rs                # pdf-extract fast path + pdfium rich path
│   │   ├── image.rs              # Image decode + optional OCR
│   │   └── markdown.rs           # Markdown passthrough + cleanup
│   ├── ocr/
│   │   ├── mod.rs                # OCR engine trait
│   │   ├── recognizer.rs         # OpenOCR RepSVTR ONNX inference
│   │   ├── detector.rs           # (Phase 2) RepViT DB detection
│   │   ├── preprocess.rs         # SIMD resize + normalization
│   │   └── decode.rs             # CTC decoder + dictionary
│   ├── normalizer.rs             # Markdown normalization
│   ├── chunker.rs                # Heading/page-aware chunking
│   ├── cache.rs                  # Content-addressed result cache
│   └── stream.rs                 # NDJSON / SSE emitter
├── models/                       # .gitignore'd — ONNX models at runtime
│   ├── rec_model.onnx
│   ├── det_model.onnx            # Phase 2
│   └── ppocr_keys_v1.txt
└── tests/
    ├── test_web.rs
    ├── test_pdf.rs
    ├── test_ocr.rs
    └── fixtures/
        ├── sample.html
        ├── sample.pdf
        └── sample.png
```

---

## 7. Implementation Phases

### Phase 1 — Minimal Viable Service (Web + Fast PDF)

**Goal:** `POST /v1/extract/url` returns clean Markdown for web pages and text-native PDFs.

#### Tasks

- [ ] **P1.1** Project scaffold — `cargo init`, Cargo.toml, feature flags
- [ ] **P1.2** Config module — CLI args via clap, env var overrides, `RuntimeConfig`
- [ ] **P1.3** Server module — axum router, shared state, tracing middleware, request limits
- [ ] **P1.4** Resolver/classifier — URL normalization, arxiv upgrade, MIME sniffing, `SourceDescriptor`
- [ ] **P1.5** Web pipeline — reqwest fetch → readability extract → html2md → chunks
- [ ] **P1.6** PDF fast pipeline — pdf-extract in-memory by-page → quality scoring → chunks
- [ ] **P1.7** Normalizer — common Markdown cleanup (strip excessive whitespace, fix broken lists/tables)
- [ ] **P1.8** Chunker — heading-aware + page-aware chunking with overlap
- [ ] **P1.9** NDJSON streaming — emit metadata + chunks + done events progressively
- [ ] **P1.10** Integration tests — web page extraction, PDF extraction, arxiv URL resolution
- [ ] **P1.11** Health endpoint, `/v1/formats` metadata, error responses

**Deliverable:** A service that handles `summary https://arxiv.org/abs/2603.08938` end-to-end for text-native PDFs.

### Phase 2 — GPU OCR Engine

**Goal:** Add GPU-accelerated OCR for scanned PDFs and images.

#### Tasks

- [ ] **P2.1** Export OpenOCR ONNX models — use `tools/toonnx.py` for rec (and det) models
- [ ] **P2.2** OCR recognizer module — port existing prototype (session pool, SIMD preprocess, CTC decode)
- [ ] **P2.3** SIMD preprocessing — `fast_image_resize` with AVX2/SSE4.1 auto-detection
- [ ] **P2.4** Session pool — round-robin multi-session with `tokio::sync::Mutex`
- [ ] **P2.5** TensorRT/CUDA execution providers — feature-gated, priority-ordered
- [ ] **P2.6** `/v1/ocr` endpoint — direct recognition API for pre-cropped images
- [ ] **P2.7** TensorRT engine caching + FP16 — reduce cold-start, improve throughput
- [ ] **P2.8** Benchmark — measure QPS at batch sizes 1/4/8/16/32, session pool sizes 2/4/8
- [ ] **P2.9** Integration with PDF rich path — auto-escalate to OCR when text quality is poor

**Deliverable:** OCR at > 1000 recognitions/sec on RTX 3090 for line images.

### Phase 3 — Full OCR Pipeline (Detection + Recognition)

**Goal:** Accept full page/document images and detect + crop + recognize text regions.

#### Tasks

- [ ] **P3.1** Export RepViT DB detector to ONNX
- [ ] **P3.2** Detector module — ONNX inference, limit_side_len resize, DB post-processing
- [ ] **P3.3** DB post-processing port — binarization, contour detection, box scoring, unclip
- [ ] **P3.4** Crop extraction — sort detected boxes, crop from original image
- [ ] **P3.5** Full OCR pipeline — detect → crop → batch recognize → merge text
- [ ] **P3.6** Page-level OCR for PDFs — render page → detect → recognize → structured text
- [ ] **P3.7** Reading order reconstruction — sort text regions into natural reading order

**Deliverable:** Full document OCR from raw page images with structured text output.

### Phase 4 — Rich PDF Fallback

**Goal:** pdfium-render integration for layout-sensitive PDFs.

#### Tasks

- [ ] **P4.1** pdfium-render integration — page text extraction + bitmap rendering
- [ ] **P4.2** Bitmap reuse pool — avoid per-page allocation overhead
- [ ] **P4.3** Quality-based escalation — auto-detect when pdf-extract fails, escalate to pdfium
- [ ] **P4.4** Selective page OCR — only OCR pages where pdfium text extraction also fails
- [ ] **P4.5** Image manifest — optional page image references for downstream agents
- [ ] **P4.6** Figure extraction — detect and extract embedded figures/charts

**Deliverable:** Robust handling of any PDF type including scanned documents.

### Phase 5 — Production Hardening

**Goal:** Cache, observability, deployment.

#### Tasks

- [ ] **P5.1** Content-addressed cache — `sha256(bytes) + mode + options` → cached result
- [ ] **P5.2** Cache storage backend — in-memory LRU + optional disk/Redis
- [ ] **P5.3** Concurrency limits — semaphore-based per-pipeline limits, backpressure
- [ ] **P5.4** Host-level fetch budgets — avoid slow-site head-of-line blocking
- [ ] **P5.5** Prometheus metrics — request count, latency histograms, cache hit rate, OCR usage
- [ ] **P5.6** Dockerfile — multi-stage build, NVIDIA base image, model volume mount
- [ ] **P5.7** Docker Compose — service + optional Redis cache
- [ ] **P5.8** Load testing — wrk/k6 against all endpoints
- [ ] **P5.9** Graceful shutdown — drain in-flight requests on SIGTERM

**Deliverable:** Production-ready deployment on RTX 3090 host.

---

## 8. Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Web page → first chunk | < 500ms | Includes fetch + readability + chunking |
| Text PDF → first chunk | < 300ms | In-memory extraction, no rendering |
| OCR recognition (batch=1) | < 15ms | Single line image, TensorRT FP16 |
| OCR recognition (batch=32) | < 50ms | Batch amortization on RTX 3090 |
| OCR throughput | > 1000 rec/s | Sustained on RTX 3090, session pool=4 |
| Full page OCR (det+rec) | < 200ms | Phase 3, single page |
| Concurrent requests | 64+ | Bounded by semaphore, not by crash |

---

## 9. Performance Tactics

### Must-Do

1. **Reuse `reqwest::Client`** — single global instance, internal connection pool
2. **In-memory extraction** — avoid temp files for PDF and image processing
3. **SIMD preprocessing** — `fast_image_resize` with AVX2/SSE4.1 auto-detection
4. **Session pool** — multiple ONNX sessions, round-robin for GPU concurrency
5. **TensorRT priority** — FP16, engine caching, timing cache for max RTX 3090 throughput
6. **Progressive escalation** — cheap extraction first, OCR/render only when needed
7. **Stream early** — NDJSON chunks as soon as first page/section is ready
8. **Bitmap reuse** — pool `PdfBitmap` allocations for multi-page render
9. **Split I/O and compute** — async I/O tasks on Tokio, CPU-bound work on `spawn_blocking` or rayon

### Avoid

- ❌ Per-request HTTP client construction
- ❌ Unconditional OCR on all documents
- ❌ Unconditional PDF page rendering
- ❌ Temp-file-heavy processing
- ❌ Synchronous fetch in the request hot path
- ❌ Unbounded concurrency (always use semaphores)

---

## 10. ONNX Model Preparation

### Recognition Model Export

```bash
# Clone OpenOCR
git clone https://github.com/Topdu/OpenOCR.git
cd OpenOCR

# Download pretrained RepSVTR mobile weights
# (follow OpenOCR model zoo instructions)

# Export to ONNX
python tools/toonnx.py \
  --config configs/rec/svtrv2/repsvtr_ch.yml \
  --model_path /path/to/repsvtr_mobile.pth \
  --save_path /models/openocr/rec_model.onnx
```

### Detection Model Export (Phase 3)

```bash
python tools/toonnx.py \
  --config configs/det/dbnet/repvit_db.yml \
  --model_path /path/to/repvit_db_mobile.pth \
  --save_path /models/openocr/det_model.onnx
```

### Dictionary

```bash
cp tools/utils/ppocr_keys_v1.txt /models/openocr/ppocr_keys_v1.txt
```

---

## 11. Deployment

### Run Script

```bash
#!/usr/bin/env bash
set -euo pipefail

export RUST_LOG=info

cargo run --release --features tensorrt-ep -- \
  --model /models/openocr/rec_model.onnx \
  --dict /models/openocr/ppocr_keys_v1.txt \
  --host 0.0.0.0 \
  --port 8080 \
  --session-pool 4 \
  --max-batch 32 \
  --intra-threads 1 \
  --inter-threads 1 \
  --device-id 0
```

### Dockerfile (Phase 5)

```dockerfile
FROM nvidia/cuda:12.2-runtime-ubuntu22.04 AS runtime
# Install ONNX Runtime with TensorRT
# Copy binary + models
# EXPOSE 8080
# ENTRYPOINT ["./doc2msg"]
```

### Usage from Agent CLI

```bash
# From Copilot CLI, Codex CLI, or Claude Code:
curl -s http://localhost:8080/v1/extract/url \
  -H 'Content-Type: application/json' \
  -d '{"url": "https://arxiv.org/abs/2603.08938", "stream": true}'

# Direct OCR:
curl -s http://localhost:8080/v1/ocr \
  -H 'Content-Type: application/json' \
  -d '{"images": ["base64-encoded-image..."]}'
```

---

## 12. Existing Prototype

A working OCR runtime prototype already exists and compiles cleanly:

```
session-state/files/openocr-rtx3090-runtime/
├── Cargo.toml          # Pinned: ort=2.0.0-rc.10, image=0.25.8
├── src/
│   ├── main.rs         # axum server, /health, /v1/rec endpoints
│   └── recognizer.rs   # Session pool, SIMD preprocess, CTC decode
└── run.sh              # Example launch with --features tensorrt-ep
```

**Status:** `cargo check --no-default-features` passes. GPU feature builds (`cuda-ep`, `tensorrt-ep`) require a host with ONNX Runtime GPU libraries.

This prototype covers Phase 2 scope and should be absorbed into the full service in Phase 2.

---

## 13. Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| OCR model | OpenOCR RepSVTR Mobile | CTC = fast, ONNX-native, throughput-optimized |
| GPU runtime | `ort` (ONNX Runtime) | TensorRT/CUDA EPs, no Python, minimal overhead |
| Not Candle | — | ONNX Runtime + TensorRT is faster path to NVIDIA-specific speedups |
| Not GOT-OCR 2.0 | — | Too heavy for max QPS on single 3090 |
| PDF fast path | `pdf-extract` | Pure Rust, in-memory, no native deps |
| PDF rich path | `pdfium-render` | Best text + render fidelity, but has native dep |
| HTML extraction | `readability` | Good article extraction; wrap behind trait for swappability |
| Markdown conversion | `html2md` | Proper DOM walk, not just tag stripping |
| Image resize | `fast_image_resize` | SIMD-native (AVX2/SSE4.1/Neon), optional rayon |
| Async runtime | Tokio | Industry standard, axum's native runtime |
| Streaming format | NDJSON | Simpler than SSE, works with curl, easy to parse |

---

## 14. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| PDFium native dependency | Deployment complexity | Dockerfile with pre-built pdfium; Phase 4 only |
| `readability` crate maintenance | May drift on modern HTML | Wrap behind trait; swap to `scraper` + custom rules if needed |
| ONNX model version mismatch | Inference failures | Pin ONNX opset version during export; test in CI |
| TensorRT cold start | First request slow (~30s) | Engine caching; warmup on startup |
| `ort` API instability (pre-2.0) | Breaking changes between RCs | Pin exact version; test before upgrading |
| GPU OOM with large batches | Service crash | Bounded batch sizes; configurable max_batch |
| Scanned PDF quality variance | Poor OCR results | Quality scoring; fallback to larger model or flag for human review |

---

## 15. Reference Projects

| Project | Value |
|---------|-------|
| [kreuzberg](https://github.com/kreuzberg-dev/kreuzberg) | Full document intelligence framework in Rust — study API design, handlers, extraction pipeline |
| [OpenOCR](https://github.com/Topdu/OpenOCR) | Source models, configs, ONNX export tools, preprocessing reference |
| [deepseek-ocr.rs](https://github.com/TimmyOVO/deepseek-ocr.rs) | Rust+Candle OCR serving reference (alternative runtime approach) |
| [ort examples](https://github.com/pykeio/ort) | ONNX Runtime Rust binding patterns, execution provider setup |

---

## 16. Success Criteria

1. ✅ `summary https://arxiv.org/abs/2603.08938` returns clean Markdown in < 3 seconds (text PDF)
2. ✅ Web page extraction returns agent-friendly Markdown with proper heading structure
3. ✅ Scanned PDF pages are OCR'd automatically when text extraction fails
4. ✅ OCR throughput > 1000 recognitions/second on RTX 3090
5. ✅ Streaming response — agent sees first chunk in < 1 second for web pages
6. ✅ Service handles 64+ concurrent requests without crashing
7. ✅ Zero Python dependencies in production runtime
