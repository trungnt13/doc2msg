# AGENTS.md — Doc2Agent

## Project Overview

Doc2Agent is a Rust microservice that converts documents (web pages, PDFs, Markdown, images) into clean, chunked Markdown text streamed as NDJSON. It serves LLM agent CLIs — Codex CLI, Copilot CLI, Claude Code — so they can ingest arbitrary URLs and files without native document parsing.

**Core principle:** Progressive Escalation — try the cheapest extraction first, escalate to GPU OCR only when needed.

**Stack:** Rust 1.85+, axum 0.8, reqwest 0.12, readability + html2md, pdf-extract, ort (ONNX Runtime) for OCR, Tokio, clap 4.5, tracing.

---

## Setup & Commands

| Action | Command |
|--------|---------|
| Install deps | `cargo build` (fetches all crates) |
| Build (debug) | `cargo build` |
| Build (release) | `cargo build --release` |
| Build + CUDA OCR | `cargo build --release --features cuda-ep` |
| Build + TensorRT OCR | `cargo build --release --features tensorrt-ep` |
| Check (no codegen) | `cargo check` |
| Test | `cargo test` |
| Lint | `cargo clippy -- -D warnings` |
| Format | `cargo fmt` |
| Format check | `cargo fmt -- --check` |
| Run server | `cargo run --release -- --host 0.0.0.0 --port 8080` |
| Pre-commit | `cargo fmt -- --check && cargo clippy -- -D warnings && cargo test` |

---

## Project Structure

```
doc2msg/
├── AGENTS.md              # ← you are here
├── Cargo.toml             # Workspace manifest, feature flags
├── README.md              # User-facing docs
├── run.sh                 # Dev convenience launcher
├── ai-docs/
│   └── doc2agent-implementation-plan.md  # Full design doc
├── src/
│   ├── main.rs            # CLI args (clap derive), server bootstrap
│   ├── config.rs          # RuntimeConfig, env var overrides, feature toggles
│   ├── server.rs          # Axum router, middleware stack, AppState
│   ├── resolver.rs        # URL fetch, MIME sniffing, SourceDescriptor
│   ├── pipeline/
│   │   ├── mod.rs         # Pipeline trait + kind-based dispatcher
│   │   ├── web.rs         # HTML → readability → html2md
│   │   ├── pdf.rs         # pdf-extract (fast) + pdfium-render (rich, Phase 4)
│   │   ├── image.rs       # Image decode → optional OCR
│   │   └── markdown.rs    # Markdown passthrough + cleanup
│   ├── ocr/
│   │   ├── mod.rs         # OCR engine trait
│   │   ├── recognizer.rs  # OpenOCR RepSVTR ONNX inference
│   │   ├── detector.rs    # RepViT DB text detection (Phase 3)
│   │   ├── preprocess.rs  # SIMD resize + normalization (fast_image_resize)
│   │   └── decode.rs      # CTC beam decoder + dictionary
│   ├── normalizer.rs      # Markdown cleanup, whitespace normalization
│   ├── chunker.rs         # Heading/page-aware splitting (1200-2200 chars)
│   ├── cache.rs           # Content-addressed result cache
│   └── stream.rs          # NDJSON / SSE emitter
├── models/                # ONNX models (gitignored, managed externally)
└── tests/
    ├── test_web.rs        # Web pipeline integration tests
    ├── test_pdf.rs        # PDF pipeline integration tests
    ├── test_ocr.rs        # OCR pipeline integration tests
    └── fixtures/          # Sample HTML, PDF, PNG for tests
```

---

## Architecture

### Progressive Escalation Pipeline

Every request follows one path:

```
fetch → classify(MIME) → cheap extraction → quality check → [rich render/OCR if needed] → normalize → chunk → stream
```

Document types (`SourceKind`): `Web`, `Pdf`, `Image`, `Markdown`, `PlainText`. All converge to `DocumentOutput` which contains title, canonical URL, normalized Markdown, chunks, and diagnostics.

### NDJSON Streaming

Responses stream as newline-delimited JSON events:

```
{"event":"metadata","title":"...","url":"...","source_kind":"Web"}
{"event":"chunk","id":0,"text":"...","section":"Introduction","token_estimate":340}
{"event":"chunk","id":1,"text":"...","section":"Methods","token_estimate":380}
{"event":"done","chunks_total":12,"latency_ms":420}
```

Chunks are emitted as soon as each page/section is ready — do not buffer the entire document.

### API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/extract/url` | POST | Extract from URL (mode: auto/fast/rich/ocr) |
| `/v1/extract/bytes` | POST | Extract from uploaded file bytes |
| `/v1/ocr` | POST | Direct OCR for pre-cropped images |
| `/health` | GET | Service health + GPU availability |

---

## Code Style & Conventions

### Error Handling

Use `thiserror` for typed library errors, `anyhow` only in `main.rs` / CLI glue:

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum PipelineError {
    #[error("fetch failed: {0}")]
    Fetch(#[from] reqwest::Error),
    #[error("extraction failed: {0}")]
    Extraction(String),
    #[error("ocr failed: {0}")]
    Ocr(String),
}

pub async fn extract(url: &str) -> Result<DocumentOutput, PipelineError> {
    let response = client.get(url).send().await?;  // auto-converts via #[from]
    // ...
    todo!()
}
```

### Async Patterns

- **I/O-bound** (HTTP, file reads): run directly on Tokio runtime.
- **CPU-bound** (PDF extraction, image processing, OCR pre/post): wrap in `tokio::task::spawn_blocking`.
- Never block the Tokio runtime with synchronous computation or blocking I/O.

### Naming

| Kind | Convention | Example |
|------|-----------|---------|
| Files, functions, variables | `snake_case` | `resolve_url`, `chunk_text` |
| Types, traits, enums | `CamelCase` | `SourceKind`, `Pipeline` |
| Constants | `SCREAMING_SNAKE` | `MAX_CHUNK_SIZE` |
| Feature flags | `kebab-case` | `cuda-ep`, `tensorrt-ep` |

### Logging

Use `tracing` macros with structured fields:

```rust
tracing::info!(url = %url, kind = ?source.source_kind, "starting extraction");
tracing::debug!(chunks = result.chunks.len(), latency_ms = elapsed, "extraction complete");
tracing::warn!(err = %e, "OCR fallback failed, returning partial result");
```

### Other Rules

- No `unwrap()` or `expect()` in library code (`src/*`) — always propagate with `?`.
- All extractors implement the `Pipeline` trait defined in `src/pipeline/mod.rs`.
- CLI args use clap derive macros; every CLI flag has an env var override.
- Global `reqwest::Client` lives in `AppState` — never construct a new one per request.

---

## Testing

| Command | Scope |
|---------|-------|
| `cargo test` | All tests |
| `cargo test --test test_web` | Web pipeline only |
| `cargo test --test test_pdf` | PDF pipeline only |
| `cargo test --test test_ocr` | OCR pipeline only |

- **Integration tests** live in `tests/` and use `axum::test` helpers for HTTP-level assertions.
- **Fixtures** go in `tests/fixtures/` (sample HTML, PDF, PNG files).
- Every new pipeline or feature must include corresponding tests.
- OCR tests are gated behind the presence of model files in `models/`.

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| OpenOCR RepSVTR Mobile for OCR | CTC decoding is fast, ONNX-native, no autoregressive overhead |
| `ort` (ONNX Runtime) over Candle | TensorRT/CUDA execution providers for 10-50× GPU speedups |
| `pdf-extract` for fast PDF path | Pure Rust, zero native dependencies, handles text-layer PDFs |
| `pdfium-render` for rich PDF path (Phase 4) | Renders scanned/image-heavy pages to bitmap for OCR |
| `readability` for HTML extraction | Battle-tested content extraction; wrapped behind trait for swappability |
| NDJSON over SSE | Simpler framing, works with plain `curl`, no event-source boilerplate |
| Feature flags for GPU (`cuda-ep`, `tensorrt-ep`) | CPU-only builds work everywhere; GPU is opt-in |

---

## Workflow Guidelines

1. Branch off `main` with descriptive names: `phase1/web-pipeline`, `fix/chunker-overlap`.
2. Keep PRs focused — one phase or feature per PR.
3. Run before every commit:
   ```sh
   cargo fmt -- --check && cargo clippy -- -D warnings && cargo test
   ```
4. All changes go through PR review — do not push directly to `main`.
5. Reference the implementation plan phase in PR descriptions.

---

## Boundaries

### Always Do

- Run `cargo check` after any code change.
- Use `?` operator for error propagation — never `unwrap()` in `src/`.
- Wrap CPU-bound work in `spawn_blocking` (PDF parsing, image resize, OCR inference).
- Reuse the global `reqwest::Client` from `AppState`.
- Preserve streaming behavior — emit NDJSON chunks incrementally.
- Use in-memory buffers for document processing, not temp files.

### Ask First

- Adding new dependencies to `Cargo.toml`.
- Changing API response schemas or endpoint signatures.
- Modifying the `Pipeline` trait signature.
- Changing OCR model files or configuration.
- Altering chunking defaults (1200–2200 chars, 100–200 char overlap).
- Adding new feature flags.

### Never Do

- Modify files in `models/` — ONNX models are managed externally.
- Commit `.env` files, secrets, or API keys.
- Use `unwrap()` or `expect()` in library code (`src/*`).
- Create temp files for document processing — use in-memory buffers.
- Construct a new `reqwest::Client` per request — reuse `AppState.client`.
- OCR all documents unconditionally — progressive escalation only.
- Block the Tokio runtime with synchronous I/O or heavy computation.
- Ignore `cargo clippy` warnings — they must be zero.

---

## Implementation Phases

| Phase | Scope | Status |
|-------|-------|--------|
| **1 — Web + Fast PDF** | axum server, reqwest fetcher, readability + html2md, pdf-extract, chunker, NDJSON streaming | **Up next** |
| 2 — GPU OCR Engine | ort/ONNX Runtime, RepSVTR recognizer, session pool, SIMD preprocessing | Stubs only |
| 3 — Full OCR Pipeline | RepViT DB detection + recognition, end-to-end image → text | Stubs only |
| 4 — Rich PDF Fallback | pdfium-render, bitmap reuse, quality-based escalation to OCR | Stubs only |
| 5 — Production Hardening | Content-addressed cache, Prometheus metrics, Docker image, load testing | Stubs only |

Source file stubs exist for all phases. Implement Phase 1 fully before moving to Phase 2.

---

## Performance Constraints

These are hard requirements — do not introduce regressions:

| Metric | Target |
|--------|--------|
| Web page → first chunk | < 500ms |
| Text PDF → first chunk | < 300ms |
| OCR recognition (batch=1) | < 15ms |
| OCR throughput (RTX 3090) | > 1000 rec/s |
| Concurrent requests | 64+ |

**Implementation rules:**

- Single global `reqwest::Client` — connection-pooled, reused across all requests.
- SIMD preprocessing via `fast_image_resize` (AVX2/SSE4.1/Neon auto-selected).
- OCR session pool: round-robin across N `ort::Session` instances, mutex-protected.
- Execution provider priority: TensorRT EP → CUDA EP → CPU fallback.
- Stream NDJSON chunks as soon as first page/section is ready — never buffer entire documents.
- Reuse `PdfBitmap` allocations across pages (Phase 4) to avoid per-page heap allocation.
- Split I/O (Tokio tasks) from compute (`spawn_blocking` or rayon thread pool).

---

## Reference

- **Full implementation plan:** [`ai-docs/doc2agent-implementation-plan.md`](ai-docs/doc2agent-implementation-plan.md)
- **axum:** <https://docs.rs/axum/0.8>
- **reqwest:** <https://docs.rs/reqwest/0.12>
- **ort (ONNX Runtime):** <https://docs.rs/ort>
- **pdf-extract:** <https://docs.rs/pdf-extract>
- **readability:** <https://docs.rs/readability>
- **html2md:** <https://docs.rs/html2md>
- **fast_image_resize:** <https://docs.rs/fast_image_resize>
- **tracing:** <https://docs.rs/tracing>
- **clap:** <https://docs.rs/clap/4.5>
- **thiserror:** <https://docs.rs/thiserror>
