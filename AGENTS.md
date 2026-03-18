# AGENTS.md — Doc2Msg

Rust microservice that converts documents (web pages, PDFs, Markdown, images) into chunked Markdown streamed as NDJSON. Serves LLM agent CLIs (Codex, Copilot, Claude Code).

**Core principle:** Progressive Escalation — cheapest extraction first, GPU OCR only when needed.

---

## Setup

**Requirements:** Rust 1.85+ (via `rustup`), system C compiler (for native deps).

```sh
cargo build                  # debug build — fetches all crates
cargo build --release        # release build
cargo build --release --features cuda-ep      # GPU build (CUDA)
cargo build --release --features tensorrt-ep  # GPU build (TensorRT)
```

**Optional runtime dependencies:**
- **libpdfium** shared library — needed if `--pdfium-enabled` is set (rich PDF rendering)
- **ONNX model files** in `models/` — needed for OCR (`--model-path`, `--dict-path`)

---

## Commands

| Action | Command |
|--------|---------|
| Check (fast, no codegen) | `cargo check` |
| Build | `cargo build` |
| Test all | `cargo test` |
| Test web pipeline | `cargo test --test test_web` |
| Test PDF pipeline | `cargo test --test test_pdf` |
| Test image pipeline | `cargo test --test test_image` |
| Test OCR | `cargo test --test test_ocr` |
| Lint | `cargo clippy -- -D warnings` |
| Format check | `cargo fmt -- --check` |
| **Pre-commit** | `cargo fmt -- --check && cargo clippy -- -D warnings && cargo test` |
| Run server | `cargo run --release -- --host 0.0.0.0 --port 3000` |
| OCR benchmark | `cargo run --bin ocr_benchmark` |

---

## Project Structure

| Path | Purpose |
|------|---------|
| `src/main.rs` | CLI args (clap derive), server bootstrap, graceful shutdown |
| `src/config.rs` | `RuntimeConfig` — 16 fields, all with `DOC2MSG_*` env var overrides |
| `src/server.rs` | Axum router, middleware stack (timeout, body limit, tracing, gzip), `AppState` |
| `src/resolver.rs` | URL fetch, MIME sniffing, `SourceDescriptor` + `SourceKind` classification |
| `src/pipeline/mod.rs` | `dispatch()` — routes `SourceKind` to the matching pipeline's `extract()` method |
| `src/pipeline/web.rs` | HTML → readability → html2md extraction |
| `src/pipeline/pdf.rs` | PDF extraction: fast text → pdfium fallback → OCR escalation |
| `src/pipeline/image.rs` | Image decode → OCR recognition |
| `src/pipeline/markdown.rs` | Markdown/plaintext passthrough + normalization |
| `src/ocr/` | ONNX-based OCR: detection (`detector.rs`), recognition (`recognizer.rs`), preprocessing, CTC decode |
| `src/chunker.rs` | Heading/page-aware splitting (1200–2200 chars, 100–200 char overlap) |
| `src/normalizer.rs` | Markdown cleanup — strip HTML tags, fix whitespace |
| `src/cache.rs` | Content-addressed in-memory result cache with LRU eviction |
| `src/metrics.rs` | Prometheus metrics (latency, throughput, errors per route) |
| `src/stream.rs` | NDJSON / SSE streaming emitter |
| `src/pdfium.rs` | Pdfium runtime binding — library loading, page rendering |
| `tests/` | Integration tests + `fixtures/` (sample.html, test.pdf) |
| `ai-docs/` | Design specs and execution records |
| `models/` | ONNX models — **managed externally, do not modify** |

---

## Architecture

```
fetch → classify(MIME) → cheap extraction → quality check → [rich render/OCR if needed] → normalize → chunk → stream
```

- **Document types** (`SourceKind`): `Web`, `Pdf`, `Image`, `Markdown`, `PlainText` → all converge to `DocumentOutput`
- **Dispatch**: `src/pipeline/mod.rs::dispatch()` pattern-matches on `SourceKind` and calls the corresponding pipeline's `extract()` method
- **Streaming**: NDJSON chunks emitted as each section is ready — never buffer entire documents

### API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Service health + GPU availability |
| `/metrics` | GET | Prometheus metrics |
| `/v1/extract/url` | POST | Extract from URL |
| `/v1/extract/bytes` | POST | Extract from uploaded file bytes |
| `/v1/ocr` | POST | Direct OCR for pre-cropped images |
| `/v1/formats` | GET | List supported formats |

---

## Code Conventions

**Error handling** — `thiserror` in library code, `anyhow` only in `main.rs`:

```rust
#[derive(thiserror::Error, Debug)]
pub enum PipelineError {
    #[error("fetch failed: {0}")]
    Fetch(#[from] reqwest::Error),
    #[error("extraction failed: {0}")]
    Extraction(String),
}

// ✅ Good — propagate with ?
pub async fn extract(url: &str) -> Result<DocumentOutput, PipelineError> {
    let resp = client.get(url).send().await?;
    // ...
}

// ❌ Bad — never unwrap in src/
let resp = client.get(url).send().await.unwrap();
```

**Async** — I/O-bound work runs on Tokio directly; CPU-bound work goes in `spawn_blocking`:

```rust
// ✅ Good — heavy PDF parsing off the async runtime
let text = tokio::task::spawn_blocking(move || {
    pdf_extract::extract_text_from_mem(&bytes)
}).await??;

// ❌ Bad — blocks the Tokio runtime, starves concurrent requests
let text = pdf_extract::extract_text_from_mem(&bytes)?;
```

**Logging** — `tracing` macros with structured fields:

```rust
tracing::info!(url = %url, kind = ?source.source_kind, "starting extraction");
tracing::warn!(err = %e, "OCR fallback failed, returning partial result");
```

**Naming:** `snake_case` (files, fns, vars), `CamelCase` (types, traits), `SCREAMING_SNAKE` (constants), `kebab-case` (feature flags).

---

## Testing

- **Integration tests** in `tests/` use `axum_test` for HTTP-level assertions
- **Fixtures** in `tests/fixtures/` — `sample.html`, `test.pdf`
- **OCR tests** require model files in `models/` — they're skipped if models are absent
- **Latency tests** in `test_latency.rs` validate performance targets
- Every new pipeline or feature must include corresponding tests

---

## Boundaries

**Always:**
- Run `cargo check` after code changes
- Propagate errors with `?` — no `unwrap()` / `expect()` in `src/`
- Wrap CPU-bound work in `spawn_blocking` (PDF parsing, image processing, OCR)
- Reuse the global `reqwest::Client` from `AppState` — never construct per request
- Emit NDJSON incrementally — never buffer entire documents
- Use in-memory buffers, not temp files

**Ask first:**
- Adding dependencies to `Cargo.toml` — each dep is attack surface and compile-time cost
- Changing API response schemas or endpoint signatures
- Modifying the `dispatch()` function or `extract()` method signatures
- Changing chunking defaults (1200–2200 chars, 100–200 char overlap)
- Adding feature flags or modifying OCR model configuration

**Never:**
- Modify files in `models/` — ONNX models are managed externally
- Commit `.env` files, secrets, or API keys
- OCR all documents unconditionally — progressive escalation only
- Ignore `cargo clippy` warnings — they must be zero
- Block the Tokio runtime with synchronous I/O or heavy computation

---

## Git Workflow

- Use [Conventional Commits](https://www.conventionalcommits.org/) with scope: `feat(chunker):`, `fix(ocr):`, `docs(readme):`
- One commit per logical task; each commit must compile and pass tests
- Run `cargo fmt -- --check && cargo clippy -- -D warnings && cargo test` before every commit
- Branch names: `phase1/web-pipeline`, `fix/chunker-overlap`
- Design docs and execution records live in `ai-docs/` — see existing files for format
