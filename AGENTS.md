# AGENTS.md — Doc2Agent

Rust microservice that converts documents (web pages, PDFs, Markdown, images) into chunked Markdown streamed as NDJSON. Serves LLM agent CLIs (Codex, Copilot, Claude Code).

**Core principle:** Progressive Escalation — cheapest extraction first, GPU OCR only when needed.

---

## Agent Workflow

### 1. Commit when done

After completing a task, **commit your changes** with a clear, conventional message. Why: uncommitted work is invisible to other agents and easy to lose.

```sh
# Validate first — never commit broken code
cargo fmt -- --check && cargo clippy -- -D warnings && cargo test

# Then commit with a descriptive message
git add -A && git commit -m "feat(pipeline): implement web extraction with readability"
```

- One commit per logical task (not per file).
- Use [Conventional Commits](https://www.conventionalcommits.org/): `feat`, `fix`, `refactor`, `docs`, `test`, `chore`.
- Include scope: `feat(chunker):`, `fix(ocr):`, `docs(readme):`.
- If a task spans multiple logical steps, multiple commits are fine — each should compile and pass tests.

### 2. Use worktrees for parallel work

When working on independent tasks simultaneously, use **git worktrees** instead of branches in the same checkout. Why: worktrees give each task its own working directory, eliminating merge conflicts and partial states from concurrent edits.

```sh
# Create a worktree for a parallel task
git worktree add ../doc2msg-phase2 -b phase2/ocr-engine

# Work in that directory independently
cd ../doc2msg-phase2

# When done, merge and clean up
cd ../doc2msg
git merge phase2/ocr-engine
git worktree remove ../doc2msg-phase2
```

- Use worktrees when two tasks touch different modules and can proceed independently.
- Name worktree directories `<repo>-<task>` for clarity.
- Always remove worktrees after merging to avoid stale checkouts.

### 3. Keep documentation concise with reasoning

When writing or updating docs, **explain why, not just what**. Why: "what" becomes obvious from reading code — "why" is the part that gets lost.

- Every rule or constraint should have a brief rationale (inline or parenthetical).
- Prefer tables and bullet points over prose paragraphs.
- Delete documentation that restates what the code already says.
- Update docs in the same commit as the code change they describe.

### 4. Archive plans and records in `ai-docs/`

All design plans and execution records live in `ai-docs/`. Why: the live session plan is ephemeral, while future agents need durable records of what was designed, actually built, and deferred.

**Two document types:**

| Type | Naming | Purpose |
|------|--------|---------|
| Design spec | `<project>-design-spec.md` | Architecture, API design, phase roadmap — the *intent* |
| Execution record | `<project>-execution-record.md` | What was built, validated, deferred — the *outcome* |

- Create the design spec when starting a multi-phase effort.
- Create the execution record before closing the task if the work spanned multiple phases, deployments, or benchmarks.
- For focused efforts (bug fixes, optimizations, refactors), use `<project>-<topic>-record.md` — e.g., `doc2agent-latency-optimization-record.md`.
- The execution record must include a **Deviations from Design Spec** section listing anything designed but not implemented, changed, or deferred.

**Every document in `ai-docs/` must begin with YAML frontmatter:**

```md
---
status: completed          # completed | in-progress | abandoned
goal: <high-level goal>
prompt: <user request or concise paraphrase that kicked off the work>
created: <ISO-8601 timestamp>
finished: <ISO-8601 timestamp or empty if in-progress>
---
```

**Execution record body** should capture:

- phase-by-phase execution summary
- key code paths/files changed
- deployment and validation details
- benchmark/runtime evidence
- deviations from design spec
- caveats, blockers, or environment-specific notes

If there is an active session `plan.md`, keep it current during execution and then publish the durable archive to `ai-docs/` when the effort is complete.

**Before marking a task complete**, verify:
- [ ] Session plan archived to `ai-docs/<project>-<topic>-record.md` with YAML frontmatter
- [ ] Deviations from original plan documented
- [ ] File names follow `<project>-` prefix convention

---

## Commands

| Action | Command |
|--------|---------|
| Build | `cargo build` |
| Release build | `cargo build --release` |
| Check (fast, no codegen) | `cargo check` |
| Test | `cargo test` |
| Lint | `cargo clippy -- -D warnings` |
| Format | `cargo fmt` |
| **Pre-commit (run before every commit)** | `cargo fmt -- --check && cargo clippy -- -D warnings && cargo test` |
| Run server | `cargo run --release -- --host 0.0.0.0 --port 8080` |
| GPU build (CUDA) | `cargo build --release --features cuda-ep` |
| GPU build (TensorRT) | `cargo build --release --features tensorrt-ep` |

---

## Architecture

```
fetch → classify(MIME) → cheap extraction → quality check → [rich render/OCR if needed] → normalize → chunk → stream
```

Document types (`SourceKind`): `Web`, `Pdf`, `Image`, `Markdown`, `PlainText` → all converge to `DocumentOutput` (title, URL, Markdown, chunks, diagnostics).

Responses stream as NDJSON — chunks emitted as soon as each section is ready, never buffered whole.

### Key paths

| Path | Purpose |
|------|---------|
| `src/main.rs` | CLI args (clap derive), server bootstrap |
| `src/server.rs` | Axum router, middleware, `AppState` |
| `src/resolver.rs` | URL fetch, MIME sniffing |
| `src/pipeline/` | Extraction per doc type — all implement `Pipeline` trait in `mod.rs` |
| `src/ocr/` | ONNX-based OCR (detection + recognition) |
| `src/chunker.rs` | Heading/page-aware splitting (1200–2200 chars) |
| `src/stream.rs` | NDJSON emitter |
| `tests/` | Integration tests + `fixtures/` |
| `ai-docs/` | Full design doc and implementation plan |

---

## Code Conventions

**Error handling:** `thiserror` in library code, `anyhow` only in `main.rs`. Why: typed errors make pipeline failures actionable; `anyhow` is for top-level "print and exit".

**Async:** I/O-bound work runs on Tokio directly. CPU-bound work (PDF parsing, image processing, OCR) goes in `spawn_blocking`. Why: blocking the Tokio runtime starves all concurrent requests.

**No `unwrap()` / `expect()` in `src/`** — propagate with `?`. Why: panics crash the server for all clients, not just the bad request.

**Single global `reqwest::Client`** in `AppState`. Why: connection pooling; creating a client per request leaks sockets.

**Logging:** `tracing` macros with structured fields (`url = %url, kind = ?kind`).

**Naming:** `snake_case` (files, fns, vars), `CamelCase` (types, traits), `SCREAMING_SNAKE` (constants), `kebab-case` (feature flags).

---

## Boundaries

**Always:**
- Run `cargo check` after code changes.
- Propagate errors with `?`.
- Wrap CPU work in `spawn_blocking`.
- Emit NDJSON incrementally — never buffer entire documents.
- Use in-memory buffers, not temp files. Why: temp files add I/O latency and cleanup burden.

**Ask first:**
- Adding dependencies to `Cargo.toml`. Why: each dep is an attack surface and compile-time cost.
- Changing API schemas, `Pipeline` trait, or chunking defaults.
- Adding feature flags or modifying OCR models.

**Never:**
- Modify `models/` — ONNX models are managed externally.
- Commit `.env`, secrets, or API keys.
- Construct `reqwest::Client` per request.
- OCR unconditionally — progressive escalation only.
- Ignore `cargo clippy` warnings.

---

## Design Decisions

| Decision | Why |
|----------|-----|
| NDJSON over SSE | Simpler framing, works with plain `curl`, no event-source boilerplate |
| `readability` for HTML | Battle-tested extraction; wrapped behind trait for swappability |
| `pdf-extract` for fast PDF | Pure Rust, zero native deps, handles text-layer PDFs |
| `ort` (ONNX Runtime) for OCR | TensorRT/CUDA execution providers give 10–50× GPU speedup |
| Feature flags for GPU | CPU-only builds work everywhere; GPU is opt-in |

---

## Implementation Phases

| Phase | Scope | Status |
|-------|-------|--------|
| 1 — Web + Fast PDF | axum server, reqwest fetcher, readability + html2md, pdf-extract, chunker, NDJSON streaming | Complete |
| 2 — GPU OCR Engine | ort/ONNX Runtime, RepSVTR recognizer, session pool, SIMD preprocessing | Complete |
| 3 — Full OCR Pipeline | RepViT DB detection + recognition, end-to-end image → text | Complete |
| 4 — Rich PDF Fallback | pdfium-render, bitmap reuse, quality-based escalation to OCR | Complete |
| 5 — Production Hardening | Content-addressed cache, Prometheus metrics, Docker, load testing | Complete |

Implement phases sequentially. Design: [`ai-docs/doc2agent-design-spec.md`](ai-docs/doc2agent-design-spec.md) · Execution record: [`ai-docs/doc2agent-execution-record.md`](ai-docs/doc2agent-execution-record.md)

---

## Performance Targets

| Metric | Target | Why it matters |
|--------|--------|----------------|
| Web → first chunk | < 500ms | Agents timeout on slow responses |
| Text PDF → first chunk | < 300ms | Common doc type, must feel instant |
| OCR (batch=1) | < 15ms | Bottleneck for scanned pages |
| Concurrent requests | 64+ | Multiple agents hit the service simultaneously |
