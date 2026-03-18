use axum::body::Body;
use axum::http::{header, Response, StatusCode};
use serde::Serialize;

use crate::chunker::DocumentOutput;

#[derive(Serialize)]
#[serde(tag = "event")]
enum NdjsonEvent {
    #[serde(rename = "metadata")]
    Metadata {
        title: Option<String>,
        url: Option<String>,
        source_kind: String,
        page_count: Option<u32>,
        ocr_used: bool,
        render_used: bool,
        fallback_used: bool,
        fallback_reason: Option<String>,
        text_quality_score: Option<f32>,
    },
    #[serde(rename = "chunk")]
    Chunk {
        id: String,
        text: String,
        section: Option<String>,
        token_estimate: usize,
    },
    #[serde(rename = "done")]
    Done {
        chunks_total: usize,
        latency_ms: u128,
        pipeline_used: String,
        ocr_used: bool,
        render_used: bool,
        fallback_used: bool,
        fallback_reason: Option<String>,
        text_quality_score: Option<f32>,
    },
}

/// Convert a `DocumentOutput` into an NDJSON streaming response.
///
/// Each line is a JSON object terminated by `\n`. Events are emitted in order:
/// metadata → chunk(s) → done.
pub fn ndjson_stream(output: DocumentOutput) -> Response<Body> {
    let mut lines: Vec<String> = Vec::with_capacity(output.chunks.len() + 2);

    // 1. Metadata event
    let metadata = NdjsonEvent::Metadata {
        title: output.title,
        url: output.canonical_url,
        source_kind: output.diagnostics.pipeline_used.clone(),
        page_count: output.metadata.page_count,
        ocr_used: output.diagnostics.ocr_used,
        render_used: output.diagnostics.render_used,
        fallback_used: output.diagnostics.fallback_used,
        fallback_reason: output.diagnostics.fallback_reason.clone(),
        text_quality_score: output.diagnostics.text_quality_score,
    };
    lines.push(serde_json::to_string(&metadata).unwrap_or_default());

    // 2. Chunk events
    for chunk in &output.chunks {
        let event = NdjsonEvent::Chunk {
            id: chunk.id.clone(),
            text: chunk.text.clone(),
            section: chunk.section.clone(),
            token_estimate: chunk.token_estimate,
        };
        lines.push(serde_json::to_string(&event).unwrap_or_default());
    }

    // 3. Done event
    let done = NdjsonEvent::Done {
        chunks_total: output.chunks.len(),
        latency_ms: output.diagnostics.latency_ms,
        pipeline_used: output.diagnostics.pipeline_used,
        ocr_used: output.diagnostics.ocr_used,
        render_used: output.diagnostics.render_used,
        fallback_used: output.diagnostics.fallback_used,
        fallback_reason: output.diagnostics.fallback_reason,
        text_quality_score: output.diagnostics.text_quality_score,
    };
    lines.push(serde_json::to_string(&done).unwrap_or_default());

    let body = lines.join("\n") + "\n";

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/x-ndjson")
        .body(Body::from(body))
        .expect("failed to build response")
}

/// Convert a `DocumentOutput` into a plain JSON response (non-streaming).
pub fn json_response(output: &DocumentOutput) -> Response<Body> {
    let body = serde_json::to_string(output).unwrap_or_default();
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body))
        .expect("failed to build response")
}
