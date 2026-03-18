use axum::body::Body;
use axum::http::{header, Response, StatusCode};
use serde::Serialize;
use thiserror::Error;

use std::sync::Arc;

use crate::chunker::DocumentOutput;

#[derive(Error, Debug)]
pub enum StreamError {
    #[error("serialization failed: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("http response build failed: {0}")]
    Http(axum::http::Error),
}

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
/// metadata → chunk(s) → done. Each event is serialized and flushed individually
/// via `Body::from_stream()`, so the client receives data incrementally.
///
/// Accepts `Arc<DocumentOutput>` to avoid deep-cloning cached values. When the
/// caller holds the only reference (`Arc::strong_count == 1`),
/// `Arc::unwrap_or_clone` extracts the inner value at zero cost.
pub fn ndjson_stream(output: Arc<DocumentOutput>) -> Result<Response<Body>, StreamError> {
    let output = Arc::unwrap_or_clone(output);
    let chunks_total = output.chunks.len();

    // Build all events: metadata, chunks, done
    let mut events: Vec<NdjsonEvent> = Vec::with_capacity(chunks_total + 2);

    events.push(NdjsonEvent::Metadata {
        title: output.title,
        url: output.canonical_url,
        source_kind: output.diagnostics.pipeline_used.clone(),
        page_count: output.metadata.page_count,
        ocr_used: output.diagnostics.ocr_used,
        render_used: output.diagnostics.render_used,
        fallback_used: output.diagnostics.fallback_used,
        fallback_reason: output.diagnostics.fallback_reason.clone(),
        text_quality_score: output.diagnostics.text_quality_score,
    });

    for chunk in output.chunks {
        events.push(NdjsonEvent::Chunk {
            id: chunk.id,
            text: chunk.text,
            section: chunk.section,
            token_estimate: chunk.token_estimate,
        });
    }

    events.push(NdjsonEvent::Done {
        chunks_total,
        latency_ms: output.diagnostics.latency_ms,
        pipeline_used: output.diagnostics.pipeline_used,
        ocr_used: output.diagnostics.ocr_used,
        render_used: output.diagnostics.render_used,
        fallback_used: output.diagnostics.fallback_used,
        fallback_reason: output.diagnostics.fallback_reason,
        text_quality_score: output.diagnostics.text_quality_score,
    });

    // Stream each event as a serialized NDJSON line
    let stream = tokio_stream::iter(events.into_iter().map(|event| {
        serde_json::to_string(&event)
            .map(|line| bytes::Bytes::from(line + "\n"))
            .map_err(StreamError::Serialization)
    }));

    let body = Body::from_stream(stream);

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/x-ndjson")
        .body(body)
        .map_err(StreamError::Http)
}

/// Convert a `DocumentOutput` into a plain JSON response (non-streaming).
pub fn json_response(output: &DocumentOutput) -> Result<Response<Body>, StreamError> {
    let body = serde_json::to_string(output)?;
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body))
        .map_err(StreamError::Http)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunker::{Chunk, DocumentMetadata, DocumentOutput, PipelineDiagnostics};

    fn sample_output(chunks: Vec<Chunk>) -> DocumentOutput {
        DocumentOutput {
            title: Some("Test".to_string()),
            canonical_url: None,
            markdown: "test".to_string(),
            chunks,
            metadata: DocumentMetadata {
                page_count: None,
                word_count: 1,
                char_count: 4,
            },
            diagnostics: PipelineDiagnostics {
                pipeline_used: "test".to_string(),
                ocr_used: false,
                render_used: false,
                fallback_used: false,
                fallback_reason: None,
                text_quality_score: None,
                latency_ms: 1,
            },
            image_manifest: None,
        }
    }

    fn make_chunk(id: &str, text: &str) -> Chunk {
        Chunk {
            id: id.to_string(),
            text: text.to_string(),
            section: None,
            page_start: None,
            page_end: None,
            char_count: text.len(),
            token_estimate: text.len() / 4,
        }
    }

    /// Intent: Valid DocumentOutput with chunks produces well-formed NDJSON (metadata → chunks → done).
    #[tokio::test]
    async fn test_ndjson_stream_well_formed() {
        let output = sample_output(vec![make_chunk("c01", "hello world")]);
        let response = ndjson_stream(Arc::new(output)).unwrap();

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();
        let lines: Vec<serde_json::Value> = text
            .lines()
            .filter(|l| !l.is_empty())
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();

        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0]["event"], "metadata");
        assert_eq!(lines[1]["event"], "chunk");
        assert_eq!(lines[2]["event"], "done");
    }

    /// Intent: DocumentOutput with zero chunks emits only metadata and done events.
    #[tokio::test]
    async fn test_ndjson_stream_zero_chunks() {
        let output = sample_output(vec![]);
        let response = ndjson_stream(Arc::new(output)).unwrap();

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();
        let lines: Vec<serde_json::Value> = text
            .lines()
            .filter(|l| !l.is_empty())
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();

        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["event"], "metadata");
        assert_eq!(lines[1]["event"], "done");
        assert_eq!(lines[1]["chunks_total"], 0);
    }

    /// Intent: Chunk text containing newlines and JSON special chars is properly escaped in NDJSON.
    #[tokio::test]
    async fn test_ndjson_stream_special_chars() {
        let output = sample_output(vec![make_chunk("c01", "line1\nline2\t\"quoted\"")]);
        let response = ndjson_stream(Arc::new(output)).unwrap();

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();

        // Each NDJSON line should be valid JSON
        for line in text.lines().filter(|l| !l.is_empty()) {
            let parsed: Result<serde_json::Value, _> = serde_json::from_str(line);
            assert!(parsed.is_ok(), "line should be valid JSON: {}", line);
        }

        // Verify the special chars survived round-trip
        let chunk_line: serde_json::Value = text
            .lines()
            .filter(|l| !l.is_empty())
            .nth(1)
            .map(|l| serde_json::from_str(l).unwrap())
            .unwrap();
        assert_eq!(chunk_line["text"], "line1\nline2\t\"quoted\"");
    }
}
