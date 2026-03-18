use std::time::Instant;

use crate::chunker::{
    chunk_markdown, compute_metadata, extract_title, DocumentOutput, PipelineDiagnostics,
    DEFAULT_MAX_CHUNK_CHARS,
};
use crate::normalizer::normalize_markdown;
use crate::resolver::SourceDescriptor;

/// Markdown passthrough pipeline: normalize and chunk existing markdown/plain text.
pub struct MarkdownPipeline;

impl MarkdownPipeline {
    pub async fn extract(&self, source: SourceDescriptor) -> anyhow::Result<DocumentOutput> {
        let start = Instant::now();

        let raw_text = String::from_utf8_lossy(&source.raw_bytes).to_string();
        let markdown = normalize_markdown(&raw_text);
        let title = extract_title(&markdown);
        let chunks = chunk_markdown(&markdown, DEFAULT_MAX_CHUNK_CHARS);
        let metadata = compute_metadata(&markdown, None);

        let pipeline_name = match source.source_kind {
            crate::resolver::SourceKind::PlainText => "plaintext",
            _ => "markdown",
        };

        let diagnostics = PipelineDiagnostics {
            pipeline_used: pipeline_name.to_string(),
            ocr_used: false,
            render_used: false,
            fallback_used: false,
            fallback_reason: None,
            text_quality_score: None,
            latency_ms: start.elapsed().as_millis(),
        };

        Ok(DocumentOutput {
            title,
            canonical_url: source.canonical_url,
            markdown,
            chunks,
            metadata,
            diagnostics,
            image_manifest: None,
        })
    }
}
