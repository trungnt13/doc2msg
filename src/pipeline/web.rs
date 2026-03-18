use std::io::Cursor;
use std::time::Instant;

use url::Url;

use crate::chunker::{
    chunk_markdown, compute_metadata, extract_title, DocumentOutput, PipelineDiagnostics,
    DEFAULT_MAX_CHUNK_CHARS,
};
use crate::normalizer::normalize_markdown;
use crate::resolver::SourceDescriptor;

/// Web extraction pipeline: HTML → readability → html2md → normalize → chunk.
pub struct WebPipeline;

impl WebPipeline {
    pub async fn extract(&self, source: SourceDescriptor) -> anyhow::Result<DocumentOutput> {
        let start = Instant::now();

        // Convert bytes to string (lossy handles non-UTF-8 pages gracefully)
        let html = String::from_utf8_lossy(&source.raw_bytes).to_string();

        // Parse URL for readability; fall back to a dummy URL if absent
        let url_str = source
            .canonical_url
            .clone()
            .unwrap_or_else(|| "https://example.com".to_string());
        let parsed_url = Url::parse(&url_str)
            .unwrap_or_else(|_| Url::parse("https://example.com").expect("static URL is valid"));

        // Extract main content using readability (CPU-intensive DOM parsing)
        let extracted = tokio::task::spawn_blocking(move || {
            let mut cursor = Cursor::new(html.into_bytes());
            readability::extractor::extract(&mut cursor, &parsed_url)
        })
        .await?
        .map_err(|e| anyhow::anyhow!("readability extraction failed: {e}"))?;

        // Convert extracted HTML content to Markdown
        let markdown_raw = html2md::parse_html(&extracted.content);

        // Normalize whitespace, headings, collapse blank lines
        let markdown = normalize_markdown(&markdown_raw);

        // Title: prefer readability's title, fall back to first # heading
        let title = if !extracted.title.is_empty() {
            Some(extracted.title)
        } else {
            extract_title(&markdown)
        };

        // Chunk the markdown
        let chunks = chunk_markdown(&markdown, DEFAULT_MAX_CHUNK_CHARS);

        // Build output
        let metadata = compute_metadata(&markdown, None);
        let diagnostics = PipelineDiagnostics {
            pipeline_used: "web".to_string(),
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
