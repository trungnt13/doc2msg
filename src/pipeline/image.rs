use std::time::Instant;

use anyhow::Context;

use crate::chunker::{
    chunk_markdown, compute_metadata, extract_title, DocumentOutput, PipelineDiagnostics,
    DEFAULT_MAX_CHUNK_CHARS,
};
use crate::normalizer::normalize_markdown;
use crate::ocr::shared_document_ocr;
use crate::resolver::SourceDescriptor;

/// Image pipeline: decode image bytes, run full OCR, and wrap text as markdown.
pub struct ImagePipeline;

pub(crate) fn recognize_markdown_from_image(image: &image::DynamicImage) -> anyhow::Result<String> {
    let ocr = shared_document_ocr()?;
    let result = ocr.recognize_document(image)?;
    Ok(result.markdown)
}

impl ImagePipeline {
    pub async fn extract(&self, source: SourceDescriptor) -> anyhow::Result<DocumentOutput> {
        let start = Instant::now();
        let bytes = source.raw_bytes.clone();
        let url = source.canonical_url.clone();

        let ocr_result = tokio::task::spawn_blocking(move || -> anyhow::Result<String> {
            let image = image::load_from_memory(&bytes)
                .context("failed to decode image bytes for OCR extraction")?;
            recognize_markdown_from_image(&image)
        })
        .await
        .context("image OCR worker task failed")??;

        let markdown = normalize_markdown(&ocr_result);
        let title = extract_title(&markdown).or_else(|| {
            markdown
                .lines()
                .map(str::trim)
                .find(|line| !line.is_empty())
                .map(|line| line.to_string())
        });
        let chunks = chunk_markdown(&markdown, DEFAULT_MAX_CHUNK_CHARS);
        let metadata = compute_metadata(&markdown, Some(1));
        let diagnostics = PipelineDiagnostics {
            pipeline_used: "image-ocr".to_string(),
            ocr_used: true,
            render_used: false,
            fallback_used: false,
            fallback_reason: None,
            text_quality_score: None,
            latency_ms: start.elapsed().as_millis(),
        };

        Ok(DocumentOutput {
            title,
            canonical_url: url,
            markdown,
            chunks,
            metadata,
            diagnostics,
            image_manifest: None,
        })
    }
}
