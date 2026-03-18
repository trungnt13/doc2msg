use std::collections::{BTreeSet, HashMap};
use std::time::Instant;

use anyhow::Context;
use image::{DynamicImage, ImageBuffer, Luma, Rgb};

use crate::chunker::{
    chunk_markdown, compute_metadata, extract_title, DocumentOutput, ImageRef, PipelineDiagnostics,
    DEFAULT_MAX_CHUNK_CHARS,
};
use crate::normalizer::normalize_markdown;
use crate::pipeline::image::recognize_markdown_from_image;
use crate::resolver::SourceDescriptor;

/// PDF extraction pipeline with fast (text-based) and rich (render+OCR) paths.
pub struct PdfPipeline;

const OCR_ESCALATION_QUALITY_THRESHOLD: f32 = 0.5;
const LOW_CHARS_PER_PAGE_THRESHOLD: f32 = 50.0;
const HIGH_EMPTY_PAGE_RATIO_THRESHOLD: f32 = 0.3;
const HIGH_REPLACEMENT_RATIO_THRESHOLD: f32 = 0.05;
const EMPTY_PAGE_CHAR_THRESHOLD: usize = 50;
const QUALITY_COMPARISON_EPSILON: f32 = 0.01;

struct FastPdfExtraction {
    pages: Vec<String>,
    quality: TextQuality,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CapabilityStatus {
    Available,
    Unavailable(String),
}

impl CapabilityStatus {
    fn unavailable_reason(&self) -> Option<&str> {
        match self {
            Self::Available => None,
            Self::Unavailable(reason) => Some(reason.as_str()),
        }
    }

    fn is_available(&self) -> bool {
        matches!(self, Self::Available)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum PdfEscalationPlan {
    Fast {
        fallback_reason: Option<String>,
    },
    Pdfium {
        allow_ocr: bool,
        fallback_reason: String,
    },
    EmbeddedImageOcr {
        fallback_reason: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QualityTrigger {
    LowCharsPerPage,
    HighEmptyPageRatio,
    ReplacementCharacterIssues,
    MissingTextLayer,
}

#[derive(Debug, Clone)]
struct PageTextQuality {
    page_number: u32,
    score: f32,
    char_count: usize,
    replacement_ratio: f32,
    triggers: Vec<QualityTrigger>,
}

impl PageTextQuality {
    fn needs_ocr(&self) -> bool {
        self.score <= OCR_ESCALATION_QUALITY_THRESHOLD
            || self.triggers.contains(&QualityTrigger::MissingTextLayer)
    }
}

#[derive(Debug, Clone)]
struct TextQuality {
    score: f32,
    chars_per_page: f32,
    median_chars_per_page: f32,
    page_count: u32,
    empty_page_ratio: f32,
    replacement_ratio: f32,
    missing_text_layer: bool,
    triggers: Vec<QualityTrigger>,
    page_assessments: Vec<PageTextQuality>,
}

impl TextQuality {
    fn needs_escalation(&self) -> bool {
        self.score < OCR_ESCALATION_QUALITY_THRESHOLD || !self.triggers.is_empty()
    }

    fn issue_summary(&self) -> String {
        if self.triggers.is_empty() {
            return "no quality issues detected".to_string();
        }

        let mut issues = Vec::new();
        for trigger in &self.triggers {
            match trigger {
                QualityTrigger::LowCharsPerPage => issues.push(format!(
                    "low chars/page (avg {:.1}, median {:.1})",
                    self.chars_per_page, self.median_chars_per_page
                )),
                QualityTrigger::HighEmptyPageRatio => issues.push(format!(
                    "high empty-page ratio ({:.0}%)",
                    self.empty_page_ratio * 100.0
                )),
                QualityTrigger::ReplacementCharacterIssues => issues.push(format!(
                    "replacement-character issues ({:.1}%)",
                    self.replacement_ratio * 100.0
                )),
                QualityTrigger::MissingTextLayer => issues.push("missing text layer".to_string()),
            }
        }

        issues.join(", ")
    }

    fn poor_page_numbers(&self) -> Vec<u32> {
        self.page_assessments
            .iter()
            .filter(|page| page.needs_ocr())
            .map(|page| page.page_number)
            .collect()
    }
}

impl PdfPipeline {
    fn build_output_from_raw_text(
        &self,
        source: &SourceDescriptor,
        raw_text: String,
        page_count: u32,
        diagnostics: PipelineDiagnostics,
        image_manifest: Option<Vec<ImageRef>>,
    ) -> DocumentOutput {
        let markdown = normalize_markdown(&raw_text);
        let title = infer_document_title(&markdown);
        let chunks = chunk_markdown(&markdown, DEFAULT_MAX_CHUNK_CHARS);
        let metadata = compute_metadata(&markdown, Some(page_count));

        DocumentOutput {
            title,
            canonical_url: source.canonical_url.clone(),
            markdown,
            chunks,
            metadata,
            diagnostics,
            image_manifest,
        }
    }

    fn build_output_from_pages(
        &self,
        source: &SourceDescriptor,
        mut pages: Vec<(u32, String)>,
        diagnostics: PipelineDiagnostics,
        image_manifest: Option<Vec<ImageRef>>,
    ) -> DocumentOutput {
        pages.sort_by_key(|(page_number, _)| *page_number);
        let page_count = pages
            .iter()
            .map(|(page_number, _)| *page_number)
            .max()
            .unwrap_or_default();
        let raw_text = render_page_markdown(pages);

        self.build_output_from_raw_text(source, raw_text, page_count, diagnostics, image_manifest)
    }

    async fn extract_fast_text(
        &self,
        source: &SourceDescriptor,
    ) -> anyhow::Result<FastPdfExtraction> {
        let bytes = source.raw_bytes.clone();

        let pages = tokio::task::spawn_blocking(move || {
            pdf_extract::extract_text_from_mem_by_pages(&bytes)
        })
        .await??;

        let quality = assess_text_quality(&pages);
        Ok(FastPdfExtraction { pages, quality })
    }

    fn build_fast_output(
        &self,
        source: &SourceDescriptor,
        start: Instant,
        fast: &FastPdfExtraction,
        fallback_used: bool,
        fallback_reason: Option<String>,
    ) -> DocumentOutput {
        let pages = fast
            .pages
            .iter()
            .enumerate()
            .map(|(index, page)| {
                let page_number = u32::try_from(index + 1).unwrap_or(u32::MAX);
                (page_number, normalize_pdf_text(page))
            })
            .collect();

        let diagnostics = PipelineDiagnostics {
            pipeline_used: "pdf-fast".to_string(),
            ocr_used: false,
            render_used: false,
            fallback_used,
            fallback_reason,
            text_quality_score: Some(fast.quality.score),
            latency_ms: start.elapsed().as_millis(),
        };

        self.build_output_from_pages(source, pages, diagnostics, None)
    }

    async fn extract_embedded_page_images(
        &self,
        source: &SourceDescriptor,
    ) -> anyhow::Result<Vec<(u32, DynamicImage)>> {
        let bytes = source.raw_bytes.clone();

        tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<(u32, DynamicImage)>> {
            let document = pdf_extract::Document::load_mem(bytes.as_ref())
                .context("failed to parse PDF for OCR fallback")?;
            let mut page_images = Vec::new();

            for (page_number, page_id) in document.get_pages() {
                let images = document
                    .get_page_images(page_id)
                    .with_context(|| format!("failed to inspect page {page_number} images"))?;

                let best_image = images
                    .iter()
                    .filter_map(|image| {
                        decode_embedded_pdf_image(image)
                            .ok()
                            .map(|decoded| ((image.width * image.height), decoded))
                    })
                    .max_by_key(|(area, _)| *area)
                    .map(|(_, image)| image);

                if let Some(image) = best_image {
                    page_images.push((page_number, image));
                }
            }

            Ok(page_images)
        })
        .await
        .context("PDF OCR fallback worker task failed")?
    }

    pub async fn extract_pdfium_page_text(
        &self,
        source: &SourceDescriptor,
    ) -> anyhow::Result<Vec<String>> {
        crate::pdfium::extract_text_from_bytes(source.raw_bytes.clone().to_vec())
            .await
            .map(|pages| pages.into_iter().map(|page| page.text).collect())
    }

    async fn render_pdfium_page_assets(
        &self,
        source: &SourceDescriptor,
        page_numbers: &[u32],
    ) -> anyhow::Result<Vec<crate::pdfium::PdfiumRenderedPage>> {
        crate::pdfium::render_selected_pages_for_ocr(
            source.raw_bytes.clone().to_vec(),
            page_numbers.to_vec(),
        )
        .await
    }

    pub async fn render_pdfium_pages(
        &self,
        source: &SourceDescriptor,
    ) -> anyhow::Result<Vec<(u32, DynamicImage)>> {
        self.render_pdfium_page_assets(source, &[])
            .await
            .map(|pages| {
                pages
                    .into_iter()
                    .map(|page| (page.page_number, page.image))
                    .collect()
            })
    }

    async fn recognize_rendered_pages(
        &self,
        rendered_pages: Vec<(u32, DynamicImage)>,
    ) -> anyhow::Result<Vec<(u32, String)>> {
        let mut ocr_pages = Vec::with_capacity(rendered_pages.len());

        for (page_number, image) in rendered_pages {
            let markdown =
                tokio::task::spawn_blocking(move || recognize_markdown_from_image(&image))
                    .await
                    .with_context(|| {
                        format!("OCR worker task failed for rendered PDF page {page_number}")
                    })??;
            ocr_pages.push((page_number, markdown));
        }

        Ok(ocr_pages)
    }

    pub async fn extract_with_page_ocr(
        &self,
        source: &SourceDescriptor,
    ) -> anyhow::Result<DocumentOutput> {
        let start = Instant::now();
        let rendered_pages = self.render_pdfium_page_assets(source, &[]).await?;
        let image_manifest =
            image_manifest_from_rendered_pages(&rendered_pages, "pdfium-rendered page image");
        let ocr_pages = self
            .recognize_rendered_pages(
                rendered_pages
                    .into_iter()
                    .map(|page| (page.page_number, page.image))
                    .collect(),
            )
            .await?;
        let final_quality = assess_text_quality(&sorted_page_texts(ocr_pages.clone()));
        let diagnostics = PipelineDiagnostics {
            pipeline_used: "pdf-page-ocr".to_string(),
            ocr_used: true,
            render_used: true,
            fallback_used: false,
            fallback_reason: None,
            text_quality_score: Some(final_quality.score),
            latency_ms: start.elapsed().as_millis(),
        };

        Ok(self.build_output_from_pages(source, ocr_pages, diagnostics, image_manifest))
    }

    async fn extract_with_embedded_image_ocr(
        &self,
        source: &SourceDescriptor,
        start: Instant,
        fast: &FastPdfExtraction,
        fallback_reason: &str,
    ) -> anyhow::Result<Option<DocumentOutput>> {
        let embedded_images = self.extract_embedded_page_images(source).await?;
        if embedded_images.is_empty() {
            return Ok(None);
        }

        let ocr_page_count = embedded_images.len();
        let image_manifest =
            image_manifest_from_dynamic_images(&embedded_images, "embedded PDF page image");
        let mut ocr_pages = HashMap::with_capacity(embedded_images.len());
        for (page_number, image) in embedded_images {
            let markdown =
                tokio::task::spawn_blocking(move || recognize_markdown_from_image(&image))
                    .await
                    .with_context(|| {
                        format!("OCR worker task failed for PDF page {page_number}")
                    })??;
            ocr_pages.insert(page_number, markdown);
        }

        let page_count = pdf_page_count(&fast.pages);
        let combined_pages = fast
            .pages
            .iter()
            .enumerate()
            .map(|(index, page)| {
                let page_number = u32::try_from(index + 1).unwrap_or(u32::MAX);
                ocr_pages
                    .remove(&page_number)
                    .unwrap_or_else(|| normalize_pdf_text(page))
            })
            .collect::<Vec<_>>();
        let final_quality = assess_text_quality(&combined_pages);

        if final_quality.score + QUALITY_COMPARISON_EPSILON < fast.quality.score {
            return Ok(None);
        }

        let diagnostics = PipelineDiagnostics {
            pipeline_used: "pdf-embedded-image-ocr".to_string(),
            ocr_used: true,
            render_used: false,
            fallback_used: true,
            fallback_reason: Some(format!(
                "{fallback_reason}; embedded-image OCR repaired {}/{} page(s)",
                ocr_page_count, page_count
            )),
            text_quality_score: Some(final_quality.score),
            latency_ms: start.elapsed().as_millis(),
        };

        let numbered_pages = combined_pages
            .into_iter()
            .enumerate()
            .map(|(index, page)| (u32::try_from(index + 1).unwrap_or(u32::MAX), page))
            .collect();

        Ok(Some(self.build_output_from_pages(
            source,
            numbered_pages,
            diagnostics,
            image_manifest,
        )))
    }

    async fn extract_with_pdfium_fallback(
        &self,
        source: &SourceDescriptor,
        start: Instant,
        fast: &FastPdfExtraction,
        fallback_reason: &str,
        allow_ocr: bool,
        ocr_status: &CapabilityStatus,
    ) -> anyhow::Result<DocumentOutput> {
        let pdfium_pages = self.extract_pdfium_page_text(source).await?;
        anyhow::ensure!(!pdfium_pages.is_empty(), "pdfium returned no pages");

        let pdfium_quality = assess_text_quality(&pdfium_pages);
        let normalized_pdfium_pages = pdfium_pages
            .iter()
            .map(|page| normalize_pdf_text(page))
            .collect::<Vec<_>>();
        let poor_pages = pdfium_quality.poor_page_numbers();

        if allow_ocr && !poor_pages.is_empty() {
            let render_pages = BTreeSet::from_iter(poor_pages.iter().copied())
                .into_iter()
                .collect::<Vec<_>>();
            let rendered_pages = self
                .render_pdfium_page_assets(source, &render_pages)
                .await?;
            anyhow::ensure!(
                !rendered_pages.is_empty(),
                "pdfium render returned no pages for selective OCR"
            );
            let image_manifest =
                image_manifest_from_rendered_pages(&rendered_pages, "pdfium-rendered page image");
            let ocr_pages = self
                .recognize_rendered_pages(
                    rendered_pages
                        .into_iter()
                        .map(|page| (page.page_number, page.image))
                        .collect(),
                )
                .await?;
            let repaired_page_count = ocr_pages.len();
            let mut ocr_pages: HashMap<u32, String> = ocr_pages.into_iter().collect();
            let combined_pages = normalized_pdfium_pages
                .iter()
                .enumerate()
                .map(|(index, page)| {
                    let page_number = u32::try_from(index + 1).unwrap_or(u32::MAX);
                    ocr_pages
                        .remove(&page_number)
                        .unwrap_or_else(|| page.clone())
                })
                .collect::<Vec<_>>();
            let final_quality = assess_text_quality(&combined_pages);

            if final_quality.score + QUALITY_COMPARISON_EPSILON < pdfium_quality.score {
                return Ok(self.build_pdfium_text_output(
                    source,
                    start,
                    normalized_pdfium_pages,
                    &pdfium_quality,
                    format!(
                        "{fallback_reason}; pdfium text quality {:.2}; selective OCR on {} page(s) did not improve quality",
                        pdfium_quality.score,
                        render_pages.len()
                    ),
                ));
            }

            if final_quality.score + QUALITY_COMPARISON_EPSILON < fast.quality.score {
                return Ok(self.build_fast_output(
                    source,
                    start,
                    fast,
                    true,
                    Some(format!(
                        "{fallback_reason}; pdfium selective OCR quality {:.2} did not beat fast-path quality {:.2}",
                        final_quality.score, fast.quality.score
                    )),
                ));
            }

            let diagnostics = PipelineDiagnostics {
                pipeline_used: "pdf-pdfium-ocr".to_string(),
                ocr_used: true,
                render_used: true,
                fallback_used: true,
                fallback_reason: Some(format!(
                    "{fallback_reason}; pdfium text quality {:.2}; selective OCR repaired {}/{} poor page(s)",
                    pdfium_quality.score,
                    repaired_page_count,
                    render_pages.len()
                )),
                text_quality_score: Some(final_quality.score),
                latency_ms: start.elapsed().as_millis(),
            };

            let numbered_pages = combined_pages
                .into_iter()
                .enumerate()
                .map(|(index, page)| (u32::try_from(index + 1).unwrap_or(u32::MAX), page))
                .collect();

            return Ok(self.build_output_from_pages(
                source,
                numbered_pages,
                diagnostics,
                image_manifest,
            ));
        }

        let fallback_reason = if poor_pages.is_empty() {
            format!(
                "{fallback_reason}; pdfium text extraction improved quality to {:.2}",
                pdfium_quality.score
            )
        } else if let Some(ocr_reason) = ocr_status.unavailable_reason() {
            format!(
                "{fallback_reason}; pdfium text quality {:.2} still has poor pages {:?}; OCR unavailable: {ocr_reason}",
                pdfium_quality.score, poor_pages
            )
        } else {
            format!(
                "{fallback_reason}; pdfium text quality {:.2} still has poor pages {:?}",
                pdfium_quality.score, poor_pages
            )
        };

        if pdfium_quality.score + QUALITY_COMPARISON_EPSILON < fast.quality.score {
            return Ok(self.build_fast_output(
                source,
                start,
                fast,
                true,
                Some(format!(
                    "{fallback_reason}; pdfium text quality {:.2} did not improve over fast-path quality {:.2}",
                    pdfium_quality.score, fast.quality.score
                )),
            ));
        }

        Ok(self.build_pdfium_text_output(
            source,
            start,
            normalized_pdfium_pages,
            &pdfium_quality,
            fallback_reason,
        ))
    }

    fn build_pdfium_text_output(
        &self,
        source: &SourceDescriptor,
        start: Instant,
        pages: Vec<String>,
        quality: &TextQuality,
        fallback_reason: String,
    ) -> DocumentOutput {
        let diagnostics = PipelineDiagnostics {
            pipeline_used: "pdf-pdfium".to_string(),
            ocr_used: false,
            render_used: false,
            fallback_used: true,
            fallback_reason: Some(fallback_reason),
            text_quality_score: Some(quality.score),
            latency_ms: start.elapsed().as_millis(),
        };
        let numbered_pages = pages
            .into_iter()
            .enumerate()
            .map(|(index, page)| (u32::try_from(index + 1).unwrap_or(u32::MAX), page))
            .collect();

        self.build_output_from_pages(source, numbered_pages, diagnostics, None)
    }

    pub async fn extract(&self, source: SourceDescriptor) -> anyhow::Result<DocumentOutput> {
        let start = Instant::now();
        let fast = self.extract_fast_text(&source).await?;
        tracing::debug!(
            quality_score = fast.quality.score,
            chars_per_page = fast.quality.chars_per_page,
            median_chars_per_page = fast.quality.median_chars_per_page,
            page_count = fast.quality.page_count,
            empty_page_ratio = fast.quality.empty_page_ratio,
            replacement_ratio = fast.quality.replacement_ratio,
            missing_text_layer = fast.quality.missing_text_layer,
            quality_issues = %fast.quality.issue_summary(),
            "PDF text quality assessment"
        );

        if !fast.quality.needs_escalation() {
            return Ok(self.build_fast_output(&source, start, &fast, false, None));
        }

        // Check pdfium and OCR availability in parallel — both involve potentially
        // expensive I/O (library binding, ONNX model loading) on first call.
        let (pdfium_result, ocr_result) = tokio::join!(
            tokio::task::spawn_blocking(crate::pdfium::ensure_available),
            tokio::task::spawn_blocking(crate::ocr::shared_document_ocr),
        );
        let pdfium_status = match pdfium_result {
            Ok(Ok(())) => CapabilityStatus::Available,
            Ok(Err(error)) => CapabilityStatus::Unavailable(error.to_string()),
            Err(join_error) => CapabilityStatus::Unavailable(join_error.to_string()),
        };
        let ocr_status = match ocr_result {
            Ok(Ok(_)) => CapabilityStatus::Available,
            Ok(Err(error)) => CapabilityStatus::Unavailable(error.to_string()),
            Err(join_error) => CapabilityStatus::Unavailable(join_error.to_string()),
        };

        match plan_escalation(&fast.quality, &pdfium_status, &ocr_status) {
            PdfEscalationPlan::Fast { fallback_reason } => Ok(self.build_fast_output(
                &source,
                start,
                &fast,
                fallback_reason.is_some(),
                fallback_reason,
            )),
            PdfEscalationPlan::Pdfium {
                allow_ocr,
                fallback_reason,
            } => match self
                .extract_with_pdfium_fallback(
                    &source,
                    start,
                    &fast,
                    &fallback_reason,
                    allow_ocr,
                    &ocr_status,
                )
                .await
            {
                Ok(output) => Ok(output),
                Err(error) => {
                    tracing::warn!(err = %error, "pdfium fallback failed");
                    if ocr_status.is_available() {
                        if let Some(output) = self
                            .extract_with_embedded_image_ocr(
                                &source,
                                start,
                                &fast,
                                &format!("{fallback_reason}; pdfium fallback failed: {error}"),
                            )
                            .await?
                        {
                            return Ok(output);
                        }
                    }

                    Ok(self.build_fast_output(
                        &source,
                        start,
                        &fast,
                        true,
                        Some(format!(
                            "{fallback_reason}; pdfium fallback failed: {error}"
                        )),
                    ))
                }
            },
            PdfEscalationPlan::EmbeddedImageOcr { fallback_reason } => {
                if let Some(output) = self
                    .extract_with_embedded_image_ocr(&source, start, &fast, &fallback_reason)
                    .await?
                {
                    Ok(output)
                } else {
                    Ok(self.build_fast_output(
                        &source,
                        start,
                        &fast,
                        true,
                        Some(format!(
                            "{fallback_reason}; embedded-image OCR could not improve the document"
                        )),
                    ))
                }
            }
        }
    }
}

fn pdf_page_count(pages: &[String]) -> u32 {
    u32::try_from(pages.len()).unwrap_or(u32::MAX)
}

fn plan_escalation(
    quality: &TextQuality,
    pdfium_status: &CapabilityStatus,
    ocr_status: &CapabilityStatus,
) -> PdfEscalationPlan {
    if !quality.needs_escalation() {
        return PdfEscalationPlan::Fast {
            fallback_reason: None,
        };
    }

    let trigger_summary = quality.issue_summary();
    let base_reason = format!(
        "fast PDF text quality {:.2} triggered escalation: {trigger_summary}",
        quality.score
    );

    match pdfium_status {
        CapabilityStatus::Available => PdfEscalationPlan::Pdfium {
            allow_ocr: ocr_status.is_available(),
            fallback_reason: base_reason,
        },
        CapabilityStatus::Unavailable(pdfium_reason) => match ocr_status {
            CapabilityStatus::Available => PdfEscalationPlan::EmbeddedImageOcr {
                fallback_reason: format!(
                    "{base_reason}; pdfium unavailable: {pdfium_reason}"
                ),
            },
            CapabilityStatus::Unavailable(ocr_reason) => PdfEscalationPlan::Fast {
                fallback_reason: Some(format!(
                    "{base_reason}; pdfium unavailable: {pdfium_reason}; OCR unavailable: {ocr_reason}"
                )),
            },
        },
    }
}

fn infer_document_title(markdown: &str) -> Option<String> {
    extract_title(markdown).or_else(|| {
        markdown
            .lines()
            .find(|line| {
                let trimmed = line.trim();
                !trimmed.is_empty() && trimmed.len() > 3 && !trimmed.starts_with("<!--")
            })
            .map(|line| line.trim().to_string())
    })
}

fn render_page_markdown(mut pages: Vec<(u32, String)>) -> String {
    pages.sort_by_key(|(page_number, _)| *page_number);
    pages
        .into_iter()
        .map(|(page_number, page_text)| {
            format!("<!-- page {page_number} -->\n{}", page_text.trim())
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn sorted_page_texts(mut pages: Vec<(u32, String)>) -> Vec<String> {
    pages.sort_by_key(|(page_number, _)| *page_number);
    pages.into_iter().map(|(_, text)| text).collect()
}

fn image_manifest_from_rendered_pages(
    pages: &[crate::pdfium::PdfiumRenderedPage],
    label: &str,
) -> Option<Vec<ImageRef>> {
    let manifest = pages
        .iter()
        .map(|page| ImageRef {
            page: page.page_number,
            url: None,
            alt: Some(format!(
                "{label} for page {} ({}x{})",
                page.page_number, page.width, page.height
            )),
        })
        .collect::<Vec<_>>();

    (!manifest.is_empty()).then_some(manifest)
}

fn image_manifest_from_dynamic_images(
    pages: &[(u32, DynamicImage)],
    label: &str,
) -> Option<Vec<ImageRef>> {
    let manifest = pages
        .iter()
        .map(|(page_number, image)| ImageRef {
            page: *page_number,
            url: None,
            alt: Some(format!(
                "{label} for page {} ({}x{})",
                page_number,
                image.width(),
                image.height()
            )),
        })
        .collect::<Vec<_>>();

    (!manifest.is_empty()).then_some(manifest)
}

fn normalize_pdf_text(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '\u{FB00}' => out.push_str("ff"),
            '\u{FB01}' => out.push_str("fi"),
            '\u{FB02}' => out.push_str("fl"),
            '\u{FB03}' => out.push_str("ffi"),
            '\u{FB04}' => out.push_str("ffl"),
            '\u{FB05}' | '\u{FB06}' => out.push_str("st"),
            '\u{2018}' | '\u{2019}' | '\u{201A}' => out.push('\''),
            '\u{201C}' | '\u{201D}' | '\u{201E}' => out.push('"'),
            '\u{2013}' => out.push('-'),
            '\u{2014}' => out.push_str("--"),
            '\u{2012}' | '\u{2015}' => out.push('-'),
            '\u{00A0}' | '\u{2007}' | '\u{202F}' | '\u{2060}' => out.push(' '),
            '\u{00AD}' | '\u{200B}' | '\u{200C}' | '\u{200D}' | '\u{FEFF}' => {}
            '\u{2022}' => out.push_str("* "),
            '\u{2026}' => out.push_str("..."),
            '\u{00B7}' => out.push('.'),
            '\u{2212}' => out.push('-'),
            '\u{00D7}' => out.push('x'),
            '\u{2032}' => out.push('\''),
            '\u{2033}' => out.push('"'),
            '\u{00C0}'..='\u{00C5}' => out.push('A'),
            '\u{00C7}' => out.push('C'),
            '\u{00C8}'..='\u{00CB}' => out.push('E'),
            '\u{00CC}'..='\u{00CF}' => out.push('I'),
            '\u{00D1}' => out.push('N'),
            '\u{00D2}'..='\u{00D6}' | '\u{00D8}' => out.push('O'),
            '\u{00D9}'..='\u{00DC}' => out.push('U'),
            '\u{00DD}' => out.push('Y'),
            '\u{00E0}'..='\u{00E5}' => out.push('a'),
            '\u{00E7}' => out.push('c'),
            '\u{00E8}'..='\u{00EB}' => out.push('e'),
            '\u{00EC}'..='\u{00EF}' => out.push('i'),
            '\u{00F1}' => out.push('n'),
            '\u{00F2}'..='\u{00F6}' | '\u{00F8}' => out.push('o'),
            '\u{00F9}'..='\u{00FC}' => out.push('u'),
            '\u{00FD}' | '\u{00FF}' => out.push('y'),
            '\u{00DF}' => out.push_str("ss"),
            '\u{0300}'..='\u{036F}' => {}
            '\u{00B4}' | '\u{0060}' | '\u{02CA}' | '\u{02CB}' => {}
            '\u{00A8}' | '\u{02C6}' | '\u{02DC}' => {}
            '\u{00A9}' => out.push_str("(c)"),
            '\u{00AE}' => out.push_str("(R)"),
            '\u{2122}' => out.push_str("(TM)"),
            c if !c.is_ascii() => out.push('?'),
            c => out.push(c),
        }
    }
    out
}

fn decode_embedded_pdf_image(
    image: &pdf_extract::xobject::PdfImage<'_>,
) -> anyhow::Result<DynamicImage> {
    if let Ok(decoded) = image::load_from_memory(image.content) {
        return Ok(decoded);
    }

    let width = u32::try_from(image.width).context("PDF image width is out of range")?;
    let height = u32::try_from(image.height).context("PDF image height is out of range")?;
    let bits_per_component = image.bits_per_component.unwrap_or(8);

    anyhow::ensure!(
        bits_per_component == 8,
        "unsupported PDF image bit depth: {bits_per_component}"
    );

    let stream = pdf_extract::Stream {
        dict: image.origin_dict.clone(),
        content: image.content.to_vec(),
        allows_compression: true,
        start_position: None,
    };
    let bytes = stream
        .get_plain_content()
        .context("failed to decode embedded PDF image stream")?;

    match image.color_space.as_deref() {
        Some("DeviceGray") => {
            let buffer = ImageBuffer::<Luma<u8>, _>::from_raw(width, height, bytes)
                .context("embedded grayscale PDF image buffer size mismatch")?;
            Ok(DynamicImage::ImageLuma8(buffer))
        }
        Some("DeviceRGB") => {
            let buffer = ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, bytes)
                .context("embedded RGB PDF image buffer size mismatch")?;
            Ok(DynamicImage::ImageRgb8(buffer))
        }
        other => anyhow::bail!("unsupported embedded PDF image color space: {:?}", other),
    }
}

fn assess_page_text_quality(page_number: u32, page: &str) -> PageTextQuality {
    let char_count = page.trim().chars().count();
    let total_chars = page.chars().count();
    let replacement_chars = page.chars().filter(|ch| *ch == '\u{FFFD}').count();
    let replacement_ratio = if total_chars > 0 {
        replacement_chars as f32 / total_chars as f32
    } else {
        0.0
    };

    let mut score = 1.0f32;
    let mut triggers = Vec::new();

    if char_count < EMPTY_PAGE_CHAR_THRESHOLD {
        score -= 0.5;
        triggers.push(QualityTrigger::LowCharsPerPage);
    }
    if replacement_ratio > HIGH_REPLACEMENT_RATIO_THRESHOLD {
        score -= 0.25;
        triggers.push(QualityTrigger::ReplacementCharacterIssues);
    }
    if char_count == 0 {
        score -= 0.25;
        triggers.push(QualityTrigger::MissingTextLayer);
    }

    PageTextQuality {
        page_number,
        score: score.max(0.0),
        char_count,
        replacement_ratio,
        triggers,
    }
}

fn assess_text_quality(pages: &[String]) -> TextQuality {
    let page_count = pdf_page_count(pages);
    let page_assessments = pages
        .iter()
        .enumerate()
        .map(|(index, page)| {
            assess_page_text_quality(u32::try_from(index + 1).unwrap_or(u32::MAX), page)
        })
        .collect::<Vec<_>>();
    let total_chars: usize = page_assessments.iter().map(|page| page.char_count).sum();
    let chars_per_page = if page_count > 0 {
        total_chars as f32 / page_count as f32
    } else {
        0.0
    };

    let mut char_counts = page_assessments
        .iter()
        .map(|page| page.char_count)
        .collect::<Vec<_>>();
    char_counts.sort_unstable();
    let median_chars_per_page = median_usize(&char_counts);

    let empty_pages = page_assessments
        .iter()
        .filter(|page| page.char_count < EMPTY_PAGE_CHAR_THRESHOLD)
        .count();
    let empty_page_ratio = if page_count > 0 {
        empty_pages as f32 / page_count as f32
    } else {
        1.0
    };

    let replacement_chars: usize = pages
        .iter()
        .map(|page| page.chars().filter(|ch| *ch == '\u{FFFD}').count())
        .sum();
    let total_raw_chars: usize = pages.iter().map(|page| page.chars().count()).sum();
    let replacement_ratio = if total_raw_chars > 0 {
        replacement_chars as f32 / total_raw_chars as f32
    } else {
        0.0
    };
    let missing_text_layer = total_chars == 0;

    let mut score = 1.0f32;
    let mut triggers = Vec::new();

    if median_chars_per_page < LOW_CHARS_PER_PAGE_THRESHOLD {
        score -= 0.4;
        triggers.push(QualityTrigger::LowCharsPerPage);
    }
    if empty_page_ratio > HIGH_EMPTY_PAGE_RATIO_THRESHOLD {
        score -= 0.25;
        triggers.push(QualityTrigger::HighEmptyPageRatio);
    }
    if replacement_ratio > HIGH_REPLACEMENT_RATIO_THRESHOLD {
        score -= 0.2;
        triggers.push(QualityTrigger::ReplacementCharacterIssues);
    }
    if missing_text_layer {
        score -= 0.5;
        triggers.push(QualityTrigger::MissingTextLayer);
    }

    TextQuality {
        score: score.max(0.0),
        chars_per_page,
        median_chars_per_page,
        page_count,
        empty_page_ratio,
        replacement_ratio,
        missing_text_layer,
        triggers,
        page_assessments,
    }
}

fn median_usize(values: &[usize]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }

    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] as f32 + values[middle] as f32) / 2.0
    } else {
        values[middle] as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;

    #[tokio::test]
    async fn test_pdf_extract_from_test_file() {
        let pdf_bytes = std::fs::read("tests/test.pdf").expect("tests/test.pdf should exist");
        let source = SourceDescriptor {
            canonical_url: Some("file:///test.pdf".to_string()),
            source_kind: crate::resolver::SourceKind::Pdf,
            mime: "application/pdf".to_string(),
            filename: Some("test.pdf".to_string()),
            raw_bytes: Bytes::from(pdf_bytes),
        };

        let pipeline = PdfPipeline;
        let result = pipeline
            .extract(source)
            .await
            .expect("extraction should succeed");

        assert!(!result.markdown.is_empty(), "markdown should not be empty");
        assert!(!result.chunks.is_empty(), "should have at least one chunk");
        assert_eq!(result.diagnostics.pipeline_used, "pdf-fast");
        assert!(!result.diagnostics.ocr_used);
        assert!(!result.diagnostics.render_used);
        assert!(!result.diagnostics.fallback_used);
        assert!(result.diagnostics.fallback_reason.is_none());
        assert!(result.diagnostics.text_quality_score.is_some());
        assert!(result.metadata.page_count.is_some());
        assert!(result.metadata.word_count > 0);
    }

    #[test]
    fn test_assess_quality_good_text() {
        let pages = vec![
            "This is a page with substantial text content that should be considered good quality."
                .to_string(),
            "Another page with reasonable content for quality assessment purposes.".to_string(),
        ];
        let quality = assess_text_quality(&pages);
        assert!(quality.score > 0.5);
        assert!(quality.chars_per_page > 50.0);
        assert!(quality.empty_page_ratio < 0.3);
        assert!(quality.triggers.is_empty());
    }

    #[test]
    fn test_assess_quality_empty_pages_flags_missing_text_layer() {
        let pages = vec!["".to_string(), "".to_string(), "".to_string()];
        let quality = assess_text_quality(&pages);
        assert!(quality.score < 0.5);
        assert!(quality.empty_page_ratio > 0.9);
        assert!(quality.missing_text_layer);
        assert!(quality.triggers.contains(&QualityTrigger::MissingTextLayer));
        assert!(quality
            .triggers
            .contains(&QualityTrigger::HighEmptyPageRatio));
    }

    #[test]
    fn test_assess_quality_flags_replacement_char_issues() {
        let pages = vec!["valid text \u{FFFD}\u{FFFD}\u{FFFD} more text".to_string()];
        let quality = assess_text_quality(&pages);
        assert!(quality
            .triggers
            .contains(&QualityTrigger::ReplacementCharacterIssues));
        assert!(quality.replacement_ratio > HIGH_REPLACEMENT_RATIO_THRESHOLD);
    }

    #[test]
    fn test_poor_page_numbers_select_only_bad_pages() {
        let pages = vec![
            "This page has plenty of text and should stay as text.".repeat(3),
            "".to_string(),
            "Tiny".to_string(),
        ];
        let quality = assess_text_quality(&pages);
        assert_eq!(quality.poor_page_numbers(), vec![2, 3]);
        assert!(quality.page_assessments[1].replacement_ratio <= 1.0);
    }

    #[test]
    fn test_plan_escalation_prefers_pdfium_when_available() {
        let quality = TextQuality {
            score: 0.1,
            chars_per_page: 0.0,
            median_chars_per_page: 0.0,
            page_count: 1,
            empty_page_ratio: 1.0,
            replacement_ratio: 0.0,
            missing_text_layer: true,
            triggers: vec![QualityTrigger::MissingTextLayer],
            page_assessments: vec![PageTextQuality {
                page_number: 1,
                score: 0.0,
                char_count: 0,
                replacement_ratio: 0.0,
                triggers: vec![QualityTrigger::MissingTextLayer],
            }],
        };

        let route = plan_escalation(
            &quality,
            &CapabilityStatus::Available,
            &CapabilityStatus::Available,
        );
        assert_eq!(
            route,
            PdfEscalationPlan::Pdfium {
                allow_ocr: true,
                fallback_reason:
                    "fast PDF text quality 0.10 triggered escalation: missing text layer"
                        .to_string(),
            }
        );
    }

    #[test]
    fn test_plan_escalation_uses_embedded_image_ocr_when_pdfium_unavailable() {
        let quality = TextQuality {
            score: 0.1,
            chars_per_page: 0.0,
            median_chars_per_page: 0.0,
            page_count: 1,
            empty_page_ratio: 1.0,
            replacement_ratio: 0.0,
            missing_text_layer: true,
            triggers: vec![QualityTrigger::MissingTextLayer],
            page_assessments: vec![PageTextQuality {
                page_number: 1,
                score: 0.0,
                char_count: 0,
                replacement_ratio: 0.0,
                triggers: vec![QualityTrigger::MissingTextLayer],
            }],
        };

        let route = plan_escalation(
            &quality,
            &CapabilityStatus::Unavailable("pdfium missing".to_string()),
            &CapabilityStatus::Available,
        );
        assert_eq!(
            route,
            PdfEscalationPlan::EmbeddedImageOcr {
                fallback_reason: "fast PDF text quality 0.10 triggered escalation: missing text layer; pdfium unavailable: pdfium missing".to_string(),
            }
        );
    }

    #[test]
    fn test_plan_escalation_returns_fast_when_all_fallbacks_unavailable() {
        let quality = TextQuality {
            score: 0.1,
            chars_per_page: 0.0,
            median_chars_per_page: 0.0,
            page_count: 1,
            empty_page_ratio: 1.0,
            replacement_ratio: 0.0,
            missing_text_layer: true,
            triggers: vec![QualityTrigger::MissingTextLayer],
            page_assessments: vec![PageTextQuality {
                page_number: 1,
                score: 0.0,
                char_count: 0,
                replacement_ratio: 0.0,
                triggers: vec![QualityTrigger::MissingTextLayer],
            }],
        };

        let route = plan_escalation(
            &quality,
            &CapabilityStatus::Unavailable("pdfium missing".to_string()),
            &CapabilityStatus::Unavailable("OCR missing".to_string()),
        );
        assert_eq!(
            route,
            PdfEscalationPlan::Fast {
                fallback_reason: Some("fast PDF text quality 0.10 triggered escalation: missing text layer; pdfium unavailable: pdfium missing; OCR unavailable: OCR missing".to_string()),
            }
        );
    }

    #[test]
    fn test_render_page_markdown_orders_pages_by_page_number() {
        let raw_text = render_page_markdown(vec![
            (2, "Second page".to_string()),
            (1, "First page".to_string()),
        ]);

        assert_eq!(
            raw_text,
            "<!-- page 1 -->\nFirst page\n\n<!-- page 2 -->\nSecond page"
        );
    }
}
