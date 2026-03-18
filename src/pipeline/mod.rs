pub mod image;
pub mod markdown;
pub mod pdf;
pub mod web;

use crate::chunker::DocumentOutput;
use crate::resolver::{SourceDescriptor, SourceKind};

/// Dispatch a source descriptor to the appropriate pipeline.
pub async fn dispatch(source: SourceDescriptor) -> anyhow::Result<DocumentOutput> {
    match source.source_kind {
        SourceKind::Web => {
            let pipeline = web::WebPipeline;
            pipeline.extract(source).await
        }
        SourceKind::Pdf => {
            let pipeline = pdf::PdfPipeline;
            pipeline.extract(source).await
        }
        SourceKind::Markdown | SourceKind::PlainText => {
            let pipeline = markdown::MarkdownPipeline;
            pipeline.extract(source).await
        }
        SourceKind::Image => {
            let pipeline = image::ImagePipeline;
            pipeline.extract(source).await
        }
    }
}
