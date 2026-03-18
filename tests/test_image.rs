use std::env;
use std::io::Cursor;
use std::path::{Path, PathBuf};

use anyhow::Result;
use bytes::Bytes;
use doc2msg::pipeline;
use doc2msg::pipeline::image::ImagePipeline;
use doc2msg::resolver::{SourceDescriptor, SourceKind};
use image::{DynamicImage, ImageBuffer, ImageFormat, Rgb};

fn local_full_ocr_assets_available() -> bool {
    resolve_existing_path(
        &["DOC2MSG_DET_MODEL", "DOC2MSG_TEST_DET_MODEL"],
        "models/det_model.onnx",
    )
    .is_some()
        && resolve_existing_path(
            &[
                "DOC2MSG_MODEL_PATH",
                "DOC2MSG_REC_MODEL",
                "DOC2MSG_TEST_REC_MODEL",
            ],
            "models/rec_model.onnx",
        )
        .is_some()
        && resolve_existing_path(
            &["DOC2MSG_DICT_PATH", "DOC2MSG_TEST_REC_DICT"],
            "models/ppocr_keys_v1.txt",
        )
        .is_some()
}

fn resolve_existing_path(env_keys: &[&str], repo_relative: &str) -> Option<PathBuf> {
    env_keys
        .iter()
        .filter_map(|key| env::var_os(key).map(PathBuf::from))
        .find(|path| path.is_file())
        .or_else(|| {
            let fallback = Path::new(env!("CARGO_MANIFEST_DIR")).join(repo_relative);
            fallback.is_file().then_some(fallback)
        })
}

#[tokio::test]
async fn image_pipeline_runs_when_local_models_are_available() -> Result<()> {
    if !local_full_ocr_assets_available() {
        eprintln!(
            "skipping full-image OCR integration test; provide detector, recognizer, and dictionary assets in models/ or via DOC2MSG_* env vars"
        );
        return Ok(());
    }

    let image = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(320, 120, Rgb([255, 255, 255])));
    let mut encoded = Cursor::new(Vec::new());
    image.write_to(&mut encoded, ImageFormat::Png)?;

    let source = SourceDescriptor {
        canonical_url: Some("file:///blank.png".to_string()),
        source_kind: SourceKind::Image,
        mime: "image/png".to_string(),
        filename: Some("blank.png".to_string()),
        raw_bytes: Bytes::from(encoded.into_inner()),
    };

    let pipeline = ImagePipeline;
    let result = pipeline.extract(source).await?;

    assert_eq!(result.diagnostics.pipeline_used, "image-ocr");
    assert!(result.diagnostics.ocr_used);
    assert!(!result.diagnostics.render_used);
    assert_eq!(result.metadata.page_count, Some(1));

    Ok(())
}

#[tokio::test]
async fn dispatch_routes_image_sources_through_image_ocr_pipeline_when_local_models_are_available(
) -> Result<()> {
    if !local_full_ocr_assets_available() {
        eprintln!(
            "skipping image dispatch integration test; provide detector, recognizer, and dictionary assets in models/ or via DOC2MSG_* env vars"
        );
        return Ok(());
    }

    let image = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(320, 120, Rgb([255, 255, 255])));
    let mut encoded = Cursor::new(Vec::new());
    image.write_to(&mut encoded, ImageFormat::Png)?;

    let source = SourceDescriptor {
        canonical_url: Some("file:///dispatch-blank.png".to_string()),
        source_kind: SourceKind::Image,
        mime: "image/png".to_string(),
        filename: Some("dispatch-blank.png".to_string()),
        raw_bytes: Bytes::from(encoded.into_inner()),
    };

    let result = pipeline::dispatch(source).await?;

    assert_eq!(result.diagnostics.pipeline_used, "image-ocr");
    assert!(result.diagnostics.ocr_used);
    assert!(!result.diagnostics.render_used);
    assert!(!result.diagnostics.fallback_used);
    assert!(result.diagnostics.fallback_reason.is_none());

    Ok(())
}
