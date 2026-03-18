use std::env;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Context;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use bytes::Bytes;
use doc2agent::config::RuntimeConfig;
use doc2agent::pdfium;
use doc2agent::pipeline::pdf::PdfPipeline;
use doc2agent::resolver::{SourceDescriptor, SourceKind};
use doc2agent::server::{build_router, AppState};
use image::{DynamicImage, ImageBuffer, ImageFormat, Rgb};
use tower::ServiceExt;

fn create_test_app() -> axum::Router {
    let config = RuntimeConfig {
        host: "127.0.0.1".to_string(),
        port: 0,
        request_timeout_secs: 30,
        max_request_body_bytes: 50 * 1024 * 1024,
        extraction_concurrency: 4,
        ocr_concurrency: 1,
        per_host_fetch_concurrency: None,
        model_path: None,
        dict_path: None,
        pdfium_enabled: false,
        pdfium_lib_path: None,
        session_pool_size: 1,
        max_batch: 1,
        intra_threads: 1,
        inter_threads: 1,
        device_id: -1,
    };
    let state = Arc::new(AppState::new(config).expect("test app state"));
    build_router(state)
}

/// Helper: parse NDJSON response body into a vec of serde_json::Value.
async fn parse_ndjson(response: axum::http::Response<Body>) -> Vec<serde_json::Value> {
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body_str = String::from_utf8(body.to_vec()).unwrap();
    body_str
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

fn local_full_ocr_assets_available() -> bool {
    resolve_existing_path(
        &["DOC2AGENT_DET_MODEL", "DOC2AGENT_TEST_DET_MODEL"],
        "models/det_model.onnx",
    )
    .is_some()
        && resolve_existing_path(
            &[
                "DOC2AGENT_MODEL_PATH",
                "DOC2AGENT_REC_MODEL",
                "DOC2AGENT_TEST_REC_MODEL",
            ],
            "models/rec_model.onnx",
        )
        .is_some()
        && resolve_existing_path(
            &["DOC2AGENT_DICT_PATH", "DOC2AGENT_TEST_REC_DICT"],
            "models/ppocr_keys_v1.txt",
        )
        .is_some()
}

fn local_pdfium_available() -> anyhow::Result<()> {
    pdfium::ensure_available().with_context(|| {
        "pdfium is unavailable for tests; set DOC2AGENT_PDFIUM_ENABLED=1 and optionally DOC2AGENT_PDFIUM_LIB_PATH=/path/to/libpdfium"
    })
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

fn build_image_only_pdf() -> anyhow::Result<Vec<u8>> {
    use pdf_extract::content::{Content, Operation};
    use pdf_extract::{dictionary, Document, Object, Stream};

    let image = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(320, 120, Rgb([255, 255, 255])));
    let mut encoded = Cursor::new(Vec::new());
    image.write_to(&mut encoded, ImageFormat::Jpeg)?;

    let mut doc = Document::with_version("1.5");
    let pages_id = doc.new_object_id();
    let image_id = doc.add_object(Stream::new(
        dictionary! {
            "Type" => "XObject",
            "Subtype" => "Image",
            "Width" => 320,
            "Height" => 120,
            "ColorSpace" => "DeviceRGB",
            "BitsPerComponent" => 8,
            "Filter" => "DCTDecode",
        },
        encoded.into_inner(),
    ));
    let content = Content {
        operations: vec![
            Operation::new("q", vec![]),
            Operation::new(
                "cm",
                vec![
                    595.into(),
                    0.into(),
                    0.into(),
                    842.into(),
                    0.into(),
                    0.into(),
                ],
            ),
            Operation::new("Do", vec![Object::Name(b"Im1".to_vec())]),
            Operation::new("Q", vec![]),
        ],
    };
    let content_id = doc.add_object(Stream::new(dictionary! {}, content.encode()?));
    let page_id = doc.add_object(dictionary! {
        "Type" => "Page",
        "Parent" => pages_id,
        "Contents" => content_id,
        "MediaBox" => vec![0.into(), 0.into(), 595.into(), 842.into()],
    });
    doc.add_xobject(page_id, b"Im1", image_id)?;

    let pages = dictionary! {
        "Type" => "Pages",
        "Kids" => vec![page_id.into()],
        "Count" => 1,
        "MediaBox" => vec![0.into(), 0.into(), 595.into(), 842.into()],
    };
    doc.objects.insert(pages_id, Object::Dictionary(pages));

    let catalog_id = doc.add_object(dictionary! {
        "Type" => "Catalog",
        "Pages" => pages_id,
    });
    doc.trailer.set("Root", catalog_id);

    let mut pdf = Vec::new();
    doc.save_to(&mut pdf)?;
    Ok(pdf)
}

// -----------------------------------------------------------------------
// PDF extraction via /v1/extract/bytes
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_pdf_extract_bytes() {
    let app = create_test_app();
    let pdf_bytes = std::fs::read("tests/test.pdf").expect("test.pdf should exist");

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/extract/bytes")
                .header("content-type", "application/pdf")
                .body(Body::from(pdf_bytes))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let events = parse_ndjson(response).await;
    assert!(
        events.len() >= 3,
        "should have metadata + chunk(s) + done, got {}",
        events.len()
    );

    // First event: metadata
    let metadata = &events[0];
    assert_eq!(metadata["event"], "metadata");
    assert_eq!(metadata["source_kind"], "pdf-fast");
    assert_eq!(metadata["fallback_used"], false);
    assert_eq!(metadata["ocr_used"], false);
    assert!(
        metadata["page_count"].as_u64().unwrap() > 0,
        "PDF should report page count"
    );

    // Middle events: chunks
    for event in &events[1..events.len() - 1] {
        assert_eq!(event["event"], "chunk");
        assert!(event["id"].is_string());
        assert!(!event["text"].as_str().unwrap().is_empty());
        assert!(event["token_estimate"].as_u64().unwrap() > 0);
    }

    // Last event: done
    let done = events.last().unwrap();
    assert_eq!(done["event"], "done");
    assert!(done["chunks_total"].as_u64().unwrap() > 0);
    assert_eq!(done["pipeline_used"], "pdf-fast");
    assert_eq!(done["fallback_used"], false);
    assert_eq!(done["ocr_used"], false);
}

// -----------------------------------------------------------------------
// PDF extraction produces meaningful text
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_pdf_extract_has_content() {
    let app = create_test_app();
    let pdf_bytes = std::fs::read("tests/test.pdf").expect("test.pdf should exist");

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/extract/bytes")
                .header("content-type", "application/pdf")
                .body(Body::from(pdf_bytes))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let events = parse_ndjson(response).await;
    let total_text_len: usize = events
        .iter()
        .filter(|e| e["event"] == "chunk")
        .map(|e| e["text"].as_str().unwrap().len())
        .sum();

    assert!(
        total_text_len > 50,
        "PDF extraction should produce meaningful text, got {} chars",
        total_text_len
    );
}

// -----------------------------------------------------------------------
// PDF MIME auto-detection via magic bytes (no content-type header)
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_pdf_auto_detect_mime() {
    let app = create_test_app();
    let pdf_bytes = std::fs::read("tests/test.pdf").expect("test.pdf should exist");

    // Send without content-type — resolver should sniff %PDF magic bytes
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/extract/bytes")
                .header("content-type", "application/octet-stream")
                .body(Body::from(pdf_bytes))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let events = parse_ndjson(response).await;
    assert_eq!(events[0]["event"], "metadata");
    assert_eq!(events[0]["source_kind"], "pdf-fast");
    assert_eq!(events[0]["fallback_used"], false);
}

#[tokio::test]
async fn poor_quality_pdf_routes_to_best_available_ocr_fallback_when_assets_are_available(
) -> anyhow::Result<()> {
    if !local_full_ocr_assets_available() {
        eprintln!(
            "skipping PDF OCR fallback integration test; provide detector, recognizer, and dictionary assets in models/ or via DOC2AGENT_* env vars"
        );
        return Ok(());
    }

    let source = SourceDescriptor {
        canonical_url: Some("file:///scanned.pdf".to_string()),
        source_kind: SourceKind::Pdf,
        mime: "application/pdf".to_string(),
        filename: Some("scanned.pdf".to_string()),
        raw_bytes: Bytes::from(build_image_only_pdf()?),
    };

    let result = PdfPipeline.extract(source).await?;

    let pdfium_available = local_pdfium_available().is_ok();

    assert_eq!(
        result.diagnostics.pipeline_used,
        if pdfium_available {
            "pdf-pdfium-ocr"
        } else {
            "pdf-embedded-image-ocr"
        }
    );
    assert!(result.diagnostics.ocr_used);
    assert_eq!(result.diagnostics.render_used, pdfium_available);
    assert!(result.diagnostics.fallback_used);
    assert!(result.diagnostics.text_quality_score.is_some());
    assert!(result
        .image_manifest
        .as_ref()
        .map(|manifest| !manifest.is_empty())
        .unwrap_or(false));
    assert!(result
        .diagnostics
        .fallback_reason
        .as_deref()
        .unwrap_or_default()
        .contains(if pdfium_available {
            "selective OCR"
        } else {
            "embedded-image OCR"
        }));

    Ok(())
}

#[tokio::test]
async fn pdfium_extracts_page_text_when_available() -> anyhow::Result<()> {
    if let Err(error) = local_pdfium_available() {
        eprintln!("skipping pdfium text extraction test: {error}");
        return Ok(());
    }

    let source = SourceDescriptor {
        canonical_url: Some("file:///test.pdf".to_string()),
        source_kind: SourceKind::Pdf,
        mime: "application/pdf".to_string(),
        filename: Some("test.pdf".to_string()),
        raw_bytes: Bytes::from(std::fs::read("tests/test.pdf")?),
    };

    let pages = PdfPipeline.extract_pdfium_page_text(&source).await?;
    let total_chars: usize = pages.iter().map(|page| page.trim().len()).sum();

    assert!(!pages.is_empty(), "pdfium should return at least one page");
    assert!(
        total_chars > 50,
        "pdfium page text should contain meaningful content, got {total_chars} chars"
    );

    Ok(())
}

#[tokio::test]
async fn pdfium_renders_pages_when_available() -> anyhow::Result<()> {
    if let Err(error) = local_pdfium_available() {
        eprintln!("skipping pdfium render test: {error}");
        return Ok(());
    }

    let source = SourceDescriptor {
        canonical_url: Some("file:///test.pdf".to_string()),
        source_kind: SourceKind::Pdf,
        mime: "application/pdf".to_string(),
        filename: Some("test.pdf".to_string()),
        raw_bytes: Bytes::from(std::fs::read("tests/test.pdf")?),
    };

    let pages = PdfPipeline.render_pdfium_pages(&source).await?;

    assert!(!pages.is_empty(), "pdfium should render at least one page");
    assert!(
        pages
            .iter()
            .all(|(_, image)| image.width() > 0 && image.height() > 0),
        "rendered pdfium images should have non-zero dimensions"
    );

    Ok(())
}

#[tokio::test]
async fn pdfium_page_ocr_runs_when_pdfium_and_models_are_available() -> anyhow::Result<()> {
    if let Err(error) = local_pdfium_available() {
        eprintln!("skipping pdfium page OCR test: {error}");
        return Ok(());
    }
    if !local_full_ocr_assets_available() {
        eprintln!(
            "skipping pdfium page OCR test; provide detector, recognizer, and dictionary assets in models/ or via DOC2AGENT_* env vars"
        );
        return Ok(());
    }

    let source = SourceDescriptor {
        canonical_url: Some("file:///test.pdf".to_string()),
        source_kind: SourceKind::Pdf,
        mime: "application/pdf".to_string(),
        filename: Some("test.pdf".to_string()),
        raw_bytes: Bytes::from(std::fs::read("tests/test.pdf")?),
    };

    let result = PdfPipeline.extract_with_page_ocr(&source).await?;

    assert_eq!(result.diagnostics.pipeline_used, "pdf-page-ocr");
    assert!(result.diagnostics.ocr_used);
    assert!(result.diagnostics.render_used);
    assert!(!result.diagnostics.fallback_used);
    assert!(result.diagnostics.fallback_reason.is_none());
    assert!(result.diagnostics.text_quality_score.is_some());
    assert!(result
        .image_manifest
        .as_ref()
        .map(|manifest| !manifest.is_empty())
        .unwrap_or(false));
    assert!(
        result.metadata.page_count.unwrap_or_default() > 0,
        "page OCR should retain PDF page count metadata"
    );
    assert!(
        result.markdown.contains("<!-- page 1 -->"),
        "page OCR output should retain page markers"
    );

    Ok(())
}
