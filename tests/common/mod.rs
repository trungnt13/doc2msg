#![allow(dead_code)]

use std::env;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::body::Body;
use base64::{engine::general_purpose::STANDARD, Engine as _};
use doc2agent::config::RuntimeConfig;
use doc2agent::server::{build_router, AppState};
use image::{DynamicImage, ImageBuffer, ImageFormat, Rgb};

/// Create a default RuntimeConfig for testing with OCR disabled and minimal resource usage.
pub fn test_config() -> RuntimeConfig {
    RuntimeConfig {
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
    }
}

/// Build an axum Router backed by a custom RuntimeConfig for testing.
pub fn create_test_app_with_config(config: RuntimeConfig) -> axum::Router {
    let state = Arc::new(AppState::new(config).expect("test app state"));
    build_router(state)
}

/// Build an axum Router backed by the default test config.
pub fn create_test_app() -> axum::Router {
    create_test_app_with_config(test_config())
}

/// Parse an NDJSON response body into a vector of JSON values.
pub async fn parse_ndjson(response: axum::http::Response<Body>) -> Vec<serde_json::Value> {
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

/// Encode a solid-color PNG image as base64 for OCR testing.
pub fn encode_png_base64(color: [u8; 3]) -> String {
    let image = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(160, 48, Rgb(color)));
    let mut cursor = std::io::Cursor::new(Vec::new());
    image.write_to(&mut cursor, ImageFormat::Png).unwrap();
    STANDARD.encode(cursor.into_inner())
}

/// Resolve a file path from environment variables or a repo-relative fallback.
/// Checks each env var in order; if none are set or point to existing files,
/// falls back to `CARGO_MANIFEST_DIR/repo_relative`.
pub fn resolve_existing_path(env_keys: &[&str], repo_relative: &str) -> Option<PathBuf> {
    env_keys
        .iter()
        .filter_map(|key| env::var_os(key).map(PathBuf::from))
        .find(|path| path.is_file())
        .or_else(|| {
            let fallback = Path::new(env!("CARGO_MANIFEST_DIR")).join(repo_relative);
            fallback.is_file().then_some(fallback)
        })
}

/// Check whether all OCR assets (detector, recognizer, dictionary) are available locally.
pub fn local_full_ocr_assets_available() -> bool {
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
