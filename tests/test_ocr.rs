use std::env;
use std::path::PathBuf;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use doc2agent::config::RuntimeConfig;
use doc2agent::server::{build_router, AppState};
use image::{DynamicImage, ImageBuffer, ImageFormat, Rgb};
use tower::ServiceExt;

fn local_model_paths() -> Option<(PathBuf, PathBuf)> {
    let model_path = env::var_os("DOC2AGENT_TEST_REC_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/rec_model.onnx"));
    let dict_path = env::var_os("DOC2AGENT_TEST_REC_DICT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/ppocr_keys_v1.txt"));

    if model_path.is_file() && dict_path.is_file() {
        Some((model_path, dict_path))
    } else {
        None
    }
}

fn create_test_app(model_path: String, dict_path: String) -> axum::Router {
    let config = RuntimeConfig {
        host: "127.0.0.1".to_string(),
        port: 0,
        request_timeout_secs: 30,
        max_request_body_bytes: 50 * 1024 * 1024,
        extraction_concurrency: 4,
        ocr_concurrency: 1,
        per_host_fetch_concurrency: None,
        model_path: Some(model_path),
        dict_path: Some(dict_path),
        pdfium_enabled: false,
        pdfium_lib_path: None,
        session_pool_size: 1,
        max_batch: 2,
        intra_threads: 1,
        inter_threads: 1,
        device_id: -1,
    };
    let state = Arc::new(AppState::new(config).expect("test app state"));
    build_router(state)
}

fn encode_png_base64(color: [u8; 3]) -> String {
    let image = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(160, 48, Rgb(color)));
    let mut cursor = std::io::Cursor::new(Vec::new());
    image.write_to(&mut cursor, ImageFormat::Png).unwrap();
    STANDARD.encode(cursor.into_inner())
}

#[tokio::test]
async fn ocr_endpoint_runs_when_local_models_are_available() {
    let Some((model_path, dict_path)) = local_model_paths() else {
        eprintln!(
            "skipping OCR endpoint integration test; provide local model assets at models/ or via DOC2AGENT_TEST_REC_MODEL and DOC2AGENT_TEST_REC_DICT"
        );
        return;
    };

    let app = create_test_app(
        model_path.to_string_lossy().into_owned(),
        dict_path.to_string_lossy().into_owned(),
    );
    let raw_base64 = encode_png_base64([255, 255, 255]);
    let body = serde_json::json!({
        "images": [
            raw_base64,
            format!("data:image/png;base64,{}", encode_png_base64([0, 0, 0]))
        ]
    });

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/ocr")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["model"], "openocr-repsvtr");
    assert_eq!(json["batch_size"], 2);
    assert!(json["latency_ms"].as_u64().is_some());

    let items = json["items"].as_array().unwrap();
    assert_eq!(items.len(), 2);
    for item in items {
        assert!(item["text"].is_string());
        let confidence = item["confidence"].as_f64().unwrap();
        assert!(confidence.is_finite());
        assert!(confidence >= 0.0);
    }

    let health_response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(health_response.status(), StatusCode::OK);

    let health_body = axum::body::to_bytes(health_response.into_body(), usize::MAX)
        .await
        .unwrap();
    let health_json: serde_json::Value = serde_json::from_slice(&health_body).unwrap();
    assert_eq!(health_json["ocr_available"], true);
}
