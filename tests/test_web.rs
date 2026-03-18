use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use doc2agent::config::RuntimeConfig;
use doc2agent::server::{build_router, AppState};
use image::{DynamicImage, ImageBuffer, ImageFormat, Rgb};
use tower::ServiceExt;

fn test_config() -> RuntimeConfig {
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

fn create_test_app_with_config(config: RuntimeConfig) -> axum::Router {
    let state = Arc::new(AppState::new(config));
    build_router(state)
}

fn create_test_app() -> axum::Router {
    create_test_app_with_config(test_config())
}

fn encode_png_base64(color: [u8; 3]) -> String {
    let image = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(160, 48, Rgb(color)));
    let mut cursor = std::io::Cursor::new(Vec::new());
    image.write_to(&mut cursor, ImageFormat::Png).unwrap();
    STANDARD.encode(cursor.into_inner())
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

// -----------------------------------------------------------------------
// Health endpoint
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_health_endpoint() {
    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["status"], "ok");
    assert!(json["version"].is_string());
    let pipelines = json["pipelines"].as_array().unwrap();
    assert!(pipelines.len() >= 2);
    assert_eq!(json["ocr_available"], false);
}

// -----------------------------------------------------------------------
// Formats endpoint
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_formats_endpoint() {
    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/formats")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    let supported = json["supported"].as_array().unwrap();
    assert!(supported.len() >= 4, "should list at least 4 formats");

    let kinds: Vec<&str> = supported
        .iter()
        .map(|e| e["kind"].as_str().unwrap())
        .collect();
    assert!(kinds.contains(&"web"));
    assert!(kinds.contains(&"pdf"));
    assert!(kinds.contains(&"markdown"));
    assert!(kinds.contains(&"plaintext"));
}

// -----------------------------------------------------------------------
// Web extraction via /v1/extract/bytes with HTML
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_web_extract_bytes() {
    let app = create_test_app();
    let html_bytes = std::fs::read("tests/fixtures/sample.html").expect("sample.html should exist");

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/extract/bytes")
                .header("content-type", "text/html")
                .body(Body::from(html_bytes))
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
    assert_eq!(events[0]["event"], "metadata");
    assert_eq!(events[0]["source_kind"], "web");

    // Middle events: chunks
    for event in &events[1..events.len() - 1] {
        assert_eq!(event["event"], "chunk");
        assert!(event["id"].is_string());
        assert!(event["text"].as_str().unwrap().len() > 0);
        assert!(event["token_estimate"].as_u64().unwrap() > 0);
    }

    // Last event: done
    let done = events.last().unwrap();
    assert_eq!(done["event"], "done");
    assert!(done["chunks_total"].as_u64().unwrap() > 0);
    assert_eq!(done["pipeline_used"], "web");
}

// -----------------------------------------------------------------------
// Web extraction preserves article content
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_web_extract_content_quality() {
    let app = create_test_app();
    let html_bytes = std::fs::read("tests/fixtures/sample.html").expect("sample.html should exist");

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/extract/bytes")
                .header("content-type", "text/html")
                .body(Body::from(html_bytes))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let events = parse_ndjson(response).await;
    let chunk_texts: String = events
        .iter()
        .filter(|e| e["event"] == "chunk")
        .map(|e| e["text"].as_str().unwrap().to_string())
        .collect::<Vec<_>>()
        .join(" ");

    // Key article content should appear in the extracted chunks
    assert!(
        chunk_texts.contains("section one")
            || chunk_texts.contains("Section One")
            || chunk_texts.to_lowercase().contains("section one"),
        "Extracted text should contain article content"
    );
}

// -----------------------------------------------------------------------
// Markdown passthrough via /v1/extract/bytes
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_markdown_extract_bytes() {
    let app = create_test_app();
    let md_content = b"# My Document\n\nThis is a paragraph.\n\n## Section\n\nMore content here.";

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/extract/bytes")
                .header("content-type", "text/markdown")
                .body(Body::from(md_content.to_vec()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let events = parse_ndjson(response).await;
    assert!(events.len() >= 3);

    assert_eq!(events[0]["event"], "metadata");
    assert_eq!(events[0]["source_kind"], "markdown");

    let done = events.last().unwrap();
    assert_eq!(done["event"], "done");
    assert_eq!(done["pipeline_used"], "markdown");
}

// -----------------------------------------------------------------------
// Plain text via /v1/extract/bytes
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_plaintext_extract_bytes() {
    let app = create_test_app();
    let text_content = b"This is a plain text document with enough content for extraction.";

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/extract/bytes")
                .header("content-type", "text/plain")
                .body(Body::from(text_content.to_vec()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let events = parse_ndjson(response).await;
    assert!(events.len() >= 3);

    assert_eq!(events[0]["event"], "metadata");
    assert_eq!(events[0]["source_kind"], "plaintext");

    let done = events.last().unwrap();
    assert_eq!(done["event"], "done");
    assert_eq!(done["pipeline_used"], "plaintext");
}

// -----------------------------------------------------------------------
// Empty body returns 400
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_extract_bytes_empty_body() {
    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/extract/bytes")
                .header("content-type", "text/html")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

// -----------------------------------------------------------------------
// OCR endpoint validation / availability
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_ocr_endpoint_returns_not_implemented_when_unconfigured() {
    let app = create_test_app();
    let body = serde_json::json!({
        "images": [encode_png_base64([255, 255, 255])]
    });

    let response = app
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

    assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["status"], 501);
    assert!(json["error"]
        .as_str()
        .unwrap()
        .contains("OCR is not configured"));
}

#[tokio::test]
async fn test_ocr_endpoint_rejects_invalid_payload() {
    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/ocr")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"images":"not-an-array"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["status"], 400);
    assert!(json["error"]
        .as_str()
        .unwrap()
        .contains("invalid OCR request payload"));
}

#[tokio::test]
async fn test_ocr_endpoint_rejects_invalid_base64() {
    let app = create_test_app();
    let body = serde_json::json!({
        "images": ["data:image/png;base64,not-valid-base64"]
    });

    let response = app
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

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["status"], 400);
    assert!(json["error"]
        .as_str()
        .unwrap()
        .contains("invalid base64 OCR image payload"));
}

#[tokio::test]
async fn test_ocr_endpoint_returns_service_unavailable_for_incomplete_config() {
    let mut config = test_config();
    config.model_path = Some("models/rec_model.onnx".to_string());
    let app = create_test_app_with_config(config);
    let body = serde_json::json!({
        "images": [encode_png_base64([255, 255, 255])]
    });

    let response = app
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

    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["status"], 503);
    assert!(json["error"]
        .as_str()
        .unwrap()
        .contains("configuration is incomplete"));
}

#[tokio::test]
async fn test_metrics_endpoint_exposes_prometheus_metrics() {
    let app = create_test_app();
    let body = b"metrics fixture body".to_vec();

    let health_response = app
        .clone()
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

    let first_extract = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/extract/bytes")
                .header("content-type", "text/plain")
                .body(Body::from(body.clone()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(first_extract.status(), StatusCode::OK);

    let cached_extract = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/extract/bytes")
                .header("content-type", "text/plain")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(cached_extract.status(), StatusCode::OK);

    let ocr_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/ocr")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "images": [encode_png_base64([255, 255, 255])]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(ocr_response.status(), StatusCode::NOT_IMPLEMENTED);

    let metrics_response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(metrics_response.status(), StatusCode::OK);
    assert_eq!(
        metrics_response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok()),
        Some("text/plain; version=0.0.4; charset=utf-8")
    );

    let body = axum::body::to_bytes(metrics_response.into_body(), usize::MAX)
        .await
        .unwrap();
    let metrics = String::from_utf8(body.to_vec()).unwrap();

    assert!(metrics.contains("# HELP doc2agent_requests_total"));
    assert!(metrics.contains("doc2agent_requests_total{route=\"/health\",status=\"200\"} 1"));
    assert!(metrics.contains("doc2agent_requests_total{route=\"/v1/ocr\",status=\"501\"} 1"));
    assert!(metrics.contains("# TYPE doc2agent_request_duration_ms histogram"));
    assert!(metrics.contains("doc2agent_request_duration_ms_count{route=\"/health\"} 1"));
    assert!(metrics.contains("doc2agent_cache_hits_total{route=\"/v1/extract/bytes\"} 1"));
    assert!(metrics.contains("doc2agent_cache_misses_total{route=\"/v1/extract/bytes\"} 1"));
    assert!(metrics.contains("doc2agent_cache_hit_ratio{route=\"/v1/extract/bytes\"} 0.500000"));
    assert!(metrics.contains("doc2agent_ocr_requests_total 1"));
    assert!(metrics.contains("doc2agent_ocr_usage_total 0"));
}
