use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::{
    body::Body,
    extract::{rejection::JsonRejection, Request, State},
    http::{header, StatusCode},
    middleware::{from_fn_with_state, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use base64::{
    engine::general_purpose::{STANDARD, STANDARD_NO_PAD},
    Engine as _,
};
use image::DynamicImage;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::{Notify, OwnedSemaphorePermit, Semaphore};
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;

use crate::cache::{Cache, InMemoryCache};
use crate::config::RuntimeConfig;
use crate::metrics::{
    route_label, MetricsCollector, PROMETHEUS_CONTENT_TYPE, ROUTE_EXTRACT_BYTES, ROUTE_EXTRACT_URL,
};
use crate::ocr::recognizer::Recognizer;
use crate::ocr::{OcrResult, OCR_MODEL_NAME};

const DEFAULT_CACHE_ENTRIES: usize = 256;

struct PerHostFetchBudget {
    limit: usize,
    semaphores: Mutex<HashMap<String, Arc<Semaphore>>>,
}

impl PerHostFetchBudget {
    fn new(limit: usize) -> Self {
        Self {
            limit,
            semaphores: Mutex::new(HashMap::new()),
        }
    }

    async fn acquire(&self, url: &str) -> Result<Option<OwnedSemaphorePermit>, AppError> {
        let Some(host) = url_host_key(url) else {
            return Ok(None);
        };

        let semaphore = {
            let mut semaphores = lock_mutex(&self.semaphores);
            semaphores
                .entry(host)
                .or_insert_with(|| Arc::new(Semaphore::new(self.limit)))
                .clone()
        };

        semaphore
            .acquire_owned()
            .await
            .map(Some)
            .map_err(|_| AppError::Internal("per-host fetch limiter is unavailable".to_string()))
    }
}

enum OcrService {
    Ready(Arc<Recognizer>),
    NotConfigured { reason: String },
    Unavailable { reason: String },
}

impl OcrService {
    fn from_config(config: &RuntimeConfig) -> Self {
        match (config.model_path.as_deref(), config.dict_path.as_deref()) {
            (Some(model_path), Some(dict_path)) => match Recognizer::new(
                model_path,
                dict_path,
                config.session_pool_size,
                config.max_batch,
                config.intra_threads,
                config.inter_threads,
                config.device_id,
            ) {
                Ok(recognizer) => Self::Ready(Arc::new(recognizer)),
                Err(error) => {
                    tracing::warn!(
                        error = %error,
                        model_path,
                        dict_path,
                        "failed to initialize OCR recognizer; /v1/ocr will return 503"
                    );
                    Self::Unavailable {
                        reason: format!("OCR recognizer is unavailable: {error}"),
                    }
                }
            },
            (None, None) => Self::NotConfigured {
                reason: "OCR is not configured; set both DOC2AGENT_MODEL_PATH and DOC2AGENT_DICT_PATH to enable /v1/ocr".to_string(),
            },
            _ => Self::Unavailable {
                reason: "OCR configuration is incomplete; set both model_path and dict_path".to_string(),
            },
        }
    }

    fn available(&self) -> bool {
        matches!(self, Self::Ready(_))
    }

    fn recognizer(&self) -> Result<Arc<Recognizer>, AppError> {
        match self {
            Self::Ready(recognizer) => Ok(Arc::clone(recognizer)),
            Self::NotConfigured { reason } => Err(AppError::NotImplemented(reason.clone())),
            Self::Unavailable { reason } => Err(AppError::ServiceUnavailable(reason.clone())),
        }
    }
}

/// Shared application state.
pub struct AppState {
    pub config: RuntimeConfig,
    pub client: reqwest::Client,
    pub cache: InMemoryCache,
    pub metrics: MetricsCollector,
    pub extraction_limiter: Arc<Semaphore>,
    pub ocr_limiter: Arc<Semaphore>,
    fetch_budget: Option<PerHostFetchBudget>,
    in_flight_requests: AtomicUsize,
    shutting_down: AtomicBool,
    drain_notify: Notify,
    ocr_service: OcrService,
}

impl AppState {
    pub fn new(config: RuntimeConfig) -> Self {
        crate::pdfium::install_runtime_config(&config);
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .connect_timeout(Duration::from_secs(10))
            .pool_idle_timeout(Duration::from_secs(90))
            .redirect(reqwest::redirect::Policy::limited(10))
            .gzip(true)
            .brotli(true)
            .build()
            .expect("failed to build reqwest client");
        let ocr_service = OcrService::from_config(&config);
        Self {
            extraction_limiter: Arc::new(Semaphore::new(config.extraction_concurrency)),
            ocr_limiter: Arc::new(Semaphore::new(config.ocr_concurrency)),
            metrics: MetricsCollector::default(),
            fetch_budget: config
                .per_host_fetch_concurrency
                .map(PerHostFetchBudget::new),
            in_flight_requests: AtomicUsize::new(0),
            shutting_down: AtomicBool::new(false),
            drain_notify: Notify::new(),
            config,
            client,
            cache: InMemoryCache::new(DEFAULT_CACHE_ENTRIES),
            ocr_service,
        }
    }

    pub fn in_flight_requests(&self) -> usize {
        self.in_flight_requests.load(Ordering::SeqCst)
    }

    pub fn begin_shutdown(&self) -> bool {
        let was_running = !self.shutting_down.swap(true, Ordering::SeqCst);
        self.drain_notify.notify_waiters();
        was_running
    }

    pub fn is_shutting_down(&self) -> bool {
        self.shutting_down.load(Ordering::SeqCst)
    }

    pub async fn wait_for_drain(&self, timeout: Duration) -> Result<(), usize> {
        let deadline = tokio::time::Instant::now() + timeout;

        loop {
            let remaining = self.in_flight_requests();
            if remaining == 0 {
                return Ok(());
            }

            let wait_time = deadline.saturating_duration_since(tokio::time::Instant::now());
            if wait_time.is_zero() {
                return Err(remaining);
            }

            if tokio::time::timeout(wait_time, self.drain_notify.notified())
                .await
                .is_err()
            {
                return Err(self.in_flight_requests());
            }
        }
    }

    async fn acquire_extraction_permit(&self) -> Result<OwnedSemaphorePermit, AppError> {
        Arc::clone(&self.extraction_limiter)
            .acquire_owned()
            .await
            .map_err(|_| AppError::Internal("extraction limiter is unavailable".to_string()))
    }

    async fn acquire_ocr_permit(&self) -> Result<OwnedSemaphorePermit, AppError> {
        Arc::clone(&self.ocr_limiter)
            .acquire_owned()
            .await
            .map_err(|_| AppError::Internal("OCR limiter is unavailable".to_string()))
    }

    async fn acquire_fetch_permit(
        &self,
        url: &str,
    ) -> Result<Option<OwnedSemaphorePermit>, AppError> {
        match &self.fetch_budget {
            Some(budget) => budget.acquire(url).await,
            None => Ok(None),
        }
    }

    fn ocr_available(&self) -> bool {
        self.ocr_service.available()
    }

    fn ocr_recognizer(&self) -> Result<Arc<Recognizer>, AppError> {
        self.ocr_service.recognizer()
    }

    fn request_started(&self) {
        self.in_flight_requests.fetch_add(1, Ordering::SeqCst);
    }

    fn request_finished(&self) {
        let previous = self
            .in_flight_requests
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |current| {
                current.checked_sub(1)
            })
            .unwrap_or(0);

        if previous == 0 {
            tracing::warn!("in-flight request accounting underflow");
        }

        self.drain_notify.notify_waiters();
    }
}

/// Build the Axum router with all routes and middleware.
pub fn build_router(state: Arc<AppState>) -> Router {
    let timeout = Duration::from_secs(state.config.request_timeout_secs);
    let body_limit = state.config.max_request_body_bytes;

    Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics_handler))
        .route("/v1/extract/url", post(extract_url))
        .route("/v1/extract/bytes", post(extract_bytes))
        .route("/v1/ocr", post(ocr))
        .route("/v1/formats", get(formats))
        .layer(TraceLayer::new_for_http())
        .layer(TimeoutLayer::with_status_code(
            StatusCode::REQUEST_TIMEOUT,
            timeout,
        ))
        .layer(RequestBodyLimitLayer::new(body_limit))
        .layer(from_fn_with_state(Arc::clone(&state), track_in_flight))
        .with_state(state)
}

fn lock_mutex<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn url_host_key(url: &str) -> Option<String> {
    reqwest::Url::parse(url)
        .ok()?
        .host_str()
        .map(|host| host.to_ascii_lowercase())
}

async fn track_in_flight(
    State(state): State<Arc<AppState>>,
    request: Request,
    next: Next,
) -> Response {
    let route = route_label(request.uri().path());
    let started_at = std::time::Instant::now();
    state.request_started();
    let response = next.run(request).await;
    let latency_ms = started_at.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
    state
        .metrics
        .record_request(route, response.status(), latency_ms);
    state.request_finished();
    response
}

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    version: &'static str,
    pipelines: Vec<&'static str>,
    ocr_available: bool,
}

async fn health(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(HealthResponse {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
        pipelines: vec!["web", "pdf", "markdown", "plaintext"],
        ocr_available: state.ocr_available(),
    })
}

async fn metrics_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, PROMETHEUS_CONTENT_TYPE)],
        state.metrics.render_prometheus(),
    )
}

// ---------------------------------------------------------------------------
// Formats
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct FormatsResponse {
    supported: Vec<FormatEntry>,
}

#[derive(Serialize)]
struct FormatEntry {
    kind: &'static str,
    mime_types: Vec<&'static str>,
    description: &'static str,
}

async fn formats() -> impl IntoResponse {
    Json(FormatsResponse {
        supported: vec![
            FormatEntry {
                kind: "web",
                mime_types: vec!["text/html"],
                description: "Web pages via readability + html2md",
            },
            FormatEntry {
                kind: "pdf",
                mime_types: vec!["application/pdf"],
                description: "PDF text extraction via pdf-extract",
            },
            FormatEntry {
                kind: "markdown",
                mime_types: vec!["text/markdown"],
                description: "Markdown passthrough with normalization",
            },
            FormatEntry {
                kind: "plaintext",
                mime_types: vec!["text/plain"],
                description: "Plain text with minimal processing",
            },
        ],
    })
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

enum AppError {
    BadRequest(String),
    NotImplemented(String),
    ServiceUnavailable(String),
    Internal(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            AppError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg),
            AppError::NotImplemented(msg) => (StatusCode::NOT_IMPLEMENTED, msg),
            AppError::ServiceUnavailable(msg) => (StatusCode::SERVICE_UNAVAILABLE, msg),
            AppError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
        };
        let body = serde_json::json!({
            "error": message,
            "status": status.as_u16()
        });
        (status, Json(body)).into_response()
    }
}

// ---------------------------------------------------------------------------
// Extraction endpoints
// ---------------------------------------------------------------------------

fn default_mode() -> String {
    "auto".to_string()
}

fn default_true() -> bool {
    true
}

fn update_hash_field(hasher: &mut Sha256, name: &str, value: &[u8]) {
    hasher.update((name.len() as u64).to_be_bytes());
    hasher.update(name.as_bytes());
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value);
}

fn update_hash_optional_text(hasher: &mut Sha256, name: &str, value: Option<&str>) {
    if let Some(value) = value {
        update_hash_field(hasher, &format!("{name}:present"), &[1]);
        update_hash_field(hasher, name, value.as_bytes());
    } else {
        update_hash_field(hasher, &format!("{name}:present"), &[0]);
    }
}

fn normalize_mime(mime: Option<&str>) -> Option<String> {
    mime.map(|value| {
        value
            .split(';')
            .next()
            .unwrap_or(value)
            .trim()
            .to_ascii_lowercase()
    })
    .filter(|value| !value.is_empty())
}

fn url_cache_key(url: &str, mode: &str, max_pages: Option<u32>) -> String {
    let mut hasher = Sha256::new();
    update_hash_field(&mut hasher, "kind", b"url");
    update_hash_field(&mut hasher, "url", url.as_bytes());
    update_hash_field(&mut hasher, "mode", mode.trim().as_bytes());

    let max_pages_value = max_pages.map(|value| value.to_string());
    update_hash_optional_text(&mut hasher, "max_pages", max_pages_value.as_deref());

    format!("{:x}", hasher.finalize())
}

fn bytes_cache_key(body: &[u8], mime: Option<&str>, filename: Option<&str>) -> String {
    let mut hasher = Sha256::new();
    update_hash_field(&mut hasher, "kind", b"bytes");
    update_hash_field(&mut hasher, "body", body);

    let normalized_mime = normalize_mime(mime);
    update_hash_optional_text(&mut hasher, "mime", normalized_mime.as_deref());
    update_hash_optional_text(&mut hasher, "filename", filename);

    format!("{:x}", hasher.finalize())
}

#[derive(Deserialize)]
struct ExtractUrlRequest {
    url: String,
    #[serde(default = "default_mode")]
    mode: String,
    #[serde(default = "default_true")]
    stream: bool,
    #[allow(dead_code)]
    max_pages: Option<u32>,
}

async fn extract_url(
    State(state): State<Arc<AppState>>,
    Json(body): Json<ExtractUrlRequest>,
) -> Result<Response<Body>, AppError> {
    tracing::info!(url = %body.url, mode = %body.mode, "extract_url request");

    let cache_key = url_cache_key(&body.url, &body.mode, body.max_pages);
    if let Some(output) = state.cache.get(&cache_key) {
        state.metrics.record_cache_hit(ROUTE_EXTRACT_URL);
        tracing::debug!(cache_key = %cache_key, url = %body.url, mode = %body.mode, "extract_url cache hit");
        return if body.stream {
            Ok(crate::stream::ndjson_stream(output))
        } else {
            Ok(crate::stream::json_response(&output))
        };
    }

    state.metrics.record_cache_miss(ROUTE_EXTRACT_URL);
    tracing::debug!(cache_key = %cache_key, url = %body.url, mode = %body.mode, "extract_url cache miss");

    let _fetch_permit = state.acquire_fetch_permit(&body.url).await?;
    let source = crate::resolver::resolve_url(&state.client, &body.url)
        .await
        .map_err(|e| AppError::BadRequest(format!("failed to resolve URL: {e}")))?;

    let _extraction_permit = state.acquire_extraction_permit().await?;
    let output = crate::pipeline::dispatch(source)
        .await
        .map_err(|e| AppError::Internal(format!("extraction failed: {e}")))?;

    if output.diagnostics.ocr_used {
        state.metrics.record_ocr_usage();
    }
    state.cache.put(cache_key, output.clone());

    if body.stream {
        Ok(crate::stream::ndjson_stream(output))
    } else {
        Ok(crate::stream::json_response(&output))
    }
}

async fn extract_bytes(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    body: bytes::Bytes,
) -> Result<Response<Body>, AppError> {
    let content_type = headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .map(String::from);
    let filename = headers
        .get("x-filename")
        .and_then(|v| v.to_str().ok())
        .map(String::from);

    tracing::info!(
        content_type = ?content_type,
        filename = ?filename,
        size = body.len(),
        "extract_bytes request"
    );

    let cache_key = bytes_cache_key(body.as_ref(), content_type.as_deref(), filename.as_deref());
    if let Some(output) = state.cache.get(&cache_key) {
        state.metrics.record_cache_hit(ROUTE_EXTRACT_BYTES);
        tracing::debug!(
            cache_key = %cache_key,
            content_type = ?content_type,
            filename = ?filename,
            "extract_bytes cache hit"
        );
        return Ok(crate::stream::ndjson_stream(output));
    }

    state.metrics.record_cache_miss(ROUTE_EXTRACT_BYTES);
    tracing::debug!(
        cache_key = %cache_key,
        content_type = ?content_type,
        filename = ?filename,
        "extract_bytes cache miss"
    );

    let source = crate::resolver::resolve_bytes(body, filename.as_deref(), content_type.as_deref())
        .map_err(|e| AppError::BadRequest(format!("failed to classify content: {e}")))?;

    let _extraction_permit = state.acquire_extraction_permit().await?;
    let output = crate::pipeline::dispatch(source)
        .await
        .map_err(|e| AppError::Internal(format!("extraction failed: {e}")))?;

    if output.diagnostics.ocr_used {
        state.metrics.record_ocr_usage();
    }
    state.cache.put(cache_key, output.clone());

    Ok(crate::stream::ndjson_stream(output))
}

#[derive(Deserialize)]
struct OcrRequest {
    images: Vec<String>,
}

#[derive(Serialize)]
struct OcrResponse {
    model: &'static str,
    batch_size: usize,
    latency_ms: u64,
    items: Vec<OcrResult>,
}

fn decode_base64_image(encoded: &str) -> Result<DynamicImage, AppError> {
    let encoded = encoded.trim();
    if encoded.is_empty() {
        return Err(AppError::BadRequest(
            "OCR image payload must not be empty".to_string(),
        ));
    }

    let image_bytes = if let Some(data_url) = encoded.strip_prefix("data:") {
        let (metadata, payload) = data_url.split_once(',').ok_or_else(|| {
            AppError::BadRequest("OCR data URL is missing a comma separator".to_string())
        })?;
        let metadata = metadata.to_ascii_lowercase();
        let media_type = metadata.split(';').next().unwrap_or_default();
        if !media_type.is_empty() && !media_type.starts_with("image/") {
            return Err(AppError::BadRequest(
                "OCR data URL must use an image MIME type".to_string(),
            ));
        }
        if !metadata.split(';').any(|segment| segment == "base64") {
            return Err(AppError::BadRequest(
                "OCR data URL images must be base64-encoded".to_string(),
            ));
        }
        decode_base64_bytes(payload)?
    } else {
        decode_base64_bytes(encoded)?
    };

    image::load_from_memory(&image_bytes)
        .map_err(|error| AppError::BadRequest(format!("OCR image could not be decoded: {error}")))
}

fn decode_base64_bytes(encoded: &str) -> Result<Vec<u8>, AppError> {
    STANDARD
        .decode(encoded)
        .or_else(|_| STANDARD_NO_PAD.decode(encoded))
        .map_err(|error| AppError::BadRequest(format!("invalid base64 OCR image payload: {error}")))
}

async fn ocr(
    State(state): State<Arc<AppState>>,
    payload: Result<Json<OcrRequest>, JsonRejection>,
) -> Result<impl IntoResponse, AppError> {
    let Json(body) = payload.map_err(|error| {
        AppError::BadRequest(format!(
            "invalid OCR request payload: {}",
            error.body_text()
        ))
    })?;
    if body.images.is_empty() {
        return Err(AppError::BadRequest(
            "OCR request must include at least one image".to_string(),
        ));
    }

    let batch_size = body.images.len();
    let decode_started_at = std::time::Instant::now();
    let images = tokio::task::spawn_blocking(move || {
        body.images
            .into_iter()
            .enumerate()
            .map(|(index, image)| {
                decode_base64_image(&image).map_err(|error| match error {
                    AppError::BadRequest(message) => {
                        AppError::BadRequest(format!("images[{index}]: {message}"))
                    }
                    other => other,
                })
            })
            .collect::<Result<Vec<_>, _>>()
    })
    .await
    .map_err(|error| AppError::Internal(format!("OCR decode worker task failed: {error}")))??;

    let recognizer = state.ocr_recognizer()?;
    let _ocr_permit = state.acquire_ocr_permit().await?;
    let items = tokio::task::spawn_blocking(move || {
        recognizer
            .recognize_batch(&images)
            .map_err(|error| AppError::Internal(format!("OCR recognition failed: {error}")))
    })
    .await
    .map_err(|error| AppError::Internal(format!("OCR worker task failed: {error}")))??;
    let latency_ms = decode_started_at
        .elapsed()
        .as_millis()
        .min(u128::from(u64::MAX)) as u64;

    Ok(Json(OcrResponse {
        model: OCR_MODEL_NAME,
        batch_size,
        latency_ms,
        items,
    }))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use axum::body::{to_bytes, Body};
    use axum::http::{Request, StatusCode};
    use axum::routing::get;
    use serde_json::json;
    use tower::ServiceExt;

    use super::{build_router, bytes_cache_key, url_cache_key, AppState};
    use crate::cache::Cache;
    use crate::chunker::{Chunk, DocumentMetadata, DocumentOutput, PipelineDiagnostics};
    use crate::config::RuntimeConfig;

    fn test_config() -> RuntimeConfig {
        RuntimeConfig {
            host: "127.0.0.1".to_string(),
            port: 0,
            request_timeout_secs: 30,
            max_request_body_bytes: 50 * 1024 * 1024,
            extraction_concurrency: 4,
            ocr_concurrency: 1,
            per_host_fetch_concurrency: Some(2),
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

    fn sample_output(title: &str, text: &str) -> DocumentOutput {
        DocumentOutput {
            title: Some(title.to_string()),
            canonical_url: Some("cached://doc".to_string()),
            markdown: text.to_string(),
            chunks: vec![Chunk {
                id: "c01".to_string(),
                text: text.to_string(),
                section: Some("cached".to_string()),
                page_start: None,
                page_end: None,
                char_count: text.len(),
                token_estimate: text.len() / 4,
            }],
            metadata: DocumentMetadata {
                page_count: None,
                word_count: text.split_whitespace().count(),
                char_count: text.len(),
            },
            diagnostics: PipelineDiagnostics {
                pipeline_used: "cached".to_string(),
                ocr_used: false,
                render_used: false,
                fallback_used: false,
                fallback_reason: None,
                text_quality_score: None,
                latency_ms: 1,
            },
            image_manifest: None,
        }
    }

    async fn parse_ndjson(response: axum::response::Response) -> Vec<serde_json::Value> {
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("response body should be readable");
        String::from_utf8(body.to_vec())
            .expect("response body should be utf-8")
            .lines()
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_str(line).expect("line should parse as json"))
            .collect()
    }

    async fn parse_json(response: axum::response::Response) -> serde_json::Value {
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("response body should be readable");
        serde_json::from_slice(&body).expect("response body should parse as json")
    }

    async fn start_test_content_server() -> (String, tokio::task::JoinHandle<()>) {
        let app = axum::Router::new().route(
            "/doc",
            get(|| async {
                axum::response::Html(
                    "<html><head><title>Fixture</title></head><body><article><h1>Fixture</h1><p>Remote content.</p></article></body></html>",
                )
            }),
        );

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let addr = listener
            .local_addr()
            .expect("listener should have local addr");
        let handle = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        (format!("http://{addr}/doc"), handle)
    }

    #[tokio::test]
    async fn extract_bytes_uses_cached_output() {
        let state = Arc::new(AppState::new(test_config()));
        let app = build_router(Arc::clone(&state));

        let body = b"Original document body".to_vec();
        let first_response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/extract/bytes")
                    .header("content-type", "text/plain; charset=utf-8")
                    .header("x-filename", "note.txt")
                    .body(Body::from(body.clone()))
                    .expect("request should build"),
            )
            .await
            .expect("request should succeed");

        assert_eq!(first_response.status(), StatusCode::OK);

        let cache_key = bytes_cache_key(&body, Some("text/plain; charset=utf-8"), Some("note.txt"));
        assert!(
            state.cache.get(&cache_key).is_some(),
            "successful extraction should be cached"
        );

        state.cache.put(
            cache_key,
            sample_output("cached bytes", "cached bytes body"),
        );

        let cached_response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/extract/bytes")
                    .header("content-type", "text/plain; charset=utf-8")
                    .header("x-filename", "note.txt")
                    .body(Body::from(body))
                    .expect("request should build"),
            )
            .await
            .expect("request should succeed");

        let events = parse_ndjson(cached_response).await;
        assert_eq!(events[0]["title"], "cached bytes");
        assert_eq!(events[0]["source_kind"], "cached");
        assert_eq!(events[1]["text"], "cached bytes body");
    }

    #[tokio::test]
    async fn extract_url_uses_cached_output_and_ignores_stream_flag() {
        let (url, server_handle) = start_test_content_server().await;
        let state = Arc::new(AppState::new(test_config()));
        let app = build_router(Arc::clone(&state));

        let first_response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/extract/url")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "url": url,
                            "mode": "auto",
                            "stream": true,
                            "max_pages": 2
                        })
                        .to_string(),
                    ))
                    .expect("request should build"),
            )
            .await
            .expect("request should succeed");

        assert_eq!(first_response.status(), StatusCode::OK);

        let cache_key = url_cache_key(&url, "auto", Some(2));
        assert!(
            state.cache.get(&cache_key).is_some(),
            "successful URL extraction should be cached"
        );
        state
            .cache
            .put(cache_key, sample_output("cached url", "cached url body"));

        server_handle.abort();
        let _ = server_handle.await;

        let cached_response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/extract/url")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "url": url,
                            "mode": "auto",
                            "stream": false,
                            "max_pages": 2
                        })
                        .to_string(),
                    ))
                    .expect("request should build"),
            )
            .await
            .expect("request should succeed");

        assert_eq!(cached_response.status(), StatusCode::OK);

        let json = parse_json(cached_response).await;
        assert_eq!(json["title"], "cached url");
        assert_eq!(json["markdown"], "cached url body");
        assert_eq!(json["diagnostics"]["pipeline_used"], "cached");
    }

    #[test]
    fn cache_keys_ignore_stream_but_vary_with_relevant_inputs() {
        let first = url_cache_key("https://example.com/doc", "auto", Some(1));
        let same = url_cache_key("https://example.com/doc", "auto", Some(1));
        let different_mode = url_cache_key("https://example.com/doc", "rich", Some(1));
        let different_pages = url_cache_key("https://example.com/doc", "auto", Some(2));
        let different_bytes = bytes_cache_key(b"abc", Some("text/plain"), Some("a.txt"));
        let same_bytes = bytes_cache_key(b"abc", Some("text/plain; charset=utf-8"), Some("a.txt"));
        let different_filename = bytes_cache_key(b"abc", Some("text/plain"), Some("b.txt"));

        assert_eq!(first, same);
        assert_ne!(first, different_mode);
        assert_ne!(first, different_pages);
        assert_eq!(different_bytes, same_bytes);
        assert_ne!(different_bytes, different_filename);
    }

    #[tokio::test]
    async fn extract_bytes_waits_for_extraction_capacity_and_tracks_in_flight_requests() {
        let mut config = test_config();
        config.extraction_concurrency = 1;
        let state = Arc::new(AppState::new(config));
        let held_permit = Arc::clone(&state.extraction_limiter)
            .acquire_owned()
            .await
            .expect("permit should be acquired");
        let app = build_router(Arc::clone(&state));

        let request = Request::builder()
            .method("POST")
            .uri("/v1/extract/bytes")
            .header("content-type", "text/plain")
            .body(Body::from("queued body"))
            .expect("request should build");

        let handle = tokio::spawn(async move {
            app.oneshot(request)
                .await
                .expect("request should complete once capacity is available")
        });

        for _ in 0..20 {
            if state.in_flight_requests() == 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        assert_eq!(
            state.in_flight_requests(),
            1,
            "request should be counted as in-flight while waiting for extraction capacity"
        );

        drop(held_permit);

        let response = handle.await.expect("task should join");
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(state.in_flight_requests(), 0);
    }

    #[tokio::test]
    async fn wait_for_drain_times_out_then_completes_after_request_finishes() {
        let mut config = test_config();
        config.extraction_concurrency = 1;
        let state = Arc::new(AppState::new(config));
        let held_permit = Arc::clone(&state.extraction_limiter)
            .acquire_owned()
            .await
            .expect("permit should be acquired");
        let app = build_router(Arc::clone(&state));

        let request = Request::builder()
            .method("POST")
            .uri("/v1/extract/bytes")
            .header("content-type", "text/plain")
            .body(Body::from("queued body"))
            .expect("request should build");

        let handle = tokio::spawn(async move {
            app.oneshot(request)
                .await
                .expect("request should complete once capacity is available")
        });

        for _ in 0..20 {
            if state.in_flight_requests() == 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        state.begin_shutdown();
        assert_eq!(
            state.wait_for_drain(Duration::from_millis(50)).await,
            Err(1)
        );

        drop(held_permit);
        let response = handle.await.expect("task should join");
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(state.wait_for_drain(Duration::from_secs(1)).await, Ok(()));
        assert!(state.is_shutting_down());
    }
}
