#![allow(dead_code)]

pub mod cache;
pub mod chunker;
pub mod config;
pub mod metrics;
pub mod normalizer;
pub mod ocr;
pub mod pdfium;
pub mod pipeline;
pub mod resolver;
pub mod server;
pub mod stream;

use clap::Parser;

/// CLI arguments for the doc2msg microservice.
#[derive(Parser, Debug)]
#[command(
    name = "doc2msg",
    about = "Ultra-fast document to agent-friendly output microservice"
)]
pub struct CliArgs {
    /// Host address to bind
    #[arg(long, env = "DOC2MSG_HOST", default_value = "0.0.0.0")]
    pub host: String,

    /// Port to listen on
    #[arg(long, env = "DOC2MSG_PORT", default_value_t = 3000)]
    pub port: u16,

    /// Request timeout in seconds
    #[arg(long, env = "DOC2MSG_REQUEST_TIMEOUT", default_value_t = 60)]
    pub request_timeout: u64,

    /// Maximum request body size in bytes (default 50 MiB)
    #[arg(long, env = "DOC2MSG_MAX_BODY_SIZE", default_value_t = 50 * 1024 * 1024)]
    pub max_body_size: usize,

    /// Maximum number of concurrent extraction jobs
    #[arg(long, env = "DOC2MSG_EXTRACTION_CONCURRENCY", default_value_t = 16)]
    pub extraction_concurrency: usize,

    /// Maximum number of concurrent OCR jobs (defaults to session pool size when omitted)
    #[arg(long, env = "DOC2MSG_OCR_CONCURRENCY")]
    pub ocr_concurrency: Option<usize>,

    /// Optional per-host concurrency limit for outbound URL fetches
    #[arg(long, env = "DOC2MSG_PER_HOST_FETCH_CONCURRENCY")]
    pub per_host_fetch_concurrency: Option<usize>,

    /// Path to ONNX model file (Phase 2+)
    #[arg(long, env = "DOC2MSG_MODEL_PATH")]
    pub model_path: Option<String>,

    /// Path to dictionary file for CTC decoding (Phase 2+)
    #[arg(long, env = "DOC2MSG_DICT_PATH")]
    pub dict_path: Option<String>,

    /// Enable runtime pdfium binding for Phase 4 PDF text extraction and rendering
    #[arg(long, env = "DOC2MSG_PDFIUM_ENABLED", default_value_t = false)]
    pub pdfium_enabled: bool,

    /// Optional path to libpdfium (either the shared library or its containing directory)
    #[arg(long, env = "DOC2MSG_PDFIUM_LIB_PATH")]
    pub pdfium_lib_path: Option<String>,

    /// Number of sessions in the ORT session pool
    #[arg(long, env = "DOC2MSG_SESSION_POOL_SIZE", default_value_t = 2)]
    pub session_pool_size: usize,

    /// Maximum batch size for OCR inference
    #[arg(long, env = "DOC2MSG_MAX_BATCH", default_value_t = 16)]
    pub max_batch: usize,

    /// Number of intra-op threads for ORT
    #[arg(long, env = "DOC2MSG_INTRA_THREADS", default_value_t = 4)]
    pub intra_threads: usize,

    /// Number of inter-op threads for ORT
    #[arg(long, env = "DOC2MSG_INTER_THREADS", default_value_t = 1)]
    pub inter_threads: usize,

    /// GPU device ID (-1 for CPU)
    #[arg(long, env = "DOC2MSG_DEVICE_ID", default_value_t = -1)]
    pub device_id: i32,
}
