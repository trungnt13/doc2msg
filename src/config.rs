/// Runtime configuration derived from CLI args.
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub host: String,
    pub port: u16,
    // Phase 1: core server config
    pub request_timeout_secs: u64,
    pub max_request_body_bytes: usize,
    pub extraction_concurrency: usize,
    pub ocr_concurrency: usize,
    pub per_host_fetch_concurrency: Option<usize>,
    // Phase 2+: OCR config (optional)
    pub model_path: Option<String>,
    pub dict_path: Option<String>,
    pub pdfium_enabled: bool,
    pub pdfium_lib_path: Option<String>,
    pub session_pool_size: usize,
    pub max_batch: usize,
    pub intra_threads: usize,
    pub inter_threads: usize,
    pub device_id: i32,
}

impl RuntimeConfig {
    pub fn from_cli(args: &crate::CliArgs) -> Self {
        Self {
            host: args.host.clone(),
            port: args.port,
            request_timeout_secs: args.request_timeout,
            max_request_body_bytes: args.max_body_size,
            extraction_concurrency: normalize_limit(args.extraction_concurrency, 16),
            ocr_concurrency: normalize_limit(
                args.ocr_concurrency.unwrap_or(args.session_pool_size),
                1,
            ),
            per_host_fetch_concurrency: args
                .per_host_fetch_concurrency
                .map(|limit| normalize_limit(limit, 1)),
            model_path: args.model_path.clone(),
            dict_path: args.dict_path.clone(),
            pdfium_enabled: args.pdfium_enabled,
            pdfium_lib_path: args.pdfium_lib_path.clone(),
            session_pool_size: args.session_pool_size,
            max_batch: args.max_batch,
            intra_threads: args.intra_threads,
            inter_threads: args.inter_threads,
            device_id: args.device_id,
        }
    }
}

fn normalize_limit(limit: usize, default: usize) -> usize {
    if limit == 0 {
        default
    } else {
        limit
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::RuntimeConfig;
    use crate::CliArgs;

    #[test]
    fn from_cli_uses_runtime_hardening_defaults() {
        let args = CliArgs::parse_from(["doc2msg"]);
        let config = RuntimeConfig::from_cli(&args);

        assert_eq!(config.extraction_concurrency, 16);
        assert_eq!(config.ocr_concurrency, args.session_pool_size);
        assert_eq!(config.per_host_fetch_concurrency, None);
    }

    #[test]
    fn from_cli_normalizes_zero_limits() {
        let args = CliArgs::parse_from([
            "doc2msg",
            "--extraction-concurrency",
            "0",
            "--ocr-concurrency",
            "0",
            "--per-host-fetch-concurrency",
            "0",
        ]);
        let config = RuntimeConfig::from_cli(&args);

        assert_eq!(config.extraction_concurrency, 16);
        assert_eq!(config.ocr_concurrency, 1);
        assert_eq!(config.per_host_fetch_concurrency, Some(1));
    }

    #[test]
    fn from_cli_sets_pdfium_runtime_config() {
        let args = CliArgs::parse_from([
            "doc2msg",
            "--pdfium-enabled",
            "--pdfium-lib-path",
            "/opt/pdfium/libpdfium.so",
        ]);
        let config = RuntimeConfig::from_cli(&args);

        assert!(config.pdfium_enabled);
        assert_eq!(
            config.pdfium_lib_path.as_deref(),
            Some("/opt/pdfium/libpdfium.so")
        );
    }
}
