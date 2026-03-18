use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::sync::Mutex;

use axum::http::StatusCode;

pub const PROMETHEUS_CONTENT_TYPE: &str = "text/plain; version=0.0.4; charset=utf-8";

pub const ROUTE_HEALTH: &str = "/health";
pub const ROUTE_METRICS: &str = "/metrics";
pub const ROUTE_FORMATS: &str = "/v1/formats";
pub const ROUTE_EXTRACT_URL: &str = "/v1/extract/url";
pub const ROUTE_EXTRACT_BYTES: &str = "/v1/extract/bytes";
pub const ROUTE_OCR: &str = "/v1/ocr";
pub const ROUTE_UNMATCHED: &str = "unmatched";

const KNOWN_ROUTES: [&str; 7] = [
    ROUTE_HEALTH,
    ROUTE_METRICS,
    ROUTE_FORMATS,
    ROUTE_EXTRACT_URL,
    ROUTE_EXTRACT_BYTES,
    ROUTE_OCR,
    ROUTE_UNMATCHED,
];
const CACHE_ROUTES: [&str; 2] = [ROUTE_EXTRACT_URL, ROUTE_EXTRACT_BYTES];
const LATENCY_BUCKETS_MS: [u64; 9] = [5, 10, 25, 50, 100, 250, 500, 1_000, 5_000];

#[derive(Default)]
pub struct MetricsCollector {
    inner: Mutex<MetricsState>,
}

#[derive(Default)]
struct MetricsState {
    requests_total: BTreeMap<(&'static str, u16), u64>,
    request_latency: BTreeMap<&'static str, LatencyStats>,
    cache_hits: BTreeMap<&'static str, u64>,
    cache_misses: BTreeMap<&'static str, u64>,
    ocr_requests: u64,
    ocr_usage: u64,
}

#[derive(Clone)]
struct LatencyStats {
    bucket_counts: Vec<u64>,
    sum_ms: u64,
    count: u64,
}

impl Default for LatencyStats {
    fn default() -> Self {
        Self {
            bucket_counts: vec![0; LATENCY_BUCKETS_MS.len() + 1],
            sum_ms: 0,
            count: 0,
        }
    }
}

impl LatencyStats {
    fn record(&mut self, latency_ms: u64) {
        self.count = self.count.saturating_add(1);
        self.sum_ms = self.sum_ms.saturating_add(latency_ms);

        let index = LATENCY_BUCKETS_MS
            .iter()
            .position(|bucket| latency_ms <= *bucket)
            .unwrap_or(LATENCY_BUCKETS_MS.len());
        if let Some(bucket) = self.bucket_counts.get_mut(index) {
            *bucket = bucket.saturating_add(1);
        }
    }
}

impl MetricsCollector {
    pub fn record_request(&self, route: &'static str, status: StatusCode, latency_ms: u64) {
        let mut inner = self.lock_state();
        let status = status.as_u16();

        inner
            .requests_total
            .entry((route, status))
            .and_modify(|count| *count = count.saturating_add(1))
            .or_insert(1);

        inner
            .request_latency
            .entry(route)
            .or_default()
            .record(latency_ms);

        if route == ROUTE_OCR {
            inner.ocr_requests = inner.ocr_requests.saturating_add(1);
        }
    }

    pub fn record_cache_hit(&self, route: &'static str) {
        let mut inner = self.lock_state();
        inner
            .cache_hits
            .entry(route)
            .and_modify(|count| *count = count.saturating_add(1))
            .or_insert(1);
    }

    pub fn record_cache_miss(&self, route: &'static str) {
        let mut inner = self.lock_state();
        inner
            .cache_misses
            .entry(route)
            .and_modify(|count| *count = count.saturating_add(1))
            .or_insert(1);
    }

    pub fn record_ocr_usage(&self) {
        let mut inner = self.lock_state();
        inner.ocr_usage = inner.ocr_usage.saturating_add(1);
    }

    pub fn render_prometheus(&self) -> String {
        let inner = self.lock_state();
        let mut output = String::new();

        output
            .push_str("# HELP doc2msg_requests_total Total HTTP requests by route and status.\n");
        output.push_str("# TYPE doc2msg_requests_total counter\n");
        for ((route, status), count) in &inner.requests_total {
            let _ = writeln!(
                output,
                "doc2msg_requests_total{{route=\"{route}\",status=\"{status}\"}} {count}"
            );
        }

        output.push_str("# HELP doc2msg_request_duration_ms Request latency in milliseconds.\n");
        output.push_str("# TYPE doc2msg_request_duration_ms histogram\n");
        for route in KNOWN_ROUTES {
            let stats = inner
                .request_latency
                .get(route)
                .cloned()
                .unwrap_or_default();
            let mut cumulative = 0_u64;

            for (index, bucket) in LATENCY_BUCKETS_MS.iter().enumerate() {
                cumulative = cumulative.saturating_add(stats.bucket_counts[index]);
                let _ = writeln!(
                    output,
                    "doc2msg_request_duration_ms_bucket{{route=\"{route}\",le=\"{bucket}\"}} {cumulative}"
                );
            }

            cumulative = cumulative.saturating_add(stats.bucket_counts[LATENCY_BUCKETS_MS.len()]);
            let _ = writeln!(
                output,
                "doc2msg_request_duration_ms_bucket{{route=\"{route}\",le=\"+Inf\"}} {cumulative}"
            );
            let _ = writeln!(
                output,
                "doc2msg_request_duration_ms_sum{{route=\"{route}\"}} {}",
                stats.sum_ms
            );
            let _ = writeln!(
                output,
                "doc2msg_request_duration_ms_count{{route=\"{route}\"}} {}",
                stats.count
            );
        }

        output.push_str("# HELP doc2msg_ocr_requests_total Total OCR endpoint requests.\n");
        output.push_str("# TYPE doc2msg_ocr_requests_total counter\n");
        let _ = writeln!(
            output,
            "doc2msg_ocr_requests_total {}",
            inner.ocr_requests
        );

        output.push_str(
            "# HELP doc2msg_ocr_usage_total Total extraction operations that used OCR.\n",
        );
        output.push_str("# TYPE doc2msg_ocr_usage_total counter\n");
        let _ = writeln!(output, "doc2msg_ocr_usage_total {}", inner.ocr_usage);

        output.push_str("# HELP doc2msg_cache_hits_total Total cache hits by route.\n");
        output.push_str("# TYPE doc2msg_cache_hits_total counter\n");
        for route in CACHE_ROUTES {
            let count = inner.cache_hits.get(route).copied().unwrap_or(0);
            let _ = writeln!(
                output,
                "doc2msg_cache_hits_total{{route=\"{route}\"}} {count}"
            );
        }

        output.push_str("# HELP doc2msg_cache_misses_total Total cache misses by route.\n");
        output.push_str("# TYPE doc2msg_cache_misses_total counter\n");
        for route in CACHE_ROUTES {
            let count = inner.cache_misses.get(route).copied().unwrap_or(0);
            let _ = writeln!(
                output,
                "doc2msg_cache_misses_total{{route=\"{route}\"}} {count}"
            );
        }

        output.push_str("# HELP doc2msg_cache_hit_ratio Cache hit ratio by route.\n");
        output.push_str("# TYPE doc2msg_cache_hit_ratio gauge\n");
        for route in CACHE_ROUTES {
            let hits = inner.cache_hits.get(route).copied().unwrap_or(0);
            let misses = inner.cache_misses.get(route).copied().unwrap_or(0);
            let total = hits.saturating_add(misses);
            let ratio = if total == 0 {
                0.0
            } else {
                hits as f64 / total as f64
            };

            let _ = writeln!(
                output,
                "doc2msg_cache_hit_ratio{{route=\"{route}\"}} {ratio:.6}"
            );
        }

        output
    }

    fn lock_state(&self) -> std::sync::MutexGuard<'_, MetricsState> {
        match self.inner.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }
}

pub fn route_label(path: &str) -> &'static str {
    match path {
        ROUTE_HEALTH => ROUTE_HEALTH,
        ROUTE_METRICS => ROUTE_METRICS,
        ROUTE_FORMATS => ROUTE_FORMATS,
        ROUTE_EXTRACT_URL => ROUTE_EXTRACT_URL,
        ROUTE_EXTRACT_BYTES => ROUTE_EXTRACT_BYTES,
        ROUTE_OCR => ROUTE_OCR,
        _ => ROUTE_UNMATCHED,
    }
}
