use std::cmp;
use std::io::Cursor;
use std::time::Instant;

use anyhow::{anyhow, ensure, Context};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use clap::Parser;
use image::{DynamicImage, ImageBuffer, ImageFormat, Rgb};
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Debug, Parser)]
#[command(
    name = "ocr_benchmark",
    about = "Benchmark the /v1/ocr endpoint with synthetic image payloads"
)]
struct Args {
    /// Base URL for the Doc2Agent server.
    #[arg(long, default_value = "http://127.0.0.1:3000")]
    base_url: String,

    /// Number of images sent in each OCR request.
    #[arg(long)]
    batch_size: usize,

    /// Number of concurrent client workers.
    #[arg(long, default_value_t = 1)]
    concurrency: usize,

    /// Timed requests executed by each worker.
    #[arg(long, default_value_t = 10)]
    requests_per_worker: usize,

    /// Untimed warmup requests executed before measurement.
    #[arg(long, default_value_t = 3)]
    warmup_requests: usize,

    /// Synthetic image width in pixels.
    #[arg(long, default_value_t = 320)]
    image_width: u32,

    /// Synthetic image height in pixels.
    #[arg(long, default_value_t = 48)]
    image_height: u32,

    /// Optional label describing the execution path, e.g. cpu or cuda-ep.
    #[arg(long, default_value = "unknown")]
    runtime_label: String,

    /// Session pool size used by the server for this run.
    #[arg(long)]
    session_pool_size: usize,

    /// Whether GPU process detection was observed for the server.
    #[arg(long, default_value_t = false)]
    gpu_process_detected: bool,

    /// GPU memory used by the server process, if observed.
    #[arg(long)]
    gpu_memory_mib: Option<u64>,

    /// Pretty-print JSON output.
    #[arg(long, default_value_t = false)]
    pretty: bool,
}

#[derive(Debug, Serialize)]
struct BenchmarkRun {
    runtime_label: String,
    base_url: String,
    session_pool_size: usize,
    batch_size: usize,
    concurrency: usize,
    requests_per_worker: usize,
    warmup_requests: usize,
    total_requests: usize,
    total_images: usize,
    image_width: u32,
    image_height: u32,
    response_model: String,
    gpu_process_detected: bool,
    gpu_memory_mib: Option<u64>,
    observed_latency_ms: StatsSummary,
    server_latency_ms: StatsSummary,
    request_throughput_per_sec: f64,
    image_throughput_per_sec: f64,
}

#[derive(Debug, Clone, Serialize)]
struct StatsSummary {
    count: usize,
    min: f64,
    max: f64,
    mean: f64,
    p50: f64,
    p95: f64,
    p99: f64,
}

#[derive(Debug, Clone)]
struct Sample {
    observed_latency_ms: f64,
    server_latency_ms: f64,
    response_model: String,
}

#[derive(Debug, Deserialize)]
struct OcrResponse {
    model: String,
    batch_size: usize,
    latency_ms: u64,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    validate_args(&args)?;

    let client = Client::builder()
        .pool_max_idle_per_host(cmp::max(args.concurrency, 1))
        .build()
        .context("failed to build benchmark HTTP client")?;
    let payload_images = build_payload_images(args.batch_size, args.image_width, args.image_height)
        .context("failed to build synthetic OCR payload images")?;

    for _ in 0..args.warmup_requests {
        send_request(&client, &args.base_url, &payload_images).await?;
    }

    let started_at = Instant::now();
    let mut tasks = tokio::task::JoinSet::new();
    for _worker in 0..args.concurrency {
        let client = client.clone();
        let base_url = args.base_url.clone();
        let payload_images = payload_images.clone();
        let requests_per_worker = args.requests_per_worker;
        tasks.spawn(async move {
            let mut samples = Vec::with_capacity(requests_per_worker);
            for _request in 0..requests_per_worker {
                samples.push(send_request(&client, &base_url, &payload_images).await?);
            }
            Ok::<Vec<Sample>, anyhow::Error>(samples)
        });
    }

    let mut samples = Vec::with_capacity(args.concurrency * args.requests_per_worker);
    while let Some(joined) = tasks.join_next().await {
        let worker_samples =
            joined.map_err(|error| anyhow!("benchmark worker task failed: {error}"))??;
        samples.extend(worker_samples);
    }
    let elapsed_secs = started_at.elapsed().as_secs_f64();
    ensure!(
        !samples.is_empty(),
        "benchmark produced no samples; increase requests-per-worker"
    );

    let response_model = samples[0].response_model.clone();
    ensure!(
        samples
            .iter()
            .all(|sample| sample.response_model == response_model),
        "benchmark responses used inconsistent OCR models"
    );

    let observed_latencies = samples
        .iter()
        .map(|sample| sample.observed_latency_ms)
        .collect::<Vec<_>>();
    let server_latencies = samples
        .iter()
        .map(|sample| sample.server_latency_ms)
        .collect::<Vec<_>>();

    let total_requests = samples.len();
    let total_images = total_requests * args.batch_size;
    let run = BenchmarkRun {
        runtime_label: args.runtime_label,
        base_url: trim_trailing_slash(&args.base_url).to_string(),
        session_pool_size: args.session_pool_size,
        batch_size: args.batch_size,
        concurrency: args.concurrency,
        requests_per_worker: args.requests_per_worker,
        warmup_requests: args.warmup_requests,
        total_requests,
        total_images,
        image_width: args.image_width,
        image_height: args.image_height,
        response_model,
        gpu_process_detected: args.gpu_process_detected,
        gpu_memory_mib: args.gpu_memory_mib,
        observed_latency_ms: summarize(&observed_latencies),
        server_latency_ms: summarize(&server_latencies),
        request_throughput_per_sec: total_requests as f64 / elapsed_secs,
        image_throughput_per_sec: total_images as f64 / elapsed_secs,
    };

    let json = if args.pretty {
        serde_json::to_string_pretty(&run)
    } else {
        serde_json::to_string(&run)
    }
    .context("failed to serialize benchmark run")?;
    println!("{json}");
    Ok(())
}

fn validate_args(args: &Args) -> anyhow::Result<()> {
    ensure!(args.batch_size > 0, "batch-size must be greater than zero");
    ensure!(
        args.concurrency > 0,
        "concurrency must be greater than zero"
    );
    ensure!(
        args.requests_per_worker > 0,
        "requests-per-worker must be greater than zero"
    );
    ensure!(
        args.image_width > 0,
        "image-width must be greater than zero"
    );
    ensure!(
        args.image_height > 0,
        "image-height must be greater than zero"
    );
    ensure!(
        args.session_pool_size > 0,
        "session-pool-size must be greater than zero"
    );
    Ok(())
}

fn trim_trailing_slash(base_url: &str) -> &str {
    base_url.trim_end_matches('/')
}

fn build_payload_images(batch_size: usize, width: u32, height: u32) -> anyhow::Result<Vec<String>> {
    let variant_count = cmp::max(batch_size.min(8), 1);
    let variants = (0..variant_count)
        .map(|variant| synthetic_image_data_url(width, height, variant))
        .collect::<anyhow::Result<Vec<_>>>()?;

    Ok((0..batch_size)
        .map(|index| variants[index % variants.len()].clone())
        .collect())
}

fn synthetic_image_data_url(width: u32, height: u32, variant: usize) -> anyhow::Result<String> {
    let mut image = ImageBuffer::from_pixel(width, height, Rgb([255, 255, 255]));

    let margin = 4 + (variant as u32 % 3);
    let stripe_height = cmp::max(4, height / 6);
    let block_width = cmp::max(6, width / 12);
    let ink = [
        24_u8.saturating_add((variant as u8).saturating_mul(7)),
        24,
        24_u8.saturating_add((variant as u8).saturating_mul(5)),
    ];

    for x in margin..width.saturating_sub(margin) {
        image.put_pixel(x, margin, Rgb([210, 210, 210]));
        image.put_pixel(x, height.saturating_sub(margin + 1), Rgb([210, 210, 210]));
    }

    for y in margin..height.saturating_sub(margin) {
        image.put_pixel(margin, y, Rgb([210, 210, 210]));
        image.put_pixel(width.saturating_sub(margin + 1), y, Rgb([210, 210, 210]));
    }

    for line in 0..3_u32 {
        let top = margin + line * (stripe_height + 2);
        if top >= height.saturating_sub(margin) {
            break;
        }

        let y_end = cmp::min(top + stripe_height, height.saturating_sub(margin));
        let offset = ((variant as u32 + line) * 11) % block_width.max(1);
        for y in top..y_end {
            for x in margin..width.saturating_sub(margin) {
                let column = x.saturating_sub(margin) + offset;
                let is_ink = column % block_width < block_width.saturating_sub(2)
                    && column % (block_width * 2).max(1) < block_width;
                if is_ink {
                    image.put_pixel(x, y, Rgb(ink));
                }
            }
        }
    }

    let dynamic = DynamicImage::ImageRgb8(image);
    let mut cursor = Cursor::new(Vec::new());
    dynamic
        .write_to(&mut cursor, ImageFormat::Png)
        .context("failed to encode synthetic OCR benchmark image")?;
    Ok(format!(
        "data:image/png;base64,{}",
        STANDARD.encode(cursor.into_inner())
    ))
}

async fn send_request(
    client: &Client,
    base_url: &str,
    payload_images: &[String],
) -> anyhow::Result<Sample> {
    let url = format!("{}/v1/ocr", trim_trailing_slash(base_url));
    let started_at = Instant::now();
    let response = client
        .post(url)
        .json(&serde_json::json!({ "images": payload_images }))
        .send()
        .await
        .context("failed to send OCR benchmark request")?;
    let observed_latency_ms = started_at.elapsed().as_secs_f64() * 1000.0;
    let status = response.status();
    let body = response
        .bytes()
        .await
        .context("failed to read OCR benchmark response body")?;

    if !status.is_success() {
        let body = String::from_utf8_lossy(&body);
        return Err(anyhow!(
            "OCR benchmark request failed with status {status}: {body}"
        ));
    }

    let parsed: OcrResponse =
        serde_json::from_slice(&body).context("failed to parse OCR benchmark response JSON")?;
    ensure!(
        parsed.batch_size == payload_images.len(),
        "OCR response batch size {} did not match request batch size {}",
        parsed.batch_size,
        payload_images.len()
    );

    Ok(Sample {
        observed_latency_ms,
        server_latency_ms: parsed.latency_ms as f64,
        response_model: parsed.model,
    })
}

fn summarize(samples: &[f64]) -> StatsSummary {
    let mut sorted = samples.to_vec();
    sorted.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    let count = sorted.len();
    let min = *sorted.first().unwrap_or(&0.0);
    let max = *sorted.last().unwrap_or(&0.0);
    let mean = if count == 0 {
        0.0
    } else {
        sorted.iter().sum::<f64>() / count as f64
    };

    StatsSummary {
        count,
        min,
        max,
        mean,
        p50: percentile(&sorted, 0.50),
        p95: percentile(&sorted, 0.95),
        p99: percentile(&sorted, 0.99),
    }
}

fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }

    let index = ((sorted.len() - 1) as f64 * quantile).round() as usize;
    sorted[index.min(sorted.len() - 1)]
}
