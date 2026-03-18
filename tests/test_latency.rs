mod common;

use std::time::Instant;

use axum::body::Body;
use axum::http::Request;
use tower::ServiceExt;

use common::create_test_app;

/// Intent: Verify that the markdown passthrough pipeline (no parsing, no network)
/// completes a full round-trip in under 50ms, ensuring minimal overhead.
#[tokio::test]
#[ignore]
async fn test_markdown_passthrough_latency() {
    let app = create_test_app();

    let markdown = "# Benchmark Document\n\n\
        This is a paragraph used for latency benchmarking. It contains enough text \
        to exercise the chunker without being large enough to dominate timing. \
        Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod \
        tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, \
        quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo \
        consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse \
        cillum dolore eu fugiat nulla pariatur.";

    let request = Request::builder()
        .method("POST")
        .uri("/v1/extract/bytes")
        .header("content-type", "text/markdown")
        .body(Body::from(markdown))
        .unwrap();

    let start = Instant::now();
    let response = app.oneshot(request).await.unwrap();
    let _body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let elapsed = start.elapsed();

    eprintln!("markdown passthrough: {:?}", elapsed);
    assert!(
        elapsed.as_millis() < 50,
        "markdown passthrough took {elapsed:?}, expected < 50ms"
    );
}

/// Intent: Verify that the HTML extraction pipeline (readability + html2md)
/// completes a full round-trip in under 500ms for a typical article-sized page.
#[tokio::test]
#[ignore]
async fn test_html_extraction_latency() {
    let app = create_test_app();

    let html = r#"<html><head><title>Benchmark Article</title></head><body><article>
        <h1>Benchmark Article</h1>
        <p>This is the first paragraph of a benchmark article. It contains enough text to
        exercise the readability extraction and html2md conversion pipeline. The purpose is
        to measure end-to-end latency for a realistic HTML document.</p>
        <h2>Background</h2>
        <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor
        incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud
        exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.</p>
        <h2>Methods</h2>
        <p>Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu
        fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa
        qui officia deserunt mollit anim id est laborum.</p>
        <h2>Results</h2>
        <p>Curabitur pretium tincidunt lacus. Nulla gravida orci a odio. Nullam varius, turpis
        et commodo pharetra, est eros bibendum elit, nec luctus magna felis sollicitudin mauris.
        Integer in mauris eu nibh euismod gravida.</p>
        <h2>Conclusion</h2>
        <p>Praesent dapibus, neque id cursus faucibus, tortor neque egestas augue, eu vulputate
        magna eros eu erat. Aliquam erat volutpat. Nam dui mi, tincidunt quis, accumsan porttitor,
        facilisis luctus, metus.</p>
    </article></body></html>"#;

    let request = Request::builder()
        .method("POST")
        .uri("/v1/extract/bytes")
        .header("content-type", "text/html")
        .body(Body::from(html))
        .unwrap();

    let start = Instant::now();
    let response = app.oneshot(request).await.unwrap();
    let _body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let elapsed = start.elapsed();

    eprintln!("html extraction: {:?}", elapsed);
    assert!(
        elapsed.as_millis() < 500,
        "html extraction took {elapsed:?}, expected < 500ms"
    );
}

/// Intent: Verify that a cache hit on `/v1/extract/bytes` returns in under 5ms,
/// confirming the in-memory cache bypasses the extraction pipeline entirely.
#[tokio::test]
#[ignore]
async fn test_cache_hit_latency() {
    let app = create_test_app();

    let content =
        "# Cached Document\n\nThis content is submitted twice to measure cache hit speed.";

    // First request: populates the cache (cold miss).
    let req1 = Request::builder()
        .method("POST")
        .uri("/v1/extract/bytes")
        .header("content-type", "text/markdown")
        .body(Body::from(content))
        .unwrap();
    let resp1 = app.clone().oneshot(req1).await.unwrap();
    let _body1 = axum::body::to_bytes(resp1.into_body(), usize::MAX)
        .await
        .unwrap();

    // Second request: should be a cache hit.
    let req2 = Request::builder()
        .method("POST")
        .uri("/v1/extract/bytes")
        .header("content-type", "text/markdown")
        .body(Body::from(content))
        .unwrap();
    let start = Instant::now();
    let resp2 = app.oneshot(req2).await.unwrap();
    let _body2 = axum::body::to_bytes(resp2.into_body(), usize::MAX)
        .await
        .unwrap();
    let elapsed = start.elapsed();

    eprintln!("cache hit: {:?}", elapsed);
    assert!(
        elapsed.as_millis() < 5,
        "cache hit took {elapsed:?}, expected < 5ms"
    );
}
