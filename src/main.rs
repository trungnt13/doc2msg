use std::sync::Arc;
use std::time::Duration;

use clap::Parser;

use doc2agent::config::RuntimeConfig;
use doc2agent::server::{build_router, AppState};
use doc2agent::CliArgs;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "doc2agent=info,tower_http=info".into()),
        )
        .init();

    let args = CliArgs::parse();
    let config = RuntimeConfig::from_cli(&args);

    let state = Arc::new(AppState::new(config.clone()));
    let app = build_router(Arc::clone(&state));

    let addr = format!("{}:{}", config.host, config.port);
    tracing::info!(addr = %addr, "starting doc2agent server");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
    let shutdown_state = Arc::clone(&state);
    let drain_timeout = Duration::from_secs(config.request_timeout_secs);

    tokio::spawn(async move {
        await_shutdown_signal(&shutdown_state, shutdown_tx, drain_timeout).await;
    });

    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = shutdown_rx.await;
        })
        .await?;
    tracing::info!("doc2agent server stopped");
    Ok(())
}

async fn await_shutdown_signal(
    state: &AppState,
    shutdown_tx: tokio::sync::oneshot::Sender<()>,
    drain_timeout: Duration,
) {
    if let Err(error) = tokio::signal::ctrl_c().await {
        tracing::error!(error = %error, "failed to install signal handler");
        return;
    }

    let first_signal = state.begin_shutdown();
    let in_flight = state.in_flight_requests();
    tracing::info!(
        in_flight,
        timeout_secs = drain_timeout.as_secs(),
        first_signal,
        "shutdown signal received; stopping new work and draining in-flight requests"
    );

    let _ = shutdown_tx.send(());

    match state.wait_for_drain(drain_timeout).await {
        Ok(()) => tracing::info!("all in-flight requests drained before timeout"),
        Err(remaining) => tracing::warn!(
            remaining_in_flight = remaining,
            timeout_secs = drain_timeout.as_secs(),
            "timed out waiting for in-flight requests to drain"
        ),
    }
}
