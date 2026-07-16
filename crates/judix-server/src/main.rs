//! Judix HTTP server (portable axum + tokio). Day-1 scope: `GET /health` so we
//! have a live public URL first (blueprint §11.1). Scoring routes + SSE land
//! Day 1.3 / Day 2.
//!
//! Deployed as a Docker image on any container host (Koyeb / Render / Fly / etc).
//! Binds `$PORT` (hosts inject it). The router is built by [`build_app`] so route
//! logic stays independent of the entrypoint.

use axum::{routing::get, Json, Router};
use serde_json::{json, Value};

/// Build the application router. Kept separate from `main` for reuse and testing.
pub fn build_app() -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/", get(root))
}

/// Liveness probe. Deployed first so there is a reachable URL from hour one, and
/// hit by the keep-warm ping to prevent free-tier scale-to-zero.
async fn health() -> Json<Value> {
    Json(json!({ "ok": true, "service": "judix", "version": env!("CARGO_PKG_VERSION") }))
}

/// Placeholder root until the web UI ships (Day 2, §11.5).
async fn root() -> Json<Value> {
    Json(json!({
        "service": "judix",
        "tagline": "Real-time, per-turn evaluation for AI agents & RAG. A deterministic Rust engine scores; a model only explains.",
        "endpoints": ["/health", "/score/agent (soon)", "/score/rag (soon)", "/demo/:id (soon)"]
    }))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "judix_server=info".into()),
        )
        .init();

    // Container hosts inject $PORT; default to 8000 for local dev.
    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8000);
    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));

    let app = build_app();
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("judix-server listening on http://{addr}");
    axum::serve(listener, app).await?;
    Ok(())
}
