//! Judix HTTP server (portable axum + tokio). Deployed as a Docker image on any
//! container host (Render / Koyeb / Fly / HF Spaces). Binds `$PORT`.
//!
//! Routes:
//!   GET  /health        liveness probe (also hit by the keep-warm ping)
//!   GET  /              service info
//!   POST /score/agent   deterministic agent-trajectory scoring (no key needed)
//!   POST /score/rag     RAG scoring (needs the model layer — stubbed until key)
//!   GET  /demo/:id      built-in demo fixtures (clean|wrong_tool|rag_hallucination)

use axum::{
    extract::Path,
    http::StatusCode,
    response::{Html, IntoResponse},
    routing::{get, post},
    Json, Router,
};
use judix_core::scoring::score_agent;
use judix_core::types::AgentTrace;
use serde_json::{json, Value};

// Assets embedded at compile time so they work in the slim runtime image (which
// doesn't ship the source `web/` or `demos/` directories).
const INDEX_HTML: &str = include_str!("../../../web/index.html");
const DEMO_CLEAN: &str = include_str!("../../../demos/clean.json");
const DEMO_WRONG_TOOL: &str = include_str!("../../../demos/wrong_tool.json");
const DEMO_RAG: &str = include_str!("../../../demos/rag_hallucination.json");

/// Build the application router. Kept separate from `main` for reuse and testing.
pub fn build_app() -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/", get(root))
        .route("/api", get(api_info))
        .route("/score/agent", post(score_agent_handler))
        .route("/score/rag", post(score_rag_handler))
        .route("/demo/{id}", get(demo_handler))
}

async fn health() -> Json<Value> {
    Json(json!({ "ok": true, "service": "judix", "version": env!("CARGO_PKG_VERSION") }))
}

/// Serve the playground UI at the root so judges see the visual scorer, not JSON.
async fn root() -> Html<&'static str> {
    Html(INDEX_HTML)
}

/// Machine-readable service/endpoint info (the old root payload).
async fn api_info() -> Json<Value> {
    Json(json!({
        "service": "judix",
        "tagline": "Real-time, per-turn evaluation for AI agents & RAG. A deterministic Rust engine scores; a model only explains.",
        "endpoints": {
            "GET /health": "liveness",
            "POST /score/agent": "score an agent trace (deterministic; model explanations when JUDIX_API_KEY is set)",
            "POST /score/rag": "score a RAG triple (requires the model layer)",
            "GET /demo/:id": "clean | wrong_tool | rag_hallucination"
        }
    }))
}

/// Score an agent trace. The deterministic metrics (tool-call F1, loop detection)
/// are computed with zero model calls; `step_relevance`/`goal_drift` will be added
/// by the model layer once `JUDIX_API_KEY` is configured (Day 1.3).
async fn score_agent_handler(Json(trace): Json<AgentTrace>) -> impl IntoResponse {
    // No model metrics yet — deterministic-first. `deterministic_share` will read
    // 100% until the model layer is wired in.
    let report = score_agent(&trace, &[], 0, 0.0);
    (StatusCode::OK, Json(json!(report)))
}

/// RAG scoring is model-dependent (claim decomposition + verification), so until
/// the model layer ships this returns a clear, honest `model_required` signal
/// rather than fabricating a score.
async fn score_rag_handler(Json(_triple): Json<Value>) -> impl IntoResponse {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(json!({
            "status": "model_required",
            "message": "RAG faithfulness needs the model layer (claim decomposition + verification). Set JUDIX_API_KEY to enable it."
        })),
    )
}

/// Serve a built-in demo fixture so the playground can one-click load examples.
async fn demo_handler(Path(id): Path<String>) -> impl IntoResponse {
    let body = match id.as_str() {
        "clean" => DEMO_CLEAN,
        "wrong_tool" => DEMO_WRONG_TOOL,
        "rag_hallucination" => DEMO_RAG,
        _ => {
            return (
                StatusCode::NOT_FOUND,
                Json(json!({ "error": "unknown demo id", "valid": ["clean", "wrong_tool", "rag_hallucination"] })),
            )
        }
    };
    // The fixtures are valid JSON; parse so we return application/json, not text.
    match serde_json::from_str::<Value>(body) {
        Ok(v) => (StatusCode::OK, Json(v)),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("demo parse error: {e}") })),
        ),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "judix_server=info".into()),
        )
        .init();

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
