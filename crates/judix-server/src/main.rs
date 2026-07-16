use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{Html, IntoResponse},
    routing::{get, post},
    Json, Router,
};
use judix_core::model::ModelClient;
use judix_core::scoring::{score_agent, score_rag};
use judix_core::types::{AgentTrace, RagTriple};
use serde_json::{json, Value};
use std::sync::Arc;

const INDEX_HTML: &str = include_str!("../../../web/index.html");
const DEMO_CLEAN: &str = include_str!("../../../demos/clean.json");
const DEMO_WRONG_TOOL: &str = include_str!("../../../demos/wrong_tool.json");
const DEMO_RAG: &str = include_str!("../../../demos/rag_hallucination.json");

#[derive(Clone)]
struct AppState {
    model: Option<Arc<ModelClient>>,
}

pub fn build_app() -> Router {
    let model = ModelClient::from_env().map(Arc::new);
    if model.is_some() {
        tracing::info!("model layer enabled (JUDIX_BASE_URL is set)");
    } else {
        tracing::info!("model layer disabled — deterministic-only mode");
    }
    let state = AppState { model };

    Router::new()
        .route("/health", get(health))
        .route("/", get(root))
        .route("/api", get(api_info))
        .route("/score/agent", post(score_agent_handler))
        .route("/score/rag", post(score_rag_handler))
        .route("/demo/{id}", get(demo_handler))
        .with_state(state)
}

async fn health() -> Json<Value> {
    Json(json!({ "ok": true, "service": "judix", "version": env!("CARGO_PKG_VERSION") }))
}

async fn root() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn api_info() -> Json<Value> {
    Json(json!({
        "service": "judix",
        "tagline": "Real-time, per-turn evaluation for AI agents & RAG.",
        "endpoints": {
            "GET /health": "liveness",
            "POST /score/agent": "score an agent trace",
            "POST /score/rag": "score a RAG triple",
            "GET /demo/:id": "clean | wrong_tool | rag_hallucination"
        }
    }))
}

async fn score_agent_handler(
    State(state): State<AppState>,
    Json(trace): Json<AgentTrace>,
) -> impl IntoResponse {
    let t0 = std::time::Instant::now();

    let model_metrics = match &state.model {
        Some(client) => client.score_agent_steps(&trace).await,
        None => vec![],
    };

    let latency = t0.elapsed().as_millis() as u64;
    let report = score_agent(&trace, &model_metrics, latency, 0.0);
    (StatusCode::OK, Json(json!(report)))
}

async fn score_rag_handler(
    State(state): State<AppState>,
    Json(triple): Json<RagTriple>,
) -> impl IntoResponse {
    let client = match &state.model {
        Some(c) => c,
        None => {
            return (
                StatusCode::NOT_IMPLEMENTED,
                Json(json!({
                    "status": "model_required",
                    "message": "RAG scoring needs the model layer. Set JUDIX_BASE_URL to your OmniRoute endpoint to enable it."
                })),
            )
        }
    };

    let t0 = std::time::Instant::now();
    match client.score_rag_triple(&triple).await {
        Ok((metrics, spans)) => {
            let latency = t0.elapsed().as_millis() as u64;
            let report = score_rag(metrics, spans, latency, 0.0);
            (StatusCode::OK, Json(json!(report)))
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("model error: {e}") })),
        ),
    }
}

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
