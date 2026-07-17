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

    if let Some(client) = state.model.clone() {
        tokio::spawn(prewarm(client));
    }

    Router::new()
        .route("/health", get(health))
        .route("/", get(root))
        .route("/api", get(api_info))
        .route("/score/agent", post(score_agent_handler))
        .route("/score/rag", post(score_rag_handler))
        .route("/demo/{id}", get(demo_handler))
        .with_state(state)
}

/// Score the built-in demo fixtures once at boot, in the background, so the first
/// visitor never pays the cold path.
///
/// Measured: a warm RAG score is ~0.5s, but the very first one took 13.7s in
/// production vs 2.24s locally. The gap is the free tier's 0.1 CPU — the initial TLS
/// handshake to the provider plus JSON work on a starved core, and RAG needs two
/// sequential model waves (decompose, then verify), so that cost is paid twice. A
/// judge's first click is exactly that cold path.
///
/// This warms both the HTTPS connection pool and the response cache, so the demo
/// buttons are instant. It's deliberately fire-and-forget: it must never delay
/// startup or the health check, and a failure here is harmless (the next real
/// request just pays the normal cost).
async fn prewarm(client: Arc<ModelClient>) {
    let t0 = std::time::Instant::now();

    for (name, raw) in [("clean", DEMO_CLEAN), ("wrong_tool", DEMO_WRONG_TOOL)] {
        match serde_json::from_str::<AgentTrace>(raw) {
            Ok(trace) => {
                client.score_agent_steps(&trace).await;
                tracing::info!(demo = name, "prewarmed");
            }
            Err(e) => tracing::warn!(demo = name, error = %e, "prewarm parse failed"),
        }
    }
    match serde_json::from_str::<RagTriple>(DEMO_RAG) {
        Ok(triple) => match client.score_rag_triple(&triple).await {
            Ok(_) => tracing::info!(demo = "rag_hallucination", "prewarmed"),
            Err(e) => tracing::warn!(demo = "rag_hallucination", error = %e, "prewarm failed"),
        },
        Err(e) => tracing::warn!(demo = "rag_hallucination", error = %e, "prewarm parse failed"),
    }

    tracing::info!(ms = t0.elapsed().as_millis() as u64, "demo prewarm complete");
}

/// Liveness + config visibility. `model_layer` reports whether the explanation
/// layer is live, so a deploy can be verified without poking a scoring endpoint
/// (deterministic scoring works either way; only the model metrics need a key).
async fn health(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "ok": true,
        "service": "judix",
        "version": env!("CARGO_PKG_VERSION"),
        "model_layer": if state.model.is_some() { "enabled" } else { "disabled (set JUDIX_API_KEY)" },
        "model_fast": std::env::var("JUDIX_MODEL_FAST").unwrap_or_else(|_| "-".into()),
    }))
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

    let (model_metrics, cost) = match &state.model {
        Some(client) => client.score_agent_steps(&trace).await,
        None => (vec![], 0.0),
    };

    let latency = t0.elapsed().as_millis() as u64;
    let report = score_agent(&trace, &model_metrics, latency, cost);
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
                    "message": "RAG scoring needs the model layer (claim decomposition + verification). Set JUDIX_BASE_URL and JUDIX_API_KEY to an OpenAI-compatible endpoint (e.g. Gemini) to enable it."
                })),
            )
        }
    };

    let t0 = std::time::Instant::now();
    match client.score_rag_triple(&triple).await {
        Ok((metrics, spans, any_contradiction, cost)) => {
            let latency = t0.elapsed().as_millis() as u64;
            let report = score_rag(metrics, spans, any_contradiction, latency, cost);
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
