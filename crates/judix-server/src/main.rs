use axum::{
    extract::{Path, Request, State},
    http::{HeaderMap, StatusCode},
    middleware::{self, Next},
    response::sse::{Event, KeepAlive, Sse},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use futures::{Stream, StreamExt};
use judix_core::model::{ModelClient, RagEvent};
use judix_core::scoring::{score_agent, score_rag};
use judix_core::types::{AgentTrace, MetricResult, RagTriple};
use serde_json::{json, Value};
use std::convert::Infallible;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

const INDEX_HTML: &str = include_str!("../../../web/index.html");
const DEMO_CLEAN: &str = include_str!("../../../demos/clean.json");
const DEMO_WRONG_TOOL: &str = include_str!("../../../demos/wrong_tool.json");
const DEMO_RAG: &str = include_str!("../../../demos/rag_hallucination.json");

/// Scoring requests allowed per client IP per minute, before `429`.
///
/// Sized against the real cost, not a round number: with `JUDIX_MAX_MODEL_STEPS` at 40 a
/// single request can spend 80 model calls, and the free tiers total ~1-2k/day. 20/min is
/// far more than a human clicking demos will ever need (the playground fires one request
/// per click) while turning "one curl loop drains the quota" into something an attacker
/// has to work at from many IPs.
const DEFAULT_RATE_LIMIT_PER_MIN: u32 = 20;

#[derive(Clone)]
struct AppState {
    model: Option<Arc<ModelClient>>,
    /// Fixed-window hit counts per client IP. Entries expire 60s after first insert,
    /// which *is* the window reset — no sweeper task needed.
    hits: moka::future::Cache<String, Arc<AtomicU32>>,
    rate_limit: u32,
}

/// The client's IP as seen through Render's Cloudflare edge.
///
/// Deliberately NOT `ConnectInfo<SocketAddr>`: behind a proxy that returns the *edge's*
/// address, so every visitor on earth would share one bucket and the first judge to click
/// twice would rate-limit everyone else.
///
/// Header order is a security decision. `CF-Connecting-IP` is written by Cloudflare and
/// overwrites anything the client sent, so it can be trusted. `X-Forwarded-For`'s first
/// entry is client-supplied — trusting it first would let an attacker forge a fresh
/// identity per request and bypass the limiter entirely. It's only a fallback for running
/// behind a different proxy.
fn client_ip(headers: &HeaderMap) -> String {
    for name in ["cf-connecting-ip", "x-forwarded-for"] {
        if let Some(v) = headers.get(name).and_then(|v| v.to_str().ok()) {
            let first = v.split(',').next().unwrap_or(v).trim();
            if !first.is_empty() {
                return first.to_string();
            }
        }
    }
    // No proxy headers (e.g. local dev): one shared bucket is fine.
    "direct".to_string()
}

/// Fixed-window per-IP rate limit, applied only to the endpoints that spend money.
async fn rate_limit_mw(
    State(state): State<AppState>,
    req: Request,
    next: Next,
) -> Result<Response, (StatusCode, [(&'static str, &'static str); 1], Json<Value>)> {
    let ip = client_ip(req.headers());
    let counter = state
        .hits
        .get_with(ip.clone(), async { Arc::new(AtomicU32::new(0)) })
        .await;
    let n = counter.fetch_add(1, Ordering::Relaxed) + 1;

    if n > state.rate_limit {
        tracing::warn!(ip = %ip, hits = n, limit = state.rate_limit, "rate limited");
        return Err((
            StatusCode::TOO_MANY_REQUESTS,
            [("retry-after", "60")],
            Json(json!({
                "error": "rate_limited",
                "message": format!(
                    "More than {} scoring requests in a minute. Scoring spends model calls on a free tier, so it's capped per IP. Retry in 60s — /demo/* and /health are not limited.",
                    state.rate_limit
                ),
            })),
        ));
    }
    Ok(next.run(req).await)
}

pub fn build_app() -> Router {
    let model = ModelClient::from_env().map(Arc::new);
    if model.is_some() {
        tracing::info!("model layer enabled (JUDIX_BASE_URL is set)");
    } else {
        tracing::info!("model layer disabled — deterministic-only mode");
    }
    let rate_limit = std::env::var("JUDIX_RATE_LIMIT_PER_MIN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_RATE_LIMIT_PER_MIN);
    let state = AppState {
        model,
        // TTL == the window: an entry created by the first hit expires 60s later, which
        // resets the count. Bounded at 10k IPs so the limiter can't itself be a memory
        // DoS via spoofed/rotating addresses.
        hits: moka::future::Cache::builder()
            .max_capacity(10_000)
            .time_to_live(Duration::from_secs(60))
            .build(),
        rate_limit,
    };
    tracing::info!(rate_limit, "scoring rate limit (per IP per minute)");

    if let Some(client) = state.model.clone() {
        tokio::spawn(prewarm(client));
    }

    // Only the routes that spend model calls are limited. /health must stay open — the
    // keep-warm pinger hits it every 10 min and a 429 there would let the service sleep.
    // /demo/* and / are static and free.
    let scoring = Router::new()
        .route("/score/agent", post(score_agent_handler))
        .route("/score/rag", post(score_rag_handler))
        // Streaming variants (§9.5). Kept as separate routes rather than content
        // negotiation on the existing paths, so the documented JSON API — the CLI,
        // scripts/stress.sh, every curl in the README — keeps working untouched.
        .route("/score/agent/stream", post(score_agent_stream))
        .route("/score/rag/stream", post(score_rag_stream))
        .route_layer(middleware::from_fn_with_state(state.clone(), rate_limit_mw));

    Router::new()
        .route("/health", get(health))
        .route("/", get(root))
        .route("/api", get(api_info))
        .route("/demo/{id}", get(demo_handler))
        .merge(scoring)
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
    // Pace between fixtures. Each agent demo fires 10 concurrent calls, so warming all
    // three back-to-back burst ~25 requests and rate-limited *itself* — the RAG demo ran
    // last and lost, failing to warm at all, which left the money demo cold for the very
    // first judge. Nothing is waiting on this task, so spending a minute is free.
    const GAP: Duration = Duration::from_secs(8);

    let t0 = std::time::Instant::now();

    for (name, raw) in [("clean", DEMO_CLEAN), ("wrong_tool", DEMO_WRONG_TOOL)] {
        match serde_json::from_str::<AgentTrace>(raw) {
            Ok(trace) => {
                client.score_agent_steps(&trace).await;
                tracing::info!(demo = name, "prewarmed");
            }
            Err(e) => tracing::warn!(demo = name, error = %e, "prewarm parse failed"),
        }
        tokio::time::sleep(GAP).await;
    }

    match serde_json::from_str::<RagTriple>(DEMO_RAG) {
        Ok(triple) => {
            // Retry once after a longer pause: this is the RAG money demo, and a cold
            // first click on it costs ~13s.
            for attempt in 0..2 {
                match client.score_rag_triple(&triple).await {
                    Ok(_) => {
                        tracing::info!(demo = "rag_hallucination", "prewarmed");
                        break;
                    }
                    Err(e) => {
                        tracing::warn!(demo = "rag_hallucination", attempt, error = %e, "prewarm failed");
                        tokio::time::sleep(GAP * 2).await;
                    }
                }
            }
        }
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
        // Just the size of the failover pool, not the model names: a mis-pasted
        // JUDIX_EXTRA_PROVIDERS still shows up as a smaller-than-expected count
        // (the same silent-degradation signal), without exposing providers publicly.
        "model_pool_size": state.model.as_ref().map(|m| m.pool_summary().len()).unwrap_or(0),
        // Which commit is ACTUALLY serving traffic. Render injects RENDER_GIT_COMMIT.
        // Without this a stale deploy is invisible — this service silently served a
        // build from ~10 commits back for a full day because nothing reported the
        // running SHA. The deploy workflow polls this to prove a deploy really landed.
        "commit": std::env::var("RENDER_GIT_COMMIT")
            .ok()
            .map(|c| c.chars().take(7).collect::<String>())
            .unwrap_or_else(|| "local".into()),
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

/// Boundary guards: reject inputs that cannot be meaningfully scored, with a clear 422,
/// before spending any model calls. These validate the request shape only; the
/// deterministic scoring engine is untouched.
fn validate_trace(trace: &AgentTrace) -> Result<(), String> {
    for (i, s) in trace.steps.iter().enumerate() {
        if s.kind == "tool_call" {
            let named = s.name.as_ref().map(|n| !n.trim().is_empty()).unwrap_or(false);
            if !named {
                return Err(format!(
                    "step {i}: a \"tool_call\" step must carry a non-empty \"name\"."
                ));
            }
            if let Some(a) = &s.args {
                if !a.is_object() && !a.is_null() {
                    return Err(format!(
                        "step {i}: \"args\" must be a JSON object, e.g. {{\"area\":\"downtown\"}}."
                    ));
                }
            }
        }
    }
    Ok(())
}

fn validate_triple(t: &RagTriple) -> Result<(), String> {
    if t.answer.trim().is_empty() {
        return Err("\"answer\" must not be empty; there is nothing to ground.".to_string());
    }
    if t.contexts.iter().all(|c| c.trim().is_empty()) {
        return Err("\"contexts\" must contain at least one non-empty source.".to_string());
    }
    Ok(())
}

async fn score_agent_handler(
    State(state): State<AppState>,
    Json(trace): Json<AgentTrace>,
) -> impl IntoResponse {
    if let Err(msg) = validate_trace(&trace) {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(json!({ "error": "invalid_input", "message": msg })),
        );
    }
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
    if let Err(msg) = validate_triple(&triple) {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(json!({ "error": "invalid_input", "message": msg })),
        );
    }
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

// ---------------------------------------------------------------------------
// SSE streaming (§9.5)
// ---------------------------------------------------------------------------
//
// The product claim is "real-time": the deterministic engine has an answer in ~1ms, but
// the JSON endpoints hold it hostage until the slowest model call returns. Streaming
// lets the part that costs $0 and needs no network paint immediately, with each model
// explanation layering in as it arrives.
//
// Event protocol (each `data:` is one JSON object):
//   event: deterministic  → a complete AgentReport from the engine alone. Render it NOW.
//   event: metric         → {step_index, metric} — one model metric that just landed.
//   event: claims         → {claims:[…]} — RAG only: decomposed claims + spans.
//   event: done           → the final recomposed report (weights + hard caps applied).
//   event: error          → {message} — terminal.
//
// Client note: browsers cannot POST with `EventSource`, so consume this with `fetch()`
// + a ReadableStream reader, not `new EventSource(...)`.

/// Serialize a value into an SSE event, degrading to an `error` event rather than
/// killing the stream.
fn sse_event(name: &str, value: &Value) -> Event {
    match Event::default().event(name).json_data(value) {
        Ok(e) => e,
        Err(e) => Event::default()
            .event("error")
            .data(json!({ "message": format!("serialize {name}: {e}") }).to_string()),
    }
}

/// Stream an agent score: deterministic metrics first, model metrics as they land.
async fn score_agent_stream(
    State(state): State<AppState>,
    Json(trace): Json<AgentTrace>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = async_stream::stream! {
        if let Err(msg) = validate_trace(&trace) {
            yield Ok(sse_event("error", &json!({ "status": "invalid_input", "message": msg })));
            return;
        }
        let t0 = std::time::Instant::now();

        // 1. The hero: real scores, zero model calls, ~1ms — on screen before any
        //    network round-trip has even started.
        let det = score_agent(&trace, &[], t0.elapsed().as_millis() as u64, 0.0);
        yield Ok(sse_event("deterministic", &json!(det)));

        let Some(client) = state.model.clone() else {
            // Keyless: the deterministic report IS the final report.
            yield Ok(sse_event("done", &json!(det)));
            return;
        };

        // 2. Model metrics, emitted in completion order (not step order).
        let mut per_step: Vec<Vec<MetricResult>> = vec![Vec::new(); trace.steps.len()];
        let mut cost = 0.0f64;
        let metrics = client.stream_agent_metrics(&trace);
        futures::pin_mut!(metrics);
        while let Some((step_index, metric, c)) = metrics.next().await {
            cost += c;
            yield Ok(sse_event("metric", &json!({
                "step_index": step_index,
                "metric": metric,
            })));
            if let Some(slot) = per_step.get_mut(step_index) {
                slot.push(metric);
            }
        }

        // 3. Recompose with the model metrics included so weights and hard caps apply to
        //    the full picture — streamed metrics are individual facts, this is the verdict.
        let full = score_agent(&trace, &per_step, t0.elapsed().as_millis() as u64, cost);
        yield Ok(sse_event("done", &json!(full)));
    };

    Sse::new(stream).keep_alive(KeepAlive::default())
}

/// Stream a RAG score. There are no deterministic RAG metrics to lead with (the
/// faithfulness *ratio* is computed in Rust, but only from model verdicts), so this
/// emits the claim decomposition as soon as it exists — a judge sees the answer broken
/// into claims and spans before the composite verdict is assembled.
async fn score_rag_stream(
    State(state): State<AppState>,
    Json(triple): Json<RagTriple>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = async_stream::stream! {
        if let Err(msg) = validate_triple(&triple) {
            yield Ok(sse_event("error", &json!({ "status": "invalid_input", "message": msg })));
            return;
        }
        let t0 = std::time::Instant::now();

        let Some(client) = state.model.clone() else {
            yield Ok(sse_event("error", &json!({
                "status": "model_required",
                "message": "RAG scoring needs the model layer. Set JUDIX_BASE_URL and JUDIX_API_KEY.",
            })));
            return;
        };

        let events = client.stream_rag(&triple);
        futures::pin_mut!(events);
        while let Some(ev) = events.next().await {
            match ev {
                RagEvent::Claims(claims) => {
                    yield Ok(sse_event("claims", &json!({ "claims": claims })));
                }
                RagEvent::Metric(m) => {
                    yield Ok(sse_event("metric", &json!({ "metric": m })));
                }
                RagEvent::Done { metrics, spans, any_contradiction, cost } => {
                    let report = score_rag(
                        metrics, spans, any_contradiction,
                        t0.elapsed().as_millis() as u64, cost,
                    );
                    yield Ok(sse_event("done", &json!(report)));
                }
                RagEvent::Error(e) => {
                    yield Ok(sse_event("error", &json!({ "message": e })));
                }
            }
        }
    };

    Sse::new(stream).keep_alive(KeepAlive::default())
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

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{HeaderMap, HeaderValue, Request, StatusCode};
    use http_body_util::BodyExt;
    use tower::ServiceExt; // for `oneshot`

    // Serializes tests that mutate process-global env, so a rate-limit test's
    // JUDIX_RATE_LIMIT_PER_MIN can't bleed into a concurrent test's build_app().
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Build the router under a given env, keyless. The lock is held ONLY across the
    /// synchronous env-read + `build_app()` (all env is consumed there), and dropped
    /// before the caller awaits — so no MutexGuard ever crosses an `.await`.
    fn app_with(env: &[(&str, &str)]) -> Router {
        let _g = ENV_LOCK.lock().unwrap();
        std::env::remove_var("JUDIX_API_KEY");
        for (k, v) in env {
            std::env::set_var(k, v);
        }
        let app = build_app();
        for (k, _) in env {
            std::env::remove_var(k);
        }
        app
    }

    fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
        use axum::http::HeaderName;
        let mut h = HeaderMap::new();
        for (k, v) in pairs {
            h.insert(
                HeaderName::from_bytes(k.as_bytes()).unwrap(),
                HeaderValue::from_str(v).unwrap(),
            );
        }
        h
    }

    // --- client_ip: the anti-spoofing invariant the whole rate limiter rests on ---

    #[test]
    fn client_ip_prefers_cf_over_xff() {
        // CF-Connecting-IP is written by Cloudflare; XFF's first hop is client-supplied.
        // Trusting XFF first would let an attacker forge a fresh identity per request.
        let h = headers(&[("cf-connecting-ip", "1.1.1.1"), ("x-forwarded-for", "9.9.9.9")]);
        assert_eq!(client_ip(&h), "1.1.1.1");
    }

    #[test]
    fn client_ip_xff_takes_first_hop() {
        let h = headers(&[("x-forwarded-for", "2.2.2.2, 3.3.3.3")]);
        assert_eq!(client_ip(&h), "2.2.2.2");
    }

    #[test]
    fn client_ip_falls_back_to_direct() {
        assert_eq!(client_ip(&HeaderMap::new()), "direct");
    }

    // --- HTTP surface (keyless: model layer is None, deterministic path) ---

    async fn body_json(resp: axum::response::Response) -> serde_json::Value {
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn health_exposes_all_documented_fields() {
        let resp = app_with(&[])
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        for field in ["ok", "service", "version", "model_layer", "model_pool_size", "commit"] {
            assert!(v.get(field).is_some(), "/health missing `{field}`: {v}");
        }
        // /health must not leak model/provider names publicly.
        for leaked in ["model_fast", "model_pool"] {
            assert!(v.get(leaked).is_none(), "/health should not expose `{leaked}`: {v}");
        }
    }

    #[tokio::test]
    async fn score_agent_works_with_no_key() {
        // The money demo must score fully on the deterministic engine alone.
        let resp = app_with(&[])
            .oneshot(
                Request::post("/score/agent")
                    .header("content-type", "application/json")
                    .body(Body::from(DEMO_WRONG_TOOL))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        assert_eq!(v["deterministic_share"], 1.0, "keyless must be 100% deterministic");
        assert!(v["run_quality"].as_f64().unwrap() < 60.0, "wrong_tool must be capped red/amber");
    }

    #[tokio::test]
    async fn rag_without_model_returns_501() {
        let resp = app_with(&[])
            .oneshot(
                Request::post("/score/rag")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"question":"q","contexts":["c"],"answer":"a"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
        assert_eq!(body_json(resp).await["status"], "model_required");
    }

    // --- rate limiter ---

    async fn post_agent(app: &Router, ip: &str) -> StatusCode {
        app.clone()
            .oneshot(
                Request::post("/score/agent")
                    .header("content-type", "application/json")
                    .header("cf-connecting-ip", ip)
                    .body(Body::from(r#"{"goal":"x","steps":[]}"#))
                    .unwrap(),
            )
            .await
            .unwrap()
            .status()
    }

    #[tokio::test]
    async fn rate_limit_returns_429_after_cap() {
        let app = app_with(&[("JUDIX_RATE_LIMIT_PER_MIN", "2")]);
        assert_eq!(post_agent(&app, "7.7.7.7").await, StatusCode::OK);
        assert_eq!(post_agent(&app, "7.7.7.7").await, StatusCode::OK);
        assert_eq!(post_agent(&app, "7.7.7.7").await, StatusCode::TOO_MANY_REQUESTS);
        // A DIFFERENT IP has its own bucket and is unaffected.
        assert_eq!(post_agent(&app, "8.8.8.8").await, StatusCode::OK);
    }

    #[tokio::test]
    async fn health_and_demo_are_never_rate_limited() {
        let app = app_with(&[("JUDIX_RATE_LIMIT_PER_MIN", "1")]);
        // Same IP, well past the cap of 1 — these routes are exempt.
        for _ in 0..5 {
            let h = app
                .clone()
                .oneshot(
                    Request::get("/health")
                        .header("cf-connecting-ip", "6.6.6.6")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(h.status(), StatusCode::OK);
            let d = app
                .clone()
                .oneshot(
                    Request::get("/demo/clean")
                        .header("cf-connecting-ip", "6.6.6.6")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(d.status(), StatusCode::OK);
        }
    }
}
