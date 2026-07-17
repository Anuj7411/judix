//! The explanation (model-judge) layer (§7 / §8). The deterministic engine stays
//! the core and computes first; the model only *explains* and does RAG claim
//! verification. Every check returns `{score, confidence, reason}` (source =
//! Model). Two OpenAI-compatible providers are wired via env:
//!
//! | Role   | base URL                  | key                    | model               |
//! |--------|---------------------------|------------------------|---------------------|
//! | fast   | `JUDIX_BASE_URL`          | `JUDIX_API_KEY`        | `JUDIX_MODEL_FAST`  |
//! | strong | `JUDIX_STRONG_BASE_URL`* | `JUDIX_STRONG_API_KEY`*| `JUDIX_MODEL_STRONG`|
//!
//! *Strong falls back to the fast provider's URL/key when unset. Intended prod
//! wiring: fast = Gemini Flash (`gemini-flash-latest`), strong = Groq
//! (`llama-3.3-70b-versatile`) as the low-confidence escalation.
//!
//! If `JUDIX_API_KEY` is unset the whole layer is disabled (`from_env` → `None`)
//! and every model metric is reported `na` — the app still runs keyless on the
//! deterministic engine alone.

use crate::cache::ModelCache;
use crate::deterministic;
use crate::types::{AgentTrace, ClaimSpan, MetricResult, MetricSource, RagTriple};
use futures::future;
use reqwest::Client;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Semaphore;

/// Cap on in-flight model calls.
///
/// Measured, not guessed. The ~5/min figure is specific to `gemini-2.5-flash`;
/// `gemini-3.1-flash-lite` served 14/14 concurrent trivial requests with zero 429s,
/// and Groq's own headers report 1000 req/day + 12k tokens/min. But a *real* load of
/// full-size traces does throttle 3.1-flash-lite, so concurrency alone isn't the
/// answer — `chat` fails over to the second provider on 429. A 5-step trace fires 10
/// calls, so 8 sends a whole trace in ~2 waves while bounding a pathological input.
const MAX_CONCURRENCY: usize = 8;
/// How many times to retry a 429/503 before giving up on a single call.
const MAX_RETRIES: u32 = 4;
/// Ceiling on any single backoff sleep.
const MAX_BACKOFF_MS: u64 = 12_000;

// Output budgets, sized to what each check actually returns.
//
// These are NOT a micro-optimisation. Providers reserve `max_tokens` against your quota
// at request time, spent or not — Groq's own 429 shows it: prompt 116 + max_tokens 2048
// = "Requested 2164" against a 100k tokens/day free tier. A blanket 2048 was therefore
// billing ~2.1k tokens to receive a ~50-token JSON object, burning ~93% of the daily
// budget on nothing and capping us at ~46 calls/day. Sized properly it's ~270.
//
// Set them generously enough that a truncated response (which fails to parse and costs a
// retry) stays impossible.
/// `{score, confidence, reason}` — one short sentence.
const MAX_TOKENS_SCORE: u32 = 256;
/// A claims array with a verbatim quote per claim.
const MAX_TOKENS_DECOMPOSE: u32 = 1536;
/// One verdict object per claim.
const MAX_TOKENS_VERIFY: u32 = 1024;

/// How long a throttled provider is skipped before we try it again.
///
/// Long enough to matter (a per-minute limit clears, a daily quota obviously doesn't),
/// short enough that a brief spike doesn't sideline a healthy provider for long.
const PROVIDER_COOLDOWN_SECS: u64 = 45;

/// How many steps of a trace get **model** checks in a single request.
///
/// Security bound, not a performance knob. The model layer fires 2 calls per step with
/// no natural ceiling, and a minimal step is ~30 bytes — so ~60k steps fit inside axum's
/// 2MB body limit, which would mean ~120k model calls from ONE unauthenticated request.
/// Against a 1000-req/day free tier that is a trivial denial-of-wallet: a single curl
/// exhausts the quota and takes the public demo down.
///
/// Steps past the cap still get **full deterministic scoring** — that path is free,
/// local, and O(n), so trace size is only ever a threat via the model layer.
const DEFAULT_MAX_MODEL_STEPS: usize = 40;

/// §7.2 — judge one step against the goal, including constraints encoded in tool args.
const REL_SYS: &str = "You evaluate ONE step of an AI agent's trajectory against the user's GOAL. \
    The goal may contain explicit constraints (e.g. 'avoiding downtown'). Rate how relevant \
    and necessary this step is to achieving the goal. HEAVILY penalize any step that violates \
    an explicit constraint in the goal, and penalize steps that make no progress. \
    Respond ONLY as JSON {\"score\":0-100,\"confidence\":0.0-1.0,\"reason\":\"one short sentence\"}. \
    100 = essential and compliant, 50 = tangential, 0 = irrelevant or constraint-violating.";

/// §7.3 — judge the trajectory so far against the original goal.
const DRIFT_SYS: &str = "You evaluate whether an AI agent is still pursuing its ORIGINAL GOAL given \
    the TRAJECTORY so far. Respond ONLY as JSON {\"score\":0-100,\"confidence\":0.0-1.0,\"reason\":\"one short sentence\"}. \
    100 = fully on the original goal, 0 = fully drifted or abandoned. Penalize repeated \
    no-progress actions and constraint violations.";

/// §7.5 — does the answer address the question?
const AR_SYS: &str = "Rate how directly the ANSWER addresses the QUESTION. Respond ONLY as JSON \
    {\"score\":0-100,\"confidence\":0.0-1.0,\"reason\":\"one short sentence\"}. 100 = fully answers, 0 = off-topic.";

/// §7.5 — signal-to-noise of the retrieved contexts.
const CP_SYS: &str = "Given the QUESTION, the retrieved CONTEXTS, and the ANSWER, rate the signal-to-noise \
    of the contexts: what fraction are actually relevant/necessary to answer the question. \
    Respond ONLY as JSON {\"score\":0-100,\"confidence\":0.0-1.0,\"reason\":\"one short sentence\"}. \
    100 = all contexts relevant, 0 = all noise.";

/// §7.5 — do the contexts contain everything the answer needs?
const CR_SYS: &str = "Given the QUESTION, the CONTEXTS, and the ANSWER, rate context recall: do the contexts \
    contain all the information needed to support the answer? Respond ONLY as JSON \
    {\"score\":0-100,\"confidence\":0.0-1.0,\"reason\":\"one short sentence\"}. 100 = fully covered, 0 = key info missing.";

/// One OpenAI-compatible endpoint (base URL + key + model name).
#[derive(Clone)]
struct Provider {
    base_url: String,
    api_key: String,
    model: String,
}

#[derive(Clone)]
pub struct ModelClient {
    http: Client,
    fast: Provider,
    strong: Provider,
    cache: ModelCache,
    sem: Arc<Semaphore>,
    /// Escalate a check to the strong model when the fast model reports confidence
    /// below this (§8 default 0.5, tunable via `JUDIX_ESCALATE_BELOW`). Judges are
    /// chronically overconfident — Gemini self-reports ≥0.5 even on genuinely
    /// ambiguous input — so this is exposed rather than hard-coded.
    escalate_below: f32,
    /// See [`DEFAULT_MAX_MODEL_STEPS`]. Bounds model calls per request so an
    /// oversized trace can't drain the quota.
    max_model_steps: usize,
    /// Circuit breaker: providers currently known to be throttled, keyed by base URL.
    /// Entries expire on their own (moka TTL), which *is* the breaker closing again.
    ///
    /// Without this, a provider that is out of quota **for the day** still gets tried
    /// first on every request, burning the full retry ladder before failing over —
    /// measured at 42s per request once Gemini's 500/day free quota ran out. Retrying a
    /// daily quota is pointless; skip it and go straight to the provider that works.
    cooling: moka::future::Cache<String, ()>,
}

// --- Chat wire types -------------------------------------------------------

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Deserialize)]
struct Choice {
    message: Msg,
}

#[derive(Deserialize)]
struct Msg {
    content: Option<String>,
    // Reasoning models sometimes leave `content` null and put text here.
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    reasoning: Option<String>,
}

impl Msg {
    fn text(&self) -> String {
        for s in [&self.content, &self.reasoning_content, &self.reasoning]
            .into_iter()
            .flatten()
        {
            if !s.trim().is_empty() {
                return s.clone();
            }
        }
        String::new()
    }
}

#[derive(Deserialize, Default)]
struct Usage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
}

/// The output of one chat call: the text plus the $ it cost (0 on a cache hit).
struct ChatOut {
    text: String,
    cost_usd: f64,
}

/// Why a single attempt failed. `Throttled` is recoverable — we fail over to the
/// other provider (separate quota) rather than sleeping on a busy one.
enum CallErr {
    Throttled { retry_ms: Option<u64> },
    Other(String),
}

// --- Strict structured-output types ---------------------------------------

#[derive(Deserialize)]
struct ScoreOut {
    #[serde(default = "half")]
    score: f64,
    #[serde(default = "point_seven")]
    confidence: f64,
    #[serde(default)]
    reason: String,
}
fn half() -> f64 {
    50.0
}
fn point_seven() -> f64 {
    0.7
}

#[derive(Deserialize)]
struct ClaimsOut {
    #[serde(default)]
    claims: Vec<ClaimItem>,
}
#[derive(Deserialize)]
struct ClaimItem {
    #[serde(default)]
    text: String,
    /// Verbatim substring of the answer this claim is drawn from, so we can map
    /// an unsupported claim back to a char-span for red highlighting.
    #[serde(default)]
    quote: String,
}

#[derive(Deserialize)]
struct VerifyOut {
    #[serde(default)]
    results: Vec<VerifyItem>,
}
#[derive(Deserialize)]
struct VerifyItem {
    #[serde(default)]
    id: u64,
    /// "supported" | "unsupported" | "contradicted". A contradiction is a factual
    /// conflict with the context (a hallucination) — strictly worse than having no
    /// evidence, and treated as a critical failure by `score_rag`.
    #[serde(default)]
    status: String,
}

/// A decomposed claim before any grounding verdict exists, with its char-span in the
/// answer. Deliberately carries no `supported`/`contradicted` flag: emitting a claim
/// early with `supported: false` would render it red before it had been checked.
#[derive(Debug, Clone, Serialize)]
pub struct PendingClaim {
    pub start: usize,
    pub end: usize,
    pub text: String,
}

/// Incremental RAG scoring events (§9.5), in emission order.
pub enum RagEvent {
    /// The answer split into claims + spans. Available a full wave before any verdict.
    Claims(Vec<PendingClaim>),
    /// One metric landed.
    Metric(MetricResult),
    /// Terminal: everything needed to compose the final report.
    Done {
        metrics: Vec<MetricResult>,
        spans: Vec<ClaimSpan>,
        any_contradiction: bool,
        cost: f64,
    },
    /// Terminal.
    Error(String),
}

/// Grounding verdict for one claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClaimStatus {
    Supported,
    Unsupported,
    Contradicted,
}

impl ClaimStatus {
    fn parse(s: &str) -> Self {
        match s.trim().to_lowercase().as_str() {
            "supported" => ClaimStatus::Supported,
            "contradicted" => ClaimStatus::Contradicted,
            _ => ClaimStatus::Unsupported,
        }
    }
}

impl ModelClient {
    /// Build from env. Returns `None` (keyless / deterministic-only) unless both
    /// `JUDIX_BASE_URL` and `JUDIX_API_KEY` are set.
    pub fn from_env() -> Option<Self> {
        let base_url = std::env::var("JUDIX_BASE_URL").ok()?;
        let api_key = std::env::var("JUDIX_API_KEY").ok().filter(|k| !k.is_empty())?;
        let model_fast =
            std::env::var("JUDIX_MODEL_FAST").unwrap_or_else(|_| "gemini-flash-latest".into());
        let model_strong = std::env::var("JUDIX_MODEL_STRONG")
            .unwrap_or_else(|_| "llama-3.3-70b-versatile".into());

        let fast = Provider {
            base_url: base_url.clone(),
            api_key: api_key.clone(),
            model: model_fast,
        };
        // Strong provider falls back to the fast provider's URL/key when its own
        // env vars are absent (so a single-provider setup still works).
        let strong = Provider {
            base_url: std::env::var("JUDIX_STRONG_BASE_URL").unwrap_or(base_url),
            api_key: std::env::var("JUDIX_STRONG_API_KEY").unwrap_or(api_key),
            model: model_strong,
        };

        Some(Self {
            http: Client::new(),
            fast,
            strong,
            // 6h, not the §8 baseline of 1h: the cache key is
            // (check, model, normalized_input), so an entry is a pure function of its
            // input — TTL is an eviction policy, not a correctness knob, and a stale
            // answer is impossible (a different model or input is a different key).
            // A longer window keeps the prewarmed demo fixtures hot across a judging
            // session instead of expiring after an hour and re-paying the cold path.
            cache: ModelCache::new(
                1000,
                std::env::var("JUDIX_CACHE_TTL_SECS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(21_600),
            ),
            sem: Arc::new(Semaphore::new(MAX_CONCURRENCY)),
            // Empirically 0.6, not the spec's literal 0.5: Gemini expresses "I'm
            // unsure" as *exactly* 0.5 and never dips below it, so `< 0.5` never
            // fires and the strong model is never consulted. 0.6 catches that 0.5
            // uncertainty signal while leaving confident calls (0.7–1.0) alone.
            escalate_below: std::env::var("JUDIX_ESCALATE_BELOW")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.6),
            max_model_steps: std::env::var("JUDIX_MAX_MODEL_STEPS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(DEFAULT_MAX_MODEL_STEPS),
            cooling: moka::future::Cache::builder()
                .max_capacity(8)
                .time_to_live(Duration::from_secs(PROVIDER_COOLDOWN_SECS))
                .build(),
        })
    }

    /// Per-model list-price rate (USD per 1M tokens: input, output). Free-tier use
    /// bills $0, but exposing the notional cost lets the UI say "this would cost
    /// $X at list price — you paid $0." Unknown / self-hosted models → 0.
    fn rate(model: &str) -> (f64, f64) {
        let m = model.to_lowercase();
        if m.contains("gemini") && m.contains("flash") {
            (0.075, 0.30)
        } else if m.contains("llama-3.3-70b") {
            (0.59, 0.79)
        } else if m.contains("gpt-oss-120b") {
            (0.15, 0.60)
        } else {
            (0.0, 0.0)
        }
    }

    /// A single attempt against one provider. No retries, no cache — the caller
    /// (`chat`) owns the failover/backoff policy.
    async fn chat_once(
        &self,
        provider: &Provider,
        check: &str,
        system: &str,
        user: &str,
        max_tokens: u32,
    ) -> Result<ChatOut, CallErr> {
        let url = format!("{}/chat/completions", provider.base_url.trim_end_matches('/'));
        let body = json!({
            "model": provider.model,
            "messages": [
                { "role": "system", "content": system },
                { "role": "user",   "content": user   }
            ],
            "temperature": 0.0,
            // Reserved against the provider's quota whether we use it or not — see the
            // MAX_TOKENS_* consts. Do not replace with a blanket value.
            "max_tokens": max_tokens,
            "stream": false,
            // OpenAI structured-output: force a JSON object body. We still parse
            // tolerantly (see `parse`) so providers that ignore this still work.
            "response_format": { "type": "json_object" }
        });

        let permit = self
            .sem
            .acquire()
            .await
            .map_err(|e| CallErr::Other(e.to_string()))?;
        let call_start = std::time::Instant::now();
        let sent = self
            .http
            .post(&url)
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", provider.api_key))
            .json(&body)
            .send()
            .await;
        drop(permit);

        let res = match sent {
            Ok(r) => r,
            Err(e) => return Err(CallErr::Other(e.to_string())),
        };

        if res.status() == 429 || res.status() == 503 {
            let header_hint = Self::retry_after_header(&res);
            let status = res.status().as_u16();
            let text = res.text().await.unwrap_or_default();
            // Gemini puts its hint in the BODY ("retryDelay": "11s"), not in a
            // Retry-After header — reading only the header made us retry after
            // 250ms against an 11s cooldown and burn every attempt.
            let retry_ms = header_hint.or_else(|| Self::retry_delay_body(&text));
            tracing::warn!(check, model = %provider.model, status, ?retry_ms, "provider throttled");
            return Err(CallErr::Throttled { retry_ms });
        }
        if !res.status().is_success() {
            let status = res.status();
            let text = res.text().await.unwrap_or_default();
            let short: String = text.chars().take(200).collect();
            return Err(CallErr::Other(format!("model API {status}: {short}")));
        }

        let chat: ChatResponse = res.json().await.map_err(|e| CallErr::Other(e.to_string()))?;
        let text = chat.choices.first().map(|c| c.message.text()).unwrap_or_default();
        let usage = chat.usage.unwrap_or_default();
        let (rin, rout) = Self::rate(&provider.model);
        let cost_usd =
            (usage.prompt_tokens as f64 * rin + usage.completion_tokens as f64 * rout) / 1_000_000.0;
        tracing::debug!(
            check, model = %provider.model, ms = call_start.elapsed().as_millis() as u64,
            "model call ok"
        );
        Ok(ChatOut { text, cost_usd })
    }

    /// Run `check` against `primary`, **failing over to `secondary` the moment
    /// `primary` throttles**, and only backing off if both are busy.
    ///
    /// The two providers hold independent quotas (Gemini and Groq), so a 429 on one
    /// is not a reason to sleep — it's a reason to use the other. Sleeping on a busy
    /// provider while a second key sat idle is what exhausted the retry budget and
    /// dropped metrics to `na` under load.
    ///
    /// Cached under `primary.model` regardless of which provider actually answered:
    /// the cache key is the *logical route* (check + input), and either provider's
    /// answer is a valid response to the same question.
    async fn chat(
        &self,
        primary: &Provider,
        secondary: &Provider,
        check: &str,
        system: &str,
        user: &str,
        max_tokens: u32,
    ) -> Result<ChatOut, String> {
        let cache_input = format!("{system}\n---\n{user}");
        if let Some(cached) = self.cache.get(check, &primary.model, &cache_input).await {
            return Ok(ChatOut { text: cached, cost_usd: 0.0 });
        }

        let mut last_err: Option<String> = None;
        let mut throttled = false;
        for attempt in 0..=MAX_RETRIES {
            let mut throttle_hint: Option<u64> = None;

            // Prefer a provider that isn't in cooldown. If both are cooling we still try
            // them (rather than fail instantly) — the breaker is there to reorder work
            // away from a dead provider, not to refuse to do it.
            let both = [primary, secondary];
            let mut order: Vec<&Provider> = both
                .iter()
                .copied()
                .filter(|p| !self.is_cooling(p))
                .collect();
            if order.is_empty() {
                order = both.to_vec();
            }

            for (idx, provider) in order.into_iter().enumerate() {
                // Skip the duplicate call when both roles point at one provider.
                if idx == 1 && secondary.model == primary.model && secondary.base_url == primary.base_url {
                    continue;
                }
                match self.chat_once(provider, check, system, user, max_tokens).await {
                    Ok(out) => {
                        if idx == 1 {
                            tracing::info!(check, from = %primary.model, to = %provider.model,
                                "failed over to secondary provider");
                        }
                        self.cache
                            .insert(check, &primary.model, &cache_input, out.text.clone())
                            .await;
                        return Ok(out);
                    }
                    Err(CallErr::Throttled { retry_ms }) => {
                        throttled = true;
                        throttle_hint = throttle_hint.or(retry_ms);
                        self.mark_cooling(provider).await;
                        continue; // try the other provider before sleeping
                    }
                    Err(CallErr::Other(e)) => {
                        last_err = Some(e);
                        continue;
                    }
                }
            }

            // Both providers unavailable — now a wait is actually warranted. Honor
            // the provider's own hint when it gave one, else exponential backoff.
            if attempt < MAX_RETRIES {
                let delay = throttle_hint.unwrap_or_else(|| Self::backoff_ms(attempt));
                tracing::warn!(check, attempt, delay_ms = delay, "all providers busy — backing off");
                tokio::time::sleep(Duration::from_millis(delay)).await;
            }
        }
        // Report what actually happened. A pure-throttle exhaustion used to surface as
        // "no attempt made" (the untouched default of last_err), which is the opposite
        // of the truth and sent me hunting the wrong bug.
        Err(match (last_err, throttled) {
            (Some(e), _) => format!("all providers failed after {MAX_RETRIES} retries: {e}"),
            (None, true) => {
                format!("all providers rate-limited after {MAX_RETRIES} retries")
            }
            (None, false) => format!("all providers unavailable after {MAX_RETRIES} retries"),
        })
    }

    /// Is this provider currently sidelined for throttling?
    fn is_cooling(&self, p: &Provider) -> bool {
        self.cooling.contains_key(&p.base_url)
    }

    /// Sideline a provider that just throttled, so the next request tries the other one
    /// first instead of re-paying the retry ladder against a quota that hasn't reset.
    async fn mark_cooling(&self, p: &Provider) {
        if !self.is_cooling(p) {
            tracing::warn!(model = %p.model, secs = PROVIDER_COOLDOWN_SECS, "provider cooling down");
        }
        self.cooling.insert(p.base_url.clone(), ()).await;
    }

    /// Milliseconds to wait per the `Retry-After` header (sent in seconds).
    fn retry_after_header(res: &reqwest::Response) -> Option<u64> {
        let secs: u64 = res
            .headers()
            .get(reqwest::header::RETRY_AFTER)?
            .to_str()
            .ok()?
            .trim()
            .parse()
            .ok()?;
        Some((secs * 1000).min(MAX_BACKOFF_MS))
    }

    /// Pull Gemini's `"retryDelay": "11s"` out of a 429 body.
    fn retry_delay_body(body: &str) -> Option<u64> {
        let at = body.find("retryDelay")?;
        let rest = &body[at..];
        let start = rest.find(':')?;
        let seg: String = rest[start..]
            .chars()
            .skip_while(|c| !c.is_ascii_digit())
            .take_while(|c| c.is_ascii_digit() || *c == '.')
            .collect();
        let secs: f64 = seg.parse().ok()?;
        Some(((secs * 1000.0) as u64).min(MAX_BACKOFF_MS))
    }

    /// Backoff for attempt N, in milliseconds: 250, 500, 1000, 2000 … capped.
    ///
    /// Deliberately sub-second to start. A seconds-scale ladder (4/8/16/20s) turns
    /// one transient 503 into ~48s of sleeping and dominates the whole request —
    /// which is exactly what made a 10-call trace take 52s despite the provider
    /// having ample quota. Jitter (derived from the attempt, not a clock, to keep
    /// scoring reproducible) avoids retry convoys when a wave throttles together.
    fn backoff_ms(attempt: u32) -> u64 {
        let base = 250u64 << attempt.min(5);
        let jitter = (attempt as u64 * 37) % 100;
        (base + jitter).min(MAX_BACKOFF_MS)
    }

    /// Tolerant JSON extraction → strict serde. Tries raw parse, ```json fences,
    /// then the first balanced `{`/`[`.
    fn parse<T: DeserializeOwned>(text: &str) -> Option<T> {
        let t = text.trim();
        if let Ok(v) = serde_json::from_str::<T>(t) {
            return Some(v);
        }
        if let Some(start) = t.find("```") {
            let after = &t[start + 3..];
            let body_start = after.find('\n').map(|n| n + 1).unwrap_or(0);
            if let Some(end) = after[body_start..].find("```") {
                let block = after[body_start..body_start + end].trim();
                if let Ok(v) = serde_json::from_str::<T>(block) {
                    return Some(v);
                }
            }
        }
        for (i, c) in t.char_indices() {
            if c == '{' || c == '[' {
                if let Ok(v) = serde_json::from_str::<T>(&t[i..]) {
                    return Some(v);
                }
            }
        }
        None
    }

    /// Run a `{score,confidence,reason}` check on the fast model, escalating to the
    /// strong model when the fast model's confidence < 0.5 (§8). Returns the metric
    /// plus total $ spent (both calls, if escalated).
    async fn scored_check(
        &self,
        check: &str,
        system: &str,
        user: &str,
    ) -> (MetricResult, f64) {
        let (fast_out, mut cost) = match self.chat(&self.fast, &self.strong, check, system, user, MAX_TOKENS_SCORE).await {
            Ok(o) => (Self::parse::<ScoreOut>(&o.text), o.cost_usd),
            Err(e) => {
                return (
                    MetricResult::na(check, MetricSource::Model)
                        .with_reason(format!("model call failed: {e}")),
                    0.0,
                )
            }
        };

        let fast = match fast_out {
            Some(s) => s,
            None => {
                return (
                    MetricResult::na(check, MetricSource::Model)
                        .with_reason("unparseable model response"),
                    cost,
                )
            }
        };

        // Escalate low-confidence fast results to the strong model.
        if (fast.confidence as f32) < self.escalate_below {
            if let Ok(o) = self.chat(&self.strong, &self.fast, check, system, user, MAX_TOKENS_SCORE).await {
                cost += o.cost_usd;
                if let Some(strong) = Self::parse::<ScoreOut>(&o.text) {
                    let mut m = MetricResult::model(
                        check,
                        strong.score as f32,
                        strong.confidence as f32,
                        strong.reason,
                    );
                    // We escalated because the fast pass was uncertain — flag it.
                    m.low_confidence = true;
                    return (m, cost);
                }
            }
        }

        let m = MetricResult::model(check, fast.score as f32, fast.confidence as f32, fast.reason);
        (m, cost)
    }

    // ------------------------------------------------------------------
    // Agent checks (§7.2 / §7.3)
    // ------------------------------------------------------------------

    /// Render each step as `(step_description, trajectory_prefix)`.
    ///
    /// Synchronous and cheap, so the async calls that follow have no ordering
    /// constraint between them: `goal_drift` needs only the trajectory *prefix*, which
    /// is fully determined here.
    fn prepare_steps(trace: &AgentTrace) -> Vec<(String, String)> {
        let mut trajectory = String::new();
        let mut prepared = Vec::with_capacity(trace.steps.len());
        for (i, step) in trace.steps.iter().enumerate() {
            let name = step.name.clone().unwrap_or_else(|| step.kind.clone());
            // Include tool-call args so the judge can see WHAT was called (e.g.
            // area=downtown), plus any result/content. Without the args a relevance
            // judge is blind to constraint violations encoded in the arguments.
            let args = step
                .args
                .as_ref()
                .map(|a| format!(" args={a}"))
                .unwrap_or_default();
            let content = step
                .content
                .clone()
                .or_else(|| step.result.clone())
                .map(|c| format!(" → {c}"))
                .unwrap_or_default();
            let step_desc = format!("[{}] {name}{args}{content}", step.kind);
            trajectory.push_str(&format!("Step {i}: {step_desc}\n"));
            prepared.push((step_desc, trajectory.clone()));
        }
        prepared
    }

    /// Score `step_relevance` + `goal_drift` for every step, all model calls fired
    /// concurrently. Returns per-step metrics and total model $.
    pub async fn score_agent_steps(&self, trace: &AgentTrace) -> (Vec<Vec<MetricResult>>, f64) {
        let prepared = Self::prepare_steps(trace);
        let scored = prepared.len().min(self.max_model_steps);

        let futures = prepared.iter().take(scored).map(|(step_desc, traj)| async move {
            let rel_user = format!("GOAL: {}\n\nSTEP: {step_desc}", trace.goal);
            let drift_user = format!("ORIGINAL GOAL: {}\n\nTRAJECTORY SO FAR:\n{traj}", trace.goal);
            let ((rel, c1), (drift, c2)) = futures::future::join(
                self.scored_check("step_relevance", REL_SYS, &rel_user),
                self.scored_check("goal_drift", DRIFT_SYS, &drift_user),
            )
            .await;
            (vec![rel, drift], c1 + c2)
        });

        let results = futures::future::join_all(futures).await;
        let cost = results.iter().map(|(_, c)| c).sum();
        let mut metrics: Vec<Vec<MetricResult>> = results.into_iter().map(|(m, _)| m).collect();

        // Say so, rather than silently returning fewer metrics than there are steps.
        for _ in scored..prepared.len() {
            metrics.push(Self::capped_metrics(self.max_model_steps));
        }
        (metrics, cost)
    }

    /// `na` metrics explaining that a step was past the model-call budget. Being
    /// explicit matters: a silently unscored step looks identical to a bug, and `na`
    /// metrics are excluded from the composite rather than dragging it down.
    fn capped_metrics(max: usize) -> Vec<MetricResult> {
        let reason =
            format!("step beyond the first {max} — model checks capped per request (JUDIX_MAX_MODEL_STEPS); deterministic scoring still applied");
        vec![
            MetricResult::na("step_relevance", MetricSource::Model).with_reason(reason.clone()),
            MetricResult::na("goal_drift", MetricSource::Model).with_reason(reason),
        ]
    }

    /// Same checks as [`score_agent_steps`], but yielded **as each one lands** rather
    /// than after the whole wave (§9.5).
    ///
    /// `score_agent_steps` waits on `join_all`, so the caller blocks for the slowest
    /// call. That's fine for the JSON API, which has one response to fill. For SSE the
    /// whole point is that a judge sees each explanation the instant it arrives, on top
    /// of deterministic metrics that were already on screen in ~1ms. `FuturesUnordered`
    /// polls every call concurrently and completes them out of order, so the stream is
    /// ordered by *latency*, not by step index — each item carries its own `step_index`.
    pub fn stream_agent_metrics<'a>(
        &'a self,
        trace: &'a AgentTrace,
    ) -> impl futures::Stream<Item = (usize, MetricResult, f64)> + 'a {
        let prepared = Self::prepare_steps(trace);
        let scored = prepared.len().min(self.max_model_steps);
        let futs = futures::stream::FuturesUnordered::new();

        // Same per-request model-call budget as `score_agent_steps` — streaming must not
        // be a way around the cap.
        for (i, (step_desc, traj)) in prepared.into_iter().enumerate().take(scored) {
            let rel_user = format!("GOAL: {}\n\nSTEP: {step_desc}", trace.goal);
            futs.push(future::Either::Left(async move {
                let (m, c) = self.scored_check("step_relevance", REL_SYS, &rel_user).await;
                (i, m, c)
            }));
            let drift_user = format!("ORIGINAL GOAL: {}\n\nTRAJECTORY SO FAR:\n{traj}", trace.goal);
            futs.push(future::Either::Right(async move {
                let (m, c) = self.scored_check("goal_drift", DRIFT_SYS, &drift_user).await;
                (i, m, c)
            }));
        }
        futs
    }

    // ------------------------------------------------------------------
    // RAG checks (§7.4 / §7.5)
    // ------------------------------------------------------------------

    /// (a) Decompose the answer into atomic claims (each with a verbatim quote).
    async fn decompose_claims(&self, answer: &str) -> Result<(Vec<(String, String)>, f64), String> {
        let system = "Decompose the ANSWER into atomic factual claims — each a single, independently \
            verifiable statement. For each claim also return `quote`: the EXACT verbatim substring of \
            the answer it is drawn from (copied character-for-character, no paraphrasing). \
            Respond ONLY as JSON {\"claims\":[{\"id\":1,\"text\":\"...\",\"quote\":\"...\"}]}.";
        let out = self
            .chat(
                &self.fast,
                &self.strong,
                "rag_decompose",
                system,
                &format!("ANSWER:\n{answer}"),
                MAX_TOKENS_DECOMPOSE,
            )
            .await?;
        let parsed: ClaimsOut = Self::parse(&out.text)
            .ok_or_else(|| format!("unparseable decomposition: {}", out.text))?;
        let claims = parsed
            .claims
            .into_iter()
            .map(|c| {
                let quote = if c.quote.trim().is_empty() {
                    c.text.clone()
                } else {
                    c.quote
                };
                (c.text, quote)
            })
            .collect();
        Ok((claims, out.cost_usd))
    }

    /// (b) Verify ALL claims against the contexts in one batched call.
    async fn verify_claims(
        &self,
        claims: &[(String, String)],
        contexts: &[String],
    ) -> Result<(Vec<ClaimStatus>, f64), String> {
        let system = "Given CONTEXTS and a numbered list of CLAIMS, classify EACH claim against the \
            contexts with one of three statuses:\n\
            - \"supported\": the claim can be directly inferred from some context.\n\
            - \"contradicted\": the claim CONFLICTS with a fact stated in a context (e.g. the context \
              says 14 days and the claim says 30 days). This is a hallucination.\n\
            - \"unsupported\": the contexts neither support nor contradict the claim.\n\
            Respond ONLY as JSON {\"results\":[{\"id\":1,\"status\":\"supported\",\"context_index\":1,\"reason\":\"...\"}]}.";
        let ctx = contexts
            .iter()
            .enumerate()
            .map(|(i, c)| format!("[Context {}]: {c}", i + 1))
            .collect::<Vec<_>>()
            .join("\n");
        let claim_list = claims
            .iter()
            .enumerate()
            .map(|(i, (text, _))| format!("{}. {text}", i + 1))
            .collect::<Vec<_>>()
            .join("\n");
        let user = format!("CONTEXTS:\n{ctx}\n\nCLAIMS:\n{claim_list}");

        let out = self
            .chat(&self.fast, &self.strong, "rag_verify", system, &user, MAX_TOKENS_VERIFY)
            .await?;
        let parsed: VerifyOut = Self::parse(&out.text)
            .ok_or_else(|| format!("unparseable verification: {}", out.text))?;

        // Map verdicts back by 1-based id; default unsupported if the model omits one.
        let mut statuses = vec![ClaimStatus::Unsupported; claims.len()];
        for r in parsed.results {
            let idx = r.id as usize;
            if idx >= 1 && idx <= claims.len() {
                statuses[idx - 1] = ClaimStatus::parse(&r.status);
            }
        }
        Ok((statuses, out.cost_usd))
    }

    /// Same work as [`score_rag_triple`], but emitted incrementally (§9.5).
    ///
    /// RAG has no deterministic metric to lead with — the faithfulness *ratio* is
    /// computed in Rust, but only from model verdicts, so there is nothing to show at
    /// 1ms the way the agent path can. What it does have is a natural seam: claims
    /// exist after the decompose wave, a full wave before any grounding verdict. So
    /// emit them, letting the UI paint the answer broken into spans while the checks
    /// are still running, instead of holding everything until the last one lands.
    pub fn stream_rag<'a>(
        &'a self,
        triple: &'a RagTriple,
    ) -> impl futures::Stream<Item = RagEvent> + 'a {
        async_stream::stream! {
            // Wave 1: decompose.
            let (claims, decompose_cost) = match self.decompose_claims(&triple.answer).await {
                Ok(v) => v,
                Err(e) => {
                    yield RagEvent::Error(e);
                    return;
                }
            };

            let located: Vec<(String, Option<(usize, usize)>)> = claims
                .iter()
                .map(|(text, quote)| {
                    (text.clone(), deterministic::find_span(&triple.answer, quote))
                })
                .collect();

            yield RagEvent::Claims(
                located
                    .iter()
                    .map(|(text, span)| PendingClaim {
                        start: span.map(|(s, _)| s).unwrap_or(0),
                        end: span.map(|(_, e)| e).unwrap_or(0),
                        text: text.clone(),
                    })
                    .collect(),
            );

            // Wave 2: grounding + the three relevancy/context checks, all concurrent.
            let (verify_res, ar, cp, cr) = futures::future::join4(
                self.verify_claims(&claims, &triple.contexts),
                self.scored_check("answer_relevancy", AR_SYS, &Self::ar_user(triple)),
                self.scored_check("context_precision", CP_SYS, &Self::ctx_user(triple)),
                self.scored_check("context_recall", CR_SYS, &Self::ctx_user(triple)),
            )
            .await;

            let (statuses, verify_cost) = match verify_res {
                Ok(v) => v,
                Err(e) => {
                    yield RagEvent::Error(e);
                    return;
                }
            };

            let (faithfulness, spans, any_contradiction) =
                Self::compose_faithfulness(&located, &statuses, &claims);

            let (ar_m, ar_c) = ar;
            let (cp_m, cp_c) = cp;
            let (cr_m, cr_c) = cr;
            let metrics = vec![faithfulness, ar_m, cp_m, cr_m];
            for m in &metrics {
                yield RagEvent::Metric(m.clone());
            }

            yield RagEvent::Done {
                metrics,
                spans,
                any_contradiction,
                cost: decompose_cost + verify_cost + ar_c + cp_c + cr_c,
            };
        }
    }

    /// Full RAG scoring: faithfulness (decompose → batched verify → ratio in Rust,
    /// with unsupported spans), plus answer_relevancy, context_precision, and
    /// context_recall. Returns `(metrics, spans, any_contradiction, cost)`, where
    /// `any_contradiction` drives the critical-fail cap in `score_rag`.
    pub async fn score_rag_triple(
        &self,
        triple: &RagTriple,
    ) -> Result<(Vec<MetricResult>, Vec<ClaimSpan>, bool, f64), String> {
        // Step 1: decompose (must precede verify).
        let (claims, decompose_cost) = self.decompose_claims(&triple.answer).await?;

        // Step 2: verify all claims (batched) concurrently with the three
        // relevancy/context checks — they're independent.
        let (verify_res, ar, cp, cr) = futures::future::join4(
            self.verify_claims(&claims, &triple.contexts),
            self.scored_check("answer_relevancy", AR_SYS, &Self::ar_user(triple)),
            self.scored_check("context_precision", CP_SYS, &Self::ctx_user(triple)),
            self.scored_check("context_recall", CR_SYS, &Self::ctx_user(triple)),
        )
        .await;

        let (statuses, verify_cost) = verify_res?;
        let located: Vec<(String, Option<(usize, usize)>)> = claims
            .iter()
            .map(|(text, quote)| (text.clone(), deterministic::find_span(&triple.answer, quote)))
            .collect();
        let (faithfulness, spans, any_contradiction) =
            Self::compose_faithfulness(&located, &statuses, &claims);

        let (ar_m, ar_c) = ar;
        let (cp_m, cp_c) = cp;
        let (cr_m, cr_c) = cr;
        Ok((
            vec![faithfulness, ar_m, cp_m, cr_m],
            spans,
            any_contradiction,
            decompose_cost + verify_cost + ar_c + cp_c + cr_c,
        ))
    }

    fn ar_user(triple: &RagTriple) -> String {
        format!("QUESTION: {}\nANSWER: {}", triple.question, triple.answer)
    }

    fn ctx_user(triple: &RagTriple) -> String {
        let ctx_joined = triple
            .contexts
            .iter()
            .enumerate()
            .map(|(i, c)| format!("[Context {}]: {c}", i + 1))
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "QUESTION: {}\n\nCONTEXTS:\n{ctx_joined}\n\nANSWER: {}",
            triple.question, triple.answer
        )
    }

    /// Turn per-claim verdicts into the faithfulness metric + highlightable spans.
    /// Shared by the batch and streaming paths so they can never drift apart.
    fn compose_faithfulness(
        located: &[(String, Option<(usize, usize)>)],
        statuses: &[ClaimStatus],
        claims: &[(String, String)],
    ) -> (MetricResult, Vec<ClaimSpan>, bool) {
        let supported_count = statuses
            .iter()
            .filter(|s| **s == ClaimStatus::Supported)
            .count();
        let contradicted_count = statuses
            .iter()
            .filter(|s| **s == ClaimStatus::Contradicted)
            .count();
        let any_contradiction = contradicted_count > 0;
        let faith_score = deterministic::faithfulness_ratio(supported_count, claims.len());

        // Map each claim to a char-span via its verbatim quote, so the UI can
        // highlight ungrounded text (contradicted = red, unsupported = amber).
        let spans = located
            .iter()
            .zip(statuses.iter())
            .map(|((text, span), status)| ClaimSpan {
                start: span.map(|(s, _)| s).unwrap_or(0),
                end: span.map(|(_, e)| e).unwrap_or(0),
                text: text.clone(),
                supported: *status == ClaimStatus::Supported,
                contradicted: *status == ClaimStatus::Contradicted,
            })
            .collect();

        // A ratio dilutes severity: 3 harmless-correct claims outvote 1 catastrophically
        // wrong one. Name the contradiction explicitly so the headline can't read "fine".
        let faith_reason = if any_contradiction {
            format!(
                "{supported_count}/{} claims grounded — {contradicted_count} CONTRADICTS the context (hallucination)",
                claims.len()
            )
        } else {
            format!(
                "{supported_count}/{} claims grounded in the contexts",
                claims.len()
            )
        };
        (
            MetricResult::model("faithfulness", faith_score, 0.9, faith_reason),
            spans,
            any_contradiction,
        )
    }
}
