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
use reqwest::Client;
use serde::de::DeserializeOwned;
use serde::Deserialize;
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
    ) -> Result<ChatOut, CallErr> {
        let url = format!("{}/chat/completions", provider.base_url.trim_end_matches('/'));
        let body = json!({
            "model": provider.model,
            "messages": [
                { "role": "system", "content": system },
                { "role": "user",   "content": user   }
            ],
            "temperature": 0.0,
            "max_tokens": 2048,
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
    ) -> Result<ChatOut, String> {
        let cache_input = format!("{system}\n---\n{user}");
        if let Some(cached) = self.cache.get(check, &primary.model, &cache_input).await {
            return Ok(ChatOut { text: cached, cost_usd: 0.0 });
        }

        let mut last_err = String::from("no attempt made");
        for attempt in 0..=MAX_RETRIES {
            let mut throttle_hint: Option<u64> = None;

            for (idx, provider) in [primary, secondary].into_iter().enumerate() {
                // Skip the duplicate call when both roles point at one provider.
                if idx == 1 && secondary.model == primary.model && secondary.base_url == primary.base_url {
                    continue;
                }
                match self.chat_once(provider, check, system, user).await {
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
                        throttle_hint = throttle_hint.or(retry_ms);
                        continue; // try the other provider before sleeping
                    }
                    Err(CallErr::Other(e)) => {
                        last_err = e;
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
        Err(format!("all providers exhausted after {MAX_RETRIES} retries: {last_err}"))
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
        let (fast_out, mut cost) = match self.chat(&self.fast, &self.strong, check, system, user).await {
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
            if let Ok(o) = self.chat(&self.strong, &self.fast, check, system, user).await {
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

    /// Score `step_relevance` + `goal_drift` for every step, all model calls fired
    /// concurrently. Returns per-step metrics and total model $.
    pub async fn score_agent_steps(&self, trace: &AgentTrace) -> (Vec<Vec<MetricResult>>, f64) {
        // Precompute each step's description and the trajectory prefix up to it
        // (synchronous — no ordering constraint on the async calls).
        let mut trajectory = String::new();
        let mut prepared: Vec<(String, String)> = Vec::with_capacity(trace.steps.len());
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

        let rel_sys = "You evaluate ONE step of an AI agent's trajectory against the user's GOAL. \
            The goal may contain explicit constraints (e.g. 'avoiding downtown'). Rate how relevant \
            and necessary this step is to achieving the goal. HEAVILY penalize any step that violates \
            an explicit constraint in the goal, and penalize steps that make no progress. \
            Respond ONLY as JSON {\"score\":0-100,\"confidence\":0.0-1.0,\"reason\":\"one short sentence\"}. \
            100 = essential and compliant, 50 = tangential, 0 = irrelevant or constraint-violating.";
        let drift_sys = "You evaluate whether an AI agent is still pursuing its ORIGINAL GOAL given \
            the TRAJECTORY so far. Respond ONLY as JSON {\"score\":0-100,\"confidence\":0.0-1.0,\"reason\":\"one short sentence\"}. \
            100 = fully on the original goal, 0 = fully drifted or abandoned. Penalize repeated \
            no-progress actions and constraint violations.";

        let futures = prepared.iter().map(|(step_desc, traj)| async move {
            let rel_user = format!("GOAL: {}\n\nSTEP: {step_desc}", trace.goal);
            let drift_user = format!("ORIGINAL GOAL: {}\n\nTRAJECTORY SO FAR:\n{traj}", trace.goal);
            let ((rel, c1), (drift, c2)) = futures::future::join(
                self.scored_check("step_relevance", rel_sys, &rel_user),
                self.scored_check("goal_drift", drift_sys, &drift_user),
            )
            .await;
            (vec![rel, drift], c1 + c2)
        });

        let results = futures::future::join_all(futures).await;
        let cost = results.iter().map(|(_, c)| c).sum();
        let metrics = results.into_iter().map(|(m, _)| m).collect();
        (metrics, cost)
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
            .chat(&self.fast, &self.strong, "rag_decompose", system, &format!("ANSWER:\n{answer}"))
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

        let out = self.chat(&self.fast, &self.strong, "rag_verify", system, &user).await?;
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
        let ar_sys = "Rate how directly the ANSWER addresses the QUESTION. Respond ONLY as JSON \
            {\"score\":0-100,\"confidence\":0.0-1.0,\"reason\":\"one short sentence\"}. 100 = fully answers, 0 = off-topic.";
        let cp_sys = "Given the QUESTION, the retrieved CONTEXTS, and the ANSWER, rate the signal-to-noise \
            of the contexts: what fraction are actually relevant/necessary to answer the question. \
            Respond ONLY as JSON {\"score\":0-100,\"confidence\":0.0-1.0,\"reason\":\"one short sentence\"}. \
            100 = all contexts relevant, 0 = all noise.";
        let cr_sys = "Given the QUESTION, the CONTEXTS, and the ANSWER, rate context recall: do the contexts \
            contain all the information needed to support the answer? Respond ONLY as JSON \
            {\"score\":0-100,\"confidence\":0.0-1.0,\"reason\":\"one short sentence\"}. 100 = fully covered, 0 = key info missing.";
        let ctx_joined = triple
            .contexts
            .iter()
            .enumerate()
            .map(|(i, c)| format!("[Context {}]: {c}", i + 1))
            .collect::<Vec<_>>()
            .join("\n");
        let ar_user = format!("QUESTION: {}\nANSWER: {}", triple.question, triple.answer);
        let cp_user = format!(
            "QUESTION: {}\n\nCONTEXTS:\n{ctx_joined}\n\nANSWER: {}",
            triple.question, triple.answer
        );
        let cr_user = cp_user.clone();

        let (verify_res, ar, cp, cr) = futures::future::join4(
            self.verify_claims(&claims, &triple.contexts),
            self.scored_check("answer_relevancy", ar_sys, &ar_user),
            self.scored_check("context_precision", cp_sys, &cp_user),
            self.scored_check("context_recall", cr_sys, &cr_user),
        )
        .await;

        let (statuses, verify_cost) = verify_res?;
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
        let mut spans = Vec::with_capacity(claims.len());
        for ((claim_text, quote), status) in claims.iter().zip(statuses.iter()) {
            let span = deterministic::find_span(&triple.answer, quote);
            spans.push(ClaimSpan {
                start: span.map(|(s, _)| s).unwrap_or(0),
                end: span.map(|(_, e)| e).unwrap_or(0),
                text: claim_text.clone(),
                supported: *status == ClaimStatus::Supported,
                contradicted: *status == ClaimStatus::Contradicted,
            });
        }

        // A ratio dilutes severity: 3 harmless-correct claims outvote 1 catastrophically
        // wrong one. Name the contradiction explicitly so the headline can't read "fine".
        let faith_reason = if any_contradiction {
            format!(
                "{supported_count}/{} claims grounded — {contradicted_count} CONTRADICTS the context (hallucination)",
                claims.len()
            )
        } else {
            format!("{supported_count}/{} claims grounded in the contexts", claims.len())
        };
        let faithfulness = MetricResult::model("faithfulness", faith_score, 0.9, faith_reason);

        let (ar_m, ar_c) = ar;
        let (cp_m, cp_c) = cp;
        let (cr_m, cr_c) = cr;
        let total_cost = decompose_cost + verify_cost + ar_c + cp_c + cr_c;

        Ok((
            vec![faithfulness, ar_m, cp_m, cr_m],
            spans,
            any_contradiction,
            total_cost,
        ))
    }
}
