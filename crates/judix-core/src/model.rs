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
        for c in [&self.content, &self.reasoning_content, &self.reasoning] {
            if let Some(s) = c {
                if !s.trim().is_empty() {
                    return s.clone();
                }
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
    #[serde(default)]
    supported: bool,
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
            cache: ModelCache::new(1000, 3600),
        })
    }

    /// Per-model list-price rate (USD per 1M tokens: input, output). Free-tier use
    /// bills $0, but exposing the notional cost lets the UI say "this would cost
    /// $X at list price — you paid $0." Unknown models (e.g. local OmniRoute) → 0.
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

    /// One chat call against a specific provider. Requests a JSON object via
    /// `response_format`, caches by `(check, model, input)`, and computes cost
    /// from token usage. A cache hit returns instantly at $0.
    async fn chat(
        &self,
        provider: &Provider,
        check: &str,
        system: &str,
        user: &str,
    ) -> Result<ChatOut, String> {
        let cache_input = format!("{system}\n---\n{user}");
        if let Some(cached) = self.cache.get(check, &provider.model, &cache_input).await {
            return Ok(ChatOut {
                text: cached,
                cost_usd: 0.0,
            });
        }

        let url = format!(
            "{}/chat/completions",
            provider.base_url.trim_end_matches('/')
        );
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

        let res = self
            .http
            .post(&url)
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", provider.api_key))
            .json(&body)
            .send()
            .await
            .map_err(|e| e.to_string())?;

        if !res.status().is_success() {
            let status = res.status();
            let text = res.text().await.unwrap_or_default();
            return Err(format!("model API {status}: {text}"));
        }

        let chat: ChatResponse = res.json().await.map_err(|e| e.to_string())?;
        let text = chat.choices.first().map(|c| c.message.text()).unwrap_or_default();
        let usage = chat.usage.unwrap_or_default();
        let (rin, rout) = Self::rate(&provider.model);
        let cost_usd = (usage.prompt_tokens as f64 * rin + usage.completion_tokens as f64 * rout)
            / 1_000_000.0;

        self.cache
            .insert(check, &provider.model, &cache_input, text.clone())
            .await;
        Ok(ChatOut { text, cost_usd })
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
        let (fast_out, mut cost) = match self.chat(&self.fast, check, system, user).await {
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
        if fast.confidence < 0.5 {
            if let Ok(o) = self.chat(&self.strong, check, system, user).await {
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
            .chat(&self.strong_or_fast(), "rag_decompose", system, &format!("ANSWER:\n{answer}"))
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
    ) -> Result<(Vec<bool>, f64), String> {
        let system = "Given CONTEXTS and a numbered list of CLAIMS, decide for EACH claim whether it \
            is supported by the contexts. `supported` is true ONLY if the claim can be directly \
            inferred from some context; if it conflicts with or goes beyond the contexts, it is false. \
            Respond ONLY as JSON {\"results\":[{\"id\":1,\"supported\":true,\"context_index\":1,\"reason\":\"...\"}]}.";
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

        let out = self.chat(&self.fast, "rag_verify", system, &user).await?;
        let parsed: VerifyOut = Self::parse(&out.text)
            .ok_or_else(|| format!("unparseable verification: {}", out.text))?;

        // Map verdicts back by 1-based id; default unsupported if the model omits one.
        let mut supported = vec![false; claims.len()];
        for r in parsed.results {
            let idx = r.id as usize;
            if idx >= 1 && idx <= claims.len() {
                supported[idx - 1] = r.supported;
            }
        }
        Ok((supported, out.cost_usd))
    }

    fn strong_or_fast(&self) -> Provider {
        self.strong.clone()
    }

    /// Full RAG scoring: faithfulness (decompose → batched verify → ratio in Rust,
    /// with unsupported spans), plus answer_relevancy, context_precision, and
    /// context_recall. Returns `(metrics, unsupported_spans, cost)`.
    pub async fn score_rag_triple(
        &self,
        triple: &RagTriple,
    ) -> Result<(Vec<MetricResult>, Vec<ClaimSpan>, f64), String> {
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

        let (supported, verify_cost) = verify_res?;
        let supported_count = supported.iter().filter(|s| **s).count();
        let faith_score = deterministic::faithfulness_ratio(supported_count, claims.len());

        // Map each claim to a char-span via its verbatim quote (unsupported → red).
        let mut spans = Vec::with_capacity(claims.len());
        for ((claim_text, quote), &ok) in claims.iter().zip(supported.iter()) {
            let span = deterministic::find_span(&triple.answer, quote);
            spans.push(ClaimSpan {
                start: span.map(|(s, _)| s).unwrap_or(0),
                end: span.map(|(_, e)| e).unwrap_or(0),
                text: claim_text.clone(),
                supported: ok,
            });
        }

        let faithfulness = MetricResult::model(
            "faithfulness",
            faith_score,
            0.9,
            format!("{supported_count}/{} claims grounded in the contexts", claims.len()),
        );

        let (ar_m, ar_c) = ar;
        let (cp_m, cp_c) = cp;
        let (cr_m, cr_c) = cr;
        let total_cost = decompose_cost + verify_cost + ar_c + cp_c + cr_c;

        Ok((vec![faithfulness, ar_m, cp_m, cr_m], spans, total_cost))
    }
}
