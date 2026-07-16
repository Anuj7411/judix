use crate::cache::ModelCache;
use crate::deterministic;
use crate::types::{ClaimSpan, MetricResult, MetricSource, RagTriple};
use reqwest::Client;
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Clone)]
pub struct ModelClient {
    http: Client,
    base_url: String,
    api_key: Option<String>,
    model_fast: String,
    model_strong: String,
    cache: ModelCache,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: Msg,
}

#[derive(Deserialize)]
struct Msg {
    content: Option<String>,
    /// Some routed models (reasoning models) leave `content` null and put the
    /// answer in `reasoning_content` / `reasoning`. Fall back to those.
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    reasoning: Option<String>,
}

impl Msg {
    /// The usable text: prefer `content`, else fall back to reasoning fields.
    fn text(&self) -> String {
        let c = self.content.clone().unwrap_or_default();
        if !c.trim().is_empty() {
            return c;
        }
        if let Some(rc) = &self.reasoning_content {
            if !rc.trim().is_empty() {
                return rc.clone();
            }
        }
        self.reasoning.clone().unwrap_or_default()
    }
}

impl ModelClient {
    /// Create a client from env vars. Returns `None` when no model endpoint is
    /// configured (deterministic-only mode).
    ///
    /// | Env var              | Default                          |
    /// |----------------------|----------------------------------|
    /// | `JUDIX_BASE_URL`     | *(must be set to enable)*        |
    /// | `JUDIX_API_KEY`      | *(optional bearer token)*        |
    /// | `JUDIX_MODEL_FAST`   | `auto/best-chat`                 |
    /// | `JUDIX_MODEL_STRONG` | `auto/best-coding`               |
    pub fn from_env() -> Option<Self> {
        let base_url = std::env::var("JUDIX_BASE_URL").ok()?;
        Some(Self {
            http: Client::new(),
            base_url,
            api_key: std::env::var("JUDIX_API_KEY").ok(),
            model_fast: std::env::var("JUDIX_MODEL_FAST")
                .unwrap_or_else(|_| "auto/best-chat".into()),
            model_strong: std::env::var("JUDIX_MODEL_STRONG")
                .unwrap_or_else(|_| "auto/best-coding".into()),
            cache: ModelCache::new(1000, 3600),
        })
    }

    async fn chat(&self, model: &str, system: &str, user: &str) -> Result<String, String> {
        let cache_key = format!("{system}\n---\n{user}");
        if let Some(cached) = self.cache.get(model, &cache_key).await {
            return Ok(cached);
        }

        let url = format!("{}/chat/completions", self.base_url);
        let mut req = self
            .http
            .post(&url)
            .header("Content-Type", "application/json");
        if let Some(key) = &self.api_key {
            req = req.header("Authorization", format!("Bearer {key}"));
        }

        let body = json!({
            "model": model,
            "messages": [
                { "role": "system", "content": system },
                { "role": "user",   "content": user   }
            ],
            "temperature": 0.0,
            "max_tokens": 2048,
            // OmniRoute (and OpenAI-compatible gateways) stream by default via SSE.
            // We want a single JSON body we can deserialize in one shot.
            "stream": false
        });

        let res = req.json(&body).send().await.map_err(|e| e.to_string())?;
        if !res.status().is_success() {
            let status = res.status();
            let text = res.text().await.unwrap_or_default();
            return Err(format!("model API {status}: {text}"));
        }

        let chat: ChatResponse = res.json().await.map_err(|e| e.to_string())?;
        let content = chat
            .choices
            .first()
            .map(|c| c.message.text())
            .unwrap_or_default();

        self.cache.insert(model, &cache_key, content.clone()).await;
        Ok(content)
    }

    /// Best-effort JSON extraction: tries raw parse, ```json fences, first `{`/`[`.
    fn parse_json(text: &str) -> Option<Value> {
        let t = text.trim();
        if let Ok(v) = serde_json::from_str(t) {
            return Some(v);
        }
        // ```json ... ``` or ``` ... ```
        if let Some(start) = t.find("```") {
            let after = &t[start + 3..];
            // skip optional language tag on the first line
            let body_start = after.find('\n').map(|n| n + 1).unwrap_or(0);
            if let Some(end) = after[body_start..].find("```") {
                let block = after[body_start..body_start + end].trim();
                if let Ok(v) = serde_json::from_str(block) {
                    return Some(v);
                }
            }
        }
        // first `{` or `[`
        for (i, c) in t.char_indices() {
            if c == '{' || c == '[' {
                if let Ok(v) = serde_json::from_str(&t[i..]) {
                    return Some(v);
                }
            }
        }
        None
    }

    // ------------------------------------------------------------------
    // Agent metrics
    // ------------------------------------------------------------------

    pub async fn step_relevance(
        &self,
        goal: &str,
        step_kind: &str,
        step_name: &str,
        step_content: &str,
    ) -> MetricResult {
        let system = "You evaluate whether an AI agent's step is relevant to its stated goal.\n\
            Return ONLY a JSON object: {\"score\": <0-100>, \"confidence\": <0.0-1.0>, \"reason\": \"<one sentence>\"}.\n\
            100 = directly advances the goal, 50 = tangentially related, 0 = irrelevant.";
        let user = format!(
            "Goal: {goal}\nStep type: {step_kind}\nStep name: {step_name}\nContent: {step_content}"
        );

        match self.chat(&self.model_fast, system, &user).await {
            Ok(text) => match Self::parse_json(&text) {
                Some(v) => MetricResult::model(
                    "step_relevance",
                    v["score"].as_f64().unwrap_or(50.0) as f32,
                    v["confidence"].as_f64().unwrap_or(0.7) as f32,
                    v["reason"].as_str().unwrap_or("").to_string(),
                ),
                None => MetricResult::na("step_relevance", MetricSource::Model)
                    .with_reason(format!("unparseable model response: {text}")),
            },
            Err(e) => MetricResult::na("step_relevance", MetricSource::Model)
                .with_reason(format!("model call failed: {e}")),
        }
    }

    pub async fn goal_drift(&self, goal: &str, trajectory: &str) -> MetricResult {
        let system = "You evaluate whether an AI agent has drifted from its original goal.\n\
            Return ONLY a JSON object: {\"score\": <0-100>, \"confidence\": <0.0-1.0>, \"reason\": \"<one sentence>\"}.\n\
            100 = perfectly on track, 50 = some drift, 0 = completely lost.";
        let user = format!("Original goal: {goal}\n\nTrajectory so far:\n{trajectory}");

        match self.chat(&self.model_fast, system, &user).await {
            Ok(text) => match Self::parse_json(&text) {
                Some(v) => MetricResult::model(
                    "goal_drift",
                    v["score"].as_f64().unwrap_or(50.0) as f32,
                    v["confidence"].as_f64().unwrap_or(0.7) as f32,
                    v["reason"].as_str().unwrap_or("").to_string(),
                ),
                None => MetricResult::na("goal_drift", MetricSource::Model)
                    .with_reason(format!("unparseable model response: {text}")),
            },
            Err(e) => MetricResult::na("goal_drift", MetricSource::Model)
                .with_reason(format!("model call failed: {e}")),
        }
    }

    /// Score model metrics (step_relevance + goal_drift) for every step.
    /// Returns one `Vec<MetricResult>` per step, suitable for `score_agent()`.
    ///
    /// All model calls run **concurrently** — the per-step metrics are independent
    /// (goal_drift depends only on the trajectory *prefix*, which we precompute
    /// synchronously). For an N-step trace this fires 2N calls in one wave instead
    /// of 2N sequential round-trips, cutting wall-clock from ~N·latency to ~latency.
    pub async fn score_agent_steps(
        &self,
        trace: &crate::types::AgentTrace,
    ) -> Vec<Vec<MetricResult>> {
        // Precompute each step's (kind, name, content) and the trajectory prefix
        // up to and including that step — no awaits, so no ordering constraint.
        let mut trajectory = String::new();
        let mut prepared: Vec<(String, String, String, String)> =
            Vec::with_capacity(trace.steps.len());
        for (i, step) in trace.steps.iter().enumerate() {
            let name = step.name.clone().unwrap_or_else(|| step.kind.clone());
            let content = step
                .content
                .clone()
                .or_else(|| step.result.clone())
                .unwrap_or_default();
            trajectory.push_str(&format!("Step {i}: [{}] {name} {content}\n", step.kind));
            prepared.push((step.kind.clone(), name, content, trajectory.clone()));
        }

        // Fire every step's two metrics concurrently.
        let futures = prepared.iter().map(|(kind, name, content, traj)| async move {
            let (rel, drift) = futures::future::join(
                self.step_relevance(&trace.goal, kind, name, content),
                self.goal_drift(&trace.goal, traj),
            )
            .await;
            vec![rel, drift]
        });
        futures::future::join_all(futures).await
    }

    // ------------------------------------------------------------------
    // RAG metrics
    // ------------------------------------------------------------------

    /// A decomposed claim, paired with the verbatim `quote` from the answer it came
    /// from (so we can map it back to a char-span for red-highlighting).
    async fn decompose_claims(&self, answer: &str) -> Result<Vec<(String, String)>, String> {
        let system = "Decompose the answer into atomic factual claims. Each claim must be a single, \
            independently verifiable statement.\n\
            For each claim, also return `quote`: the EXACT verbatim substring of the answer \
            (copied character-for-character, no paraphrasing) that the claim is drawn from.\n\
            Return ONLY a JSON array of objects: [{\"claim\": \"...\", \"quote\": \"...\"}].";
        let user = format!("Answer:\n{answer}");

        let text = self.chat(&self.model_strong, system, &user).await?;
        let arr = Self::parse_json(&text)
            .and_then(|v| v.as_array().cloned())
            .ok_or_else(|| format!("expected JSON array of claims, got: {text}"))?;
        Ok(arr
            .iter()
            .filter_map(|v| {
                let claim = v["claim"].as_str()?.to_string();
                // Fall back to the claim text itself if no quote was returned.
                let quote = v["quote"].as_str().unwrap_or(&claim).to_string();
                Some((claim, quote))
            })
            .collect())
    }

    /// A per-claim grounding verdict. `Contradicted` is a factual conflict with the
    /// context (a hallucination) — worse than merely `Unsupported` (no evidence).
    async fn verify_claim(
        &self,
        claim: &str,
        contexts: &[String],
    ) -> Result<(ClaimStatus, String), String> {
        let system = "Verify a claim against the context passages.\n\
            Return ONLY a JSON object: {\"status\": \"supported\"|\"unsupported\"|\"contradicted\", \"reason\": \"<one sentence>\"}.\n\
            - \"supported\": the claim is directly inferable from the contexts.\n\
            - \"contradicted\": the claim conflicts with a fact stated in the contexts (a hallucination).\n\
            - \"unsupported\": the contexts neither support nor contradict the claim.";
        let ctx = contexts
            .iter()
            .enumerate()
            .map(|(i, c)| format!("[Context {}]: {c}", i + 1))
            .collect::<Vec<_>>()
            .join("\n");
        let user = format!("Claim: {claim}\n\nContexts:\n{ctx}");

        let text = self.chat(&self.model_strong, system, &user).await?;
        let v = Self::parse_json(&text)
            .ok_or_else(|| format!("expected JSON verification, got: {text}"))?;
        let status = match v["status"].as_str().unwrap_or("unsupported") {
            "supported" => ClaimStatus::Supported,
            "contradicted" => ClaimStatus::Contradicted,
            _ => ClaimStatus::Unsupported,
        };
        Ok((status, v["reason"].as_str().unwrap_or("").to_string()))
    }

    async fn answer_relevancy(&self, question: &str, answer: &str) -> MetricResult {
        let system = "Evaluate whether the answer is relevant to the question.\n\
            Return ONLY a JSON object: {\"score\": <0-100>, \"confidence\": <0.0-1.0>, \"reason\": \"<one sentence>\"}.\n\
            100 = perfectly answers the question, 50 = partially relevant, 0 = off-topic.";
        let user = format!("Question: {question}\nAnswer: {answer}");

        match self.chat(&self.model_fast, system, &user).await {
            Ok(text) => match Self::parse_json(&text) {
                Some(v) => MetricResult::model(
                    "answer_relevancy",
                    v["score"].as_f64().unwrap_or(50.0) as f32,
                    v["confidence"].as_f64().unwrap_or(0.7) as f32,
                    v["reason"].as_str().unwrap_or("").to_string(),
                ),
                None => MetricResult::na("answer_relevancy", MetricSource::Model),
            },
            Err(_) => MetricResult::na("answer_relevancy", MetricSource::Model),
        }
    }

    /// Full RAG pipeline: decompose → verify each claim (concurrently) → compute
    /// faithfulness. `answer_relevancy` runs concurrently with the verifications.
    /// Returns `(metrics, spans, any_contradiction)`; the caller applies the
    /// contradiction hard-cap in `score_rag`.
    pub async fn score_rag_triple(
        &self,
        triple: &RagTriple,
    ) -> Result<(Vec<MetricResult>, Vec<ClaimSpan>, bool), String> {
        // Step 1 (must be first): decompose the answer into atomic claims + quotes.
        let claims = self.decompose_claims(&triple.answer).await?;

        // Step 2: verify every claim concurrently, and run answer_relevancy in the
        // same wave (it's independent of the verifications).
        let verify_all = futures::future::join_all(
            claims
                .iter()
                .map(|(claim, _quote)| self.verify_claim(claim, &triple.contexts)),
        );
        let (verdicts, relevancy) = futures::future::join(
            verify_all,
            self.answer_relevancy(&triple.question, &triple.answer),
        )
        .await;

        let mut supported_count = 0usize;
        let mut any_contradiction = false;
        let mut spans = Vec::with_capacity(claims.len());

        for ((claim, quote), verdict) in claims.iter().zip(verdicts) {
            let status = verdict.map(|(s, _)| s).unwrap_or(ClaimStatus::Unsupported);
            let supported = status == ClaimStatus::Supported;
            if supported {
                supported_count += 1;
            }
            if status == ClaimStatus::Contradicted {
                any_contradiction = true;
            }
            // Map the claim back to the answer via its verbatim quote.
            let char_span = deterministic::find_span(&triple.answer, quote);
            spans.push(ClaimSpan {
                start: char_span.map(|(s, _)| s).unwrap_or(0),
                end: char_span.map(|(_, e)| e).unwrap_or(0),
                text: claim.clone(),
                supported,
            });
        }

        let faith_score = deterministic::faithfulness_ratio(supported_count, claims.len());
        let faith_reason = if any_contradiction {
            format!(
                "{supported_count}/{} claims supported — contains a contradiction (hallucination)",
                claims.len()
            )
        } else {
            format!("{supported_count}/{} claims supported", claims.len())
        };
        let faithfulness = MetricResult::model("faithfulness", faith_score, 0.85, faith_reason);

        Ok((vec![faithfulness, relevancy], spans, any_contradiction))
    }
}

/// Per-claim grounding verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClaimStatus {
    Supported,
    Unsupported,
    Contradicted,
}
