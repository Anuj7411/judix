//! Core data types for Judix: inputs (traces / RAG triples), metric results, and
//! reports. All types are `serde`-serializable so the server can stream them as
//! SSE/JSON and the CLI can print them. See blueprint §9.2 / §9.3.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Where a metric's score came from.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MetricSource {
    /// Computed by the Rust engine with **no** model call.
    Deterministic,
    /// Produced by the AI model (explanation layer).
    Model,
}

/// Derived, purely-visual color band (§5.2). Thresholds: ≥80 green, 50–79 amber, <50 red.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Band {
    Green,
    Amber,
    Red,
}

impl Band {
    /// Map any 0–100 score to a band. Values are clamped conceptually — a score
    /// above 100 is still green, below 0 still red.
    pub fn from_score(score: f32) -> Band {
        if score >= 80.0 {
            Band::Green
        } else if score >= 50.0 {
            Band::Amber
        } else {
            Band::Red
        }
    }
}

/// A single metric's result (§5.1). Deterministic metrics carry `pass` + `raw_value`;
/// model metrics carry `confidence` + `reason`. Uncomputable metrics set `na`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricResult {
    pub name: String,
    pub score: f32,
    pub band: Band,
    pub source: MetricSource,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pass: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_value: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    pub na: bool,
    pub low_confidence: bool,
}

impl MetricResult {
    /// Construct a deterministic metric with a computed score, pass flag, and raw value.
    pub fn deterministic(name: &str, score: f32, pass: bool, raw_value: f64) -> Self {
        MetricResult {
            name: name.to_string(),
            score,
            band: Band::from_score(score),
            source: MetricSource::Deterministic,
            pass: Some(pass),
            raw_value: Some(raw_value),
            confidence: None,
            reason: None,
            na: false,
            low_confidence: false,
        }
    }

    /// Construct a model metric with a score, confidence, and one-line reason.
    /// Flags `low_confidence` when confidence < 0.5 (§5.6).
    pub fn model(name: &str, score: f32, confidence: f32, reason: impl Into<String>) -> Self {
        MetricResult {
            name: name.to_string(),
            score,
            band: Band::from_score(score),
            source: MetricSource::Model,
            pass: None,
            raw_value: None,
            confidence: Some(confidence),
            reason: Some(reason.into()),
            na: false,
            low_confidence: confidence < 0.5,
        }
    }

    /// Construct a "not applicable" metric — excluded from composite scoring.
    pub fn na(name: &str, source: MetricSource) -> Self {
        MetricResult {
            name: name.to_string(),
            score: 0.0,
            band: Band::Amber,
            source,
            pass: None,
            raw_value: None,
            confidence: None,
            reason: None,
            na: true,
            low_confidence: false,
        }
    }

    /// Attach a reason string (chainable).
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }
}

/// Per-step composite result (§5.3).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepScore {
    pub index: usize,
    pub label: String,
    pub metrics: Vec<MetricResult>,
    pub step_quality: f32,
    pub band: Band,
    /// True when the step had no computable (non-`na`) metrics, so it is excluded
    /// from the run mean. Not in the original spec struct but needed to represent
    /// model-only steps evaluated before any model key is configured.
    pub na: bool,
}

/// Run-level agent report (§5.4). `run_quality` is the headline number.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentReport {
    pub run_quality: f32,
    pub band: Band,
    pub steps: Vec<StepScore>,
    pub latency_ms: u64,
    pub model_cost_usd: f64,
    /// Fraction (0..1) of scored metrics computed with no model call.
    pub deterministic_share: f32,
}

/// A claim mapped back to a char-span in the RAG answer, for red highlighting (§7.4).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimSpan {
    pub start: usize,
    pub end: usize,
    pub text: String,
    pub supported: bool,
    /// True when the claim actively *conflicts* with the contexts (a hallucination),
    /// as opposed to merely lacking evidence. Lets the UI show red vs amber, and
    /// drives the RAG critical-fail cap.
    #[serde(default)]
    pub contradicted: bool,
}

/// RAG triple report (§5.5).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagReport {
    pub rag_quality: f32,
    pub band: Band,
    pub metrics: Vec<MetricResult>,
    pub unsupported_spans: Vec<ClaimSpan>,
    pub latency_ms: u64,
    pub model_cost_usd: f64,
}

// ---------------------------------------------------------------------------
// Input types (§9.3)
// ---------------------------------------------------------------------------

/// One step of an agent trajectory. `kind` is typically "llm" or "tool_call".
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStep {
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub args: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

impl AgentStep {
    pub fn is_tool_call(&self) -> bool {
        self.kind == "tool_call" && self.name.is_some()
    }
}

/// A full agent trace to be scored.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTrace {
    pub goal: String,
    pub steps: Vec<AgentStep>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_tools: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_schemas: Option<HashMap<String, Value>>,
}

/// A RAG (question, contexts, answer) triple to be scored.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagTriple {
    pub question: String,
    pub contexts: Vec<String>,
    pub answer: String,
}
