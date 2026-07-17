//! Deterministic scoring algorithms (§6) — NO model, NO network, NO clock, NO
//! randomness. Same input always yields the same output. This is the product's
//! hero: it catches wrong tool calls and loops instantly, for $0.

use crate::types::{AgentTrace, MetricResult, MetricSource};
use serde_json::Value;
use std::collections::HashMap;

/// Sliding-window size for loop detection (§6.2).
pub const LOOP_WINDOW: usize = 5;
/// Penalty subtracted from tool-call score when args fail schema validation (§6.1).
const ARG_INVALID_PENALTY: f32 = 20.0;

/// Produce a canonical string form of a JSON value with object keys sorted, so
/// that two semantically-identical tool-call arg objects hash identically
/// regardless of key order.
pub fn canonical_json(v: &Value) -> String {
    match v {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let inner: Vec<String> = keys
                .iter()
                .map(|k| {
                    format!(
                        "{}:{}",
                        serde_json::to_string(k).unwrap_or_else(|_| format!("\"{k}\"")),
                        canonical_json(&map[*k])
                    )
                })
                .collect();
            format!("{{{}}}", inner.join(","))
        }
        Value::Array(items) => {
            let inner: Vec<String> = items.iter().map(canonical_json).collect();
            format!("[{}]", inner.join(","))
        }
        other => other.to_string(),
    }
}

/// A tool call extracted from a trace, with its originating step index.
#[derive(Debug, Clone)]
pub struct ToolCallView {
    pub index: usize,
    pub name: String,
    pub args: Value,
}

/// Extract every tool-call step from a trace, in order.
pub fn tool_calls(trace: &AgentTrace) -> Vec<ToolCallView> {
    trace
        .steps
        .iter()
        .enumerate()
        .filter(|(_, s)| s.is_tool_call())
        .map(|(i, s)| ToolCallView {
            index: i,
            name: s.name.clone().unwrap_or_default(),
            args: s.args.clone().unwrap_or(Value::Null),
        })
        .collect()
}

/// Compute an F1 from an intersection count and the two side sizes, guarding
/// against zero-division. Two empty sides is a perfect match (nothing expected,
/// nothing called); one empty side is a complete miss.
fn f1(intersection: f64, pred_size: f64, expected_size: f64) -> f64 {
    if pred_size == 0.0 && expected_size == 0.0 {
        return 1.0;
    }
    if pred_size == 0.0 || expected_size == 0.0 {
        return 0.0;
    }
    let precision = intersection / pred_size;
    let recall = intersection / expected_size;
    if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}

/// Multiset counts of a slice of strings.
fn counts(items: &[String]) -> HashMap<&str, usize> {
    let mut m: HashMap<&str, usize> = HashMap::new();
    for it in items {
        *m.entry(it.as_str()).or_insert(0) += 1;
    }
    m
}

/// **Tool-call correctness (§6.1).** Final F1 = 0.5·set_f1 + 0.5·bag_f1, over the
/// tool names actually called vs. the golden `expected_tools`. Invalid args (per a
/// supplied JSON schema) force `pass=false` and subtract 20 (floor 0). Returns an
/// `na` metric when there is no golden label.
pub fn tool_call_correctness(trace: &AgentTrace) -> MetricResult {
    let name = "tool_call_correctness";
    let expected = match &trace.expected_tools {
        Some(e) => e.clone(),
        None => return MetricResult::na(name, MetricSource::Deterministic),
    };

    let calls = tool_calls(trace);
    let predicted: Vec<String> = calls.iter().map(|c| c.name.clone()).collect();

    // --- Set F1: unique tool names ---
    let pred_set: std::collections::BTreeSet<&str> = predicted.iter().map(|s| s.as_str()).collect();
    let exp_set: std::collections::BTreeSet<&str> = expected.iter().map(|s| s.as_str()).collect();
    let set_inter = pred_set.intersection(&exp_set).count() as f64;
    let set_f1 = f1(set_inter, pred_set.len() as f64, exp_set.len() as f64);

    // --- Bag (multiset) F1: ordered-call overlap by name ---
    let pred_counts = counts(&predicted);
    let exp_counts = counts(&expected);
    let mut bag_inter = 0.0;
    for (tool, pc) in &pred_counts {
        if let Some(ec) = exp_counts.get(tool) {
            bag_inter += (*pc).min(*ec) as f64;
        }
    }
    let bag_f1 = f1(bag_inter, predicted.len() as f64, expected.len() as f64);

    let final_f1 = 0.5 * set_f1 + 0.5 * bag_f1;
    let mut score = (final_f1 * 100.0) as f32;

    // --- Arg-schema validation (§6.3) ---
    let (args_ok, validation_errors) = validate_all_args(trace, &calls);
    if !args_ok {
        score = (score - ARG_INVALID_PENALTY).max(0.0);
    }

    let pass = args_ok && final_f1 >= 0.8;
    let mut metric = MetricResult::deterministic(name, score, pass, final_f1);
    if !validation_errors.is_empty() {
        metric = metric.with_reason(format!("Invalid tool args: {}", validation_errors.join("; ")));
    }
    metric
}

/// Validate each tool call's args against its schema (if one is supplied).
/// Returns `(all_valid, error_messages)`.
fn validate_all_args(trace: &AgentTrace, calls: &[ToolCallView]) -> (bool, Vec<String>) {
    let schemas = match &trace.tool_schemas {
        Some(s) => s,
        None => return (true, Vec::new()),
    };
    let mut all_ok = true;
    let mut errors = Vec::new();
    for call in calls {
        if let Some(schema) = schemas.get(&call.name) {
            // A malformed schema can't validate anything — skip it rather than
            // punishing the agent for a bad golden label.
            if let Ok(validator) = jsonschema::validator_for(schema) {
                // `validate` returns the validation errors as an iterator in its
                // `Err` arm; collect them into a readable reason.
                if let Err(errs) = validator.validate(&call.args) {
                    all_ok = false;
                    let detail: Vec<String> = errs.map(|e| e.to_string()).collect();
                    errors.push(format!(
                        "{} (step {}): {}",
                        call.name,
                        call.index,
                        if detail.is_empty() {
                            "schema violation".to_string()
                        } else {
                            detail.join(", ")
                        }
                    ));
                }
            }
        }
    }
    (all_ok, errors)
}

/// **Loop / repeat detection (§6.2).** For every step, returns `Some(metric)` for
/// tool-call steps and `None` for others. A tool call is a "repeat" if an identical
/// `(name, canonical_args)` hash appears earlier within the sliding window; the
/// score is `max(0, 100 - repeat_count*25)`, `pass = repeat_count == 0`.
pub fn loop_free_metrics(trace: &AgentTrace) -> Vec<Option<MetricResult>> {
    let name = "loop_free";
    let mut window: Vec<String> = Vec::with_capacity(LOOP_WINDOW);
    let mut out: Vec<Option<MetricResult>> = Vec::with_capacity(trace.steps.len());

    for step in &trace.steps {
        if !step.is_tool_call() {
            out.push(None);
            continue;
        }
        let hash = format!(
            "{}::{}",
            step.name.clone().unwrap_or_default(),
            canonical_json(step.args.as_ref().unwrap_or(&Value::Null))
        );
        // Count identical hashes already in the window (i.e. prior repeats).
        let repeat_count = window.iter().filter(|h| **h == hash).count();
        let score = (100.0 - (repeat_count as f32) * 25.0).max(0.0);
        let pass = repeat_count == 0;
        out.push(Some(MetricResult::deterministic(
            name,
            score,
            pass,
            repeat_count as f64,
        )));

        // Slide the window (keep the most recent LOOP_WINDOW tool-call hashes).
        window.push(hash);
        if window.len() > LOOP_WINDOW {
            window.remove(0);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// RAG deterministic piece (§5.5 / §7.4): the faithfulness *ratio* is computed
// here in Rust from the model's per-claim support verdicts.
// ---------------------------------------------------------------------------

/// Compute `faithfulness = supported/total * 100`. With zero claims we treat the
/// answer as vacuously faithful (100) rather than dividing by zero.
pub fn faithfulness_ratio(supported: usize, total: usize) -> f32 {
    if total == 0 {
        return 100.0;
    }
    (supported as f32) / (total as f32) * 100.0
}

/// Locate a claim's text inside the answer and return its char-span `(start, end)`.
/// Returns `None` if the claim text isn't found verbatim.
pub fn find_span(answer: &str, claim_text: &str) -> Option<(usize, usize)> {
    let start = answer.find(claim_text)?;
    Some((start, start + claim_text.len()))
}
