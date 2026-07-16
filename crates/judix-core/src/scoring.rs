//! Composite scoring math (§5). Turns per-metric results into Step/Run/Rag
//! quality scores, applying the renormalized weighting and the hard caps.

use crate::deterministic;
use crate::types::{
    AgentReport, AgentTrace, Band, MetricResult, MetricSource, RagReport, StepScore,
};

/// Agent step-metric weights (§5.3).
fn agent_weight(metric_name: &str) -> f32 {
    match metric_name {
        "tool_call_correctness" => 0.30,
        "loop_free" => 0.15,
        "step_relevance" => 0.30,
        "goal_drift" => 0.25,
        _ => 0.0,
    }
}

/// RAG metric weights (§5.5).
fn rag_weight(metric_name: &str) -> f32 {
    match metric_name {
        "faithfulness" => 0.40,
        "answer_relevancy" => 0.25,
        "context_precision" => 0.20,
        "context_recall" => 0.15,
        _ => 0.0,
    }
}

/// Weighted average over the available (non-`na`) metrics, weights renormalized
/// to sum to 1 across whatever is present. Returns `None` if nothing is available.
fn weighted_average(metrics: &[MetricResult], weight_of: impl Fn(&str) -> f32) -> Option<f32> {
    let mut num = 0.0f32;
    let mut den = 0.0f32;
    for m in metrics {
        if m.na {
            continue;
        }
        let w = weight_of(&m.name);
        if w == 0.0 {
            continue;
        }
        num += m.score * w;
        den += w;
    }
    if den == 0.0 {
        None
    } else {
        Some(num / den)
    }
}

/// Does this step contain a critical deterministic FAIL (a looping step, or a
/// failed tool-call check)? Such a step is forced red (§5.3 hard rule).
fn has_loop_fail(metrics: &[MetricResult]) -> bool {
    metrics
        .iter()
        .any(|m| m.name == "loop_free" && m.pass == Some(false))
}

fn has_critical_fail(metrics: &[MetricResult]) -> bool {
    metrics.iter().any(|m| {
        (m.name == "loop_free" || m.name == "tool_call_correctness") && m.pass == Some(false)
    })
}

/// Compose one step's metrics into a `StepScore`, applying the loop cap.
pub fn score_step(index: usize, label: String, metrics: Vec<MetricResult>) -> StepScore {
    match weighted_average(&metrics, agent_weight) {
        None => StepScore {
            index,
            label,
            metrics,
            step_quality: 0.0,
            band: Band::Amber,
            na: true,
        },
        Some(mut q) => {
            // Hard rule: a looping step is a real failure — cap at 49 (force red).
            if has_loop_fail(&metrics) {
                q = q.min(49.0);
            }
            StepScore {
                index,
                label,
                metrics,
                step_quality: q,
                band: Band::from_score(q),
                na: false,
            }
        }
    }
}

/// Build a full `AgentReport` from a trace plus optional per-step model metrics.
///
/// `model_metrics[i]` holds the model-produced metrics (step_relevance, goal_drift)
/// for step `i`. Pass empty vecs when no model is configured — those metrics simply
/// don't count toward the composite, and `deterministic_share` reflects that.
pub fn score_agent(
    trace: &AgentTrace,
    model_metrics: &[Vec<MetricResult>],
    latency_ms: u64,
    model_cost_usd: f64,
) -> AgentReport {
    // Run-level deterministic tool-call correctness, attached to every tool step.
    let tool_metric = deterministic::tool_call_correctness(trace);
    let loop_metrics = deterministic::loop_free_metrics(trace);

    let mut steps: Vec<StepScore> = Vec::with_capacity(trace.steps.len());
    let mut det_count = 0usize;
    let mut total_count = 0usize;
    let mut any_critical_fail = false;

    for (i, step) in trace.steps.iter().enumerate() {
        let mut metrics: Vec<MetricResult> = Vec::new();

        if step.is_tool_call() {
            metrics.push(tool_metric.clone());
            if let Some(Some(lm)) = loop_metrics.get(i) {
                metrics.push(lm.clone());
            }
        }
        // Model metrics for this step (step_relevance, goal_drift), if any.
        if let Some(mm) = model_metrics.get(i) {
            metrics.extend(mm.iter().cloned());
        }

        for m in &metrics {
            if m.na {
                continue;
            }
            total_count += 1;
            if m.source == MetricSource::Deterministic {
                det_count += 1;
            }
        }
        if has_critical_fail(&metrics) {
            any_critical_fail = true;
        }

        let label = step
            .name
            .clone()
            .unwrap_or_else(|| format!("{} step", step.kind));
        steps.push(score_step(i, label, metrics));
    }

    // Run quality = mean of available (non-na) step qualities.
    let available: Vec<f32> = steps
        .iter()
        .filter(|s| !s.na)
        .map(|s| s.step_quality)
        .collect();
    let mut run_quality = if available.is_empty() {
        0.0
    } else {
        available.iter().sum::<f32>() / available.len() as f32
    };
    // Cap at 59 if any step has a critical FAIL (loop or tool_call FAIL) (§5.4).
    if any_critical_fail {
        run_quality = run_quality.min(59.0);
    }

    let deterministic_share = if total_count == 0 {
        1.0
    } else {
        det_count as f32 / total_count as f32
    };

    AgentReport {
        run_quality,
        band: Band::from_score(run_quality),
        steps,
        latency_ms,
        model_cost_usd,
        deterministic_share,
    }
}

/// Compose RAG metrics into a `RagReport`, applying the faithfulness cap (§5.5):
/// if `faithfulness < 50`, cap `rag_quality` at 49 (red).
pub fn score_rag(
    metrics: Vec<MetricResult>,
    unsupported_spans: Vec<crate::types::ClaimSpan>,
    latency_ms: u64,
    model_cost_usd: f64,
) -> RagReport {
    let faithfulness = metrics
        .iter()
        .find(|m| m.name == "faithfulness" && !m.na)
        .map(|m| m.score);

    let mut rag_quality = weighted_average(&metrics, rag_weight).unwrap_or(0.0);
    if let Some(f) = faithfulness {
        if f < 50.0 {
            rag_quality = rag_quality.min(49.0);
        }
    }

    RagReport {
        rag_quality,
        band: Band::from_score(rag_quality),
        metrics,
        unsupported_spans,
        latency_ms,
        model_cost_usd,
    }
}
