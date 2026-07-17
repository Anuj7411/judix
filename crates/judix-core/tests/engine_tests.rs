//! Deterministic-engine unit tests (§6.4). These run with NO model and NO network,
//! and pin the exact scoring behavior for the three demo fixtures plus edge cases.

use judix_core::deterministic::{
    canonical_json, faithfulness_ratio, find_span, loop_free_metrics, tool_call_correctness,
    tool_calls,
};
use judix_core::scoring::{score_agent, score_rag};
use judix_core::types::{AgentStep, AgentTrace, Band, ClaimSpan, MetricResult, MetricSource};
use serde_json::json;

fn load_trace(name: &str) -> AgentTrace {
    let path = format!("{}/../../demos/{}", env!("CARGO_MANIFEST_DIR"), name);
    let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse {name}: {e}"))
}

// ---------------------------------------------------------------------------
// canonical_json — key order must not matter
// ---------------------------------------------------------------------------

#[test]
fn canonical_json_is_key_order_independent() {
    let a = json!({ "area": "downtown", "party_size": 2, "day": "Friday" });
    let b = json!({ "day": "Friday", "party_size": 2, "area": "downtown" });
    assert_eq!(canonical_json(&a), canonical_json(&b));
}

#[test]
fn canonical_json_distinguishes_different_values() {
    let a = json!({ "area": "downtown" });
    let b = json!({ "area": "midtown" });
    assert_ne!(canonical_json(&a), canonical_json(&b));
}

// ---------------------------------------------------------------------------
// clean.json — happy path → green
// ---------------------------------------------------------------------------

#[test]
fn clean_demo_is_green_with_perfect_tool_f1() {
    let trace = load_trace("clean.json");

    let tc = tool_call_correctness(&trace);
    assert!(!tc.na, "expected_tools present, should compute");
    assert_eq!(tc.raw_value, Some(1.0), "clean trace should have F1 = 1.0");
    assert_eq!(tc.pass, Some(true));
    assert_eq!(tc.score, 100.0);

    // No loops anywhere.
    let loops = loop_free_metrics(&trace);
    for lm in loops.into_iter().flatten() {
        assert_eq!(lm.pass, Some(true), "clean trace must have no loops");
    }

    let report = score_agent(&trace, &[], 0, 0.0);
    assert!(
        report.run_quality >= 80.0,
        "clean run_quality should be green, got {}",
        report.run_quality
    );
    assert_eq!(report.band, Band::Green);
    // With no model configured, 100% of scored metrics are deterministic.
    assert_eq!(report.deterministic_share, 1.0);
}

// ---------------------------------------------------------------------------
// wrong_tool.json — the money demo → wrong tool + loop, red steps
// ---------------------------------------------------------------------------

#[test]
fn wrong_tool_demo_fails_tool_check_and_flags_loops() {
    let trace = load_trace("wrong_tool.json");

    // Three identical downtown searches, expected search+check+book → F1 well below 0.8.
    let tc = tool_call_correctness(&trace);
    assert_eq!(tc.pass, Some(false));
    assert!(
        tc.raw_value.unwrap() < 0.5,
        "F1 should be low, got {:?}",
        tc.raw_value
    );

    // The 2nd and 3rd identical calls must be flagged as loops (pass=false).
    let loops = loop_free_metrics(&trace);
    let loop_fails = loops
        .iter()
        .flatten()
        .filter(|m| m.pass == Some(false))
        .count();
    assert!(
        loop_fails >= 2,
        "expected >=2 loop failures, got {loop_fails}"
    );

    let report = score_agent(&trace, &[], 0, 0.0);
    // Deterministic-only (no model): F1 0.417 + loop caps put the headline at
    // ~51.5 (Amber) with the two repeat steps already Red. The run goes fully
    // Red (<50) once the model's low step_relevance/goal_drift land (§5.3 weights
    // 0.30+0.25). Pin the deterministic band precisely to catch score drift.
    assert!(
        report.run_quality > 50.0 && report.run_quality <= 59.0,
        "deterministic run_quality should be ~51.5 (amber, critical-fail capped), got {}",
        report.run_quality
    );
    assert_eq!(report.band, Band::Amber);

    // The repeated-call steps must be individually red (loop cap → <50).
    let red_tool_steps = report
        .steps
        .iter()
        .filter(|s| s.band == Band::Red)
        .count();
    assert!(
        red_tool_steps >= 2,
        "expected >=2 red steps from the loop, got {red_tool_steps}"
    );
}

// ---------------------------------------------------------------------------
// RAG faithfulness ratio + cap (§5.5) — testable without a model
// ---------------------------------------------------------------------------

/// Every score the engine emits must lie inside the 0–100 range it advertises.
/// The clean demo used to report run_quality = 100.00001: renormalizing a weighted
/// average divides by a sum of weights with no exact binary representation, and the
/// existing one-sided `>= 80.0` assertion happily accepted the overflow.
#[test]
fn all_scores_stay_within_0_to_100() {
    for name in ["clean.json", "wrong_tool.json"] {
        let trace = load_trace(name);
        let report = score_agent(&trace, &[], 0, 0.0);
        assert!(
            (0.0..=100.0).contains(&report.run_quality),
            "{name}: run_quality {} outside 0..=100",
            report.run_quality
        );
        for s in &report.steps {
            assert!(
                (0.0..=100.0).contains(&s.step_quality),
                "{name}: step {} quality {} outside 0..=100",
                s.index,
                s.step_quality
            );
            for m in &s.metrics {
                assert!(
                    (0.0..=100.0).contains(&m.score),
                    "{name}: step {} metric {} score {} outside 0..=100",
                    s.index,
                    m.name,
                    m.score
                );
            }
        }
    }
}

/// The deterministic engine must stay safe at any trace size: it's the path that
/// always runs, on every request, before any cap applies. A 10k-step trace inside
/// axum's 2MB body limit is trivially reachable by an attacker, so O(n²) here (or a
/// panic) would be a free DoS. Loop detection uses a fixed-size sliding window, so
/// this should be linear and effectively instant.
#[test]
fn deterministic_engine_handles_a_huge_trace_fast() {
    let steps: Vec<AgentStep> = (0..10_000)
        .map(|i| AgentStep {
            kind: "tool_call".into(),
            name: Some("search".into()),
            args: Some(json!({ "q": format!("query {i}") })),
            result: Some("ok".into()),
            content: None,
        })
        .collect();
    let trace = AgentTrace {
        goal: "stress".into(),
        steps,
        expected_tools: Some(vec!["search".into()]),
        tool_schemas: None,
    };

    let t0 = std::time::Instant::now();
    let report = score_agent(&trace, &[], 0, 0.0);
    let elapsed = t0.elapsed();

    assert_eq!(report.steps.len(), 10_000);
    assert!((0.0..=100.0).contains(&report.run_quality));
    assert!(
        elapsed.as_secs() < 5,
        "10k-step deterministic scoring took {elapsed:?} — should be ~linear"
    );
}

#[test]
fn faithfulness_ratio_math() {
    assert_eq!(faithfulness_ratio(1, 2), 50.0);
    assert_eq!(faithfulness_ratio(2, 2), 100.0);
    assert_eq!(faithfulness_ratio(0, 3), 0.0);
    assert_eq!(faithfulness_ratio(0, 0), 100.0); // vacuously faithful
}

#[test]
fn rag_quality_capped_when_faithfulness_below_50() {
    // Mirrors rag_hallucination.json: 1 of 2 claims supported → faithfulness 50,
    // but push it just under to exercise the cap, with other metrics high.
    let metrics = vec![
        MetricResult::model("faithfulness", 49.0, 0.9, "1 of ~2 claims grounded"),
        MetricResult::model("answer_relevancy", 95.0, 0.9, "directly answers"),
        MetricResult::model("context_precision", 90.0, 0.9, "relevant contexts"),
        MetricResult::model("context_recall", 90.0, 0.9, "covers the answer"),
    ];
    let report = score_rag(metrics, vec![], false, 0, 0.0);
    assert!(
        report.rag_quality <= 49.0,
        "ungrounded answer must be capped, got {}",
        report.rag_quality
    );
    assert_eq!(report.band, Band::Red);
}

#[test]
fn contradiction_caps_rag_quality_even_when_ratio_looks_healthy() {
    // Mirrors rag_hallucination.json under real Gemini decomposition: 3 of 4 claims
    // are grounded (faithfulness 75) and the answer reads perfectly relevant — but
    // one claim CONTRADICTS the context ("30 days" vs the policy's 14). Without the
    // critical-fail cap the ratio dilutes that to 61.6 (amber, "mostly fine") even
    // though acting on the answer causes real harm.
    let metrics = vec![
        MetricResult::model("faithfulness", 75.0, 0.9, "3/4 grounded — 1 CONTRADICTS the context"),
        MetricResult::model("answer_relevancy", 100.0, 1.0, "directly answers"),
        MetricResult::model("context_precision", 33.0, 0.9, "answer contradicts context 1"),
        MetricResult::model("context_recall", 0.0, 0.9, "context says 14 days, answer says 30"),
    ];
    let report = score_rag(metrics, vec![], true, 0, 0.0);
    assert!(
        report.rag_quality <= 49.0,
        "a contradicted claim is a critical failure and must force red, got {}",
        report.rag_quality
    );
    assert_eq!(report.band, Band::Red);
}

#[test]
fn find_span_locates_unsupported_claim() {
    let answer = "You have 30 days from delivery to return an item.";
    let span = find_span(answer, "30 days").unwrap();
    assert_eq!(&answer[span.0..span.1], "30 days");
    let _span = ClaimSpan {
        start: span.0,
        end: span.1,
        text: "30 days".into(),
        supported: false,
        contradicted: true,
    };
}

// ---------------------------------------------------------------------------
// Edge cases (§6.4): empty trace, single step, all-repeats, missing/invalid args
// ---------------------------------------------------------------------------

#[test]
fn empty_trace_does_not_panic() {
    let trace = AgentTrace {
        goal: "nothing".into(),
        steps: vec![],
        expected_tools: None,
        tool_schemas: None,
    };
    let report = score_agent(&trace, &[], 0, 0.0);
    assert_eq!(report.steps.len(), 0);
    assert_eq!(report.run_quality, 0.0);
}

#[test]
fn no_expected_tools_makes_tool_metric_na() {
    let trace = AgentTrace {
        goal: "book".into(),
        steps: vec![AgentStep {
            kind: "tool_call".into(),
            name: Some("search".into()),
            args: Some(json!({"q": "x"})),
            result: None,
            content: None,
        }],
        expected_tools: None,
        tool_schemas: None,
    };
    let tc = tool_call_correctness(&trace);
    assert!(tc.na);
    assert_eq!(tc.source, MetricSource::Deterministic);
}

#[test]
fn all_repeats_increments_repeat_count() {
    let step = |args| AgentStep {
        kind: "tool_call".into(),
        name: Some("search".into()),
        args: Some(args),
        result: None,
        content: None,
    };
    let trace = AgentTrace {
        goal: "loop".into(),
        steps: vec![
            step(json!({"a": 1})),
            step(json!({"a": 1})),
            step(json!({"a": 1})),
        ],
        expected_tools: None,
        tool_schemas: None,
    };
    let loops: Vec<_> = loop_free_metrics(&trace).into_iter().flatten().collect();
    assert_eq!(loops[0].raw_value, Some(0.0)); // first: no prior
    assert_eq!(loops[1].raw_value, Some(1.0)); // one prior identical
    assert_eq!(loops[2].raw_value, Some(2.0)); // two prior identical
    assert_eq!(loops[2].score, 50.0); // 100 - 2*25
}

#[test]
fn invalid_args_force_fail_and_penalty() {
    // party_size should be an integer; pass a string to violate the schema.
    let schema = json!({
        "type": "object",
        "properties": { "party_size": { "type": "integer" } },
        "required": ["party_size"]
    });
    let mut schemas = std::collections::HashMap::new();
    schemas.insert("search".to_string(), schema);

    let trace = AgentTrace {
        goal: "book".into(),
        steps: vec![AgentStep {
            kind: "tool_call".into(),
            name: Some("search".into()),
            args: Some(json!({ "party_size": "two" })),
            result: None,
            content: None,
        }],
        expected_tools: Some(vec!["search".into()]),
        tool_schemas: Some(schemas),
    };
    let tc = tool_call_correctness(&trace);
    assert_eq!(tc.pass, Some(false), "invalid args must fail");
    assert!(tc.reason.is_some(), "should explain the violation");
    // F1 is 1.0 (search called == expected) → 100, minus 20 penalty = 80.
    assert_eq!(tc.score, 80.0);
}

#[test]
fn valid_args_pass_schema() {
    let schema = json!({
        "type": "object",
        "properties": { "party_size": { "type": "integer" } },
        "required": ["party_size"]
    });
    let mut schemas = std::collections::HashMap::new();
    schemas.insert("search".to_string(), schema);

    let trace = AgentTrace {
        goal: "book".into(),
        steps: vec![AgentStep {
            kind: "tool_call".into(),
            name: Some("search".into()),
            args: Some(json!({ "party_size": 2 })),
            result: None,
            content: None,
        }],
        expected_tools: Some(vec!["search".into()]),
        tool_schemas: Some(schemas),
    };
    let tc = tool_call_correctness(&trace);
    assert_eq!(tc.pass, Some(true));
    assert_eq!(tc.score, 100.0);
}

#[test]
fn tool_calls_extraction_skips_llm_steps() {
    let trace = load_trace("clean.json");
    let calls = tool_calls(&trace);
    assert_eq!(calls.len(), 3, "clean has 3 tool calls among 5 steps");
    assert_eq!(calls[0].name, "search_restaurants");
}
