//! Judix CLI — score an agent trace from the terminal with the **deterministic engine
//! only** (no model, no key, $0): tool-call correctness (F1), loop detection, and schema
//! validation, exactly as the server computes them before any model call.
//!
//! Judix does NOT watch your agents on its own — nothing can; a scorer has to be handed
//! each turn. So the CLI has two jobs:
//!   * `judix` / `judix demo`      -> stream a real run scoring live in the terminal, so a
//!                                    fresh install can *see* per-turn scoring with no setup.
//!   * `judix <trace>` / `| judix` -> score a trace you hand it (file, `-`, or a pipe).
//!
//! RAG scoring is model-powered (claim decomposition + grounding) and has no deterministic
//! component, so it lives on the server; the CLI detects a RAG triple and says so.

use judix_core::scoring::score_agent;
use judix_core::types::{AgentReport, AgentTrace, Band};
use serde_json::Value;
use std::io::{IsTerminal, Read};
use std::time::Duration;

/// A real trace, embedded at compile time so `judix demo` needs no repo checkout.
const DEMO_TRACE: &str = include_str!("../../../demos/wrong_tool.json");

const HELP: &str = "\
judix — score an agent trace with the deterministic engine (no model, no key, $0).

usage:
  judix                     watch a demo run score live in your terminal (no setup)
  judix demo                same as above, explicitly
  judix <trace.json>        score a trace file
  judix -                   score a trace read from stdin
  <producer> | judix        score a trace piped in

score a real trace with no local files:
  curl -s https://judix-8piu.onrender.com/demo/wrong_tool | judix

a trace is JSON shaped like { \"goal\": \"...\", \"steps\": [ ... ] }. To score every turn of
YOUR agent live, wire the one-line SDK hook into it (see the README). RAG triples
({question, contexts, answer}) are model-scored on the server.";

fn read_stdin() -> String {
    let mut buf = String::new();
    if let Err(e) = std::io::stdin().read_to_string(&mut buf) {
        eprintln!("error reading stdin: {e}");
        std::process::exit(1);
    }
    buf
}

fn main() {
    let arg = std::env::args().nth(1);
    let (raw, source) = match arg.as_deref() {
        Some("-h") | Some("--help") => {
            println!("{HELP}");
            return;
        }
        // Bare `judix demo`, or a bare `judix` typed at a prompt, shows scoring happening
        // instead of an error — the answer to "I installed it, now what do I look at?".
        Some("demo") => {
            run_demo();
            return;
        }
        Some("-") => (read_stdin(), "-".to_string()),
        Some(path) => match std::fs::read_to_string(path) {
            Ok(r) => (r, path.to_string()),
            Err(e) => {
                eprintln!("error reading {path}: {e}\n\n{HELP}");
                std::process::exit(1);
            }
        },
        None => {
            if std::io::stdin().is_terminal() {
                run_demo();
                return;
            }
            (read_stdin(), "-".to_string())
        }
    };

    // Empty input (e.g. `judix < /dev/null`) should teach, not throw a cryptic JSON error.
    if raw.trim().is_empty() {
        eprintln!("no trace to score.\n\n{HELP}");
        std::process::exit(2);
    }

    // Parse loosely first so we can branch on shape and give a useful message, instead of
    // deserializing straight into AgentTrace and dying on "missing field `goal`".
    let value: Value = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: input ({source}) is not valid JSON: {e}");
            std::process::exit(1);
        }
    };

    // A RAG triple has no deterministic metrics; point the user at the server.
    let is_rag = value.get("answer").is_some() && value.get("contexts").is_some()
        || value.get("type").and_then(Value::as_str) == Some("rag");
    if is_rag {
        let data_ref = if source == "-" { "@-".to_string() } else { format!("@{source}") };
        eprintln!(
            "This looks like a RAG triple. RAG faithfulness is model-powered and has no\n\
             deterministic score, so it runs on the server, not the CLI:\n\
             \n\
             \x20   curl -X POST https://judix-8piu.onrender.com/score/rag \\\n\
             \x20        -H 'content-type: application/json' --data-binary {data_ref}\n"
        );
        std::process::exit(2);
    }

    let trace: AgentTrace = match serde_json::from_value(value) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: expected an agent trace (an object with `goal` and `steps`): {e}");
            std::process::exit(1);
        }
    };

    let report = score_agent(&trace, &[], 0, 0.0);
    match serde_json::to_string_pretty(&report) {
        Ok(json) => println!("{json}"),
        Err(e) => {
            eprintln!("error serializing report: {e}");
            std::process::exit(1);
        }
    }
}

fn band_ansi(b: &Band, color: bool) -> &'static str {
    if !color {
        return "";
    }
    match b {
        Band::Green => "\x1b[32m",
        Band::Amber => "\x1b[33m",
        Band::Red => "\x1b[31m",
    }
}

fn band_word(b: &Band) -> &'static str {
    match b {
        Band::Green => "GREEN",
        Band::Amber => "AMBER",
        Band::Red => "RED",
    }
}

/// Stream the embedded run through the deterministic engine turn by turn, so the user
/// *watches* per-turn scoring happen — the "live context scoring" the site promises,
/// with zero pasting and zero wiring.
fn run_demo() {
    let tty = std::io::stdout().is_terminal();
    let color = tty && std::env::var_os("NO_COLOR").is_none();
    let (bold, dim, reset) = if color {
        ("\x1b[1m", "\x1b[2m", "\x1b[0m")
    } else {
        ("", "", "")
    };
    let sleep = |ms: u64| {
        if tty {
            std::thread::sleep(Duration::from_millis(ms));
        }
    };

    let trace: AgentTrace =
        serde_json::from_str(DEMO_TRACE).expect("embedded demo trace must be valid");
    let report: AgentReport = score_agent(&trace, &[], 0, 0.0);

    println!();
    println!("  {bold}judix{reset} — live per-turn scoring   {dim}deterministic engine · no model · $0{reset}");
    println!("  {dim}demo run: wrong_tool{reset}");
    println!("  {dim}goal:{reset} {}", trace.goal);
    println!();

    for step in &report.steps {
        sleep(340);
        let col = band_ansi(&step.band, color);
        let label = &step.label;
        if step.na {
            // No deterministic metrics on this step (a model-only step, e.g. plan/reply).
            println!("  {dim}[{:02}] {label:<22}   —   needs a model key for this turn{reset}", step.index);
            continue;
        }
        let f1 = step
            .metrics
            .iter()
            .find(|m| m.name == "tool_call_correctness")
            .and_then(|m| m.raw_value);
        let loopv = step
            .metrics
            .iter()
            .find(|m| m.name == "loop_free")
            .map(|m| m.score.round() as i32);
        let mut extra = String::new();
        if let Some(f1) = f1 {
            extra.push_str(&format!("F1 {f1:.2}"));
        }
        if let Some(l) = loopv {
            if !extra.is_empty() {
                extra.push_str("  ·  ");
            }
            extra.push_str(&format!("loop {l}"));
        }
        let sq = step.step_quality.round() as i32;
        let band = band_word(&step.band);
        println!("  {dim}[{:02}]{reset} {label:<22} {col}{sq:>3}  {band:<5}{reset}  {dim}{extra}{reset}", step.index);
    }

    sleep(360);
    println!();
    println!("  {dim}{}{reset}", "─".repeat(52));
    let rc = band_ansi(&report.band, color);
    let rq = report.run_quality.round() as i32;
    println!(
        "  run quality  {rc}{bold}{rq}{reset}  {rc}{}{reset}   {dim}{:.0}% scored with no model call{reset}",
        band_word(&report.band),
        report.deterministic_share * 100.0
    );
    println!();
    println!("  {dim}This is the deterministic half — the loop and the wrong tool call are caught");
    println!("  with zero model calls. Add a model key and the intent scoring (relevance, goal");
    println!("  drift) pulls this same run to red, the number you see on the site.{reset}");
    println!();
    println!("  {bold}now:{reset}");
    println!("    score your own trace     {dim}judix your_trace.json    (or  … | judix){reset}");
    println!("    score a hosted demo      {dim}curl -s https://judix-8piu.onrender.com/demo/wrong_tool | judix{reset}");
    println!("    score YOUR agent live    {dim}wire the one-line SDK hook into it — see the README{reset}");
    println!("    add the model's \"why\"    {dim}set JUDIX_API_KEY to any OpenAI-compatible provider{reset}");
    println!();
}
