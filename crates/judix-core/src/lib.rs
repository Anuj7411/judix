//! # judix-core
//!
//! The deterministic evaluation engine at the heart of Judix. A pure-Rust engine
//! computes real numeric scores for agent trajectories and RAG triples with **zero
//! LLM calls**; an AI model (in the optional `model` feature) is used only to
//! *explain* results and decompose RAG claims.
//!
//! Build order (blueprint §11): the `deterministic` + `scoring` modules are the
//! hero and are fully unit-tested before any model integration.

pub mod deterministic;
pub mod scoring;
pub mod types;

#[cfg(feature = "model")]
pub mod cache;
#[cfg(feature = "model")]
pub mod model;

pub use types::{
    AgentReport, AgentStep, AgentTrace, Band, ClaimSpan, MetricResult, MetricSource, RagReport,
    RagTriple, StepScore,
};
