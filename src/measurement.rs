//! Cache-only request observation, separate from billed spend.
//!
//! Once enabled, chat clients cannot submit live or batch requests, even if
//! their caller forgot to set cached-only. Cached outputs are never fabricated.
use crate::chat_completions::{ChatRequest, ChatUsage};
use serde::Serialize;
use std::io::Write;
use std::sync::{Mutex, OnceLock};

struct Measurement {
    file: std::fs::File,
    stage: String,
}
static MEASUREMENT: OnceLock<Mutex<Measurement>> = OnceLock::new();
static MISSES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Number of independent uncached requests observed so far (not billed calls).
pub fn misses() -> u64 {
    MISSES.load(std::sync::atomic::Ordering::Relaxed)
}

/// Enable process-wide cache measurement, writing a new JSONL file.
pub fn start(path: &std::path::Path) -> std::io::Result<()> {
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    MEASUREMENT
        .set(Mutex::new(Measurement {
            file,
            stage: "unattributed".into(),
        }))
        .map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                "measurement already enabled",
            )
        })
}
/// True once measurement has been enabled; it cannot be disabled mid-run.
pub fn enabled() -> bool {
    MEASUREMENT.get().is_some()
}
/// Set the sequential runner's stage label before constructing its requests.
/// Do not run unrelated stages concurrently while using this process label.
pub fn stage(label: &str) {
    if let Some(measurement) = MEASUREMENT.get() {
        measurement.lock().unwrap().stage = label.into();
    }
}

#[derive(Serialize)]
struct Observation<'a> {
    stage: &'a str,
    model: &'a str,
    requested_service_tier: Option<&'a str>,
    planned_transport: &'a str,
    schema: &'a crate::chat_completions::ResponseFormat,
    cache_key: &'a str,
    cached_model: Option<&'a str>,
    cached_usage: Option<ChatUsage>,
    input_estimate_tokens: usize,
    estimate_method: &'static str,
}

pub(crate) fn observe(
    request: &ChatRequest,
    cache_key: &str,
    cached: Option<(&str, ChatUsage)>,
    planned_transport: &str,
) {
    let Some(measurement) = MEASUREMENT.get() else {
        return;
    };
    if cached.is_none() {
        MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    let mut measurement = measurement.lock().unwrap();
    let (input_estimate_tokens, estimate_method) =
        if let Some((_, usage)) = cached {
            (
                usage.prompt_tokens as usize,
                "actual_cached_response_prompt_usage",
            )
        } else {
            let encoded = serde_json::to_string(request).unwrap();
            match tiktoken_rs::bpe_for_model(&request.model) {
            Ok(tokenizer) => (
                tokenizer.encode_ordinary(&encoded).len(),
                "model_tokenizer_on_serialized_request; JSON framing approximate",
            ),
            Err(_) => (
                tiktoken_rs::o200k_base_singleton().encode_ordinary(&encoded).len(),
                "o200k_base_fallback_not_confirmed_for_model; serialized JSON framing approximate",
            ),
        }
        };
    let mut row = serde_json::to_vec(&Observation {
        stage: &measurement.stage,
        model: &request.model,
        requested_service_tier: request.service_tier.as_deref(),
        planned_transport,
        schema: &request.response_format,
        cache_key,
        cached_model: cached.map(|(model, _)| model),
        cached_usage: cached.map(|(_, usage)| usage),
        input_estimate_tokens,
        estimate_method,
    })
    .unwrap();
    row.push(b'\n');
    measurement
        .file
        .write_all(&row)
        .expect("write cache measurement");
}
