//! Text classification with OpenAI's Decisions API.
//!
//! Choice questions only; the endpoint does not support the Batch API. Requests
//! are sent verbatim, with no added prompt or reordered choices.

use std::{collections::HashSet, num::NonZeroUsize, path::PathBuf, sync::Mutex, time::Duration};

use lru::LruCache;
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;

use crate::retry::{is_transient_api_error, RetryKind, LIVE_REQUEST_ATTEMPTS};
use crate::utils::{api_key, ApiKeyError};

/// A choice value. Boolean `true` and the string `"true"` are distinct.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChoiceValue {
    /// A text label.
    String(String),
    /// A boolean label.
    Bool(bool),
}

impl From<&str> for ChoiceValue {
    fn from(value: &str) -> Self {
        Self::String(value.into())
    }
}
impl From<String> for ChoiceValue {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}
impl From<bool> for ChoiceValue {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

/// One of the finite alternatives offered to the model.
#[derive(Debug, Clone, Serialize)]
pub struct ChoiceOption {
    /// The value returned if this option is selected.
    pub value: ChoiceValue,
    /// Optional guidance about this option.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

impl ChoiceOption {
    /// Create an option without a description.
    pub fn new(value: impl Into<ChoiceValue>) -> Self {
        Self {
            value: value.into(),
            description: None,
        }
    }
}

/// A question with 2–255 distinct choices, in caller-supplied order.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename = "choice")]
pub struct ChoiceQuestion {
    /// Optional identifier echoed by the answer.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// What the model should decide about the input.
    pub instructions: String,
    /// Available answers; their order is preserved on the wire and in the cache key.
    pub choices: Vec<ChoiceOption>,
}

/// Probability assigned to an option.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChoiceProbability {
    /// The option's typed value.
    pub value: ChoiceValue,
    /// Probability reported by the model.
    pub probability: f64,
}

/// A successful choice answer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChoiceAnswer {
    /// The question's name, or null for an unnamed question.
    pub name: Option<String>,
    /// Selected option.
    pub choice: ChoiceValue,
    /// Model confidence, not necessarily the selected option's probability.
    pub confidence: f64,
    /// Distribution over the supplied options.
    pub probabilities: Vec<ChoiceProbability>,
}

/// Each question can independently receive an answer or a refusal.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DecisionAnswer {
    /// The model selected an option.
    Choice(ChoiceAnswer),
    /// The model declined this question; the API provides no refusal message.
    Refusal {
        /// The question's name, or null for an unnamed question.
        name: Option<String>,
    },
}

/// Token accounting returned by Decisions (including provider prompt-cache usage).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    /// Total input tokens.
    pub input_tokens: u64,
    /// Input cache accounting.
    pub input_tokens_details: InputTokensDetails,
    /// Total output tokens.
    pub output_tokens: u64,
    /// Output reasoning accounting.
    pub output_tokens_details: OutputTokensDetails,
    /// Total tokens.
    pub total_tokens: u64,
}

/// Provider prompt-cache accounting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputTokensDetails {
    /// Tokens read from the provider cache.
    pub cached_tokens: u64,
    /// Tokens written to the provider cache.
    pub cache_write_tokens: u64,
}

/// Reasoning-token accounting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputTokensDetails {
    /// Reasoning tokens consumed.
    pub reasoning_tokens: u64,
}

/// Answers in the same order as the questions, including any refusals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionResponse {
    /// Model that produced the response.
    pub model: String,
    /// One result per question.
    pub answers: Vec<DecisionAnswer>,
    /// Original API usage (also preserved on local cache hits).
    pub usage: Usage,
}

/// Request, transport, response, or persistent-cache failure.
#[derive(Debug, thiserror::Error)]
pub enum DecisionsError {
    /// Invalid questions detected before sending a request.
    #[error("Invalid Decisions request: {0}")]
    InvalidRequest(String),
    /// HTTP transport failure.
    #[error("Decisions request failed: {0}")]
    Request(#[from] reqwest::Error),
    /// Non-success HTTP response, with the original body for diagnosis.
    #[error("Decisions HTTP {status}: {body}")]
    Http {
        /// HTTP status.
        status: reqwest::StatusCode,
        /// Raw response body.
        body: String,
    },
    /// The response was not valid Decisions JSON.
    #[error("Invalid Decisions JSON: {0}")]
    Json(#[from] serde_json::Error),
    /// The response does not match the submitted questions.
    #[error("Invalid Decisions response: {0}")]
    InvalidResponse(String),
    /// Persistent cache I/O or compression failure.
    #[error("Decisions cache error: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Serialize)]
struct DecisionRequest<'a> {
    model: &'a str,
    input: &'a str,
    questions: &'a [ChoiceQuestion],
}

fn cache_key(body: &str, url: &url::Url) -> String {
    let hash = xxhash_rust::xxh3::xxh3_128(format!("{body} /:/ {url}").as_bytes());
    format!("tysm-v1-decisions_request-{hash:032x}.zstd")
}

/// Pooled, concurrency-limited client with optional persistent exact-response caching.
pub struct DecisionsClient {
    api_key: String,
    model: String,
    base_url: url::Url,
    http_client: reqwest::Client,
    semaphore: Semaphore,
    cache_directory: Option<PathBuf>,
    lru: Mutex<LruCache<String, DecisionResponse>>,
}

impl DecisionsClient {
    /// Create a client with a maximum of 16 simultaneous HTTP requests.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.into(),
            base_url: "https://api.openai.com/v1/".parse().unwrap(),
            http_client: crate::utils::pooled_client(),
            semaphore: Semaphore::new(16),
            cache_directory: None,
            lru: Mutex::new(LruCache::new(NonZeroUsize::new(1024).unwrap())),
        }
    }

    /// Read `OPENAI_API_KEY`, also loading `.env` when that feature is enabled.
    pub fn from_env(model: impl Into<String>) -> Result<Self, ApiKeyError> {
        Ok(Self::new(api_key()?, model))
    }

    /// Set the API base URL (for example `https://api.openai.com/v1/`).
    /// Panics if the URL is invalid.
    pub fn with_url(mut self, url: impl Into<String>) -> Self {
        let url = url.into();
        self.base_url = format!("{}/", url.trim_end_matches('/')).parse().unwrap();
        self
    }

    /// Set the maximum simultaneous HTTP requests. Panics if zero.
    pub fn with_max_concurrent_requests(mut self, max: usize) -> Self {
        assert!(max > 0, "Maximum concurrency must be positive");
        self.semaphore = Semaphore::new(max);
        self
    }

    /// Persist responses in an osmo store, shared with fresh clients using this directory.
    pub fn with_cache_directory(mut self, directory: impl Into<PathBuf>) -> Self {
        self.cache_directory = Some(directory.into());
        self
    }

    /// Ask choice questions about unchanged text input.
    ///
    /// Refusals are returned explicitly alongside choice answers and cached like
    /// other valid results. HTTP and invalid responses are not cached.
    /// Transport and transient API failures get up to four total attempts, honoring
    /// integer-seconds `Retry-After` headers. Backoff does not hold a concurrency slot.
    /// Concurrent identical cache misses are not deduplicated. No Batch API
    /// submissions are performed.
    pub async fn decide(
        &self,
        context: &str,
        questions: &[ChoiceQuestion],
    ) -> Result<DecisionResponse, DecisionsError> {
        if questions.is_empty() {
            return Err(DecisionsError::InvalidRequest(
                "At least one question is required".into(),
            ));
        }
        for question in questions {
            let unique: HashSet<_> = question.choices.iter().map(|c| &c.value).collect();
            if !(2..=255).contains(&question.choices.len())
                || unique.len() != question.choices.len()
            {
                return Err(DecisionsError::InvalidRequest(
                    "Each question needs 2–255 unique choices".into(),
                ));
            }
        }
        let body = serde_json::to_string(&DecisionRequest {
            model: &self.model,
            input: context,
            questions,
        })?;
        let url = self.base_url.join("decisions").unwrap();
        let key = cache_key(&body, &url);
        if let Some(response) = self.lru.lock().unwrap().get(&key).cloned() {
            return Ok(response);
        }
        if let Some(directory) = &self.cache_directory {
            if let Some(compressed) = crate::cache::read_from_cache_dir(directory, &key).await {
                let response = serde_json::from_slice(&zstd::decode_all(compressed.as_slice())?)?;
                validate(&response, questions)?;
                self.lru.lock().unwrap().put(key, response.clone());
                return Ok(response);
            }
        }
        let text = self.request_with_retries(&url, &body).await?;
        let response: DecisionResponse = serde_json::from_str(&text)?;
        validate(&response, questions)?;
        if let Some(directory) = &self.cache_directory {
            let compressed = zstd::encode_all(text.as_bytes(), 3)?;
            crate::cache::write_to_cache_dir(directory, &key, &compressed).await?;
        }
        self.lru.lock().unwrap().put(key, response.clone());
        Ok(response)
    }

    async fn request_with_retries(
        &self,
        url: &url::Url,
        body: &str,
    ) -> Result<String, DecisionsError> {
        let mut attempt = 1;
        loop {
            let (error, delay) = match self.request_once(url, body).await {
                Ok((status, _, text)) if status.is_success() => return Ok(text),
                Ok((status, retry_after, text)) => {
                    #[derive(Deserialize)]
                    struct ApiError {
                        error: crate::OpenAiError,
                    }
                    let transient = status == reqwest::StatusCode::TOO_MANY_REQUESTS
                        || status.is_server_error()
                        || serde_json::from_str::<ApiError>(&text)
                            .is_ok_and(|e| is_transient_api_error(&e.error));
                    let delay = transient.then(|| RetryKind::Capacity.delay(attempt, retry_after));
                    (DecisionsError::Http { status, body: text }, delay)
                }
                Err(error) => (
                    DecisionsError::Request(error),
                    Some(RetryKind::Transport.delay(attempt, None)),
                ),
            };
            match delay.filter(|_| attempt < LIVE_REQUEST_ATTEMPTS) {
                Some(delay) => {
                    log::warn!("Decisions request to {} failed on attempt {attempt}/{LIVE_REQUEST_ATTEMPTS} ({error}), retrying in {}s", self.model, delay.as_secs());
                    tokio::time::sleep(delay).await;
                    attempt += 1;
                }
                None => return Err(error),
            }
        }
    }

    // The permit lives only for this HTTP exchange, never for retry backoff.
    async fn request_once(
        &self,
        url: &url::Url,
        body: &str,
    ) -> Result<(reqwest::StatusCode, Option<Duration>, String), reqwest::Error> {
        let _permit = self.semaphore.acquire().await.unwrap();
        let response = self
            .http_client
            .post(url.clone())
            .bearer_auth(&self.api_key)
            .header("Content-Type", "application/json")
            .body(body.to_owned())
            .send()
            .await?;
        let status = response.status();
        let retry_after = response
            .headers()
            .get(reqwest::header::RETRY_AFTER)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse().ok())
            .map(Duration::from_secs);
        Ok((status, retry_after, response.text().await?))
    }
}

fn validate(
    response: &DecisionResponse,
    questions: &[ChoiceQuestion],
) -> Result<(), DecisionsError> {
    let invalid = || {
        DecisionsError::InvalidResponse(
            "Answers must match question count, names, and choices".into(),
        )
    };
    if response.answers.len() != questions.len() {
        return Err(invalid());
    }
    for (answer, question) in response.answers.iter().zip(questions) {
        let name = match answer {
            DecisionAnswer::Choice(answer) => {
                if !question.choices.iter().any(|c| c.value == answer.choice) {
                    return Err(invalid());
                }
                &answer.name
            }
            DecisionAnswer::Refusal { name } => name,
        };
        if name != &question.name {
            return Err(invalid());
        }
    }
    Ok(())
}
