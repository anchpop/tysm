//! Chat completions are the most common way to interact with the OpenAI API.
//! This module provides a client for interacting with the ChatGPT API.
//!
//! It also provides a batch API for processing large numbers of requests asynchronously.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::RwLock;

use dashmap::DashMap;
use reqwest::Client;
use schemars::{schema_for, transform::Transform, JsonSchema, Schema};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::Semaphore;
use xxhash_rust::const_xxh3::xxh3_64 as const_xxh3;

use crate::batch::{Batch, BatchClient, BatchRequestItem, BatchStatus};
use crate::schema::OpenAiTransform;
use crate::utils::{api_key, OpenAiApiKeyError};
use crate::OpenAiError;
use log::{debug, info, warn};

/// To use this library, you need to create a [`ChatClient`]. This contains various information needed to interact with the ChatGPT API,
/// such as the API key, the model to use, and the URL of the API.
///
/// ```rust
/// # use tysm::chat_completions::ChatClient;
/// // Create a client with your API key and model
/// let client = ChatClient::new("sk-1234567890", "gpt-4o");
/// ```
///
/// ```rust
/// # use tysm::chat_completions::ChatClient;
/// // Create a client using an API key stored in an `OPENAI_API_KEY` environment variable.
/// // (This will also look for an `.env` file in the current directory.)
/// let client = ChatClient::from_env("gpt-4o").unwrap();
/// ```
#[non_exhaustive]
pub struct ChatClient {
    /// The API key to use for the ChatGPT API.
    pub api_key: String,
    /// The URL of the ChatGPT API. Customize this if you are using a custom API that is compatible with OpenAI's.
    pub base_url: url::Url,
    /// The subpath to the chat-completions endpoint. By default, this is `chat/completions`.
    pub chat_completions_path: String,
    /// The model to use for the ChatGPT API.
    pub model: String,
    /// A cache of recent responses.
    pub lru: DashMap<String, String>,
    /// This client's token consumption (as reported by the API). Batch API requests are tracked separately in `batch_usage`.
    pub usage: RwLock<ChatUsage>,
    /// This client's token consumption via the Batch API (as reported by the API). Tracked
    /// separately from `usage` because OpenAI bills batch requests at 50% of the standard price.
    pub batch_usage: RwLock<ChatUsage>,
    /// Dollars billed so far, accumulated one request at a time.
    ///
    /// Kept beside the token counters rather than derived from them because a
    /// model's rate can depend on a single request's prompt size (see
    /// [`crate::model_prices::LongContext`]) — pricing the summed tokens of
    /// many short requests would bill them all at the long-context premium.
    ///
    /// `None` once any request used a model we have no price for, since the
    /// total is unknowable from then on.
    pub spend: RwLock<Option<f64>>,
    /// The directory in which to cache responses to requests
    pub cache_directory: Option<PathBuf>,

    /// The backup cache directory to check if a file is not found in the main cache directory
    pub backup_cache_directory: Option<PathBuf>,

    /// The service tier to use for requests (e.g., "flex")
    pub service_tier: Option<String>,

    /// The prompt cache key to send with requests. A routing hint for provider-side
    /// prompt caching (OpenAI `prompt_cache_key`): requests sharing a key are routed to
    /// the same cache, making prefix hits reliable. On GPT-5.6+ setting it is required
    /// for dependable cache matching. Does not affect response content.
    pub prompt_cache_key: Option<String>,

    /// The reasoning effort to use for requests (e.g., "low", "medium", "high")
    pub reasoning_effort: Option<String>,

    /// Extra body to be provided when making requests
    pub extra_body: Option<serde_json::Value>,

    /// How few uncached requests a batch may contain before it is sent as
    /// ordinary live calls instead.
    ///
    /// The Batch API halves the price but pays a fixed cost in latency —
    /// uploading the request file and waiting in the queue — which is worth it
    /// for a thousand requests and absurd for five. Retries feel this most: once
    /// most responses are cached, what remains is a handful of stragglers that
    /// would otherwise wait on a whole batch round trip.
    ///
    /// Zero (the default) always batches; `usize::MAX` never submits a new
    /// batch (see [`ChatClient::with_no_batch`]).
    pub small_batch_threshold: usize,

    /// Semaphore to limit the maximum number of concurrent requests
    pub semaphore: Semaphore,

    /// Shared HTTP client with connection pooling
    pub http_client: Client,

    /// If true, all uncached requests will fail with [`ChatError::CacheMiss`] (or, in a
    /// batch, a per-item [`IndividualChatError::CacheMiss`]) instead of
    /// hitting the API. Useful for testing or offline usage.
    pub cached_only: bool,

    /// Clients whose caches are checked in order after this client's cache. These clients are
    /// never allowed to make API requests through this client. Configure them newest-to-oldest.
    cache_fallbacks: Vec<ChatClient>,
}

/// The role of a message.
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub enum Role {
    /// The user is sending the message.
    #[serde(rename = "user")]
    User,
    /// The assistant is sending the message.
    #[serde(rename = "assistant")]
    Assistant,
    /// The system is sending the message.
    #[serde(rename = "system")]
    System,
}

/// A message to send to the ChatGPT API.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ChatMessage {
    /// The role of user sending the message.
    pub role: Role,

    /// The content of the message. It is a vector of [`ChatMessageContent`]s,
    /// which allows you to include images in the message.
    pub content: Vec<ChatMessageContent>,
}

impl ChatMessage {
    /// Create a new [`ChatMessage`].
    pub fn new(role: Role, content: Vec<ChatMessageContent>) -> Self {
        Self { role, content }
    }

    /// Create a new [`ChatMessage`] with the user role.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: vec![ChatMessageContent::Text {
                text: content.into(),
            }],
        }
    }

    /// Create a new [`ChatMessage`] with the assistant role.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: vec![ChatMessageContent::Text {
                text: content.into(),
            }],
        }
    }

    /// Create a new [`ChatMessage`] with the system role.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: vec![ChatMessageContent::Text {
                text: content.into(),
            }],
        }
    }
}

/// The content of a message.
///
/// Currently, only text and image URLs are supported.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatMessageContent {
    /// A textual message.
    Text {
        /// The text of the message.
        text: String,
    },
    /// An image URL.
    /// The image URL can also be a base64 encoded image.
    /// example:
    /// ```rust
    /// use tysm::chat_completions::{ChatMessageContent, ImageUrl};
    ///
    /// let base64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";
    /// let content = ChatMessageContent::ImageUrl {
    ///     image: ImageUrl {
    ///         url: format!("data:image/png;base64,{base64_image}"),
    ///     },
    /// };
    /// ```
    ImageUrl {
        /// The image URL.
        #[serde(rename = "image_url")]
        image: ImageUrl,
    },
    /// Audio input as base64-encoded data.
    /// ```rust,no_run
    /// use tysm::chat_completions::{ChatMessageContent, InputAudio};
    ///
    /// let audio_bytes = std::fs::read("audio.wav").unwrap();
    /// let content = ChatMessageContent::InputAudio {
    ///     input_audio: InputAudio::wav(audio_bytes),
    /// };
    /// ```
    InputAudio {
        /// The audio data.
        input_audio: InputAudio,
    },
}

/// An image URL. OpenAI will accept a link to an image, or a base64 encoded image.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ImageUrl {
    /// The image URL.
    pub url: String,
}

/// Base64-encoded audio input.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct InputAudio {
    /// Base64-encoded audio data.
    pub data: String,
    /// The audio format (e.g. "wav", "mp3").
    pub format: String,
}

impl InputAudio {
    /// Create an `InputAudio` from raw WAV bytes.
    pub fn wav(bytes: impl AsRef<[u8]>) -> Self {
        use base64::Engine;
        Self {
            data: base64::engine::general_purpose::STANDARD.encode(bytes),
            format: "wav".to_string(),
        }
    }

    /// Create an `InputAudio` from raw MP3 bytes.
    pub fn mp3(bytes: impl AsRef<[u8]>) -> Self {
        use base64::Engine;
        Self {
            data: base64::engine::general_purpose::STANDARD.encode(bytes),
            format: "mp3".to_string(),
        }
    }
}

/// A request to the ChatGPT API. You probably will not need to use this directly,
/// but it is public because it is still exposed in errors.
#[derive(Serialize, Clone, Debug)]
pub struct ChatRequest {
    /// The model to use for the ChatGPT API.
    pub model: String,
    /// The messages to send to the API.
    pub messages: Vec<ChatMessage>,
    /// The response format to use for the ChatGPT API.
    pub response_format: ResponseFormat,

    /// The service tier to use for the request (e.g., "flex")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,

    /// The prompt cache key for provider-side prompt-cache routing
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,

    /// The reasoning effort to use for the request (e.g., "low", "medium", "high")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,

    /// Extra fields to be included in the request body
    #[serde(flatten)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra_body: Option<serde_json::Value>,
}

impl ChatRequest {
    fn cache_key(&self) -> String {
        // Create a version of the request without service_tier or prompt_cache_key for
        // caching. Neither affects response content (one is a scheduling tier, the other
        // a provider-side cache-routing hint), so requests differing only in those fields
        // share the same cached response.
        // Note: reasoning_effort IS included in the cache key since it affects output
        let mut cacheable = self.clone();
        cacheable.service_tier = None;
        cacheable.prompt_cache_key = None;

        let serialized = serde_json::to_string(&cacheable).unwrap();
        let id = const_xxh3(serialized.as_bytes());
        format!("tysm-v1-chat_request-{}.zstd", id)
    }

    // LEGACY CACHE KEY MIGRATION (can be removed in a future version once caches have been migrated)
    //
    // Prior to v0.17.1, SchemaFormat had an explicit `additionalProperties: false` field
    // AND OpenAiTransform also inserted `additionalProperties` into every schema object
    // (including primitives). This produced duplicate keys in the serialized JSON, which
    // changed the cache key hash. This method reproduces the old serialization so we can
    // find and migrate cached responses.
    fn legacy_cache_key(&self) -> String {
        let mut cacheable = self.clone();
        cacheable.service_tier = None;
        cacheable.prompt_cache_key = None;

        let serialized = serde_json::to_string(&LegacyChatRequest::from(cacheable)).unwrap();
        let id = const_xxh3(serialized.as_bytes());
        format!("tysm-v1-chat_request-{}.zstd", id)
    }
}

// LEGACY TYPES FOR CACHE KEY MIGRATION (can be removed in a future version)
//
// These types reproduce the old serialization format where SchemaFormat had an
// explicit `additionalProperties: false` field that duplicated the one added by
// OpenAiTransform. They exist solely to compute legacy cache keys for migration.

#[derive(Serialize)]
struct LegacySchemaFormat {
    #[serde(rename = "additionalProperties")]
    additional_properties: bool,
    #[serde(flatten)]
    schema: Schema,
}

#[derive(Serialize)]
struct LegacyJsonSchemaFormat {
    name: String,
    strict: bool,
    schema: LegacySchemaFormat,
}

#[derive(Serialize)]
#[serde(tag = "type")]
enum LegacyResponseFormat {
    #[serde(rename = "json_schema")]
    JsonSchema { json_schema: LegacyJsonSchemaFormat },
    #[serde(rename = "json_object")]
    JsonObject,
    #[serde(rename = "text")]
    Text,
}

#[derive(Serialize)]
struct LegacyChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    response_format: LegacyResponseFormat,
    #[serde(skip_serializing_if = "Option::is_none")]
    service_tier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<String>,
    #[serde(flatten)]
    #[serde(skip_serializing_if = "Option::is_none")]
    extra_body: Option<serde_json::Value>,
}

impl From<ChatRequest> for LegacyChatRequest {
    fn from(req: ChatRequest) -> Self {
        let response_format = match req.response_format {
            ResponseFormat::JsonSchema { json_schema } => {
                // Re-apply the old transform that added additionalProperties to ALL
                // schema objects (including primitives like string/integer)
                let mut schema = json_schema.schema.schema;
                LegacyOpenAiTransform.transform(&mut schema);

                LegacyResponseFormat::JsonSchema {
                    json_schema: LegacyJsonSchemaFormat {
                        name: json_schema.name,
                        strict: json_schema.strict,
                        schema: LegacySchemaFormat {
                            additional_properties: false,
                            schema,
                        },
                    },
                }
            }
            ResponseFormat::JsonObject => LegacyResponseFormat::JsonObject,
            ResponseFormat::Text => LegacyResponseFormat::Text,
        };

        LegacyChatRequest {
            model: req.model,
            messages: req.messages,
            response_format,
            service_tier: req.service_tier,
            reasoning_effort: req.reasoning_effort,
            extra_body: req.extra_body,
        }
    }
}

/// The old OpenAiTransform added `additionalProperties: false` to ALL schema objects,
/// not just object-type ones. We need to reproduce that for legacy cache key computation.
struct LegacyOpenAiTransform;

impl schemars::transform::Transform for LegacyOpenAiTransform {
    fn transform(&mut self, schema: &mut Schema) {
        if let Some(obj) = schema.as_object_mut() {
            if obj.get("$ref").is_none() {
                obj.insert(
                    "additionalProperties".to_string(),
                    serde_json::Value::Bool(false),
                );
            }
        }
        schemars::transform::transform_subschemas(self, schema);
    }
}
// END LEGACY TYPES

/// An object specifying the format that the model must output.
/// `ResponseFormat::JsonSchema` enables Structured Outputs which ensures the model will match your supplied JSON schema
#[derive(Serialize, Debug, Clone)]
#[serde(tag = "type")]
pub enum ResponseFormat {
    /// The model is constrained to return a JSON object of the specified schema.
    #[serde(rename = "json_schema")]
    JsonSchema {
        /// The schema.
        /// Often generated with `JsonSchemaFormat::new()`.
        json_schema: JsonSchemaFormat,
    },

    /// The model is constrained to return a JSON object, but the schema is not enforced.
    #[serde(rename = "json_object")]
    JsonObject,

    /// The model is not constrained to any specific format.
    #[serde(rename = "text")]
    Text,
}

/// The format of a JSON schema.
#[derive(Serialize, Debug, Clone)]
pub struct JsonSchemaFormat {
    /// The name of the schema. It's not clear whether this is actually used anywhere by OpenAI.
    pub name: String,
    /// Whether the schema is strict. (For openai, you always want this to be true.)
    pub strict: bool,
    /// The schema.
    pub schema: SchemaFormat,
}

impl JsonSchemaFormat {
    /// Create a new `JsonSchemaFormat`.
    pub fn new<T: JsonSchema>() -> Self {
        let mut schema = schema_for!(T);
        let name = tynm::type_name::<T>();
        let name = if name.is_empty() {
            "response".to_string()
        } else {
            name
        };

        OpenAiTransform.transform(&mut schema);

        Self::from_schema(schema, &name)
    }

    /// Create a new `JsonSchemaFormat` from a `Schema`.
    pub fn from_schema(schema: Schema, ty_name: &str) -> Self {
        Self {
            name: ty_name.to_string(),
            strict: true,
            schema: SchemaFormat { schema },
        }
    }
}

/// A JSON schema format wrapper.
/// The `additionalProperties` constraint is handled by [`OpenAiTransform`](crate::schema::OpenAiTransform)
/// which adds it only to object-type schemas.
#[derive(Serialize, Debug, Clone)]
pub struct SchemaFormat {
    /// The schema.
    #[serde(flatten)]
    pub schema: Schema,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub(crate) struct ChatMessageResponse {
    pub role: Role,
    pub content: Option<String>,

    /// When using Structured Outputs with user-generated input, OpenAI models may occasionally refuse to fulfill the request for safety reasons. Since a refusal does not necessarily follow the schema supplied in response_format, the API response will include a new field called refusal to indicate that the model refused to fulfill the request.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub refusal: Option<String>,
}

impl ChatMessageResponse {
    fn content(self) -> Result<String, String> {
        if let Some(refusal) = self.refusal {
            if !refusal.trim().is_empty() {
                return Err(refusal);
            }
        }

        // if there's no refusal, we assume that there is content
        let content = self.content.unwrap();

        Ok(content)
    }
}

#[derive(Deserialize, Debug, Clone, Serialize)]
struct ChatResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    system_fingerprint: Option<String>,
    choices: Vec<ChatChoice>,
    usage: ChatUsage,
}

#[derive(Deserialize, Debug, Clone, Serialize)]
struct ChatChoice {
    index: u8,
    message: ChatMessageResponse,
    logprobs: Option<serde_json::Value>,
    finish_reason: String,
}

#[derive(Deserialize, Debug, Serialize)]
enum ChatResponseOrError {
    #[serde(rename = "error")]
    Error(OpenAiError),

    #[serde(untagged)]
    Response(ChatResponse),
}

/// The token consumption of the chat-completions API.
#[derive(Deserialize, Debug, Default, Clone, Copy, Eq, PartialEq, Serialize)]
pub struct ChatUsage {
    /// The number of tokens used for the prompt.
    pub prompt_tokens: u32,
    /// The number of tokens used for the completion.
    pub completion_tokens: u32,
    /// The total number of tokens used.
    pub total_tokens: u32,

    /// Details about the prompt tokens (such as whether they were cached).
    #[serde(default)]
    pub prompt_token_details: Option<PromptTokenDetails>,
    /// Details about the completion tokens for reasoning models
    #[serde(default)]
    pub completion_token_details: Option<CompletionTokenDetails>,
}

/// Includes details about the prompt tokens.
/// Currently, only contains the number of cached tokens.
#[derive(Deserialize, Debug, Default, Clone, Copy, Eq, PartialEq, Serialize)]
pub struct PromptTokenDetails {
    /// OpenAI automatically caches tokens that are used in a previous request.
    /// This reduces input cost.
    pub cached_tokens: u32,
}

/// Includes details about the completion tokens for reasoning models
#[derive(Deserialize, Debug, Default, Clone, Copy, Eq, PartialEq, Serialize)]
pub struct CompletionTokenDetails {
    /// The number of tokens used for reasoning.
    pub reasoning_tokens: u32,
    /// The number of accepted tokens from the reasoning model.
    pub accepted_prediction_tokens: u32,
    /// The number of rejected tokens from the reasoning model.
    /// (These tokens are still counted towards the cost of the request)
    pub rejected_prediction_tokens: u32,
}

impl std::ops::AddAssign for ChatUsage {
    fn add_assign(&mut self, rhs: Self) {
        self.prompt_tokens += rhs.prompt_tokens;
        self.completion_tokens += rhs.completion_tokens;
        self.total_tokens += rhs.total_tokens;

        self.prompt_token_details = match (self.prompt_token_details, rhs.prompt_token_details) {
            (Some(lhs), Some(rhs)) => Some(lhs + rhs),
            (None, Some(rhs)) => Some(rhs),
            (Some(lhs), None) => Some(lhs),
            (None, None) => None,
        };
        self.completion_token_details =
            match (self.completion_token_details, rhs.completion_token_details) {
                (Some(lhs), Some(rhs)) => Some(lhs + rhs),
                (None, Some(rhs)) => Some(rhs),
                (Some(lhs), None) => Some(lhs),
                (None, None) => None,
            };
    }
}

impl std::ops::Add for PromptTokenDetails {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            cached_tokens: self.cached_tokens + rhs.cached_tokens,
        }
    }
}

impl std::ops::Add for CompletionTokenDetails {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            reasoning_tokens: self.reasoning_tokens + rhs.reasoning_tokens,
            accepted_prediction_tokens: self.accepted_prediction_tokens
                + rhs.accepted_prediction_tokens,
            rejected_prediction_tokens: self.rejected_prediction_tokens
                + rhs.rejected_prediction_tokens,
        }
    }
}

/// How many times a live chat request is attempted before its error is returned.
const LIVE_REQUEST_ATTEMPTS: u32 = 4;

/// Whether an API error is one a later, identical request could succeed at.
///
/// The flex service tier answers `flex_unavailable` whenever OpenAI has no
/// spare capacity; rate limits and server faults are the same shape of
/// problem. In each case the request is fine and only the moment is wrong.
pub(crate) fn is_transient_api_error(error: &OpenAiError) -> bool {
    let code = error.code.as_deref().unwrap_or_default();
    matches!(
        code,
        "flex_unavailable" | "resource_unavailable" | "rate_limit_exceeded" | "server_error"
    ) || matches!(
        error.r#type.as_str(),
        "resource_unavailable" | "rate_limit_error" | "server_error"
    )
}

/// How long to wait before retrying `error`, or `None` when retrying it would
/// fail the same way however many times it is tried.
///
/// A transport error is usually a pooled keep-alive connection the server has
/// already closed, surfacing on its next use; reconnecting fixes it at once.
/// Capacity frees up on a scale of minutes rather than milliseconds, so those
/// errors wait far longer. Anything else - a malformed request, a schema the
/// model cannot satisfy - is returned as is.
fn retry_delay(error: &ChatError, attempt: u32) -> Option<std::time::Duration> {
    use std::time::Duration;
    match error {
        ChatError::RequestError(_) => Some(Duration::from_secs(1 << attempt)),
        ChatError::ApiError(error, _) if is_transient_api_error(error) => {
            Some(Duration::from_secs(15 << (attempt - 1)))
        }
        _ => None,
    }
}

/// Errors that can occur when interacting with the ChatGPT API.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum ChatError {
    /// An error occurred when sending the request to the API.
    #[error("Request error: {0}")]
    RequestError(#[from] reqwest::Error),

    /// An error occurred when serializing the request to JSON.
    #[error("JSON serialization error: {0}")]
    JsonSerializeError(serde_json::Error, ChatRequest),

    /// The API did not return a JSON object.
    #[error("API did not return a JSON object: {response} (request: {request})")]
    ApiDidNotReturnJson {
        /// The response from the API.
        response: String,
        /// The request that was sent to the API.
        request: String,
        /// The error that occurred when parsing the response.
        #[source]
        error: serde_json::Error,
    },

    /// The API returned a response could not be parsed into the structure expected of OpenAI responses
    #[error("API returned a response could not be parsed into the structure expected of OpenAI responses: `{response:#}` (request: `{request}`)")]
    ApiParseError {
        /// The response from the API.
        response: serde_json::Value,
        /// The error that occurred when parsing the response.
        #[source]
        error: serde_json::Error,
        /// The request that was sent to the API.
        request: String,
    },

    /// An error occurred when deserializing the response from the API.
    #[error("API returned an error response for request {1}")]
    ApiError(#[source] OpenAiError, String),

    /// The API returned a response that was not a valid JSON object.
    #[error("There was a problem with the API response")]
    ChatError(#[from] IndividualChatError),

    /// IO error (usually occurs when reading from the cache).
    #[error("IO error")]
    IoError(#[from] std::io::Error),

    /// The API did not return any choices.
    #[error("No choices returned from API")]
    NoChoices,

    /// The request was not found in the cache and the client is in cached-only mode.
    #[error("Cache miss: request not found in cache (cached_only mode is enabled)")]
    CacheMiss,
}

/// Errors that can occur when sending many chat requests via the batch API.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum BatchChatError {
    /// An error occurred when uploading the file to the API.
    #[error("Error uploading file")]
    FileUploadError(#[from] crate::files::FilesError),

    /// An error occurred when sending the request to the API.
    #[error("Error getting batch results")]
    GetBatchResultsError(#[from] crate::batch::GetBatchResultsError),

    /// An error occurred when creating the batch.
    #[error("Error creating batch")]
    CreateBatchError(#[from] crate::batch::CreateBatchError),

    /// An error occurred when waiting for the batch to complete.
    #[error("Error waiting for batch to complete")]
    WaitForBatchError(#[from] crate::batch::WaitForBatchError),

    /// Batch item error.
    #[error("Batch item error")]
    BatchItemError(#[from] crate::batch::BatchItemError),

    /// An error occurred when sending the request to the API.
    #[error("Chat completions error for request with custom id `{1}`")]
    OpenAiError(#[source] OpenAiError, String),

    /// A custom ID in the batch request was not found in the results.
    #[error("Custom ID `{0}` not found in results")]
    CustomIdNotFound(String),

    /// The batch has no choices.
    #[error("The result for Custom ID `{0}` has no choices")]
    BatchNoChoices(String),

    /// The API returned a response could not be parsed into the structure expected of OpenAI responses
    #[error("API returned a response could not be parsed into the structure expected of OpenAI responses: {response}")]
    ApiParseError {
        /// The error that occurred when parsing the response.
        #[source]
        error: serde_json::Error,
        /// The response from the API.
        response: String,
    },

    /// An error occurred when listing the batches.
    #[error("Error listing batches")]
    ListBatchesError(#[from] crate::batch::ListBatchesError),

    /// IO error while persisting a successful batch response in the local cache.
    #[error("IO error while writing batch response cache")]
    CacheIoError(#[from] std::io::Error),
}

/// Errors that can occur when sending many chat requests via the batch API.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum IndividualChatError {
    /// The API returned a response that did not conform to the given schema.
    #[error(
        "API returned a response that did not conform to the given schema: `{schema}` (response: `{response}`)"
    )]
    ResponseNotConformantToSchema {
        /// The error that occurred when deserializing the response.
        #[source]
        error: serde_json::Error,
        /// The response from the API.
        response: String,
        /// The schema that the response was supposed to conform to.
        schema: String,
    },

    /// The API refused to fulfill the request.
    #[error("The API refused to fulfill the request: `{0}`")]
    Refusal(String),

    /// The request failed outright.
    ///
    /// Only produced when a batch's requests are sent live (see the
    /// small-batch threshold or `with_no_batch`): a transport or API failure belongs
    /// to that one request, and reporting it per-item keeps the live path's
    /// result shape identical to the batch path's.
    #[error("The request failed: `{0}`")]
    Other(String),

    /// The request was not in any configured cache and the client is in
    /// cached-only mode (see [`ChatClient::with_cached_only`]).
    #[error("Cache miss: request not found in cache (cached_only mode is enabled)")]
    CacheMiss,
}

/// Live requests in flight at once when a batch is sent live instead.
const LIVE_CONCURRENCY: usize = 16;

/// Cache lookups in flight at once while a batch is checked against the cache.
const CACHE_LOOKUP_CONCURRENCY: usize = 32;

impl ChatClient {
    /// Create a new [`ChatClient`].
    /// If the API key is in the environment, you can use the [`Self::from_env`] method instead.
    ///
    /// ```rust
    /// use tysm::chat_completions::ChatClient;
    ///
    /// let client = ChatClient::new("sk-1234567890", "gpt-4o");
    /// ```
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: url::Url::parse("https://api.openai.com/v1/").unwrap(),
            chat_completions_path: "chat/completions".to_string(),
            model: model.into(),
            lru: DashMap::new(),
            usage: RwLock::new(ChatUsage::default()),
            batch_usage: RwLock::new(ChatUsage::default()),
            spend: RwLock::new(Some(0.0)),
            cache_directory: None,
            backup_cache_directory: None,
            service_tier: None,
            prompt_cache_key: None,
            reasoning_effort: None,
            extra_body: None,
            small_batch_threshold: 0,
            semaphore: Semaphore::new(100),
            http_client: crate::utils::pooled_client(),
            cached_only: false,
            cache_fallbacks: Vec::new(),
        }
    }

    /// How few uncached requests a batch may contain before it is sent as
    /// ordinary live calls instead. Zero (the default) always batches.
    pub fn with_small_batch_threshold(mut self, threshold: usize) -> Self {
        self.small_batch_threshold = threshold;
        self
    }

    /// Never submit a new batch: every `batch_*` call still harvests a
    /// matching existing batch (cancelling it if it is in flight), then sends
    /// the remaining uncached requests as ordinary live calls, concurrently,
    /// returning the same per-item results. For when the Batch API is
    /// unavailable — an org whose batches all fail validation, say — or a
    /// backlog of nearly finished batches should be cashed in now, and a run
    /// should finish at live prices rather than not at all.
    pub fn with_no_batch(self) -> Self {
        self.with_small_batch_threshold(usize::MAX)
    }

    /// Set the cache directory for the client.
    ///
    /// The cache directory will be used to persistently cache all responses to requests.
    pub fn with_cache_directory(mut self, cache_directory: impl Into<PathBuf>) -> Self {
        let cache_directory = cache_directory.into();

        if cache_directory.exists() && cache_directory.is_file() {
            panic!("Cache directory is a file");
        }

        self.cache_directory = Some(cache_directory);
        self
    }

    /// Set the backup cache directory for the client.
    ///
    /// If a cached file is not found in the main cache directory, the backup cache directory
    /// will be checked. If found there, the file will be moved to the main cache directory.
    pub fn with_backup_cache_directory(
        mut self,
        backup_cache_directory: impl Into<PathBuf>,
    ) -> Self {
        let backup_cache_directory = backup_cache_directory.into();

        if backup_cache_directory.exists() && backup_cache_directory.is_file() {
            panic!("Backup cache directory is a file");
        }

        self.backup_cache_directory = Some(backup_cache_directory);
        self
    }

    /// Set the service tier for requests (e.g., "flex")
    pub fn with_service_tier(mut self, service_tier: impl Into<String>) -> Self {
        self.service_tier = Some(service_tier.into());
        self
    }

    /// Set the prompt cache key for requests. A provider-side cache-routing hint
    /// (OpenAI `prompt_cache_key`); use one key per stable prompt prefix. Like the
    /// service tier, it does not affect response content, so it is excluded from the
    /// local response-cache key — setting or changing it never invalidates the cache.
    pub fn with_prompt_cache_key(mut self, prompt_cache_key: impl Into<String>) -> Self {
        self.prompt_cache_key = Some(prompt_cache_key.into());
        self
    }

    /// Set the reasoning effort for requests (e.g., "low", "medium", "high")
    pub fn with_reasoning_effort(mut self, reasoning_effort: impl Into<String>) -> Self {
        self.reasoning_effort = Some(reasoning_effort.into());
        self
    }

    /// Set extra fields to be included in the request body
    pub fn with_extra_body(mut self, extra_body: serde_json::Value) -> Self {
        self.extra_body = Some(extra_body);
        self
    }

    /// Set the maximum number of concurrent requests allowed
    pub fn with_max_concurrent_requests(self, max: usize) -> Self {
        Self {
            semaphore: Semaphore::new(max),
            ..self
        }
    }

    /// If set, all uncached requests will fail with [`ChatError::CacheMiss`] (or, in a
    /// batch, a per-item [`IndividualChatError::CacheMiss`]) instead of
    /// hitting the API. Useful for testing or offline usage.
    pub fn with_cached_only(self) -> Self {
        Self {
            cached_only: true,
            ..self
        }
    }

    /// Check another client's cache after this client's cache, without ever allowing the
    /// fallback client to make an API request. Add multiple fallbacks newest-to-oldest.
    ///
    /// This is useful when migrating models: configure the new model as `self`, then add the
    /// old model as a cache fallback. Requests reuse this client's cache first, then valid
    /// old-model responses, and only then call this client's API.
    /// Multiple fallbacks are checked in the order they are added.
    pub fn with_cache_fallback(mut self, fallback: ChatClient) -> Self {
        self.cache_fallbacks.push(fallback);
        self
    }

    /// Sets the base URL
    ///
    /// ```
    /// # use tysm::chat_completions::ChatClient;
    /// let api_key = "YOUR ANTHROPIC API KEY HERE";
    /// let client = ChatClient::new(api_key, "claude-3-7-sonnet-20250219").with_url("https://api.anthropic.com/v1/");
    /// ```
    ///
    /// or...
    ///
    /// ```
    /// # use tysm::chat_completions::ChatClient;
    /// let api_key = "YOUR GEMINI API KEY HERE";
    /// let client = ChatClient::new(api_key, "gemini-2.0-flash").with_url("https://generativelanguage.googleapis.com/v1beta/openai/");
    /// ```
    ///
    /// Panics if the argument is not a valid URL.
    pub fn with_url(self, url: impl Into<String>) -> Self {
        let url = url.into();
        let url = if url.ends_with('/') {
            url
        } else {
            format!("{}/", url)
        };

        let url = url::Url::parse(&url).unwrap();
        Self {
            base_url: url,
            ..self
        }
    }

    fn chat_completions_url(&self) -> url::Url {
        self.base_url.join(&self.chat_completions_path).unwrap()
    }

    /// Create a new [`ChatClient`].
    /// This will use the `OPENAI_API_KEY` environment variable to set the API key.
    /// It will also look in the `.env` file for an `OPENAI_API_KEY` variable (using dotenv).
    ///
    /// ```rust
    /// # use tysm::chat_completions::ChatClient;
    /// let client = ChatClient::from_env("gpt-4o").unwrap();
    /// ```
    pub fn from_env(model: impl Into<String>) -> Result<Self, OpenAiApiKeyError> {
        Ok(Self::new(api_key()?, model))
    }

    /// Send a chat message to the API and deserialize the response into the given type.
    ///
    /// ```rust
    /// # use tysm::chat_completions::ChatClient;
    /// #  let client = {
    /// #     let my_api = url::Url::parse("https://g7edusstdonmn3vxdh3qdypkrq0wzttx.lambda-url.us-east-1.on.aws/v1/").unwrap();
    /// #     ChatClient::from_env("gpt-4o").unwrap().with_url(my_api.to_string())
    /// # };
    ///
    /// #[derive(serde::Deserialize, Debug, schemars::JsonSchema)]
    /// struct CityName {
    ///     english: String,
    ///     local: String,
    /// }
    ///
    /// # tokio_test::block_on(async {
    /// let response: CityName = client.chat("What is the capital of Portugal?").await.unwrap();
    ///
    /// assert_eq!(response.english, "Lisbon");
    /// assert_eq!(response.local, "Lisboa");
    /// # })
    /// ```
    ///
    /// Responses are cached in the client, so sending the same request twice
    /// will return the same response.
    ///
    /// **Important:** The response type must implement the `JsonSchema` trait
    /// from the `schemars` crate.
    pub async fn chat<T: DeserializeOwned + JsonSchema>(
        &self,
        prompt: impl Into<String>,
    ) -> Result<T, ChatError> {
        self.chat_with_system_prompt("", prompt).await
    }

    /// Send a chat message to the API and deserialize the response into the given type.
    /// The first argument, the system prompt, is used to tell the AI how to behave during the conversation.
    ///
    /// ```rust
    /// # use tysm::chat_completions::ChatClient;
    /// #  let client = {
    /// #     let my_api = url::Url::parse("https://g7edusstdonmn3vxdh3qdypkrq0wzttx.lambda-url.us-east-1.on.aws/v1/").unwrap();
    /// #     ChatClient::from_env("gpt-4o").unwrap().with_url(my_api.to_string())
    /// # };
    ///
    /// #[derive(serde::Deserialize, Debug, schemars::JsonSchema)]
    /// struct CityName {
    ///     english: String,
    ///     local: String,
    /// }
    ///
    /// # tokio_test::block_on(async {
    /// let response: CityName = client.chat_with_system_prompt("You are an expert in cities", "What is the capital of Portugal?").await.unwrap();
    ///
    /// assert_eq!(response.english, "Lisbon");
    /// assert_eq!(response.local, "Lisboa");
    /// # })
    /// ```
    pub async fn chat_with_system_prompt<T: DeserializeOwned + JsonSchema>(
        &self,
        system_prompt: impl Into<String>,
        prompt: impl Into<String>,
    ) -> Result<T, ChatError> {
        let prompt = prompt.into();
        let system_prompt = system_prompt.into();

        let messages = vec![
            ChatMessage::system(system_prompt),
            ChatMessage::user(prompt),
        ];
        self.chat_with_messages::<T>(messages).await
    }

    /// Send a sequence of chat messages to the API and deserialize the response into the given type.
    /// This is useful for more advanced use cases like chatbots, multi-turn conversations, or when you need to use [Vision](https://platform.openai.com/docs/guides/vision).
    ///
    /// ```rust
    /// # use tysm::chat_completions::ChatClient;
    /// #  let client = {
    /// #     let my_api = url::Url::parse("https://g7edusstdonmn3vxdh3qdypkrq0wzttx.lambda-url.us-east-1.on.aws/v1/").unwrap();
    /// #     ChatClient::from_env("gpt-4o").unwrap().with_url(my_api.to_string())
    /// # };
    ///
    /// #[derive(serde::Deserialize, Debug, schemars::JsonSchema)]
    /// struct CityName {
    ///     english: String,
    ///     local: String,
    /// }
    ///
    /// # use tysm::chat_completions::ChatMessageContent;
    /// # use tysm::chat_completions::Role;
    /// # use tysm::chat_completions::ChatMessage;
    /// # tokio_test::block_on(async {
    /// let response: CityName = client.chat_with_messages(vec![
    ///     ChatMessage {
    ///         role: Role::System,
    ///         content: vec![ChatMessageContent::Text {
    ///             text: "You are an expert on cities.".to_string(),
    ///         }],
    ///     },
    ///     ChatMessage {
    ///         role: Role::User,
    ///         content: vec![ChatMessageContent::Text {
    ///             text: "What is the capital of Portugal?".to_string(),
    ///         }],
    ///     }
    /// ]).await.unwrap();
    ///
    /// assert_eq!(response.english, "Lisbon");
    /// assert_eq!(response.local, "Lisboa");
    /// # })
    /// ```
    pub async fn chat_with_messages<T: DeserializeOwned + JsonSchema>(
        &self,
        messages: Vec<ChatMessage>,
    ) -> Result<T, ChatError> {
        let json_schema = JsonSchemaFormat::new::<T>();

        let response_format = ResponseFormat::JsonSchema {
            json_schema: json_schema.clone(),
        };

        let chat_response = self
            .chat_with_messages_raw_mapped(messages, response_format, |chat_response| {
                Self::decode_json(&chat_response).map_err(|e| {
                    ChatError::ChatError(IndividualChatError::ResponseNotConformantToSchema {
                        error: e,
                        response: chat_response.trim().to_string(),
                        schema: serde_json::to_string(&json_schema.schema).unwrap(),
                    })
                })
            })
            .await?;

        Ok(chat_response)
    }

    /// Send a sequence of chat messages to the API. It's called "chat_with_messages_raw" because it allows you to specify any response format, and doesn't attempt to deserialize the chat completion.
    pub async fn chat_with_messages_raw(
        &self,
        messages: Vec<ChatMessage>,
        response_format: ResponseFormat,
    ) -> Result<String, ChatError> {
        self.chat_with_messages_raw_mapped(messages, response_format, Ok)
            .await
    }

    /// Send a sequence of chat messages to the API, then map the response to a different type. The response will only be cached if the mapping succeeds.
    async fn chat_with_messages_raw_mapped<T>(
        &self,
        messages: Vec<ChatMessage>,
        response_format: ResponseFormat,
        map_response: impl Fn(String) -> Result<T, ChatError>,
    ) -> Result<T, ChatError> {
        let chat_request = ChatRequest {
            model: self.model.clone(),
            messages,
            response_format,
            service_tier: self.service_tier.clone(),
            prompt_cache_key: self.prompt_cache_key.clone(),
            reasoning_effort: self.reasoning_effort.clone(),
            extra_body: self.extra_body.clone(),
        };

        let chat_request_str = serde_json::to_string(&chat_request).unwrap();

        let process_result = |cached_response: String| -> Result<(T, ChatUsage), ChatError> {
            let cached_response = match serde_json::from_str::<serde_json::Value>(&cached_response)
            {
                Ok(response) => response,
                Err(error) => {
                    return Err(ChatError::ApiDidNotReturnJson {
                        response: cached_response,
                        request: chat_request_str,
                        error,
                    });
                }
            };

            let chat_response: ChatResponseOrError =
                match serde_json::from_value(cached_response.clone()) {
                    Ok(response) => response,
                    Err(error) => {
                        let chat_request_str = if chat_request_str.len() > 100 {
                            chat_request_str
                                .chars()
                                .take(100)
                                .chain("...".chars())
                                .collect()
                        } else {
                            chat_request_str.clone()
                        };
                        return Err(ChatError::ApiParseError {
                            response: cached_response.clone(),
                            error,
                            request: chat_request_str,
                        });
                    }
                };
            let response = match chat_response {
                ChatResponseOrError::Response(response) => response,
                ChatResponseOrError::Error(error) => {
                    let chat_request_str = if chat_request_str.len() > 100 {
                        chat_request_str
                            .chars()
                            .take(100)
                            .chain("...".chars())
                            .collect()
                    } else {
                        chat_request_str.clone()
                    };
                    return Err(ChatError::ApiError(error, chat_request_str));
                }
            };
            let choice = response
                .choices
                .into_iter()
                .next()
                .ok_or(ChatError::NoChoices)?;
            let chat_response = choice
                .message
                .content()
                .map_err(IndividualChatError::Refusal)?;

            map_response(chat_response).map(|mapped_response| (mapped_response, response.usage))
        };

        if let Some(cached_response) = self
            .chat_cached(&chat_request, process_result.clone())
            .await
        {
            match cached_response {
                Ok((result, _usage)) => {
                    debug!("Using cached response from current model {}", self.model);
                    return Ok(result);
                }
                Err(error) => warn!(
                    "Ignoring invalid cached response from current model {}: {}",
                    self.model, error
                ),
            }
        }

        for fallback in &self.cache_fallbacks {
            let fallback_request = ChatRequest {
                model: fallback.model.clone(),
                messages: chat_request.messages.clone(),
                response_format: chat_request.response_format.clone(),
                service_tier: fallback.service_tier.clone(),
                prompt_cache_key: fallback.prompt_cache_key.clone(),
                reasoning_effort: fallback.reasoning_effort.clone(),
                extra_body: fallback.extra_body.clone(),
            };

            if let Some(cached_response) = fallback
                .chat_cached(&fallback_request, process_result.clone())
                .await
            {
                match cached_response {
                    Ok((result, _usage)) => {
                        debug!(
                            "Using cached response from fallback model {}",
                            fallback.model
                        );
                        return Ok(result);
                    }
                    Err(error) => {
                        warn!(
                            "Ignoring invalid cached response from fallback model {}: {}",
                            fallback.model, error
                        );
                    }
                }
            }
        }

        if self.cached_only {
            return Err(ChatError::CacheMiss);
        }
        let (chat_response, result, usage) = {
            let mut attempt = 1;
            loop {
                let outcome = match self.chat_uncached(&chat_request).await {
                    Ok(response) => process_result.clone()(response.clone())
                        .map(|(result, usage)| (response, result, usage)),
                    Err(error) => Err(error),
                };
                match outcome {
                    Ok(outcome) => break outcome,
                    Err(error) => {
                        match retry_delay(&error, attempt)
                            .filter(|_| attempt < LIVE_REQUEST_ATTEMPTS)
                        {
                            Some(delay) => {
                                warn!(
                                    "Chat request to {} failed on attempt \
                                     {attempt}/{LIVE_REQUEST_ATTEMPTS} ({error}), retrying in {}s",
                                    self.model,
                                    delay.as_secs()
                                );
                                tokio::time::sleep(delay).await;
                                attempt += 1;
                            }
                            None => return Err(error),
                        }
                    }
                }
            }
        };
        *self.usage.write().unwrap() += usage;
        self.record_spend(self.service_tier.as_deref(), usage, 1.0);

        // cache the response
        {
            let chat_request_cache_key = chat_request.cache_key();
            let chat_request = serde_json::to_string(&chat_request)
                .map_err(|e| ChatError::JsonSerializeError(e, chat_request.clone()))?;

            if let Some(cache_directory) = &self.cache_directory {
                // Compress the response with zstd before writing to disk
                let compressed = zstd::encode_all(chat_response.as_bytes(), 3)?;
                crate::cache::write_to_cache_dir(
                    cache_directory,
                    &chat_request_cache_key,
                    &compressed,
                )
                .await?;
            }

            self.lru.insert(chat_request, chat_response);
        }

        Ok(result)
    }

    /// Send chat messages to the batch API and deserialize the responses into the given type.
    ///
    /// This goes through the batch API, which is cheaper and has higher ratelimits, but is much higher-latency. The responses to the batch API stick around in OpenAI's servers for some time, and before starting a new batch request, `tysm` will automatically check if that same request has been made before (and reuse it if so).
    pub async fn batch_chat<T: DeserializeOwned + JsonSchema>(
        &self,
        prompts: Vec<impl Into<String>>,
        on_progress: impl FnMut(&crate::batch::Batch),
    ) -> Result<Vec<Result<T, IndividualChatError>>, BatchChatError> {
        self.batch_chat_with_system_prompt("", prompts, on_progress)
            .await
    }

    /// Send objects through the Batch API using `prompt` to render each object, returning
    /// every original object paired with its result. This is the chat equivalent of
    /// [`crate::embeddings::EmbeddingsClient::embed_fn`].
    pub async fn batch_chat_fn<'a, I, S, T>(
        &self,
        items: &'a [I],
        prompt: impl Fn(&'a I) -> S,
        on_progress: impl FnMut(&crate::batch::Batch),
    ) -> Result<Vec<(&'a I, Result<T, IndividualChatError>)>, BatchChatError>
    where
        S: Into<String>,
        T: DeserializeOwned + JsonSchema,
    {
        self.batch_chat_with_system_prompt_fn("", items, prompt, on_progress)
            .await
    }

    /// Send a batch of chat messages to the API and deserialize the responses into the given type.
    /// The first argument, the system prompt, is used to tell the AI how to behave during the conversations.
    ///
    /// This goes through the batch API, which is cheaper and has higher ratelimits, but is much higher-latency. The responses to the batch API stick around in OpenAI's servers for some time, and before starting a new batch request, `tysm` will automatically check if that same request has been made before (and reuse it if so).
    pub async fn batch_chat_with_system_prompt<T: DeserializeOwned + JsonSchema>(
        &self,
        system_prompt: impl Into<String> + Clone,
        prompts: Vec<impl Into<String>>,
        on_progress: impl FnMut(&crate::batch::Batch),
    ) -> Result<Vec<Result<T, IndividualChatError>>, BatchChatError> {
        let prompts = prompts
            .into_iter()
            .map(|prompt| {
                let prompt = prompt.into();
                let system_prompt = system_prompt.clone().into();

                vec![
                    ChatMessage::system(system_prompt),
                    ChatMessage::user(prompt),
                ]
            })
            .collect();

        self.batch_chat_with_messages(prompts, on_progress).await
    }

    /// Send objects through the Batch API with a shared system prompt, returning every
    /// original object paired with its result.
    pub async fn batch_chat_with_system_prompt_fn<'a, I, S, T>(
        &self,
        system_prompt: impl Into<String> + Clone,
        items: &'a [I],
        prompt: impl Fn(&'a I) -> S,
        on_progress: impl FnMut(&crate::batch::Batch),
    ) -> Result<Vec<(&'a I, Result<T, IndividualChatError>)>, BatchChatError>
    where
        S: Into<String>,
        T: DeserializeOwned + JsonSchema,
    {
        let prompts = items.iter().map(&prompt).collect::<Vec<_>>();
        let results = self
            .batch_chat_with_system_prompt(system_prompt, prompts, on_progress)
            .await?;
        Ok(items.iter().zip(results).collect())
    }

    /// Send a batch of sequences of chat messages to the API and deserialize the responses into the given type.
    /// This is useful for more advanced use cases like chatbots, multi-turn conversations, or when you need to use [Vision](https://platform.openai.com/docs/guides/vision).
    ///
    /// This goes through the batch API, which is cheaper and has higher ratelimits, but is much higher-latency. The responses to the batch API stick around in OpenAI's servers for some time, and before starting a new batch request, `tysm` will automatically check if that same request has been made before (and reuse it if so).
    pub async fn batch_chat_with_messages<T: DeserializeOwned + JsonSchema>(
        &self,
        messages: Vec<Vec<ChatMessage>>,
        on_progress: impl FnMut(&crate::batch::Batch),
    ) -> Result<Vec<Result<T, IndividualChatError>>, BatchChatError> {
        let json_schema = JsonSchemaFormat::new::<T>();

        let response_format = ResponseFormat::JsonSchema {
            json_schema: json_schema.clone(),
        };

        // Deserialization is handed *down* rather than applied to the results,
        // so a response that does not fit the schema is never written to the
        // cache. Mapping afterwards would cache it first and then reject it,
        // making the failure permanent.
        let schema = serde_json::to_string(&json_schema.schema).unwrap();
        self.batch_chat_with_messages_raw_mapped(
            messages
                .into_iter()
                .map(|m| (m, response_format.clone()))
                .collect(),
            on_progress,
            move |raw| {
                Self::decode_json(&raw).map_err(|error| {
                    IndividualChatError::ResponseNotConformantToSchema {
                        error,
                        response: raw.trim().to_string(),
                        schema: schema.clone(),
                    }
                })
            },
        )
        .await
    }

    /// Send objects through the Batch API using `messages` to build each conversation,
    /// returning every original object paired with its result.
    pub async fn batch_chat_with_messages_fn<'a, I, T>(
        &self,
        items: &'a [I],
        messages: impl Fn(&'a I) -> Vec<ChatMessage>,
        on_progress: impl FnMut(&crate::batch::Batch),
    ) -> Result<Vec<(&'a I, Result<T, IndividualChatError>)>, BatchChatError>
    where
        T: DeserializeOwned + JsonSchema,
    {
        let messages = items.iter().map(messages).collect();
        let results = self.batch_chat_with_messages(messages, on_progress).await?;
        Ok(items.iter().zip(results).collect())
    }

    fn request_for_messages(
        &self,
        messages: Vec<ChatMessage>,
        response_format: ResponseFormat,
    ) -> ChatRequest {
        ChatRequest {
            model: self.model.clone(),
            messages,
            response_format,
            service_tier: self.service_tier.clone(),
            prompt_cache_key: self.prompt_cache_key.clone(),
            reasoning_effort: self.reasoning_effort.clone(),
            extra_body: self.extra_body.clone(),
        }
    }

    async fn cached_batch_content(
        &self,
        request: &ChatRequest,
    ) -> Option<Result<String, IndividualChatError>> {
        let clients = std::iter::once(self)
            .chain(self.cache_fallbacks.iter())
            .collect::<Vec<_>>();

        for client in clients {
            let candidate = ChatRequest {
                model: client.model.clone(),
                messages: request.messages.clone(),
                response_format: request.response_format.clone(),
                service_tier: client.service_tier.clone(),
                prompt_cache_key: client.prompt_cache_key.clone(),
                reasoning_effort: client.reasoning_effort.clone(),
                extra_body: client.extra_body.clone(),
            };
            let Some(Ok(raw)) = client.chat_cached(&candidate, Ok).await else {
                continue;
            };
            let Ok(response) = serde_json::from_str::<ChatResponseOrError>(&raw) else {
                warn!(
                    "Ignoring malformed cached batch response for {}",
                    client.model
                );
                continue;
            };
            match response {
                ChatResponseOrError::Response(response) => {
                    let Some(choice) = response.choices.into_iter().next() else {
                        continue;
                    };
                    return Some(
                        choice
                            .message
                            .content()
                            .map_err(IndividualChatError::Refusal),
                    );
                }
                ChatResponseOrError::Error(_) => continue,
            }
        }
        None
    }

    async fn cache_batch_response(
        &self,
        request: &ChatRequest,
        response: &ChatResponse,
    ) -> Result<(), std::io::Error> {
        let raw = serde_json::to_string(response).expect("ChatResponse is serializable");
        if let Some(cache_directory) = &self.cache_directory {
            let compressed = zstd::encode_all(raw.as_bytes(), 3)?;
            crate::cache::write_to_cache_dir(cache_directory, &request.cache_key(), &compressed)
                .await?;
        }
        self.lru.insert(
            serde_json::to_string(request).expect("ChatRequest is serializable"),
            raw,
        );
        Ok(())
    }

    /// Send a batch of sequences of chat messages to the API. It's called "chat_with_messages_raw" because it allows you to specify any response format, and doesn't attempt to deserialize the chat completion.
    ///
    /// This goes through the batch API, which is cheaper and has higher ratelimits, but is much higher-latency. The responses to the batch API stick around in OpenAI's servers for some time, and before starting a new batch request, `tysm` will automatically check if that same request has been made before (and reuse it if so).
    pub async fn batch_chat_with_messages_raw(
        &self,
        prompts: Vec<(Vec<ChatMessage>, ResponseFormat)>,
        on_progress: impl FnMut(&crate::batch::Batch),
    ) -> Result<Vec<Result<String, IndividualChatError>>, BatchChatError> {
        self.batch_chat_with_messages_raw_mapped(prompts, on_progress, Ok)
            .await
    }

    /// Send a batch of chat requests, then map each response to a different
    /// type. A response is cached only if its mapping succeeds.
    ///
    /// The mapping is taken as an argument, rather than applied by the caller
    /// afterwards, for the same reason the live path does it
    /// ([`Self::chat_with_messages_raw_mapped`]): a response we cannot use is a
    /// response we must not cache. Caching first and rejecting second would
    /// serve the bad response back as a hit on every future run, so re-running
    /// could never repair it.
    async fn batch_chat_with_messages_raw_mapped<T>(
        &self,
        prompts: Vec<(Vec<ChatMessage>, ResponseFormat)>,
        mut on_progress: impl FnMut(&crate::batch::Batch),
        map_response: impl Fn(String) -> Result<T, IndividualChatError>,
    ) -> Result<Vec<Result<T, IndividualChatError>>, BatchChatError> {
        info!("Starting batch chat with {} prompts", prompts.len());

        let mut output = (0..prompts.len()).map(|_| None).collect::<Vec<_>>();
        let mut misses = Vec::new();
        // Cache lookups run concurrently: each one walks this client's cache and
        // then every fallback cache, and a mostly-warm batch of tens of
        // thousands of prompts spends its whole life here if they are awaited
        // one at a time. Unordered so one slow lookup never holds up the rest;
        // `output` is indexed and `misses` is re-sorted, so nothing downstream
        // can tell.
        use futures::StreamExt as _;
        let map_response = &map_response;
        let mut lookups = futures::stream::iter(prompts.into_iter().enumerate())
            .map(|(index, (messages, response_format))| async move {
                let request = self.request_for_messages(messages, response_format);
                // A cached entry that will not map is not an answer, so it counts as
                // a miss and gets asked again; the fresh response then replaces it.
                // Returning the stored failure instead would make it permanent — the
                // request would never be retried and re-running could never repair
                // it. This also heals entries written before caching was gated on
                // the mapping succeeding.
                let cached = self
                    .cached_batch_content(&request)
                    .await
                    .map(|cached| cached.and_then(map_response));
                (index, request, cached)
            })
            .buffer_unordered(CACHE_LOOKUP_CONCURRENCY);
        // Consumed as results arrive rather than collected, so a hit's request
        // is dropped as soon as it is known to be one.
        while let Some((index, request, cached)) = lookups.next().await {
            match cached {
                Some(Ok(value)) => output[index] = Some(Ok(value)),
                Some(Err(_)) | None => misses.push((index, request)),
            }
        }
        misses.sort_by_key(|(index, _)| *index);

        if misses.is_empty() {
            return Ok(output.into_iter().map(Option::unwrap).collect());
        }
        if self.cached_only {
            // Cached-only is a per-request condition, as on the live path: the
            // hits are still answers, and callers already handle per-item
            // errors, so a miss is reported in its slot rather than failing
            // the whole batch.
            warn!(
                "cached-only mode: {} of {} batch request(s) not found in cache",
                misses.len(),
                output.len()
            );
            for (index, _) in misses {
                output[index] = Some(Err(IndividualChatError::CacheMiss));
            }
            return Ok(output.into_iter().map(Option::unwrap).collect());
        }

        // Send the uncached requests live instead of batching when there are
        // too few to be worth a batch's fixed overhead (or always, under
        // `with_no_batch`). Decided on `misses`, not the caller's total, so a
        // retry whose responses are nearly all cached skips the queue entirely.
        let send_live = misses.len() <= self.small_batch_threshold;
        let batch_client = BatchClient::from(self);
        let existing = match self.find_batch_by_hash(&batch_client, &misses).await {
            Ok(existing) => existing,
            // Live mode must work against an endpoint that has no Batch API at
            // all; there is nothing to harvest there.
            Err(error) if send_live => {
                warn!("could not look for an existing batch, sending live: {error}");
                None
            }
            Err(error) => return Err(error),
        };
        if let Some(mut batch) = existing {
            if !batch.status.is_terminal() {
                if send_live && batch.status != BatchStatus::Cancelling {
                    warn!(
                        "sending live: cancelling in-flight batch {} for these requests \
                         and harvesting its completed results",
                        batch.id
                    );
                    // The listing may be stale: a batch that finished in the
                    // meantime rejects cancellation, and its results are
                    // exactly what the wait below returns.
                    if let Err(error) = batch_client.cancel_batch(&batch.id).await {
                        warn!("could not cancel batch {}: {error}", batch.id);
                    }
                }
                batch = batch_client
                    .wait_for_batch(&batch.id, &mut on_progress)
                    .await?;
            }
            // An old batch whose output file is gone (OpenAI keeps them for
            // thirty days) is worth nothing, not a reason to stop: its
            // requests are simply still misses.
            if let Err(error) = self
                .harvest_batch(
                    &batch_client,
                    &batch,
                    true,
                    &mut misses,
                    &mut output,
                    map_response,
                )
                .await
            {
                warn!("could not harvest batch {}: {error}", batch.id);
            }
        }
        if misses.is_empty() {
            return Ok(output.into_iter().map(Option::unwrap).collect());
        }

        if send_live {
            info!(
                "Sending {} uncached request(s) live instead of batching",
                misses.len()
            );
            // Uses the *mapped* live call so this path caches on exactly the
            // same condition as the batch path — a response that fails the
            // caller's mapping is not written to the cache either way.
            let mapped = |raw: String| map_response(raw).map_err(ChatError::ChatError);
            let mapped = &mapped;
            let results: Vec<(usize, Result<T, IndividualChatError>)> =
                futures::stream::iter(misses)
                    .map(|(index, request)| async move {
                        let result = self
                            .chat_with_messages_raw_mapped(
                                request.messages.clone(),
                                request.response_format.clone(),
                                mapped,
                            )
                            .await;
                        let result = match result {
                            Ok(response) => Ok(response),
                            // A live failure becomes this request's error rather
                            // than failing the whole call, matching what the
                            // batch path returns per item.
                            Err(ChatError::ChatError(e)) => Err(e),
                            Err(e) => Err(IndividualChatError::Other(e.to_string())),
                        };
                        (index, result)
                    })
                    .buffer_unordered(LIVE_CONCURRENCY)
                    .collect()
                    .await;
            for (index, result) in results {
                output[index] = Some(result);
            }
            return Ok(output.into_iter().map(Option::unwrap).collect());
        }

        // Never resubmit the successful portion of a partial batch.
        let batch = self.submit_batch(&batch_client, &misses).await?;
        let batch = batch_client
            .wait_for_batch(&batch.id, &mut on_progress)
            .await?;
        self.harvest_batch(
            &batch_client,
            &batch,
            false,
            &mut misses,
            &mut output,
            map_response,
        )
        .await?;
        if let Some((_, request)) = misses.first() {
            return Err(BatchChatError::CustomIdNotFound(Self::batch_custom_id(
                request,
            )));
        }
        Ok(output.into_iter().map(Option::unwrap).collect())
    }

    fn batch_request_hash(request: &ChatRequest) -> u64 {
        const_xxh3(serde_json::to_string(request).unwrap().as_bytes())
    }

    fn batch_custom_id(request: &ChatRequest) -> String {
        format!("request-{}", Self::batch_request_hash(request))
    }

    fn batch_hash(misses: &[(usize, ChatRequest)]) -> String {
        misses
            .iter()
            .map(|(_, request)| Self::batch_request_hash(request))
            .collect::<HashSet<_>>()
            .into_iter()
            .fold(0u64, u64::wrapping_add)
            .to_string()
    }

    async fn find_batch_by_hash(
        &self,
        client: &BatchClient,
        misses: &[(usize, ChatRequest)],
    ) -> Result<Option<Batch>, BatchChatError> {
        let hash = Self::batch_hash(misses);
        Ok(client.list_batches().await?.into_iter().find(|batch| {
            batch.status != BatchStatus::Failed
                && batch.metadata.as_ref().and_then(|m| m.get("request_hash")) == Some(&hash)
        }))
    }

    async fn submit_batch(
        &self,
        client: &BatchClient,
        misses: &[(usize, ChatRequest)],
    ) -> Result<Batch, BatchChatError> {
        let requests = misses
            .iter()
            .map(|(_, request)| {
                let id = Self::batch_custom_id(request);
                (id.clone(), BatchRequestItem::new_chat(id, request.clone()))
            })
            .collect::<HashMap<_, _>>()
            .into_values()
            .collect::<Vec<_>>();
        let content = client.create_batch_content(&requests);
        let file = client
            .files_client
            .upload_bytes("batch_request", content, crate::files::FilePurpose::Batch)
            .await?;
        Ok(client
            .create_batch(
                file.id,
                HashMap::from([("request_hash".to_owned(), Self::batch_hash(misses))]),
            )
            .await?)
    }

    /// Fill `output` from a terminal batch's results and drop the filled
    /// slots from `misses`. With `retry_failures`, a failed item (an API
    /// error, a refusal, or a response the caller's mapping rejects) stays a
    /// miss so it is asked again; otherwise it is reported in its slot. The
    /// first suits a batch found from an earlier run — a failure is never
    /// cached, so the same batch would otherwise replay it forever — and the
    /// second the batch this call just ran.
    async fn harvest_batch<T>(
        &self,
        client: &BatchClient,
        batch: &Batch,
        retry_failures: bool,
        misses: &mut Vec<(usize, ChatRequest)>,
        output: &mut [Option<Result<T, IndividualChatError>>],
        map_response: &impl Fn(String) -> Result<T, IndividualChatError>,
    ) -> Result<(), BatchChatError> {
        let expected = misses
            .iter()
            .map(|(_, r)| Self::batch_custom_id(r))
            .collect::<HashSet<_>>();
        let mut results = client.get_batch_results(batch).await?;
        // Only completed files can still be settling. Explicit failures have IDs
        // too, so they do not cause pointless file-download retries.
        if batch.status == BatchStatus::Completed && batch.output_file_id.is_some() {
            for delay_secs in [2u64, 5, 15, 30, 60] {
                let returned = results
                    .iter()
                    .map(|r| r.custom_id.clone())
                    .collect::<HashSet<_>>();
                if expected.is_subset(&returned) {
                    break;
                }
                info!(
                    "batch {} output file is missing IDs; re-fetching in {delay_secs}s",
                    batch.id
                );
                tokio::time::sleep(std::time::Duration::from_secs(delay_secs)).await;
                results = client.get_batch_results(batch).await?;
            }
        }
        // Every item the file mentions is resolved here, one way or the other:
        // a well-formed success, or the reason it is not. Only IDs the file
        // never mentions stay misses to be asked again.
        let mut responses: HashMap<String, Result<ChatResponse, String>> = HashMap::new();
        for item in results {
            if !expected.contains(&item.custom_id) {
                continue;
            }
            let response = match (item.error, item.response) {
                (Some(error), _) => Err(error.to_string()),
                (None, None) => Err("batch item had neither response nor error".to_owned()),
                (None, Some(response)) => match serde_json::from_value(response.body) {
                    Ok(ChatResponseOrError::Response(response)) => Ok(response),
                    Ok(ChatResponseOrError::Error(error)) => Err(error.to_string()),
                    Err(error) => Err(format!("malformed response: {error}")),
                },
            };
            if let Err(error) = &response {
                warn!("batch {} item {}: {error}", batch.id, item.custom_id);
            }
            responses.insert(item.custom_id, response);
        }
        // Account once per billed response, not per duplicate prompt or download.
        for response in responses.values().flatten() {
            *self.batch_usage.write().unwrap() += response.usage;
            self.record_spend(None, response.usage, 0.5);
        }
        let mut cached = HashSet::new();
        for (index, request) in misses.iter() {
            let id = Self::batch_custom_id(request);
            // Mapped per slot: T need not be Clone, and duplicate prompts must
            // not consume one another's response.
            let value = match responses.get(&id) {
                None => continue,
                Some(Err(error)) => Err(IndividualChatError::Other(error.clone())),
                Some(Ok(response)) => match response.choices.first() {
                    None => Err(IndividualChatError::Other(
                        "response had no choices".to_owned(),
                    )),
                    Some(choice) => {
                        let value = choice
                            .message
                            .clone()
                            .content()
                            .map_err(IndividualChatError::Refusal)
                            .and_then(map_response);
                        // The cache write lives inside the `Ok` branch so a
                        // refusal or a response that fails the caller's
                        // mapping is never stored.
                        if value.is_ok() && cached.insert(id) {
                            self.cache_batch_response(request, response).await?;
                        }
                        value
                    }
                },
            };
            if value.is_err() && retry_failures {
                continue;
            }
            output[*index] = Some(value);
        }
        info!(
            "harvested {} of {} results from batch {} ({})",
            cached.len(),
            expected.len(),
            batch.id,
            batch.status
        );
        misses.retain(|(index, _)| output[*index].is_none());
        Ok(())
    }

    async fn chat_cached<T>(
        &self,
        chat_request: &ChatRequest,
        map_response: impl FnOnce(String) -> Result<T, ChatError>,
    ) -> Option<Result<T, ChatError>> {
        let chat_request_cache_key = chat_request.cache_key();
        // LEGACY CACHE KEY MIGRATION (can be removed in a future version)
        let legacy_cache_key = chat_request.legacy_cache_key();
        let chat_request = serde_json::to_string(chat_request).ok()?;

        // First, check the cache
        if let Some(response) = self.lru.get(&chat_request) {
            return Some(map_response(response.clone()));
        }

        // Then, check the cache directory (sharded, then flat)
        let cache_directory = self.cache_directory.as_ref()?;
        if !cache_directory.exists() {
            panic!(
                "Cache directory does not exist: {}",
                cache_directory.display()
            );
        }

        // Helper to search for a cache key across main and backup cache directories.
        // Returns the compressed data if found, and copies it to the main cache under
        // `copy_as_key` if it was found elsewhere.
        let find_in_cache = |key: &str, copy_as_key: &str| {
            let cache_directory = cache_directory.clone();
            let backup_cache_directory = self.backup_cache_directory.clone();
            let key = key.to_string();
            let copy_as_key = copy_as_key.to_string();
            async move {
                // Check main cache directory
                if let Some(data) = crate::cache::read_from_cache_dir(&cache_directory, &key).await
                {
                    // If found under a different key, copy to the canonical key
                    if key != copy_as_key {
                        let _ =
                            crate::cache::write_to_cache_dir(&cache_directory, &copy_as_key, &data)
                                .await;
                    }
                    return Some(data);
                }

                // Check backup cache directory
                if let Some(backup) = &backup_cache_directory {
                    if backup.exists() {
                        if let Some(data) = crate::cache::read_from_cache_dir(backup, &key).await {
                            // Copy to main cache under the canonical key
                            let _ = crate::cache::write_to_cache_dir(
                                &cache_directory,
                                &copy_as_key,
                                &data,
                            )
                            .await;
                            return Some(data);
                        }
                    }
                }

                None
            }
        };

        // Read the compressed data from disk, checking sharded then flat paths,
        // then falling back to backup cache directory
        let compressed_data = if let Some(data) =
            find_in_cache(&chat_request_cache_key, &chat_request_cache_key).await
        {
            data
        }
        // LEGACY CACHE KEY MIGRATION (can be removed in a future version)
        // Try the legacy cache key (pre-v0.17.1 format with duplicate additionalProperties).
        // If found, copy to the new cache key so future lookups use the new format.
        else if legacy_cache_key != chat_request_cache_key {
            find_in_cache(&legacy_cache_key, &chat_request_cache_key).await?
        }
        // END LEGACY CACHE KEY MIGRATION
        else {
            return None;
        };

        // Decompress the data
        let decompressed_data = zstd::decode_all(compressed_data.as_slice()).ok()?;

        // Convert bytes back to string
        let response = String::from_utf8(decompressed_data).ok()?;

        Some(map_response(response))
    }

    async fn chat_uncached(&self, chat_request: &ChatRequest) -> Result<String, ChatError> {
        let _permit = self.semaphore.acquire().await.unwrap();

        let response = self
            .http_client
            .post(self.chat_completions_url())
            .header("Authorization", format!("Bearer {}", self.api_key.clone()))
            .header("Content-Type", "application/json")
            .json(chat_request)
            .send()
            .await?
            .text()
            .await?;

        Ok(response)
    }

    fn decode_json<T: DeserializeOwned>(json: &str) -> Result<T, serde_json::Error> {
        match serde_json::from_str(json) {
            Ok(chat_response) => Ok(chat_response),
            Err(e) => {
                // try decoding each line separately
                {
                    let lines = json.lines();
                    for line in lines {
                        if let Ok(chat_response) = serde_json::from_str(line) {
                            return Ok(chat_response);
                        }
                    }
                }

                // give up
                Err(e)
            }
        }
    }

    /// Returns how many tokens have been used so far, excluding Batch API requests
    /// (see [`batch_usage`](Self::batch_usage)).
    ///
    /// Does not double-count tokens used in cached responses.
    pub fn usage(&self) -> ChatUsage {
        *self.usage.read().unwrap()
    }

    /// Returns how many tokens have been used via the Batch API so far.
    pub fn batch_usage(&self) -> ChatUsage {
        *self.batch_usage.read().unwrap()
    }

    /// Attempts to compute the cost in dollars of the usage of this client,
    /// including Batch API usage at its 50% discount.
    ///
    /// This is provided on a best-effort basis. The prices are hardcoded into
    /// the library (as OpenAI doesn't provide an API to get API pricing info),
    /// and may be out of date or unavailable for the model you're using.
    /// If you notice the prices being out of date, [please leave an issue](https://github.com/not-pizza/tysm)!
    pub fn cost(&self) -> Option<f64> {
        *self.spend.read().unwrap()
    }

    /// Price one request and add it to the running total, `discount` scaling
    /// the result (the Batch API bills at half).
    ///
    /// Priced here, per request, rather than by pricing the accumulated token
    /// counts later: a model with a long-context premium charges by how large
    /// one prompt was, which the totals no longer remember.
    fn record_spend(&self, service_tier: Option<&str>, usage: ChatUsage, discount: f64) {
        let mut spend = self.spend.write().unwrap();
        match crate::model_prices::cost_of_call(&self.model, service_tier, usage) {
            Some(dollars) => {
                if let Some(total) = spend.as_mut() {
                    *total += dollars * discount;
                }
            }
            // One unpriced request makes the total unknowable, and it stays
            // that way — a later priced request must not resurrect it.
            None => *spend = None,
        }
    }
}

#[test]
fn test_deser() {
    let s = r#"
{
    "choices": [
        {
        "finish_reason": "stop",
        "index": 0,
        "logprobs": null,
        "message": {
            "content": "Hey there! When replying to someone who's asked about what you're studying, it's all about how you present it. Even if you think math might sound boring, you can share why you find it interesting or how it applies to everyday life. Try saying something like, \"I'm actually diving into the world of math! It's fascinating because [insert a fun fact about your studies or why you chose it]. What about you? What are you passionate about?\" This way, you're flipping the script from just stating your major to sharing your enthusiasm!",
            "role": "assistant"
        }
        }
    ],
    "created": 1714696172,
    "id": "chatcmpl-9Kb5oqHOdNRLuFJHCTQFOeU516mU8",
    "model": "gpt-4-0125-preview",
    "object": "chat.completion",
    "system_fingerprint": null,
    "usage": {
        "completion_tokens": 123,
        "prompt_tokens": 188,
        "total_tokens": 311
    }
}
"#;
    let _chat_response: ChatResponse = serde_json::from_str(s).unwrap();
}

#[test]
fn cost_includes_batch_usage_at_half_price() {
    let client = ChatClient::new("sk-test", "gpt-4o");
    let usage = ChatUsage {
        prompt_tokens: 1_000_000,
        completion_tokens: 1_000_000,
        total_tokens: 2_000_000,
        prompt_token_details: None,
        completion_token_details: None,
    };

    client.record_spend(None, usage, 1.0);
    let live_only = client.cost().unwrap();

    client.record_spend(None, usage, 0.5);
    let with_batch = client.cost().unwrap();

    // The same usage again via the Batch API should cost exactly half as much more.
    assert!((with_batch - live_only * 1.5).abs() < 1e-9);
}

#[test]
fn an_unpriced_request_makes_the_total_unknowable() {
    let client = ChatClient::new("sk-test", "some-model-we-have-no-price-for");
    let usage = ChatUsage {
        prompt_tokens: 1_000,
        completion_tokens: 1_000,
        total_tokens: 2_000,
        prompt_token_details: None,
        completion_token_details: None,
    };

    // A client that has done nothing has spent nothing, whatever its model.
    assert_eq!(client.cost(), Some(0.0));

    client.record_spend(None, usage, 1.0);
    assert_eq!(client.cost(), None);
}

/// The client accumulates dollars, not tokens, so a long run of short requests
/// is never mistaken for one long-context request.
#[test]
fn spend_accumulates_per_request_not_from_summed_tokens() {
    let client = ChatClient::new("sk-test", "gpt-5.6-luna");
    let short = ChatUsage {
        prompt_tokens: 50_000,
        completion_tokens: 0,
        total_tokens: 50_000,
        prompt_token_details: None,
        completion_token_details: None,
    };

    // Six 50k requests sum to 300k tokens, past luna's 272k threshold — but
    // each was billed on its own at the cheap card: 0.3M @ $0.20 = $0.06.
    for _ in 0..6 {
        client.record_spend(None, short, 1.0);
    }
    let spent = client.cost().unwrap();
    assert!((spent - 0.06).abs() < 1e-9, "{spent}");

    // Those same 300k tokens arriving as a single request cross the threshold
    // and cost twice as much.
    let one_long = ChatUsage {
        prompt_tokens: 300_000,
        total_tokens: 300_000,
        ..short
    };
    let fresh = ChatClient::new("sk-test", "gpt-5.6-luna");
    fresh.record_spend(None, one_long, 1.0);
    let spent_at_once = fresh.cost().unwrap();
    assert!((spent_at_once - 0.12).abs() < 1e-9, "{spent_at_once}");
}

#[test]
fn service_tier_excluded_from_cache_key() {
    // Create two identical requests except for service_tier and prompt_cache_key,
    // neither of which affects response content
    let request1 = ChatRequest {
        model: "gpt-4o".to_string(),
        messages: vec![ChatMessage::user("test")],
        response_format: ResponseFormat::Text,
        service_tier: None,
        prompt_cache_key: None,
        reasoning_effort: None,
        extra_body: None,
    };

    let request2 = ChatRequest {
        model: "gpt-4o".to_string(),
        messages: vec![ChatMessage::user("test")],
        response_format: ResponseFormat::Text,
        service_tier: Some("flex".to_string()),
        prompt_cache_key: Some("my-prompt-cache-key".to_string()),
        reasoning_effort: None,
        extra_body: None,
    };

    // The cache keys should be identical even though service_tier and
    // prompt_cache_key differ
    assert_eq!(request1.cache_key(), request2.cache_key());

    // An unset prompt_cache_key must vanish from the serialization entirely, so cache
    // keys hashed before the field existed still match — adding the field must not
    // orphan existing on-disk caches.
    let serialized = serde_json::to_string(&request1).unwrap();
    assert!(
        !serialized.contains("prompt_cache_key"),
        "None prompt_cache_key leaked into serialization: {serialized}"
    );

    // Test that reasoning_effort IS included in cache key (different reasoning_effort = different cache)
    let request3 = ChatRequest {
        model: "gpt-4o".to_string(),
        messages: vec![ChatMessage::user("test")],
        response_format: ResponseFormat::Text,
        service_tier: None,
        prompt_cache_key: None,
        reasoning_effort: Some("high".to_string()),
        extra_body: None,
    };

    assert_ne!(request1.cache_key(), request3.cache_key());

    // Verify that different messages produce different cache keys
    let request4 = ChatRequest {
        model: "gpt-4o".to_string(),
        messages: vec![ChatMessage::user("different message")],
        response_format: ResponseFormat::Text,
        service_tier: Some("flex".to_string()),
        prompt_cache_key: Some("my-prompt-cache-key".to_string()),
        reasoning_effort: Some("high".to_string()),
        extra_body: None,
    };

    assert_ne!(request1.cache_key(), request4.cache_key());
}

#[test]
fn schema_has_no_duplicate_additional_properties() {
    use schemars::JsonSchema;

    #[derive(JsonSchema)]
    #[allow(dead_code)]
    struct TestStruct {
        name: String,
        age: u32,
    }

    let schema = JsonSchemaFormat::new::<TestStruct>();
    let serialized = serde_json::to_string_pretty(&schema).unwrap();
    println!("{serialized}");

    // additionalProperties should only appear on object-type schemas, not on primitives,
    // and should never be duplicated
    let count = serialized.matches("additionalProperties").count();
    assert_eq!(
        count, 1,
        "Expected exactly 1 additionalProperties (on the root object) but found {count} in:\n{serialized}"
    );

    // Verify the legacy cache key differs from the new one (proving migration is needed)
    let request = ChatRequest {
        model: "gpt-4o".to_string(),
        messages: vec![ChatMessage::user("test")],
        response_format: ResponseFormat::JsonSchema {
            json_schema: JsonSchemaFormat::new::<TestStruct>(),
        },
        service_tier: None,
        prompt_cache_key: None,
        reasoning_effort: None,
        extra_body: None,
    };
    assert_ne!(
        request.cache_key(),
        request.legacy_cache_key(),
        "Legacy and new cache keys should differ for structured output requests"
    );

    // But for non-schema requests, they should be the same
    let text_request = ChatRequest {
        model: "gpt-4o".to_string(),
        messages: vec![ChatMessage::user("test")],
        response_format: ResponseFormat::Text,
        service_tier: None,
        prompt_cache_key: None,
        reasoning_effort: None,
        extra_body: None,
    };
    assert_eq!(
        text_request.cache_key(),
        text_request.legacy_cache_key(),
        "Legacy and new cache keys should match for non-schema requests"
    );
}

#[cfg(test)]
#[tokio::test]
#[ignore]
async fn openai_structured_output() {
    use schemars::JsonSchema;
    use serde::Deserialize;

    #[derive(Deserialize, Debug, JsonSchema)]
    #[allow(dead_code)]
    struct CapitalCity {
        city: String,
        country: String,
    }

    #[cfg(feature = "dotenvy")]
    dotenvy::dotenv().ok();
    let client = ChatClient::from_env("gpt-4o-mini").unwrap();

    let result: CapitalCity = client.chat("What is the capital of France?").await.unwrap();
    assert_eq!(result.city, "Paris");
}

#[cfg(test)]
#[tokio::test]
#[ignore]
async fn gemini_structured_output() {
    use schemars::JsonSchema;
    use serde::Deserialize;

    #[derive(Deserialize, Debug, JsonSchema)]
    #[allow(dead_code)]
    struct CapitalCity {
        city: String,
        country: String,
    }

    #[cfg(feature = "dotenvy")]
    dotenvy::dotenv().ok();
    let api_key = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");

    let client = ChatClient::new(api_key, "gemini-2.5-flash")
        .with_url("https://generativelanguage.googleapis.com/v1beta/openai/");

    let result: CapitalCity = client.chat("What is the capital of France?").await.unwrap();
    assert_eq!(result.city, "Paris");
}

#[cfg(test)]
#[tokio::test]
#[ignore]
async fn gemini_audio_transcription() {
    use schemars::JsonSchema;
    use serde::Deserialize;

    #[derive(Deserialize, Debug, JsonSchema)]
    #[allow(dead_code)]
    struct Transcription {
        text: String,
    }

    #[cfg(feature = "dotenvy")]
    dotenvy::dotenv().ok();
    let api_key = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");

    let client = ChatClient::new(api_key, "gemini-3-flash-preview")
        .with_url("https://generativelanguage.googleapis.com/v1beta/openai/");

    let audio_bytes = std::fs::read("test_fixtures/harvard.wav").unwrap();

    let result: Transcription = client
        .chat_with_messages(vec![ChatMessage::new(
            Role::User,
            vec![
                ChatMessageContent::InputAudio {
                    input_audio: InputAudio::wav(audio_bytes),
                },
                ChatMessageContent::Text {
                    text: "Transcribe this audio exactly.".to_string(),
                },
            ],
        )])
        .await
        .unwrap();

    // Harvard sentences - just check a few key phrases are present
    println!("Transcription: {}", result.text);
    let text = result.text.to_lowercase();
    assert!(
        text.contains("stale smell of old beer"),
        "Expected 'stale smell of old beer' in transcription, got: {}",
        result.text
    );
}

#[cfg(test)]
#[test]
fn cache_fallbacks_preserve_insertion_order() {
    let client = ChatClient::new("unused", "new-model")
        .with_cache_fallback(ChatClient::new("unused", "oldest-model"))
        .with_cache_fallback(ChatClient::new("unused", "newer-model"));

    assert_eq!(client.cache_fallbacks.len(), 2);
    assert_eq!(client.cache_fallbacks[0].model, "oldest-model");
    assert_eq!(client.cache_fallbacks[1].model, "newer-model");
}

#[cfg(test)]
#[tokio::test]
async fn batch_fn_pairs_objects_and_uses_fallback_cache_without_network() {
    #[derive(Debug, serde::Deserialize, serde::Serialize, schemars::JsonSchema, PartialEq)]
    struct Answer {
        value: String,
    }

    let temp = tempfile::tempdir().unwrap();
    let old_cache = temp.path().join("old");
    let new_cache = temp.path().join("new");
    std::fs::create_dir_all(&old_cache).unwrap();
    std::fs::create_dir_all(&new_cache).unwrap();

    let old = ChatClient::new("unused", "old-model").with_cache_directory(&old_cache);
    let response_format = ResponseFormat::JsonSchema {
        json_schema: JsonSchemaFormat::new::<Answer>(),
    };
    let request = old.request_for_messages(
        vec![
            ChatMessage::system("return a value"),
            ChatMessage::user("alpha"),
        ],
        response_format,
    );
    old.cache_batch_response(
        &request,
        &ChatResponse {
            id: "cached".into(),
            object: "chat.completion".into(),
            created: 0,
            model: "old-model".into(),
            system_fingerprint: None,
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessageResponse {
                    role: Role::Assistant,
                    content: Some(r#"{"value":"cached answer"}"#.into()),
                    refusal: None,
                },
                logprobs: None,
                finish_reason: "stop".into(),
            }],
            usage: ChatUsage::default(),
        },
    )
    .await
    .unwrap();

    let client = ChatClient::new("unused", "new-model")
        .with_cache_directory(&new_cache)
        .with_cache_fallback(old)
        .with_cached_only();
    let items = vec!["alpha".to_string()];
    let results = client
        .batch_chat_with_system_prompt_fn::<_, _, Answer>(
            "return a value",
            &items,
            |item| item.clone(),
            |_| {},
        )
        .await
        .unwrap();

    assert!(std::ptr::eq(results[0].0, &items[0]));
    assert_eq!(
        results[0].1.as_ref().unwrap(),
        &Answer {
            value: "cached answer".into()
        }
    );
}

#[cfg(test)]
#[tokio::test]
async fn batch_prefers_current_model_cache_over_fallback_cache() {
    #[derive(Debug, serde::Deserialize, serde::Serialize, schemars::JsonSchema, PartialEq)]
    struct Answer {
        value: String,
    }

    async fn cache_answer(client: &ChatClient, value: &str) {
        let request = client.request_for_messages(
            vec![
                ChatMessage::system("return a value"),
                ChatMessage::user("alpha"),
            ],
            ResponseFormat::JsonSchema {
                json_schema: JsonSchemaFormat::new::<Answer>(),
            },
        );
        client
            .cache_batch_response(
                &request,
                &ChatResponse {
                    id: "cached".into(),
                    object: "chat.completion".into(),
                    created: 0,
                    model: client.model.clone(),
                    system_fingerprint: None,
                    choices: vec![ChatChoice {
                        index: 0,
                        message: ChatMessageResponse {
                            role: Role::Assistant,
                            content: Some(format!(r#"{{"value":"{value}"}}"#)),
                            refusal: None,
                        },
                        logprobs: None,
                        finish_reason: "stop".into(),
                    }],
                    usage: ChatUsage::default(),
                },
            )
            .await
            .unwrap();
    }

    let temp = tempfile::tempdir().unwrap();
    let current = ChatClient::new("unused", "current-model")
        .with_cache_directory(temp.path().join("current"));
    let fallback = ChatClient::new("unused", "fallback-model")
        .with_cache_directory(temp.path().join("fallback"));
    cache_answer(&current, "current").await;
    cache_answer(&fallback, "fallback").await;

    let client = current.with_cache_fallback(fallback).with_cached_only();
    let items = vec!["alpha".to_string()];
    let results = client
        .batch_chat_with_system_prompt_fn::<_, _, Answer>(
            "return a value",
            &items,
            Clone::clone,
            |_| {},
        )
        .await
        .unwrap();

    assert_eq!(results[0].1.as_ref().unwrap().value, "current");
}

#[cfg(test)]
#[tokio::test]
async fn cached_only_batch_returns_hits_and_per_item_misses() {
    #[derive(Debug, serde::Deserialize, serde::Serialize, schemars::JsonSchema, PartialEq)]
    struct Answer {
        value: String,
    }

    let temp = tempfile::tempdir().unwrap();
    let client = ChatClient::new("unused", "model").with_cache_directory(temp.path());
    let request = client.request_for_messages(
        vec![
            ChatMessage::system("return a value"),
            ChatMessage::user("alpha"),
        ],
        ResponseFormat::JsonSchema {
            json_schema: JsonSchemaFormat::new::<Answer>(),
        },
    );
    client
        .cache_batch_response(
            &request,
            &ChatResponse {
                id: "cached".into(),
                object: "chat.completion".into(),
                created: 0,
                model: client.model.clone(),
                system_fingerprint: None,
                choices: vec![ChatChoice {
                    index: 0,
                    message: ChatMessageResponse {
                        role: Role::Assistant,
                        content: Some(r#"{"value":"cached"}"#.into()),
                        refusal: None,
                    },
                    logprobs: None,
                    finish_reason: "stop".into(),
                }],
                usage: ChatUsage::default(),
            },
        )
        .await
        .unwrap();

    let client = client.with_cached_only();
    let items = vec!["alpha".to_string(), "beta".to_string()];
    let results = client
        .batch_chat_with_system_prompt_fn::<_, _, Answer>(
            "return a value",
            &items,
            Clone::clone,
            |_| {},
        )
        .await
        .expect("a cached-only batch reports misses per item, not as a batch failure");

    assert_eq!(results[0].1.as_ref().unwrap().value, "cached");
    assert!(matches!(results[1].1, Err(IndividualChatError::CacheMiss)));
}

#[cfg(test)]
mod retry_tests {
    use super::*;

    fn api_error(r#type: &str, code: Option<&str>) -> ChatError {
        ChatError::ApiError(
            OpenAiError {
                r#type: r#type.to_owned(),
                code: code.map(str::to_owned),
                message: "message".to_owned(),
                param: None,
            },
            "request".to_owned(),
        )
    }

    #[test]
    fn capacity_errors_are_retried_and_wait_longer_than_a_reconnect() {
        // The flex tier's "out of capacity" answer is the case this exists for.
        let flex = api_error("resource_unavailable", Some("flex_unavailable"));
        let delay = retry_delay(&flex, 1).expect("flex_unavailable is retryable");
        assert_eq!(delay.as_secs(), 15);
        // Backoff grows, because capacity frees up over minutes.
        assert_eq!(retry_delay(&flex, 2).unwrap().as_secs(), 30);
        assert_eq!(retry_delay(&flex, 3).unwrap().as_secs(), 60);
    }

    #[test]
    fn rate_limits_and_server_faults_are_retried() {
        for error in [
            api_error("rate_limit_error", Some("rate_limit_exceeded")),
            api_error("server_error", None),
            api_error("resource_unavailable", None),
        ] {
            assert!(retry_delay(&error, 1).is_some(), "should retry: {error:?}");
        }
    }

    #[test]
    fn a_bad_request_is_never_retried() {
        // A request the API rejects on its merits fails the same way forever.
        for error in [
            api_error("invalid_request_error", Some("context_length_exceeded")),
            api_error("invalid_request_error", None),
        ] {
            assert!(
                retry_delay(&error, 1).is_none(),
                "should not retry: {error:?}"
            );
        }
        assert!(retry_delay(&ChatError::NoChoices, 1).is_none());
    }
    // A scripted HTTP server also asserts that no unexpected live requests,
    // duplicate submissions, or output-file retries occur.
    async fn server(
        steps: Vec<(&'static str, String)>,
    ) -> (url::Url, tokio::task::JoinHandle<Vec<String>>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/v1/", listener.local_addr().unwrap())
            .parse()
            .unwrap();
        let task = tokio::spawn(async move {
            let mut requests = Vec::new();
            for (expected, body) in steps {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut request = Vec::new();
                loop {
                    let mut buf = [0; 4096];
                    let n = socket.read(&mut buf).await.unwrap();
                    assert_ne!(n, 0);
                    request.extend_from_slice(&buf[..n]);
                    if let Some(end) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                        let headers = String::from_utf8_lossy(&request[..end]);
                        let len = headers
                            .lines()
                            .find_map(|line| {
                                line.to_ascii_lowercase()
                                    .strip_prefix("content-length: ")
                                    .map(|s| s.parse::<usize>().unwrap())
                            })
                            .unwrap_or(0);
                        if request.len() >= end + 4 + len {
                            break;
                        }
                    }
                }
                let request = String::from_utf8_lossy(&request).into_owned();
                assert!(
                    request.starts_with(expected),
                    "expected {expected}, got {request}"
                );
                requests.push(request);
                socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes()).await.unwrap();
            }
            requests
        });
        (url, task)
    }

    fn batch_json(status: &str, output: Option<&str>, hash: &str) -> serde_json::Value {
        serde_json::json!({
            "id": "batch-test", "object": "batch", "endpoint": "/v1/chat/completions",
            "input_file_id": "input", "completion_window": "24h", "status": status,
            "output_file_id": output, "created_at": 0,
            "request_counts": {"total": 2, "completed": 1, "failed": 0},
            "metadata": {"request_hash": hash}
        })
    }

    fn response_json(content: &str) -> serde_json::Value {
        serde_json::json!({
            "id": "response", "object": "chat.completion", "created": 0, "model": "gpt-4o-mini",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        })
    }

    fn result_line(id: String, content: &str) -> String {
        serde_json::json!({"id": "item", "custom_id": id,
            "response": {"status_code": 200, "request_id": "req", "body": response_json(content)}
        })
        .to_string()
    }

    #[tokio::test]
    async fn wait_returns_cancelled_and_expired_batches_and_empty_results() {
        for status in ["cancelled", "expired", "completed"] {
            let (url, task) = server(vec![(
                "GET /v1/batches/batch-test ",
                batch_json(status, None, "0").to_string(),
            )])
            .await;
            let mut chat = ChatClient::new("unused", "gpt-4o-mini");
            chat.base_url = url;
            let client = BatchClient::from(&chat);
            let mut polls = 0;
            let batch = client
                .wait_for_batch("batch-test", |_| polls += 1)
                .await
                .unwrap();
            assert_eq!(polls, 1);
            assert!(batch.status.is_terminal());
            assert!(client.get_batch_results(&batch).await.unwrap().is_empty());
            task.await.unwrap();
        }
    }

    #[tokio::test]
    async fn wait_rides_out_a_rate_limited_status_poll() {
        let rate_limited = serde_json::json!({"error": {
            "message": "You've exceeded the 100 request(s) every 1 minute(s) rate limit",
            "type": "invalid_request_error", "param": null, "code": "rate_limit_exceeded"
        }})
        .to_string();
        let (url, task) = server(vec![
            ("GET /v1/batches/batch-test ", rate_limited),
            (
                "GET /v1/batches/batch-test ",
                batch_json("completed", None, "0").to_string(),
            ),
        ])
        .await;
        let mut chat = ChatClient::new("unused", "gpt-4o-mini");
        chat.base_url = url;
        let batch = BatchClient::from(&chat)
            .wait_for_batch("batch-test", |_| {})
            .await
            .unwrap();
        assert!(batch.status.is_terminal());
        task.await.unwrap();
    }

    #[tokio::test]
    async fn partial_batch_is_harvested_cached_and_only_remaining_request_goes_live() {
        // Exercise terminal harvesting and cancellation of still-running work.
        for status in ["cancelled", "expired", "in_progress", "cancelling"] {
            let cache = tempfile::tempdir().unwrap();
            let mut client = ChatClient::new("unused", "gpt-4o-mini")
                .with_cache_directory(cache.path())
                .with_no_batch();
            let prompts = vec![
                (vec![ChatMessage::user("alpha")], ResponseFormat::Text),
                (vec![ChatMessage::user("beta")], ResponseFormat::Text),
                (vec![ChatMessage::user("alpha")], ResponseFormat::Text),
            ];
            let requests = prompts
                .iter()
                .enumerate()
                .map(|(i, (m, f))| (i, client.request_for_messages(m.clone(), f.clone())))
                .collect::<Vec<_>>();
            let hash = ChatClient::batch_hash(&requests);
            let batch = batch_json(status, Some("output"), &hash);
            let mut steps = vec![(
                "GET /v1/batches?limit=100 ",
                serde_json::json!({"object":"list", "data":[batch], "has_more":false}).to_string(),
            )];
            if status == "in_progress" {
                steps.push((
                    "POST /v1/batches/batch-test/cancel ",
                    batch_json("cancelling", None, &hash).to_string(),
                ));
            }
            if matches!(status, "in_progress" | "cancelling") {
                steps.push((
                    "GET /v1/batches/batch-test ",
                    batch_json("cancelled", Some("output"), &hash).to_string(),
                ));
            }
            steps.push((
                "GET /v1/files/output/content ",
                result_line(ChatClient::batch_custom_id(&requests[0].1), "harvested"),
            ));
            steps.push((
                "POST /v1/chat/completions ",
                response_json("live").to_string(),
            ));
            let (url, task) = server(steps).await;
            client.base_url = url;
            let results = client
                .batch_chat_with_messages_raw(prompts.clone(), |_| {})
                .await
                .unwrap();
            assert_eq!(
                results.into_iter().collect::<Result<Vec<_>, _>>().unwrap(),
                ["harvested", "live", "harvested"]
            );
            assert_eq!(client.batch_usage().total_tokens, 15);
            assert_eq!(client.usage.read().unwrap().total_tokens, 15);
            let network = task.await.unwrap();
            let live = network.last().unwrap();
            assert!(live.contains("beta"));
            assert!(!live.contains("alpha"));
            // A new client proves persistence, not just the in-memory cache.
            let cached = ChatClient::new("unused", "gpt-4o-mini")
                .with_cache_directory(cache.path())
                .with_cached_only();
            let results = cached
                .batch_chat_with_messages_raw(prompts, |_| {})
                .await
                .unwrap();
            assert_eq!(
                results.into_iter().collect::<Result<Vec<_>, _>>().unwrap(),
                ["harvested", "live", "harvested"]
            );
        }
    }
    #[tokio::test]
    async fn harvest_reports_item_errors_refusals_and_mapping_failures_per_item() {
        let mut client = ChatClient::new("unused", "gpt-4o-mini");
        let mut misses = (0..6)
            .map(|i| {
                (
                    i,
                    client.request_for_messages(
                        vec![ChatMessage::user(i.to_string())],
                        ResponseFormat::Text,
                    ),
                )
            })
            .collect::<Vec<_>>();
        let requests = misses.iter().map(|(_, r)| r.clone()).collect::<Vec<_>>();
        let id = |i: usize| ChatClient::batch_custom_id(&requests[i]);
        let mut refusal = response_json("unused");
        refusal["choices"][0]["message"] = serde_json::json!({"role":"assistant", "refusal":"no"});
        let body_item = |id: String, body: serde_json::Value| {
            serde_json::json!({
                "id":"item", "custom_id":id,
                "response":{"status_code":200, "request_id":"req", "body":body}
            })
            .to_string()
        };
        let lines = [
            result_line(id(0), "good"),
            serde_json::json!({"id":"item", "custom_id":id(1), "error":{"code":"failed", "message":"failed"}}).to_string(),
            body_item(id(2), serde_json::json!({"error":{"type":"invalid_request_error", "message":"bad"}})),
            body_item(id(3), refusal),
            result_line(id(4), "bad mapping"),
            body_item(id(5), serde_json::json!({"malformed":"response"})),
        ].join("\n");
        // All IDs are present: a completed batch must not retry this file.
        let reject_bad_mapping = |s: String| {
            if s == "bad mapping" {
                Err(IndividualChatError::Other(s))
            } else {
                Ok(s)
            }
        };
        let (url, task) = server(vec![("GET /v1/files/output/content ", lines.clone())]).await;
        client.base_url = url;
        let batch = serde_json::from_value(batch_json("completed", Some("output"), "0")).unwrap();
        let mut output = (0..6).map(|_| None).collect::<Vec<_>>();
        client
            .harvest_batch(
                &BatchClient::from(&client),
                &batch,
                false,
                &mut misses,
                &mut output,
                &reject_bad_mapping,
            )
            .await
            .unwrap();
        assert_eq!(output[0].take().unwrap().unwrap(), "good");
        // Every ID the file mentions is resolved: failures are per-item
        // errors, not misses, and none of them is cached.
        assert!(misses.is_empty());
        for slot in &mut output[1..] {
            assert!(slot.take().unwrap().is_err());
        }
        assert_eq!(client.batch_usage().total_tokens, 45);
        for request in &requests[1..] {
            assert!(client.cached_batch_content(request).await.is_none());
        }
        task.await.unwrap();

        // The same file from a batch found on a later run: failures are
        // retried instead, so they stay misses.
        let (url, task) = server(vec![("GET /v1/files/output/content ", lines)]).await;
        client.base_url = url;
        let mut misses = requests.iter().cloned().enumerate().collect::<Vec<_>>();
        let mut output = (0..6).map(|_| None).collect::<Vec<_>>();
        client
            .harvest_batch(
                &BatchClient::from(&client),
                &batch,
                true,
                &mut misses,
                &mut output,
                &reject_bad_mapping,
            )
            .await
            .unwrap();
        assert_eq!(output[0].take().unwrap().unwrap(), "good");
        assert_eq!(
            misses.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
            [1, 2, 3, 4, 5]
        );
        task.await.unwrap();
    }

    #[tokio::test]
    async fn items_the_api_failed_to_run_are_read_from_the_error_file() {
        let mut client = ChatClient::new("unused", "gpt-4o-mini");
        let mut misses = (0..2)
            .map(|i| {
                (
                    i,
                    client.request_for_messages(
                        vec![ChatMessage::user(i.to_string())],
                        ResponseFormat::Text,
                    ),
                )
            })
            .collect::<Vec<_>>();
        let id = |i: usize| ChatClient::batch_custom_id(&misses[i].1);
        // As OpenAI wrote it for one request of a 30,899-request batch.
        let failed = serde_json::json!({"id":"item", "custom_id":id(1), "error":null,
            "response":{"status_code":500, "request_id":"",
                "body":{"error":{"message":"BatchAPI failed to execute task in batch"}}}
        })
        .to_string();
        // One download per file: the item is not missing, so no re-fetch.
        let (url, task) = server(vec![
            ("GET /v1/files/output/content ", result_line(id(0), "good")),
            ("GET /v1/files/errors/content ", failed),
        ])
        .await;
        client.base_url = url;
        let mut batch = batch_json("completed", Some("output"), "0");
        batch["error_file_id"] = "errors".into();
        let batch = serde_json::from_value(batch).unwrap();
        let mut output = (0..2).map(|_| None).collect::<Vec<_>>();
        client
            .harvest_batch(
                &BatchClient::from(&client),
                &batch,
                false,
                &mut misses,
                &mut output,
                &|s: String| Ok::<_, IndividualChatError>(s),
            )
            .await
            .unwrap();
        assert!(misses.is_empty());
        assert_eq!(output[0].take().unwrap().unwrap(), "good");
        assert!(output[1].take().unwrap().is_err());
        task.await.unwrap();
    }

    #[tokio::test]
    async fn replacement_batch_contains_only_missing_subset_and_its_hash() {
        let mut client = ChatClient::new("unused", "gpt-4o-mini");
        let prompts = ["alpha", "beta"]
            .map(|s| (vec![ChatMessage::user(s)], ResponseFormat::Text))
            .to_vec();
        let requests = prompts
            .iter()
            .enumerate()
            .map(|(i, (m, f))| (i, client.request_for_messages(m.clone(), f.clone())))
            .collect::<Vec<_>>();
        let old = batch_json(
            "cancelled",
            Some("old-output"),
            &ChatClient::batch_hash(&requests),
        );
        let remaining_hash = ChatClient::batch_hash(&requests[1..]);
        let new = batch_json("completed", Some("new-output"), &remaining_hash).to_string();
        let (url, task) = server(vec![
            ("GET /v1/batches?limit=100 ", serde_json::json!({"object":"list", "data":[old], "has_more":false}).to_string()),
            ("GET /v1/files/old-output/content ", result_line(ChatClient::batch_custom_id(&requests[0].1), "old")),
            ("POST /v1/files ", serde_json::json!({"id":"input", "object":"file", "bytes":1, "created_at":0, "filename":"batch", "purpose":"batch"}).to_string()),
            ("POST /v1/batches ", new.clone()),
            ("GET /v1/batches/batch-test ", new),
            ("GET /v1/files/new-output/content ", result_line(ChatClient::batch_custom_id(&requests[1].1), "new")),
        ]).await;
        client.base_url = url;
        let values = client
            .batch_chat_with_messages_raw(prompts, |_| {})
            .await
            .unwrap();
        assert_eq!(
            values.into_iter().collect::<Result<Vec<_>, _>>().unwrap(),
            ["old", "new"]
        );
        let network = task.await.unwrap();
        assert!(network[2].contains("beta"));
        assert!(!network[2].contains("alpha"));
        assert!(network[3].contains(&remaining_hash));
        assert_eq!(client.batch_usage().total_tokens, 30);
    }

    #[tokio::test]
    async fn unreadable_output_file_of_an_old_batch_is_a_miss_not_an_error() {
        let mut client = ChatClient::new("unused", "gpt-4o-mini").with_no_batch();
        let prompts = vec![(vec![ChatMessage::user("alpha")], ResponseFormat::Text)];
        let requests = vec![(
            0,
            client.request_for_messages(prompts[0].0.clone(), ResponseFormat::Text),
        )];
        let batch = batch_json(
            "completed",
            Some("gone"),
            &ChatClient::batch_hash(&requests),
        );
        let (url, task) = server(vec![
            (
                "GET /v1/batches?limit=100 ",
                serde_json::json!({"object":"list", "data":[batch], "has_more":false}).to_string(),
            ),
            (
                "GET /v1/files/gone/content ",
                "not a result line".to_owned(),
            ),
            (
                "POST /v1/chat/completions ",
                response_json("live").to_string(),
            ),
        ])
        .await;
        client.base_url = url;
        let results = client
            .batch_chat_with_messages_raw(prompts, |_| {})
            .await
            .unwrap();
        assert_eq!(
            results.into_iter().collect::<Result<Vec<_>, _>>().unwrap(),
            ["live"]
        );
        task.await.unwrap();
    }
}
