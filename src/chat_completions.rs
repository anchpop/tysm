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

use crate::batch::{BatchResponseItem, BatchStatus};
use crate::schema::OpenAiTransform;
use crate::utils::{api_key, OpenAiApiKeyError};
use crate::OpenAiError;
use log::{debug, info};

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
    /// This client's token consumption (as reported by the API). Batch requests will not affect `usage`.
    pub usage: RwLock<ChatUsage>,
    /// The directory in which to cache responses to requests
    pub cache_directory: Option<PathBuf>,

    /// The backup cache directory to check if a file is not found in the main cache directory
    pub backup_cache_directory: Option<PathBuf>,

    /// The service tier to use for requests (e.g., "flex")
    pub service_tier: Option<String>,

    /// The reasoning effort to use for requests (e.g., "low", "medium", "high")
    pub reasoning_effort: Option<String>,

    /// Extra body to be provided when making requests
    pub extra_body: Option<serde_json::Value>,

    /// Semaphore to limit the maximum number of concurrent requests
    pub semaphore: Semaphore,

    /// Shared HTTP client with connection pooling
    pub http_client: Client,

    /// If true, all uncached requests will fail with [`ChatError::CacheMiss`] instead of
    /// hitting the API. Useful for testing or offline usage.
    pub cached_only: bool,

    /// The S3-compatible bucket (e.g. Cloudflare R2) to mirror the on-disk cache to.
    /// When set together with [`cache_directory`](Self::cache_directory), the cache is
    /// pulled from the bucket on first use and pushed back via [`Self::flush_cache`].
    #[cfg(feature = "cache-sync")]
    pub cache_bucket: Option<crate::cache_sync::CacheBucket>,
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
        // Create a version of the request without service_tier for caching
        // This ensures requests with different service tiers share the same cache
        // Note: reasoning_effort IS included in the cache key since it affects output
        let mut cacheable = self.clone();
        cacheable.service_tier = None;

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
}

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
            cache_directory: None,
            backup_cache_directory: None,
            service_tier: None,
            reasoning_effort: None,
            extra_body: None,
            semaphore: Semaphore::new(100),
            http_client: crate::utils::pooled_client(),
            cached_only: false,
            #[cfg(feature = "cache-sync")]
            cache_bucket: None,
        }
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

    /// Mirror the on-disk cache to an S3-compatible bucket (e.g. Cloudflare R2).
    ///
    /// Requires [`Self::with_cache_directory`] to have been set. On the first request the
    /// local cache directory is warmed from the bucket (once per process, even across
    /// clients sharing the directory); call [`Self::flush_cache`] at the end of your run
    /// to upload new entries back.
    ///
    /// Credentials are read from the environment: `R2_ACCOUNT_ID` (or `R2_ENDPOINT`),
    /// `R2_ACCESS_KEY_ID` (or `AWS_ACCESS_KEY_ID`), and `R2_SECRET_ACCESS_KEY` (or
    /// `AWS_SECRET_ACCESS_KEY`).
    #[cfg(feature = "cache-sync")]
    pub fn with_cache_bucket(mut self, bucket: impl Into<String>) -> Self {
        if self.cache_directory.is_none() {
            panic!("with_cache_bucket requires with_cache_directory to be set first");
        }
        self.cache_bucket = Some(crate::cache_sync::CacheBucket::new(bucket));
        self
    }

    /// Set an optional key prefix within the cache bucket (default: empty).
    ///
    /// Panics if [`Self::with_cache_bucket`] has not been called first.
    #[cfg(feature = "cache-sync")]
    pub fn with_cache_bucket_prefix(mut self, prefix: impl Into<String>) -> Self {
        let prefix = prefix.into();
        let prefix = prefix.trim_matches('/').to_string();
        match &mut self.cache_bucket {
            Some(bucket) => bucket.prefix = prefix,
            None => panic!("with_cache_bucket_prefix requires with_cache_bucket to be set first"),
        }
        self
    }

    /// Warm the local cache directory from the configured bucket. Best-effort and runs at
    /// most once per process per (bucket, directory). Called automatically before each
    /// request; exposed for callers that want to warm the cache eagerly.
    #[cfg(feature = "cache-sync")]
    pub async fn ensure_cache_warmed(&self) {
        if let (Some(dir), Some(bucket)) = (&self.cache_directory, &self.cache_bucket) {
            crate::cache_sync::ensure_pulled(dir, bucket).await;
        }
    }

    /// Upload any local cache entries not yet in the bucket. Call this once at the end of
    /// a run. Idempotent and deduplicated across clients sharing a cache directory.
    ///
    /// Returns the number of objects transferred. No-op (returns `Ok` with zeroed stats)
    /// if no cache bucket or directory is configured.
    #[cfg(feature = "cache-sync")]
    pub async fn flush_cache(
        &self,
    ) -> Result<crate::cache_sync::CacheSyncStats, crate::cache_sync::CacheSyncError> {
        match (&self.cache_directory, &self.cache_bucket) {
            (Some(dir), Some(bucket)) => crate::cache_sync::flush(dir, bucket).await,
            _ => Ok(crate::cache_sync::CacheSyncStats::default()),
        }
    }

    /// Set the service tier for requests (e.g., "flex")
    pub fn with_service_tier(mut self, service_tier: impl Into<String>) -> Self {
        self.service_tier = Some(service_tier.into());
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

    /// If set, all uncached requests will fail with [`ChatError::CacheMiss`] instead of
    /// hitting the API. Useful for testing or offline usage.
    pub fn with_cached_only(self) -> Self {
        Self {
            cached_only: true,
            ..self
        }
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
    /// #     ChatClient {
    /// #         base_url: my_api,
    /// #         ..ChatClient::from_env("gpt-4o").unwrap()
    /// #     }
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
    /// #     ChatClient {
    /// #         base_url: my_api,
    /// #         ..ChatClient::from_env("gpt-4o").unwrap()
    /// #     }
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
    /// #     ChatClient {
    /// #         base_url: my_api,
    /// #         ..ChatClient::from_env("gpt-4o").unwrap()
    /// #     }
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
        #[cfg(feature = "cache-sync")]
        self.ensure_cache_warmed().await;

        let chat_request = ChatRequest {
            model: self.model.clone(),
            messages,
            response_format,
            service_tier: self.service_tier.clone(),
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

        let chat_response = if let Some(cached_response) = self
            .chat_cached(&chat_request, process_result.clone())
            .await
        {
            debug!("Using cached response");
            let (result, _usage) = cached_response?;
            result
        } else {
            if self.cached_only {
                return Err(ChatError::CacheMiss);
            }
            let chat_response = self.chat_uncached(&chat_request).await?;
            let (result, usage) = process_result(chat_response.clone())?;
            *self.usage.write().unwrap() += usage;

            // cache the response
            {
                let chat_request_cache_key = chat_request.cache_key();
                let chat_request = serde_json::to_string(&chat_request)
                    .map_err(|e| ChatError::JsonSerializeError(e, chat_request.clone()))?;

                if let Some(cache_directory) = &self.cache_directory {
                    // Compress the response with zstd before writing to disk
                    let compressed = zstd::encode_all(chat_response.as_bytes(), 3)?;
                    crate::utils::write_to_cache_dir(
                        cache_directory,
                        &chat_request_cache_key,
                        &compressed,
                    )
                    .await?;
                }

                self.lru.insert(chat_request, chat_response.clone());
            }
            result
        };

        Ok(chat_response)
    }

    /// Send chat messages to the batch API and deserialize the responses into the given type.
    ///
    /// This goes through the batch API, which is cheaper and has higher ratelimits, but is much higher-latency. The responses to the batch API stick around in OpenAI's servers for some time, and before starting a new batch request, `tysm` will automatically check if that same request has been made before (and reuse it if so).
    pub async fn batch_chat<T: DeserializeOwned + JsonSchema>(
        &self,
        prompts: Vec<impl Into<String>>,
    ) -> Result<Vec<Result<T, IndividualChatError>>, BatchChatError> {
        self.batch_chat_with_system_prompt("", prompts).await
    }

    /// Send a batch of chat messages to the API and deserialize the responses into the given type.
    /// The first argument, the system prompt, is used to tell the AI how to behave during the conversations.
    ///
    /// This goes through the batch API, which is cheaper and has higher ratelimits, but is much higher-latency. The responses to the batch API stick around in OpenAI's servers for some time, and before starting a new batch request, `tysm` will automatically check if that same request has been made before (and reuse it if so).
    pub async fn batch_chat_with_system_prompt<T: DeserializeOwned + JsonSchema>(
        &self,
        system_prompt: impl Into<String> + Clone,
        prompts: Vec<impl Into<String>>,
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

        self.batch_chat_with_messages(prompts).await
    }

    /// Send a batch of sequences of chat messages to the API and deserialize the responses into the given type.
    /// This is useful for more advanced use cases like chatbots, multi-turn conversations, or when you need to use [Vision](https://platform.openai.com/docs/guides/vision).
    ///
    /// This goes through the batch API, which is cheaper and has higher ratelimits, but is much higher-latency. The responses to the batch API stick around in OpenAI's servers for some time, and before starting a new batch request, `tysm` will automatically check if that same request has been made before (and reuse it if so).
    pub async fn batch_chat_with_messages<T: DeserializeOwned + JsonSchema>(
        &self,
        messages: Vec<Vec<ChatMessage>>,
    ) -> Result<Vec<Result<T, IndividualChatError>>, BatchChatError> {
        let json_schema = JsonSchemaFormat::new::<T>();

        let response_format = ResponseFormat::JsonSchema {
            json_schema: json_schema.clone(),
        };

        let chat_responses = self
            .batch_chat_with_messages_raw(
                messages
                    .into_iter()
                    .map(|m| (m, response_format.clone()))
                    .collect(),
            )
            .await?;

        let chat_responses: Vec<Result<T, _>> = chat_responses
            .into_iter()
            .map(|chat_response| {
                let chat_response = chat_response?;
                Self::decode_json(&chat_response).map_err(|e| {
                    IndividualChatError::ResponseNotConformantToSchema {
                        error: e,
                        response: chat_response.trim().to_string(),
                        schema: serde_json::to_string(&json_schema.schema).unwrap(),
                    }
                })
            })
            .collect::<Vec<Result<_, IndividualChatError>>>();

        Ok(chat_responses)
    }

    /// Send a batch of sequences of chat messages to the API. It's called "chat_with_messages_raw" because it allows you to specify any response format, and doesn't attempt to deserialize the chat completion.
    ///
    /// This goes through the batch API, which is cheaper and has higher ratelimits, but is much higher-latency. The responses to the batch API stick around in OpenAI's servers for some time, and before starting a new batch request, `tysm` will automatically check if that same request has been made before (and reuse it if so).
    pub async fn batch_chat_with_messages_raw(
        &self,
        prompts: Vec<(Vec<ChatMessage>, ResponseFormat)>,
    ) -> Result<Vec<Result<String, IndividualChatError>>, BatchChatError> {
        use crate::batch::{BatchClient, BatchRequestItem};

        info!("Starting batch chat with {} prompts", prompts.len());

        let batch_client = BatchClient::from(self);

        let (custom_ids, requests) = prompts
            .into_iter()
            .map(|(messages, response_format)| {
                let request_str = format!("{messages:?}, {response_format:?}, {:?}", self.model);
                let request_hash = const_xxh3(request_str.as_bytes());
                let custom_id = format!("request-{}", request_hash);
                (
                    (custom_id.clone(), request_hash),
                    (
                        request_hash,
                        BatchRequestItem::new_chat(
                            custom_id,
                            ChatRequest {
                                model: self.model.clone(),
                                messages,
                                response_format,
                                service_tier: self.service_tier.clone(),
                                reasoning_effort: self.reasoning_effort.clone(),
                                extra_body: self.extra_body.clone(),
                            },
                        ),
                    ),
                )
            })
            .unzip::<_, _, Vec<_>, HashMap<_, _>>();
        let requests = requests.values().cloned().collect::<Vec<_>>();

        let (custom_ids, hashes) = custom_ids.into_iter().unzip::<_, _, Vec<_>, HashSet<_>>();
        let request_hash = hashes
            .into_iter()
            .fold(0, |acc: u64, hash: u64| acc.wrapping_add(hash));

        // list the batches to see if we already have a batch for this request
        let all_batches = batch_client.list_batches().await?;
        let batch = all_batches
            .iter()
            .find(|batch| {
                let still_active = [
                    BatchStatus::Completed,
                    BatchStatus::InProgress,
                    BatchStatus::Validating,
                    BatchStatus::Finalizing,
                ]
                .contains(&batch.status);
                if !still_active {
                    return false;
                }

                batch
                    .metadata
                    .as_ref()
                    .cloned()
                    .unwrap_or_default()
                    .get("request_hash")
                    .map(|s| s == &request_hash.to_string())
                    .unwrap_or_default()
            })
            .cloned();

        // If the batch already exists, use it. Otherwise, create a new one.
        let batch = if let Some(batch) = batch {
            info!("Reusing existing batch");
            batch
        } else {
            info!("No batch with matching hash found found, creating a new one");
            // Create the batch content
            let content = batch_client.create_batch_content(&requests);

            // Upload the content directly
            let file_obj = batch_client
                .files_client
                .upload_bytes("batch_request", content, crate::files::FilePurpose::Batch)
                .await?;

            batch_client
                .create_batch(
                    file_obj.id,
                    std::collections::HashMap::from([(
                        "request_hash".to_string(),
                        request_hash.to_string(),
                    )]),
                )
                .await?
        };

        let batch = batch_client.wait_for_batch(&batch.id).await?;

        let results = batch_client.get_batch_results(&batch).await?;

        let results = results
            .into_iter()
            .map(
                |BatchResponseItem {
                     id: _,
                     custom_id,
                     response,
                     error,
                 }| {
                    if let Some(error) = error {
                        return Err(BatchChatError::BatchItemError(error));
                    }
                    // in this case, we assume that response is not None
                    let response = response.unwrap().body;
                    let response: ChatResponseOrError = serde_json::from_value(response.clone())
                        .map_err(|e| BatchChatError::ApiParseError {
                            error: e,
                            response: response.to_string(),
                        })?;

                    Ok((custom_id, response))
                },
            )
            .collect::<Result<Vec<_>, _>>()?;

        let results = results
            .into_iter()
            .map(|(custom_id, response)| match response {
                ChatResponseOrError::Response(response) => Ok((custom_id, response)),
                ChatResponseOrError::Error(error) => {
                    Err(BatchChatError::OpenAiError(error, custom_id))
                }
            })
            .collect::<Result<HashMap<_, _>, BatchChatError>>()?;

        let results = custom_ids
            .into_iter()
            .map(|custom_id| {
                results
                    .get(&custom_id)
                    .ok_or(BatchChatError::CustomIdNotFound(custom_id.clone()))
                    .and_then(|response| {
                        response
                            .choices
                            .first()
                            .ok_or(BatchChatError::BatchNoChoices(custom_id))
                    })
                    .map(|choice| {
                        choice
                            .message
                            .clone()
                            .content()
                            .map_err(IndividualChatError::Refusal)
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(results)
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
                if let Some(data) = crate::utils::read_from_cache_dir(&cache_directory, &key).await
                {
                    // If found under a different key, copy to the canonical key
                    if key != copy_as_key {
                        let _ =
                            crate::utils::write_to_cache_dir(&cache_directory, &copy_as_key, &data)
                                .await;
                    }
                    return Some(data);
                }

                // Check backup cache directory
                if let Some(backup) = &backup_cache_directory {
                    if backup.exists() {
                        if let Some(data) = crate::utils::read_from_cache_dir(backup, &key).await {
                            // Copy to main cache under the canonical key
                            let _ = crate::utils::write_to_cache_dir(
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

    /// Returns how many tokens have been used so far.
    ///
    /// Does not double-count tokens used in cached responses.
    pub fn usage(&self) -> ChatUsage {
        *self.usage.read().unwrap()
    }

    /// Attempts to compute the cost in dollars of the usage of this client.
    ///
    /// This is provided on a best-effort basis. The prices are hardcoded into
    /// the library (as OpenAI doesn't provide an API to get API pricing info),
    /// and may be out of date or unavailable for the model you're using.
    /// If you notice the prices being out of date, [please leave an issue](https://github.com/not-pizza/tysm)!
    pub fn cost(&self) -> Option<f64> {
        let usage = self.usage();
        crate::model_prices::cost(&self.model, self.service_tier.as_deref(), usage)
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
fn service_tier_excluded_from_cache_key() {
    // Create two identical requests except for service_tier
    let request1 = ChatRequest {
        model: "gpt-4o".to_string(),
        messages: vec![ChatMessage::user("test")],
        response_format: ResponseFormat::Text,
        service_tier: None,
        reasoning_effort: None,
        extra_body: None,
    };

    let request2 = ChatRequest {
        model: "gpt-4o".to_string(),
        messages: vec![ChatMessage::user("test")],
        response_format: ResponseFormat::Text,
        service_tier: Some("flex".to_string()),
        reasoning_effort: None,
        extra_body: None,
    };

    // The cache keys should be identical even though service_tier differs
    assert_eq!(request1.cache_key(), request2.cache_key());

    // Test that reasoning_effort IS included in cache key (different reasoning_effort = different cache)
    let request3 = ChatRequest {
        model: "gpt-4o".to_string(),
        messages: vec![ChatMessage::user("test")],
        response_format: ResponseFormat::Text,
        service_tier: None,
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
