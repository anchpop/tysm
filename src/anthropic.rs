//! Native Anthropic Messages and Message Batches transport.
//!
//! Wire contracts checked against platform.claude.com/docs on 2026-10-08.
//! The common chat layer owns typed validation and the response cache. This
//! adapter translates native messages into its response envelope; it never
//! routes Claude through OpenAI's compatibility endpoint.

use super::*;
use futures::{StreamExt, TryStreamExt};
use serde_json::{json, Value};
use std::time::{Duration, Instant};

/// The native API used for chat and batch requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    /// OpenAI Chat Completions and the file-based Batch API (also compatible hosts).
    OpenAi,
    /// Anthropic Messages and Message Batches.
    Anthropic,
}

impl Provider {
    /// Infer the provider: `claude-*` uses Anthropic, everything else OpenAI.
    pub fn for_model(model: &str) -> Self {
        if model.starts_with("claude-") {
            Self::Anthropic
        } else {
            Self::OpenAi
        }
    }

    pub(super) fn base_url(self) -> &'static str {
        match self {
            Self::OpenAi => "https://api.openai.com/v1/",
            Self::Anthropic => "https://api.anthropic.com/v1/",
        }
    }

    pub(super) fn http_client(self) -> Client {
        match self {
            Self::OpenAi => crate::utils::pooled_client(),
            // x-api-key is a custom header: reqwest does not strip it from
            // cross-origin redirects like it strips Authorization.
            Self::Anthropic => {
                crate::utils::pooled_client_with_redirects(reqwest::redirect::Policy::none())
            }
        }
    }

    pub(super) fn api_key_env(self) -> &'static str {
        match self {
            Self::OpenAi => "OPENAI_API_KEY",
            Self::Anthropic => "ANTHROPIC_API_KEY",
        }
    }
}

fn invalid(message: impl Into<String>) -> ChatError {
    ChatError::Provider(message.into())
}

/// Normalize only schema nodes, not arbitrary values in `enum` or `default`.
#[derive(Default)]
struct AnthropicTransform {
    unsupported: Option<String>,
}
impl Transform for AnthropicTransform {
    fn transform(&mut self, schema: &mut Schema) {
        if let Some(obj) = schema.as_object_mut() {
            obj.remove("$schema");
            let mut constraints = Vec::new();
            for key in [
                "minimum",
                "maximum",
                "exclusiveMinimum",
                "exclusiveMaximum",
                "multipleOf",
                "minLength",
                "maxLength",
                "maxItems",
                "uniqueItems",
            ] {
                if let Some(value) = obj.remove(key) {
                    constraints.push(format!("{key}: {value}"));
                }
            }
            if obj
                .get("minItems")
                .and_then(Value::as_u64)
                .is_some_and(|n| n > 1)
            {
                constraints.push(format!("minItems: {}", obj.remove("minItems").unwrap()));
            }
            if !constraints.is_empty() {
                let previous = obj.get("description").and_then(Value::as_str).unwrap_or("");
                obj.insert(
                    "description".into(),
                    json!(format!(
                        "{previous} Constraints: {}.",
                        constraints.join(", ")
                    )),
                );
            }
            if obj
                .get("additionalProperties")
                .is_some_and(|v| v.is_object() || *v == json!(true))
            {
                self.unsupported = Some("Anthropic structured outputs cannot represent maps with arbitrary keys; use a list of key/value entries".into());
            }
            if obj.get("type") == Some(&json!("object")) || obj.contains_key("properties") {
                obj.insert("additionalProperties".into(), json!(false));
            }
            if let Some(one_of) = obj.remove("oneOf") {
                obj.insert("anyOf".into(), one_of);
            }
        }
        schemars::transform::transform_subschemas(self, schema);
    }
}

pub(super) fn request_body(request: &ChatRequest) -> Result<Value, ChatError> {
    let mut system = Vec::new();
    let mut messages = Vec::new();
    for message in &request.messages {
        let mut content = Vec::new();
        for block in &message.content {
            content.push(match block {
                // Anthropic rejects empty text blocks, and `chat` sends an
                // empty system prompt.
                ChatMessageContent::Text { text } if text.is_empty() => continue,
                ChatMessageContent::Text { text } => json!({"type":"text", "text":text}),
                ChatMessageContent::ImageUrl { image } => {
                    if matches!(message.role, Role::System)
                        || matches!(image.detail, Some(ImageDetail::Low | ImageDetail::High))
                    {
                        return Err(invalid(
                            "Anthropic images cannot have a system role or OpenAI image detail",
                        ));
                    }
                    let source = if let Some(data) = image.url.strip_prefix("data:") {
                        let (media_type, data) = data
                            .split_once(";base64,")
                            .ok_or_else(|| invalid("Expected a base64 image data URL"))?;
                        json!({"type":"base64", "media_type":media_type, "data":data})
                    } else {
                        json!({"type":"url", "url":image.url})
                    };
                    json!({"type":"image", "source":source})
                }
                ChatMessageContent::InputAudio { .. } => {
                    return Err(invalid(
                        "Anthropic Messages does not accept OpenAI input_audio",
                    ))
                }
            });
        }
        if matches!(message.role, Role::System) {
            if !messages.is_empty() {
                return Err(invalid(
                    "Anthropic system prompts must precede conversation messages",
                ));
            }
            system.extend(content);
        } else {
            messages.push(json!({"role":message.role, "content":content}));
        }
    }
    let mut body = json!({"model":request.model, "messages":messages, "max_tokens":16384});
    if !system.is_empty() {
        body["system"] = json!(system);
    }
    match &request.response_format {
        ResponseFormat::JsonSchema { json_schema } => {
            let mut schema = json_schema.schema.schema.clone();
            let mut transform = AnthropicTransform::default();
            transform.transform(&mut schema);
            if let Some(error) = transform.unsupported {
                return Err(invalid(error));
            }
            body["output_config"] = json!({"format":{"type":"json_schema", "schema":schema}});
        }
        ResponseFormat::Text => {}
        ResponseFormat::JsonObject => {
            return Err(invalid(
                "Anthropic requires a JSON schema; use typed chat or ResponseFormat::JsonSchema",
            ))
        }
    }
    if let Some(effort) = &request.reasoning_effort {
        let effort = match effort.as_str() {
            "minimal" => "low",
            "low" | "medium" | "high" | "xhigh" | "max" => effort,
            _ => return Err(invalid("Anthropic effort must be minimal, low, medium, high, xhigh, or max; thinking cannot be disabled on current models")),
        };
        body["thinking"] = json!({"type":"adaptive"});
        body["output_config"]["effort"] = json!(effort);
    }
    if let Some(tier) = &request.service_tier {
        if !matches!(tier.as_str(), "auto" | "standard_only") {
            return Err(invalid("Anthropic service_tier must be auto or standard_only (OpenAI flex is not supported)"));
        }
        body["service_tier"] = json!(tier);
    }
    if request.prompt_cache_key.is_some() {
        return Err(invalid(
            "prompt_cache_key is OpenAI-only; use Anthropic cache_control via with_extra_body",
        ));
    }
    if let Some(extra) = &request.extra_body {
        let extra = extra
            .as_object()
            .ok_or_else(|| invalid("extra_body must be an object"))?;
        for (key, value) in extra {
            if matches!(
                key.as_str(),
                "model" | "messages" | "system" | "stream" | "response_format"
            ) {
                return Err(invalid(format!(
                    "extra_body cannot override {key} for Anthropic"
                )));
            }
            if key == "output_config" {
                let config = value
                    .as_object()
                    .ok_or_else(|| invalid("output_config must be an object"))?;
                for (key, value) in config {
                    if key == "format" {
                        return Err(invalid(
                            "Use response_format to configure the output schema",
                        ));
                    }
                    body["output_config"][key] = value.clone();
                }
            } else {
                body[key] = value.clone();
            }
        }
    }
    Ok(body)
}

#[derive(Deserialize)]
struct Message {
    id: String,
    model: String,
    content: Vec<Value>,
    stop_reason: String,
    usage: Usage,
}

#[derive(Deserialize)]
struct Usage {
    input_tokens: u32,
    output_tokens: u32,
    #[serde(default)]
    cache_read_input_tokens: u32,
    #[serde(default)]
    cache_creation_input_tokens: u32,
    #[serde(default)]
    cache_creation: CacheCreation,
}

#[derive(Default, Deserialize)]
struct CacheCreation {
    #[serde(default)]
    ephemeral_5m_input_tokens: u32,
    #[serde(default)]
    ephemeral_1h_input_tokens: u32,
}

impl From<Usage> for ChatUsage {
    fn from(usage: Usage) -> Self {
        let prompt =
            usage.input_tokens + usage.cache_read_input_tokens + usage.cache_creation_input_tokens;
        let hour = usage.cache_creation.ephemeral_1h_input_tokens;
        Self {
            prompt_tokens: prompt,
            completion_tokens: usage.output_tokens,
            total_tokens: prompt + usage.output_tokens,
            prompt_tokens_details: Some(PromptTokenDetails {
                cached_tokens: usage.cache_read_input_tokens,
                cache_creation_5m_tokens: usage
                    .cache_creation
                    .ephemeral_5m_input_tokens
                    .max(usage.cache_creation_input_tokens.saturating_sub(hour)),
                cache_creation_1h_tokens: hour,
            }),
            completion_tokens_details: None,
        }
    }
}

fn normalize(message: Message) -> ChatResponse {
    let content = message
        .content
        .iter()
        .filter_map(|b| (b["type"] == "text").then(|| b["text"].as_str()).flatten())
        .collect::<String>();
    ChatResponse {
        id: message.id,
        object: "message".into(),
        created: 0,
        model: message.model,
        system_fingerprint: None,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessageResponse {
                role: Role::Assistant,
                refusal: (message.stop_reason == "refusal").then(|| content.clone()),
                content: Some(content),
            },
            logprobs: None,
            finish_reason: message.stop_reason,
        }],
        usage: message.usage.into(),
    }
}

#[derive(Deserialize)]
struct MessageBatch {
    id: String,
    processing_status: String,
    request_counts: Counts,
    results_url: Option<String>,
}

#[derive(Default, Deserialize)]
struct Counts {
    processing: u32,
    succeeded: u32,
    errored: u32,
    canceled: u32,
    expired: u32,
}

impl MessageBatch {
    fn progress(&self) -> Batch {
        let c = &self.request_counts;
        Batch {
            id: self.id.clone(),
            object: "message_batch".into(),
            endpoint: "/v1/messages".into(),
            errors: None,
            input_file_id: String::new(),
            completion_window: "24h".into(),
            status: match self.processing_status.as_str() {
                "ended" => BatchStatus::Completed,
                "canceling" => BatchStatus::Cancelling,
                _ => BatchStatus::InProgress,
            },
            output_file_id: None,
            error_file_id: None,
            created_at: 0,
            in_progress_at: None,
            expires_at: None,
            completed_at: None,
            failed_at: None,
            expired_at: None,
            request_counts: crate::batch::BatchRequestCounts {
                total: c.processing + c.succeeded + c.errored + c.canceled + c.expired,
                completed: c.succeeded,
                failed: c.errored + c.canceled + c.expired,
            },
            metadata: None,
        }
    }
}

#[derive(Deserialize)]
struct ResultLine {
    custom_id: String,
    result: MessageResult,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum MessageResult {
    Succeeded { message: Message },
    Errored { error: Value },
    Canceled,
    Expired,
}
impl MessageResult {
    fn response(self) -> Result<ChatResponse, String> {
        match self {
            Self::Succeeded { message } => Ok(normalize(message)),
            Self::Errored { error } => Err(format!("Anthropic batch item error: {error}")),
            Self::Canceled => Err("Anthropic batch item canceled".into()),
            Self::Expired => Err("Anthropic batch item expired".into()),
        }
    }
}

const MAX_BATCH_REQUESTS: usize = 100_000;
// Decimal MB is conservative relative to the service's 256 MB boundary.
const MAX_BATCH_BYTES: usize = 256_000_000;

fn split_requests(
    items: Vec<Value>,
    max_count: usize,
    max_bytes: usize,
) -> Result<Vec<Value>, ChatError> {
    let overhead = b"{\"requests\":[]}".len();
    let mut chunks = Vec::new();
    let mut current = Vec::new();
    let mut bytes = overhead;
    for item in items {
        let size = serde_json::to_vec(&item).unwrap().len();
        if size + overhead > max_bytes {
            return Err(invalid(
                "A single Anthropic batch request exceeds the batch byte limit",
            ));
        }
        let comma = usize::from(!current.is_empty());
        if current.len() == max_count || bytes + comma + size > max_bytes {
            chunks.push(json!({"requests":current}));
            current = Vec::new();
            bytes = overhead;
        }
        bytes += size + usize::from(!current.is_empty());
        current.push(item);
    }
    if !current.is_empty() {
        chunks.push(json!({"requests":current}));
    }
    Ok(chunks)
}

impl ChatClient {
    pub(super) fn json_schema_format<T: JsonSchema>(&self) -> JsonSchemaFormat {
        match self.provider {
            Provider::OpenAi => JsonSchemaFormat::new::<T>(),
            Provider::Anthropic => {
                JsonSchemaFormat::from_schema(schema_for!(T), &tynm::type_name::<T>())
            }
        }
    }

    /// Select a native provider explicitly, resetting its base URL and HTTP
    /// client. Call `with_url` afterwards for a proxy. The API key is not changed.
    pub fn with_provider(mut self, provider: Provider) -> Self {
        self.provider = provider;
        self.base_url = url::Url::parse(provider.base_url()).unwrap();
        self.http_client = provider.http_client();
        self
    }

    pub(super) fn cache_key(&self, request: &ChatRequest) -> String {
        match self.provider {
            Provider::OpenAi => request.cache_key(),
            Provider::Anthropic => {
                // Include *every* input, even scheduling/cache hints. No OpenAI
                // legacy-cache fallback is allowed for this namespace. Bump the
                // adapter version if the defaults or translation change.
                let bytes = serde_json::to_vec(&(
                    &request.model,
                    &request.messages,
                    &request.response_format,
                    &request.service_tier,
                    &request.prompt_cache_key,
                    &request.reasoning_effort,
                    &request.extra_body,
                ))
                .unwrap();
                format!(
                    "tysm-anthropic-v1-{:032x}.zstd",
                    xxhash_rust::xxh3::xxh3_128(&bytes)
                )
            }
        }
    }

    async fn anthropic_http(
        &self,
        method: reqwest::Method,
        url: url::Url,
        body: Option<&Value>,
    ) -> Result<reqwest::Response, ChatError> {
        for attempt in 0..LIVE_REQUEST_ATTEMPTS {
            let mut request = self
                .http_client
                .request(method.clone(), url.clone())
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", "2023-06-01")
                .timeout(Duration::from_secs(600));
            if let Some(body) = body {
                request = request.json(body);
            }
            let response = request.send().await;
            let mut delay = Duration::from_secs(2u64 << attempt);
            let error = match response {
                Err(error) => ChatError::RequestError(error),
                Ok(response) => {
                    let status = response.status();
                    if status.is_success() {
                        return Ok(response);
                    }
                    if let Some(seconds) = response
                        .headers()
                        .get("retry-after")
                        .and_then(|v| v.to_str().ok())
                        .and_then(|v| v.parse::<u64>().ok())
                    {
                        delay = Duration::from_secs(seconds);
                    }
                    let text = response.text().await?;
                    let error = ChatError::HttpStatus {
                        status: status.as_u16(),
                        body: text,
                    };
                    if !(status.as_u16() == 429
                        || status.as_u16() == 408
                        || status.as_u16() == 409
                        || status.is_server_error())
                    {
                        return Err(error);
                    }
                    error
                }
            };
            if attempt + 1 == LIVE_REQUEST_ATTEMPTS {
                return Err(error);
            }
            warn!(
                "Anthropic request failed ({error}); retrying in {}s",
                delay.as_secs()
            );
            tokio::time::sleep(delay).await;
        }
        unreachable!()
    }

    async fn anthropic_json<T: DeserializeOwned>(
        &self,
        method: reqwest::Method,
        path: &str,
        body: Option<&Value>,
    ) -> Result<T, ChatError> {
        Ok(self
            .anthropic_http(
                method,
                self.base_url
                    .join(path)
                    .map_err(|e| invalid(e.to_string()))?,
                body,
            )
            .await?
            .json()
            .await?)
    }

    pub(super) async fn anthropic_chat(&self, request: &ChatRequest) -> Result<String, ChatError> {
        let body = request_body(request)?;
        let message: Message = self
            .anthropic_json(reqwest::Method::POST, "messages", Some(&body))
            .await?;
        let response = normalize(message);
        // Bill refusals, truncations and nonconformant outputs too. They cost
        // tokens even though the common validation layer will not cache them.
        *self.usage.write().unwrap() += response.usage;
        self.record_spend(request.service_tier.as_deref(), response.usage, 1.0);
        Ok(serde_json::to_string(&response).unwrap())
    }

    fn pending_key(&self, request: &ChatRequest) -> String {
        format!("anthropic-pending-{}", self.cache_key(request))
    }

    async fn pending_batch(&self, request: &ChatRequest) -> Option<String> {
        let key = self.pending_key(request);
        if let Some(id) = self.lru.get(&key) {
            return (!id.is_empty()).then(|| id.clone());
        }
        let bytes = crate::cache::read_from_cache_dir(self.cache_directory.as_ref()?, &key).await?;
        String::from_utf8(bytes).ok().filter(|id| !id.is_empty())
    }

    async fn set_pending(&self, request: &ChatRequest, id: &str) -> Result<(), std::io::Error> {
        let key = self.pending_key(request);
        if let Some(dir) = &self.cache_directory {
            crate::cache::write_to_cache_dir(dir, &key, id.as_bytes()).await?;
        }
        self.lru.insert(key, id.into());
        Ok(())
    }

    /// Cancel an Anthropic Message Batch. Polling/harvesting remains the
    /// responsibility of the batch call; cancellation is asynchronous.
    pub async fn cancel_anthropic_batch(&self, id: &str) -> Result<(), ChatError> {
        if self.provider != Provider::Anthropic {
            return Err(invalid("Not an Anthropic client"));
        }
        if !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') || id.is_empty() {
            return Err(invalid("Invalid Anthropic batch id"));
        }
        let _: MessageBatch = self
            .anthropic_json(
                reqwest::Method::POST,
                &format!("messages/batches/{id}/cancel"),
                None,
            )
            .await?;
        Ok(())
    }

    async fn wait_anthropic_batch(
        &self,
        id: &str,
        cancel: bool,
        progress: &mut impl FnMut(&Batch),
    ) -> Result<MessageBatch, ChatError> {
        let mut cancelling = None;
        loop {
            let batch: MessageBatch = self
                .anthropic_json(
                    reqwest::Method::GET,
                    &format!("messages/batches/{id}"),
                    None,
                )
                .await?;
            progress(&batch.progress());
            if batch.processing_status == "ended" {
                return Ok(batch);
            }
            if !matches!(
                batch.processing_status.as_str(),
                "in_progress" | "canceling"
            ) {
                return Err(invalid(format!(
                    "Unknown batch status {}",
                    batch.processing_status
                )));
            }
            // Never cancel a slow batch: queues legitimately sit without
            // visible progress for hours, and Anthropic expires them at 24h.
            if cancel && cancelling.is_none() {
                match self.cancel_anthropic_batch(id).await {
                    Ok(()) => {}
                    // The job can finish between the GET and cancellation.
                    Err(ChatError::HttpStatus {
                        status: 400 | 409, ..
                    }) => {}
                    Err(error) => return Err(error),
                }
                cancelling = Some(Instant::now());
            }
            if cancelling.is_some_and(|started| started.elapsed() > Duration::from_secs(600)) {
                return Err(invalid(format!(
                    "Batch {id} has not finished canceling; rerun to resume and harvest"
                )));
            }
            tokio::time::sleep(Duration::from_secs(5)).await;
        }
    }

    async fn anthropic_results(
        &self,
        batch: &MessageBatch,
    ) -> Result<HashMap<String, Result<ChatResponse, String>>, ChatError> {
        let Some(url) = &batch.results_url else {
            return Ok(HashMap::new());
        };
        let url = url::Url::parse(url).map_err(|e| invalid(e.to_string()))?;
        // Never forward credentials to an arbitrary origin returned in JSON.
        if url.origin() != self.base_url.origin() {
            return Err(invalid("Anthropic results_url has a different origin"));
        }
        let response = self.anthropic_http(reqwest::Method::GET, url, None).await?;
        let stream = response.bytes_stream().map_err(std::io::Error::other);
        let reader = tokio_util::io::StreamReader::new(stream);
        use tokio::io::{AsyncBufReadExt, BufReader};
        let mut lines = BufReader::new(reader).lines();
        let mut results = HashMap::new();
        while let Some(line) = lines.next_line().await? {
            if line.trim().is_empty() {
                continue;
            }
            let item: ResultLine = serde_json::from_str(&line)
                .map_err(|e| invalid(format!("Malformed Anthropic batch result: {e}")))?;
            if results
                .insert(item.custom_id, item.result.response())
                .is_some()
            {
                return Err(invalid("Duplicate custom_id in Anthropic results"));
            }
        }
        Ok(results)
    }

    pub(super) async fn anthropic_batch<T>(
        &self,
        misses: Vec<(usize, ChatRequest)>,
        mut output: Vec<Option<Result<T, IndividualChatError>>>,
        progress: &mut impl FnMut(&Batch),
        map: &impl Fn(String) -> Result<T, IndividualChatError>,
    ) -> Result<Vec<Result<T, IndividualChatError>>, BatchChatError> {
        // Deduplicate before deciding whether the small-batch threshold applies.
        let mut unique = std::collections::BTreeMap::new();
        for (_, request) in &misses {
            unique.entry(self.cache_key(request)).or_insert(request);
        }
        let live = unique.len() <= self.small_batch_threshold;
        let mut jobs: std::collections::BTreeMap<String, Vec<String>> = Default::default();
        let mut fresh = Vec::new();
        for (key, request) in &unique {
            // Validate even when resuming; invalid configuration must not hide
            // behind an unrelated old job.
            let params = request_body(request)?;
            if let Some(id) = self.pending_batch(request).await {
                jobs.entry(id).or_default().push(key.clone());
            } else {
                fresh.push(json!({"custom_id":custom_id(key), "params":params}));
            }
        }
        // Submit every chunk before waiting on any of them. Batch round trips
        // are hours, not milliseconds, so serial submission is unacceptable.
        if !live {
            for body in split_requests(fresh, MAX_BATCH_REQUESTS, MAX_BATCH_BYTES)? {
                let batch: MessageBatch = self
                    .anthropic_json(reqwest::Method::POST, "messages/batches", Some(&body))
                    .await?;
                let ids: HashSet<_> = body["requests"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|r| r["custom_id"].as_str().unwrap())
                    .collect();
                let keys: Vec<_> = unique
                    .keys()
                    .filter(|key| ids.contains(custom_id(key).as_str()))
                    .cloned()
                    .collect();
                for key in &keys {
                    self.set_pending(unique[key], &batch.id).await?;
                }
                progress(&batch.progress());
                jobs.insert(batch.id, keys);
            }
        }
        let mut responses = HashMap::new();
        for (id, keys) in jobs {
            let downloaded = match self.wait_anthropic_batch(&id, live, progress).await {
                Ok(batch) => self.anthropic_results(&batch).await,
                Err(error) => Err(error),
            };
            let mut results = match downloaded {
                Ok(results) => results,
                Err(ChatError::HttpStatus { status: 404, .. }) => {
                    // Results expire 29 days after creation. Forget the stale
                    // handle so a rerun can submit these requests again.
                    for key in keys {
                        responses.insert(
                            key,
                            Err(format!(
                                "Batch {id} is no longer available; rerun to resubmit"
                            )),
                        );
                    }
                    continue;
                }
                Err(error) => return Err(error.into()),
            };
            for key in keys {
                let result = results.remove(&custom_id(&key)).unwrap_or_else(|| {
                    Err(format!("Batch {id} omitted custom_id {}", custom_id(&key)))
                });
                if let Ok(response) = &result {
                    *self.batch_usage.write().unwrap() += response.usage;
                    self.record_spend(None, response.usage, 0.5);
                }
                responses.insert(key, result);
            }
        }
        let mut cached = HashSet::new();
        for (index, request) in &misses {
            let key = self.cache_key(request);
            if let Some(response) = responses.get(&key) {
                let value = match response {
                    Ok(response) => response_content(response).and_then(map),
                    Err(error) => Err(IndividualChatError::Other(error.clone())),
                };
                if value.is_ok() && cached.insert(key.clone()) {
                    self.cache_batch_response(request, response.as_ref().unwrap())
                        .await?;
                }
                if value.is_ok() || !live {
                    output[*index] = Some(value);
                }
            }
        }
        // Clear pending IDs only after successes are safely in the response
        // cache. Interrupted harvesting can therefore be resumed without a bill.
        for key in responses.keys() {
            self.set_pending(unique[key], "").await?;
        }
        if live {
            let remaining: Vec<_> = unique
                .iter()
                .filter(|(key, _)| !cached.contains(*key))
                .map(|(key, request)| (key.clone(), (*request).clone()))
                .collect();
            let results: Vec<_> = futures::stream::iter(remaining)
                .map(|(key, request)| async move {
                    let value = self
                        .chat_with_messages_raw_mapped(
                            request.messages.clone(),
                            request.response_format.clone(),
                            |raw| {
                                map(raw.clone()).map_err(ChatError::ChatError)?;
                                Ok(raw)
                            },
                        )
                        .await;
                    (key.clone(), value.map_err(std::sync::Arc::new))
                })
                .buffer_unordered(LIVE_CONCURRENCY)
                .collect()
                .await;
            let results: HashMap<_, _> = results.into_iter().collect();
            for (index, request) in &misses {
                if output[*index].is_some() {
                    continue;
                }
                output[*index] = Some(match &results[&self.cache_key(request)] {
                    Ok(raw) => map(raw.clone()),
                    Err(error) => Err(SharedChatError(error.clone()).into()),
                });
            }
        }
        Ok(output.into_iter().map(Option::unwrap).collect())
    }
}

#[cfg(test)]
#[path = "anthropic_tests.rs"]
mod tests;

fn custom_id(key: &str) -> String {
    // 32 ASCII hex characters: always within Anthropic's [A-Za-z0-9_-]{1,64}.
    format!("{:032x}", xxhash_rust::xxh3::xxh3_128(key.as_bytes()))
}

pub(super) fn response_content(response: &ChatResponse) -> Result<String, IndividualChatError> {
    let choice = response
        .choices
        .first()
        .ok_or_else(|| IndividualChatError::Other("Empty response".into()))?;
    if choice.finish_reason == "refusal" {
        return Err(IndividualChatError::Refusal(
            choice.message.content.clone().unwrap_or_default(),
        ));
    }
    if !matches!(choice.finish_reason.as_str(), "end_turn" | "stop_sequence") {
        return Err(IndividualChatError::Other(format!(
            "Anthropic response did not finish: {}",
            choice.finish_reason
        )));
    }
    choice
        .message
        .clone()
        .content()
        .map_err(IndividualChatError::Refusal)
}
