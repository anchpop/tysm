//! Batch API for processing large numbers of requests asynchronously.
//!
//! The Batch API allows you to send asynchronous groups of requests with 50% lower costs,
//! a separate pool of significantly higher rate limits, and a clear 24-hour turnaround time.
//!
//! See the examples/ for more information.

use log::{debug, info, warn};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::io::Write;

use std::time::Duration;
use thiserror::Error;
use tokio::time::sleep;

use crate::chat_completions::{ChatClient, ChatRequest};
use crate::files::{FilePurpose, FilesClient, FilesError};
use crate::utils::remove_trailing_slash;
use crate::OpenAiError;

/// A client for batching requests to the OpenAI API.
pub struct BatchClient {
    /// The API key to use for the ChatGPT API.
    pub api_key: String,
    /// The URL of the ChatGPT API. Customize this if you are using a custom API that is compatible with OpenAI's.
    pub base_url: url::Url,
    /// The subpath to the batches endpoint. By default, this is `batches`.
    pub batches_path: String,
    /// the endpoint whose calls we want to batch
    pub endpoint: String,
    /// The model to use for the ChatGPT API.
    pub model: String,
    /// The client to use for file operations.
    pub files_client: FilesClient,
    /// Shared HTTP client with connection pooling
    pub http_client: Client,
}

impl From<&ChatClient> for BatchClient {
    fn from(client: &ChatClient) -> Self {
        Self {
            api_key: client.api_key.clone(),
            base_url: client.base_url.clone(),
            batches_path: "batches/".to_string(),
            endpoint: "/v1/chat/completions".to_string(),
            model: client.model.clone(),
            files_client: FilesClient::from(client),
            http_client: client.http_client.clone(),
        }
    }
}

/// Errors that can occur when uploading a batch file.
#[derive(Error, Debug)]
pub enum UploadBatchFileError {
    /// An error occurred when uploading the file to the API.
    #[error("Error uploading file")]
    FileUploadError(#[from] FilesError),
}

/// Errors that can occur when creating a batch.
#[derive(Error, Debug)]
pub enum CreateBatchError {
    /// An error occurred when sending the request to the API.
    #[error("Error sending request to the API")]
    RequestError(#[from] reqwest::Error),

    /// An error occurred when deserializing JSON.
    #[error("JSON error when parsing {1}")]
    JsonParseError(#[source] serde_json::Error, String),

    /// An error occurred with the OpenAI API.
    #[error("OpenAI API error: {0}")]
    OpenAiError(#[from] OpenAiError),
}

/// Errors that can occur when getting the status of a batch.
#[derive(Error, Debug)]
pub enum GetBatchStatusError {
    /// An error occurred when sending the request to the API.
    #[error("Error sending request to the API")]
    RequestError(#[from] reqwest::Error),

    /// An error occurred when deserializing JSON.
    #[error("JSON error when parsing {1}")]
    JsonParseError(#[source] serde_json::Error, String),

    /// An error occurred with the OpenAI API.
    #[error("OpenAI API error: {0}")]
    OpenAiError(#[from] OpenAiError),
}

/// How many *consecutive* failed status polls end a wait. Isolated failures
/// reset the count, so this bounds only a sustained outage — roughly 25
/// minutes with the backoff below, after which the batch is left running for
/// a later call to resume.
const MAX_CONSECUTIVE_POLL_FAILURES: u32 = 10;

/// Errors that can occur when waiting for a batch to complete.
#[derive(Error, Debug)]
pub enum WaitForBatchError {
    /// An error occurred when getting the batch status.
    #[error("Error getting batch status")]
    GetBatchStatusError(#[from] GetBatchStatusError),

    /// The batch failed.
    #[error("Batch {id} failed: {error}")]
    BatchFailed {
        /// The ID of the batch.
        id: String,
        /// The error message.
        error: String,
    },
    /// Timeout waiting for batch to complete.
    #[error("Timeout waiting for batch to complete: {0}")]
    BatchTimeout(String),
}

/// Errors that can occur when getting the results of a batch.
#[derive(Error, Debug)]
pub enum GetBatchResultsError {
    /// The batch is not completed.
    #[error("Batch is not completed: {0}")]
    BatchNotCompleted(BatchStatus),

    /// The batch has no output file.
    #[error("Batch has no output file")]
    BatchNoOutputFile(String),

    /// An error occurred when downloading the output file.
    #[error("File error: {0}")]
    DownloadFileError(#[from] FilesError),

    /// An error occurred when deserializing JSON.
    #[error("JSON error when parsing {1}")]
    JsonParseError(#[source] serde_json::Error, String),
}

/// Errors that can occur when listing batches.
#[derive(Error, Debug)]
pub enum CancelBatchError {
    /// An error occured when sending the request to the API.
    #[error("Error sending request to the API")]
    RequestError(#[from] reqwest::Error),

    /// An error occurred when deserializing JSON.
    #[error("JSON error when parsing {1}")]
    JsonParseError(#[source] serde_json::Error, String),

    /// An error occurred with the OpenAI API.
    #[error("OpenAI API error: {0}")]
    OpenAiError(#[from] OpenAiError),
}

/// Errors that can occur when listing batches.
#[derive(Error, Debug)]
pub enum ListBatchesError {
    /// An error occured when sending the request to the API.
    #[error("Error sending request to the API")]
    RequestError(#[from] reqwest::Error),

    /// An error occurred when deserializing JSON.
    #[error("JSON error when parsing {1}")]
    JsonParseError(#[source] serde_json::Error, String),

    /// An error occurred with the OpenAI API.
    #[error("OpenAI API error: {0}")]
    OpenAiError(#[from] OpenAiError),
}

/// An error body from the API. OpenAI wraps it as `{"error": {...}}`; a bare
/// error object is accepted too. Reading only the bare shape turned every API
/// error on the batch endpoints (rate limits included) into a JSON parse
/// error that no caller could recognize or retry.
fn parse_api_error(text: &str) -> Result<OpenAiError, serde_json::Error> {
    #[derive(Deserialize)]
    struct Wrapped {
        error: OpenAiError,
    }
    serde_json::from_str::<Wrapped>(text)
        .map(|wrapped| wrapped.error)
        .or_else(|_| serde_json::from_str(text))
}

/// A request item for a batch.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BatchRequestItem {
    /// A unique identifier for this request.
    pub custom_id: String,
    /// The HTTP method to use for this request.
    pub method: String,
    /// The URL to send this request to.
    pub url: String,
    /// The body of the request.
    pub body: Value,
}

/// A response item from a batch.
#[derive(Deserialize, Debug, Clone)]
pub struct BatchResponseItem {
    /// The ID of this response.
    pub id: String,
    /// The custom ID that was provided in the request.
    pub custom_id: String,
    /// The response from the API.
    pub response: Option<BatchItemResponse>,
    /// The error from the API, if any.
    pub error: Option<BatchItemError>,
}

/// A response from a batch item.
#[derive(Deserialize, Debug, Clone)]
pub struct BatchItemResponse {
    /// The HTTP status code of the response.
    pub status_code: u16,
    /// The request ID of the response.
    pub request_id: String,
    /// The body of the response.
    pub body: Value,
}

/// An error from a batch item.
#[derive(Deserialize, Debug, Clone, Error)]
#[error("Batch item error: ({code}) {message}")]
pub struct BatchItemError {
    /// The error code.
    #[serde(default)]
    pub code: String,
    /// The error message.
    pub message: String,
}

/// A batch object.
#[derive(Deserialize, Debug, Clone)]
pub struct Batch {
    /// The ID of the batch.
    pub id: String,
    /// The object type, always "batch".
    pub object: String,
    /// The endpoint that this batch is for.
    pub endpoint: String,
    /// Any errors that occurred during batch creation.
    pub errors: Option<Value>,
    /// The ID of the input file.
    pub input_file_id: String,
    /// The completion window for this batch.
    pub completion_window: String,
    /// The status of the batch.
    pub status: BatchStatus,
    /// The ID of the output file, if available.
    pub output_file_id: Option<String>,
    /// The ID of the error file, if available.
    pub error_file_id: Option<String>,
    /// When the batch was created.
    pub created_at: u64,
    /// When the batch started processing.
    pub in_progress_at: Option<u64>,
    /// When the batch expires.
    pub expires_at: Option<u64>,
    /// When the batch completed.
    pub completed_at: Option<u64>,
    /// When the batch failed.
    pub failed_at: Option<u64>,
    /// When the batch expired.
    pub expired_at: Option<u64>,
    /// The number of requests in the batch.
    pub request_counts: BatchRequestCounts,
    /// Custom metadata for the batch.
    pub metadata: Option<HashMap<String, String>>,
}

/// The status of a batch.
#[derive(Deserialize, Debug, Clone, Copy, PartialEq, Eq, Ord, PartialOrd)]
pub enum BatchStatus {
    /// the input file is being validated before the batch can begin
    #[serde(rename = "validating")]
    Validating,
    /// the input file has failed the validation process
    #[serde(rename = "failed")]
    Failed,
    /// the input file was successfully validated and the batch is currently being run
    #[serde(rename = "in_progress")]
    InProgress,
    /// the batch has completed and the results are being prepared
    #[serde(rename = "finalizing")]
    Finalizing,
    /// the batch has been completed and the results are ready
    #[serde(rename = "completed")]
    Completed,
    /// the batch was not able to be completed within the 24-hour time window
    #[serde(rename = "expired")]
    Expired,
    /// the batch is being cancelled (may take up to 10 minutes)
    #[serde(rename = "cancelling")]
    Cancelling,
    /// the batch was cancelled
    #[serde(rename = "cancelled")]
    Cancelled,
}

impl BatchStatus {
    pub(crate) fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Cancelled | Self::Expired | Self::Failed
        )
    }
}

impl std::fmt::Display for BatchStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// The number of requests in a batch.
#[derive(Deserialize, Debug, Clone)]
pub struct BatchRequestCounts {
    /// The total number of requests in the batch.
    pub total: u32,
    /// The number of completed requests in the batch.
    pub completed: u32,
    /// The number of failed requests in the batch.
    pub failed: u32,
}

/// A list of batches.
#[derive(Deserialize, Debug, Clone)]
pub struct BatchList {
    /// The list of batches.
    pub data: Vec<Batch>,
    /// The object type, always "list".
    pub object: String,
    /// Whether there are more batches to fetch.
    pub has_more: bool,
}

impl BatchRequestItem {
    /// Create a new batch request item for the chat completions API.
    ///
    /// The body is the same serialized [`ChatRequest`] a live call sends, so
    /// every option (reasoning effort, prompt-cache key, extra body fields)
    /// reaches the Batch API too. Only the service tier is dropped: a batch
    /// runs on its own tier.
    pub fn new_chat(custom_id: impl Into<String>, mut chat_request: ChatRequest) -> Self {
        chat_request.service_tier = None;
        let body = serde_json::to_value(&chat_request).expect("ChatRequest serializes");
        Self {
            custom_id: custom_id.into(),
            method: "POST".to_string(),
            url: "/v1/chat/completions".to_string(),
            body,
        }
    }

    /// Create a new batch request item for the embeddings API.
    pub fn new_embedding(
        custom_id: impl Into<String>,
        model: impl Into<String>,
        input: Vec<String>,
    ) -> Self {
        Self {
            custom_id: custom_id.into(),
            method: "POST".to_string(),
            url: "/v1/embeddings".to_string(),
            body: serde_json::json!({
                "model": model.into(),
                "input": input,
            }),
        }
    }

    /// Create a new batch request item for the completions API.
    pub fn new_completion(
        custom_id: impl Into<String>,
        model: impl Into<String>,
        prompt: impl Into<String>,
    ) -> Self {
        Self {
            custom_id: custom_id.into(),
            method: "POST".to_string(),
            url: "/v1/completions".to_string(),
            body: serde_json::json!({
                "model": model.into(),
                "prompt": prompt.into(),
                "max_tokens": 1000,
            }),
        }
    }

    /// Create a new batch request item for the responses API.
    pub fn new_response(
        custom_id: impl Into<String>,
        model: impl Into<String>,
        prompt: impl Into<String>,
    ) -> Self {
        Self {
            custom_id: custom_id.into(),
            method: "POST".to_string(),
            url: "/v1/responses".to_string(),
            body: serde_json::json!({
                "model": model.into(),
                "prompt": prompt.into(),
                "max_tokens": 1000,
            }),
        }
    }
}

impl BatchClient {
    fn batches_url(&self) -> url::Url {
        self.base_url.join(&self.batches_path).unwrap()
    }

    /// Create batch file content from a list of batch request items.
    ///
    /// Returns the serialized JSONL content as bytes.
    pub fn create_batch_content(&self, requests: &[BatchRequestItem]) -> Vec<u8> {
        let mut content = Vec::new();

        // Write each request as a JSON line
        for request in requests {
            let json = serde_json::to_string(request).unwrap(); // cannot panic
            writeln!(&mut content, "{}", json).unwrap(); // writing to memory cannot fail
        }

        content
    }

    /// Create a batch file from a list of batch request items.
    pub async fn upload_batch_file(
        &self,
        filename: impl AsRef<str>,
        requests: &[BatchRequestItem],
    ) -> Result<String, UploadBatchFileError> {
        // Create the batch content
        let content = self.create_batch_content(requests);

        // Upload the content directly
        let file_obj = self
            .files_client
            .upload_bytes(filename.as_ref(), content, FilePurpose::Batch)
            .await?;

        info!(
            "Batch file {} uploaded with ID {}",
            filename.as_ref(),
            file_obj.id
        );

        Ok(file_obj.id)
    }

    /// Create a batch from a file ID.
    pub async fn create_batch(
        &self,
        input_file_id: impl AsRef<str>,
        metadata: HashMap<String, String>,
    ) -> Result<Batch, CreateBatchError> {
        let url = remove_trailing_slash(self.batches_url());
        let response = self
            .http_client
            .post(url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({
                "input_file_id": input_file_id.as_ref(),
                "endpoint": &self.endpoint,
                "completion_window": "24h",
                "metadata": metadata,
            }))
            .send()
            .await?;

        let response_text = response.text().await?;
        let batch: Result<Batch, serde_json::Error> = serde_json::from_str(&response_text);

        match batch {
            Ok(batch) => {
                info!(
                    "Batch {} created with file id {}",
                    batch.id,
                    input_file_id.as_ref()
                );
                Ok(batch)
            }
            Err(e) => {
                // Try to parse as an OpenAI error
                let error = parse_api_error(&response_text);
                match error {
                    Ok(error) => Err(CreateBatchError::OpenAiError(error)),
                    Err(_) => Err(CreateBatchError::JsonParseError(e, response_text)),
                }
            }
        }
    }

    /// Get the status of a batch.
    pub async fn get_batch_status(&self, batch_id: &str) -> Result<Batch, GetBatchStatusError> {
        let url = self.batches_url().join(batch_id).unwrap();
        let response = self
            .http_client
            .get(url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .send()
            .await?;

        let response_text = response.text().await?;
        let batch: Result<Batch, serde_json::Error> = serde_json::from_str(&response_text);

        match batch {
            Ok(batch) => Ok(batch),
            Err(e) => {
                // Try to parse as an OpenAI error
                let error = parse_api_error(&response_text);
                match error {
                    Ok(error) => Err(GetBatchStatusError::OpenAiError(error)),
                    Err(_) => Err(GetBatchStatusError::JsonParseError(e, response_text)),
                }
            }
        }
    }

    /// Wait until a batch is completed, cancelled, or expired, calling `on_progress` after every poll.
    /// Cancelling batches are polled until their partial results are ready.
    pub async fn wait_for_batch(
        &self,
        batch_id: &str,
        mut on_progress: impl FnMut(&Batch),
    ) -> Result<Batch, WaitForBatchError> {
        let mut seconds_waited = 0;
        let mut failed_polls = 0u32;

        loop {
            let batch = match self.get_batch_status(batch_id).await {
                Ok(batch) => {
                    failed_polls = 0;
                    batch
                }
                Err(e)
                    if match &e {
                        GetBatchStatusError::RequestError(_)
                        | GetBatchStatusError::JsonParseError(..) => true,
                        GetBatchStatusError::OpenAiError(error) => {
                            crate::chat_completions::is_transient_api_error(error)
                        }
                    } =>
                {
                    // A batch can run for hours, and every status poll opens a
                    // fresh connection — so sooner or later one poll hits a
                    // dropped connection, a DNS blip, or a laptop switching
                    // networks. A gateway in front of the API can also answer
                    // mid-outage with a plain-text "upstream connect error"
                    // body, which surfaces here as a JSON parse failure rather
                    // than a transport one; it means the same thing. Neither
                    // must abort a wait that is already hours deep: the batch
                    // itself is fine on OpenAI's side. Treat both as "still
                    // waiting" and poll again, as well as API errors that only
                    // mean "not now" (rate limits, overload, server faults);
                    // any other error the API reports propagates immediately.
                    //
                    // Only *consecutive* failures count toward giving up, so
                    // isolated blips are free however long the wait runs, but
                    // an API that is genuinely down or has changed shape ends
                    // the wait in minutes instead of hanging for a day. Giving
                    // up is cheap: the batch keeps running server-side and a
                    // later call resumes it rather than resubmitting the work.
                    failed_polls += 1;
                    if failed_polls >= MAX_CONSECUTIVE_POLL_FAILURES {
                        warn!(
                            "giving up on batch {batch_id} after {failed_polls} consecutive \
                             failed status polls; the batch is still running and can be resumed"
                        );
                        return Err(e.into());
                    }
                    if seconds_waited >= 86400 {
                        return Err(WaitForBatchError::BatchTimeout(batch_id.to_string()));
                    }
                    // Back off as the failures pile up, so a struggling API is
                    // not hammered every 30 seconds while it recovers.
                    let delay = (30 * u64::from(failed_polls)).min(300);
                    let cause = std::error::Error::source(&e)
                        .map(|s| format!(": {s}"))
                        .unwrap_or_default();
                    warn!(
                        "transient error polling batch {batch_id} status \
                         ({failed_polls}/{MAX_CONSECUTIVE_POLL_FAILURES}), \
                         retrying in {delay} seconds: {e}{cause}"
                    );
                    sleep(Duration::from_secs(delay)).await;
                    seconds_waited += delay;
                    continue;
                }
                Err(e) => return Err(e.into()),
            };
            on_progress(&batch);

            match batch.status {
                BatchStatus::Completed | BatchStatus::Cancelled | BatchStatus::Expired => {
                    return Ok(batch)
                }
                BatchStatus::Failed => {
                    return Err(WaitForBatchError::BatchFailed {
                        id: batch_id.to_string(),
                        error: batch.errors.unwrap_or_default().to_string(),
                    })
                }
                BatchStatus::Cancelling
                | BatchStatus::InProgress
                | BatchStatus::Validating
                | BatchStatus::Finalizing => {
                    // Still in progress, wait and try again
                    if seconds_waited >= 86400 {
                        return Err(WaitForBatchError::BatchTimeout(batch_id.to_string()));
                    }

                    let delay = 5;
                    info!(
                        "batch {} is still in progress, waiting {} seconds",
                        batch_id, delay
                    );
                    sleep(Duration::from_secs(delay)).await;
                    seconds_waited += delay;
                }
            }
        }
    }

    /// Download available results, including partial results from cancelled or expired batches.
    /// Items the API failed to run (say, a 500 on one request) are not in the
    /// output file but in the error file, so both are read: an item missing
    /// from the output file is not necessarily missing from the batch.
    /// Terminal batches without either file have no results.
    pub async fn get_batch_results(
        &self,
        batch: &Batch,
    ) -> Result<Vec<BatchResponseItem>, GetBatchResultsError> {
        if batch.output_file_id.is_none() && !batch.status.is_terminal() {
            return Err(GetBatchResultsError::BatchNotCompleted(batch.status));
        }

        let mut results = Vec::new();
        for file_id in [&batch.output_file_id, &batch.error_file_id]
            .into_iter()
            .flatten()
        {
            let content = self.download_batch_file(batch, file_id).await?;
            debug!("Got results for batch {}: {}", batch.id, content);
            for line in content.lines() {
                let result: BatchResponseItem = serde_json::from_str(line)
                    .map_err(|e| GetBatchResultsError::JsonParseError(e, content.clone()))?;
                results.push(result);
            }
        }

        Ok(results)
    }

    /// Available results may represent hours of completed work, so a
    /// transient transport failure gets a few retries rather than bubbling up
    /// and discarding the wait.
    async fn download_batch_file(
        &self,
        batch: &Batch,
        file_id: &str,
    ) -> Result<String, GetBatchResultsError> {
        let mut attempt = 0;
        loop {
            match self.files_client.download_file(file_id).await {
                Ok(content) => return Ok(content),
                Err(FilesError::RequestError(e)) if attempt < 5 => {
                    attempt += 1;
                    let delay = 10 * attempt;
                    warn!(
                        "transient error downloading results for batch {}, \
                         retrying in {delay} seconds: {e}",
                        batch.id
                    );
                    sleep(Duration::from_secs(delay)).await;
                }
                Err(e) => return Err(e.into()),
            }
        }
    }

    /// Cancel a batch.
    pub async fn cancel_batch(&self, batch_id: &str) -> Result<Batch, CancelBatchError> {
        let response = self
            .http_client
            .post(
                self.batches_url()
                    .join(&format!("{batch_id}/cancel"))
                    .unwrap(),
            )
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .send()
            .await?;

        let response_text = response.text().await?;
        let batch: Result<Batch, serde_json::Error> = serde_json::from_str(&response_text);

        match batch {
            Ok(batch) => Ok(batch),
            Err(e) => {
                // Try to parse as an OpenAI error
                let error = parse_api_error(&response_text);
                match error {
                    Ok(error) => Err(CancelBatchError::OpenAiError(error)),
                    Err(_) => Err(CancelBatchError::JsonParseError(e, response_text)),
                }
            }
        }
    }

    /// List all batches.
    ///
    /// This method will automatically handle pagination by repeatedly calling
    /// `list_batches_limited` until all batches have been retrieved.
    pub async fn list_batches(&self) -> Result<Vec<Batch>, ListBatchesError> {
        let mut all_batches = Vec::new();
        let mut last_batch_id = None;

        loop {
            let batch_list = self.list_batch_page(last_batch_id.as_deref()).await?;

            if batch_list.data.is_empty() {
                break;
            }

            // Get the ID of the last batch for pagination
            if let Some(last_batch) = batch_list.data.last() {
                last_batch_id = Some(last_batch.id.clone());
            }

            all_batches.extend(batch_list.data);

            // If there are no more batches, break
            if !batch_list.has_more {
                break;
            }
        }

        Ok(all_batches)
    }

    /// One page of the batch list, as large as the API allows, retried while
    /// the API says to slow down.
    ///
    /// Listing walks an organization's whole batch history, and many batch
    /// calls starting at once each walk it to look for a reusable batch. At
    /// the default 20 per page that exceeds the list endpoint's 100 requests a
    /// minute; a rate limit is only the wrong moment, so it is waited out.
    async fn list_batch_page(&self, after: Option<&str>) -> Result<BatchList, ListBatchesError> {
        const PAGE: u32 = 100;
        const ATTEMPTS: u32 = 10;
        let mut attempt = 1;
        loop {
            match self.list_batches_limited(Some(PAGE), after).await {
                Err(ListBatchesError::OpenAiError(error))
                    if attempt < ATTEMPTS
                        && crate::chat_completions::is_transient_api_error(&error) =>
                {
                    let wait = Duration::from_secs(15 * u64::from(attempt));
                    warn!("listing batches: {error}; retrying in {wait:?}");
                    sleep(wait).await;
                    attempt += 1;
                }
                result => return result,
            }
        }
    }

    /// List all batches.
    async fn list_batches_limited(
        &self,
        limit: Option<u32>,
        after: Option<&str>,
    ) -> Result<BatchList, ListBatchesError> {
        let mut url = self.batches_url();

        // Add query parameters
        let mut query_params = Vec::new();
        if let Some(limit) = limit {
            query_params.push(format!("limit={}", limit));
        }
        if let Some(after) = after {
            query_params.push(format!("after={}", after));
        }
        if !query_params.is_empty() {
            url.set_query(Some(&query_params.join("&")));
        }

        let response = self
            .http_client
            .get(remove_trailing_slash(url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .send()
            .await?;

        let response_text = response.text().await?;
        let batch_list: Result<BatchList, serde_json::Error> = serde_json::from_str(&response_text);

        match batch_list {
            Ok(batch_list) => Ok(batch_list),
            Err(e) => {
                // Try to parse as an OpenAI error
                let error = parse_api_error(&response_text);
                match error {
                    Ok(error) => Err(ListBatchesError::OpenAiError(error)),
                    Err(_) => Err(ListBatchesError::JsonParseError(e, response_text)),
                }
            }
        }
    }
}

#[test]
fn test_batch_request_serialization() {
    use serde_json::json;
    let request = BatchRequestItem {
        custom_id: "request-1".to_string(),
        method: "POST".to_string(),
        url: "/v1/chat/completions".to_string(),
        body: json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello world!"}
            ],
            "max_tokens": 1000
        }),
    };

    let serialized = serde_json::to_string(&request).unwrap();
    assert!(serialized.contains("custom_id"));
    assert!(serialized.contains("request-1"));
    assert!(serialized.contains("method"));
    assert!(serialized.contains("POST"));
    assert!(serialized.contains("url"));
    assert!(serialized.contains("/v1/chat/completions"));
    assert!(serialized.contains("body"));
    assert!(serialized.contains("gpt-4o"));
    assert!(serialized.contains("helpful assistant"));
    assert!(serialized.contains("Hello world!"));
}

#[test]
fn batch_body_carries_every_live_request_option() {
    use crate::chat_completions::{ChatMessage, ResponseFormat};
    let request = ChatRequest {
        model: "gpt-6-luna".into(),
        messages: vec![ChatMessage::user("Hello world!")],
        response_format: ResponseFormat::Text,
        service_tier: Some("flex".into()),
        prompt_cache_key: Some("proofread-fra".into()),
        reasoning_effort: Some("low".into()),
        extra_body: Some(serde_json::json!({"verbosity": "low"})),
    };
    let body = BatchRequestItem::new_chat("request-1", request).body;
    assert_eq!(body["model"], "gpt-6-luna");
    assert_eq!(body["reasoning_effort"], "low");
    assert_eq!(body["prompt_cache_key"], "proofread-fra");
    assert_eq!(body["verbosity"], "low");
    assert_eq!(body["messages"][0]["content"][0]["text"], "Hello world!");
    assert!(
        body.get("service_tier").is_none(),
        "a batch runs on its own tier"
    );
}

#[test]
fn api_errors_parse_in_the_wrapped_shape_the_api_sends() {
    let body = r#"{
      "error": {
        "message": "You've exceeded the 100 request(s) every 1 minute(s) rate limit, please slow down and try again.",
        "type": "invalid_request_error",
        "param": null,
        "code": "rate_limit_exceeded"
      }
    }"#;
    let error = parse_api_error(body).unwrap();
    assert_eq!(error.code.as_deref(), Some("rate_limit_exceeded"));
    assert!(crate::chat_completions::is_transient_api_error(&error));
    let bare = r#"{"type": "server_error", "message": "oops", "code": null, "param": null}"#;
    assert_eq!(parse_api_error(bare).unwrap().r#type, "server_error");
}
