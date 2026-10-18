//! Files API for interacting with OpenAI's file management endpoints.
//! This module provides a client for uploading, listing, retrieving, and deleting files.

use reqwest::{multipart, Client};
use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;
use tokio::fs::File;
use tokio_util::codec::{BytesCodec, FramedRead};

use crate::{
    utils::{api_key, remove_trailing_slash, ApiKeyError},
    OpenAiError,
};

/// A client for interacting with the OpenAI Files API.
pub struct FilesClient {
    /// The API key to use for the OpenAI API.
    pub api_key: String,
    /// The base URL of the OpenAI API.
    pub base_url: url::Url,
    /// The path to the Files API.
    pub files_path: String,
    /// Shared HTTP client with connection pooling
    pub http_client: Client,
}

impl From<&crate::chat_completions::ChatClient> for FilesClient {
    fn from(client: &crate::chat_completions::ChatClient) -> Self {
        Self {
            api_key: client.api_key.clone(),
            base_url: client.base_url.clone(),
            files_path: "files/".to_string(),
            http_client: client.http_client.clone(),
        }
    }
}
/// The purpose of a file in the OpenAI API.
#[derive(Serialize, Clone, Copy)]
#[serde(rename_all = "snake_case")]
pub enum FilePurpose {
    /// For fine-tuning models
    #[serde(rename = "fine-tune")]
    FineTune,
    /// For assistants
    #[serde(rename = "assistants")]
    Assistants,
    /// For batch jobs
    #[serde(rename = "batch")]
    Batch,
    /// For user data
    #[serde(rename = "user_data")]
    UserData,
    /// For vision models
    #[serde(rename = "vision")]
    Vision,
    /// For evals
    #[serde(rename = "evals")]
    Evals,
}

impl std::fmt::Debug for FilePurpose {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FilePurpose::FineTune => write!(f, "fine-tune"),
            FilePurpose::Assistants => write!(f, "assistants"),
            FilePurpose::Batch => write!(f, "batch"),
            FilePurpose::UserData => write!(f, "user_data"),
            FilePurpose::Vision => write!(f, "vision"),
            FilePurpose::Evals => write!(f, "evals"),
        }
    }
}

impl std::fmt::Display for FilePurpose {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// A file in the OpenAI API.
#[derive(Debug, Deserialize)]
pub struct FileObject {
    /// The ID of the file.
    pub id: String,
    /// The object type, always "file".
    pub object: String,
    /// The size of the file in bytes.
    pub bytes: u64,
    /// When the file was created.
    pub created_at: u64,
    /// The name of the file.
    pub filename: String,
    /// The purpose of the file.
    pub purpose: String,
}

#[derive(Debug, Deserialize)]
enum UploadFileResponse {
    #[serde(rename = "error")]
    Error(OpenAiError),
    #[serde(untagged)]
    File(FileObject),
}

/// A list of files in the OpenAI API.
#[derive(Debug, Deserialize)]
pub struct FileList {
    /// The list of files.
    pub data: Vec<FileObject>,
    /// The object type, always "list".
    pub object: String,
}

/// Errors that can occur when interacting with the Files API.
#[derive(Error, Debug)]
pub enum FilesError {
    /// An error occurred when sending the request to the API.
    #[error("Request error: {0}")]
    RequestError(#[from] reqwest::Error),

    /// An error occurred when deserializing the response from the API.
    #[error("API {url} returned an unknown response: {response}")]
    ApiParseError {
        /// The URL of the API that returned the error.
        url: String,
        /// The response from the API.
        response: String,
        /// The error that occurred when deserializing the response.
        #[source]
        error: serde_json::Error,
    },

    /// An error occurred when deserializing the response from the API.
    #[error("API returned an error response")]
    ApiError(#[from] OpenAiError),

    /// An error occurred when reading an on-disk file.
    #[error("File error: {0}")]
    IoError(#[from] std::io::Error),

    /// The file path is invalid.
    #[error("Invalid file path")]
    InvalidFilePath,
}

impl FilesClient {
    /// Create a new [`FilesClient`].
    /// If the API key is in the environment, you can use the [`Self::from_env`] method instead.
    ///
    /// ```rust
    /// use tysm::files::FilesClient;
    ///
    /// let client = FilesClient::new("sk-1234567890");
    /// ```
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: url::Url::parse("https://api.openai.com/v1/").unwrap(),
            files_path: "files/".to_string(),
            http_client: crate::utils::pooled_client(),
        }
    }

    fn files_url(&self) -> url::Url {
        self.base_url.join(&self.files_path).unwrap()
    }

    /// Create a new [`FilesClient`].
    /// This will use the `OPENAI_API_KEY` environment variable to set the API key.
    /// It will also look in the `.env` file for an `OPENAI_API_KEY` variable (using dotenv).
    ///
    /// ```rust,no_run
    /// # use tysm::files::FilesClient;
    /// let client = FilesClient::from_env().unwrap();
    /// ```
    pub fn from_env() -> Result<Self, ApiKeyError> {
        Ok(Self::new(api_key()?))
    }

    /// Upload a file to the OpenAI API from a file path.
    ///
    /// ```rust,no_run
    /// # use tysm::files::{FilesClient, FilePurpose};
    /// # use tokio_test::block_on;
    /// # block_on(async {
    /// let client = FilesClient::from_env().unwrap();
    /// let file = client.upload_file("mydata.jsonl", FilePurpose::FineTune).await.unwrap();
    /// println!("Uploaded file: {}", file.id);
    /// # });
    /// ```
    pub async fn upload_file(
        &self,
        file_path: impl AsRef<Path>,
        purpose: FilePurpose,
    ) -> Result<FileObject, FilesError> {
        let file_path = file_path.as_ref();
        let file_name = file_path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or(FilesError::InvalidFilePath)?;

        let file = File::open(file_path).await?;
        let stream = FramedRead::new(file, BytesCodec::new());
        let file_part = multipart::Part::stream(reqwest::Body::wrap_stream(stream))
            .file_name(file_name.to_string());

        let form = multipart::Form::new()
            .text("purpose", format!("{:?}", purpose).to_lowercase())
            .part("file", file_part);

        let url = remove_trailing_slash(self.files_url());
        let response = self
            .http_client
            .post(url.clone())
            .header("Authorization", format!("Bearer {}", self.api_key))
            .multipart(form)
            .send()
            .await?;

        let response_text = response.text().await?;

        let file_object: UploadFileResponse =
            serde_json::from_str(&response_text).map_err(|e| FilesError::ApiParseError {
                url: url.to_string(),
                response: response_text.clone(),
                error: e,
            })?;

        match file_object {
            UploadFileResponse::File(file) => Ok(file),
            UploadFileResponse::Error(error) => Err(FilesError::ApiError(error)),
        }
    }

    /// Upload file content directly from bytes to the OpenAI API.
    ///
    /// ```rust,no_run
    /// # use tysm::files::{FilesClient, FilePurpose};
    /// # use tokio_test::block_on;
    /// # block_on(async {
    /// let client = FilesClient::from_env().unwrap();
    /// let content = "{ \"prompt\": \"example\", \"completion\": \"response\" }\n".as_bytes().to_vec();
    /// let file = client.upload_bytes("mydata.jsonl", content, FilePurpose::FineTune).await.unwrap();
    /// println!("Uploaded file: {}", file.id);
    /// # });
    /// ```
    pub async fn upload_bytes(
        &self,
        filename: &str,
        bytes: Vec<u8>,
        purpose: FilePurpose,
    ) -> Result<FileObject, FilesError> {
        let file_part = multipart::Part::bytes(bytes).file_name(filename.to_string());

        let form = multipart::Form::new()
            .text("purpose", format!("{:?}", purpose).to_lowercase())
            .part("file", file_part);

        let url = remove_trailing_slash(self.files_url());
        let response = self
            .http_client
            .post(url.clone())
            .header("Authorization", format!("Bearer {}", self.api_key))
            .multipart(form)
            .send()
            .await?;

        let response_text = response.text().await?;

        let file_object: UploadFileResponse =
            serde_json::from_str(&response_text).map_err(|e| FilesError::ApiParseError {
                url: url.to_string(),
                response: response_text.clone(),
                error: e,
            })?;

        match file_object {
            UploadFileResponse::File(file) => Ok(file),
            UploadFileResponse::Error(error) => Err(FilesError::ApiError(error)),
        }
    }

    /// List all files in the OpenAI API.
    ///
    /// ```rust,no_run
    /// # use tysm::files::FilesClient;
    /// # use tokio_test::block_on;
    /// # block_on(async {
    /// let client = FilesClient::from_env().unwrap();
    /// let files = client.list_files().await.unwrap();
    /// for file in files.data {
    ///     println!("File: {} ({})", file.filename, file.id);
    /// }
    /// # });
    /// ```
    pub async fn list_files(&self) -> Result<FileList, FilesError> {
        self.get_json(self.files_url()).await
    }

    /// Retrieve a file from the OpenAI API.
    ///
    /// ```rust,no_run
    /// # use tysm::files::FilesClient;
    /// # use tokio_test::block_on;
    /// # block_on(async {
    /// let client = FilesClient::from_env().unwrap();
    /// let file = client.retrieve_file("file-abc123").await.unwrap();
    /// println!("File: {} ({})", file.filename, file.id);
    /// # });
    /// ```
    pub async fn retrieve_file(&self, file_id: &str) -> Result<FileObject, FilesError> {
        self.get_json(self.files_url().join(file_id).unwrap()).await
    }

    async fn get_json<T: serde::de::DeserializeOwned>(
        &self,
        url: url::Url,
    ) -> Result<T, FilesError> {
        let response = crate::retry::get_text(
            self.http_client
                .get(url.clone())
                .header("Authorization", format!("Bearer {}", self.api_key)),
        )
        .await?;
        serde_json::from_str(&response).map_err(|error| FilesError::ApiParseError {
            url: url.to_string(),
            response,
            error,
        })
    }

    /// Delete a file from the OpenAI API.
    ///
    /// ```rust,no_run
    /// # use tysm::files::FilesClient;
    /// # use tokio_test::block_on;
    /// # block_on(async {
    /// let client = FilesClient::from_env().unwrap();
    /// let deleted = client.delete_file("file-abc123").await.unwrap();
    /// println!("Deleted: {}", deleted.id);
    /// # });
    /// ```
    pub async fn delete_file(&self, file_id: &str) -> Result<DeletedFile, FilesError> {
        let response = self
            .http_client
            .delete(self.files_url().join(file_id).unwrap())
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await?;

        let deleted_file = response.json::<DeletedFile>().await?;
        Ok(deleted_file)
    }

    /// Download a file from the OpenAI API.
    ///
    /// ```rust,no_run
    /// # use tysm::files::FilesClient;
    /// # use tokio_test::block_on;
    /// # block_on(async {
    /// let client = FilesClient::from_env().unwrap();
    /// let content = client.download_file("file-abc123").await.unwrap();
    /// println!("File content: {}", content);
    /// # });
    /// ```
    pub async fn download_file(&self, file_id: &str) -> Result<String, FilesError> {
        let url = self
            .files_url()
            .join(&format!("{file_id}/content"))
            .unwrap();
        Ok(crate::retry::get_text(
            self.http_client
                .get(url)
                .header("Authorization", format!("Bearer {}", self.api_key)),
        )
        .await?)
    }
}

/// Response from deleting a file.
#[derive(Debug, Deserialize)]
pub struct DeletedFile {
    /// The ID of the deleted file.
    pub id: String,
    /// The object type, always "file".
    pub object: String,
    /// Whether the file was deleted.
    pub deleted: bool,
}
