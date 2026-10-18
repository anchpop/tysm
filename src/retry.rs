//! Shared request retry policy.

/// Read an idempotent endpoint, including its body, with the same backoff as
/// live requests. Retry HTTP status independently of the provider's JSON error
/// shape: gateways often return HTML or plain text instead.
pub(crate) async fn get_text(request: reqwest::RequestBuilder) -> Result<String, reqwest::Error> {
    let mut attempt = 1;
    loop {
        let mut retry_after = None;
        let result = async {
            let response = request
                .try_clone()
                .expect("GET requests are cloneable")
                .send()
                .await?;
            retry_after = response
                .headers()
                .get(reqwest::header::RETRY_AFTER)
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.parse::<u64>().ok())
                .map(Duration::from_secs);
            response.error_for_status()?.text().await
        }
        .await;
        match result {
            Err(error) if attempt < LIVE_REQUEST_ATTEMPTS && transient_http_error(&error) => {
                let kind = if error.status().is_some() {
                    RetryKind::Capacity
                } else {
                    RetryKind::Transport
                };
                let delay = kind.delay(attempt, retry_after);
                log::warn!("idempotent GET failed; retrying in {delay:?}: {error}");
                tokio::time::sleep(delay).await;
                attempt += 1;
            }
            result => return result,
        }
    }
}

pub(crate) fn transient_http_error(error: &reqwest::Error) -> bool {
    match error.status() {
        Some(status) => {
            status.is_server_error() || status == reqwest::StatusCode::TOO_MANY_REQUESTS
        }
        None => error.is_connect() || error.is_timeout() || error.is_request() || error.is_body(),
    }
}

use std::time::Duration;

use crate::OpenAiError;

/// How many times a live request is attempted before its error is returned.
pub(crate) const LIVE_REQUEST_ATTEMPTS: u32 = 4;

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

/// A transport error is usually a pooled keep-alive connection the server has
/// already closed, surfacing on its next use; reconnecting fixes it at once.
/// Capacity frees up on a scale of minutes rather than milliseconds, so those
/// errors wait far longer. Anything else (a malformed request, a schema the
/// model cannot satisfy) is not retried at all.
pub(crate) enum RetryKind {
    Transport,
    Capacity,
}

impl RetryKind {
    /// Attempts start at one. Reconnect quickly; give capacity time to recover.
    /// A provider's explicit delay takes precedence over the default backoff.
    pub(crate) fn delay(self, attempt: u32, retry_after: Option<Duration>) -> Duration {
        retry_after.unwrap_or_else(|| match self {
            Self::Transport => Duration::from_secs(1 << attempt),
            Self::Capacity => Duration::from_secs(15 << (attempt - 1)),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    async fn server(
        statuses: Vec<u16>,
        body: String,
    ) -> (url::Url, tokio::task::JoinHandle<Vec<String>>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/v1/", listener.local_addr().unwrap())
            .parse()
            .unwrap();
        let task = tokio::spawn(async move {
            let mut paths = Vec::new();
            for status in statuses {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut request = Vec::new();
                while !request.windows(4).any(|s| s == b"\r\n\r\n") {
                    let mut buf = [0; 4096];
                    let n = socket.read(&mut buf).await.unwrap();
                    assert_ne!(n, 0);
                    request.extend_from_slice(&buf[..n]);
                }
                paths.push(
                    String::from_utf8(request)
                        .unwrap()
                        .lines()
                        .next()
                        .unwrap()
                        .to_owned(),
                );
                if status == 0 {
                    continue;
                } // close without an HTTP response
                let text = if status == 200 {
                    body.as_str()
                } else {
                    "<html>gateway unavailable</html>"
                };
                socket.write_all(format!("HTTP/1.1 {status} Test\r\nContent-Length: {}\r\nRetry-After: 0\r\nConnection: close\r\n\r\n{text}", text.len()).as_bytes()).await.unwrap();
            }
            paths
        });
        (url, task)
    }

    #[tokio::test]
    async fn batch_gets_retry_closed_connections_rate_limits_and_gateway_pages() {
        for endpoint in ["list", "status", "download"] {
            let body = match endpoint {
                "list" => {
                    serde_json::json!({"object":"list", "data":[], "has_more":false}).to_string()
                }
                "status" => serde_json::json!({
                    "id":"batch-test", "object":"batch", "endpoint":"/v1/chat/completions",
                    "input_file_id":"input", "completion_window":"24h", "status":"completed",
                    "created_at":0, "request_counts":{"total":1,"completed":1,"failed":0}
                })
                .to_string(),
                _ => "result line".into(),
            };
            let (url, task) = server(vec![0, 429, 502, 200], body).await;
            let mut chat = crate::chat_completions::ChatClient::new("unused", "test-model");
            chat.base_url = url;
            let batch = crate::batch::BatchClient::from(&chat);
            match endpoint {
                "list" => assert!(batch.list_batches().await.unwrap().is_empty()),
                "status" => assert_eq!(
                    batch.get_batch_status("batch-test").await.unwrap().id,
                    "batch-test"
                ),
                _ => assert_eq!(
                    crate::files::FilesClient::from(&chat)
                        .download_file("output")
                        .await
                        .unwrap(),
                    "result line"
                ),
            }
            let paths = task.await.unwrap();
            assert_eq!(paths.len(), 4);
            assert!(paths.iter().all(|path| path == &paths[0]));
        }
    }

    #[tokio::test]
    async fn permanent_http_errors_are_not_retried() {
        let (url, task) = server(vec![401], String::new()).await;
        let error = get_text(reqwest::Client::new().get(url)).await.unwrap_err();
        assert_eq!(error.status().unwrap().as_u16(), 401);
        assert_eq!(task.await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn exhausted_listing_retries_never_send_live_even_in_no_batch_mode() {
        let (url, task) = server(vec![503; LIVE_REQUEST_ATTEMPTS as usize], String::new()).await;
        let mut chat =
            crate::chat_completions::ChatClient::new("unused", "test-model").with_no_batch();
        chat.base_url = url;
        let error = chat
            .batch_chat_with_messages_raw(
                vec![(
                    vec![crate::chat_completions::ChatMessage::user("hello")],
                    crate::chat_completions::ResponseFormat::Text,
                )],
                |_| {},
            )
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            crate::chat_completions::BatchChatError::ListBatchesError(_)
        ));
        let paths = task.await.unwrap();
        assert_eq!(paths.len(), LIVE_REQUEST_ATTEMPTS as usize);
        assert!(paths
            .iter()
            .all(|path| path.starts_with("GET /v1/batches?limit=100 ")));
    }
}
