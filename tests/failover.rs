use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tysm::chat_completions::{ChatClient, ChatError, ChatMessage, ResponseFormat};

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
struct Answer {
    value: String,
}

fn response(content: &str) -> serde_json::Value {
    serde_json::json!({
        "id": "test", "object": "chat.completion", "created": 0, "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    })
}

#[tokio::test]
async fn errors_fail_over_with_separate_caches_and_nested_errors() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/v1/", listener.local_addr().unwrap());
    let requests = Arc::new(Mutex::new(Vec::<serde_json::Value>::new()));
    let received = requests.clone();
    let server = tokio::spawn(async move {
        loop {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut bytes = Vec::new();
            let header_end = loop {
                let mut buffer = [0; 4096];
                let count = stream.read(&mut buffer).await.unwrap();
                assert!(count > 0);
                bytes.extend_from_slice(&buffer[..count]);
                if let Some(end) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
                    break end + 4;
                }
            };
            let headers = String::from_utf8_lossy(&bytes[..header_end]);
            let length: usize = headers
                .lines()
                .find_map(|line| {
                    line.to_ascii_lowercase()
                        .strip_prefix("content-length: ")
                        .map(|value| value.parse().unwrap())
                })
                .unwrap();
            while bytes.len() < header_end + length {
                let mut buffer = [0; 4096];
                let count = stream.read(&mut buffer).await.unwrap();
                assert!(count > 0);
                bytes.extend_from_slice(&buffer[..count]);
            }
            let request: serde_json::Value = serde_json::from_slice(&bytes[header_end..]).unwrap();
            let body = match request["model"].as_str().unwrap() {
                "api-error" => serde_json::json!({"error": {"message": "depleted credits", "type": "invalid_request_error", "code": "insufficient_quota", "param": null}}),
                "bad-envelope" => serde_json::json!({"unexpected": true}),
                "bad-schema" => response(r#"{"value":42}"#),
                "no-choices" => { let mut body = response(""); body["choices"] = serde_json::json!([]); body },
                "refusal" => { let mut body = response(""); body["choices"][0]["message"]["refusal"] = "no".into(); body },
                _ => response(r#"{"value":"ok"}"#),
            }.to_string();
            received.lock().unwrap().push(request);
            stream.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes()).await.unwrap();
        }
    });
    let client = |model| ChatClient::new("not-a-key", model).with_url(&url);
    for model in [
        "api-error",
        "bad-envelope",
        "bad-schema",
        "no-choices",
        "refusal",
    ] {
        requests.lock().unwrap().clear();
        let primary = client(model).with_failover(client("gpt-4o").with_reasoning_effort("low"));
        for _ in 0..2 {
            assert_eq!(primary.chat::<Answer>("test").await.unwrap().value, "ok");
        }
        assert!(primary.lru.is_empty());
        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 3); // Retry primary, reuse only the failover's cache.
        assert_eq!(requests[0]["model"], model);
        assert_eq!(requests[1]["model"], "gpt-4o");
        assert_eq!(requests[1]["reasoning_effort"], "low");
        assert_eq!(requests[0]["messages"], requests[1]["messages"]);
        assert_eq!(
            requests[0]["response_format"],
            requests[1]["response_format"]
        );
        assert_eq!(requests[2]["model"], model);
        assert_eq!(
            primary.usage.read().unwrap().total_tokens,
            if matches!(model, "bad-schema" | "no-choices" | "refusal") {
                4
            } else {
                0
            }
        );
    }
    let nested =
        client("api-error").with_failover(client("bad-envelope").with_failover(client("gpt-4o")));
    assert_eq!(nested.chat::<Answer>("nested").await.unwrap().value, "ok");
    let failed = client("api-error")
        .with_failover(client("bad-envelope").with_failover(client("bad-schema")));
    let error = failed.chat::<Answer>("nested failure").await.unwrap_err();
    let ChatError::Failover { primary, failover } = error else {
        panic!("expected combined failure")
    };
    assert!(matches!(*primary, ChatError::ApiError(..)));
    assert!(matches!(*failover, ChatError::Failover { .. }));
    let raw = client("api-error").with_failover(client("gpt-4o"));
    assert_eq!(
        raw.chat_with_messages_raw(vec![ChatMessage::user("raw")], ResponseFormat::Text)
            .await
            .unwrap(),
        r#"{"value":"ok"}"#
    );
    requests.lock().unwrap().clear();
    let offline = client("api-error")
        .with_failover(client("gpt-4o"))
        .with_cached_only();
    assert!(matches!(
        offline.chat::<Answer>("offline").await,
        Err(ChatError::CacheMiss)
    ));
    assert!(requests.lock().unwrap().is_empty());
    server.abort();
}
