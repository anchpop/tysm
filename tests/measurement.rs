use tysm::chat_completions::{ChatClient, ChatError, IndividualChatError};

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
struct Answer {
    _value: String,
}

#[tokio::test]
async fn measurement_blocks_live_and_batch_without_client_opt_in() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("measurement.jsonl");
    tysm::measurement::start(&path).unwrap();
    tysm::measurement::stage("test");
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    // Intentionally not .with_cached_only(): measurement itself must enforce it.
    let client = ChatClient::new("not-a-key", "gpt-4o")
        .with_url(format!("http://{}/v1/", listener.local_addr().unwrap()));
    assert!(matches!(
        client.chat::<Answer>("live miss").await,
        Err(ChatError::CacheMiss)
    ));
    let results = client
        .batch_chat::<Answer>(vec!["first", "second", "third"], |_| {})
        .await
        .unwrap();
    assert_eq!(results.len(), 3);
    assert!(results
        .iter()
        .all(|result| matches!(result, Err(IndividualChatError::CacheMiss))));
    let anthropic = ChatClient::new("not-a-key", "claude-sonnet-4-5")
        .with_url(format!("http://{}/v1/", listener.local_addr().unwrap()));
    assert!(matches!(
        anthropic.chat::<Answer>("native miss").await,
        Err(ChatError::CacheMiss)
    ));
    let native = anthropic
        .batch_chat::<Answer>(vec!["native batch miss"], |_| {})
        .await
        .unwrap();
    assert!(matches!(native[0], Err(IndividualChatError::CacheMiss)));
    let mut embeddings =
        tysm::embeddings::EmbeddingsClient::new("not-a-key", "text-embedding-3-small");
    embeddings.base_url = format!("http://{}/v1/", listener.local_addr().unwrap())
        .parse()
        .unwrap();
    assert!(matches!(
        embeddings.embed_single("embedding miss".into()).await,
        Err(tysm::embeddings::EmbeddingsError::CacheMiss)
    ));
    assert_eq!(
        listener.accept().unwrap_err().kind(),
        std::io::ErrorKind::WouldBlock
    );
    let rows: Vec<serde_json::Value> = std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(rows.len(), 6);
    assert!(rows.iter().all(|row| row["cached_usage"].is_null()));
    assert_eq!(client.cost(), Some(0.0));
    assert_eq!(tysm::chat_completions::process_spend().to_string(), "$0.00");
}
