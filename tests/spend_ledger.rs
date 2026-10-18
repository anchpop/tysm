use std::io::{Read, Write};
use tysm::spend::LedgerRow;

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
struct Answer {
    value: String,
}

#[tokio::test]
async fn snapshots_append_and_share_exact_process_totals() {
    let directory = tempfile::tempdir().unwrap();
    let guard = tysm::spend::start_ledger(directory.path(), "jpn").unwrap();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut request = [0; 16384];
        assert!(stream.read(&mut request).unwrap() > 0);
        let body = r#"{"id":"test","object":"chat.completion","created":0,"model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"{\"value\":\"ok\"}"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#;
        write!(stream,"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",body.len(),body).unwrap();
    });
    let client = tysm::chat_completions::ChatClient::new("not-a-key", "gpt-4o")
        .with_url(format!("http://{address}/v1/"));
    let answer: Answer = client.chat("test").await.unwrap();
    assert_eq!(answer.value, "ok");
    server.join().unwrap();
    tysm::spend::snapshot_label("lap-test").unwrap();
    drop(guard);
    let paths: Vec<_> = std::fs::read_dir(directory.path())
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();
    assert_eq!(paths.len(), 1);
    assert!(paths[0]
        .file_name()
        .unwrap()
        .to_string_lossy()
        .ends_with("-jpn.jsonl"));
    let rows: Vec<LedgerRow> = std::fs::read_to_string(&paths[0])
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let labels: Vec<_> = rows
        .iter()
        .filter_map(|row| match row {
            LedgerRow::Snapshot { label, .. } => Some(label.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(labels, ["start", "lap-test", "exit"]);
    let LedgerRow::Snapshot { buckets, .. } = rows.last().unwrap() else {
        panic!()
    };
    assert_eq!(buckets.len(), 1);
    assert_eq!(buckets[0].1.requests, 1);
    assert_eq!(
        buckets[0].1.known_dollars,
        tysm::chat_completions::process_spend().dollars
    );
    assert_eq!(client.cost(), Some(0.0000125));
}
