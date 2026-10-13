use serde_json::{json, Value};
use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
};
use tysm::decisions::{
    ChoiceOption, ChoiceQuestion, DecisionAnswer, DecisionResponse, DecisionsClient, DecisionsError,
};

const FIXTURE: &str = include_str!("../test_fixtures/decisions_readings.json");

fn questions() -> Vec<ChoiceQuestion> {
    vec![
        ChoiceQuestion {
            name: Some("japanese_reading".into()),
            instructions: "Read 両親".into(),
            choices: vec![
                ChoiceOption::new("りょうしん"),
                ChoiceOption::new("ふたおや"),
            ],
        },
        ChoiceQuestion {
            name: Some("mandarin_reading".into()),
            instructions: "Read 了不起".into(),
            choices: vec![
                ChoiceOption::new("liǎo bu qǐ"),
                ChoiceOption::new("liǎo bù qǐ"),
            ],
        },
    ]
}

async fn read_request(socket: &mut tokio::net::TcpStream) -> Value {
    let mut bytes = Vec::new();
    loop {
        let mut buf = [0; 4096];
        let count = socket.read(&mut buf).await.unwrap();
        assert_ne!(count, 0);
        bytes.extend_from_slice(&buf[..count]);
        if let Some(end) = bytes.windows(4).position(|v| v == b"\r\n\r\n") {
            let headers = String::from_utf8_lossy(&bytes[..end]).to_ascii_lowercase();
            let length = headers
                .lines()
                .find_map(|h| h.strip_prefix("content-length: "))
                .unwrap()
                .parse::<usize>()
                .unwrap();
            if bytes.len() >= end + 4 + length {
                assert!(headers.starts_with("post /v1/decisions http/1.1"));
                assert!(headers.contains("authorization: bearer test-key"));
                assert!(headers.contains("content-type: application/json"));
                return serde_json::from_slice(&bytes[end + 4..]).unwrap();
            }
        }
    }
}

async fn reply(socket: &mut tokio::net::TcpStream, status: u16, body: &str) {
    socket
        .write_all(
            format!(
                "HTTP/1.1 {status} Test\r\nContent-Length: {}\r\nRetry-After: 0\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .as_bytes(),
        )
        .await
        .unwrap();
}

async fn server(steps: Vec<(u16, String)>) -> (String, tokio::task::JoinHandle<Vec<Value>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/v1", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        let mut received = Vec::new();
        for (status, body) in steps {
            let (mut socket, _) = tokio::time::timeout(Duration::from_secs(10), listener.accept())
                .await
                .unwrap()
                .unwrap();
            received.push(read_request(&mut socket).await);
            reply(&mut socket, status, &body).await;
        }
        received
    });
    (url, task)
}

fn client(url: &str, cache: &std::path::Path) -> DecisionsClient {
    DecisionsClient::new("test-key", "gpt-6-luna")
        .with_url(url)
        .with_cache_directory(cache)
}

#[tokio::test]
async fn exact_wire_and_persistent_cache() {
    let dir = tempfile::tempdir().unwrap();
    let (url, task) = server(vec![(200, FIXTURE.into())]).await;
    let mut questions = questions();
    questions[0].choices[0].description = Some("Standard reading".into());
    let first = client(&url, dir.path());
    let response = first
        .decide("unchanged\n両親 / 了不起", &questions)
        .await
        .unwrap();
    let requests = task.await.unwrap(); // Listener is now closed: a cache miss must fail.
    assert_eq!(
        requests,
        vec![
            json!({"model":"gpt-6-luna", "input":"unchanged\n両親 / 了不起", "questions":[
                {"type":"choice","name":"japanese_reading","instructions":"Read 両親","choices":[{"value":"りょうしん","description":"Standard reading"},{"value":"ふたおや"}]},
                {"type":"choice","name":"mandarin_reading","instructions":"Read 了不起","choices":[{"value":"liǎo bu qǐ"},{"value":"liǎo bù qǐ"}]}
            ]})
        ]
    );
    let value = serde_json::to_value(&response).unwrap();
    assert_eq!(value, serde_json::from_str::<Value>(FIXTURE).unwrap());
    let DecisionAnswer::Choice(answer) = &response.answers[1] else {
        panic!()
    };
    assert_eq!(answer.confidence, 0.98);
    assert_eq!(answer.probabilities[0].probability, 0.99);
    drop(first);
    let cached = client(&url, dir.path())
        .decide("unchanged\n両親 / 了不起", &questions)
        .await
        .unwrap();
    assert_eq!(serde_json::to_value(cached).unwrap(), value);
}

#[tokio::test]
async fn failures_are_not_cached_but_partial_refusals_are() {
    let mut refusal: Value = serde_json::from_str(FIXTURE).unwrap();
    refusal["answers"][1] = json!({"type":"refusal", "name":"mandarin_reading"});
    let mut invalid: Value = serde_json::from_str(FIXTURE).unwrap();
    invalid["answers"] = json!([]);
    let dir = tempfile::tempdir().unwrap();
    let (url, task) = server(vec![
        (400, "{\"error\":\"invalid request\"}".into()),
        (200, "{}".into()),
        (200, invalid.to_string()),
        (200, refusal.to_string()),
    ])
    .await;
    let client = client(&url, dir.path());
    assert!(
        matches!(client.decide("text", &questions()).await, Err(DecisionsError::Http { status, body }) if status.as_u16() == 400 && body.contains("invalid request"))
    );
    assert!(matches!(
        client.decide("text", &questions()).await,
        Err(DecisionsError::Json(_))
    ));
    assert!(matches!(
        client.decide("text", &questions()).await,
        Err(DecisionsError::InvalidResponse(_))
    ));
    let response = client.decide("text", &questions()).await.unwrap();
    assert!(matches!(&response.answers[0], DecisionAnswer::Choice(_)));
    assert!(
        matches!(&response.answers[1], DecisionAnswer::Refusal { name } if name.as_deref() == Some("mandarin_reading"))
    );
    assert_eq!(task.await.unwrap().len(), 4);
    let cached = DecisionsClient::new("test-key", "gpt-6-luna")
        .with_url(&url)
        .with_cache_directory(dir.path())
        .decide("text", &questions())
        .await
        .unwrap();
    assert_eq!(serde_json::to_value(cached).unwrap(), refusal);
}

#[tokio::test]
async fn every_request_field_and_url_affect_cache_identity() {
    let dir = tempfile::tempdir().unwrap();
    let (url, task) = server(vec![(200, FIXTURE.into()); 7]).await;
    let client = client(&url, dir.path());
    client.decide("text", &questions()).await.unwrap();
    client.decide("changed text", &questions()).await.unwrap();
    let mut changed = questions();
    changed[0].instructions.push('!');
    client.decide("text", &changed).await.unwrap();
    changed = questions();
    changed[0].choices.reverse();
    client.decide("text", &changed).await.unwrap();
    changed = questions();
    changed[0].choices[1] = ChoiceOption::new("different");
    client.decide("text", &changed).await.unwrap();
    changed = questions();
    changed[0].choices[0].description = Some("different description".into());
    client.decide("text", &changed).await.unwrap();
    DecisionsClient::new("test-key", "other-model")
        .with_url(&url)
        .with_cache_directory(dir.path())
        .decide("text", &questions())
        .await
        .unwrap();
    assert_eq!(task.await.unwrap().len(), 7);
    let (other_url, task) = server(vec![(200, FIXTURE.into())]).await;
    DecisionsClient::new("test-key", "gpt-6-luna")
        .with_url(&other_url)
        .with_cache_directory(dir.path())
        .decide("text", &questions())
        .await
        .unwrap();
    assert_eq!(task.await.unwrap().len(), 1);
}

#[tokio::test]
async fn bool_and_string_choices_are_distinct_and_unnamed_is_omitted() {
    let mut response: Value = serde_json::from_str(FIXTURE).unwrap();
    response["answers"] = json!([{"type":"choice","name":null,"choice":true,"confidence":0.5,"probabilities":[{"value":true,"probability":0.75},{"value":"true","probability":0.25}]}]);
    let (url, task) = server(vec![(200, response.to_string())]).await;
    let questions = [ChoiceQuestion {
        name: None,
        instructions: "Choose".into(),
        choices: vec![ChoiceOption::new(true), ChoiceOption::new("true")],
    }];
    let result = DecisionsClient::new("test-key", "gpt-6-luna")
        .with_url(&url)
        .decide("text", &questions)
        .await
        .unwrap();
    assert_eq!(serde_json::to_value(result).unwrap(), response);
    let requests = task.await.unwrap();
    assert!(requests[0]["questions"][0].get("name").is_none());
}

#[tokio::test]
async fn invalid_choices_fail_before_http() {
    let client = DecisionsClient::new("test-key", "gpt-6-luna").with_url("http://127.0.0.1:1");
    assert!(matches!(
        client.decide("", &[]).await,
        Err(DecisionsError::InvalidRequest(_))
    ));
    for choices in [
        vec![ChoiceOption::new("one")],
        vec![ChoiceOption::new("same"); 2],
        (0..256).map(|n| ChoiceOption::new(n.to_string())).collect(),
    ] {
        let question = ChoiceQuestion {
            name: None,
            instructions: "Choose".into(),
            choices,
        };
        assert!(matches!(
            client.decide("", &[question]).await,
            Err(DecisionsError::InvalidRequest(_))
        ));
    }
}

#[test]
fn unknown_answer_types_are_not_silently_discarded() {
    let mut fixture: Value = serde_json::from_str(FIXTURE).unwrap();
    fixture["answers"][0]["type"] = json!("unexpected");
    assert!(serde_json::from_value::<DecisionResponse>(fixture).is_err());
}

#[tokio::test]
async fn retries_transient_errors_then_caches_success() {
    let dir = tempfile::tempdir().unwrap();
    let (url, task) = server(vec![
        (429, "rate limited".into()),
        (502, "bad gateway".into()),
        (409, json!({"error":{"type":"invalid_request_error","code":"flex_unavailable","message":"No capacity"}}).to_string()),
        (200, FIXTURE.into()),
    ]).await;
    let response = client(&url, dir.path())
        .decide("text", &questions())
        .await
        .unwrap();
    let requests = task.await.unwrap();
    assert_eq!(requests.len(), 4);
    assert!(requests.iter().all(|request| request == &requests[0]));
    let cached = client(&url, dir.path())
        .decide("text", &questions())
        .await
        .unwrap();
    assert_eq!(
        serde_json::to_value(cached).unwrap(),
        serde_json::to_value(response).unwrap()
    );
}

#[tokio::test]
async fn exhausted_retries_and_auth_errors_are_not_cached() {
    let dir = tempfile::tempdir().unwrap();
    let mut steps = vec![(503, "unavailable".into()); 4];
    steps.extend([(401, "bad key".into()), (200, FIXTURE.into())]);
    let (url, task) = server(steps).await;
    let client = client(&url, dir.path());
    assert!(
        matches!(client.decide("text", &questions()).await, Err(DecisionsError::Http { status, .. }) if status.as_u16() == 503)
    );
    assert!(
        matches!(client.decide("text", &questions()).await, Err(DecisionsError::Http { status, .. }) if status.as_u16() == 401)
    );
    client.decide("text", &questions()).await.unwrap();
    assert_eq!(task.await.unwrap().len(), 6);
}

#[tokio::test]
async fn reconnects_after_closed_connection() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/v1/", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let first = read_request(&mut socket).await;
        drop(socket); // No HTTP response: emulate a disconnected upstream.
        let start = tokio::time::Instant::now();
        let (mut socket, _) = listener.accept().await.unwrap();
        assert!(start.elapsed() >= Duration::from_secs(2));
        assert_eq!(read_request(&mut socket).await, first);
        reply(&mut socket, 200, FIXTURE).await;
    });
    tokio::time::timeout(Duration::from_secs(10), async {
        DecisionsClient::new("test-key", "gpt-6-luna")
            .with_url(url)
            .decide("text", &questions())
            .await
            .unwrap();
        task.await.unwrap();
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn retry_after_is_honored_without_holding_concurrency_slot() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/v1/", listener.local_addr().unwrap());
    let (sent, received) = tokio::sync::oneshot::channel();
    let task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        assert_eq!(read_request(&mut socket).await["input"], "retry me");
        let start = tokio::time::Instant::now();
        socket.write_all(b"HTTP/1.1 429 Test\r\nContent-Length: 0\r\nRetry-After: 1\r\nConnection: close\r\n\r\n").await.unwrap();
        drop(socket);
        sent.send(()).unwrap();
        // A different call must complete while the first is backing off, even
        // with max concurrency one. Holding the permit makes this time out.
        let (mut socket, _) = tokio::time::timeout(Duration::from_millis(750), listener.accept())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(read_request(&mut socket).await["input"], "other call");
        reply(&mut socket, 200, FIXTURE).await;
        let (mut socket, _) = listener.accept().await.unwrap();
        assert!(start.elapsed() >= Duration::from_secs(1));
        assert_eq!(read_request(&mut socket).await["input"], "retry me");
        reply(&mut socket, 200, FIXTURE).await;
    });
    let client = DecisionsClient::new("test-key", "gpt-6-luna")
        .with_url(url)
        .with_max_concurrent_requests(1);
    let questions = questions();
    tokio::time::timeout(Duration::from_secs(10), async {
        let other = async {
            received.await.unwrap();
            client.decide("other call", &questions).await.unwrap();
        };
        let (first, ()) = tokio::join!(client.decide("retry me", &questions), other);
        first.unwrap();
        task.await.unwrap();
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn concurrency_is_limited() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/v1/", listener.local_addr().unwrap());
    let active = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let peak_server = peak.clone();
    let task = tokio::spawn(async move {
        let mut tasks = Vec::new();
        for _ in 0..6 {
            let (mut socket, _) = listener.accept().await.unwrap();
            let active = active.clone();
            let peak = peak_server.clone();
            tasks.push(tokio::spawn(async move {
                read_request(&mut socket).await;
                let count = active.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(count, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(40)).await;
                active.fetch_sub(1, Ordering::SeqCst);
                reply(&mut socket, 200, FIXTURE).await;
            }));
        }
        for task in tasks {
            task.await.unwrap();
        }
    });
    let client = DecisionsClient::new("test-key", "gpt-6-luna")
        .with_url(url)
        .with_max_concurrent_requests(2);
    let questions = questions();
    let contexts: Vec<_> = (0..6).map(|n| n.to_string()).collect();
    let calls = contexts
        .iter()
        .map(|context| client.decide(context, &questions));
    tokio::time::timeout(Duration::from_secs(10), async {
        for result in futures::future::join_all(calls).await {
            result.unwrap();
        }
        task.await.unwrap();
    })
    .await
    .unwrap();
    assert_eq!(peak.load(Ordering::SeqCst), 2);
}
