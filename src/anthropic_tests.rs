use super::*;

#[derive(Debug, Deserialize, JsonSchema, PartialEq)]
struct Answer {
    value: u32,
}

fn client() -> ChatClient {
    ChatClient::new("test-key", "claude-haiku-5-5")
}
fn request(client: &ChatClient, text: &str) -> ChatRequest {
    client.request_for_messages(
        vec![
            ChatMessage::system("Return a value."),
            ChatMessage::user(text),
        ],
        ResponseFormat::JsonSchema {
            json_schema: client.json_schema_format::<Answer>(),
        },
    )
}
fn message(content: &str, reason: &str) -> Value {
    json!({"id":"msg_test", "model":"claude-haiku-5-5", "stop_reason":reason,
        "content":[{"type":"thinking", "thinking":""}, {"type":"text", "text":content}],
        "usage":{"input_tokens":100, "output_tokens":10}})
}
fn batch(status: &str, succeeded: u32, failed: u32) -> Value {
    json!({"id":"msgbatch_test", "processing_status":status,
        "request_counts":{"processing":if status == "ended" {0} else {1}, "succeeded":succeeded, "errored":failed, "canceled":0, "expired":0},
        "results_url":if status == "ended" { json!("{{BASE}}results") } else { Value::Null }})
}
fn result_line(client: &ChatClient, prompt: &str, result: Value) -> String {
    json!({"custom_id":custom_id(&client.cache_key(&request(client, prompt))), "result":result})
        .to_string()
}
fn success(content: &str) -> Value {
    json!({"type":"succeeded", "message":message(content,"end_turn")})
}

// A scripted local server asserts request order, body and authentication without
// ever touching real credentials or the network outside loopback.
async fn server(
    steps: Vec<(&'static str, u16, String)>,
) -> (String, tokio::task::JoinHandle<Vec<String>>) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}/v1/", listener.local_addr().unwrap());
    let base_clone = base.clone();
    let task = tokio::spawn(async move {
        let mut received = Vec::new();
        for (expected, status, body) in steps {
            let (mut socket, _) = tokio::time::timeout(Duration::from_secs(15), listener.accept())
                .await
                .unwrap()
                .unwrap();
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
                        .map(|n| n.parse::<usize>().unwrap())
                        .unwrap_or(0);
                    if bytes.len() >= end + 4 + length {
                        break;
                    }
                }
            }
            let request = String::from_utf8(bytes).unwrap();
            assert!(
                request.starts_with(expected),
                "expected {expected}, got {request}"
            );
            assert!(request.contains("x-api-key: test-key"));
            assert!(request.contains("anthropic-version: 2023-06-01"));
            assert!(!request.to_ascii_lowercase().contains("authorization:"));
            received.push(request);
            let body = body.replace("{{BASE}}", &base_clone);
            socket.write_all(format!("HTTP/1.1 {status} Test\r\nContent-Length: {}\r\nRetry-After: 0\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes()).await.unwrap();
        }
        received
    });
    (base, task)
}

#[tokio::test]
async fn batch_futures_can_be_spawned() {
    tokio::spawn(async {
        let client = client().with_cached_only();
        let results = client
            .batch_chat::<Answer>(vec!["uncached"], |_| {})
            .await
            .unwrap();
        assert!(matches!(results[0], Err(IndividualChatError::CacheMiss)));
        let items = vec!["borrowed input".to_owned()];
        let results = client
            .batch_chat_with_system_prompt_fn::<_, _, Answer>(
                "Return a value.",
                &items,
                String::as_str,
                |_| {},
            )
            .await
            .unwrap();
        assert!(matches!(results[0].1, Err(IndividualChatError::CacheMiss)));
    })
    .await
    .unwrap();
}

#[test]
fn provider_selection_and_explicit_override() {
    let native = client();
    assert_eq!(native.provider, Provider::Anthropic);
    assert_eq!(native.base_url.as_str(), "https://api.anthropic.com/v1/");
    let compatible = native.with_provider(Provider::OpenAi);
    assert_eq!(compatible.provider, Provider::OpenAi);
    assert_eq!(compatible.base_url.as_str(), "https://api.openai.com/v1/");
    assert_eq!(Provider::for_model("gpt-5.6"), Provider::OpenAi);
}

#[test]
fn empty_system_prompt_is_omitted() {
    let req = client().request_for_messages(
        vec![ChatMessage::system(""), ChatMessage::user("hello")],
        ResponseFormat::Text,
    );
    assert!(request_body(&req).unwrap().get("system").is_none());
}

#[test]
fn request_uses_native_schema_system_thinking_and_all_extra_parameters() {
    let client = client()
        .with_reasoning_effort("minimal")
        .with_extra_body(json!({
            "max_tokens":500, "cache_control":{"type":"ephemeral", "ttl":"1h"},
            "output_config":{"effort":"medium"}, "stop_sequences":["STOP"]
        }));
    let body = request_body(&request(&client, "hello")).unwrap();
    assert_eq!(body["model"], "claude-haiku-5-5");
    assert_eq!(body["system"][0]["text"], "Return a value.");
    assert_eq!(body["messages"].as_array().unwrap().len(), 1);
    assert_eq!(body["messages"][0]["role"], "user");
    assert_eq!(body["thinking"]["type"], "adaptive");
    assert_eq!(body["output_config"]["effort"], "medium");
    assert_eq!(body["output_config"]["format"]["type"], "json_schema");
    let schema = &body["output_config"]["format"]["schema"];
    assert_eq!(schema["additionalProperties"], false);
    assert!(schema.get("$schema").is_none());
    assert!(schema["properties"]["value"].get("minimum").is_none());
    assert!(schema["properties"]["value"]["description"]
        .as_str()
        .unwrap()
        .contains("minimum"));
    assert_eq!(body["cache_control"]["ttl"], "1h");
    assert_eq!(body["max_tokens"], 500);
    assert_eq!(body["stop_sequences"], json!(["STOP"]));
    assert!(body.get("response_format").is_none());
    assert!(body.get("tools").is_none());
}

#[test]
fn schema_transform_visits_schemas_not_property_names() {
    let mut schema: Schema = json!({"type":"object", "properties":{
        "minimum":{"type":"integer", "minimum":3},
        "items":{"type":"array", "items":{"type":"string", "minLength":2}, "minItems":3},
        "nested":{"type":["object","null"], "properties":{"a":{"type":"string"}}}
    }})
    .try_into()
    .unwrap();
    AnthropicTransform::default().transform(&mut schema);
    let value = schema.as_value();
    assert_eq!(value["properties"]["minimum"]["type"], "integer");
    assert!(value["properties"]["minimum"].get("minimum").is_none());
    assert!(value["properties"]["items"].get("minItems").is_none());
    assert!(value["properties"]["items"]["items"]
        .get("minLength")
        .is_none());
    assert_eq!(value["properties"]["nested"]["additionalProperties"], false);
}

#[test]
fn nested_optional_schemas_preserve_references_and_reject_maps() {
    #[derive(JsonSchema)]
    #[allow(dead_code)]
    enum Kind {
        Alpha,
        Beta,
    }
    #[derive(JsonSchema)]
    #[allow(dead_code)]
    struct Nested {
        kind: Kind,
        other: Option<Answer>,
        values: Vec<Answer>,
    }
    let c = client();
    let mut req = request(&c, "x");
    req.response_format = ResponseFormat::JsonSchema {
        json_schema: c.json_schema_format::<Nested>(),
    };
    let body = request_body(&req).unwrap();
    let schema = &body["output_config"]["format"]["schema"];
    assert!(schema["$defs"]["Answer"].is_object());
    assert!(schema["properties"]["values"]["items"]
        .get("$ref")
        .is_some());
    assert!(schema["properties"]["other"].get("anyOf").is_some());
    assert_eq!(schema["$defs"]["Answer"]["additionalProperties"], false);
    req.response_format = ResponseFormat::JsonSchema {
        json_schema: c.json_schema_format::<HashMap<String, u32>>(),
    };
    assert!(request_body(&req)
        .unwrap_err()
        .to_string()
        .contains("arbitrary keys"));
}

#[test]
fn incompatible_options_fail_locally() {
    for client in [
        client().with_service_tier("flex"),
        client().with_prompt_cache_key("routing"),
        client().with_reasoning_effort("none"),
        client().with_extra_body(json!({"stream":true})),
        client().with_extra_body(json!({"model":"different"})),
        client().with_extra_body(json!({"output_config":{"format":{}}})),
        client().with_extra_body(json!(42)),
    ] {
        assert!(request_body(&request(&client, "x")).is_err());
    }
    let mut req = request(&client(), "x");
    req.messages.push(ChatMessage::system("late"));
    assert!(request_body(&req).is_err());
    req.messages = vec![ChatMessage::new(
        Role::User,
        vec![ChatMessageContent::InputAudio {
            input_audio: InputAudio::wav([]),
        }],
    )];
    assert!(request_body(&req).is_err());
}

#[test]
fn images_translate_to_native_sources() {
    let mut req = request(&client(), "x");
    req.messages[1].content.extend([
        ChatMessageContent::ImageUrl {
            image: ImageUrl {
                url: "data:image/png;base64,YQ==".into(),
                detail: None,
            },
        },
        ChatMessageContent::ImageUrl {
            image: ImageUrl {
                url: "https://example.com/a.png".into(),
                detail: Some(ImageDetail::Auto),
            },
        },
    ]);
    let body = request_body(&req).unwrap();
    assert_eq!(
        body["messages"][0]["content"][1]["source"],
        json!({"type":"base64","media_type":"image/png","data":"YQ=="})
    );
    assert_eq!(body["messages"][0]["content"][2]["source"]["type"], "url");
}

#[test]
fn cache_keys_include_provider_and_every_input() {
    let native = client();
    let original = request(&native, "x");
    let key = native.cache_key(&original);
    assert_ne!(key, original.cache_key());
    let openai = client().with_provider(Provider::OpenAi);
    assert_eq!(openai.cache_key(&original), original.cache_key());
    let mut changed = Vec::new();
    for field in [
        "model",
        "service_tier",
        "reasoning_effort",
        "prompt_cache_key",
        "extra_body",
        "messages",
        "schema",
    ] {
        let mut r = original.clone();
        match field {
            "model" => r.model = "claude-sonnet-5-5".into(),
            "service_tier" => r.service_tier = Some("auto".into()),
            "reasoning_effort" => r.reasoning_effort = Some("low".into()),
            "prompt_cache_key" => r.prompt_cache_key = Some("hint".into()),
            "extra_body" => r.extra_body = Some(json!({"max_tokens":50})),
            "messages" => r.messages[1] = ChatMessage::user("y"),
            "schema" => r.response_format = ResponseFormat::Text,
            _ => unreachable!(),
        }
        assert_ne!(key, native.cache_key(&r), "{field}");
        changed.push(native.cache_key(&r));
    }
    assert_eq!(changed.iter().collect::<HashSet<_>>().len(), changed.len());
    assert!(custom_id(&key).len() <= 64);
    assert!(custom_id(&key)
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'));
}

#[test]
fn parses_all_batch_result_types_and_stop_reasons() {
    let row: ResultLine =
        serde_json::from_value(json!({"custom_id":"a", "result":success("{\"value\":4}")}))
            .unwrap();
    assert_eq!(
        response_content(&row.result.response().unwrap()).unwrap(),
        "{\"value\":4}"
    );
    for kind in ["errored", "expired", "canceled"] {
        let row: ResultLine = serde_json::from_value(json!({"custom_id":"a", "result":{"type":kind, "error":{"type":"error", "error":{"type":"invalid_request_error", "message":"invalid"}}}})).unwrap();
        assert!(row.result.response().is_err());
    }
    for reason in ["refusal", "max_tokens", "pause_turn", "tool_use"] {
        let response = normalize(serde_json::from_value(message("", reason)).unwrap());
        assert!(response_content(&response).is_err(), "{reason}");
    }
}

#[test]
fn batch_split_counts_serialized_utf8_bytes_and_wrapper() {
    let one = json!({"custom_id":"a", "params":{"text":"世界"}});
    let two = json!({"custom_id":"b", "params":{"text":"世界"}});
    let exact = serde_json::to_vec(&json!({"requests":[one.clone(),two.clone()]}))
        .unwrap()
        .len();
    assert_eq!(
        split_requests(vec![one.clone(), two.clone()], 100, exact)
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        split_requests(vec![one.clone(), two.clone()], 100, exact - 1)
            .unwrap()
            .len(),
        2
    );
    assert_eq!(
        split_requests(vec![one.clone(), two], 1, exact)
            .unwrap()
            .len(),
        2
    );
    assert!(split_requests(vec![one], 100, 10).is_err());
    assert!(split_requests(vec![], 100, 100).unwrap().is_empty());
}

#[test]
fn usage_and_pricing_include_both_cache_write_ttls_and_long_context() {
    let usage: Usage = serde_json::from_value(json!({
        "input_tokens":1000,"output_tokens":1000,
        "cache_read_input_tokens":2000,"cache_creation_input_tokens":7000,
        "cache_creation":{"ephemeral_5m_input_tokens":3000,"ephemeral_1h_input_tokens":4000}
    }))
    .unwrap();
    let usage: ChatUsage = usage.into();
    assert_eq!(usage.prompt_tokens, 10000);
    assert_eq!(usage.total_tokens, 11000);
    let expected =
        (1000.0 * 0.1 + 2000.0 * 0.01 + 3000.0 * 0.125 + 4000.0 * 0.2 + 1000.0 * 0.5) / 1e6;
    assert_eq!(
        crate::model_prices::cost_of_call("claude-haiku-5-5", None, usage),
        Some(expected)
    );
    assert_eq!(
        crate::model_prices::cost_of_call("claude-haiku-5-5", Some("batch"), usage),
        Some(expected / 2.0)
    );
    let c = client();
    c.record_spend(None, usage, 0.5);
    assert_eq!(c.cost(), Some(expected / 2.0));
    let mut boundary = usage;
    boundary.prompt_tokens = 100000;
    let regular = crate::model_prices::cost_of_call("claude-haiku-5-5", None, boundary).unwrap();
    boundary.prompt_tokens += 1;
    let premium = crate::model_prices::cost_of_call("claude-haiku-5-5", None, boundary).unwrap();
    assert!((premium - (regular * 5.0 + 0.5 / 1e6)).abs() < 1e-12);
    let rates = [
        ("claude-sonnet-5-5", 2.0, 0.10, 10.0),
        ("claude-opus-5-5", 4.0, 0.20, 20.0),
        ("claude-fable-5-1", 10.0, 0.25, 50.0),
    ];
    for (model, input, read, output) in rates {
        let price = crate::model_prices::price_of(model).unwrap();
        assert_eq!(
            (
                price.standard.input,
                price.standard.cached_input,
                price.standard.output
            ),
            (input, Some(read), output)
        );
    }
}

#[tokio::test]
async fn live_retries_429_and_529_but_not_400() {
    let (url, task) = server(vec![
        ("POST /v1/messages ", 429, "rate limit".into()),
        ("POST /v1/messages ", 529, "overloaded".into()),
        (
            "POST /v1/messages ",
            200,
            message("{\"value\":7}", "end_turn").to_string(),
        ),
        ("POST /v1/messages ", 400, "invalid".into()),
    ])
    .await;
    let c = client().with_url(url);
    let answer: Answer = c.chat("x").await.unwrap();
    assert_eq!(answer.value, 7);
    assert!(matches!(
        c.chat::<Answer>("y").await,
        Err(ChatError::HttpStatus { status: 400, .. })
    ));
    let sent = task.await.unwrap();
    assert_eq!(sent.len(), 4);
    assert_eq!(sent[0], sent[1]);
    assert_eq!(sent[1], sent[2]);
}

#[tokio::test]
async fn paid_failures_are_accounted_and_never_cached() {
    let (url, task) = server(vec![
        (
            "POST /v1/messages ",
            200,
            message("{\"value\":1}", "max_tokens").to_string(),
        ),
        (
            "POST /v1/messages ",
            200,
            message("", "refusal").to_string(),
        ),
        (
            "POST /v1/messages ",
            200,
            message("{\"bad\":1}", "end_turn").to_string(),
        ),
        (
            "POST /v1/messages ",
            200,
            message("{\"value\":3}", "end_turn").to_string(),
        ),
    ])
    .await;
    let c = client().with_url(url);
    for _ in 0..3 {
        assert!(c.chat::<Answer>("x").await.is_err());
        assert!(c.lru.is_empty());
    }
    assert_eq!(c.chat::<Answer>("x").await.unwrap().value, 3);
    assert_eq!(c.usage().prompt_tokens, 400);
    assert_eq!(c.chat::<Answer>("x").await.unwrap().value, 3);
    assert_eq!(c.usage().prompt_tokens, 400);
    task.await.unwrap();
}

#[tokio::test]
async fn batch_deduplicates_maps_errors_caches_success_and_recovers_failures() {
    let dir = tempfile::tempdir().unwrap();
    let c = client().with_cache_directory(dir.path());
    let rows = [
        result_line(&c, "good", success("{\"value\":9}")),
        result_line(&c, "expired", json!({"type":"expired"})),
        result_line(
            &c,
            "bad",
            json!({"type":"errored","error":{"type":"invalid_request_error","message":"bad"}}),
        ),
        result_line(&c, "canceled", json!({"type":"canceled"})),
    ]
    .join("\n");
    let (url, task) = server(vec![
        (
            "POST /v1/messages/batches ",
            200,
            batch("in_progress", 0, 0).to_string(),
        ),
        (
            "GET /v1/messages/batches/msgbatch_test ",
            200,
            batch("ended", 1, 3).to_string(),
        ),
        ("GET /v1/results ", 200, rows),
    ])
    .await;
    let c = c.with_url(url);
    let answers = c
        .batch_chat_with_system_prompt::<Answer>(
            "Return a value.",
            vec!["good", "expired", "good", "bad", "canceled"],
            |_| {},
        )
        .await
        .unwrap();
    assert_eq!(answers[0].as_ref().unwrap().value, 9);
    assert_eq!(answers[2].as_ref().unwrap().value, 9);
    for i in [1, 3, 4] {
        assert!(answers[i].is_err());
    }
    assert_eq!(c.batch_usage().prompt_tokens, 100); // not twice for a duplicate
    let requests = task.await.unwrap();
    let body: Value = serde_json::from_str(requests[0].split_once("\r\n\r\n").unwrap().1).unwrap();
    assert_eq!(body["requests"].as_array().unwrap().len(), 4);
    assert!(body["requests"][0]["params"]
        .get("reasoning_effort")
        .is_none());
    let cached = client().with_cache_directory(dir.path()).with_cached_only();
    let answer: Answer = cached
        .chat_with_system_prompt("Return a value.", "good")
        .await
        .unwrap();
    assert_eq!(answer.value, 9);
    assert!(matches!(
        cached
            .chat_with_system_prompt::<Answer>("Return a value.", "expired")
            .await,
        Err(ChatError::CacheMiss)
    ));
    assert!(c.pending_batch(&request(&c, "expired")).await.is_none());
    let repaired_rows = result_line(&c, "expired", success("{\"value\":2}"));
    let (url, task) = server(vec![
        (
            "POST /v1/messages/batches ",
            200,
            batch("in_progress", 0, 0).to_string(),
        ),
        (
            "GET /v1/messages/batches/msgbatch_test ",
            200,
            batch("ended", 1, 0).to_string(),
        ),
        ("GET /v1/results ", 200, repaired_rows),
    ])
    .await;
    let repaired = client().with_cache_directory(dir.path()).with_url(url);
    let answers = repaired
        .batch_chat_with_system_prompt::<Answer>("Return a value.", vec!["expired"], |_| {})
        .await
        .unwrap();
    assert_eq!(answers[0].as_ref().unwrap().value, 2);
    task.await.unwrap();
}

#[tokio::test]
async fn cancellation_race_still_harvests_and_expired_results_forget_pending_ids() {
    let (url, task) = server(vec![
        (
            "GET /v1/messages/batches/msgbatch_test ",
            200,
            batch("in_progress", 0, 0).to_string(),
        ),
        (
            "POST /v1/messages/batches/msgbatch_test/cancel ",
            400,
            "already ended".into(),
        ),
        (
            "GET /v1/messages/batches/msgbatch_test ",
            200,
            batch("ended", 1, 0).to_string(),
        ),
    ])
    .await;
    let c = client().with_url(url);
    let ended = c
        .wait_anthropic_batch("msgbatch_test", true, &mut |_| {})
        .await
        .unwrap();
    assert_eq!(ended.processing_status, "ended");
    task.await.unwrap();

    let dir = tempfile::tempdir().unwrap();
    let c = client().with_cache_directory(dir.path());
    c.set_pending(&request(&c, "x"), "msgbatch_test")
        .await
        .unwrap();
    let (url, task) = server(vec![
        (
            "GET /v1/messages/batches/msgbatch_test ",
            200,
            batch("ended", 1, 0).to_string(),
        ),
        ("GET /v1/results ", 404, "results expired".into()),
    ])
    .await;
    let c = c.with_url(url);
    let results = c
        .batch_chat_with_system_prompt::<Answer>("Return a value.", vec!["x"], |_| {})
        .await
        .unwrap();
    assert!(results[0]
        .as_ref()
        .unwrap_err()
        .to_string()
        .contains("rerun"));
    assert!(c.pending_batch(&request(&c, "x")).await.is_none());
    task.await.unwrap();
}

#[tokio::test]
async fn live_batch_failure_preserves_source_for_duplicate_prompts() {
    use std::error::Error;

    let (url, task) = server(vec![(
        "POST /v1/messages ",
        400,
        json!({"type":"error", "error":{"type":"invalid_request_error", "message":"Insufficient credits"}}).to_string(),
    )])
    .await;
    let c = client().with_url(url).with_small_batch_threshold(1);
    let results = c
        .batch_chat::<Answer>(vec!["x", "x"], |_| panic!("must not batch"))
        .await
        .unwrap();
    for result in results {
        let error = result.unwrap_err();
        assert!(matches!(
            error.source().unwrap().downcast_ref::<ChatError>(),
            Some(ChatError::HttpStatus { status: 400, .. })
        ));
    }
    assert_eq!(task.await.unwrap().len(), 1);
}

#[tokio::test]
async fn small_batch_counts_unique_prompts_and_uses_live_cache() {
    let (url, task) = server(vec![(
        "POST /v1/messages ",
        200,
        message("{\"value\":8}", "end_turn").to_string(),
    )])
    .await;
    let c = client().with_url(url).with_small_batch_threshold(1);
    let results = c
        .batch_chat::<Answer>(vec!["x", "x", "x"], |_| panic!("must not batch"))
        .await
        .unwrap();
    assert!(results.iter().all(|r| r.as_ref().unwrap().value == 8));
    assert_eq!(c.usage().prompt_tokens, 100);
    assert_eq!(c.batch_usage().prompt_tokens, 0);
    let c = c.with_cached_only();
    assert_eq!(c.chat::<Answer>("x").await.unwrap().value, 8);
    let cached = c
        .batch_chat::<Answer>(vec!["x"], |_| panic!("cache hit"))
        .await
        .unwrap();
    assert_eq!(cached[0].as_ref().unwrap().value, 8);
    assert_eq!(c.usage().prompt_tokens, 100);
    task.await.unwrap();
}

#[tokio::test]
async fn resumes_disk_pending_cancels_and_harvests_before_live_fallback() {
    let dir = tempfile::tempdir().unwrap();
    let old = client().with_cache_directory(dir.path());
    for text in ["good", "pending"] {
        old.set_pending(&request(&old, text), "msgbatch_test")
            .await
            .unwrap();
    }
    let rows = [
        result_line(&old, "good", success("{\"value\":1}")),
        result_line(&old, "pending", json!({"type":"canceled"})),
    ]
    .join("\n");
    let (url, task) = server(vec![
        (
            "GET /v1/messages/batches/msgbatch_test ",
            200,
            batch("in_progress", 1, 0).to_string(),
        ),
        (
            "POST /v1/messages/batches/msgbatch_test/cancel ",
            200,
            batch("canceling", 1, 0).to_string(),
        ),
        (
            "GET /v1/messages/batches/msgbatch_test ",
            200,
            batch("ended", 1, 1).to_string(),
        ),
        ("GET /v1/results ", 200, rows),
        (
            "POST /v1/messages ",
            200,
            message("{\"value\":2}", "end_turn").to_string(),
        ),
    ])
    .await;
    let c = client()
        .with_cache_directory(dir.path())
        .with_url(url)
        .with_no_batch();
    let results = c
        .batch_chat_with_system_prompt::<Answer>("Return a value.", vec!["good", "pending"], |_| {})
        .await
        .unwrap();
    assert_eq!(results[0].as_ref().unwrap().value, 1);
    assert_eq!(results[1].as_ref().unwrap().value, 2);
    assert_eq!(c.usage().prompt_tokens, 100);
    assert_eq!(c.batch_usage().prompt_tokens, 100);
    assert!(c.pending_batch(&request(&c, "pending")).await.is_none());
    task.await.unwrap();
}
