//! Paid smoke tests. Set ANTHROPIC_API_KEY and run with --ignored --nocapture.
//! Temporary local caches ensure a run actually exercises the provider.

use schemars::JsonSchema;
use serde::Deserialize;
use tysm::chat_completions::ChatClient;

#[derive(Debug, Deserialize, JsonSchema, PartialEq)]
struct Sum {
    sum: u32,
}

fn client(cache: &std::path::Path) -> ChatClient {
    ChatClient::from_env("claude-haiku-5-5")
        .unwrap()
        .with_cache_directory(cache)
        .with_reasoning_effort("low")
        .with_extra_body(serde_json::json!({"max_tokens":1024}))
}

#[tokio::test]
#[ignore = "paid Anthropic live API smoke test"]
async fn anthropic_live() {
    let dir = tempfile::tempdir().unwrap();
    let client = client(dir.path());
    let answer: Sum = client
        .chat_with_system_prompt("Return the arithmetic sum.", "What is 20 + 22?")
        .await
        .unwrap();
    assert_eq!(answer.sum, 42);
    println!(
        "live: {answer:?}; usage={:?}; cost=${:.8}",
        client.usage(),
        client.cost().unwrap()
    );
    let before = client.usage();
    let cached: Sum = client
        .chat_with_system_prompt("Return the arithmetic sum.", "What is 20 + 22?")
        .await
        .unwrap();
    assert_eq!(cached, answer);
    assert_eq!(client.usage(), before);
    println!("live rerun: cache hit, zero additional spend");
}

#[tokio::test]
#[ignore = "paid Anthropic prompt cache write/read smoke test"]
async fn anthropic_prompt_cache() {
    let first_dir = tempfile::tempdir().unwrap();
    let second_dir = tempfile::tempdir().unwrap();
    let options =
        serde_json::json!({"max_tokens":1024,"cache_control":{"type":"ephemeral","ttl":"1h"}});
    let first = client(first_dir.path()).with_extra_body(options.clone());
    let second = client(second_dir.path()).with_extra_body(options);
    let system = format!(
        "Return the arithmetic sum. Cache probe {:?}. Background notes are irrelevant: {}",
        std::time::SystemTime::now(),
        "The sky is blue and the grass is green. ".repeat(100)
    );
    let answer: Sum = first
        .chat_with_system_prompt(&system, "What is 2 + 2?")
        .await
        .unwrap();
    assert_eq!(answer.sum, 4);
    assert!(
        first
            .usage()
            .prompt_tokens_details
            .unwrap()
            .cache_creation_1h_tokens
            > 0
    );
    let answer: Sum = second
        .chat_with_system_prompt(&system, "What is 2 + 2?")
        .await
        .unwrap();
    assert_eq!(answer.sum, 4);
    assert!(second.usage().prompt_tokens_details.unwrap().cached_tokens > 0);
    println!(
        "prompt cache write: {:?}; cost=${:.8}",
        first.usage(),
        first.cost().unwrap()
    );
    println!(
        "prompt cache read: {:?}; cost=${:.8}",
        second.usage(),
        second.cost().unwrap()
    );
}

#[tokio::test]
#[ignore = "paid Anthropic Message Batches smoke test; may take up to 24 hours"]
async fn anthropic_batch() {
    let dir = tempfile::tempdir().unwrap();
    let first = client(dir.path());
    let prompts = vec!["What is 1 + 2?", "What is 3 + 4?", "What is 5 + 6?"];
    let mut previous = None;
    let answers = first
        .batch_chat_with_system_prompt::<Sum>(
            "Return the arithmetic sum.",
            prompts.clone(),
            |batch| {
                let counts = (batch.request_counts.completed, batch.request_counts.failed);
                if previous != Some(counts) {
                    println!(
                        "batch {}: {:?}, succeeded={}, failed={}",
                        batch.id, batch.status, counts.0, counts.1
                    );
                    previous = Some(counts);
                }
            },
        )
        .await
        .unwrap();
    let answers: Vec<_> = answers.into_iter().collect::<Result<_, _>>().unwrap();
    assert_eq!(
        answers.iter().map(|a| a.sum).collect::<Vec<_>>(),
        [3, 7, 11]
    );
    assert!(first.batch_usage().prompt_tokens > 0);
    assert_eq!(first.usage().total_tokens, 0);
    println!(
        "batch: {answers:?}; usage={:?}; cost=${:.8}",
        first.batch_usage(),
        first.cost().unwrap()
    );

    // A new client proves the disk cache works, and cached-only makes any
    // accidental resubmission fail. Live and batch calls use the same entries.
    let cached = client(dir.path()).with_cached_only();
    let rerun = cached
        .batch_chat_with_system_prompt::<Sum>("Return the arithmetic sum.", prompts.clone(), |_| {
            panic!("cache hit must not submit a batch")
        })
        .await
        .unwrap();
    assert_eq!(
        rerun.into_iter().collect::<Result<Vec<_>, _>>().unwrap(),
        answers
    );
    for (prompt, answer) in prompts.into_iter().zip(answers) {
        let live: Sum = cached
            .chat_with_system_prompt("Return the arithmetic sum.", prompt)
            .await
            .unwrap();
        assert_eq!(live, answer);
    }
    assert_eq!(cached.cost(), Some(0.0));
    println!("batch rerun and live reads: 3 disk cache hits each, cost=$0");
}
