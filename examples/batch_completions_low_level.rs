use std::collections::HashMap;

use tysm::{
    batch::{BatchClient, BatchRequestItem, BatchStatus},
    chat_completions::{ChatClient, ChatMessage, ChatRequest, ResponseFormat},
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create a client
    let client = ChatClient::from_env("gpt-4o").unwrap();
    let batch_client = BatchClient::from(&client);

    println!("Creating batch requests...");

    // Create batch requests
    let requests = vec![
        BatchRequestItem::new_chat(
            "request-1",
            ChatRequest {
                model: "gpt-3.5-turbo".to_string(),
                messages: vec![
                    ChatMessage::system("You are a helpful assistant."),
                    ChatMessage::user("What is the capital of France?"),
                ],
                response_format: ResponseFormat::Text,
                extra_body: None,
                service_tier: None,
                reasoning_effort: None,
                prompt_cache_key: None,
            },
        ),
        BatchRequestItem::new_chat(
            "request-2",
            ChatRequest {
                model: "gpt-3.5-turbo".to_string(),
                messages: vec![
                    ChatMessage::system("You are a helpful assistant."),
                    ChatMessage::user("What is the capital of Japan?"),
                ],
                response_format: ResponseFormat::Text,
                extra_body: None,
                service_tier: None,
                reasoning_effort: None,
                prompt_cache_key: None,
            },
        ),
        BatchRequestItem::new_chat(
            "request-3",
            ChatRequest {
                model: "gpt-3.5-turbo".to_string(),
                messages: vec![
                    ChatMessage::system("You are a helpful assistant."),
                    ChatMessage::user("What is the capital of Italy?"),
                ],
                response_format: ResponseFormat::Text,
                extra_body: None,
                service_tier: None,
                reasoning_effort: None,
                prompt_cache_key: None,
            },
        ),
    ];

    // Create a batch file
    println!("Creating batch file...");
    let file_id = batch_client
        .upload_batch_file("capitals_batch.jsonl", &requests)
        .await?;
    println!("Batch file created with ID: {}", file_id);

    // Create a batch
    println!("Creating batch...");
    let batch = batch_client
        .create_batch(
            file_id,
            HashMap::from([("name".to_string(), "My Batch".to_string())]),
        )
        .await?;
    println!("Batch created with ID: {}", batch.id);

    // List all batches
    println!("Listing all batches...");
    let batches = batch_client.list_batches().await?;
    println!("Batches: {:?}", batches.len());
    for batch in batches
        .iter()
        .filter(|b| b.status == BatchStatus::InProgress)
    {
        println!("Batch {} in progress since {}", batch.id, batch.created_at);
    }

    // Check batch status
    println!("Checking batch status...");
    let status = batch_client.get_batch_status(&batch.id).await?;
    println!("Batch status: {}", status.status);

    // Wait for batch to complete
    println!("Waiting for batch to complete...");
    println!("(This may take a while, up to 24 hours)");
    let completed_batch = batch_client
        .wait_for_batch(&batch.id, |batch| {
            println!(
                "Batch progress: {}/{} completed, {} failed ({})",
                batch.request_counts.completed,
                batch.request_counts.total,
                batch.request_counts.failed,
                batch.status
            );
        })
        .await?;
    println!("Batch completed!");

    // Get batch results
    println!("Getting batch results...");
    let results = batch_client.get_batch_results(&completed_batch).await?;

    // Display results
    println!("\nBatch Results:");
    println!("==============");
    for result in results {
        if let Some(response) = result.response {
            let content = response.body["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or("No content");
            println!("Result for {}: {}", result.custom_id, content);
        } else if let Some(error) = result.error {
            println!(
                "Error for {}: {} - {}",
                result.custom_id, error.code, error.message
            );
        }
    }

    Ok(())
}
