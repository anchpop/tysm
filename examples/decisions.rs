//! Run with OPENAI_API_KEY set: cargo run --example decisions -- /tmp/decisions-cache
use std::time::Instant;
use tysm::decisions::{ChoiceOption, ChoiceQuestion, DecisionsClient};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cache = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "./decisions-cache".into());
    let questions = [
        ChoiceQuestion {
            name: Some("japanese_reading".into()),
            instructions: "Choose the reading of the marked Japanese word in its sentence.".into(),
            choices: vec![
                ChoiceOption::new("りょうしん"),
                ChoiceOption::new("ふたおや"),
            ],
        },
        ChoiceQuestion {
            name: Some("mandarin_reading".into()),
            instructions:
                "Choose the standard Mandarin reading of the marked Chinese word in its sentence."
                    .into(),
            choices: vec![
                ChoiceOption::new("liǎo bu qǐ"),
                ChoiceOption::new("liǎo bù qǐ"),
            ],
        },
    ];
    // Fresh clients on both passes demonstrate persistent rather than in-memory caching.
    for pass in 1..=2 {
        let client = DecisionsClient::from_env("gpt-6-luna")?.with_cache_directory(&cache);
        let start = Instant::now();
        let response = client
            .decide(
                "私の【両親】は東京に住んでいます。\n你真【了不起】！",
                &questions,
            )
            .await?;
        println!(
            "pass {pass}: {:?}\n{}",
            start.elapsed(),
            serde_json::to_string_pretty(&response)?
        );
    }
    Ok(())
}
