# tysm - Thank You So Much

**Batteries-included Rust OpenAI Client**

Including...

- **Chat-Completions API**
  - Type-safe API responses via Structured Outputs
  - Automatic schema generation
  - Automatic deserialization
  - Concise interface
  - Automatic local caching of API responses
  - Batch API support
- **Embeddings API**
  - Single and batch requests supported (but not through batch API)
  - Vector similarity functions provided
- **Files API**
  - Create, list, download and delete files

The **Typed Chat Completions** feature is the most interesting part, so most of this readme will focus on that.

[crates.io](https://crates.io/crates/tysm) | [docs.rs](https://docs.rs/tysm/latest/tysm/) | [blog post](https://chadnauseam.com/coding/ai/openai-structured-outputs-are-really-useful)

## Table of Contents
- [tysm - Thank You So Much](#tysm---thank-you-so-much)
  - [Table of Contents](#table-of-contents)
  - [Usage](#usage)
  - [Setup](#setup)
    - [Automatic Caching](#automatic-caching)
    - [Persistent Cache](#persistent-cache)
    - [Syncing the cache to a bucket (Cloudflare R2 / S3)](#syncing-the-cache-to-a-bucket-cloudflare-r2--s3)
    - [Custom API URL](#custom-api-url)
      - ["I want to use Anthropic!"](#i-want-to-use-anthropic)
      - ["I want to use Gemini!"](#i-want-to-use-gemini)
      - ["I want to use Ollama!"](#i-want-to-use-ollama)
  - [Feature flags](#feature-flags)
  - [License](#license)
  - [Backstory](#backstory)
  - [Footguns](#footguns)

A strongly-typed Rust client for OpenAI's ChatGPT API that enforces type-safe responses using [Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs).

## Usage

```rust
use tysm::chat_completions::ChatClient;

/// We want names separated into `first` and `last`.
#[derive(serde::Deserialize, schemars::JsonSchema)]
struct Name {
    first: String,
    last: String,
}

async fn get_president_name() {
    // Create a client.
    // `from_env` will look for an API key under the environment
    // variable "OPENAI_API_KEY"
    // It will also look inside `.env` if such a file exists.
    let client = ChatClient::from_env("gpt-4o").unwrap();
    
    // Request a chat completion from OpenAI and
    // parse the response into our `Name` struct.
    let name: Name = client
        .chat("Who was the first US president?")
        .await
        .unwrap();

    assert_eq!(name.first, "George");
    assert_eq!(name.last, "Washington");
}
```

See the `examples/` directory for examples of the embeddings, files, and batch APIs.

There are 4 basic methods on `ChatClient`. Each one is "lower level" than the previous.

1. [`ChatClient::chat`](https://docs.rs/tysm/latest/tysm/chat_completions/struct.ChatClient.html#method.chat): send one message to the chat-completions API, and deserialize the response into the expected type.
2. [`ChatClient::chat_with_system_prompt`](https://docs.rs/tysm/latest/tysm/chat_completions/struct.ChatClient.html#method.chat_with_system_prompt): send a system prompt and a message to the chat-completions API, and deserialize the response into the expected type. The system prompt is the first parameter.
3. [`ChatClient::chat_with_messages`](https://docs.rs/tysm/latest/tysm/chat_completions/struct.ChatClient.html#method.chat_with_messages): send an arbitrary sequence of messages to the chat-completions API, and deserialize the response into the expected type.
4. [`ChatClient::chat_with_messages_raw`](https://docs.rs/tysm/latest/tysm/chat_completions/struct.ChatClient.html#method.chat_with_messages_raw): send an arbitrary sequence of messages to the chat-completions API, and return the response as-is (without deserializing).

Each one has a corresponding batch equivalent (`batch_chat`, `batch_chat_with_system_prompt`, `batch_chat_with_messages`, `batch_chat_with_messages_raw`). These go through the batch API, which is cheaper and has higher ratelimits, but is much higher-latency. The responses to the batch API stick around in OpenAI's servers for some time, and before starting a new batch request, `tysm` will automatically check if that same request has been made before (and reuse it if so). 


## Setup

1. Get an API key from [OpenAI](https://platform.openai.com/api-keys).
2. Create a `.env` file, and make it look like this:

   ```
   OPENAI_API_KEY=<your api key here>
   ```
3. Add `.env` to your `.gitignore` so you don't accidentally commit it.
4. Add the crate and the necessary dependencies to your Rust project with:
    1. `cargo add tysm serde schemars`.


### Automatic Caching

I'm a big fan of memoization. By default, the last 1024 responses will be stored inside the `ChatClient`. For this reason it can be useful to make a client just once using LazyLock (which is part of the standard library since 1.80).

```rust
use std::sync::LazyLock;
use tysm::chat_completions::ChatClient;

static CLIENT: LazyLock<ChatClient> = LazyLock::new(|| ChatClient::from_env("gpt-4o").unwrap());

fn main() {
    #[derive(tysm::Deserialize, tysm::JsonSchema)]
    struct Name {
        first: String,
        last: String,
    }

    for _ in 0..10_000 {
        // The built-in cache prevents us from going bankrupt
        let _name: Name = CLIENT.chat("Who was the first US president?").await.unwrap();
    }
}
```

### Persistent Cache

You can also save the cache to disk:

```rust
use std::sync::LazyLock;
use tysm::chat_completions::ChatClient;

fn main() {
    let client = ChatClient::from_env("gpt-4o")
        .unwrap()
        .with_cache_directory("./cache");

    #[derive(tysm::Deserialize, tysm::JsonSchema)]
    struct Name {
        first: String,
        last: String,
    }

    let _name: Name = CLIENT.chat("Who was the first US president?").await.unwrap();
    // The response will be written to a file in ./cache
    // Subsequent calls with this exact request will use the cached value instead of hitting the API. 
}
```

### Syncing the cache to a bucket (Cloudflare R2 / S3)

With the `cache-sync` feature you can back the on-disk cache onto an S3-compatible
bucket (e.g. Cloudflare R2). On the first request the local cache directory is warmed
from the bucket; call `flush_cache()` at the end of your run to upload new entries back.
This lets many machines (or CI jobs) share a warm cache without committing it to git.

```toml
[dependencies]
tysm = { version = "0.19", features = ["cache-sync"] }
```

```rust
use tysm::chat_completions::ChatClient;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let client = ChatClient::from_env("gpt-4o")?
        .with_cache_directory("./cache")
        .with_cache_bucket("my-cache-bucket");
        // .with_cache_bucket_prefix("project-a")  // optional key prefix

    #[derive(tysm::Deserialize, tysm::JsonSchema)]
    struct Name { first: String, last: String }

    // First request transparently warms ./cache from the bucket (once per process,
    // even if multiple clients share the directory).
    let _name: Name = client.chat("Who was the first US president?").await?;

    // Upload any new cache entries back to the bucket.
    client.flush_cache().await?;
    Ok(())
}
```

Credentials are read from the environment (the bucket name stays in code):

| Variable | Purpose |
| --- | --- |
| `R2_ACCOUNT_ID` | Cloudflare account id (used to build the endpoint), **or** set `R2_ENDPOINT` directly |
| `R2_ACCESS_KEY_ID` | S3 access key id (falls back to `AWS_ACCESS_KEY_ID`) |
| `R2_SECRET_ACCESS_KEY` | S3 secret key (falls back to `AWS_SECRET_ACCESS_KEY`) |

How it works: cache files are content-addressed, so the **set of file paths** identifies
the cache. A commutative fingerprint (the wrapping sum of per-file hashes) plus per-file
content hashes are stored in the bucket as a small `_tysm_manifest.json` object; when the
local fingerprint already matches, both pull and push skip the LIST/transfer entirely. The
pull is best-effort — if it fails (e.g. missing credentials) it logs a warning and serves
from a cold cache. The same API is available on `EmbeddingsClient`. Don't enable
`cache-sync` for WASM targets.

#### Mutable files (e.g. a shared translation cache)

The default assumes every file is immutable/content-addressed (true for tysm's own
responses). If your cache directory also contains a **mutable** file — say a single JSON
file mapping requests to responses that grows over time — drop a `.tysm-sync.json` at the
cache-dir root to give it a different strategy. It's synced to the bucket too, so every
machine inherits the rule automatically:

```json
{
  "files": [
    { "path": "google_translate/master_cache.json", "strategy": "json_merge" }
  ]
}
```

Strategies (`path` glob supports `*` and `?`):

- **`path`** (default) — immutable; identified by path; transferred once.
- **`content`** — mutable; identified by content hash; last-writer-wins (remote wins on
  pull, local wins on push).
- **`json_merge`** — mutable JSON object; reconciled by **unioning top-level keys** (local
  wins on collisions), so no entries are ever lost even when two machines both append.

### Custom API URL

Sometimes people want to use a different completions API. For example, I maintain a wrapper around OpenAI's API that adds a global cache. To switch the URL, just do this:

```rust
let my_api = "https://g7edusstdonmn3vxdh3qdypkrq0wzttx.lambda-url.us-east-1.on.aws/v1/";
let client = Client::from_env("gpt-4o").with_url(my_api);
```

By the way, feel free to use this endpoint if you want, but I don't promise to maintain it forever.

#### "I want to use Anthropic!"

Anthropic has some limited [OpenAI compatibility](https://docs.anthropic.com/en/api/openai-sdk). But at the time of this writing, they ignore the `response_format` parameter. This means the structured outputs stuff is not going to work. However, you can still use the `ChatClient::chat_with_messages_raw` function just fine:

```rust
use tysm::chat_completions::{ChatClient, ChatMessage, ResponseFormat};
let api_key = std::env::var("ANTHROPIC_API_KEY").unwrap();
let client = ChatClient::new(api_key, "claude-3-7-sonnet-20250219").with_url("https://api.anthropic.com/v1/");
let response = client
  .chat_with_messages_raw(
      vec![
          ChatMessage::system("System prompt goes here"),
          ChatMessage::user("User message goes here"),
      ],
      ResponseFormat::Text, // ignored
  )
  .await?;
```

However, if you can get Claude to output JSON on one line, the non-raw methods may still work. This is because, in the event that the response cannot be decoded into the expected type, `tysm` will then attempt to decode each line individually. 

The Batch API will also not work against Anthropic's API.

#### "I want to use Gemini!"

Gemini luckily does support structured outputs. So you can just use your Gemini API key and set the URL to use the OpenAI compatibility layer.

```rust
use tysm::chat_completions::ChatClient;
let api_key = std::env::var("GEMINI_API_KEY").unwrap();
let client = ChatClient::new(api_key, "gemini-2.0-flash").with_url("https://generativelanguage.googleapis.com/v1beta/openai/");
```

#### "I want to use Ollama!"

```rust
use tysm::chat_completions::ChatClient;
let api_key = "required_but_unused";
let client = ChatClient::new(api_key, "llama2").with_url("httphttp://localhost:11434/v1/");
```

## Feature flags

The following feature flags are available:

1. `dotenvy` - (enabled by default) Enables automatic loading of environment variables from a `.env` file.
2. `cache-sync` - (off by default) Mirror the on-disk cache to an S3-compatible bucket such as Cloudflare R2. See [Syncing the cache to a bucket](#syncing-the-cache-to-a-bucket-cloudflare-r2--s3). Not WASM-compatible.

Example of disabling dotenvy:
```toml
[dependencies]
tysm = { version = "0.2", default-features = false }
```

## License

This project is licensed under the MIT License.

## Backstory

The name stands for "thank you so much", which is what I say I ask ChatGPT a question and get a great answer! If you prefer, it could also stand for "**Ty**ped **S**chema **M**agic".

I like making ChatGPT-wrappers. Unfortunately the rust ecosystem for calling ChatGPT is more anemic than you would think, and it's not very complicated, so I always end up writing my own code for calling it. It's just an API endpoint after all. In my various git repos, I'd estimate I have about 5 implementations of this.

I was in the middle of writing my 6th on a lazy christmas eve when I realized that I'm too lazy to keep doing that. So I decided to solve the problem for myself once and for all.

I almost never use streaming or anything fancy like that so this library doesn't support it. I designed it with my future lazy self in mind - which is why it has `.env` support built in and has built-in caching.

The whole library is basically one file right now, so hopefully it will be easy for you to move on from once you outgrow it.

## Footguns

1. the trait bound `Books: schemars::JsonSchema` is not satisfied

```
error[E0277]: the trait bound `MyStruct: schemars::JsonSchema` is not satisfied
note: required by a bound in `ChatClient::chat`
   --> ~/coding/typed-openai/src/chatgpt.rs:198:64
    |
198 |     pub async fn chat<T: DeserializeOwned + JsonSchema>(
    |                                             ^^^^^^^^^^ required by this bound in `ChatClient::chat`
```

You probably forgot to add the in-development version of Schemars to your project. Try replacing the `schemars` entry in your Cargo.toml with this:

```toml
schemars = { version = "1", features = ["preserve_order"] }
```
