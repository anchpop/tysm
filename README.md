# tysm - Thank You So Much

**Batteries-included Rust client for OpenAI and Anthropic**

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


## Decisions API

```rust
use tysm::decisions::{ChoiceOption, ChoiceQuestion, DecisionAnswer, DecisionsClient};

let client = DecisionsClient::from_env("gpt-6-luna")?
    .with_cache_directory("./cache")
    .with_max_concurrent_requests(16);
let response = client.decide("私の両親は東京に住んでいます。", &[ChoiceQuestion {
    name: Some("reading".into()),
    instructions: "Choose the reading of 両親 in this sentence.".into(),
    choices: vec![ChoiceOption::new("りょうしん"), ChoiceOption::new("ふたおや")],
}]).await?;
for answer in response.answers {
    match answer {
        DecisionAnswer::Choice(answer) => println!("{:?}: {:?}", answer.choice, answer.probabilities),
        DecisionAnswer::Refusal { name } => eprintln!("Refused: {name:?}"),
    }
}
```

This standalone client supports text input and finite choice questions (2–255 unique
string or boolean values). Questions and choices keep their supplied order; no prompt
is added. Confidence is reported separately from option probabilities. The response
preserves the API's model and complete token usage, including provider cache details.

Live requests only: OpenAI rejects `/v1/decisions` as a Batch API endpoint. Images,
predicate/score questions, reasoning settings, and concurrent cache-miss deduplication
are not implemented. `with_url` configures a compatible base URL. The default
concurrency limit is 16. Transport failures (including closed connections), HTTP 429,
server faults, and transient API capacity errors are retried up to four total attempts.
Retries use the shared chat backoff (2/4/8 seconds for transport; 15/30/60 for capacity),
or the server's integer-seconds `Retry-After` when supplied. HTTP-date headers are not
parsed. Sleeping retries release their concurrency slot. Permanent errors and invalid
responses are returned without retrying.

Successful, validated responses (including explicit refusals) use a separate
exact-response cache namespace keyed by the full serialized request and endpoint URL.
HTTP errors and invalid responses are not cached. Persistent caching reuses results across
fresh clients; cached usage describes the original request, not new spend. As with the
other clients, cache directories contain response content: treat them as
private. Run `cargo run --example decisions -- /tmp/decisions-cache` for Japanese and
Mandarin sentence readings, full serialized responses, and cold/warm latency.

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
    // The response is appended to a shard file in ./cache
    // Subsequent calls with this exact request will use the cached value instead of hitting the API. 
}
```

The cache is an [`osmo`](https://github.com/anchpop/osmo) append-only store under the
`tysm/` namespace. Live and batch requests share entries. Use
`tysm::cache::migrate_legacy` to import older `.kv` shards or per-entry directories.

### Custom API URL

Sometimes people want to use a different completions API. For example, I maintain a wrapper around OpenAI's API that adds a global cache. To switch the URL, just do this:

```rust
let my_api = "https://g7edusstdonmn3vxdh3qdypkrq0wzttx.lambda-url.us-east-1.on.aws/v1/";
let client = Client::from_env("gpt-4o").with_url(my_api);
```

By the way, feel free to use this endpoint if you want, but I don't promise to maintain it forever.

#### "I want to use Anthropic!"

Models named `claude-*` automatically use the native Messages and Message Batches APIs, authenticated with `ANTHROPIC_API_KEY`. No URL override or OpenAI compatibility shim is involved:

```rust
use tysm::chat_completions::ChatClient;

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct Answer { value: u32 }

let client = ChatClient::from_env("claude-haiku-5-5")?
    .with_cache_directory("./cache")
    .with_reasoning_effort("low");
let answer: Answer = client.chat("What is 6 times 7?").await?;
let answers = client.batch_chat::<Answer>(vec!["What is 1 + 2?", "What is 3 + 4?"], |_| {}).await?;
```

`with_provider(Provider::Anthropic)` / `with_provider(Provider::OpenAi)` overrides inference and resets the base URL and HTTP client; call `with_url` afterwards for a proxy. The override does not replace the API key. The low-level `batch::BatchClient` and file/embedding clients remain OpenAI-specific; use `ChatClient::batch_*` for provider-neutral batching.

**Requests.** Typed calls use native `output_config.format` JSON schemas. `with_reasoning_effort` maps `low`, `medium`, `high`, `xhigh`, and `max` directly to `output_config.effort`, with adaptive thinking; OpenAI's `minimal` maps to `low`. `none` is rejected rather than silently enabling thinking (current Sonnet/Opus/Fable cannot disable it). Omitting effort leaves the model's default. Older Claude models may reject adaptive thinking or effort; this adapter targets current models. The default `max_tokens` is 16,384. Override it and other native parameters with `with_extra_body`:

```rust
let client = client.with_extra_body(serde_json::json!({
    "max_tokens": 8192,
    "cache_control": {"type": "ephemeral", "ttl": "1h"}
}));
```

This opt-in automatic prompt caching is separate from tysm's exact-response cache. Anthropic's cache hits are prefix-based and have model-specific minimum token counts; batch hits are best-effort. `prompt_cache_key` and `service_tier="flex"` are OpenAI-only and are rejected. Anthropic accepts `auto` / `standard_only` tiers. Text and URL/base64 images are supported; audio and explicit OpenAI image-detail settings are rejected. System messages must precede the conversation. Raw `Text` works; schema-less `JsonObject` does not. Extra parameters cannot replace the model, messages, system prompt, response schema, or enable streaming.

**Schemas.** The adapter closes objects, changes `oneOf` to `anyOf`, removes `$schema`, and moves unsupported numeric/string/array constraints into descriptions (including `minItems > 1`). Serde still checks the Rust result type, including integer ranges; arbitrary schemars validation annotations are not revalidated locally. Arbitrary-key maps are rejected locally rather than silently reduced to empty objects. Recursive schemas, external references, complex enums, and unsupported regexes are not emulated: Anthropic returns a non-retryable schema error. Refusals, truncations, and invalid typed responses are never cached.

**Batches.** Duplicate prompts are submitted once. Batches split at 100,000 requests or 256 MB of serialized JSON (including wrappers), and all chunks are submitted before waiting. Every succeeded/errored/canceled/expired item is mapped back to its input slot. The callback keeps the existing `Batch` shape: `object="message_batch"`, native counts, and normalized status; OpenAI-only file IDs and timestamps are empty/zero. `cancel_anthropic_batch(id)` requests asynchronous cancellation, which is polled for up to ten minutes before returning an error that can be resumed. Retryable HTTP errors (429, 529, other 5xx, 408/409, transport failures) back off and honor numeric `Retry-After`.

Pending job IDs are persisted alongside the response cache, indexed per exact request, so restarts or calls with a smaller subset resume rather than resubmit. Configure a persistent cache to survive process restarts; without one this only works on the same client. Under `with_no_batch` or the small-batch threshold, existing work is canceled/harvested before remaining misses go live. For OpenAI batches the threshold is compared with what an in-flight batch still owes, so a batch stuck on its last few requests is cancelled, harvested, and finished live. Pending IDs are cleared only after successful results are safely cached. As with any API without submission idempotency, a process dying between server acceptance and saving its batch ID can leave an untracked job.

**Cache and billing.** Anthropic has a separate versioned cache namespace, including model, schema, messages and every request parameter. Live and batch calls share entries. Existing OpenAI keys remain unchanged. Spend includes cache reads and both 5-minute/1-hour writes; batch halves all these rates. Haiku 5.5's >100,000-token premium counts *all* input tokens, including cached tokens, and is applied per request. Failed/refused/truncated responses with usage are billed too.

Offline tests run with `cargo test --workspace`. Paid smoke tests are ignored by default:

```sh
cargo test --test anthropic -- --ignored --nocapture
```

They exercise typed Haiku live calls, a real three-request batch, disk-cache reruns, batch-to-live cache sharing, and actual one-hour prompt-cache writes/reads. Set `ANTHROPIC_API_KEY` before running them.

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

Batching is configured at runtime rather than by feature: `ChatClient::with_small_batch_threshold`
sends a batch's uncached requests live when there are few enough of them, and
`with_no_batch` never submits a new batch at all. Either way a `batch_*` call first
looks for a matching existing batch. Completed, cancelled, and expired batches are
harvested; under `with_no_batch` an in-flight one is cancelled and polled until its
partial results are ready. Successfully mapped responses are cached, and only the
remaining misses go live (or, when batching, into a new, smaller batch). Item errors,
refusals, and invalid responses are not cached.

At the low level, `wait_for_batch` returns completed, cancelled, or expired
batches (and keeps polling `Cancelling`); `get_batch_results` downloads any
available output file, or returns an empty vector for a terminal batch without
one. Only completed batches retry missing IDs while their output file settles.

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
