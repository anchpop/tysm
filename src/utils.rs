/// An error that occurs when the OpenAI API key is not found in the environment.
#[derive(Debug)]
pub struct OpenAiApiKeyError(#[expect(unused)] std::env::VarError);
impl std::fmt::Display for OpenAiApiKeyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Unable to find the OpenAI API key in the environment. Please set the OPENAI_API_KEY environment variable. API keys can be found at <https://platform.openai.com/api-keys>.")
    }
}
impl std::error::Error for OpenAiApiKeyError {}

pub(crate) fn api_key() -> Result<String, OpenAiApiKeyError> {
    #[cfg(feature = "dotenvy")]
    {
        use dotenvy::dotenv;
        dotenv().ok();
    }
    std::env::var("OPENAI_API_KEY").map_err(OpenAiApiKeyError)
}

pub(crate) fn remove_trailing_slash(url: url::Url) -> url::Url {
    let mut url = url;
    let path = url.path().to_string();
    let path = path.trim_end_matches('/');
    url.set_path(path);
    url
}

/// Create a shared reqwest::Client with connection pooling configured for high concurrency.
pub(crate) fn pooled_client() -> reqwest::Client {
    reqwest::Client::builder()
        .pool_max_idle_per_host(256)
        .pool_idle_timeout(std::time::Duration::from_secs(300))
        .build()
        .expect("Failed to build HTTP client")
}

// The on-disk response cache now lives in `crate::cache` (append-only record-log shards).
