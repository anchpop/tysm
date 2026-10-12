/// An error reading the selected provider's API key from the environment.
#[derive(Debug)]
pub struct ApiKeyError {
    variable: &'static str,
}
impl std::fmt::Display for ApiKeyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Unable to read {}; set it to the provider's API key",
            self.variable
        )
    }
}
impl std::error::Error for ApiKeyError {}

pub(crate) fn api_key() -> Result<String, ApiKeyError> {
    api_key_for("OPENAI_API_KEY")
}

pub(crate) fn api_key_for(variable: &'static str) -> Result<String, ApiKeyError> {
    #[cfg(feature = "dotenvy")]
    dotenvy::dotenv().ok();
    std::env::var(variable).map_err(|_| ApiKeyError { variable })
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
    pooled_client_with_redirects(reqwest::redirect::Policy::default())
}

pub(crate) fn pooled_client_with_redirects(policy: reqwest::redirect::Policy) -> reqwest::Client {
    reqwest::Client::builder()
        .redirect(policy)
        .pool_max_idle_per_host(256)
        .pool_idle_timeout(std::time::Duration::from_secs(300))
        .build()
        .expect("Failed to build HTTP client")
}

// The on-disk response cache now lives in `crate::cache` (append-only record-log shards).
