//! Sync the on-disk response cache to/from an S3-compatible bucket (Cloudflare R2).
//!
//! The on-disk cache (see [`crate::utils`]) stores each response as a
//! content-addressed file at `<dir>/<shard>/<cache_key>`. Because the file name is a
//! hash of the request, the *set of relative paths* fully identifies the cache
//! contents. This module mirrors that set to/from a bucket:
//!
//! - **pull** (automatic, once per process per bucket+dir): download any objects that
//!   aren't already present locally, warming a cold cache.
//! - **push** ([`ChatClient::flush_cache`](crate::chat_completions::ChatClient::flush_cache),
//!   explicit): upload any local files not yet in the bucket.
//!
//! A commutative fingerprint (the wrapping sum of `xxh3` hashes of the relative paths)
//! is stored in the bucket as a small `_fingerprint` object. When the local fingerprint
//! already matches the remote one, both pull and push skip the expensive LIST/transfer.
//!
//! Credentials come from the environment; the bucket name is configured in code via
//! [`with_cache_bucket`](crate::chat_completions::ChatClient::with_cache_bucket).

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock};
use std::time::{SystemTime, UNIX_EPOCH};

use dashmap::DashMap;
use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};
use tokio::sync::{Mutex, OnceCell};

type HmacSha256 = Hmac<Sha256>;

/// The bucket a client's cache is mirrored to. Created by
/// [`with_cache_bucket`](crate::chat_completions::ChatClient::with_cache_bucket).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheBucket {
    /// The bucket name.
    pub bucket: String,
    /// An optional key prefix within the bucket (default empty).
    pub prefix: String,
}

impl CacheBucket {
    pub(crate) fn new(bucket: impl Into<String>) -> Self {
        Self {
            bucket: bucket.into(),
            prefix: String::new(),
        }
    }
}

/// Outcome of a [`pull`](ensure_pulled) or [`push`](flush) operation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheSyncStats {
    /// Number of objects downloaded from the bucket into the local cache.
    pub downloaded: usize,
    /// Number of local files uploaded to the bucket.
    pub uploaded: usize,
    /// True if the fingerprint matched and the LIST/transfer was skipped entirely.
    pub skipped: bool,
}

/// Errors that can occur while syncing the cache with a bucket.
#[derive(Debug, thiserror::Error)]
pub enum CacheSyncError {
    /// Required credentials/configuration were not found in the environment.
    #[error("cache-sync credentials not configured: {0}")]
    MissingCredentials(String),
    /// The HTTP request to the bucket failed.
    #[error("cache-sync HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    /// A filesystem error occurred while reading/writing the cache directory.
    #[error("cache-sync IO error: {0}")]
    Io(#[from] std::io::Error),
    /// The bucket returned a non-success status code.
    #[error("cache-sync request to {key} failed with status {status}: {body}")]
    BadStatus {
        /// The object key (or `?list` for a listing).
        key: String,
        /// The HTTP status code.
        status: u16,
        /// The (truncated) response body.
        body: String,
    },
}

// ===================================================================================
// Public entry points (deduped per process)
// ===================================================================================

type SyncKey = (String, PathBuf);

static PULL_ONCE: LazyLock<DashMap<SyncKey, Arc<OnceCell<()>>>> = LazyLock::new(DashMap::new);
static FLUSH_LOCK: LazyLock<DashMap<SyncKey, Arc<Mutex<()>>>> = LazyLock::new(DashMap::new);

fn sync_key(dir: &Path, bucket: &CacheBucket) -> SyncKey {
    let dir = std::fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf());
    (format!("{}/{}", bucket.bucket, bucket.prefix), dir)
}

/// Warm the local cache directory from the bucket. Runs at most once per process for a
/// given (bucket, dir); concurrent callers await the same operation. Best-effort: any
/// error is logged and swallowed so a sync failure never breaks request serving.
pub async fn ensure_pulled(dir: &Path, bucket: &CacheBucket) {
    let key = sync_key(dir, bucket);
    let cell = PULL_ONCE
        .entry(key)
        .or_insert_with(|| Arc::new(OnceCell::new()))
        .clone();

    let dir = dir.to_path_buf();
    let bucket = bucket.clone();
    cell.get_or_init(|| async move {
        match pull(&dir, &bucket).await {
            Ok(stats) if !stats.skipped => {
                log::info!(
                    "tysm cache: pulled {} object(s) from bucket {}",
                    stats.downloaded,
                    bucket.bucket
                );
            }
            Ok(_) => log::debug!("tysm cache: already in sync with bucket {}", bucket.bucket),
            Err(e) => log::warn!("tysm cache: pull from bucket {} failed: {e}", bucket.bucket),
        }
    })
    .await;
}

/// Push local cache files to the bucket. Serialized per (bucket, dir) so sibling clients
/// sharing a directory don't duplicate work; repeat calls are cheap no-ops once the
/// fingerprint matches.
pub async fn flush(dir: &Path, bucket: &CacheBucket) -> Result<CacheSyncStats, CacheSyncError> {
    let key = sync_key(dir, bucket);
    let lock = FLUSH_LOCK
        .entry(key)
        .or_insert_with(|| Arc::new(Mutex::new(())))
        .clone();
    let _guard = lock.lock().await;
    push(dir, bucket).await
}

// ===================================================================================
// Pull / push
// ===================================================================================

const FINGERPRINT_REL: &str = "_fingerprint";

async fn pull(dir: &Path, bucket: &CacheBucket) -> Result<CacheSyncStats, CacheSyncError> {
    // Ensure the cache directory exists even if the bucket is empty or unreachable, so the
    // request path's "cache directory must exist" invariant holds after warming.
    tokio::fs::create_dir_all(dir).await?;

    let cfg = R2Config::from_env()?;
    let client = crate::utils::pooled_client();

    let local = list_cache_files(dir).await?;
    let local_keys: HashSet<String> = local.iter().map(|(k, _)| k.clone()).collect();
    let local_fp = fingerprint(&local_keys);

    // Fast path: if the bucket's recorded fingerprint matches local, we're in sync.
    if let Some(remote_fp) = get_fingerprint(&client, &cfg, bucket).await? {
        if remote_fp == local_fp {
            return Ok(CacheSyncStats {
                skipped: true,
                ..Default::default()
            });
        }
    }

    let prefix = &bucket.prefix;
    let remote_keys = list_objects(&client, &cfg, &bucket.bucket, prefix).await?;

    let mut downloaded = 0;
    for full_key in remote_keys {
        let Some(rel) = strip_prefix(&full_key, prefix) else {
            continue;
        };
        if rel == FINGERPRINT_REL || local_keys.contains(rel) {
            continue;
        }
        if let Some(bytes) = get_object(&client, &cfg, &bucket.bucket, &full_key).await? {
            let dest = dir.join(rel);
            if let Some(parent) = dest.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }
            tokio::fs::write(&dest, &bytes).await?;
            downloaded += 1;
        }
    }

    Ok(CacheSyncStats {
        downloaded,
        ..Default::default()
    })
}

async fn push(dir: &Path, bucket: &CacheBucket) -> Result<CacheSyncStats, CacheSyncError> {
    let cfg = R2Config::from_env()?;
    let client = crate::utils::pooled_client();

    let local = list_cache_files(dir).await?;
    let local_keys: HashSet<String> = local.iter().map(|(k, _)| k.clone()).collect();
    let local_fp = fingerprint(&local_keys);

    if let Some(remote_fp) = get_fingerprint(&client, &cfg, bucket).await? {
        if remote_fp == local_fp {
            return Ok(CacheSyncStats {
                skipped: true,
                ..Default::default()
            });
        }
    }

    let prefix = &bucket.prefix;
    let remote_full = list_objects(&client, &cfg, &bucket.bucket, prefix).await?;
    let remote_rel: HashSet<String> = remote_full
        .iter()
        .filter_map(|k| strip_prefix(k, prefix))
        .filter(|r| *r != FINGERPRINT_REL)
        .map(|r| r.to_string())
        .collect();

    let mut uploaded = 0;
    for (rel, path) in &local {
        if remote_rel.contains(rel) {
            continue;
        }
        let bytes = tokio::fs::read(path).await?;
        put_object(&client, &cfg, &bucket.bucket, &obj_key(prefix, rel), bytes).await?;
        uploaded += 1;
    }

    // Record the fingerprint of the *union* (what the bucket now contains) so a future
    // run that holds the same set hits the fast path.
    let union: HashSet<String> = local_keys.union(&remote_rel).cloned().collect();
    let union_fp = fingerprint(&union);
    put_object(
        &client,
        &cfg,
        &bucket.bucket,
        &obj_key(prefix, FINGERPRINT_REL),
        union_fp.to_string().into_bytes(),
    )
    .await?;

    Ok(CacheSyncStats {
        uploaded,
        ..Default::default()
    })
}

async fn get_fingerprint(
    client: &reqwest::Client,
    cfg: &R2Config,
    bucket: &CacheBucket,
) -> Result<Option<u64>, CacheSyncError> {
    let key = obj_key(&bucket.prefix, FINGERPRINT_REL);
    let Some(bytes) = get_object(client, cfg, &bucket.bucket, &key).await? else {
        return Ok(None);
    };
    Ok(String::from_utf8(bytes)
        .ok()
        .and_then(|s| s.trim().parse().ok()))
}

// ===================================================================================
// Local cache directory helpers
// ===================================================================================

/// The wrapping sum of `xxh3` hashes of the relative cache paths. Commutative, so it is
/// independent of iteration order and equal iff the two directories hold the same key set.
fn fingerprint(keys: &HashSet<String>) -> u64 {
    use xxhash_rust::xxh3::xxh3_64;
    keys.iter()
        .fold(0u64, |acc, k| acc.wrapping_add(xxh3_64(k.as_bytes())))
}

/// Walk `dir`, returning `(relative-key, absolute-path)` for every file. The relative key
/// uses `/` separators (e.g. `042/<cache_key>`). Missing directory ⇒ empty list.
async fn list_cache_files(dir: &Path) -> Result<Vec<(String, PathBuf)>, std::io::Error> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let mut rd = match tokio::fs::read_dir(&d).await {
            Ok(rd) => rd,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => return Err(e),
        };
        while let Some(entry) = rd.next_entry().await? {
            let path = entry.path();
            let ft = entry.file_type().await?;
            if ft.is_dir() {
                stack.push(path);
            } else if ft.is_file() {
                let rel = path
                    .strip_prefix(dir)
                    .unwrap_or(&path)
                    .components()
                    .map(|c| c.as_os_str().to_string_lossy())
                    .collect::<Vec<_>>()
                    .join("/");
                if rel == FINGERPRINT_REL {
                    continue;
                }
                out.push((rel, path));
            }
        }
    }
    Ok(out)
}

fn obj_key(prefix: &str, rel: &str) -> String {
    if prefix.is_empty() {
        rel.to_string()
    } else {
        format!("{prefix}/{rel}")
    }
}

/// Strip `<prefix>/` from a full object key, returning the relative key.
fn strip_prefix<'a>(full_key: &'a str, prefix: &str) -> Option<&'a str> {
    if prefix.is_empty() {
        Some(full_key)
    } else {
        full_key
            .strip_prefix(prefix)
            .and_then(|r| r.strip_prefix('/'))
    }
}

// ===================================================================================
// S3 configuration & requests (SigV4 over reqwest)
// ===================================================================================

struct R2Config {
    /// `https://<host>` with no trailing slash.
    endpoint_base: String,
    host: String,
    region: String,
    access_key: String,
    secret_key: String,
}

fn env_var(name: &str) -> Option<String> {
    #[cfg(feature = "dotenvy")]
    {
        let _ = dotenvy::dotenv();
    }
    std::env::var(name).ok().filter(|s| !s.is_empty())
}

impl R2Config {
    fn from_env() -> Result<Self, CacheSyncError> {
        let endpoint = if let Some(e) = env_var("R2_ENDPOINT") {
            e
        } else {
            let acct = env_var("R2_ACCOUNT_ID").ok_or_else(|| {
                CacheSyncError::MissingCredentials("set R2_ENDPOINT or R2_ACCOUNT_ID".to_string())
            })?;
            format!("https://{acct}.r2.cloudflarestorage.com")
        };
        let url = url::Url::parse(&endpoint).map_err(|e| {
            CacheSyncError::MissingCredentials(format!("invalid R2 endpoint {endpoint:?}: {e}"))
        })?;
        let host = url
            .host_str()
            .ok_or_else(|| {
                CacheSyncError::MissingCredentials(format!("R2 endpoint has no host: {endpoint:?}"))
            })?
            .to_string();
        let endpoint_base = format!("{}://{}", url.scheme(), host);

        let access_key = env_var("R2_ACCESS_KEY_ID")
            .or_else(|| env_var("AWS_ACCESS_KEY_ID"))
            .ok_or_else(|| {
                CacheSyncError::MissingCredentials(
                    "set R2_ACCESS_KEY_ID (or AWS_ACCESS_KEY_ID)".to_string(),
                )
            })?;
        let secret_key = env_var("R2_SECRET_ACCESS_KEY")
            .or_else(|| env_var("AWS_SECRET_ACCESS_KEY"))
            .ok_or_else(|| {
                CacheSyncError::MissingCredentials(
                    "set R2_SECRET_ACCESS_KEY (or AWS_SECRET_ACCESS_KEY)".to_string(),
                )
            })?;
        let region = env_var("R2_REGION").unwrap_or_else(|| "auto".to_string());

        Ok(Self {
            endpoint_base,
            host,
            region,
            access_key,
            secret_key,
        })
    }

    /// Sign and send a request, returning `(status, body)`.
    async fn send(
        &self,
        client: &reqwest::Client,
        method: &str,
        bucket: &str,
        key: &str,
        query: &[(&str, &str)],
        body: Vec<u8>,
    ) -> Result<(reqwest::StatusCode, Vec<u8>), CacheSyncError> {
        let (date, datetime) = amz_dates(SystemTime::now());
        let payload_hash = sha256_hex(&body);

        let canonical_uri = if key.is_empty() {
            format!("/{}", uri_encode(bucket, true))
        } else {
            format!("/{}/{}", uri_encode(bucket, true), uri_encode(key, false))
        };

        let mut qp: Vec<(String, String)> = query
            .iter()
            .map(|(k, v)| (uri_encode(k, true), uri_encode(v, true)))
            .collect();
        qp.sort();
        let canonical_query = qp
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join("&");

        let headers = [
            ("host".to_string(), self.host.clone()),
            ("x-amz-content-sha256".to_string(), payload_hash.clone()),
            ("x-amz-date".to_string(), datetime.clone()),
        ];
        let authorization = authorization_header(&SignParams {
            method,
            canonical_uri: &canonical_uri,
            canonical_query: &canonical_query,
            headers: &headers,
            payload_hash: &payload_hash,
            datetime: &datetime,
            date: &date,
            region: &self.region,
            service: "s3",
            access_key: &self.access_key,
            secret_key: &self.secret_key,
        });

        let mut url = format!("{}{}", self.endpoint_base, canonical_uri);
        if !canonical_query.is_empty() {
            url.push('?');
            url.push_str(&canonical_query);
        }

        let m = reqwest::Method::from_bytes(method.as_bytes()).expect("valid method");
        let mut rb = client
            .request(m, &url)
            .header("x-amz-content-sha256", &payload_hash)
            .header("x-amz-date", &datetime)
            .header("authorization", authorization);
        if !body.is_empty() {
            rb = rb.body(body);
        }
        let resp = rb.send().await?;
        let status = resp.status();
        let bytes = resp.bytes().await?.to_vec();
        Ok((status, bytes))
    }
}

async fn list_objects(
    client: &reqwest::Client,
    cfg: &R2Config,
    bucket: &str,
    prefix: &str,
) -> Result<Vec<String>, CacheSyncError> {
    let mut keys = Vec::new();
    let mut continuation: Option<String> = None;
    let prefix_param = if prefix.is_empty() {
        String::new()
    } else {
        format!("{prefix}/")
    };

    loop {
        let mut query: Vec<(&str, &str)> = vec![("list-type", "2")];
        if !prefix_param.is_empty() {
            query.push(("prefix", &prefix_param));
        }
        if let Some(token) = &continuation {
            query.push(("continuation-token", token));
        }

        let (status, body) = cfg
            .send(client, "GET", bucket, "", &query, Vec::new())
            .await?;
        if !status.is_success() {
            return Err(CacheSyncError::BadStatus {
                key: "?list".to_string(),
                status: status.as_u16(),
                body: truncate(&String::from_utf8_lossy(&body)),
            });
        }
        let xml = String::from_utf8_lossy(&body);
        keys.extend(extract_tags(&xml, "Key"));

        match extract_tags(&xml, "NextContinuationToken")
            .into_iter()
            .next()
        {
            Some(token)
                if extract_tags(&xml, "IsTruncated")
                    .first()
                    .map(String::as_str)
                    == Some("true") =>
            {
                continuation = Some(token);
            }
            _ => break,
        }
    }
    Ok(keys)
}

async fn get_object(
    client: &reqwest::Client,
    cfg: &R2Config,
    bucket: &str,
    key: &str,
) -> Result<Option<Vec<u8>>, CacheSyncError> {
    let (status, body) = cfg
        .send(client, "GET", bucket, key, &[], Vec::new())
        .await?;
    if status.is_success() {
        Ok(Some(body))
    } else if status == reqwest::StatusCode::NOT_FOUND {
        Ok(None)
    } else {
        Err(CacheSyncError::BadStatus {
            key: key.to_string(),
            status: status.as_u16(),
            body: truncate(&String::from_utf8_lossy(&body)),
        })
    }
}

async fn put_object(
    client: &reqwest::Client,
    cfg: &R2Config,
    bucket: &str,
    key: &str,
    body: Vec<u8>,
) -> Result<(), CacheSyncError> {
    let (status, body_resp) = cfg.send(client, "PUT", bucket, key, &[], body).await?;
    if status.is_success() {
        Ok(())
    } else {
        Err(CacheSyncError::BadStatus {
            key: key.to_string(),
            status: status.as_u16(),
            body: truncate(&String::from_utf8_lossy(&body_resp)),
        })
    }
}

fn truncate(s: &str) -> String {
    s.chars().take(300).collect()
}

/// Extract the text content of every `<tag>...</tag>` occurrence, XML-unescaping the value.
fn extract_tags(xml: &str, tag: &str) -> Vec<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let mut out = Vec::new();
    let mut rest = xml;
    while let Some(start) = rest.find(&open) {
        let Some(after) = rest.get(start + open.len()..) else {
            break;
        };
        let Some(end) = after.find(&close) else {
            break;
        };
        if let Some(inner) = after.get(..end) {
            out.push(xml_unescape(inner));
        }
        rest = match after.get(end + close.len()..) {
            Some(r) => r,
            None => break,
        };
    }
    out
}

fn xml_unescape(s: &str) -> String {
    s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
}

// ===================================================================================
// SigV4 signing
// ===================================================================================

struct SignParams<'a> {
    method: &'a str,
    canonical_uri: &'a str,
    canonical_query: &'a str,
    headers: &'a [(String, String)],
    payload_hash: &'a str,
    datetime: &'a str,
    date: &'a str,
    region: &'a str,
    service: &'a str,
    access_key: &'a str,
    secret_key: &'a str,
}

/// Compute the `Authorization` header value for an AWS SigV4 (`s3`) request.
fn authorization_header(p: &SignParams) -> String {
    let mut headers = p.headers.to_vec();
    headers.sort_by(|a, b| a.0.cmp(&b.0));

    let canonical_headers: String = headers
        .iter()
        .map(|(k, v)| format!("{}:{}\n", k, v.trim()))
        .collect();
    let signed_headers = headers
        .iter()
        .map(|(k, _)| k.as_str())
        .collect::<Vec<_>>()
        .join(";");

    let canonical_request = format!(
        "{}\n{}\n{}\n{}\n{}\n{}",
        p.method,
        p.canonical_uri,
        p.canonical_query,
        canonical_headers,
        signed_headers,
        p.payload_hash
    );

    let scope = format!("{}/{}/{}/aws4_request", p.date, p.region, p.service);
    let string_to_sign = format!(
        "AWS4-HMAC-SHA256\n{}\n{}\n{}",
        p.datetime,
        scope,
        sha256_hex(canonical_request.as_bytes())
    );

    let k_date = hmac(
        format!("AWS4{}", p.secret_key).as_bytes(),
        p.date.as_bytes(),
    );
    let k_region = hmac(&k_date, p.region.as_bytes());
    let k_service = hmac(&k_region, p.service.as_bytes());
    let k_signing = hmac(&k_service, b"aws4_request");
    let signature = hex::encode(hmac(&k_signing, string_to_sign.as_bytes()));

    format!(
        "AWS4-HMAC-SHA256 Credential={}/{}, SignedHeaders={}, Signature={}",
        p.access_key, scope, signed_headers, signature
    )
}

fn sha256_hex(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    hex::encode(h.finalize())
}

fn hmac(key: &[u8], data: &[u8]) -> Vec<u8> {
    let mut mac = <HmacSha256 as Mac>::new_from_slice(key).expect("HMAC accepts any key length");
    mac.update(data);
    mac.finalize().into_bytes().to_vec()
}

/// AWS-style percent-encoding. `unreserved = A-Za-z0-9-._~`; everything else is encoded.
/// When `encode_slash` is false, `/` is left as-is (for object key paths).
fn uri_encode(s: &str, encode_slash: bool) -> String {
    let mut out = String::with_capacity(s.len());
    for &b in s.as_bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                out.push(b as char)
            }
            b'/' if !encode_slash => out.push('/'),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Format a `SystemTime` as `(YYYYMMDD, YYYYMMDDTHHMMSSZ)` in UTC, without a date library.
fn amz_dates(t: SystemTime) -> (String, String) {
    let secs = t
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let days = secs.div_euclid(86400);
    let sod = secs.rem_euclid(86400);
    let (y, m, d) = civil_from_days(days);
    let (hh, mm, ss) = (sod / 3600, (sod % 3600) / 60, sod % 60);
    (
        format!("{y:04}{m:02}{d:02}"),
        format!("{y:04}{m:02}{d:02}T{hh:02}{mm:02}{ss:02}Z"),
    )
}

/// Convert a count of days since the Unix epoch to a `(year, month, day)` civil date.
/// Howard Hinnant's `civil_from_days` algorithm.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    (y + i64::from(m <= 2), m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprint_is_order_independent() {
        let a: HashSet<String> = ["042/aaa", "100/bbb", "999/ccc"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let b: HashSet<String> = ["999/ccc", "042/aaa", "100/bbb"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(fingerprint(&a), fingerprint(&b));

        let mut c = a.clone();
        c.insert("123/ddd".to_string());
        assert_ne!(fingerprint(&a), fingerprint(&c));
    }

    #[test]
    fn civil_date_known_values() {
        // 2013-05-24 is day 15849 since the epoch.
        assert_eq!(civil_from_days(15849), (2013, 5, 24));
        // Epoch itself.
        assert_eq!(civil_from_days(0), (1970, 1, 1));
    }

    #[test]
    fn uri_encode_rules() {
        assert_eq!(uri_encode("a/b c", false), "a/b%20c");
        assert_eq!(uri_encode("a/b c", true), "a%2Fb%20c");
        assert_eq!(uri_encode("-._~AZ09", true), "-._~AZ09");
    }

    /// The official `aws-sig-v4-test-suite` `get-vanilla` vector. Locks down the
    /// canonical-request / string-to-sign / signing-key derivation without a network call.
    /// (Verified independently against the published expected signature.)
    #[test]
    fn sigv4_matches_official_get_vanilla_vector() {
        const EMPTY_SHA256: &str =
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        let headers = [
            ("host".to_string(), "example.amazonaws.com".to_string()),
            ("x-amz-date".to_string(), "20150830T123600Z".to_string()),
        ];
        let auth = authorization_header(&SignParams {
            method: "GET",
            canonical_uri: "/",
            canonical_query: "",
            headers: &headers,
            payload_hash: EMPTY_SHA256,
            datetime: "20150830T123600Z",
            date: "20150830",
            region: "us-east-1",
            service: "service",
            access_key: "AKIDEXAMPLE",
            secret_key: "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
        });
        assert_eq!(
            auth,
            "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request, \
             SignedHeaders=host;x-amz-date, \
             Signature=5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31"
        );
    }
}
