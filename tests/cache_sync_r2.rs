//! Live round-trip test for the `cache-sync` feature against a real S3-compatible
//! bucket (Cloudflare R2). Ignored by default; run explicitly once credentials exist:
//!
//! ```bash
//! export R2_ACCOUNT_ID=...          # or R2_ENDPOINT=https://<acct>.r2.cloudflarestorage.com
//! export R2_ACCESS_KEY_ID=...
//! export R2_SECRET_ACCESS_KEY=...
//! export TYSM_TEST_BUCKET=tysm-cache-test
//! cargo test -p tysm --features cache-sync --test cache_sync_r2 -- --ignored --nocapture
//! ```
//!
//! It pushes a few fake cache files to the bucket, pulls them into a *separate* local
//! directory, and verifies the contents round-trip and that the fingerprint fast-path
//! engages on a second flush.

#![cfg(feature = "cache-sync")]

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use tysm::cache_sync::{ensure_pulled, flush, CacheBucket};

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|s| !s.is_empty())
}

/// A unique key prefix per run so repeated runs don't collide (we never delete objects).
fn unique_prefix() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("tysm-test/{}-{}", std::process::id(), nanos)
}

async fn write_file(dir: &Path, rel: &str, contents: &[u8]) {
    let path = dir.join(rel);
    tokio::fs::create_dir_all(path.parent().unwrap())
        .await
        .unwrap();
    tokio::fs::write(&path, contents).await.unwrap();
}

#[tokio::test]
#[ignore = "requires live R2 credentials + TYSM_TEST_BUCKET"]
async fn r2_round_trip() {
    let Some(bucket_name) = env("TYSM_TEST_BUCKET") else {
        eprintln!("skipping: TYSM_TEST_BUCKET not set");
        return;
    };
    if env("R2_ACCESS_KEY_ID")
        .or_else(|| env("AWS_ACCESS_KEY_ID"))
        .is_none()
    {
        eprintln!("skipping: R2/AWS access key not set");
        return;
    }

    let bucket = CacheBucket {
        bucket: bucket_name,
        prefix: unique_prefix(),
    };

    let src = tempdir();
    let dst = tempdir();

    // Two fake cache entries in the sharded layout. Sync is content-agnostic.
    write_file(src.path(), "042/aaaaaaaa", b"first response").await;
    write_file(src.path(), "999/bbbbbbbb", b"second response").await;

    // Push.
    let pushed = flush(src.path(), &bucket).await.expect("push");
    assert_eq!(pushed.uploaded, 2, "expected to upload both entries");
    assert!(!pushed.skipped);

    // Second push from the same dir: fingerprint matches, so it's skipped.
    let pushed_again = flush(src.path(), &bucket).await.expect("push again");
    assert!(
        pushed_again.skipped,
        "second push should hit fingerprint fast-path"
    );

    // Pull into a *different* directory (distinct SyncKey, so the once-guard doesn't
    // short-circuit it).
    ensure_pulled(dst.path(), &bucket).await;

    let a = tokio::fs::read(dst.path().join("042/aaaaaaaa"))
        .await
        .expect("pulled file a");
    let b = tokio::fs::read(dst.path().join("999/bbbbbbbb"))
        .await
        .expect("pulled file b");
    assert_eq!(a, b"first response");
    assert_eq!(b, b"second response");

    // The pulled directory now matches the bucket: a flush from it is a no-op.
    let post_pull = flush(dst.path(), &bucket).await.expect("flush after pull");
    assert!(
        post_pull.skipped,
        "flush after a full pull should be skipped"
    );
}

/// Exercises the `json_merge` strategy: a mutable JSON-map file declared in
/// `.tysm-sync.json` is unioned (not clobbered) across machines, and the settings file
/// itself propagates through the bucket.
#[tokio::test]
#[ignore = "requires live R2 credentials + TYSM_TEST_BUCKET"]
async fn r2_json_merge_unions_entries() {
    let Some(bucket_name) = env("TYSM_TEST_BUCKET") else {
        eprintln!("skipping: TYSM_TEST_BUCKET not set");
        return;
    };
    if env("R2_ACCESS_KEY_ID")
        .or_else(|| env("AWS_ACCESS_KEY_ID"))
        .is_none()
    {
        eprintln!("skipping: R2/AWS access key not set");
        return;
    }

    let bucket = CacheBucket {
        bucket: bucket_name,
        prefix: unique_prefix(),
    };
    let settings = br#"{"files":[{"path":"shared/data.json","strategy":"json_merge"}]}"#;

    // Machine A: declares the merge rule and pushes its entries.
    let a = tempdir();
    write_file(a.path(), ".tysm-sync.json", settings).await;
    write_file(a.path(), "shared/data.json", br#"{"a":1,"common":"x"}"#).await;
    flush(a.path(), &bucket).await.expect("push A");

    // Machine B: has its own entries and NO settings file — it should inherit the rule
    // from the bucket, then *merge* rather than clobber on pull.
    let b = tempdir();
    write_file(b.path(), "shared/data.json", br#"{"b":2,"common":"y"}"#).await;
    ensure_pulled(b.path(), &bucket).await;

    assert!(
        b.path().join(".tysm-sync.json").exists(),
        "settings should be inherited from the bucket"
    );
    let merged = tokio::fs::read_to_string(b.path().join("shared/data.json"))
        .await
        .expect("merged data.json");
    assert!(
        merged.contains("\"a\":1"),
        "remote entry merged in: {merged}"
    );
    assert!(
        merged.contains("\"b\":2"),
        "local entry preserved: {merged}"
    );
    assert!(
        merged.contains("\"common\":\"y\""),
        "local value wins on collision: {merged}"
    );

    // Push B's union back, then a fresh machine C pulls the full union.
    flush(b.path(), &bucket).await.expect("push B");
    let c = tempdir();
    ensure_pulled(c.path(), &bucket).await;
    let c_data = tokio::fs::read_to_string(c.path().join("shared/data.json"))
        .await
        .expect("pulled union");
    assert!(
        c_data.contains("\"a\":1") && c_data.contains("\"b\":2"),
        "C should hold the union of both machines: {c_data}"
    );
}

/// Minimal unique temp dir without pulling in the `tempfile` crate.
fn tempdir() -> TempDir {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("tysm-cache-{}-{}", std::process::id(), nanos));
    std::fs::create_dir_all(&path).unwrap();
    TempDir { path }
}

struct TempDir {
    path: std::path::PathBuf,
}

impl TempDir {
    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}
