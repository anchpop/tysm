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
