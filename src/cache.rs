//! On-disk response cache backed by a few big append-only **record-log** shard files.
//!
//! Each cache entry is a `(cache_key, value)` pair where `value` is the zstd-compressed
//! response. Instead of one tiny file per entry (which produced millions of inodes and was
//! painful to mirror to object storage), entries are sharded into `NNN.kv` files (`000.kv`
//! … `999.kv`) — one append-only log per shard. Each record is
//! `[u32-le key_len][key][u32-le val_len][val]`, the same binary format `osmo`'s `records`
//! sync strategy understands, so a directory of these files mirrors to a bucket as ~1000
//! objects that merge losslessly across machines.
//!
//! A shard is loaded into an in-memory map on first access (lazily); writes insert into
//! the map and append a record to the file. The legacy layout (a `dir/NNN/` directory with
//! one file per entry) is migrated into the log the first time its shard is touched.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock};

use dashmap::DashMap;
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;
use xxhash_rust::const_xxh3::xxh3_64 as const_xxh3;

const SHARDS: u64 = 1000;

/// One [`ShardStore`] per canonicalized cache directory, shared process-wide so sibling
/// clients pointed at the same directory share the in-memory shard maps.
static STORES: LazyLock<DashMap<PathBuf, Arc<ShardStore>>> = LazyLock::new(DashMap::new);

fn store_for(dir: &Path) -> Arc<ShardStore> {
    let canon = std::fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf());
    STORES
        .entry(canon.clone())
        .or_insert_with(|| Arc::new(ShardStore::new(canon)))
        .clone()
}

/// Read a cached value by key, or `None` if absent.
pub(crate) async fn read_from_cache_dir(dir: &Path, cache_key: &str) -> Option<Vec<u8>> {
    store_for(dir).get(cache_key).await
}

/// Append a `(cache_key, data)` entry to its shard log (no-op if already present identically).
pub(crate) async fn write_to_cache_dir(
    dir: &Path,
    cache_key: &str,
    data: &[u8],
) -> Result<(), std::io::Error> {
    store_for(dir).put(cache_key, data).await
}

/// Force every legacy shard directory in `dir` to be folded into its `NNN.kv` log and
/// removed. Useful before mirroring the directory to object storage, so no leftover
/// per-entry files remain. Safe to call repeatedly.
pub async fn migrate(dir: &Path) -> Result<(), std::io::Error> {
    let store = store_for(dir);
    for n in 0..SHARDS as u16 {
        // Only bother if a legacy directory actually exists for this shard. Fold it to
        // disk under the shard lock without populating the in-memory map, so migrating a
        // huge cache up front doesn't pull all of it into RAM.
        if tokio::fs::metadata(store.legacy_dir(n))
            .await
            .map(|m| m.is_dir())
            .unwrap_or(false)
        {
            let lock = store.shard_lock(n);
            let _guard = lock.lock().await;
            store.fold_legacy(n).await;
        }
    }
    Ok(())
}

struct ShardStore {
    dir: PathBuf,
    shards: DashMap<u16, Arc<Mutex<Shard>>>,
}

#[derive(Default)]
struct Shard {
    loaded: bool,
    entries: HashMap<String, Vec<u8>>,
}

impl ShardStore {
    fn new(dir: PathBuf) -> Self {
        Self {
            dir,
            shards: DashMap::new(),
        }
    }

    fn shard_lock(&self, n: u16) -> Arc<Mutex<Shard>> {
        self.shards
            .entry(n)
            .or_insert_with(|| Arc::new(Mutex::new(Shard::default())))
            .clone()
    }

    fn log_path(&self, n: u16) -> PathBuf {
        self.dir.join(format!("{n:03}.kv"))
    }

    fn legacy_dir(&self, n: u16) -> PathBuf {
        self.dir.join(format!("{n:03}"))
    }

    /// Fold the legacy `dir/NNN/<key>` directory (one file per entry) into the shard log,
    /// then remove it. No-op if there's no such directory. Does not touch the in-memory map.
    /// Caller must hold the shard lock.
    async fn fold_legacy(&self, n: u16) {
        let legacy = self.legacy_dir(n);
        let Ok(mut rd) = tokio::fs::read_dir(&legacy).await else {
            return;
        };
        let mut batch = Vec::new();
        while let Ok(Some(entry)) = rd.next_entry().await {
            if entry
                .file_type()
                .await
                .map(|t| t.is_file())
                .unwrap_or(false)
            {
                if let (Some(key), Ok(data)) = (
                    entry.file_name().to_str(),
                    tokio::fs::read(entry.path()).await,
                ) {
                    encode_record(&mut batch, key, &data);
                }
            }
        }
        // Only drop the legacy directory once its contents are safely in the log — a
        // failed append must never lose cached data.
        if !batch.is_empty() {
            if let Err(e) = append(&self.log_path(n), &batch).await {
                log::warn!("tysm cache: failed to fold legacy shard {n:03}; keeping it: {e}");
                return;
            }
        }
        let _ = tokio::fs::remove_dir_all(&legacy).await;
    }

    /// Load a shard's entries into memory (migrating the legacy directory first, if any).
    async fn ensure_loaded(&self, n: u16, shard: &mut Shard) {
        if shard.loaded {
            return;
        }
        self.fold_legacy(n).await;
        if let Ok(data) = tokio::fs::read(&self.log_path(n)).await {
            for (key, val) in parse_records(&data) {
                shard.entries.insert(key, val); // later records win
            }
        }
        shard.loaded = true;
    }

    async fn get(&self, key: &str) -> Option<Vec<u8>> {
        let n = shard_of(key);
        let lock = self.shard_lock(n);
        let mut shard = lock.lock().await;
        self.ensure_loaded(n, &mut shard).await;
        shard.entries.get(key).cloned()
    }

    async fn put(&self, key: &str, data: &[u8]) -> Result<(), std::io::Error> {
        let n = shard_of(key);
        let lock = self.shard_lock(n);
        let mut shard = lock.lock().await;
        self.ensure_loaded(n, &mut shard).await;
        // Already cached identically — nothing to append.
        if shard.entries.get(key).map(Vec::as_slice) == Some(data) {
            return Ok(());
        }
        let mut buf = Vec::with_capacity(8 + key.len() + data.len());
        encode_record(&mut buf, key, data);
        append(&self.log_path(n), &buf).await?;
        shard.entries.insert(key.to_string(), data.to_vec());
        Ok(())
    }
}

/// Which shard a key belongs to (matches the legacy `cache_shard` distribution).
fn shard_of(key: &str) -> u16 {
    (const_xxh3(key.as_bytes()) % SHARDS) as u16
}

fn encode_record(buf: &mut Vec<u8>, key: &str, val: &[u8]) {
    buf.extend_from_slice(&(key.len() as u32).to_le_bytes());
    buf.extend_from_slice(key.as_bytes());
    buf.extend_from_slice(&(val.len() as u32).to_le_bytes());
    buf.extend_from_slice(val);
}

/// Parse a record log into `(key, value)` pairs, stopping at the first incomplete record
/// (tolerant of a truncated trailing append). Records with non-UTF-8 keys are skipped.
fn parse_records(data: &[u8]) -> Vec<(String, Vec<u8>)> {
    let mut out = Vec::new();
    let mut i = 0usize;
    let read_len = |at: usize| -> Option<usize> {
        data.get(at..at + 4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize)
    };
    while let Some(key_len) = read_len(i) {
        i += 4;
        let Some(key) = data.get(i..i + key_len) else {
            break;
        };
        i += key_len;
        let Some(val_len) = read_len(i) else { break };
        i += 4;
        let Some(val) = data.get(i..i + val_len) else {
            break;
        };
        i += val_len;
        if let Ok(key) = std::str::from_utf8(key) {
            out.push((key.to_string(), val.to_vec()));
        }
    }
    out
}

async fn append(path: &Path, bytes: &[u8]) -> Result<(), std::io::Error> {
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    let mut f = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .await?;
    f.write_all(bytes).await?;
    f.flush().await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unique_dir(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "tysm-cache-test-{tag}-{}-{nanos}",
            std::process::id()
        ))
    }

    #[tokio::test]
    async fn roundtrip_and_dedup() {
        let dir = unique_dir("rt");
        tokio::fs::create_dir_all(&dir).await.unwrap();

        assert_eq!(read_from_cache_dir(&dir, "missing").await, None);
        write_to_cache_dir(&dir, "k1", b"v1").await.unwrap();
        write_to_cache_dir(&dir, "k2", b"\x00\x01binary")
            .await
            .unwrap();
        assert_eq!(
            read_from_cache_dir(&dir, "k1").await.as_deref(),
            Some(&b"v1"[..])
        );
        assert_eq!(
            read_from_cache_dir(&dir, "k2").await.as_deref(),
            Some(&b"\x00\x01binary"[..])
        );

        // Writing the same key+value again must not grow the log file.
        let shard = store_for(&dir).log_path(shard_of("k1"));
        let before = tokio::fs::metadata(&shard).await.unwrap().len();
        write_to_cache_dir(&dir, "k1", b"v1").await.unwrap();
        let after = tokio::fs::metadata(&shard).await.unwrap().len();
        assert_eq!(before, after, "identical re-write should be a no-op");

        tokio::fs::remove_dir_all(&dir).await.ok();
    }

    #[tokio::test]
    async fn persists_and_reloads_from_disk() {
        let dir = unique_dir("persist");
        tokio::fs::create_dir_all(&dir).await.unwrap();
        write_to_cache_dir(&dir, "alpha", b"A").await.unwrap();
        write_to_cache_dir(&dir, "beta", b"B").await.unwrap();

        // Drop the in-memory store and read fresh from the log files.
        STORES.clear();
        assert_eq!(
            read_from_cache_dir(&dir, "alpha").await.as_deref(),
            Some(&b"A"[..])
        );
        assert_eq!(
            read_from_cache_dir(&dir, "beta").await.as_deref(),
            Some(&b"B"[..])
        );

        tokio::fs::remove_dir_all(&dir).await.ok();
    }

    #[tokio::test]
    async fn migrates_legacy_shard_directories() {
        let dir = unique_dir("migrate");
        // Lay down the old layout: dir/<shard>/<key> = value.
        let key = "legacykey";
        let shard = format!("{:03}", const_xxh3(key.as_bytes()) % SHARDS);
        let legacy = dir.join(&shard);
        tokio::fs::create_dir_all(&legacy).await.unwrap();
        tokio::fs::write(legacy.join(key), b"oldvalue")
            .await
            .unwrap();

        STORES.clear();
        // Reading the key migrates the directory into the log and returns the value.
        assert_eq!(
            read_from_cache_dir(&dir, key).await.as_deref(),
            Some(&b"oldvalue"[..])
        );
        // The legacy directory is gone; the log file exists.
        assert!(!dir.join(&shard).exists(), "legacy dir should be removed");
        assert!(
            dir.join(format!("{shard}.kv")).exists(),
            "log file should exist"
        );

        // And it survives a fresh load.
        STORES.clear();
        assert_eq!(
            read_from_cache_dir(&dir, key).await.as_deref(),
            Some(&b"oldvalue"[..])
        );

        tokio::fs::remove_dir_all(&dir).await.ok();
    }
}
