//! On-disk response cache, backed by an [`osmo`] store.
//!
//! Each entry is a `(cache_key, value)` pair where `value` is the zstd-compressed
//! response. Entries live in the osmo store rooted at the cache directory, under the
//! `tysm/` key namespace. osmo handles persistence (append-only segment files), sharing
//! across sibling clients pointed at the same directory, and — for callers that opt in —
//! syncing the directory to an S3-compatible bucket so caches can be shared across
//! machines (see `osmo::Store::pull`/`push`).

use std::collections::HashMap;
use std::path::Path;

/// The osmo key for a tysm cache entry.
fn store_key(cache_key: &str) -> String {
    format!("tysm/{cache_key}")
}

/// Read a cached value by key, or `None` if absent.
pub(crate) async fn read_from_cache_dir(dir: &Path, cache_key: &str) -> Option<Vec<u8>> {
    osmo::Store::open(dir).read(&store_key(cache_key)).await
}

/// Write a `(cache_key, data)` entry (no-op if already present identically).
pub(crate) async fn write_to_cache_dir(
    dir: &Path,
    cache_key: &str,
    data: &[u8],
) -> Result<(), std::io::Error> {
    osmo::Store::open(dir)
        .write(&store_key(cache_key), data)
        .await
        .map_err(std::io::Error::other)
}

/// Fold a pre-osmo tysm cache layout under `shard_dir` into `store`: `NNN.kv` record-log
/// shards, and (older still) `NNN/` directories with one file per entry. Entries land
/// under the `tysm/` namespace; the legacy files are deleted once imported. Returns how
/// many entries were imported. Cheap no-op when nothing legacy exists.
pub async fn migrate_legacy(
    shard_dir: &Path,
    store: &osmo::Store,
) -> Result<usize, std::io::Error> {
    const SHARDS: u16 = 1000;

    let mut kv_files = Vec::new();
    let mut legacy_dirs = Vec::new();
    for n in 0..SHARDS {
        let kv = shard_dir.join(format!("{n:03}.kv"));
        if kv.is_file() {
            kv_files.push(kv);
        }
        let dir = shard_dir.join(format!("{n:03}"));
        if dir.is_dir() {
            legacy_dirs.push(dir);
        }
    }
    if kv_files.is_empty() && legacy_dirs.is_empty() {
        return Ok(0);
    }

    // One shard at a time, deduped so the *last* record for a key wins (matching the old
    // replay semantics), lazily so peak memory is a single shard.
    let kv_entries = kv_files.iter().flat_map(|path| {
        let mut latest: HashMap<String, Vec<u8>> = HashMap::new();
        if let Ok(data) = std::fs::read(path) {
            for (key, value) in parse_v1_records(&data) {
                latest.insert(key, value);
            }
        }
        latest.into_iter()
    });
    let dir_entries = legacy_dirs.iter().flat_map(|dir| {
        let mut out = Vec::new();
        if let Ok(rd) = std::fs::read_dir(dir) {
            for entry in rd.flatten() {
                if let (Some(key), Ok(data)) = (
                    entry.file_name().to_str().map(String::from),
                    std::fs::read(entry.path()),
                ) {
                    out.push((key, data));
                }
            }
        }
        out
    });
    let imported = store
        .import(
            kv_entries
                .chain(dir_entries)
                .map(|(key, value)| (store_key(&key), value)),
        )
        .await
        .map_err(std::io::Error::other)?;

    for path in kv_files {
        std::fs::remove_file(&path)?;
    }
    for dir in legacy_dirs {
        std::fs::remove_dir_all(&dir)?;
    }
    Ok(imported)
}

/// Parse the v1 record-log format (`[u32-le key_len][key][u32-le val_len][val]`, no tag
/// byte), stopping at the first incomplete record. Records with non-UTF-8 keys are skipped.
fn parse_v1_records(data: &[u8]) -> Vec<(String, Vec<u8>)> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

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

        // Writing the same key+value again must not grow the store.
        let seg_bytes = || -> u64 {
            std::fs::read_dir(dir.join("segments"))
                .unwrap()
                .flatten()
                .filter(|e| e.file_name().to_string_lossy().ends_with(".seg"))
                .map(|e| e.metadata().unwrap().len())
                .sum()
        };
        let before = seg_bytes();
        write_to_cache_dir(&dir, "k1", b"v1").await.unwrap();
        assert_eq!(seg_bytes(), before, "identical re-write should be a no-op");

        tokio::fs::remove_dir_all(&dir).await.ok();
    }

    #[tokio::test]
    async fn migrates_v1_shards_and_legacy_directories() {
        let dir = unique_dir("migrate");
        tokio::fs::create_dir_all(&dir).await.unwrap();

        // v1 shard log with a superseded record: later record must win.
        let mut kv = Vec::new();
        for (k, v) in [("logkey", &b"old"[..]), ("logkey", b"new"), ("other", b"x")] {
            kv.extend_from_slice(&(k.len() as u32).to_le_bytes());
            kv.extend_from_slice(k.as_bytes());
            kv.extend_from_slice(&(v.len() as u32).to_le_bytes());
            kv.extend_from_slice(v);
        }
        std::fs::write(dir.join("042.kv"), kv).unwrap();

        // Even older layout: one file per entry.
        std::fs::create_dir_all(dir.join("117")).unwrap();
        std::fs::write(dir.join("117").join("dirkey"), b"dirvalue").unwrap();

        let store = osmo::Store::open_uncached(&dir);
        let n = migrate_legacy(&dir, &store).await.unwrap();
        assert_eq!(n, 3);

        assert_eq!(
            store.read("tysm/logkey").await.as_deref(),
            Some(&b"new"[..])
        );
        assert_eq!(store.read("tysm/other").await.as_deref(), Some(&b"x"[..]));
        assert_eq!(
            store.read("tysm/dirkey").await.as_deref(),
            Some(&b"dirvalue"[..])
        );
        assert!(!dir.join("042.kv").exists(), "shard log deleted");
        assert!(!dir.join("117").exists(), "legacy dir deleted");

        // And the public read path sees the migrated entries... but through a distinct
        // handle in real use; here the registry store would differ from open_uncached, so
        // read through the same store to keep the test hermetic.
        let again = migrate_legacy(&dir, &store).await.unwrap();
        assert_eq!(again, 0, "second migration is a no-op");

        tokio::fs::remove_dir_all(&dir).await.ok();
    }
}
