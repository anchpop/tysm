//! One process-wide, typed accounting state, optionally persisted as JSONL.
//!
//! Response rows are durable immediately. Full cumulative snapshots append at
//! laps, every sixty seconds, and shutdown. Never sum snapshots: use the last
//! snapshot per process, or response rows, and exclude historical batch harvests
//! when combining this ledger with the provider's batch API.
use crate::chat_completions::ChatUsage;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::{LazyLock, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

/// Newly incurred live usage versus historical batch observations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpendKind {
    /// A response to a live request made by this process.
    Live,
    /// Historical usage harvested from a batch, not necessarily billed now.
    BatchHarvest,
}
/// One billed response; unknown pricing is null, never silently zero.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpendRecord {
    /// Observation time, not necessarily billing time.
    pub observed_at_ms: u64,
    /// Process ID, disambiguated by the ledger filename's start time.
    pub pid: u32,
    /// API provider.
    pub provider: String,
    /// Requested model.
    pub model: String,
    /// Actual tier when present, otherwise requested tier.
    pub service_tier: Option<String>,
    /// Billing origin.
    pub kind: SpendKind,
    /// Actual usage, including provider cache reads/writes.
    pub usage: ChatUsage,
    /// Recorded USD at this row's rate card, or unknown.
    pub dollars: Option<f64>,
    /// Rate card used; audits must not silently reprice history.
    pub rate_card_version: String,
}
/// Dimensions of a process accounting bucket.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct BucketKey {
    /// API provider.
    pub provider: String,
    /// Requested model.
    pub model: String,
    /// Actual tier when present, otherwise requested tier.
    pub service_tier: Option<String>,
    /// Live or historical batch usage.
    pub kind: SpendKind,
}
impl BucketKey {
    fn new(provider: &str, model: &str, tier: Option<&str>, kind: SpendKind) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
            service_tier: tier.map(str::to_owned),
            kind,
        }
    }
}
/// Cumulative token counters and known subtotal. Unknowns remain explicit.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Bucket {
    /// Observed responses, including unknown-price/unknown-usage responses.
    pub requests: u64,
    /// Requests with unknown prices.
    pub unpriced_requests: u64,
    /// Responses with present but undecodable usage.
    pub unparsed_usage: u64,
    /// Sum only of known prices (not a complete total if unknowns exist).
    pub known_dollars: f64,
    /// Prompt tokens, cache reads and writes included.
    pub prompt_tokens: u64,
    /// Output tokens, reasoning included.
    pub completion_tokens: u64,
    /// Provider cache-read tokens.
    pub cached_tokens: u64,
    /// Five-minute cache-write tokens.
    pub cache_creation_5m_tokens: u64,
    /// One-hour cache-write tokens.
    pub cache_creation_1h_tokens: u64,
}
impl Bucket {
    fn add(&mut self, usage: ChatUsage, dollars: Option<f64>) {
        self.requests += 1;
        match dollars {
            Some(d) => self.known_dollars += d,
            None => self.unpriced_requests += 1,
        }
        self.prompt_tokens += u64::from(usage.prompt_tokens);
        self.completion_tokens += u64::from(usage.completion_tokens);
        if let Some(d) = usage.prompt_tokens_details {
            self.cached_tokens += u64::from(d.cached_tokens);
            self.cache_creation_5m_tokens += u64::from(d.cache_creation_5m_tokens);
            self.cache_creation_1h_tokens += u64::from(d.cache_creation_1h_tokens);
        }
    }
}
/// A ledger row. Snapshot counters are cumulative, not additive.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "record_type", rename_all = "snake_case")]
pub enum LedgerRow {
    /// A single response's recorded price and usage.
    Response {
        /// Billing observation.
        record: SpendRecord,
    },
    /// A full process snapshot, including unknown counters.
    Snapshot {
        /// Lap, periodic, exit, or signal label.
        label: String,
        /// Observation timestamp.
        observed_at_ms: u64,
        /// Rate card of the current process.
        rate_card_version: String,
        /// Full cumulative buckets.
        buckets: Vec<(BucketKey, Bucket)>,
    },
    /// A provider returned usage which this client could not decode.
    UnparsedUsage {
        /// The affected bucket.
        key: BucketKey,
        /// Raw usage evidence; never includes credentials or request content.
        usage: serde_json::Value,
        /// Observation timestamp.
        observed_at_ms: u64,
    },
}
#[derive(Default)]
struct State {
    buckets: BTreeMap<BucketKey, Bucket>,
    warned: HashSet<(String, String)>,
    file: Option<File>,
}
impl State {
    fn append(&mut self, row: &LedgerRow) -> std::io::Result<()> {
        if let Some(file) = &mut self.file {
            serde_json::to_writer(&mut *file, row)?;
            file.write_all(b"\n")?;
            file.sync_data()?;
        }
        Ok(())
    }
}
static STATE: LazyLock<Mutex<State>> = LazyLock::new(|| Mutex::new(State::default()));

/// Read typed bucket counters even when no on-disk ledger was enabled.
pub fn buckets() -> Vec<(BucketKey, Bucket)> {
    STATE
        .lock()
        .unwrap()
        .buckets
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect()
}
/// Derive the process total from the same buckets used by the ledger.
pub fn totals() -> crate::chat_completions::ProcessSpend {
    let state = STATE.lock().unwrap();
    state.buckets.values().fold(
        crate::chat_completions::ProcessSpend {
            dollars: 0.0,
            unpriced_requests: 0,
        },
        |mut total, bucket| {
            total.dollars += bucket.known_dollars;
            total.unpriced_requests += bucket.unpriced_requests;
            total
        },
    )
}
/// Append a periodic cumulative snapshot.
pub fn snapshot() -> std::io::Result<()> {
    snapshot_label("periodic")
}
/// Append a named cumulative snapshot, preserving all previous laps.
pub fn snapshot_label(label: &str) -> std::io::Result<()> {
    let mut state = STATE.lock().unwrap();
    let row = LedgerRow::Snapshot {
        label: label.into(),
        observed_at_ms: now_ms(),
        rate_card_version: crate::model_prices::RATE_CARD_VERSION.into(),
        buckets: state
            .buckets
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect(),
    };
    state.append(&row)
}
/// Joins the periodic writer and appends an exit snapshot on normal shutdown.
pub struct LedgerGuard {
    stop: std::sync::mpsc::Sender<()>,
    thread: Option<std::thread::JoinHandle<()>>,
}
impl Drop for LedgerGuard {
    fn drop(&mut self) {
        let _ = self.stop.send(());
        self.thread
            .take()
            .unwrap()
            .join()
            .expect("join spend writer");
        snapshot_label("exit").expect("persist exit spend snapshot");
    }
}
/// Start `start-pid-language.jsonl`, with no clobbering of existing evidence.
pub fn start_ledger(directory: &Path, language: &str) -> std::io::Result<LedgerGuard> {
    if !language
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-')
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "invalid ledger language label",
        ));
    }
    std::fs::create_dir_all(directory)?;
    let mut state = STATE.lock().unwrap();
    if state.file.is_some() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "ledger already started",
        ));
    }
    let path = directory.join(format!(
        "{}-{}-{language}.jsonl",
        now_ms(),
        std::process::id()
    ));
    state.file = Some(
        OpenOptions::new()
            .create_new(true)
            .append(true)
            .open(path)?,
    );
    File::open(directory)?.sync_all()?;
    drop(state);
    snapshot_label("start")?;
    let (stop, receiver) = std::sync::mpsc::channel();
    let thread = std::thread::spawn(move || {
        while receiver.recv_timeout(std::time::Duration::from_secs(60))
            == Err(std::sync::mpsc::RecvTimeoutError::Timeout)
        {
            snapshot().expect("persist periodic spend snapshot");
        }
    });
    Ok(LedgerGuard {
        stop,
        thread: Some(thread),
    })
}
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis()
        .try_into()
        .unwrap()
}

/// Price one response, never aggregated tokens. Unknown rates stay unknown.
pub fn price(model: &str, tier: Option<&str>, kind: SpendKind, usage: ChatUsage) -> Option<f64> {
    crate::model_prices::cost_of_call(
        model,
        if kind == SpendKind::BatchHarvest {
            Some("batch")
        } else {
            tier
        },
        usage,
    )
}
pub(crate) fn record(
    provider: &str,
    model: &str,
    tier: Option<&str>,
    kind: SpendKind,
    usage: ChatUsage,
) -> Option<f64> {
    let dollars = price(model, tier, kind, usage);
    let mut state = STATE.lock().unwrap();
    if dollars.is_none() && state.warned.insert((provider.into(), model.into())) {
        log::error!("UNPRICED BILLED RESPONSE: provider={provider} model={model} tier={tier:?} kind={kind:?} usage={usage:?}; subsequent unknowns counted in ledger");
    }
    state
        .buckets
        .entry(BucketKey::new(provider, model, tier, kind))
        .or_default()
        .add(usage, dollars);
    state
        .append(&LedgerRow::Response {
            record: SpendRecord {
                observed_at_ms: now_ms(),
                pid: std::process::id(),
                provider: provider.into(),
                model: model.into(),
                service_tier: tier.map(str::to_owned),
                kind,
                usage,
                dollars,
                rate_card_version: crate::model_prices::RATE_CARD_VERSION.into(),
            },
        })
        .expect("persist spend response");
    dollars
}
pub(crate) fn unparsed(
    provider: &str,
    model: &str,
    tier: Option<&str>,
    kind: SpendKind,
    usage: serde_json::Value,
) {
    let mut state = STATE.lock().unwrap();
    let key = BucketKey::new(provider, model, tier, kind);
    let bucket = state.buckets.entry(key.clone()).or_default();
    bucket.requests += 1;
    bucket.unpriced_requests += 1;
    bucket.unparsed_usage += 1;
    log::error!("UNPARSED BILLED USAGE: {key:?}: {usage}");
    state
        .append(&LedgerRow::UnparsedUsage {
            key,
            usage,
            observed_at_ms: now_ms(),
        })
        .expect("persist unparsed spend");
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unknown_is_not_zero_and_batch_is_discounted_once() {
        let usage = ChatUsage {
            prompt_tokens: 1000,
            ..Default::default()
        };
        assert_eq!(price("unknown-model", None, SpendKind::Live, usage), None);
        let live = price("gpt-4o", None, SpendKind::Live, usage).unwrap();
        assert_eq!(
            price("gpt-4o", Some("flex"), SpendKind::BatchHarvest, usage),
            Some(live / 2.0)
        );
    }
    #[test]
    fn unknown_and_historical_origin_survive_roundtrip() {
        let row = LedgerRow::Response {
            record: SpendRecord {
                observed_at_ms: 1,
                pid: 2,
                provider: "openai".into(),
                model: "unknown".into(),
                service_tier: None,
                kind: SpendKind::BatchHarvest,
                usage: ChatUsage::default(),
                dollars: None,
                rate_card_version: "test".into(),
            },
        };
        let LedgerRow::Response { record } =
            serde_json::from_str(&serde_json::to_string(&row).unwrap()).unwrap()
        else {
            panic!()
        };
        assert_eq!(record.kind, SpendKind::BatchHarvest);
        assert_eq!(record.dollars, None);
    }
}
