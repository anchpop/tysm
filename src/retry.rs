//! Shared live-request retry policy.

use std::time::Duration;

use crate::OpenAiError;

/// How many times a live request is attempted before its error is returned.
pub(crate) const LIVE_REQUEST_ATTEMPTS: u32 = 4;

/// Whether an API error is one a later, identical request could succeed at.
///
/// The flex service tier answers `flex_unavailable` whenever OpenAI has no
/// spare capacity; rate limits and server faults are the same shape of
/// problem. In each case the request is fine and only the moment is wrong.
pub(crate) fn is_transient_api_error(error: &OpenAiError) -> bool {
    let code = error.code.as_deref().unwrap_or_default();
    matches!(
        code,
        "flex_unavailable" | "resource_unavailable" | "rate_limit_exceeded" | "server_error"
    ) || matches!(
        error.r#type.as_str(),
        "resource_unavailable" | "rate_limit_error" | "server_error"
    )
}

/// A transport error is usually a pooled keep-alive connection the server has
/// already closed, surfacing on its next use; reconnecting fixes it at once.
/// Capacity frees up on a scale of minutes rather than milliseconds, so those
/// errors wait far longer. Anything else (a malformed request, a schema the
/// model cannot satisfy) is not retried at all.
pub(crate) enum RetryKind {
    Transport,
    Capacity,
}

impl RetryKind {
    /// Attempts start at one. Reconnect quickly; give capacity time to recover.
    /// A provider's explicit delay takes precedence over the default backoff.
    pub(crate) fn delay(self, attempt: u32, retry_after: Option<Duration>) -> Duration {
        retry_after.unwrap_or_else(|| match self {
            Self::Transport => Duration::from_secs(1 << attempt),
            Self::Capacity => Duration::from_secs(15 << (attempt - 1)),
        })
    }
}
