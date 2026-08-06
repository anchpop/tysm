//! The pricing of different models.
//!
//! The table itself lives in the dependency-free [`model_prices`] crate, so
//! that builds which only need to price tokens — Android/Soong among them —
//! can vendor it without dragging this client's dependency tree along. This
//! module is the thin layer that prices a [`ChatUsage`] with it.

pub use model_prices::{
    price_of, CallTokens, LongContext, ModelCost, TierCost, MODELS, RATE_CARD_VERSION,
};

use crate::chat_completions::ChatUsage;

/// The billable token counts of one request.
///
/// `completion_tokens` already includes any reasoning tokens, so
/// `completion_token_details` is deliberately not added on top of it — that
/// would charge reasoning twice.
fn tokens_of(usage: ChatUsage) -> CallTokens {
    CallTokens {
        prompt: usage.prompt_tokens,
        cached_prompt: usage
            .prompt_token_details
            .map_or(0, |details| details.cached_tokens),
        output: usage.completion_tokens,
    }
}

/// The cost in dollars of a **single** request.
///
/// Must be called once per request and the results summed. Summing the token
/// counts of several requests and pricing the total gives a different — and
/// wrong — answer for any model with a [`LongContext`] card, because the
/// premium keys off one request's prompt size.
///
/// Returns `None` for a model we have no price for.
pub fn cost_of_call(model: &str, service_tier: Option<&str>, usage: ChatUsage) -> Option<f64> {
    model_prices::cost_of_call(model, service_tier, tokens_of(usage))
}

/// As [`cost_of_call`], but keeping the input and output dollars apart — for
/// callers that report the two separately rather than only the total.
pub fn cost_of_call_split(
    model: &str,
    service_tier: Option<&str>,
    usage: ChatUsage,
) -> Option<(f64, f64)> {
    model_prices::cost_of_call_split(model, service_tier, tokens_of(usage))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat_completions::PromptTokenDetails;

    /// The table is tested in the `model-prices` crate; what needs covering
    /// here is the translation, and above all that cached tokens reach it —
    /// dropping them would quietly bill every cache hit at full price.
    #[test]
    fn chat_usage_translates_to_billable_tokens() {
        let usage = ChatUsage {
            prompt_tokens: 2_000_000,
            completion_tokens: 1_000_000,
            total_tokens: 3_000_000,
            prompt_token_details: Some(PromptTokenDetails {
                cached_tokens: 1_000_000,
            }),
            completion_token_details: None,
        };
        assert_eq!(
            tokens_of(usage),
            CallTokens {
                prompt: 2_000_000,
                cached_prompt: 1_000_000,
                output: 1_000_000,
            }
        );

        // gpt-4o: 1M uncached @2.50 + 1M cached @1.25 + 1M output @10.00
        assert_eq!(
            cost_of_call("gpt-4o", None, usage),
            Some(2.50 + 1.25 + 10.00)
        );
    }

    #[test]
    fn a_request_with_no_cache_details_bills_every_prompt_token_uncached() {
        let usage = ChatUsage {
            prompt_tokens: 1_000_000,
            completion_tokens: 0,
            total_tokens: 1_000_000,
            prompt_token_details: None,
            completion_token_details: None,
        };
        assert_eq!(tokens_of(usage).cached_prompt, 0);
        assert_eq!(cost_of_call("gpt-4o", None, usage), Some(2.50));
    }
}
