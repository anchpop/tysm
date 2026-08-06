//! The published per-token prices of hosted LLMs, as plain data.
//!
//! Rates are USD per million tokens, copied from each vendor's published
//! pricing page. The comment above a block records which page, and when.
//!
//! Three things make a model's price more than a single pair of numbers:
//!
//! * **Cached input.** Most vendors discount prompt tokens they served from
//!   their own cache.
//! * **Service tier.** OpenAI sells the same model at `flex` (cheaper, slower)
//!   and `priority` (dearer, faster) beside the standard rate.
//! * **Prompt size.** Some models re-price the *whole request* once its prompt
//!   crosses a threshold — see [`LongContext`].
//!
//! This crate deliberately has no dependencies — not even on a client library's
//! usage type — so that anything needing to price tokens can vendor it alone.

/// One request's billable token counts.
///
/// A named struct rather than three positional integers because the three are
/// the same type, and swapping cached for output silently mis-bills instead of
/// failing to compile.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CallTokens {
    /// Prompt tokens, cached ones included.
    pub prompt: u32,
    /// Prompt tokens the vendor served from cache — a subset of `prompt`,
    /// billed at the cheaper cached rate.
    pub cached_prompt: u32,
    /// Generated tokens, reasoning included. Vendors that report reasoning
    /// separately from completion must add the two before setting this, since
    /// both are billed at the output rate.
    pub output: u32,
}

/// USD per million tokens at one service tier.
#[derive(Debug, Clone, Copy)]
pub struct TierCost {
    /// The cost of uncached input tokens.
    pub input: f64,
    /// The cost of input tokens the vendor served from cache. `None` when the
    /// vendor publishes no cache discount, in which case cached tokens are
    /// billed at `input`.
    pub cached_input: Option<f64>,
    /// The cost of output tokens.
    pub output: f64,
}

impl TierCost {
    /// Bill one request's tokens at these rates, keeping the input and output
    /// halves apart.
    fn charge_split(&self, uncached_input: u32, cached_input: u32, output: u32) -> (f64, f64) {
        let cached_rate = self.cached_input.unwrap_or(self.input);
        let input =
            (self.input * uncached_input as f64 + cached_rate * cached_input as f64) / 1_000_000.0;
        (input, self.output * output as f64 / 1_000_000.0)
    }
}

/// The rates that replace a model's whole card once a single request's prompt
/// exceeds `above_prompt_tokens`.
///
/// The premium applies to the *entire* request, not merely the tokens past the
/// threshold. That is why cost must be computed per call and the dollars
/// summed — summing tokens across calls and pricing the total would push an
/// innocent series of short requests over the threshold and bill every one of
/// them at the premium.
#[derive(Debug, Clone, Copy)]
pub struct LongContext {
    /// The prompt size above which this card replaces the standard one. A
    /// prompt exactly this large is still billed at the cheaper card.
    pub above_prompt_tokens: u32,
    /// The standard-tier rates above the threshold.
    pub standard: TierCost,
    /// As [`ModelCost::flex`], above the threshold.
    pub flex: Option<TierCost>,
    /// As [`ModelCost::priority`], above the threshold.
    pub priority: Option<TierCost>,
}

/// Everything we know about one model's price.
#[derive(Debug, Clone, Copy)]
pub struct ModelCost {
    /// Matched as a prefix of the requested model, longest match winning, so
    /// that a dated id like `gpt-5.6-terra-2026-06-01` resolves to the entry
    /// for `gpt-5.6-terra`.
    pub name: &'static str,
    /// The rates everyone pays unless they ask for another tier.
    pub standard: TierCost,
    /// `None` means the vendor does not sell this model at that tier, and a
    /// request asking for it is billed at `standard`.
    pub flex: Option<TierCost>,
    /// As `flex`.
    pub priority: Option<TierCost>,
    /// `None` means this model costs the same whatever the prompt size.
    pub long_context: Option<LongContext>,
}

impl ModelCost {
    /// The rates for one request: the long-context card if the prompt crosses
    /// this model's threshold, and within that the requested tier if the
    /// vendor sells one.
    fn rates(&self, service_tier: Option<&str>, prompt_tokens: u32) -> &TierCost {
        let (standard, flex, priority) = match &self.long_context {
            Some(long) if prompt_tokens > long.above_prompt_tokens => {
                (&long.standard, &long.flex, &long.priority)
            }
            _ => (&self.standard, &self.flex, &self.priority),
        };
        match service_tier {
            Some("flex") => flex.as_ref().unwrap_or(standard),
            Some("priority") => priority.as_ref().unwrap_or(standard),
            _ => standard,
        }
    }

    const fn flex(mut self, input: f64, cached_input: Option<f64>, output: f64) -> Self {
        self.flex = Some(TierCost {
            input,
            cached_input,
            output,
        });
        self
    }

    const fn priority(mut self, input: f64, cached_input: Option<f64>, output: f64) -> Self {
        self.priority = Some(TierCost {
            input,
            cached_input,
            output,
        });
        self
    }

    const fn long_context(mut self, long: LongContext) -> Self {
        self.long_context = Some(long);
        self
    }
}

impl LongContext {
    const fn flex(mut self, input: f64, cached_input: Option<f64>, output: f64) -> Self {
        self.flex = Some(TierCost {
            input,
            cached_input,
            output,
        });
        self
    }

    const fn priority(mut self, input: f64, cached_input: Option<f64>, output: f64) -> Self {
        self.priority = Some(TierCost {
            input,
            cached_input,
            output,
        });
        self
    }
}

/// One model at its standard rates. Chain `.flex()`, `.priority()`, and
/// `.long_context()` onto it for the models that need them.
const fn model(
    name: &'static str,
    input: f64,
    cached_input: Option<f64>,
    output: f64,
) -> ModelCost {
    ModelCost {
        name,
        standard: TierCost {
            input,
            cached_input,
            output,
        },
        flex: None,
        priority: None,
        long_context: None,
    }
}

/// The standard-tier rates above `above_prompt_tokens`. Chain `.flex()` and
/// `.priority()` for the tiers the vendor also re-prices.
const fn long(
    above_prompt_tokens: u32,
    input: f64,
    cached_input: Option<f64>,
    output: f64,
) -> LongContext {
    LongContext {
        above_prompt_tokens,
        standard: TierCost {
            input,
            cached_input,
            output,
        },
        flex: None,
        priority: None,
    }
}

/// The prompt size above which the gpt-5.x family charges its long-context
/// premium. OpenAI publishes the boundary only for gpt-5.5 ("<272K context
/// length"); the rest of the family is assumed to share it.
const GPT_5_LONG_CONTEXT: u32 = 272_000;

/// Every model we know a price for.
///
/// One entry per model, carrying all of its tiers — so changing a price is a
/// single local edit, and a tier the vendor sells but we have not filled in is
/// visible as a missing `.flex()` rather than hidden in a second table.
pub const MODELS: &[ModelCost] = &[
    // Anthropic
    //
    // Both spellings of a point release are listed. The API's own ids use
    // dashes (`claude-opus-4-5-20251101`), and with only the dotted spelling
    // present the longest-prefix match falls back to the major version —
    // billing Opus 4.5 at Opus 4's $15/$75 instead of $5/$25.
    model("claude-opus-4.5", 5.0, None, 25.0),
    model("claude-opus-4-5", 5.0, None, 25.0),
    model("claude-opus-4.1", 15.0, None, 75.0),
    model("claude-opus-4-1", 15.0, None, 75.0),
    model("claude-opus-4", 15.0, None, 75.0),
    model("claude-sonnet-4.5", 3.0, None, 15.0),
    model("claude-sonnet-4-5", 3.0, None, 15.0),
    model("claude-sonnet-4", 3.0, None, 15.0),
    model("claude-haiku-4", 0.80, None, 4.0),
    model("claude-3-opus", 15.0, None, 75.0),
    model("claude-3-7-sonnet", 3.0, None, 15.0),
    model("claude-3-5-haiku", 0.80, None, 4.0),
    // Anthropic (new models, copied from https://platform.claude.com/docs/en/about-claude/pricing on 2026-07-06)
    model("claude-fable-5", 10.0, None, 50.0),
    model("claude-mythos-5", 10.0, None, 50.0),
    model("claude-opus-5", 5.0, None, 25.0),
    model("claude-opus-4-8", 5.0, None, 25.0),
    model("claude-opus-4-7", 5.0, None, 25.0),
    model("claude-opus-4-6", 5.0, None, 25.0),
    model("claude-sonnet-4-6", 3.0, None, 15.0),
    // Introductory pricing through 2026-08-31; becomes 3.0/15.0 on 2026-09-01.
    // https://platform.claude.com/docs/en/about-claude/pricing
    model("claude-sonnet-5", 2.0, None, 10.0),
    model("claude-haiku-4-5", 1.0, None, 5.0),
    // OpenAI
    // Copied from https://platform.openai.com/docs/pricing on 2026-03-17
    model("gpt-4.1", 2.00, Some(0.50), 8.00),
    model("gpt-4.1-mini", 0.40, Some(0.10), 1.60),
    model("gpt-4.1-nano", 0.10, Some(0.025), 0.40),
    model("gpt-4.5-preview", 75.00, Some(37.50), 150.00),
    model("gpt-4o", 2.50, Some(1.25), 10.00),
    model("gpt-4o-audio-preview", 2.50, None, 10.00),
    model("gpt-4o-realtime-preview", 5.00, Some(2.50), 20.00),
    model("gpt-4o-mini", 0.15, Some(0.075), 0.60),
    model("gpt-4o-mini-audio-preview", 0.15, None, 0.60),
    model("gpt-4o-mini-realtime-preview", 0.60, Some(0.30), 2.40),
    model("o1", 15.00, Some(7.50), 60.00),
    model("o1-pro", 150.00, None, 600.00),
    model("o3", 2.00, Some(0.50), 8.00).flex(1.00, Some(0.25), 4.00),
    model("o4-mini", 1.10, Some(0.275), 4.40).flex(0.55, Some(0.138), 2.20),
    model("o3-mini", 1.10, Some(0.55), 4.40),
    model("o1-mini", 1.10, Some(0.55), 4.40),
    model("gpt-4o-mini-search-preview", 0.15, None, 0.60),
    model("gpt-4o-search-preview", 2.50, None, 10.00),
    model("computer-use-preview", 3.00, None, 12.00),
    // Google Gemini
    //
    // Gemini sells the same tiers OpenAI does, and the cheap one carries most
    // background traffic, so leaving it out would bill the common case at
    // twice its rate. Flex is Google's published half-price tier (the batch
    // rate, where that is the only cheap rate published); priority follows the
    // family's 1.8x convention, which Google does not state for every model.
    //
    // Pro is the one Gemini family Google prices by prompt size: every column
    // roughly doubles above 200k.
    model("gemini-3.1-pro", 2.00, Some(0.20), 12.00)
        .flex(1.00, Some(0.20), 6.00)
        .priority(3.60, Some(0.36), 21.60)
        .long_context(
            long(200_000, 4.00, Some(0.40), 18.00)
                .flex(2.00, Some(0.40), 9.00)
                .priority(7.20, Some(0.72), 32.40),
        ),
    model("gemini-3-flash", 0.50, Some(0.05), 3.00)
        .flex(0.25, Some(0.05), 1.50)
        .priority(0.90, Some(0.09), 5.40),
    model("gemini-3.1-flash-lite", 0.25, Some(0.025), 1.50)
        .flex(0.125, Some(0.0125), 0.75)
        .priority(0.45, Some(0.045), 2.70),
    model("gemini-3.5-flash", 1.50, Some(0.15), 9.00)
        .flex(0.75, Some(0.08), 4.50)
        .priority(2.70, Some(0.27), 16.20),
    // Released 2026-07-21 (google blog: gemini-3-6-flash-3-5-flash-lite-3-5-flash-cyber)
    model("gemini-3.6-flash", 1.50, Some(0.15), 7.50)
        .flex(0.75, Some(0.075), 3.75)
        .priority(2.70, Some(0.27), 13.50),
    model("gemini-3.5-flash-lite", 0.30, Some(0.03), 2.50)
        .flex(0.15, Some(0.015), 1.25)
        .priority(0.54, Some(0.054), 4.50),
    model("gemini-2.5-pro", 1.25, Some(0.125), 10.00)
        .flex(0.625, Some(0.125), 5.00)
        .priority(2.25, Some(0.225), 18.00)
        .long_context(
            long(200_000, 2.50, Some(0.25), 15.00)
                .flex(1.25, Some(0.25), 7.50)
                .priority(4.50, Some(0.45), 27.00),
        ),
    model("gemini-2.5-flash", 0.30, Some(0.03), 2.50)
        .flex(0.15, Some(0.03), 1.25)
        .priority(0.54, Some(0.054), 4.50),
    model("gemini-2.5-flash-lite", 0.10, Some(0.01), 0.40)
        .flex(0.05, Some(0.01), 0.20)
        .priority(0.18, Some(0.018), 0.72),
    model("gemini-2.0-flash", 0.15, None, 0.60),
    model("gemini-2.0-flash-lite", 0.075, None, 0.30),
    // GPT-5 models
    //
    // Only the 5.6 family carries a `long_context` card here, because those
    // are the rates we have. OpenAI states the 272k boundary for gpt-5.5 but
    // we have not recorded what it charges past it, and whether the older
    // 5.x models tier by prompt size at all is unconfirmed — so a long
    // request on any of them bills at the short card. Filling these in is the
    // first thing to check when refreshing this table.
    // "gpt-5.6" alias routes to gpt-5.6-sol
    model("gpt-5.6", 5.00, Some(0.50), 30.00)
        .flex(2.50, Some(0.25), 15.00)
        .priority(10.00, Some(1.00), 60.00)
        .long_context(
            long(GPT_5_LONG_CONTEXT, 10.00, Some(1.00), 45.00)
                .flex(5.00, Some(0.50), 22.50)
                .priority(20.00, Some(2.00), 90.00),
        ),
    model("gpt-5.6-sol", 5.00, Some(0.50), 30.00)
        .flex(2.50, Some(0.25), 15.00)
        .priority(10.00, Some(1.00), 60.00)
        .long_context(
            long(GPT_5_LONG_CONTEXT, 10.00, Some(1.00), 45.00)
                .flex(5.00, Some(0.50), 22.50)
                .priority(20.00, Some(2.00), 90.00),
        ),
    // Cut 20% on 2026-07-30 (from 2.50/0.25/15.00); https://openai.com/pricing
    model("gpt-5.6-terra", 2.00, Some(0.20), 12.00)
        .flex(1.00, Some(0.10), 6.00)
        .priority(4.00, Some(0.40), 24.00)
        .long_context(
            long(GPT_5_LONG_CONTEXT, 4.00, Some(0.40), 18.00)
                .flex(2.00, Some(0.20), 9.00)
                .priority(8.00, Some(0.80), 36.00),
        ),
    // Cut 80% on 2026-07-30 (from 1.00/0.10/6.00); https://openai.com/pricing
    model("gpt-5.6-luna", 0.20, Some(0.02), 1.20)
        .flex(0.10, Some(0.01), 0.60)
        .priority(0.40, Some(0.04), 2.40)
        .long_context(
            long(GPT_5_LONG_CONTEXT, 0.40, Some(0.04), 1.80)
                .flex(0.20, Some(0.02), 0.90)
                .priority(0.80, Some(0.08), 3.60),
        ),
    model("gpt-5.5-pro", 30.00, None, 180.00),
    model("gpt-5.5", 5.00, Some(0.50), 30.00)
        .flex(2.50, Some(0.25), 15.00)
        .priority(12.50, Some(1.25), 75.00),
    model("gpt-5.4-pro", 30.00, None, 180.00).flex(15.00, None, 90.00),
    model("gpt-5.4", 2.50, Some(0.25), 15.00)
        .flex(1.25, Some(0.13), 7.50)
        .priority(5.00, Some(0.50), 30.00),
    model("gpt-5.4-mini", 0.75, Some(0.075), 4.50).flex(0.375, Some(0.0375), 2.25),
    model("gpt-5.4-nano", 0.20, Some(0.02), 1.25).flex(0.10, Some(0.01), 0.625),
    model("gpt-5.2-chat-latest", 1.75, Some(0.175), 14.00),
    model("gpt-5.2-pro", 21.00, None, 168.00),
    model("gpt-5.2", 1.75, Some(0.175), 14.00).flex(0.875, Some(0.0875), 7.00),
    model("gpt-5.1-chat-latest", 1.25, Some(0.125), 10.00),
    model("gpt-5.1-codex-max", 1.25, Some(0.125), 10.00),
    model("gpt-5.1-codex", 1.25, Some(0.125), 10.00),
    model("gpt-5.1", 1.25, Some(0.125), 10.00).flex(0.625, Some(0.0625), 5.00),
    model("gpt-5-chat-latest", 1.25, Some(0.125), 10.00),
    model("gpt-5-codex", 1.25, Some(0.125), 10.00),
    model("gpt-5-pro", 15.00, None, 120.00),
    model("gpt-5-mini", 0.25, Some(0.025), 2.00).flex(0.125, Some(0.0125), 1.00),
    model("gpt-5-nano", 0.05, Some(0.005), 0.40).flex(0.025, Some(0.0025), 0.20),
    model("gpt-5", 1.25, Some(0.125), 10.00).flex(0.625, Some(0.0625), 5.00),
];

/// The entry that prices `model`, or `None` if we have no price for it.
///
/// Callers that only want a dollar figure should use [`cost_of_call`]; this is
/// for inspecting the rates themselves, or asking whether a model is priced at
/// all before recording a cost against it.
pub fn price_of(model: &str) -> Option<&'static ModelCost> {
    MODELS
        .iter()
        .filter(|mc| model.starts_with(mc.name))
        .max_by_key(|mc| mc.name.len())
}

/// When this table was last checked against the vendors' pricing pages.
///
/// Stamp it onto anything that records a dollar figure, so historical token
/// counts can be repriced when a vendor moves its list prices. Bump it
/// whenever a rate in [`MODELS`] changes.
pub const RATE_CARD_VERSION: &str = "2026-07-30";

/// The cost in dollars of a **single** request.
///
/// Must be called once per request and the results summed. Summing the token
/// counts of several requests and pricing the total gives a different — and
/// wrong — answer for any model with a [`LongContext`] card, because the
/// premium keys off one request's prompt size.
///
/// Returns `None` for a model we have no price for.
pub fn cost_of_call(model: &str, service_tier: Option<&str>, tokens: CallTokens) -> Option<f64> {
    cost_of_call_split(model, service_tier, tokens).map(|(input, output)| input + output)
}

/// As [`cost_of_call`], but keeping the input and output dollars apart — for
/// callers that report the two separately rather than only the total.
pub fn cost_of_call_split(
    model: &str,
    service_tier: Option<&str>,
    tokens: CallTokens,
) -> Option<(f64, f64)> {
    let model_cost = price_of(model)?;

    let cached = tokens.cached_prompt.min(tokens.prompt);
    let uncached = tokens.prompt - cached;

    Some(model_cost.rates(service_tier, tokens.prompt).charge_split(
        uncached,
        cached,
        tokens.output,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usage(prompt: u32, cached_prompt: u32, output: u32) -> CallTokens {
        CallTokens {
            prompt,
            cached_prompt,
            output,
        }
    }

    #[test]
    fn test_cost() {
        // input: 2.50, cached_input: Some(1.25), output: 10.00
        let cost = cost_of_call("gpt-4o", None, usage(2_000_000, 1_000_000, 1_000_000));
        assert_eq!(cost, Some(2.50 + 1.25 + 10.00));
    }

    #[test]
    fn test_flex_cost() {
        // gpt-5.4 flex: input 1.25, output 7.50
        let c = cost_of_call("gpt-5.4", Some("flex"), usage(1_000_000, 0, 1_000_000));
        assert_eq!(c, Some(1.25 + 7.50));
    }

    #[test]
    fn test_priority_cost() {
        // gpt-5.4 priority: input 5.00, output 30.00
        let c = cost_of_call("gpt-5.4", Some("priority"), usage(1_000_000, 0, 1_000_000));
        assert_eq!(c, Some(5.00 + 30.00));
    }

    #[test]
    fn test_tier_falls_back_to_standard() {
        // gpt-4o has no flex pricing, should fall back to standard
        let u = usage(1_000_000, 0, 1_000_000);
        assert_eq!(
            cost_of_call("gpt-4o", None, u),
            cost_of_call("gpt-4o", Some("flex"), u)
        );
    }

    #[test]
    fn unknown_model_has_no_price() {
        assert_eq!(
            cost_of_call("no-such-model", None, usage(1_000, 0, 1_000)),
            None
        );
    }

    /// A dated model id bills at its family's rates rather than falling off
    /// the table, and the longest matching name wins over a shorter prefix.
    #[test]
    fn dated_ids_resolve_to_the_longest_matching_prefix() {
        let u = usage(1_000_000, 0, 0);
        assert_eq!(
            cost_of_call("gpt-5.6-terra-2026-06-01", None, u),
            cost_of_call("gpt-5.6-terra", None, u),
        );
        // ...and not to the shorter "gpt-5.6" alias, which is priced as sol.
        assert_ne!(
            cost_of_call("gpt-5.6-terra", None, u),
            cost_of_call("gpt-5.6", None, u),
        );
    }

    /// A point release must not resolve to its major version. Prefix matching
    /// makes that failure silent and expensive: `claude-opus-4-5-...` matching
    /// `claude-opus-4` bills at 3x the real rate, and nothing about the result
    /// looks wrong.
    #[test]
    fn dated_point_releases_resolve_to_the_point_release() {
        for (id, expected) in [
            ("claude-opus-4-5-20251101", "claude-opus-4-5"),
            ("claude-opus-4.5", "claude-opus-4.5"),
            ("claude-sonnet-4-5-20250929", "claude-sonnet-4-5"),
            ("claude-haiku-4-5-20251001", "claude-haiku-4-5"),
            ("claude-opus-4-20250514", "claude-opus-4"),
            ("claude-sonnet-4-20250514", "claude-sonnet-4"),
        ] {
            let got = price_of(id).map(|mc| mc.name);
            assert_eq!(got, Some(expected), "{id} resolved to {got:?}");
        }
    }

    /// The premium re-prices the entire request, and a prompt exactly at the
    /// threshold is still billed at the cheaper card.
    #[test]
    fn long_prompts_bill_the_whole_request_at_the_premium() {
        // 200k exactly: 0.2M @ $2.00 = $0.40.
        let at_threshold = cost_of_call("gemini-3.1-pro", None, usage(200_000, 0, 0)).unwrap();
        assert!((at_threshold - 0.40).abs() < 1e-9, "{at_threshold}");

        // One token over flips the whole request to $4.00/M, so ~2x — not a
        // surcharge on the single token past the line.
        let over = cost_of_call("gemini-3.1-pro", None, usage(200_001, 0, 0)).unwrap();
        assert!((over - 0.800004).abs() < 1e-9, "{over}");
    }

    /// The long-context card has tiers of its own; asking for flex above the
    /// threshold must not fall back to the standard long-context rate.
    #[test]
    fn long_context_has_its_own_tiers() {
        let u = usage(300_000, 0, 0);
        let standard = cost_of_call("gpt-5.6-luna", None, u).unwrap();
        let flex = cost_of_call("gpt-5.6-luna", Some("flex"), u).unwrap();
        // 0.3M @ $0.40 standard, @ $0.20 flex.
        assert!((standard - 0.12).abs() < 1e-9, "{standard}");
        assert!((flex - 0.06).abs() < 1e-9, "{flex}");
    }

    /// Pricing each call and summing is not the same as summing tokens and
    /// pricing once — the reason [`cost_of_call`] is per-request.
    #[test]
    fn summing_tokens_would_overbill_short_calls() {
        // Six 50k-token calls: each is well under the threshold, so each bills
        // at the cheap card even though they sum to 300k.
        let per_call = cost_of_call("gpt-5.6-luna", None, usage(50_000, 0, 0)).unwrap();
        let summed_correctly = per_call * 6.0;
        let priced_as_one_lump = cost_of_call("gpt-5.6-luna", None, usage(300_000, 0, 0)).unwrap();
        assert!((summed_correctly - 0.06).abs() < 1e-9, "{summed_correctly}");
        assert!(
            (priced_as_one_lump - 0.12).abs() < 1e-9,
            "{priced_as_one_lump}"
        );
    }
}
