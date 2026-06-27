//! This module contains the pricing of different models.

/// The cost of using a model, in dollars per million tokens.
#[derive(Debug, Clone)]
pub(crate) struct ModelCost {
    pub(crate) name: &'static str,
    /// The cost of input tokens, in dollars per million tokens.
    pub(crate) input: f64,
    /// The cost of cached input tokens, in dollars per million tokens.
    pub(crate) cached_input: Option<f64>,
    /// The cost of output tokens, in dollars per million tokens.
    pub(crate) output: f64,
}

pub(crate) const CHAT_COMPLETIONS: &[ModelCost] = &[
    // Anthropic
    ModelCost {
        name: "claude-opus-4.5",
        input: 5.0,
        cached_input: None,
        output: 25.0,
    },
    ModelCost {
        name: "claude-opus-4.1",
        input: 15.0,
        cached_input: None,
        output: 75.0,
    },
    ModelCost {
        name: "claude-opus-4",
        input: 15.0,
        cached_input: None,
        output: 75.0,
    },
    ModelCost {
        name: "claude-sonnet-4.5",
        input: 3.0,
        cached_input: None,
        output: 15.0,
    },
    ModelCost {
        name: "claude-sonnet-4",
        input: 3.0,
        cached_input: None,
        output: 15.0,
    },
    ModelCost {
        name: "claude-haiku-4",
        input: 0.80,
        cached_input: None,
        output: 4.0,
    },
    ModelCost {
        name: "claude-3-opus",
        input: 15.0,
        cached_input: None,
        output: 75.0,
    },
    ModelCost {
        name: "claude-3-7-sonnet",
        input: 3.0,
        cached_input: None,
        output: 15.0,
    },
    ModelCost {
        name: "claude-3-5-haiku",
        input: 0.80,
        cached_input: None,
        output: 4.0,
    },
    // Anthropic (new models, copied from https://www.anthropic.com/pricing on 2026-06-24)
    ModelCost {
        name: "claude-fable-5",
        input: 10.0,
        cached_input: None,
        output: 50.0,
    },
    ModelCost {
        name: "claude-mythos-5",
        input: 10.0,
        cached_input: None,
        output: 50.0,
    },
    ModelCost {
        name: "claude-opus-4-8",
        input: 5.0,
        cached_input: None,
        output: 25.0,
    },
    ModelCost {
        name: "claude-opus-4-7",
        input: 5.0,
        cached_input: None,
        output: 25.0,
    },
    ModelCost {
        name: "claude-opus-4-6",
        input: 5.0,
        cached_input: None,
        output: 25.0,
    },
    ModelCost {
        name: "claude-sonnet-4-6",
        input: 3.0,
        cached_input: None,
        output: 15.0,
    },
    ModelCost {
        name: "claude-haiku-4-5",
        input: 1.0,
        cached_input: None,
        output: 5.0,
    },
    // OpenAI
    // Copied from https://platform.openai.com/docs/pricing on 2026-03-17
    ModelCost {
        name: "gpt-4.1",
        input: 2.00,
        cached_input: Some(0.50),
        output: 8.00,
    },
    ModelCost {
        name: "gpt-4.1-mini",
        input: 0.40,
        cached_input: Some(0.10),
        output: 1.60,
    },
    ModelCost {
        name: "gpt-4.1-nano",
        input: 0.10,
        cached_input: Some(0.025),
        output: 0.40,
    },
    ModelCost {
        name: "gpt-4.5-preview",
        input: 75.00,
        cached_input: Some(37.50),
        output: 150.00,
    },
    ModelCost {
        name: "gpt-4o",
        input: 2.50,
        cached_input: Some(1.25),
        output: 10.00,
    },
    ModelCost {
        name: "gpt-4o-audio-preview",
        input: 2.50,
        cached_input: None,
        output: 10.00,
    },
    ModelCost {
        name: "gpt-4o-realtime-preview",
        input: 5.00,
        cached_input: Some(2.50),
        output: 20.00,
    },
    ModelCost {
        name: "gpt-4o-mini",
        input: 0.15,
        cached_input: Some(0.075),
        output: 0.60,
    },
    ModelCost {
        name: "gpt-4o-mini-audio-preview",
        input: 0.15,
        cached_input: None,
        output: 0.60,
    },
    ModelCost {
        name: "gpt-4o-mini-realtime-preview",
        input: 0.60,
        cached_input: Some(0.30),
        output: 2.40,
    },
    ModelCost {
        name: "o1",
        input: 15.00,
        cached_input: Some(7.50),
        output: 60.00,
    },
    ModelCost {
        name: "o1-pro",
        input: 150.00,
        cached_input: None,
        output: 600.00,
    },
    ModelCost {
        name: "o3",
        input: 2.00,
        cached_input: Some(0.50),
        output: 8.00,
    },
    ModelCost {
        name: "o4-mini",
        input: 1.10,
        cached_input: Some(0.275),
        output: 4.40,
    },
    ModelCost {
        name: "o3-mini",
        input: 1.10,
        cached_input: Some(0.55),
        output: 4.40,
    },
    ModelCost {
        name: "o1-mini",
        input: 1.10,
        cached_input: Some(0.55),
        output: 4.40,
    },
    ModelCost {
        name: "gpt-4o-mini-search-preview",
        input: 0.15,
        cached_input: None,
        output: 0.60,
    },
    ModelCost {
        name: "gpt-4o-search-preview",
        input: 2.50,
        cached_input: None,
        output: 10.00,
    },
    ModelCost {
        name: "computer-use-preview",
        input: 3.00,
        cached_input: None,
        output: 12.00,
    },
    // Google Gemini
    ModelCost {
        name: "gemini-3.1-pro",
        input: 2.00,
        cached_input: None,
        output: 12.00,
    },
    ModelCost {
        name: "gemini-3-flash",
        input: 0.50,
        cached_input: None,
        output: 3.00,
    },
    ModelCost {
        name: "gemini-3.1-flash-lite",
        input: 0.25,
        cached_input: None,
        output: 1.50,
    },
    ModelCost {
        name: "gemini-3.5-flash",
        input: 1.50,
        cached_input: None,
        output: 9.00,
    },
    ModelCost {
        name: "gemini-2.5-pro",
        input: 1.25,
        cached_input: Some(0.13),
        output: 10.00,
    },
    ModelCost {
        name: "gemini-2.5-flash",
        input: 0.30,
        cached_input: Some(0.03),
        output: 2.50,
    },
    ModelCost {
        name: "gemini-2.5-flash-lite",
        input: 0.10,
        cached_input: Some(0.01),
        output: 0.40,
    },
    ModelCost {
        name: "gemini-2.0-flash",
        input: 0.15,
        cached_input: None,
        output: 0.60,
    },
    ModelCost {
        name: "gemini-2.0-flash-lite",
        input: 0.075,
        cached_input: None,
        output: 0.30,
    },
    // GPT-5 models
    ModelCost {
        name: "gpt-5.5-pro",
        input: 30.00,
        cached_input: None,
        output: 180.00,
    },
    ModelCost {
        name: "gpt-5.5",
        input: 5.00,
        cached_input: Some(0.50),
        output: 30.00,
    },
    ModelCost {
        name: "gpt-5.4-pro",
        input: 30.00,
        cached_input: None,
        output: 180.00,
    },
    ModelCost {
        name: "gpt-5.4",
        input: 2.50,
        cached_input: Some(0.25),
        output: 15.00,
    },
    ModelCost {
        name: "gpt-5.4-mini",
        input: 0.75,
        cached_input: Some(0.075),
        output: 4.50,
    },
    ModelCost {
        name: "gpt-5.4-nano",
        input: 0.20,
        cached_input: Some(0.02),
        output: 1.25,
    },
    ModelCost {
        name: "gpt-5.2-chat-latest",
        input: 1.75,
        cached_input: Some(0.175),
        output: 14.00,
    },
    ModelCost {
        name: "gpt-5.2-pro",
        input: 21.00,
        cached_input: None,
        output: 168.00,
    },
    ModelCost {
        name: "gpt-5.2",
        input: 1.75,
        cached_input: Some(0.175),
        output: 14.00,
    },
    ModelCost {
        name: "gpt-5.1-chat-latest",
        input: 1.25,
        cached_input: Some(0.125),
        output: 10.00,
    },
    ModelCost {
        name: "gpt-5.1-codex-max",
        input: 1.25,
        cached_input: Some(0.125),
        output: 10.00,
    },
    ModelCost {
        name: "gpt-5.1-codex",
        input: 1.25,
        cached_input: Some(0.125),
        output: 10.00,
    },
    ModelCost {
        name: "gpt-5.1",
        input: 1.25,
        cached_input: Some(0.125),
        output: 10.00,
    },
    ModelCost {
        name: "gpt-5-chat-latest",
        input: 1.25,
        cached_input: Some(0.125),
        output: 10.00,
    },
    ModelCost {
        name: "gpt-5-codex",
        input: 1.25,
        cached_input: Some(0.125),
        output: 10.00,
    },
    ModelCost {
        name: "gpt-5-pro",
        input: 15.00,
        cached_input: None,
        output: 120.00,
    },
    ModelCost {
        name: "gpt-5-mini",
        input: 0.25,
        cached_input: Some(0.025),
        output: 2.00,
    },
    ModelCost {
        name: "gpt-5-nano",
        input: 0.05,
        cached_input: Some(0.005),
        output: 0.40,
    },
    ModelCost {
        name: "gpt-5",
        input: 1.25,
        cached_input: Some(0.125),
        output: 10.00,
    },
];

// Flex tier pricing (OpenAI)
// Copied from https://platform.openai.com/docs/pricing on 2026-03-17
pub(crate) const FLEX_PRICES: &[ModelCost] = &[
    ModelCost {
        name: "gpt-5.5",
        input: 2.50,
        cached_input: Some(0.25),
        output: 15.00,
    },
    ModelCost {
        name: "gpt-5.4",
        input: 1.25,
        cached_input: Some(0.13),
        output: 7.50,
    },
    ModelCost {
        name: "gpt-5.4-pro",
        input: 15.00,
        cached_input: None,
        output: 90.00,
    },
    ModelCost {
        name: "gpt-5.4-mini",
        input: 0.375,
        cached_input: Some(0.0375),
        output: 2.25,
    },
    ModelCost {
        name: "gpt-5.4-nano",
        input: 0.10,
        cached_input: Some(0.01),
        output: 0.625,
    },
    ModelCost {
        name: "gpt-5.2",
        input: 0.875,
        cached_input: Some(0.0875),
        output: 7.00,
    },
    ModelCost {
        name: "gpt-5.1",
        input: 0.625,
        cached_input: Some(0.0625),
        output: 5.00,
    },
    ModelCost {
        name: "gpt-5",
        input: 0.625,
        cached_input: Some(0.0625),
        output: 5.00,
    },
    ModelCost {
        name: "gpt-5-mini",
        input: 0.125,
        cached_input: Some(0.0125),
        output: 1.00,
    },
    ModelCost {
        name: "gpt-5-nano",
        input: 0.025,
        cached_input: Some(0.0025),
        output: 0.20,
    },
    ModelCost {
        name: "o3",
        input: 1.00,
        cached_input: Some(0.25),
        output: 4.00,
    },
    ModelCost {
        name: "o4-mini",
        input: 0.55,
        cached_input: Some(0.138),
        output: 2.20,
    },
];

// Priority tier pricing (OpenAI)
// Copied from https://platform.openai.com/docs/pricing on 2026-03-17
pub(crate) const PRIORITY_PRICES: &[ModelCost] = &[
    ModelCost {
        name: "gpt-5.5",
        input: 12.50,
        cached_input: Some(1.25),
        output: 75.00,
    },
    ModelCost {
        name: "gpt-5.4",
        input: 5.00,
        cached_input: Some(0.50),
        output: 30.00,
    },
];

fn lookup<'a>(table: &'a [ModelCost], model: &str) -> Option<&'a ModelCost> {
    table
        .iter()
        .filter(|mc| model.starts_with(mc.name))
        .max_by_key(|mc| mc.name.len())
}

pub(crate) fn cost(
    model: &str,
    service_tier: Option<&str>,
    usage: crate::chat_completions::ChatUsage,
) -> Option<f64> {
    let tier_table = match service_tier {
        Some("flex") => Some(FLEX_PRICES),
        Some("priority") => Some(PRIORITY_PRICES),
        _ => None,
    };
    let model_cost = tier_table
        .and_then(|t| lookup(t, model))
        .or_else(|| lookup(CHAT_COMPLETIONS, model))?;

    let (cached_prompt_tokens, uncached_prompt_tokens) =
        if let Some(details) = usage.prompt_token_details {
            (
                details.cached_tokens,
                usage.prompt_tokens - details.cached_tokens,
            )
        } else {
            (0, usage.prompt_tokens)
        };

    let usage_cost = model_cost.input * uncached_prompt_tokens as f64 / 1_000_000.0
        + model_cost.cached_input.unwrap_or(model_cost.input) * cached_prompt_tokens as f64
            / 1_000_000.0
        + model_cost.output * usage.completion_tokens as f64 / 1_000_000.0;
    Some(usage_cost)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cost() {
        let usage = crate::chat_completions::ChatUsage {
            prompt_tokens: 2000000,
            completion_tokens: 1000000,
            prompt_token_details: Some(crate::chat_completions::PromptTokenDetails {
                cached_tokens: 1000000,
            }),
            completion_token_details: None,
            total_tokens: 2000000,
        };
        let cost = cost("gpt-4o", None, usage);
        // input: 2.50,
        // cached_input: Some(1.25),
        // output: 10.00,
        assert_eq!(cost, Some(2.50 + 1.25 + 10.00));
    }

    #[test]
    fn test_flex_cost() {
        let usage = crate::chat_completions::ChatUsage {
            prompt_tokens: 1000000,
            completion_tokens: 1000000,
            prompt_token_details: None,
            completion_token_details: None,
            total_tokens: 2000000,
        };
        // gpt-5.4 flex: input 1.25, output 7.50
        let c = cost("gpt-5.4", Some("flex"), usage);
        assert_eq!(c, Some(1.25 + 7.50));
    }

    #[test]
    fn test_priority_cost() {
        let usage = crate::chat_completions::ChatUsage {
            prompt_tokens: 1000000,
            completion_tokens: 1000000,
            prompt_token_details: None,
            completion_token_details: None,
            total_tokens: 2000000,
        };
        // gpt-5.4 priority: input 5.00, output 30.00
        let c = cost("gpt-5.4", Some("priority"), usage);
        assert_eq!(c, Some(5.00 + 30.00));
    }

    #[test]
    fn test_tier_falls_back_to_standard() {
        let usage = crate::chat_completions::ChatUsage {
            prompt_tokens: 1000000,
            completion_tokens: 1000000,
            prompt_token_details: None,
            completion_token_details: None,
            total_tokens: 2000000,
        };
        // gpt-4o has no flex pricing, should fall back to standard
        let standard = cost("gpt-4o", None, usage);
        let flex = cost("gpt-4o", Some("flex"), usage);
        assert_eq!(standard, flex);
    }
}
