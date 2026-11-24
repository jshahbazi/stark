// SPDX-License-Identifier: Apache-2.0
//! Request-local reasoning budget over the Inkling assistant framing tokens.

use super::END_MESSAGE;

pub(super) const THINKING: u32 = 200_008;
const TOOL_JSON: u32 = 200_049;
const TOOL_TEXT: u32 = 200_057;
const MODEL_MESSAGE: u32 = 200_001;
const CONTENT_TEXT: u32 = 200_004;
const TOOL_ERROR: u32 = 200_022;
const END_SAMPLING: u32 = 200_006;
const OPEN_FRAME_TOKENS: usize = 2;
const CLOSE_AND_NEXT_FRAME_TOKENS: usize = 3;
const MAX_ANSWER_RESERVE: usize = 256;
const MAX_BIAS_WINDOW: usize = 60;
const BIAS_PER_TOKEN: f32 = 0.5;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum SamplingControl {
    Natural,
    BiasClose(f32),
    Close,
    BlockThinking,
}

enum Block {
    Other,
    Thinking,
    ToolJson {
        quoted: bool,
        escaped: bool,
    },
    ToolText,
}

pub(super) struct ThinkingBudget {
    limit: usize,
    answer_reserve: usize,
    used: usize,
    block: Block,
}

impl ThinkingBudget {
    pub(super) fn new(
        requested: u32,
        total: usize,
        starts_thinking: bool,
    ) -> Self {
        let answer_reserve = (total / 4).clamp(1, MAX_ANSWER_RESERVE);
        let framing = CLOSE_AND_NEXT_FRAME_TOKENS
            + if starts_thinking {
                0
            } else {
                OPEN_FRAME_TOKENS
            };
        Self {
            limit: (requested as usize).min(total.saturating_sub(answer_reserve + framing)),
            answer_reserve,
            used: 0,
            block: if starts_thinking {
                Block::Thinking
            } else {
                Block::Other
            },
        }
    }

    pub(super) fn sampling_control(
        &self,
        remaining: usize,
    ) -> SamplingControl {
        let room = remaining.saturating_sub(self.answer_reserve + CLOSE_AND_NEXT_FRAME_TOKENS);
        let remaining_thought = self.limit.saturating_sub(self.used).min(room);
        match self.block {
            Block::Thinking => {
                if remaining_thought == 0 {
                    return SamplingControl::Close;
                }
                let window = (self.limit / 4).clamp(1, MAX_BIAS_WINDOW);
                if remaining_thought <= window {
                    return SamplingControl::BiasClose((window - remaining_thought + 1) as f32 * BIAS_PER_TOKEN);
                }
                SamplingControl::Natural
            },
            Block::Other if remaining_thought == 0 => SamplingControl::BlockThinking,
            _ => SamplingControl::Natural,
        }
    }

    /// Observe committed samples only. Token text is needed solely inside JSON
    /// arguments, where a quoted delimiter is data and cannot end a block.
    pub(super) fn observe(
        &mut self,
        token: u32,
        token_text: impl FnOnce() -> Option<String>,
    ) {
        match &mut self.block {
            Block::ToolJson {
                quoted,
                escaped,
            } if token != END_MESSAGE || *quoted => {
                if let Some(text) = token_text() {
                    for byte in text.bytes() {
                        if *escaped {
                            *escaped = false;
                        } else if *quoted && byte == b'\\' {
                            *escaped = true;
                        } else if byte == b'"' {
                            *quoted = !*quoted;
                        }
                    }
                }
                return;
            },
            Block::ToolText if token != END_MESSAGE => return,
            _ => {},
        }
        match token {
            THINKING if matches!(self.block, Block::Thinking) => self.used += 1,
            THINKING => self.block = Block::Thinking,
            TOOL_JSON => {
                self.block = Block::ToolJson {
                    quoted: false,
                    escaped: false,
                }
            },
            TOOL_TEXT => self.block = Block::ToolText,
            MODEL_MESSAGE | CONTENT_TEXT | TOOL_ERROR | END_MESSAGE | END_SAMPLING => self.block = Block::Other,
            _ if matches!(self.block, Block::Thinking) => self.used += 1,
            _ => {},
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(
        policy: &mut ThinkingBudget,
        tokens: &[u32],
    ) {
        for &token in tokens {
            policy.observe(token, || None);
        }
    }

    #[test]
    fn reasoning_budget_is_shared_across_blocks_and_blocks_a_reopen() {
        let mut policy = ThinkingBudget::new(8, 100, false);
        record(&mut policy, &[MODEL_MESSAGE, THINKING, 1, 2, 3, END_MESSAGE, MODEL_MESSAGE, THINKING, 4, 5]);
        assert_eq!(policy.sampling_control(90), SamplingControl::Natural);
        record(&mut policy, &[6]);
        assert_eq!(policy.sampling_control(89), SamplingControl::BiasClose(0.5));
        record(&mut policy, &[7]);
        assert_eq!(policy.sampling_control(88), SamplingControl::BiasClose(1.0));
        record(&mut policy, &[8]);
        assert_eq!(policy.sampling_control(87), SamplingControl::Close);
        record(&mut policy, &[END_MESSAGE, MODEL_MESSAGE]);
        assert_eq!(policy.sampling_control(85), SamplingControl::BlockThinking);
        record(&mut policy, &[CONTENT_TEXT, 9, END_MESSAGE]);
        assert_eq!(policy.used, 8);
        assert_eq!(policy.sampling_control(82), SamplingControl::BlockThinking);
    }

    #[test]
    fn reserve_accounts_for_prefilled_reasoning_and_additional_framing() {
        let mut policy = ThinkingBudget::new(100, 12, false);
        assert_eq!(policy.answer_reserve, 3);
        assert_eq!(policy.limit, 4);
        record(&mut policy, &[MODEL_MESSAGE, THINKING, 1, END_MESSAGE, MODEL_MESSAGE, THINKING]);
        // Six remaining tokens must accommodate close, next frame and answer.
        assert_eq!(policy.sampling_control(6), SamplingControl::Close);
        record(&mut policy, &[END_MESSAGE]);
        assert_eq!(policy.sampling_control(5), SamplingControl::BlockThinking);

        let mut prefilled = ThinkingBudget::new(100, 12, true);
        assert_eq!(prefilled.limit, 6);
        record(&mut prefilled, &[1, 2, 3, 4, 5, 6]);
        assert_eq!(prefilled.sampling_control(6), SamplingControl::Close);
        assert_eq!(ThinkingBudget::new(100, 4, false).sampling_control(4), SamplingControl::BlockThinking);
    }

    #[test]
    fn repeated_openers_inside_reasoning_consume_budget() {
        let mut policy = ThinkingBudget::new(2, 100, false);
        record(&mut policy, &[THINKING, THINKING, THINKING]);
        assert_eq!(policy.sampling_control(97), SamplingControl::Close);
    }

    #[test]
    fn quoted_tool_delimiters_do_not_start_or_end_reasoning() {
        let mut policy = ThinkingBudget::new(1, 100, false);
        record(&mut policy, &[THINKING, 1, END_MESSAGE, MODEL_MESSAGE, TOOL_JSON]);
        for (token, text) in [
            (1, "{\"literal\":\""),
            (2, "\\"),
            (3, "\""),
            (END_MESSAGE, "<|end_message|>"),
            (THINKING, "<|content_thinking|>"),
            (4, "\"}"),
        ] {
            policy.observe(token, || Some(text.to_owned()));
            assert_eq!(policy.sampling_control(90), SamplingControl::Natural);
        }
        record(&mut policy, &[END_MESSAGE, MODEL_MESSAGE]);
        assert_eq!(policy.used, 1);
        assert_eq!(policy.sampling_control(80), SamplingControl::BlockThinking);
    }
}
