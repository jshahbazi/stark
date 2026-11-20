//! Request-scoped prompt-lookup drafts for MTP.
//!
//! An index maps every 4-gram of the request context (prompt plus committed
//! output) to the position after its latest earlier occurrence. When the
//! current tail agrees backwards with that occurrence for long enough, the
//! tokens that followed it are proposed as drafts. The token ids exist only in
//! this structure for one request: they are never logged or persisted, and the
//! index is dropped with the decode stream on completion, failure and
//! cancellation. `Debug` is deliberately not implemented.

use std::{collections::HashMap, sync::Arc};

use tokenizers::Tokenizer;

/// Gram length of the index key.
pub const GRAM: usize = 4;
/// Backward agreement needed when the agreeing span contains a line break.
pub const MIN_AGREEMENT_ACROSS_LINE: usize = 8;
/// Backward agreement needed otherwise.
pub const MIN_AGREEMENT: usize = 32;

/// Vocabulary ids whose decoded text contains a line feed. Tokenizer
/// metadata only, built once per loaded model; no request data.
pub struct NewlineTokens {
    bits: Vec<u64>,
}

impl NewlineTokens {
    pub fn from_predicate(
        vocabulary: usize,
        mut contains_newline: impl FnMut(u32) -> bool,
    ) -> Self {
        let mut bits = vec![0u64; vocabulary.div_ceil(64)];
        for id in 0..vocabulary {
            if contains_newline(id as u32) {
                bits[id / 64] |= 1 << (id % 64);
            }
        }
        Self {
            bits,
        }
    }

    /// Decodes each id alone, without skipping special tokens. A partial
    /// UTF-8 byte token decodes to a replacement character, which is not a
    /// line feed; a line feed is always a complete single byte.
    pub fn from_tokenizer(
        tokenizer: &Tokenizer,
        vocabulary: usize,
    ) -> Self {
        Self::from_predicate(vocabulary, |id| tokenizer.decode(&[id], false).is_ok_and(|text| text.contains('\n')))
    }

    pub fn contains(
        &self,
        id: u32,
    ) -> bool {
        let id = id as usize;
        self.bits.get(id / 64).is_some_and(|word| word & (1 << (id % 64)) != 0)
    }

    pub fn count(&self) -> usize {
        self.bits.iter().map(|word| word.count_ones() as usize).sum()
    }
}

/// Per-request n-gram index over the context.
pub struct PromptLookup {
    tokens: Vec<u32>,
    /// Gram ending at `e` maps to `e + 1`, for every `e + 1 < tokens.len()`.
    /// Later occurrences overwrite earlier ones.
    index: HashMap<[u32; GRAM], u32>,
    newline: Arc<NewlineTokens>,
}

fn gram(
    tokens: &[u32],
    end: usize,
) -> [u32; GRAM] {
    let mut key = [0; GRAM];
    key.copy_from_slice(&tokens[end + 1 - GRAM..=end]);
    key
}

impl PromptLookup {
    /// `context` is the prompt followed by every committed output token, the
    /// last one being the pending input of the next round.
    pub fn new(
        context: Vec<u32>,
        newline: Arc<NewlineTokens>,
    ) -> Self {
        let mut index = HashMap::with_capacity(context.len());
        for end in GRAM - 1..context.len().saturating_sub(1) {
            index.insert(gram(&context, end), (end + 1) as u32);
        }
        Self {
            tokens: context,
            index,
            newline,
        }
    }

    /// Appends one committed token.
    pub fn push(
        &mut self,
        token: u32,
    ) {
        let end = self.tokens.len().wrapping_sub(1);
        self.tokens.push(token);
        if self.tokens.len() > GRAM {
            self.index.insert(gram(&self.tokens, end), (end + 1) as u32);
        }
    }

    /// Up to `limit` tokens that followed the latest earlier occurrence of the
    /// current tail, or nothing when the agreement rule is not met.
    pub fn propose(
        &self,
        limit: usize,
    ) -> Vec<u32> {
        let n = self.tokens.len();
        if limit == 0 || n < GRAM {
            return Vec::new();
        }
        let Some(&next) = self.index.get(&gram(&self.tokens, n - 1)) else {
            return Vec::new();
        };
        let next = next as usize;
        let mut agreement = 0;
        let mut line = false;
        while agreement < MIN_AGREEMENT
            && agreement < next
            && self.tokens[next - 1 - agreement] == self.tokens[n - 1 - agreement]
        {
            line |= self.newline.contains(self.tokens[n - 1 - agreement]);
            agreement += 1;
        }
        if agreement < MIN_AGREEMENT && !(line && agreement >= MIN_AGREEMENT_ACROSS_LINE) {
            return Vec::new();
        }
        self.tokens[next..n.min(next + limit)].to_vec()
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    const NL: u32 = 10;

    fn newline() -> Arc<NewlineTokens> {
        Arc::new(NewlineTokens::from_predicate(1000, |id| id == NL))
    }

    fn lookup(tokens: &[u32]) -> PromptLookup {
        PromptLookup::new(tokens.to_vec(), newline())
    }

    #[stark_test]
    fn newline_set_is_a_bitset_over_the_vocabulary() {
        let set = NewlineTokens::from_predicate(130, |id| id % 64 == 3 || id == 129);
        assert!(set.contains(3) && set.contains(67) && set.contains(129));
        assert!(!set.contains(4) && !set.contains(130) && !set.contains(u32::MAX));
        assert_eq!(set.count(), 3);
    }

    #[stark_test]
    fn copies_after_eight_tokens_of_agreement_across_a_line_break() {
        // A line, then the same line again up to its eighth token.
        let line: Vec<u32> = vec![100, 101, 102, NL, 103, 104, 105, 106, 107, 108, 109];
        let mut tokens = line.clone();
        tokens.extend_from_slice(&line[..8]);
        assert_eq!(lookup(&tokens).propose(3), vec![107, 108, 109]);
        // The proposal may run on into the copy that follows the match.
        assert_eq!(lookup(&tokens).propose(8), vec![107, 108, 109, 100, 101, 102, NL, 103]);
        assert_eq!(lookup(&tokens).propose(1), vec![107]);
        assert!(lookup(&tokens).propose(0).is_empty());
        // Seven tokens of agreement are not enough.
        let mut short = line.clone();
        short.push(1);
        short.extend_from_slice(&line[1..8]);
        assert!(lookup(&short).propose(3).is_empty());
    }

    #[stark_test]
    fn without_a_line_break_thirty_two_tokens_are_required() {
        let run: Vec<u32> = (200..240).collect();
        let mut tokens = run.clone();
        tokens.extend_from_slice(&run[..31]);
        assert!(lookup(&tokens).propose(3).is_empty());
        tokens.push(run[31]);
        assert_eq!(lookup(&tokens).propose(3), vec![232, 233, 234]);
    }

    #[stark_test]
    fn latest_occurrence_wins_and_push_matches_rebuild() {
        let mut first: Vec<u32> = vec![NL, 1, 2, 3, 4, 5, 6, 7, 8, 50];
        let second: Vec<u32> = vec![NL, 1, 2, 3, 4, 5, 6, 7, 8, 60];
        first.extend_from_slice(&second);
        first.extend_from_slice(&second[..9]);
        assert_eq!(lookup(&first).propose(2), vec![60, NL]);
        // Incremental pushes give the same index as a fresh build.
        let mut incremental = lookup(&first[..5]);
        for &token in &first[5..] {
            incremental.push(token);
        }
        let rebuilt = lookup(&first);
        assert_eq!(incremental.index, rebuilt.index);
        assert_eq!(incremental.propose(3), rebuilt.propose(3));
        assert_eq!(incremental.len(), first.len());
    }

    #[stark_test]
    fn the_current_tail_never_matches_itself() {
        let tokens: Vec<u32> = (0..64).map(|i| 300 + i).collect();
        assert!(lookup(&tokens).propose(3).is_empty());
        assert!(lookup(&[1, 2, 3]).propose(3).is_empty());
        assert!(lookup(&[]).propose(3).is_empty());
        let mut grow = lookup(&[]);
        for token in [NL, 1, 2, 3] {
            grow.push(token);
        }
        assert!(grow.propose(1).is_empty());
    }

    #[stark_test]
    fn periodic_context_proposes_its_continuation() {
        // A short period: agreement extends across many repeats.
        let period = [NL, 7, 8, 9];
        let tokens: Vec<u32> = period.iter().copied().cycle().take(40).collect();
        assert_eq!(lookup(&tokens).propose(3), vec![NL, 7, 8]);
    }
}
