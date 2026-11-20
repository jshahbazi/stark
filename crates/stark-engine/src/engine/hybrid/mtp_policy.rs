//! Deterministic per-round MTP depth planning and prompt-lookup scheduling.
//!
//! The plan for each round is a pure function of request-visible state: the
//! acceptance history of earlier rounds, the remaining budget, the sampling
//! mode and a static cost table. No wall-clock measurement enters a plan, so
//! a seeded request reproduces the same output on every run. Attempt
//! `20261119-LILYDEPTH-01`.

use serde::{Serialize, ser::SerializeStruct};

use super::mtp_lookup::PromptLookup;

/// Static per-round wall costs, in milliseconds, for one sampling mode.
///
/// Every number is a measured uninstrumented wall time on the Lily Q4-XL
/// artifact (Apple M5 Max, 4,096 input tokens, image tower loaded); see
/// [`CostTable::LILY_Q4_XL_LILYMERGE_01`] for the source. Replace the table
/// when kernels or host sequencing change; nothing else in the policy
/// depends on its values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RoundCosts {
    /// Zero-draft round inside an MTP stream: one ordinary decode step with a
    /// complete readout and host target draw (no verify, finish or head
    /// advance). Its host encoding is not overlapped with GPU work, so it is
    /// charged as measured inside the stream, not as MTP-off decode.
    pub serial_ms: f64,
    /// Verify pass wall by verified rows; index 0 is one row. A one-row
    /// verify is never planned (zero drafts run `serial_ms`); the entry only
    /// anchors the table.
    pub verify_ms: &'static [f64],
    /// Increment per verified row past the end of `verify_ms`. Used only if
    /// the draft limit grows past the measured table.
    pub verify_extra_row_ms: f64,
    /// Draft phase wall by drafts in the round (index 0 is zero drafts):
    /// head steps, their readouts or device chain, and draft selection.
    pub draft_ms: &'static [f64],
    pub draft_extra_ms: f64,
    /// Finish pass (rollback plus head catch-up) by drafts in the round.
    pub finish_ms: &'static [f64],
    pub finish_extra_draft_ms: f64,
    /// Host target sampling and bookkeeping by drafts in the round.
    pub host_ms: &'static [f64],
    pub host_extra_draft_ms: f64,
    /// Head cache write for the pending row in a lookup round.
    pub head_advance_ms: f64,
    /// Extra wall of a lookup round's verify, which is encoded after the
    /// proposals are known instead of ahead during the draft phase.
    pub lookup_verify_extra_ms: f64,
}

/// Greedy and sampled cost tables, selected by the request's sampling mode.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CostTable {
    /// Stable identifier reported in `stats.schedule.cost_table` whenever the
    /// adaptive policy plans with this table. Seeded sampled output under the
    /// adaptive policy depends on the table's constants, so a new or retuned
    /// table must carry a new identifier.
    pub id: &'static str,
    pub greedy: RoundCosts,
    pub sampled: RoundCosts,
}

impl CostTable {
    /// Source: attempt `20261119-LILYMERGE-01`, measured on the merged worker
    /// (host round trips, GDN replay and this planner; SHA-256 prefix
    /// `d4182a67`), quiet host.
    ///
    /// - Verify, draft, finish and host wall at 1, 2 and 3 drafts: frozen
    ///   corpus quiet A/B (`runtime/quiet-ab/q1-summary.json`, four requests
    ///   per arm). Host is the round less verify, draft and finish. Greedy
    ///   rounds measured 19.92/26.17/32.96 ms, sampled (temperature 1, top-k
    ///   20, top-p 0.95) 19.72/26.05/33.17 ms; the table reproduces them
    ///   within 0.02 ms. Greedy drafts chain on the device and cost about
    ///   1.02 ms per extra draft against 1.33 ms sampled, so the draft phase
    ///   is a table by drafts. Index 0 of each verify table is the two-row
    ///   value less the two-to-three-row slope; it is not used by the planner.
    /// - `serial_ms`: mean zero-draft round in the middle of MTP streams on
    ///   4k prose (`runtime/agent-runs/merge-costs-m1-prose.json`, 896 greedy
    ///   and 104 sampled rounds), 0.5 ms above MTP-off decode (13.23 and
    ///   13.06 ms in the quiet A/B).
    /// - `head_advance_ms` and `lookup_verify_extra_ms`: lookup rounds on
    ///   copy-heavy input (`runtime/quiet-ab/q1-6-cand-copy.json`, 0.40 ms
    ///   advance; greedy lookup rounds of 2 and 3 proposals measured 24.71
    ///   and 30.50 ms, sampled 24.54 ms at 2).
    pub const LILY_Q4_XL_LILYMERGE_01: Self = Self {
        id: "lily-q4-xl-lilymerge-01",
        greedy: RoundCosts {
            serial_ms: 13.75,
            verify_ms: &[12.80, 17.83, 22.86, 28.44],
            verify_extra_row_ms: 5.58,
            draft_ms: &[0.0, 1.27, 2.29, 3.31],
            draft_extra_ms: 1.02,
            finish_ms: &[0.58, 0.75, 0.92, 1.09],
            finish_extra_draft_ms: 0.17,
            host_ms: &[0.04, 0.07, 0.10, 0.13],
            host_extra_draft_ms: 0.03,
            head_advance_ms: 0.40,
            lookup_verify_extra_ms: 0.44,
        },
        sampled: RoundCosts {
            serial_ms: 13.61,
            verify_ms: &[12.66, 17.44, 22.22, 27.70],
            verify_extra_row_ms: 5.48,
            draft_ms: &[0.0, 1.36, 2.67, 4.00],
            draft_extra_ms: 1.33,
            finish_ms: &[0.55, 0.74, 0.93, 1.21],
            finish_extra_draft_ms: 0.28,
            host_ms: &[0.13, 0.18, 0.24, 0.26],
            host_extra_draft_ms: 0.02,
            head_advance_ms: 0.40,
            lookup_verify_extra_ms: 0.75,
        },
    };

    pub fn for_mode(
        &self,
        greedy: bool,
    ) -> RoundCosts {
        if greedy {
            self.greedy
        } else {
            self.sampled
        }
    }
}

fn extended(
    table: &[f64],
    extra: f64,
    index: usize,
) -> f64 {
    match table.get(index) {
        Some(&value) => value,
        None => {
            let last = table.len().saturating_sub(1);
            table.get(last).copied().unwrap_or(0.0) + extra * (index - last) as f64
        },
    }
}

impl RoundCosts {
    fn verify(
        &self,
        rows: usize,
    ) -> f64 {
        extended(self.verify_ms, self.verify_extra_row_ms, rows.saturating_sub(1))
    }

    fn tail(
        &self,
        drafts: usize,
    ) -> f64 {
        self.verify(drafts + 1)
            + extended(self.finish_ms, self.finish_extra_draft_ms, drafts)
            + extended(self.host_ms, self.host_extra_draft_ms, drafts)
    }

    /// A head round of `drafts` proposals; zero is a serial step.
    pub fn head_round_ms(
        &self,
        drafts: usize,
    ) -> f64 {
        if drafts == 0 {
            return self.serial_ms;
        }
        extended(self.draft_ms, self.draft_extra_ms, drafts) + self.tail(drafts)
    }

    /// A prompt-lookup round: no head forward, one head cache advance, and a
    /// verify encoded after the proposals are known.
    pub fn lookup_round_ms(
        &self,
        drafts: usize,
    ) -> f64 {
        self.head_advance_ms + self.lookup_verify_extra_ms + self.tail(drafts)
    }
}

/// Exponential-average weight for each observed position once it has been
/// observed often enough. 1/16 averages over roughly the last sixteen
/// observations of a position, about 30 to 40 output tokens: short enough
/// to follow a prose-to-code switch within one response, and smooth enough
/// that a short run of rejections on prose (one draft accepted about two
/// times in three) does not drop the plan to serial steps. Exact in binary.
pub const ACCEPTANCE_WEIGHT: f64 = 0.0625;
/// Weight of the prior, in observations, while a position is new. The n-th
/// observation of a position has weight `max(1 / (n + 2), 1/16)`: the
/// estimate is the mean of the prior counted twice and the outcomes so far
/// until the fourteenth observation, then the exponential average. A
/// request on low-acceptance text leaves the opening depth after a few
/// rounds instead of the 11 to 21 a fixed 1/8 weight needed.
pub const PRIOR_OBSERVATIONS: f64 = 2.0;
/// Starting estimate for every head position. At 0.85 the planner opens at
/// two drafts in both cost tables, the recommended fixed depth.
pub const ACCEPTANCE_PRIOR: f64 = 0.85;
/// Starting estimate for every prompt-lookup position. A lookup proposal
/// needs 8 to 32 tokens of exact backward agreement, and greedy lookup
/// drafts were accepted 97 to 100 percent of the time on copied text. At the
/// head prior a lookup round could never beat a three-draft head plan on
/// code, so lookup was never tried and its estimate never learned; at 0.95
/// it is tried, and a lookup that is rejected (sampled requests at
/// temperature 1 accepted 0 to 2 of 6) falls below the head plan after one
/// or two rounds and is re-tried only as slowly as serial steps are.
pub const LOOKUP_PRIOR: f64 = 0.95;
/// Per-round pull of the first position, while unobserved, toward the
/// prior. Serial steps observe nothing, so this is what re-tries drafting
/// after a hard stretch instead of freezing (about 66 rounds to halve the
/// gap). Half the 1/48 pull of attempt `20261119-LILYDEPTH-01`: with the
/// slower 1/16 average a probe out of serial steps lasts several rounds, so
/// probes must be rarer.
pub const ACCEPTANCE_DECAY: f64 = 1.0 / 96.0;
/// Per-round pull of every later unobserved position toward the estimate of
/// the position before it. A position the plan has stopped verifying is
/// assumed about as acceptable as the last one it verifies: once the
/// shallower positions show a deeper plan would pay (code after prose, or
/// code from the first rounds), it is tried within a few rounds instead of
/// waiting for a fixed prior that never does.
pub const ACCEPTANCE_FOLLOW: f64 = 1.0 / 16.0;
/// A challenger depth must promise at least this much more expected tokens
/// per millisecond than the incumbent before the plan changes. Two percent
/// keeps a plan from flipping on one observation while letting three drafts
/// replace two on code, where it measured 3.6 percent faster.
pub const HYSTERESIS: f64 = 0.02;

/// Expected committed tokens of a round with `drafts` proposals:
/// 1 + sum over k of the product of the conditional acceptances a[0..k].
pub fn expected_tokens(
    acceptance: &[f64],
    drafts: usize,
) -> f64 {
    let mut expected = 1.0;
    let mut reach = 1.0;
    for i in 0..drafts {
        reach *= acceptance.get(i).copied().unwrap_or(0.0);
        expected += reach;
    }
    expected
}

/// Per-position conditional acceptance estimates and observation counts.
#[derive(Clone, Debug, PartialEq)]
struct Estimates {
    prior: f64,
    acceptance: Vec<f64>,
    observed: Vec<u32>,
}

impl Estimates {
    fn new(
        limit: usize,
        prior: f64,
    ) -> Self {
        Self {
            prior,
            acceptance: vec![prior; limit],
            observed: vec![0; limit],
        }
    }

    /// Positions are updated in order, so an unobserved position drifts
    /// toward the already updated estimate of the position before it.
    fn update(
        &mut self,
        proposed: usize,
        accepted: usize,
    ) {
        for i in 0..self.acceptance.len() {
            if i < proposed && i <= accepted {
                let outcome = if i < accepted {
                    1.0
                } else {
                    0.0
                };
                self.observed[i] = self.observed[i].saturating_add(1);
                let weight = (1.0 / (f64::from(self.observed[i]) + PRIOR_OBSERVATIONS)).max(ACCEPTANCE_WEIGHT);
                self.acceptance[i] += weight * (outcome - self.acceptance[i]);
            } else {
                let (target, pull) = match i {
                    0 => (self.prior, ACCEPTANCE_DECAY),
                    _ => (self.acceptance[i - 1], ACCEPTANCE_FOLLOW),
                };
                self.acceptance[i] += pull * (target - self.acceptance[i]);
            }
        }
    }
}

/// What one round runs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RoundPlan {
    /// Draft-head proposals; zero means a single-row verify.
    Head(usize),
    /// Point-mass proposals copied from the request context.
    Lookup(Vec<u32>),
}

impl RoundPlan {
    pub fn drafts(&self) -> usize {
        match self {
            Self::Head(drafts) => *drafts,
            Self::Lookup(tokens) => tokens.len(),
        }
    }
}

/// Request-scoped depth planner. Pure: its state changes only through
/// [`DepthPlanner::observe`].
#[derive(Clone, Debug)]
pub struct DepthPlanner {
    costs: RoundCosts,
    limit: usize,
    head: Estimates,
    lookup: Estimates,
    incumbent: Option<usize>,
}

impl DepthPlanner {
    pub fn new(
        costs: RoundCosts,
        limit: usize,
    ) -> Self {
        Self {
            costs,
            limit,
            head: Estimates::new(limit, ACCEPTANCE_PRIOR),
            lookup: Estimates::new(limit, LOOKUP_PRIOR),
            incumbent: None,
        }
    }

    pub fn head_rate(
        &self,
        drafts: usize,
    ) -> f64 {
        expected_tokens(&self.head.acceptance, drafts) / self.costs.head_round_ms(drafts)
    }

    pub fn lookup_rate(
        &self,
        drafts: usize,
    ) -> f64 {
        expected_tokens(&self.lookup.acceptance, drafts) / self.costs.lookup_round_ms(drafts)
    }

    /// Head depth in `0..=cap` with the best expected tokens per millisecond,
    /// keeping the incumbent unless a challenger beats it by [`HYSTERESIS`].
    /// Ties go to the smaller depth; increases are limited to one step per
    /// round. A cap below the limit (the end of the budget) does not replace
    /// the incumbent.
    pub fn head_depth(
        &mut self,
        cap: usize,
    ) -> usize {
        let cap = cap.min(self.limit);
        // Depth rises at most one step per round, so a probe out of serial
        // steps (or out of any shallow plan) tries the next depth first;
        // positions past it may hold stale, optimistic estimates.
        let reach = self.incumbent.map_or(cap, |incumbent| cap.min(incumbent + 1));
        let mut best = 0;
        for drafts in 1..=reach {
            if self.head_rate(drafts) > self.head_rate(best) {
                best = drafts;
            }
        }
        let chosen = match self.incumbent.map(|incumbent| incumbent.min(cap)) {
            Some(incumbent) if self.head_rate(best) <= self.head_rate(incumbent) * (1.0 + HYSTERESIS) => incumbent,
            _ => best,
        };
        if cap == self.limit {
            self.incumbent = Some(chosen);
        }
        chosen
    }

    /// Lookup depth in `1..=available` that beats `head` drafts, if any.
    pub fn lookup_depth(
        &self,
        available: usize,
        head: usize,
    ) -> Option<usize> {
        let available = available.min(self.limit);
        let mut best = None::<usize>;
        for drafts in 1..=available {
            if best.is_none_or(|best| self.lookup_rate(drafts) > self.lookup_rate(best)) {
                best = Some(drafts);
            }
        }
        best.filter(|&drafts| self.lookup_rate(drafts) >= self.head_rate(head))
    }

    /// Updates the estimates of the kind that ran from its outcome, and pulls
    /// every unobserved position toward its predecessor (the first toward
    /// the prior).
    pub fn observe(
        &mut self,
        lookup: bool,
        proposed: usize,
        accepted: usize,
    ) {
        let (ran, idle) = if lookup {
            (&mut self.lookup, &mut self.head)
        } else {
            (&mut self.head, &mut self.lookup)
        };
        ran.update(proposed, accepted);
        idle.update(0, 0);
    }
}

/// Request selector for depth planning.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MtpPolicy {
    /// The request's draft count every round (the default).
    #[default]
    Fixed,
    /// Per-round depth up to the request's draft count.
    Adaptive,
}

impl MtpPolicy {
    pub fn name(self) -> &'static str {
        match self {
            Self::Fixed => "fixed",
            Self::Adaptive => "adaptive",
        }
    }
}

/// Planned-depth slots in the counters. Nine covers a draft limit of eight.
pub const DEPTH_SLOTS: usize = 9;

/// Aggregate scheduling counters. No token, logit or probability data.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MtpScheduleStats {
    pub policy: MtpPolicy,
    pub lookup: bool,
    pub limit: usize,
    /// Head rounds by drafts run (index = drafts).
    pub head_rounds_by_depth: [usize; DEPTH_SLOTS],
    /// Wall seconds of those rounds, for checking the cost table. Reported
    /// only; never read by the planner.
    pub head_round_seconds_by_depth: [f64; DEPTH_SLOTS],
    pub lookup_round_seconds: f64,
    pub lookup_rounds: usize,
    pub lookup_proposed: usize,
    pub lookup_accepted: usize,
    /// Rounds without head drafts that wrote the head's cache row instead.
    pub head_advances: usize,
    pub head_advance_seconds: f64,
    /// [`CostTable::id`] of the table the adaptive policy planned with;
    /// `None` under the fixed policy, which reads no table.
    pub cost_table: Option<&'static str>,
}

impl Serialize for MtpScheduleStats {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        let slots = (self.limit + 1).min(DEPTH_SLOTS);
        let mut state = serializer.serialize_struct("MtpScheduleStats", 11 + usize::from(self.cost_table.is_some()))?;
        state.serialize_field("policy", self.policy.name())?;
        if let Some(cost_table) = self.cost_table {
            state.serialize_field("cost_table", cost_table)?;
        }
        state.serialize_field("lookup", &self.lookup)?;
        state.serialize_field("limit", &self.limit)?;
        state.serialize_field("head_rounds_by_depth", &self.head_rounds_by_depth[..slots])?;
        state.serialize_field("head_round_seconds_by_depth", &self.head_round_seconds_by_depth[..slots])?;
        state.serialize_field("lookup_round_seconds", &self.lookup_round_seconds)?;
        state.serialize_field("lookup_rounds", &self.lookup_rounds)?;
        state.serialize_field("lookup_proposed", &self.lookup_proposed)?;
        state.serialize_field("lookup_accepted", &self.lookup_accepted)?;
        state.serialize_field("head_advances", &self.head_advances)?;
        state.serialize_field("head_advance_seconds", &self.head_advance_seconds)?;
        state.end()
    }
}

/// Request-scoped round scheduler: the planner, the optional prompt-lookup
/// index and their counters. Dropped with the decode stream on completion,
/// failure and cancellation; the lookup tokens never leave memory.
pub struct RoundScheduler {
    policy: MtpPolicy,
    planner: DepthPlanner,
    lookup: Option<PromptLookup>,
    stats: MtpScheduleStats,
    /// Test-only head depths planned before the policy, capped per round.
    #[cfg(test)]
    script: std::collections::VecDeque<usize>,
}

impl RoundScheduler {
    pub fn new(
        policy: MtpPolicy,
        costs: RoundCosts,
        limit: usize,
        lookup: Option<PromptLookup>,
    ) -> Self {
        let stats = MtpScheduleStats {
            policy,
            lookup: lookup.is_some(),
            limit,
            ..MtpScheduleStats::default()
        };
        Self {
            policy,
            planner: DepthPlanner::new(costs, limit),
            lookup,
            stats,
            #[cfg(test)]
            script: std::collections::VecDeque::new(),
        }
    }

    /// Records the identifier of the cost table `costs` came from. Reported
    /// only when the adaptive policy plans with it.
    pub fn with_cost_table(
        mut self,
        id: &'static str,
    ) -> Self {
        self.stats.cost_table = (self.policy == MtpPolicy::Adaptive).then_some(id);
        self
    }

    /// An adaptive scheduler whose first rounds run the given head depths
    /// (each capped like any plan), for exercising round sequences such as a
    /// zero-draft round between draft rounds.
    #[cfg(test)]
    pub(crate) fn scripted(
        costs: RoundCosts,
        limit: usize,
        depths: impl IntoIterator<Item = usize>,
    ) -> Self {
        let mut scheduler = Self::new(MtpPolicy::Adaptive, costs, limit, None);
        scheduler.script = depths.into_iter().collect();
        scheduler
    }

    /// The plan for a round that may verify at most `cap` proposals.
    pub fn plan(
        &mut self,
        cap: usize,
    ) -> RoundPlan {
        #[cfg(test)]
        if let Some(depth) = self.script.pop_front() {
            return RoundPlan::Head(depth.min(cap));
        }
        let head = match self.policy {
            MtpPolicy::Fixed => cap,
            MtpPolicy::Adaptive => self.planner.head_depth(cap),
        };
        if cap == 0 {
            return RoundPlan::Head(head);
        }
        let Some(lookup) = &self.lookup else {
            return RoundPlan::Head(head);
        };
        let mut proposal = lookup.propose(cap);
        if proposal.is_empty() {
            return RoundPlan::Head(head);
        }
        if self.policy == MtpPolicy::Adaptive {
            match self.planner.lookup_depth(proposal.len(), head) {
                Some(drafts) => proposal.truncate(drafts),
                None => return RoundPlan::Head(head),
            }
        }
        RoundPlan::Lookup(proposal)
    }

    /// Records a finished round and indexes its committed tokens.
    pub fn observe(
        &mut self,
        plan: &RoundPlan,
        accepted: usize,
        committed: impl IntoIterator<Item = u32>,
        seconds: f64,
    ) {
        let proposed = plan.drafts();
        match plan {
            RoundPlan::Head(drafts) => {
                if let Some(slot) = self.stats.head_rounds_by_depth.get_mut(*drafts) {
                    *slot += 1;
                    self.stats.head_round_seconds_by_depth[*drafts] += seconds;
                }
            },
            RoundPlan::Lookup(_) => {
                self.stats.lookup_round_seconds += seconds;
                self.stats.lookup_rounds += 1;
                self.stats.lookup_proposed += proposed;
                self.stats.lookup_accepted += accepted;
            },
        }
        self.planner.observe(matches!(plan, RoundPlan::Lookup(_)), proposed, accepted);
        if let Some(lookup) = &mut self.lookup {
            for token in committed {
                lookup.push(token);
            }
        }
    }

    pub fn record_head_advance(
        &mut self,
        seconds: f64,
    ) {
        self.stats.head_advances += 1;
        self.stats.head_advance_seconds += seconds;
    }

    pub fn stats(&self) -> MtpScheduleStats {
        self.stats
    }
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    const GREEDY: RoundCosts = CostTable::LILY_Q4_XL_LILYMERGE_01.greedy;
    const SAMPLED: RoundCosts = CostTable::LILY_Q4_XL_LILYMERGE_01.sampled;
    const FIRST_SERIAL_LIMIT: usize = 8;
    const PROSE_SETTLE_LIMIT: usize = 6;
    const PROSE_ONE_DRAFT_ROUNDS: usize = 150;

    /// Merged-build measurements (attempt `20261119-LILYMERGE-01`): head
    /// rounds at 1/2/3 drafts, the zero-draft round inside a stream, and
    /// lookup rounds of 2 and 3 proposals (greedy) and 2 (sampled).
    #[stark_test]
    fn cost_table_reproduces_the_measured_rounds() {
        let table: [(RoundCosts, [f64; 3], f64, &[(usize, f64)]); 2] = [
            (GREEDY, [19.92, 26.17, 32.96], 13.75, &[(2, 24.71), (3, 30.50)]),
            (SAMPLED, [19.72, 26.05, 33.17], 13.61, &[(2, 24.54)]),
        ];
        for (costs, measured, serial, lookup) in table {
            for (drafts, round) in measured.into_iter().enumerate() {
                assert!((costs.head_round_ms(drafts + 1) - round).abs() < 0.05, "{drafts} {round}");
            }
            assert_eq!(costs.head_round_ms(0), serial);
            for &(drafts, round) in lookup {
                assert!((costs.lookup_round_ms(drafts) - round).abs() < 0.05, "{drafts} {round}");
            }
            assert!(costs.lookup_round_ms(3) < costs.head_round_ms(3));
            // Past the table, every phase grows by its last measured slope.
            let slope = costs.head_round_ms(5) - costs.head_round_ms(4);
            assert!(slope > costs.verify_extra_row_ms && slope < costs.verify_extra_row_ms + 3.0);
        }
        // Greedy drafts chain on the device: each extra draft is cheaper.
        assert!(GREEDY.draft_extra_ms < SAMPLED.draft_extra_ms);
    }

    #[stark_test]
    fn expected_tokens_is_one_plus_prefix_products() {
        let cases: &[(&[f64], usize, f64)] = &[
            (&[0.5, 0.5, 0.5], 0, 1.0),
            (&[0.5, 0.5, 0.5], 1, 1.5),
            (&[0.5, 0.5, 0.5], 3, 1.875),
            (&[1.0, 0.0, 1.0], 3, 2.0),
            (&[0.9, 0.8], 4, 1.0 + 0.9 + 0.72),
        ];
        for &(acceptance, drafts, expected) in cases {
            assert!((expected_tokens(acceptance, drafts) - expected).abs() < 1e-12);
        }
    }

    /// Steady per-position acceptance derived from the merged build's tokens
    /// per round pick the depth that measured fastest: the frozen corpus
    /// (greedy 2, 1.2 percent ahead of 3; sampled 1 or 2 within one percent),
    /// the fixed-corpus code arm (3), and the 4k prose inputs (1 draft, which
    /// measured above MTP off).
    #[stark_test]
    fn planner_matches_the_measured_best_depth() {
        let table: &[(RoundCosts, [f64; 3], &[usize])] = &[
            (GREEDY, [0.932, 0.907, 0.80], &[2]),
            (SAMPLED, [0.740, 0.726, 0.37], &[1, 2]),
            (GREEDY, [0.977, 0.945, 0.98], &[3]),
            (GREEDY, [0.67, 0.51, 0.09], &[1]),
            (GREEDY, [0.58, 0.50, 0.10], &[1]),
            (GREEDY, [0.20, 0.20, 0.20], &[0]),
            (SAMPLED, [0.20, 0.20, 0.20], &[0]),
            (SAMPLED, [1.0, 1.0, 1.0], &[3]),
        ];
        for (costs, acceptance, expected) in table {
            let mut planner = DepthPlanner::new(*costs, 3);
            planner.head.acceptance = acceptance.to_vec();
            assert!(expected.contains(&planner.head_depth(3)), "{acceptance:?}");
        }
    }

    #[stark_test]
    fn prior_opens_at_two_drafts_and_respects_the_cap() {
        for costs in [GREEDY, SAMPLED] {
            let mut planner = DepthPlanner::new(costs, 3);
            assert_eq!(planner.head_depth(3), 2);
            assert_eq!(planner.head_depth(1), 1);
            assert_eq!(planner.head_depth(0), 0);
            // A capped round does not replace the incumbent.
            assert_eq!(planner.incumbent, Some(2));
        }
    }

    #[stark_test]
    fn hysteresis_keeps_the_incumbent_until_a_challenger_is_two_percent_better() {
        let mut planner = DepthPlanner::new(GREEDY, 3);
        planner.head.acceptance = vec![0.932, 0.907, 0.80];
        assert_eq!(planner.head_depth(3), 2);
        // Depth 3 slightly ahead of depth 2: within the band, keep 2.
        planner.head.acceptance = vec![0.95, 0.95, 0.86];
        assert!(planner.head_rate(3) > planner.head_rate(2));
        assert!(planner.head_rate(3) <= planner.head_rate(2) * (1.0 + HYSTERESIS));
        assert_eq!(planner.head_depth(3), 2);
        // Code-like acceptance: depth 3 is clearly ahead, switch.
        planner.head.acceptance = vec![0.98, 0.97, 0.98];
        assert_eq!(planner.head_depth(3), 3);
        // Depth 1 clearly ahead: switch.
        planner.head.acceptance = vec![0.75, 0.1, 0.1];
        assert_eq!(planner.head_depth(3), 1);
        assert_eq!(planner.incumbent, Some(1));
    }

    #[stark_test]
    fn observations_weigh_the_prior_as_two_and_unseen_positions_follow_their_predecessor() {
        let mut planner = DepthPlanner::new(SAMPLED, 3);
        planner.observe(false, 3, 1);
        let p = ACCEPTANCE_PRIOR;
        let d = ACCEPTANCE_DECAY;
        let f = ACCEPTANCE_FOLLOW;
        // First observation of a position: the mean of the prior counted
        // twice and the outcome.
        let first = p + (1.0 / 3.0) * (1.0 - p);
        let second = p + (1.0 / 3.0) * (0.0 - p);
        assert_eq!(planner.head.acceptance[0], first);
        assert_eq!(planner.head.acceptance[1], second);
        // Position 2 was never verified after the rejection: it drifts
        // toward position 1.
        assert_eq!(planner.head.acceptance[2], p + f * (second - p));
        assert_eq!(planner.head.observed, vec![1, 1, 0]);
        assert_eq!(planner.lookup.acceptance, vec![LOOKUP_PRIOR; 3]);
        // The weight falls to 1/16 from the fourteenth observation on.
        let mut planner = DepthPlanner::new(SAMPLED, 3);
        let mut expected = p;
        for n in 1..=16u32 {
            planner.observe(false, 1, 0);
            let weight = (1.0 / (f64::from(n) + PRIOR_OBSERVATIONS)).max(ACCEPTANCE_WEIGHT);
            expected += weight * (0.0 - expected);
            assert_eq!(planner.head.acceptance[0], expected);
        }
        assert_eq!(1.0 / (14.0 + PRIOR_OBSERVATIONS), ACCEPTANCE_WEIGHT);
        for _ in 0..64 {
            planner.observe(false, 1, 0);
        }
        assert!(planner.head.acceptance[0] < 0.01);
        // Position 1 is never verified and follows position 0 down.
        assert!(planner.head.acceptance[1] < 0.5);
        let low = planner.head.acceptance.clone();
        planner.observe(true, 2, 2);
        let head0 = low[0] + d * (p - low[0]);
        assert_eq!(planner.head.acceptance[0], head0);
        assert_eq!(planner.head.acceptance[1], low[1] + f * (head0 - low[1]));
        let q = LOOKUP_PRIOR;
        assert!(planner.lookup.acceptance[0] > q && planner.lookup.acceptance[1] > q);
        assert_eq!(planner.lookup.acceptance[2], q + f * (planner.lookup.acceptance[1] - q));
    }

    #[stark_test]
    fn adaptive_falls_back_and_recovers_deterministically() {
        let run = || {
            let mut planner = DepthPlanner::new(SAMPLED, 3);
            let mut depths = Vec::new();
            for round in 0..300 {
                let depth = planner.head_depth(3);
                depths.push(depth);
                // A hard stretch, then an easy one.
                let accepted = if (60..160).contains(&round) {
                    0
                } else {
                    depth
                };
                planner.observe(false, depth, accepted);
            }
            depths
        };
        let depths = run();
        assert_eq!(depths, run());
        assert_eq!(&depths[..60].iter().max(), &Some(&3), "{:?}", &depths[..60]);
        assert!(depths[150] <= 1, "{:?}", &depths[140..160]);
        // Back at three drafts once acceptance recovers (4.7 percent ahead
        // of two at full acceptance, outside the band).
        assert_eq!(depths[299], 3, "{:?}", &depths[280..]);
    }

    /// Bernoulli acceptance draws from a fixed LCG, so a run is a pure
    /// function of its inputs. Returns the planned depth of every round.
    fn simulate(
        costs: RoundCosts,
        rounds: usize,
        rate: impl Fn(usize, usize) -> f64,
    ) -> Vec<usize> {
        let mut planner = DepthPlanner::new(costs, 3);
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut uniform = || {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut depths = Vec::new();
        for round in 0..rounds {
            let depth = planner.head_depth(3);
            let accepted = (0..depth).take_while(|&position| uniform() < rate(round, position)).count();
            planner.observe(false, depth, accepted);
            depths.push(depth);
        }
        depths
    }

    /// Low-acceptance text: per-position acceptance 0.3 for 150 rounds, then
    /// 0.95. The plan reaches serial steps within a few rounds, stays there
    /// apart from occasional probes, and returns to drafting once acceptance
    /// comes back.
    #[stark_test]
    fn low_acceptance_settles_at_serial_quickly_and_recovers() {
        for costs in [GREEDY, SAMPLED] {
            let depths = simulate(costs, 400, |round, _| {
                if round < 150 {
                    0.3
                } else {
                    0.95
                }
            });
            let first_serial = depths.iter().position(|&depth| depth == 0).unwrap();
            assert!(first_serial <= FIRST_SERIAL_LIMIT, "{:?}", &depths[..20]);
            let serial = depths[20..150].iter().filter(|&&depth| depth == 0).count();
            assert!(serial >= 100, "{serial} {:?}", &depths[20..150]);
            let drafting = depths[250..].iter().filter(|&&depth| depth >= 2).count();
            assert!(drafting >= 140, "{drafting} {:?}", &depths[150..250]);
        }
    }

    /// The 4k prose arm's per-position acceptance (0.67, 0.51, 0.09 from
    /// fixed one, two and three drafts): the request opens at two drafts and
    /// leaves them within a few rounds for one draft, the measured best.
    /// With the fixed 1/8 weight of attempt `20261119-LILYDEPTH-01` the
    /// opening took about 21 rounds on the real prose input.
    #[stark_test]
    fn prose_acceptance_leaves_the_opening_depth_within_a_few_rounds() {
        for costs in [GREEDY, SAMPLED] {
            let depths = simulate(costs, 200, |_, position| [0.67, 0.51, 0.09][position]);
            assert_eq!(depths[0], 2);
            let settled = depths.iter().position(|&depth| depth < 2).unwrap();
            assert!(settled <= PROSE_SETTLE_LIMIT, "{:?}", &depths[..20]);
            let one = depths[10..].iter().filter(|&&depth| depth == 1).count();
            assert!(one >= PROSE_ONE_DRAFT_ROUNDS, "{one} {:?}", &depths[10..60]);
        }
    }

    #[stark_test]
    fn lookup_is_preferred_only_when_it_beats_the_head_plan() {
        for costs in [GREEDY, SAMPLED] {
            let mut planner = DepthPlanner::new(costs, 3);
            let head = planner.head_depth(3);
            // Higher prior, no draft phase: lookup wins at its longest
            // proposal.
            assert_eq!(planner.lookup_depth(3, head), Some(3));
            assert_eq!(planner.lookup_depth(0, head), None);
            // An untried lookup still beats a three-draft plan on code, but
            // a one-token proposal does not.
            planner.head.acceptance = vec![0.98, 0.97, 0.98];
            assert_eq!(planner.head_depth(3), 3);
            assert_eq!(planner.lookup_depth(3, 3), Some(3));
            assert_eq!(planner.lookup_depth(1, 3), None);
            // One rejected lookup round at the first position is enough to
            // fall back to the head plan.
            planner.observe(true, 3, 0);
            assert_eq!(planner.lookup_depth(3, 3), None);
            planner.lookup.acceptance = vec![0.1, 0.1, 0.1];
            assert_eq!(planner.lookup_depth(3, head), None);
        }
    }

    #[stark_test]
    fn schedule_counts_rounds_and_serializes_up_to_the_limit() {
        let mut scheduler = RoundScheduler::new(MtpPolicy::Adaptive, GREEDY, 3, None);
        assert_eq!(scheduler.plan(3), RoundPlan::Head(2));
        scheduler.observe(&RoundPlan::Head(2), 2, [1, 2, 3], 0.03);
        scheduler.observe(&RoundPlan::Head(0), 0, [4], 0.02);
        scheduler.record_head_advance(0.5);
        let stats = scheduler.stats();
        assert_eq!(stats.head_rounds_by_depth[..4], [1, 0, 1, 0]);
        let value = serde_json::to_value(stats).unwrap();
        assert_eq!(value["policy"], "adaptive");
        assert_eq!(value["head_rounds_by_depth"], serde_json::json!([1, 0, 1, 0]));
        assert_eq!(value["head_advances"], 1);
        assert!(value.get("cost_table").is_none());
        let mut fixed = RoundScheduler::new(MtpPolicy::Fixed, SAMPLED, 3, None);
        assert_eq!(fixed.plan(3), RoundPlan::Head(3));
        assert_eq!(fixed.plan(1), RoundPlan::Head(1));
    }

    #[stark_test]
    fn adaptive_schedule_reports_its_cost_table_and_fixed_does_not() {
        let table = CostTable::LILY_Q4_XL_LILYMERGE_01;
        assert_eq!(table.id, "lily-q4-xl-lilymerge-01");
        let adaptive = RoundScheduler::new(MtpPolicy::Adaptive, table.greedy, 2, None).with_cost_table(table.id);
        let value = serde_json::to_value(adaptive.stats()).unwrap();
        assert_eq!(value["cost_table"], "lily-q4-xl-lilymerge-01");
        assert_eq!(value.as_object().unwrap().len(), 12);
        let fixed = RoundScheduler::new(MtpPolicy::Fixed, table.sampled, 2, None).with_cost_table(table.id);
        assert_eq!(fixed.stats().cost_table, None);
        let value = serde_json::to_value(fixed.stats()).unwrap();
        assert!(value.get("cost_table").is_none());
        assert_eq!(value.as_object().unwrap().len(), 11);
    }
}
