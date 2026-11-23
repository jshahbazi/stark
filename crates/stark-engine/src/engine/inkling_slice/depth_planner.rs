// SPDX-License-Identifier: Apache-2.0
//! Per-round draft depth for the Inkling verifier.
//!
//! The planner keeps running estimates of the conditional acceptance of each
//! draft position and of the wall time of a serial step and of a round at each
//! depth, both measured on the running request. Each round it picks the depth
//! (0 to 3) that maximizes expected tokens per millisecond. Output does not
//! depend on the depth: every depth commits exactly the target's tokens, so
//! this is purely a speed policy.

/// Maximum draft depth the verifier supports.
pub const MAX_DEPTH: usize = 3;
const ACCEPT_WEIGHT: f64 = 0.03;
const COST_WEIGHT: f64 = 0.1;
/// A depth change needs this relative gain in expected tokens per millisecond.
const HYSTERESIS: f64 = 1.03;
/// Every this many rounds the planner proposes one draft more than its
/// choice, so the estimates for later positions stay current.
const PROBE_PERIOD: usize = 16;

#[derive(Clone, Debug)]
pub struct DepthPlanner {
    /// P(position k accepted | positions before k accepted).
    accept: [f64; MAX_DEPTH],
    /// Wall milliseconds of a serial step (index 0) and of a round per depth.
    cost: [f64; MAX_DEPTH + 1],
    rounds: usize,
    current: usize,
    floor: usize,
}

impl Default for DepthPlanner {
    fn default() -> Self {
        Self::new()
    }
}

impl DepthPlanner {
    pub fn new() -> Self {
        Self {
            accept: [0.7, 0.6, 0.6],
            // Relative starting costs; replaced by measurements within a few
            // rounds of each depth.
            cost: [15.0, 20.0, 23.0, 26.0],
            rounds: 0,
            current: MAX_DEPTH,
            floor: 1,
        }
    }

    /// A planner that may also choose plain steps (depth 0).
    pub fn with_plain_steps() -> Self {
        Self {
            floor: 0,
            ..Self::new()
        }
    }

    /// Expected committed tokens of a round at `depth` (including the
    /// target's own token).
    fn expected_tokens(
        &self,
        depth: usize,
    ) -> f64 {
        let mut total = 1.0;
        let mut reach = 1.0;
        for &p in self.accept.iter().take(depth) {
            reach *= p;
            total += reach;
        }
        total
    }

    /// Depth for the next round, capped by `limit`.
    pub fn choose(
        &mut self,
        limit: usize,
    ) -> usize {
        let limit = limit.min(MAX_DEPTH);
        self.rounds += 1;
        let rate = |depth: usize| self.expected_tokens(depth) / self.cost[depth].max(1e-3);
        // At least one draft whenever drafting is allowed: a serial step never
        // won by enough in the recorded traces to justify losing the probe.
        let floor = limit.min(self.floor);
        let best = (floor..=limit).max_by(|&a, &b| rate(a).total_cmp(&rate(b)).then(a.cmp(&b))).unwrap_or(floor);
        let current = self.current.clamp(floor, limit);
        if rate(best) > HYSTERESIS * rate(current) {
            self.current = best;
        }
        let chosen = self.current.clamp(floor, limit);
        if self.rounds % PROBE_PERIOD == 0 {
            return (chosen + 1).min(limit);
        }
        chosen
    }

    /// Record a finished round: its depth, how many proposals were accepted,
    /// and its wall time.
    pub fn record(
        &mut self,
        depth: usize,
        accepted: usize,
        wall_ms: f64,
    ) {
        let depth = depth.min(MAX_DEPTH);
        if wall_ms.is_finite() && wall_ms > 0.0 {
            self.cost[depth] += COST_WEIGHT * (wall_ms - self.cost[depth]);
        }
        for position in 0..depth {
            if position > accepted {
                break;
            }
            let hit = if position < accepted {
                1.0
            } else {
                0.0
            };
            self.accept[position] += ACCEPT_WEIGHT * (hit - self.accept[position]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn low_acceptance_prefers_short_rounds() {
        let mut planner = DepthPlanner::new();
        for _ in 0..200 {
            planner.record(3, 0, 26.0);
            planner.record(1, 0, 20.0);
            planner.record(0, 0, 15.0);
        }
        assert_eq!(planner.choose(3), 1);
        assert_eq!(planner.choose(0), 0);
    }

    #[test]
    fn high_acceptance_prefers_full_depth() {
        let mut planner = DepthPlanner::new();
        for _ in 0..40 {
            planner.record(3, 3, 26.0);
        }
        assert_eq!(planner.choose(3), 3);
    }
}
