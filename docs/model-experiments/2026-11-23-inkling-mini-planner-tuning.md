# Inkling 2 Mini draft-depth planner: tuning and the hold rule

Follow-ups to the history planner of [INKROUND-06](2026-11-23-inkling-mini-decode-round.md#20261123-inkround-06--depth-policy-history-planner-confidence-cut-and-backoff), which became the worker default in `2a1d72a2` (floor 1, capped by the requested MTP count). Same host as the other 2026-11-23 records (128 GiB M5 Max, macOS 26.7.2, 94 W adapter, shared lock). Output is identical at every depth by construction and was checked identical in every worker comparison below; the planner changes speed only. Results are on the Stark 8-bit artifact unless the original is named. Nothing is released or installed.

## Worker evidence for the planner as first committed (not interleaved)

From the decode workstream's gate batch, worker decode tok/s with the first planner settings against the fixed-depth-3 worker measured earlier the same morning, not interleaved and with a different build, so indicative only: 8-bit prose 58.5 → 75.0, explain 74.6 → 82.5, code about 101 → 101.5; original prose 58.3 → 62.8, explain 69.2 → 72.2, code 88.2 → 85.0.

## 20261123-INKROUND-11 · Start near full depth and adapt faster

| Field | Value |
| --- | --- |
| Attempt ID / date | `20261123-INKROUND-11` / 2026-11-23 |
| Kind | Speculation policy tuning |
| Result | Adopted: within 1.5 percent of the best fixed depth on all 14 trace replays; worker ABBA mean 1.007 against fixed depth 3, with losses of 0.5 to 2.4 percent on code and long-context inputs that do not pass a strict no-loss rule |
| Disposition | Committed as `1206760e` (2026-11-23) by the coordinator for the prose gain, knowing the 8K and code losses |

**Change** (`depth_planner.rs` only): prior acceptance per position 0.8 / 0.75 / 0.75 (was 0.7 / 0.6 / 0.6); acceptance weight (EMA) 0.05 (was 0.03); probe one draft deeper every 32 rounds (was 16). Hysteresis and the one-draft floor unchanged.

**Why.** The long-context loss of the first settings came from the start, not the steady state: with the low prior the planner began at depth 2 and its slow estimate took most of a 256-token window to return to 3.

**Trace replay** (all 14 `trace3` runs, 7 inputs × 2 artifacts, planner rate over best fixed depth on the same trace and costs): 8-bit c4000 0.993, c8000 1.000, p4000 1.000, s73 0.997, code 1.000, explain 0.985, prose 0.996; original c4000 0.996, c8000 0.992, p4000 1.008, s73 0.999, code 0.986, explain 1.006, prose 1.009. Worst 0.985, mean 0.998; the first settings had worst 0.968 (original c4000). Against fixed depth 3 it keeps about +8 percent on 8-bit prose and +3 percent on explain.

**Worker ABBA** (8-bit artifact; fixed depth 3 against the tuned planner, same build, the only difference `planner.choose(limit)` against `limit`; two runs per arm per input; receipts `runtime/inkling-affine8/runs/abbb1` to `abbb4` (`-worker-fixed-3` and `-worker-planner-3` result files; `abbb1-worker-fixed-3.result.json` SHA-256 prefix `e0aecc68dd7ac54a`)):

| Input | Fixed tok/s | Planner tok/s | Planner / fixed | Planner rounds at depth 0 / 1 / 2 / 3 | Accepted / proposed |
| --- | --- | --- | ---: | --- | --- |
| c4000 | 131.8 / 131.2 | 129.9 / 130.8 | 0.991 | 0 / 0 / 0 / 222 | 545 / 666 |
| c8000 | 119.8 / 117.4 | 116.4 / 115.2 | 0.976 | 0 / 0 / 21 / 203 | 543 / 651 |
| chat-code | 97.7 / 96.3 | 96.2 / 95.1 | 0.986 | 1 / 10 / 88 / 205 | 463 / 801 |
| chat-explain | 80.6 / 79.7 | 80.3 / 81.3 | 1.008 | 0 / 105 / 34 / 173 | 306 / 692 |
| chat-prose | 68.1 / 67.8 | 72.7 / 75.2 | 1.088 | 0 / 124 / 103 / 53 | 176 / 489 |
| p4000 | 105.3 / 104.6 | 104.6 / 104.3 | 0.995 | 0 / 2 / 70 / 185 | 510 / 697 |

Mean ratio 1.007. At 4K the lead is held (c4000 at full depth every round). At 8K the planner gives up 2.4 percent through 21 rounds at depth 2 despite 83 percent acceptance; the two fixed runs there differ by 2 percent, so this is at the edge of noise. Code loses 1.4 percent through 88 rounds at depth 2. All four runs of each input produced identical output. During the build of these workers, a killed run left `crates/stark-generate/src/inkling.rs` with fixed depth for about three minutes around 10:24 to 10:27; it was restored and checked with `cmp`. The decode workstream flagged that anything built or committed from that file in that window would have run at fixed depth; `1206760e` was committed at 10:31, after the restore, and the committed tree was rebuilt and gated later (commit-check-3).

## 20261123-INKROUND-12 · Hold full depth while acceptance is high

| Field | Value |
| --- | --- |
| Attempt ID / date | `20261123-INKROUND-12` / 2026-11-23 |
| Kind | Speculation policy |
| Result | Adopted on trace-replay evidence only: removes the 4K and 8K loss in replays with measured-cost noise on both artifacts, leaves prose unchanged, does not fix code. Not measured at worker level. |
| Disposition | Committed in `4314ac51` (2026-11-23) |

**Diagnosis.** A noise-free trace replay of the tuned planner showed no loss on 8-bit c8000 or code (1.000), so it did not reproduce the worker. Adding 10 percent Gaussian noise to the round walls the planner records (40 seeds) reproduced the worker's pattern (8-bit c4000 0.982, c8000 0.996, code 0.984): measured-cost noise talks the planner down from depth 3. Replay script: the decode workstream's `planner_sim.py`, which reads only aggregate receipts (not in the repository).

**Variants, 8-bit artifact, 10 percent noise, planner over fixed depth 3:**

| Variant | c4000 | c8000 | code | explain | prose | p4000 | s73 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Committed tuned (INKROUND-11) | 0.982 | 0.996 | 0.984 | 1.017 | 1.074 | 0.991 | 0.991 |
| Hold full depth while every per-position estimate ≥ 0.70 | 1.000 | 1.000 | 0.984 | 1.016 | 1.074 | 0.993 | 0.991 |
| Hold 0.70 plus hysteresis 1.08 | 1.000 | 1.000 | 0.999 | 1.016 | 1.079 | 1.000 | 0.983 |
| Hold 0.70 plus hysteresis 1.10 | 1.000 | 1.000 | 1.000 | 0.999 | 1.078 | 1.000 | 0.991 |

**Rejected variants and why.** Hold 0.70 with hysteresis 1.08: without noise 8-bit explain falls to 0.952 and original s73 to 0.957 to 0.968. Hold 0.70 with hysteresis 1.10: original s73 0.956 to 0.966 and original prose from 1.056 to about 1.03 to 1.04. Hysteresis alone at 1.10 to 1.15: up to 10 percent lost on original c4000. Holding at 0.75 or 0.78: no better than 0.70. No variant removed both the long-context and the code losses without side effects.

**Adopted.** `HOLD_ACCEPTANCE = 0.70` in `depth_planner.rs`: after the hysteresis update, `choose` holds full depth while every per-position acceptance estimate is at least 0.70; new unit test `high_acceptance_holds_full_depth_despite_noisy_costs`. On the original artifact the hold takes c4000 from 0.979 to 1.000 and c8000 from 0.996 to 1.000 in the noisy replay; prose and explain unchanged; code stays at 0.984.

**Gates and what was not run.** The committed tree `4314ac51` passed the coordinator's commit-check gates (`runtime/commit-check-3/`, summary SHA-256 prefix `4d038aef3e5d6fc0`; `inkling_slice` `b7678d423f04…`, `stark-generate` `a966b11554ae…`): clean build, 167 worker tests, exact reference exact on `g4096-02` and `s4096-00`, drafting on equals off at 73, 4,150, 8,800 and 16,000 on both artifacts, a repeated replay with identical hashes, the worker smoke on both artifacts. **The planner's unit tests did not run**: the gate run's `test-planner` step exited 101 because the command was built without the `cpu` feature the engine's tests require (`stark tests require cpu backend`), and the decode workstream's own unit-test job was cancelled while waiting for the lock. The commit body says the tests "were not run"; the receipts show an attempted run that failed to compile. No worker ABBA of the hold rule has been run.

## Open

- Planner unit tests (never run successfully).
- The code loss (about 1.4 percent at worker level, 0.984 in replay) has no fix without side effects.
- A worker-level comparison of the hold rule against fixed depth 3.
- The comparison with MLX-Serve's confidence cut and backoff, which lose on Stark's cost ratios, is in INKROUND-06 and on the [differences page](inkling-mlx-serve-differences.md).
