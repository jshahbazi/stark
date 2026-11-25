# Can DFlash 2 and DSpark correct each other on Inkling 2 Mini?

Attempt `20261125-INKDRAFT-01`, November 25, 2026. This offline greedy replay measures whether an independently trained drafter covers the primary drafter's first mistake. It also prices a specific use of that candidate. The candidate occupies one existing verifier slot, with no continuation beneath the alternate. The comparison tests cross-drafter coverage. A trained reconsideration head or a DFlash selector runner-up would produce different candidates.

## Fixtures and controls

The four original-checkpoint traces have 192 output tokens each. Inputs comprise one novel-prose prompt and three coding prompts (initial, sustained, and 4K context). The target uses the original NVFP4/MXFP8 weights, native DSpark uses Stark's INT8 proposal head, and DFlash 2 uses BF16 with F32 selector corrections. The [matched trace record](2026-11-24-inkling-mini-dflash2-traces.md) identifies the checkpoints, binaries, native target hash controls and reference implementations. The cost inputs are that record's November 24 measurements on a 40-GPU-core, 128 GiB M5 Max and the separate [DFlash worker width screen](../benchmarks/2026-11-24-inkling-mini-live-dflash2.md#worker-and-width-selection).

The analysis starts at Stark revision `a83752f22aec2139769ab8efae0d6b7729732dd1`. It checks both private fixture hashes and the teacher output hash for each request. All 16 baseline combinations reproduce their archived round, first-rejection, singleton and transaction counts. A second implementation built the candidate tree explicitly and reproduced round counts, branches offered and taken, and physical packet sizes for all 32 baseline and branch walks. The experiment changes no worker or model weights. It adds no new numerical qualification, live timing or installed runtime.

## Who catches the first miss?

A baseline round starts from a known target token and walks the primary draft until rejection, exhaustion or the output cap. A catch requires the other drafter, from that same anchor, to match the target on **every** proposal through the first miss. Fully accepted rounds and rounds with no proposal contribute zero misses. Each primary and width walks its own anchors. Coding totals pool counts across its three prompts.

| Baseline first misses caught by the other drafter | Prose, 3 proposals | Coding, 3 | Prose, 7 | Coding, 7 |
| --- | ---: | ---: | ---: | ---: |
| DFlash 2 catches DSpark | 39/79 (49.4%) | 72/149 (48.3%) | 46/90 (51.1%) | 80/186 (43.0%) |
| DSpark catches DFlash 2 | 7/47 (14.9%) | 22/102 (21.6%) | 6/56 (10.7%) | 22/134 (16.4%) |

## Fixed-budget branch and M5 Max break-even

The policy compares the two proposed paths before target verification. At their first disagreement it keeps the primary path and offers the other token as a sibling, dropping the last primary proposal if necessary. A disagreement in the last physical slot cannot form a fork. Near the output cap, unused slots can hold the sibling. The primary gets the first slot at the disagreement and the sibling follows it immediately. After each verification the next anchor comes from this policy's own walk. Choosing the branch point uses no teacher output. The gate in this replay can see both completed drafter paths.

Each table row reports the median **modeled verifier-plus-commit time saved** across six historical timing blocks, divided by the number of charged events. A fork attempt pays even if its alternate is rejected. A successful correction is a fork where the verifier takes the alternate. If producing the alternate requires a second drafter call before checking for disagreement, the relevant denominator is all nonterminal draft rounds. Negative values mean the predicted target work alone increased.

| Primary ← alternate | Workload | Slots | Rounds, base → fork | Attempts / taken | ms per attempt | ms per taken | ms per draft round |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DSpark ← DFlash 2 | Prose | 3 | 96 → 80 | 52 / 25 | **8.09** | 16.83 | **5.26** |
| DSpark ← DFlash 2 | Coding | 3 | 224 → 206 | 103 / 48 | **4.32** | 9.27 | **2.17** |
| DSpark ← DFlash 2 | Prose | 7 | 91 → 74 | 73 / 35 | 3.97 | 8.27 | 3.91 |
| DSpark ← DFlash 2 | Coding | 7 | 197 → 173 | 158 / 69 | 0.89 | 2.03 | 0.81 |
| DFlash 2 ← DSpark | Prose | 3 | 73 → 75 | 49 / 7 | −1.01 | −7.06 | −0.66 |
| DFlash 2 ← DSpark | Coding | 3 | 198 → 199 | 99 / 22 | −0.23 | −1.02 | −0.11 |
| DFlash 2 ← DSpark | Prose | 7 | 62 → 60 | 59 / 7 | −1.23 | −10.33 | −1.21 |
| DFlash 2 ← DSpark | Coding | 7 | 157 → 154 | 145 / 24 | −1.25 | −7.57 | −1.19 |

At seven slots the 4K coding trace matters. Adding DFlash to DSpark saved nine rounds on initial coding, ten on sustained coding, and five at 4K. Per attempt, the target-time budgets were **2.19, 1.62 and −1.99 ms** respectively. At 4K the policy increased physical calls from 69 to 72. The same three traces at three slots gave **4.26, 4.27 and 4.50 ms**. On prose, the observed warm DFlash draft-plus-IPC call took 8.71 ms. To cover a call that expensive on every three-slot prose draft round, each saved DSpark draft round would also need to avoid about 17.2 ms of ordinary non-verifier work, before any extra branch-state costs.

### How to use the budgets

1. Pack the anchor and candidate nodes in topological order into packets of up to four rows. Charge the second packet only when it contains a matched node the target must consume. Price the predicted packet histogram with the same case's one- through four-row medians. Scale that cost so the baseline reproduces the measured verifier total for its own arm, width and timing block. Price commit separately at that baseline's mean per multirow transaction. The native singleton already includes its state update. This assumes sibling rows and branched state commits cost the same as ordinary chain rows and commits of the same size.
2. Subtract the branch estimate from the measured baseline in each block. For the coding total sum signed savings within each block before taking the median and dividing by the total attempts. Holding the price table unscaled gives the same sign in all eight grouped comparisons. At DSpark width three, this sensitivity gives 7.47 ms/attempt on prose and 4.29 on coding.
3. For a complete runtime budget, add the ordinary non-verifier work avoided in the saved draft rounds, then deduct the actual cost of producing, selecting, verifying and committing the sibling beyond the modeled chain-equivalent work. For DSpark at width three, each millisecond of ordinary non-verifier work per saved draft round adds **0.308 ms per attempted fork on prose**. On coding the three cases save 7, 8 and 3 draft rounds respectively. A common one-millisecond cost adds **0.175 ms per fork** to that pooled budget. The fixed-width DSpark full-worker clock was unavailable on these traces.
4. For DFlash-primary only, the [aggregate receipt](../benchmarks/data/2026-11-25-inkling-first-miss.json) also gives a worker estimate. For each trace and width, subtract verifier and commit time from the mean of the two original worker decode times and spread the residual over nonterminal draft rounds. Adding the resulting savings leaves **−0.87 ms/attempt on prose and −1.06 on coding** at seven slots. These are separate clocks and the worker and replay have small end-of-budget accounting differences. This estimate does not time the proposed worker.

**Disposition.** The strongest candidate on these fixtures is a low-cost corrector for a three-proposal DSpark primary. The reverse direction has negative pooled headroom at both widths. The four prompts and greedy sampling limit the inference. A larger corpus, a real gate able to act without running both drafters, and a branch-capable verifier with measured state and timing would be needed for a runtime decision.

## Reproduce

Run `uv run --no-project python docs/archive/experiments/2026-11-24-inkling-dflash2/first_miss.py runtime/dflash2/traces` from the worktree holding the private fixtures. The [evaluator](../archive/experiments/2026-11-24-inkling-dflash2/first_miss.py) returns aggregate counters and cost estimates. The [receipt](../benchmarks/data/2026-11-25-inkling-first-miss.json) binds the matched-trace and live-timing source receipt hashes and includes per-case packet counts and timing-block savings.
