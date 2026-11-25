# Inkling 2 Mini DFlash 2 runner-up at three verifier slots

Attempt `20261125-INKDRAFT-02`, November 25, 2026. The [cross-drafter replay](2026-11-25-inkling-first-miss.md) motivated testing a second proposal from DFlash's own selector. This experiment reuses a computed shortlist and requires no training.

## Plan fixed before scoring

- Use the original Inkling 2 Mini NVFP4/MXFP8 target, a greedy native teacher, and the pinned BF16 DFlash 2 drafter with F32 selector corrections from the [matched trace study](2026-11-24-inkling-mini-dflash2-traces.md). Keep the existing three-proposal DFlash path as the baseline. Use the older traces only for implementation checks.
- Freeze 12 new short-context prose prompts and 12 new short-context coding prompts before reading candidate outcomes. Give each prompt up to 192 output tokens, keeping all natural stops. Score both domains separately. The holdout concerns this experiment's policy selection. The models' original training sets are unknown.
- Consider the first two proposal positions in order. At the first position where the corrected top-two logit gap is at most `ln(2)`, offer the conditional runner-up as a sibling. This threshold corresponds to top-one versus top-two odds of at most two to one. The ordinary path keeps up to two proposals whenever a sibling takes the third slot. On confident rounds retain all three. An alternate has no child beneath it. The gate has no access to target outputs. Also report an unconditional first-position fork as a control.
- Walk each policy from the beginning of each continuation. Count a round as the target anchor plus the longest verified path. Compare rounds for identical output tokens at the same three-slot limit, wins/ties/losses across prompts, and successes and displaced continuations. Price the replay with historical M5 Max packet timings as a diagnostic. A live latency claim requires timing a real branch-capable worker.
- Advance a candidate that saves rounds on each domain and leaves time for selection and branch-state handling. Record the result for both domains even when only one improves. Preserve the baseline implementation, holdouts and threshold throughout scoring.

## Held-out results

All 24 prompts reached 192 greedy tokens. The input lengths were 65–87 tokens for prose and 73–89 for coding. Neither domain included a prompt from the earlier four-trace study. Both policies consumed the same 4,608 target tokens. The conditional runner-up saved 19 rounds on prose and 10 on coding. Always offering the first-position runner-up increased the number of rounds in both groups.

| 12 prompts per group | Ordinary three-slot DFlash | Conditional runner-up | Fewer rounds | Paired wins / ties / losses | Always fork at position one |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prose | 904 | 885 | **2.10%** | 8 / 3 / 1 | 970 |
| Coding | 840 | 830 | **1.19%** | 8 / 2 / 2 | 933 |

Tokens per round rose from 2.55 to 2.60 on prose and 2.74 to 2.78 on coding. Resampling the 12 prompts within each domain as pairs 20,000 times gave descriptive 95% percentile intervals of 0.68–3.54% fewer rounds on prose and 0.25–2.05% on coding. These intervals describe variation across this authored short-context set.

On the ordinary baseline's own anchors, the gate and alternate could recover 103 of 594 first misses on prose and 68 of 502 on coding. On the candidate's actual walk, the verifier chose the alternate on 96 of 393 prose attempts and 57 of 298 coding attempts. Giving up the third primary node cost a correct token on 57 prose attempts and 54 coding attempts. These counts describe different walks. The resulting change in the next round's anchor also affects the final total.

## M5 Max timing and correctness

At three proposals the anchor and candidate nodes fit in a single four-row packet. Near the output cap the replay charges for the actual node count. The historical M5 Max target-only packet medians used here were 17.79, 19.24, 21.72 and 23.78 ms for one through four rows. Applied to the two walks' packet counts, the estimated target time saved was 422.29 ms across prose and 235.74 ms across coding. That pays **0.479 ms per nonterminal candidate round on prose and 0.286 ms on coding**. A real selector inspects margins before knowing whether to fork, so its cost belongs on all 881 prose and 824 coding draft rounds.

The selector itself was timed on the M5 Max. At offsets 16, 64 and 128 of each of four preselected prompts (01 and 07 in each domain), build the draft hidden state once. Time the projection, selector and host readback for each arm. The ordinary arm reads seven primary IDs. The candidate reads the identical IDs, two runners-up and two corrected-score gaps. Both arms force one grouped evaluation before readback. Three fresh processes each ran 16 interleaved pairs per anchor after warmup, reversing case, offset and arm order in the middle process. Each process gave every anchor equal weight. The median paired extra cost across the three processes was **0.022 ms on prose** (0.020–0.022) and **0.016 ms on coding** (0.003–0.022). This measures the evaluation code's selector and readback in isolation.

After that selector charge, the modeled target-time margin is approximately **0.46 ms/round on prose and 0.27 ms/round on coding**. This accounting excludes the savings in ordinary drafter work when a round disappears. It also assumes a sibling costs the same to verify as a chain node at equal width. Actual tree attention, branch-specific state selection and commit, transfer and streaming remain unpriced. The study establishes a round-count win. Live end-to-end latency for the candidate remains unmeasured.

- The archived native capture executable reproduced the earlier initial-coding fixture byte for byte. Across the four historical fixtures the new selector matched all 2,304 saved first-three proposals. At every fresh teacher anchor its ordinary path matched the frozen MLX selector across all seven positions, **32,256 ID checks**.
- An independent explicit tree traversal reproduced ordinary and gated round counts, sibling attempts, catches and packet histograms for all 24 prompts. Separately, four ordinary DFlash candidate streams were regenerated using the archived evaluator and replayed through the native Metal verifier. Every one of their 768 committed readouts matched serial target logits, final states matched, and their native three-slot round counts matched the CPU replay. Native verify averaged 23.45–26.39 ms per round in these four runs.
- At the same frozen threshold, the older control prompts together gave 271 rounds under both policies. The holdout result is specific to these new prompts, this greedy sampler, original checkpoint, and this selector precision. No installed worker behavior changed.

**Disposition.** The candidate passes the held-out replay screen in both domains and warrants a live three-slot worker comparison. Qualifying a live implementation requires testing greedy parity and branch-state commit/rollback, then measuring end-to-end decode against ordinary DFlash on matched requests. The available margin is smaller for coding.

## Reproduction

The [aggregate receipt](../benchmarks/data/2026-11-25-inkling-dflash-self-correction.json) binds the prompt manifest, fixture and private selection hashes, scoring source hashes, binaries and weight identities. Prompts, teacher tokens, scores, feature tensors and per-anchor candidates remain private. The experiment used branch base `1dd7ccf82f936d876421cdc1d7ab0d9c8ede8669`, the November 24 pinned z-lab reference, and the same 40-GPU-core, 128 GiB M5 Max.

1. Render each private prompt using the original target and the thinking-off `inkling2-json-v1` template. Pass its tokenized input to the archived `inkling_slice --capture-dspark-trace` executable with the original target and bundled DSpark weights. Omit the historical teacher field to capture native greedy output through EOS or 192 tokens. This yields one private `teacher.safetensors` per prompt.
2. Run the [runner-up extractor](../archive/experiments/2026-11-24-inkling-dflash2/runner_up.py) for each fixture with `--fixture`, `--draft-dir`, `--dflash-repo`, `--target-weights` and an ignored `--output` named `selector.json` in that case's directory. Use the existing MLX runtime with `uv run --no-project --python`.
3. Run the [CPU replay](../archive/experiments/2026-11-24-inkling-dflash2/self_correction.py) with `--set fresh CASES_DIR` on the frozen cohort. Use `--set archived` for the four historical controls. The rate calculation sums rounds within each domain.
4. Run the [selector timing](../archive/experiments/2026-11-24-inkling-dflash2/selector_cost.py) on the four named fresh fixtures under the shared model-scoring lock. Use `--case-root CASES_DIR --target-weights WEIGHTS --pairs 16` with the same MLX runtime and draft revision. Run three times, adding `--reverse` on the middle run.
