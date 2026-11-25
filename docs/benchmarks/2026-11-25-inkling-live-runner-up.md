# Inkling 2 Mini DFlash 2 three-slot worker comparison

Attempt `20261125-INKDRAFT-03`, November 25, 2026. The [fixed-cohort replay](../model-experiments/2026-11-25-inkling-dflash-self-correction.md) predicted a small reduction in verifier rounds when DFlash could spend one proposal on its runner-up. This follow-up evaluates the same frozen rule through the image-loaded, streaming Stark worker.

**Decision.** Keep the ordinary policy for Inkling deployments that select DFlash. The packed runner-up saved rounds in both domains. Adjacent same-worker requests measured a 0.67% prose decode gain and a 0.02% coding gain. Both prompt-paired intervals include zero, and the unaffected control moved by about 0.7% in either direction. The implementation stays available behind the operator flag for research.

## Runtime and workload

Both arms use the same release executable on the 40-GPU-core, 128 GiB M5 Max, macOS 26.7.2, on AC power under the shared scoring lock. Both load the original NVFP4/MXFP8 Inkling 2 Mini target, its vision tower, the BF16 DFlash 2 draft and the F32 corrected selector. The request uses greedy target draws and explicitly sets three draft slots. Each prompt requests 192 output tokens at context capacity 4,352. The benchmark launches a fresh worker for each arm and warms it with one complete untimed request. The reported times exclude launch, image qualification, shader compilation and warm-up.

The 12 prose and 12 coding prompts were frozen before the replay. Input lengths are 65–87 and 73–89 tokens respectively. These same prompts test whether the measured cost of the implementation preserves the replay advantage. They provide no additional independent test of prompt generalization.

The ordinary worker spends its three slots on consecutive primary tokens. The candidate checks the first two positions for a corrected logit gap at most `ln(2)`. On the first eligible position it retains up to two primary tokens and spends the spare slot on a sibling. At a short output tail it can use a free slot. The Metal verifier evaluates the anchor and all chosen nodes together. The sibling attends to its ancestors at its logical position. The commit copies a winning sibling into its logical KV slot, restores the convolution rings, and passes only the selected target taps back to the drafter.

Decode throughput uses the worker's 191 post-first-token intervals per request. Full-request throughput uses all 192 output tokens and the client's wall clock around the streaming request. Both rates pool tokens and seconds within each domain. The worker reports separate draft, target verification and commit clocks. The published receipt contains aggregate counters and hashes. Prompts, generated text, token IDs, scores and target features stay private.

## Correctness and replay comparison

The native test passed all 12 scenarios across one prose and one coding fixture. Each selected readout and its next-step continuation matched serial execution across all 200,058 F32 logits, bit for bit. The digest of all committed attention KV and convolution state matched at both points. The external drafter received the exact selected BF16 feature rows. The tests cover both possible sibling positions, both packet sizes, ordinary primary commits and partial rollback.

A separate image-loaded worker check exercised default and explicit greedy and sampled widths, serial generation and the long-context fallback. Seeded sampled outputs matched serial. A scoped follow-up restored 4,096 input tokens and reproduced the cold result. A generated PNG completed with drafting disabled. The next text request reused zero tokens and reproduced the earlier result.

The live worker emits the initial token from prefill and reserves room for a target draw at the output cap. The archived replay charges each anchor and permits proposals through the final output token. A second replay in the timing harness accounts for the worker's token budget. The original selector capture recomputes the full draft context at each anchor, while live MLX keeps a per-layer draft KV cache and ingests successive feature chunks. Even after matching the output budget, the saved proposals do not reproduce every live draft counter. The live worker counters and the independent frozen teacher outputs are the basis for this comparison.

## Timings with separate worker launches

All 192 timed requests in four paired blocks reproduced the frozen 192-token output. Each of the four runs of a given policy took the same number of target rounds. On prose the runner-up matched the target at **98 of 388** packed forks per pass (25.3%). On coding it matched at **57 of 295** (19.3%). Some forks displace a correct third primary prediction, so the net round savings are smaller than the hit counts.

| Domain | Target rounds per pass, ordinary → candidate | Ordinary decode tok/s | Candidate decode tok/s | Pooled change | Full-request tok/s, ordinary → candidate | Decode change by block |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Prose | 896 → 881 (−1.67%) | 60.94 | 61.79 | +1.39% | 56.32 → 56.89 | −5.75%, −2.22%, +5.45%, +8.68% |
| Coding | 833 → 823 (−1.20%) | 66.95 | 65.90 | −1.57% | 61.26 → 60.21 | −5.95%, −7.54%, +3.01%, +5.02% |

The arm order was AB, BA, AB, BA. Both domains changed signs across blocks. That prompted a tighter comparison under the [playbook's single-worker pairing rule](../PERFORMANCE_PLAYBOOK.md).

## Adjacent pairs in one loaded worker

A private diagnostic build of `6af5401b` adds a single condition, `&& options.seed == 1`, to the existing self-correction gate. Greedy target draws are deterministic for both seed labels. Seed zero selects ordinary DFlash, and seed one selects the runner-up when the request uses width three. The release worker retains the operator-profile gate. This switch allows adjacent requests to use precisely the same resident target, drafter, client and compiled kernels.

The [adjacent-pair harness](../archive/experiments/2026-11-24-inkling-dflash2/live_runner_up_pairs.py) makes two passes through the same 24 prompts and reverses which policy runs first on each prompt for the second pass. Four prompts in each domain also run at width two under both seed labels to measure timing variation on an identical inference path. Each timed request checks its frozen output and the worker's actual mode. The descriptive 95% intervals resample whole prompts, keeping both arms and both repeats together.

| Domain | Ordinary decode tok/s | Runner-up decode tok/s | Decode change (95% paired prompt interval) | Full-request tok/s, ordinary → candidate | Width-two control change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prose | 73.05 | 73.54 | +0.67% (−0.88% to +2.34%) | 67.72 → 68.17 | −0.75% |
| Coding | 78.64 | 78.65 | +0.02% (−1.24% to +1.32%) | 72.38 → 72.38 | +0.71% |

The two prose passes measured +1.09% and +0.26%, with 7 of 12 prompts faster after pooling repeats. Coding measured +0.40% and −0.35%, with 6 of 12 faster. All 128 adjacent-study completions reproduced the teacher. Every width-three request also matched the production worker's accepted count, proposed count and sibling counters for its respective arm. The two labels gave identical accepted counts and round histograms in all 16 width-two control pairs.

At these same-worker rates, the saved rounds allow **0.60 ms** of additional cost per candidate round on prose and **0.43 ms** on coding. The observed added costs were **0.36 ms** and **0.42 ms**. These are averages over the full decode. To express the budget per actual fork, charge the whole overhead only to the rounds that forked. That gives about **1.35 ms per fork on prose** and **1.19 ms on coding**. The calculation multiplies ordinary decode milliseconds per round by the number of rounds saved and divides by the number of packed forks. A hit can displace a correct primary continuation, so the saving uses net target rounds.

This study covers short greedy requests on the original image-capable checkpoint and the pinned DFlash 2 drafter. Both ordinary and candidate workers were branch builds. The installed application was outside this run.

## Reproduction

The native state check is available through `cargo +1.98.1 build --release --locked -p stark-engine --features inkling-slice --example inkling_slice`. Run the resulting `inkling_slice --measure-packed-sibling TARGET TRACE SELECTOR` with an original target shard, a private teacher trace and its hash-bound selector sidecar. For each of the two tested fixtures, the check exercises a first-position sibling in three and four physical rows, a second-position sibling, a fully accepted primary path, and rejections at each position.

Build `stark-generate` with Rust 1.98.1 in release mode using the locked dependencies. Use the pinned [DFlash operator profile](../INKLING_DFLASH2.md), setting `greedy_drafts` to 3 in both launchers and `greedy_self_correction` to true in the candidate. Point both launchers to the same binary. Supply the original private cohort captured for attempt `20261125-INKDRAFT-02` and the original image-capable prepared target. Run the [paired harness](../archive/experiments/2026-11-24-inkling-dflash2/live_runner_up.py) in Stark's locked converter environment with `--cohort`, `--model`, `--ordinary`, `--candidate`, `--lock`, `--blocks 4` and an unused private `--output` path. The harness verifies every completion against its frozen native teacher. It rotates case order and alternates which worker runs first across blocks.

To repeat the adjacent pairs, build the diagnostic condition described above in a detached checkout of the same revision. Start the [single-worker harness](../archive/experiments/2026-11-24-inkling-dflash2/live_runner_up_pairs.py) with its own `--worker` launcher, the candidate profile, `--model`, `--cohort`, the same `--lock` and a new private `--output`. The [aggregate receipt](data/2026-11-25-inkling-live-runner-up.json) binds each binary, the frozen prompt manifest, target and draft weights, timing harnesses and private measurements to their hashes.
