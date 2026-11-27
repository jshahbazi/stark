# HY4STQ-PERF-01 · Hy4 STQ OpenCode performance

This study uses the original Sushi 4bpw artifact at `61528ba635a623f127c5a7b1e1f4ac2c2b0d4885`
on an M5 Max with 128 GiB. The pinned Sushi executable is v1.8.1 at
`b7b7975cc017acba1b4c0877d65543c4c5a4a381`. The Stark release builds use locked
Rust 1.98.1 dependencies and begin at `f30406719`. See the earlier
[native STQ study](2026-11-27-hy4-stq-native.md) for the artifact, vision and DFly2
qualification and the [Sushi source](https://github.com/beamivalice/sushi/blob/v1.8.1/docs/hy4-flash-arch.md)
for the adapted register-fragment schedule and cached positional reads.

## Timing controls

All GPU testing held the shared Hy4 lock. OpenCode 2.4.2 used the same
synthetic review fixtures, with 12 tool definitions, one generation per
turn and 96 greedy output tokens. Stark used the local API and connector,
fitted context capped at 49,152 positions, 4,096-row prefill and F16 cache.
Sushi used its OpenAI-compatible API, a 49,152-position cap, KV8, the
default DFly2 scheduler and disk prefix caching disabled. Both loaded vision.
The client providers render their own templates. Stock Stark measured
16,229 and 27,229 input tokens for the two fresh fixtures, and Sushi measured
15,967 and 26,967. Continued turns embed the first response.

Prefill uses computed input tokens. Decode uses the 95 emitted tokens after
the first sample. Stage sampling comes from separate instrumented runs and
is used for attribution. The fresh Stark fixtures have two uninstrumented
observations and the same-cap Sushi fixture has one at each length. Whole-request
wall includes the OpenCode invocation and excludes model startup.

## Uninstrumented OpenCode runs

Each pair is prefill and useful decode in tok/s. The tuned run uses the
original mapped STQ banks, tensor and register attention, GPU expert
planning, PAIR1 finish fusion, split-K SPEC1, partitioned verifier attention,
cached Engram reads and up to five DFly2 drafts.

| OpenCode fixture | Resident Stark | Tuned Stark, opening | Tuned Stark, repeat | Sushi at the same cap |
| --- | ---: | ---: | ---: | ---: |
| Fresh 16K | 563.81 / 27.92 | 768.31 / 48.66 | 671.57 / 49.99 | 809.84 / 45.63 |
| Continued 27K | 578.57 / 27.07 | | 747.43 / 54.40 | 796.46 / 44.42 |
| Fresh 27K | 577.81 / 28.59 | 739.68 / 48.11 | 748.60 / 49.93 | 862.46 / 45.92 |

On the repeat, Stark took 26.87 s, 22.34 s and 38.92 s across the three
OpenCode turns. Sushi took 22.42 s, 36.47 s and 33.88 s. The 27K continuation
reused 12,288 tokens in Stark and zero in Sushi. The two fresh Stark decode
rates exceeded the same-cap Sushi control on both runs. Fresh prefill was
5–17% behind at 16K and 13–14% behind at 27K. The prior Sushi 118,784-cap
run measured 1,012.80 / 55.43 at 16K and 786.69 / 36.11 at 27K fresh.
[Earlier reference](https://github.com/jshahbazi/stark/blob/865354abef495cd872488d32d206d46348271eb6/docs/model-experiments/2026-11-26-hy4-sushi-opencode.md)

The opening worker repeated the 16K fixture at 756.58 / 48.92 tok/s after
clearing its prefix. First-use 16K GPU spans were 19.07 and 18.94 s across
the two workers. Engram issues and waits accounted for 1.64 and 1.05 s.
Subsequent requests spent about 6–12 ms in that measured host portion.
The GPU and Engram spans do not explain the remaining change in first-use
wall. Engram pages are clean file pages the OS can reclaim. The reader
retains its bounded row table in process, and generation prefix state
stays in memory. The host stayed at normal pressure with no swap growth
in either timed run.

The 16K output matched at zero and five drafts. That serial replay
interrupted the prefix before the opening worker's continued 27K turn.
Its 738.28 / 51.23 tok/s with zero reuse is a separate observation.
The repeat reproduced the continued output with either 12,288 reused
tokens or a fresh prefill. In a 4,426-token text sequence, the worker then
reused 4,096 tokens with identical output. An intervening image request
ran with drafting off and the next text request reused zero tokens while
producing the same output. All three main OpenCode output hashes matched
across the two tuned workers. Stark took 100–102 s to report ready with
the normal image checks. Sushi reported ready in about 20.1 s in each of
its two processes. Those readiness boundaries differ.

The native STQ validation worker is selected with `STARK_HY4_STQ_VALIDATION=1`.
The timed configuration requests `STARK_HY4_STQ_RESIDENT=1`, `STARK_HY4_STQ_TENSOR_MLA=1`,
`STARK_HY4_STQ_REGISTER_MLA=1`, `STARK_HY4_STQ_GPU_VERIFY=1`, `STARK_HY4_STQ_VERIFY_MLA=1`,
`STARK_HY4_STQ_FUSED_FINISH=1`, `STARK_HY4_STQ_SPLIT_SPEC1=1`,
`STARK_HY4_STQ_ENGRAM_PAGE_CACHE=1` and `STARK_HY4_STQ_DRAFT_DEPTH=5`.
`STARK_HY4_CONTEXT_LIMIT=49152` limits the benchmark. The worker still fits
context from available memory and accepts shorter request windows. The
timed path remains opt-in for this artifact while the precision and
packaged-runtime qualification is open.

## Attribution

On the prior resident graph, attention accounted for the largest sampled
prefill stage. The tuned graph reused Stark's 16-query tensor MLA for the
contiguous ranges and a 16-head Sushi-derived register kernel for sparse
selection. The latter keeps queries in registers across 32-key tiles and
reuses the F16 keys as latent values. It uses three scaled F16 query pieces,
two F16 probability pieces and an F32 softmax denominator. Local prefixes
still use the original cache layout and 4,096-row restore boundary.

| Sampled prefill stage | Resident 16K | Tuned 16K | Resident 27K fresh | Tuned 27K fresh |
| --- | ---: | ---: | ---: | ---: |
| MLA | 9.50 s | 2.80 s | 18.16 s | 5.77 s |
| Attention projections | 6.23 s | 6.81 s | 11.99 s | 13.70 s |
| Routed expert products | 5.10 s | 5.11 s | 10.14 s | 10.71 s |
| IndexPool | 0.49 s | 0.52 s | 1.34 s | 1.46 s |

The tuned verifier reused the GPU plan and indirect dispatch structure,
with an STQ planner that keeps expert and router-slot order. It packs the
four Engram inputs before the single target command. The partitioned MLA
assigns each group 256 keys and merges the F32 softmax statistics with
one sink per head. At 16K and depth five, target verification took 1.72 s.
The GPU profile attributed 0.53 s to dense projections, 0.50 s to routed
products, 0.08 s to IndexPool and 0.07 s to MLA. The earlier depth-two
resident run spent 3.06 s in verification. The change bundles kernel,
submission and draft-depth changes.

The first uninstrumented combined arm kept two drafts and ran at
689/30.1 tok/s at 16K and 641/29.3 at 27K fresh. The five-draft split-K and
tiled IndexPool arm ran at 734/48.2 and 745/43.3 tok/s. It changed output
hashes from the two-draft combined arm. The follow-up disabled tiled
verifier IndexPool, used one split-K schedule at widths one through six,
and applied the same key partitioning to greedy and speculative text
decoding. The 27K fresh input produced the same 96 token IDs with zero,
two and five drafts in that worker. Those IDs matched the first two-draft
combined arm. Both combined arms changed the output hash from the original
resident graph. These fixtures establish consistency within the tested
arithmetic and do not qualify the change on a broader corpus.

## Component screens and rejected work

| Attempt | Result and disposition |
| --- | --- |
| HY4STQ-PERF-01a · PAIR1 finish and order reduction | Real layer-one packed banks matched all 7,864,320 F32 values. Full routed GPU time fell from 7.090 to 6.711 ms at 1,024 rows and from 25.470 to 23.934 ms at 4,096 rows. The integrated flag applies at 1,024 rows and up. |
| HY4STQ-PERF-01b · SPEC1 K6 split-K | Nine real projection shapes passed at every width from one through six with F32 split partials and the previous F16 product boundary. Selected paired component times were 0.17–0.90 of stock at one, three and six rows. Worst raw relative RMS from the initial F64 checks was 1.88e-5. Prompts and K4 keep the regular packed product. |
| HY4STQ-PERF-01c · Per-head absorbed K/V product | The F16-split prototype reproduced all 22,020,096 stored BF16 weight words and measured at most 2.04e-7 relative RMS against F64 on the tested outputs. A 64-head 4,096-row K projection was 17% slower while V saved 0.38 ms. Kept the incumbent. |
| HY4STQ-PERF-01d · Pipeline admission and harness retries | Two early arms never reached ready because Metal rejected the PAIR1 lane-index types. The corrected component and worker passed. The first 49K Sushi control tried a model-prefix reset rejected by the server. The valid comparison uses separate workers for the fresh 27K arm. The lifecycle harness first lacked Pillow and then submitted a nonstreaming replay with `stream_options`. The corrected rerun passed with a stdlib-generated PNG. |
| HY4STQ-PERF-01e · 32-bit SPEC1 K6 fragment funnels | Each of two patterns matched all 283,115,520 component results. The Sushi-style pattern reduced gate and query projection time 3–6%, with mixed results on output projections. The fully 32-bit pattern regressed most shapes. A larger apparent sliding output gain lacked a stable paired control. Both stayed out of the graph. |
| HY4STQ-PERF-01f · Tiled verifier IndexPool | The targeted scorer can accumulate differently at singleton and batched widths, which can change near-tied group rankings. The early five-draft combined run changed the generated IDs relative to the later draft-equivalent runs. The tiled verifier experiment was removed. Prefill retains its existing tiled scorer. |
| HY4STQ-PERF-01g · Reuse packed K6 fragments across four row stripes | All 113,246,208 F16 component values matched at 4,096 rows. Across nine paired timings, GPU time increased 3% for the sparse query, 9% for the sliding gate and 27% for the sliding output. Kept the two-stripe product. |

The repeat still needs about 5.1 s less prefill time on the fresh 27K
fixture to reach the same-cap Sushi observation. That is about 37% of
the separately sampled attention-projection time. The tested per-head,
fragment and four-stripe changes lack that savings budget. A separate
first-use trace is also required to account for the host wait that moved
between the two 16K runs. Those gaps and the broader precision corpus
remain open.

## Evidence identities

Only aggregate timings and hashes are retained here. The fixture inputs,
raw output IDs and private worker receipts are excluded from the repository.

| Evidence | SHA-256 |
| --- | --- |
| Resident opening timing receipt | `bb19ab585866d169317ef369b59ffe7322c3f368d6b437418a7f05dbc36282d6` |
| Resident stage profile | `7f8553377a19bc74e7ac58c8a0fe002d9e175d410b3db0d8bbdf6146a0df128a` |
| Two-draft combination receipt | `918473779b55589bd6691055b06de41892c83f994feea05c331d994bc83e552f` |
| First five-draft combination receipt | `83718b0521246c7cbb5128f6e0457cfa9daff1c9db28625d330ba31a96167c76` |
| Tuned sampled profile and draft agreement | `aaba65a12665d83152db8ab875bc9c8629a7426b299c1442982192f991c9a589` |
| Tuned sampled worker executable | `8c3ba8866506bbe4ac3200c4e6d813c6493dfa5b102845d4eba5fe99961b8f51` |
| Tuned uninstrumented worker executable | `3e133b969c2c991b692b9dd6f7d63051c263a6f910797e026420b161007e0875` |
| Tuned uninstrumented timing and 16K greedy comparison | `8ceb9d2ddfd30623438cf526d5d9d21ec99e6335d5060a95e236bed0751118ab` |
| Repeat worker executable | `bae58bb788455c077db3b72460b070b455a27500e095d9d19a524aab18acfa83` |
| Repeat timing and image/prefix lifecycle | `c536922a786d5d618ff2f462bc6d1d1f3bbc4a1a4cea532475cbf099a9c1b0f3` |
| SPEC1 width and split component receipt | `43487082109bd79e4ac642ea13b958b8954e0d812b064f1db2e7e142006b6dfa` |
| Absorbed K/V component receipt | `ab461cd1f91799b4cf889c39b67ee56d6136d1e1ccda56ed8d45d8303ccb4977` |
| 32-bit packed fragment component receipt | `3e13cc55c829820d14e3f1e552009e34872b226a87287de504716f4c2c2ad218` |
| Four-stripe packed projection component receipt | `143c81f2442a493ff43d374795c35ddf32fe9ea28bd5b1c6575d4d4a4bc7fea9` |
| Sushi 49,152-cap OpenCode receipt | `d61c686ff5a425e3c2f9167a158e429b128088ce37b75244c6db810fb184ac3a` |

## Local checks

The locked Hy4 release worker built successfully. Clippy with warnings
denied and the pinned formatter passed. The generation library passed
183 tests with six ignored. The converter passed 726 focused tests after
the downloadable Hy4 catalog assertion was moved to the install-planning
view. The browse view still has independent coverage for the Sushi entry.
Ruff passed on both changed Python files.
