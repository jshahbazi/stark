# Dense Qwen27 packed weights and MTP follow-up

This follow-up preserves the [original source-bound study](2026-11-06-qwen38-27b-oq4e-mtp.md)
and qualifies new artifacts separately. The checkpoint remains
`Jundot/Qwen3.8-27B-oQ4e-mtp` at
`7dd276b77fcb27c982ebd8d8a434539345713fea`. The host is Apple M5 Max,
128 GiB unified memory, macOS 26.7.2. Generator builds use Rust 1.98.1 with
locked dependencies. Tests use the pinned nightly toolchain.

The [approach and decision ledger](2026-11-06-qwen38-27b-approaches.md)
records the retained changes, unsuccessful candidates, numerical repairs and
investigated-only ideas across both stages of this work.

## Packing-only result

The first sealed packing worker measured **48.957 versus 45.250 useful decode
tokens/s**, an **8.19% median increase**, in a balanced native comparison. Both
arms completed six 768-token responses. Every measured completion hash matched
across the two arms. These are **unscreened diagnostic observations**;
they do not establish a throughput floor or meet the 70.8 tokens/s objective.

| Control | Value |
| --- | --- |
| Prompts | Same three private code prompts; thinking disabled |
| Sampling | Temperature 0.6, top-k 20, top-p 0.95, seed 42 |
| MTP | Three drafts; exact host rejection sampling |
| Context capacity | 8,192 tokens; identical formatted inputs |
| Order | Widened, packed, packed, widened blocks |
| Per block | One excluded warmup, then one response per prompt |
| Prefix reuse | Disabled for the throughput comparison |
| Workers | Loaded sequentially; one active model at a time |

| Prompt index | Widened observations, tok/s | Packed observations, tok/s |
| --- | --- | --- |
| 0 | 40.660, 39.687 | 39.927, 41.971 |
| 1 | 45.602, 46.598 | 48.483, 49.431 |
| 2 | 44.899, 47.605 | 50.807, 50.579 |

The packed file is **17,078,038,512 bytes**, versus **21,197,528,944 bytes**
for the earlier widened file: 4,119,490,432 fewer bytes. Input embeddings retain
their admitted eight-bit representation. Uniform five/six-bit affine matrices
retain the source packed bytes, and heterogeneous projections retain contiguous
row partitions at their original widths. No target code or affine coefficient
is requantized.

Restoring four-bit rows initially changed the widened implementation's lane
reduction, and smaller partitions changed GEMM split-K planning. That candidate
failed the predeclared whole-model numerical gate and was repaired before the
packing measurement. Partitioned linears now retain their original fused output
dimension for planning, and use widened-eight-bit row arithmetic with the actual
packed codes. Uniform five/six-bit linears also retain eight-bit planning.

The repaired artifact matched the widened artifact at **all 24,583,680 compared
logits**: three prefill rows and 32 teacher-forced positions per prompt. Maximum
absolute error, RMS error and mean KL were zero; all 99 top-token choices matched.
The packed artifact's scalar versus four-row verification was also bit exact
over those 99 rows. These are bounded checks on the named artifact.

| Packing-stage identity | SHA-256 |
| --- | --- |
| Widened worker | `6d43ffe67f4087cf864c8e65b244c04c30530999ddf725986f47586e4be4511f` |
| Packing-only worker | `5ddbced8172ee2fb986fb51a1e8968342388dfdf0ccc90cdac1506e8c8aac082` |
| Packing-stage Rust/Metal source bundle | `0c60f27211f64a0159e96bddb45a412c1b47b86346f7c7f7200c82b4d7e25d05` |
| Packed weights | `bdde7b725452290d9c0d1fb7faf6571a0c792ed78690069e0d0819c02306e8cd` |
| Configuration, unchanged at this stage | `bd59a055d26a850aa41adec396b66eeb8aa7fab38e51217a71bb538ae60bcad7` |
| Tokenizer, unchanged | `87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4` |

## Draft-only readout

The inspected [pinned oMLX driver](https://github.com/jundot/omlx/blob/2165e75726ba83d9651dbaced61c97728da77bba/omlx/patches/mlx_lm_mtp/batch_generator.py#L1820-L1870)
uses a coarse head followed by accurate candidate rescoring. Stark's optional
`mtp_config.draft_readout_bits` selects three or four bits; omission retains
the complete original draft readout. This is a new proposal policy. Target
verification always reads the complete original vocabulary head.

The auxiliary matrix is derived once at load from the original unsigned affine
matrix. It drops low code bits within each group, scales the affine coefficient
by the corresponding power of two, and rounds a midpoint-adjusted bias to BF16.
It does not change target weights. The three-bit head occupies 556,236,800 bytes
and the four-bit head 715,161,600 bytes for this artifact, excluding allocator
overhead and temporary scores.

Each step ranks coarse F32 scores on the GPU, selects 64 token IDs, and rescores
those IDs with the original BF16-output head. The host rejection sampler uses
the actual resulting proposal distribution. Unselected proposal entries have
finite minimum-F32 logits, whose probabilities underflow to zero; the target
and rejection residual retain the complete target support. Exact-rejection
mass tests cover support holes and this finite masking policy.

A component experiment in pinned MLX 0.32.4 measured 2.016 ms for the exact
six-bit head, 1.228 ms for three-bit plus 64-candidate rescoring, and 1.493 ms for
four-bit plus rescoring (24 interleaved observations per arm). These are external
head-component timings, not native or whole-model speedups. Six synthetic rows
retained all 120 exact top-20 candidates in both shortlists.

The native complete score/shortlist/rescore check uses the same six synthetic
rows. GPU selection matches F32 reference ranking, each gathered BF16 score
matches its full-head score bit for bit, and all 120 top-20 candidates survive
for both widths. A first trial passed F32 storage to a BF16-only selector;
that trial is excluded. The selector now requires an explicit data type, with
BF16 compatibility and F32 precision/buffer tests.

## Predictor state and command scheduling

The predictor retains the already-valid root KV after drafting. Catch-up replays
only accepted proposal tokens, paired with the preceding committed target
hidden rows. A root-only acceptance requires no replay. Zero-draft rounds still
prime the root. Cache-only priming and repair preserve the full projection,
normalization, RoPE and KV preparation geometry while omitting attention output,
FFN and final normalization whose results are unused.

Target commit, predictor catch-up and ownership of the last target hidden row
share one command completion. A three-draft round therefore has five normal
completions rather than the earlier eight. Draft sampling remains the existing
host policy. `rollback_seconds` now covers the combined commit/catch-up/copy
phase; its boundary differs from the original report.

The retained prefix remains model-instance-bound, in memory, on the 1,024-token
grid. Coarse width is an immutable model property; the exact selected request
depth remains part of the reuse key. Request-local
accepted-prefix histograms contain counts only and are not retained across
requests. Failure/cancellation/restart/publication rules remain unchanged.

## Follow-up disposition

Initial greedy screening preserved all completion hashes, but reference
throughput drifted about 15% between blocks while Time Machine and macOS cache
reclamation were active. The user authorized pausing the backup for controlled
measurements and restarting it afterward.

With the backup paused, an interleaved comparison of two preloaded workers
measured pooled medians of 46.808 tokens/s with the complete draft head and
46.146 with the three-bit head. Every greedy completion hash matched. Quiet
admission passed, but background CPU bursts disqualified several requests;
only two complete pairs passed their request screens. Their small improvements
are insufficient to select the approximate head as the default. It remains an
explicit opt-in. The two-aligned-word unpack trial was removed because it lacked
a reliable measured gain. Its private source and receipts are retained.

## Longer verification

A private experiment raised the configured limit to seven drafts. That path
was removed after the timing trial. The shipped implementation keeps a
three-draft maximum and recommendation; requests still default to zero drafts.

The target has 48 DeltaNet layers and 16 full-attention layers. The experiment
extended DeltaNet's accepted-prefix histories through eight input rows.
Full attention initially failed the numerical gate at this batch length. Its
repaired, narrowly selected path transposed the prepared head-major queries
once, then executed the existing scalar attention core with the exact visible
KV length for each row. These experimental additions were subsequently removed.

The repaired four-row-tile candidate passed a whole-model audit covering six
contexts, including lengths 1,029, 1,030 and 2,056, and 32 teacher-forced positions
per context. All **49,167,360 logits across 198 rows** matched exactly between
the widened and packed artifacts and between scalar and eight-row verification.
The attention component also matched scalar execution in 32 comparisons across
rows five through eight and eight context boundaries.

In the initial four-row-tile pilot, seven drafts advanced 3.85 tokens per round
versus 3.02 with three drafts, but target GPU time rose from 47–50 ms to 79–85 ms.
Seven-draft throughput was 35–37 tokens/s versus 48–50 for three drafts on the
same 128-token greedy response. All completion hashes matched. That trial
required two target-weight scans per verification.

A second experiment reused weights across all eight rows, preserving the
scalar lane arithmetic. Its whole-model audit again matched all 49,167,360
logits exactly. Fixed-depth greedy screening then measured two 128-token
responses per prompt at each depth, in ascending and descending depth order.
All 42 completions matched their prompt's reference hash.

| Drafts | Pooled median useful decode tok/s |
| --- | ---: |
| 1 | 32.408 |
| 2 | 39.044 |
| 3 | 38.633 |
| 4 | 35.095 |
| 5 | 32.312 |
| 6 | 27.620 |
| 7 | 26.232 |

These are unscreened diagnostics with visible throughput drift, so they do
not establish a qualified speed comparison. Longer chains lost overall in
both tiling trials; a small five-draft improvement on the third prompt did not
offset losses on the others. Both longer-chain implementations were removed
from the shipped source. Their complete source snapshots, worker binaries and
receipts remain private. The full-row experiment's source bundle is
`c9347ae802ab841df4dde94dd076da9fe2b6a6eb9d987f4c5d82369e3988ba86`,
and its worker SHA-256 is
`237b3ed9da18c5b5d4e44bbedd803a1a84f3734c229d5775b719405863d84b7a`.

Request-local accepted-length histograms support depth analysis. This change
does not implement an online adaptive draft-depth controller.

## Final shipped source and validation

The final source retains lossless packed weights, row partitions, optional
three/four-bit proposal heads, cache-only predictor priming/repair and the
combined finish command. Its maximum and recommendation remain three drafts;
drafting defaults to zero and the original draft head remains the default.

| Final identity | SHA-256 |
| --- | --- |
| Rust/Metal source bundle, 752 files | `3235babeb1b3700bb5c56c0fa51fa8745a5b0d6a1ff582c61fd3ccaa19b166ee` |
| Separately built generator, Rust 1.98.1 | `8d2b01e20c680fc96bf975e1018991a9c38134af7acfd34496c78480e8ba8609` |

The final build repeated the packing audit: all 24,583,680 logits across 99
rows matched exactly between the widened and packed artifacts and between
scalar and four-row target verification. Maximum/RMS error and mean KL were
zero, with all top-token choices equal.

The packed artifact with its original draft head and the separately configured
three-bit and four-bit proposal artifacts each passed **40 prefix comparisons
and six invalidation controls**. Each was compared against fresh execution
under the same policy, including greedy and seeded sampling, exact selected
draft depths, 1,024-token reuse boundaries, context/scope changes, cancellation,
errors and restart. These are three separate artifact-bound results. The
different proposal policies are not required to produce identical seeded
completions to one another.

The separately built optimized scorer matched complete-vocabulary bit digests
between the widened and packed artifacts for six contexts, including the three
longer contexts above. All 12 replies per artifact, covering cold execution and
immediate cache hits, matched. This is a new-artifact scoring comparison; it is
not a replay of the historical frozen Uzu scoring fixture.

The final native shortlist/rescore test also passed for both proposal widths:
six rows each, exact F32 shortlist ranking, bit-exact original BF16 rescoring,
and 120/120 exact top-20 candidates covered for each width.

The complete CPU/Metal suite passed 817 engine tests and 12 integration or
documentation tests, with 91 ignored engine tests. Separate generator tests
passed 119 cases with two ignored; optimized scorer/probe tests passed 51.
The converter suite passed 6,141 cases with 90 skipped. Benchmark/prefix harness
tests passed 35 cases and seven subtests. Formatting, Ruff and repository policy
checks passed. An independent source review found a non-BF16 row-partition
admission gap; the final loader rejects that unsupported precision before
loading its child matrices.

## Final throughput

The final generator measured **51.483 useful decode tokens/s median** across
six 768-token sampled responses, with the original draft head and three drafts.
All six requests passed the unchanged host screen on their first attempt;
there were no excluded attempts. Every response reached its budget and every
completion hash matched the corresponding earlier packing-comparison result.
The **70.8 tokens/s target was not met**.

| Prompt index | Useful decode tok/s |
| --- | --- |
| 0 | 42.008, 42.789 |
| 1 | 51.138, 51.828 |
| 2 | 53.664, 56.784 |

The protocol used the same three private code prompts, thinking disabled,
temperature 0.6, top-k 20, top-p 0.95, seed 42, an 8,192-token context capacity,
one excluded warmup and prefix reuse disabled. Only one model was active;
compilation and other validation had finished. Time Machine was paused with
the user's authorization. Each measured request followed at least ten seconds
of observed quiet admission. The maximum observed external CPU total was
92.581%, external model CPU was zero, and the maximum thermal state was one.
All pageout and swapout deltas were zero.

This is a new host-screened result on the named build. The earlier packing-only
A/B was unscreened, so comparing its 48.957 median with this 51.483 median does
not isolate the benefit of the later predictor changes. Neither run reproduces
the public 70.8 result, whose original artifact revision was not pinned.
The [public aggregate data](data/qwen38-27b-packed-followup-2026-11-06.json)
bind the final build, artifact/configuration, corpus and harness hashes and
retain each measured request's aggregate host checks.
