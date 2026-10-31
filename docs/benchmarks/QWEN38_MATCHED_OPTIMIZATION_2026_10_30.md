# Qwen3.8 matched-checkpoint Metal optimization

Date: 2026-10-30. Implementation commit: `74e6c5c2d612f58555e8a85a2013dad23f02d9d7`.
All 797 frozen input files match this commit byte for byte.
[Machine-readable aggregates](data/qwen38-matched-optimization-2026-10-30.json).

## Matched ABBA benchmark

On the same M5 Max host, connected to normal AC power with Automatic power
mode, the final candidate's maximum measured useful decode rate was
66.8075 tokens/s versus 61.1711 tokens/s for MLX Serve: +9.2142% by the
predeclared maximum-request statistic. The user confirmed other apps were
closed for this run; that is a user-controlled host condition, not independent
proof that no external GPU work occurred.

| Engine | Maximum useful decode tok/s | Measured requests |
| --- | ---: | ---: |
| Stark | **66.81** | 10 |
| MLX-Serve 26.10.3 | **61.17** | 10 |

Measured request ranges were 57.24–66.81 and 55.65–61.17 tok/s respectively.
The JSON retains secondary endpoint rates, pooled rates, prefill and wall times.

Rates are tokens/s. Primary decode rate is 255 useful post-first-token outputs
per request divided by each executor's internal decode time; prefill and
transport are excluded. The 256-output endpoint rate is secondary. MLX's timer
includes its unused final forward, which is not subtracted. The engines use
distinct timers. Per-engine repeat outputs were exact; cross-engine numerical
and output parity were not established.

The workload used 4,096 input tokens, 256 output tokens, temperature 1,
top-k 20, top-p 0.95, seed 42; four load blocks ran in Stark/MLX/MLX/Stark order. Each block contained six
calls for that block's engine, with its first call excluded as warmup. Two
blocks per engine produced ten measured requests each. Both engines used checkpoint repository
`ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit`, revision
`788c07a8725642bccf13a0a3b8a21954ab84b4f2`, at the same checkpoint path and
with identical stored weights and quantization. No weight conversion or
requantization, speculation, or prefix reuse was used. MLX Serve was 26.10.3.
The checkpoint manifest SHA-256 is
`11246ebbea8a24a988517ebeb59623b857df89120e5ae3becd48807194542ef6`.

## Candidate identity and precision

The candidate generator binary SHA-256 is
`3946afcf18b9a024b93ee05dfa55952e468a9038dce8fc0a9c2e5b4e660a7a05`; the
scorer binary SHA-256 is
`f778948e8e40caa7add367c172ff75fc7914c4fbe59f5e71fd46b2aee65d4305`. Both
source receipts bind source manifest
`a5135b3a0e0222730a86b999ae91411a3bd2ff1b17003a588e4e19cdb2666921`, source
tree manifest
`301a5b1f92901035edc78678cbc14ed1b8c1d96aacc2ee1cdddcf337c2ad7cc2`, and
Cargo.lock SHA-256
`9ee091260fa8c09b1919885bdbf42538bdd5b4f832258ad84b4321375b19e6e5`.
Build receipts record Rust 1.98.1.

The mixed policy uses BF16 activations and primitive boundaries, Q8 affine-64
KV cache, F32 GDN ValueKey state, F32 accumulation where specified, and
BF16-rounded logical vocabulary-head values widened
to F32 storage. Stored checkpoint weight values and quantization remain
unchanged. MTP, drafter, and PLD are disabled. The exact policy and sealed
source identities are recorded by the source-bound receipts. The verified checkpoint inventory
contains 113 files totaling 107,326,639,344 bytes.

The official policy selects stock-order HC reads and an eligible WR1 write/read
pair, ValueKey GDN decode, TB12 prefill with a contiguous-lane Q/K tree, a
BF16-valued Q4 head, and two admitted Q8 GDN projection shapes. Affine KV and
normalization/gate boundaries have explicit selectors. Other descriptors retain
their previous routes. Scoped compiler math modes leave existing default calls
unchanged. [Source lineage and licenses](../../vendor/lily-core/UPSTREAM.md)
record the per-operation boundaries.

## Three-run engineering diagnostics

These are separate Stark-only diagnostics, each with one warmup and three
measured runs on the 4K/256, temperature-1, top-k-20, top-p-.95, seed-42
workload. They are not ABBA results, not paired to MLX Serve, and do not
attribute differences to an individual change.

| Candidate diagnostic | Maximum decode | Minimum decode | Pooled decode | Pooled prefill |
| --- | ---: | ---: | ---: | ---: |
| `attention-compat-v1` | 42.65 | 41.79 | 42.21 | 1,880.27 |
| `gdn-compat-v1` | 41.06 | 40.31 | 40.66 | 2,145.74 |
| `hc-tail-fusion-v1` | 42.98 | 42.02 | 42.56 | 1,937.55 |
| `moe-compat-v1` | 42.28 | 41.32 | 41.82 | 1,966.67 |
| `hc-full-safe-v1` | 66.84 | 63.40 | 64.79 | 1,734.76 |
| `gdn-value-key-v1` | 67.37 | 65.06 | 65.87 | 1,732.47 |
| `official-holdout-v1` | 64.97 | 63.17 | 63.93 | 1,639.28 |
| `head-stock-qmv-v4b` | 66.67 | 63.48 | 64.90 | 1,750.57 |
| `head-detailed-profile-v1` (unprofiled diagnostic) | 66.15 | 65.54 | 65.88 | 1,765.88 |
| `gdn-pipelined-prefill-v1` | 65.91 | 63.05 | 64.65 | 1,707.39 |
| `gdn-q8-stock-v2` | 65.59 | 60.34 | 62.82 | 1,649.98 |
| `gdn-qk-contiguous-v1` | 65.50 | 63.85 | 64.46 | 1,737.96 |
| `hc-write-pair-v1` | 65.80 | 63.06 | 64.41 | 1,747.44 |
| `final-official-fast-v1` | 66.17 | 63.15 | 64.54 | 1,732.96 |

WR1's maximum changed by approximately 0.46% from its immediate three-run
predecessor, with exact sampled output hashes. This marginal difference does
not establish a reliable isolated gain. The final ABBA result above supplies
the current engine comparison.

## Source-bound component checks

The final candidate archived 797 source files. Its component-validation record
binds the source manifest above and reports 683 passed, 0 failed, 46 ignored
physical native tests; the separate vendor-core default suite reports 69
passed, 0 failed, 35 ignored. Packaging checks passed 20; repository policy
and scoped formatting checks passed. Specific stock/reference checks include:

| Area | Exact recorded evidence | Scope limit |
| --- | --- | --- |
| HC | 7 stock controls; 192 exact stock stage bindings | Component/stage evidence only |
| GDN | 3 Q8 projection controls; 3 pipelined controls; 2 dense Q/K controls; complete-chain stock rows 65 and 129 | Does not establish whole-model quality |
| Head | 4 head controls; stock Q4 head fixtures passed at output widths 128 and 256 | Full-vocabulary stock Q4 head hash is not qualified |
| MoE affine-4 | 3 current-stock stage matches; 9 compiler-mode stock-stage matches | Compact fixture covers 10 experts; full E=512 host admission is not qualified by it |

Additional source-bound controls recorded 3 model-lifecycle passes and 1
manual-holdout/failure-fence pass. The retained-Lily v3 correctness control
matches its reference output hash for a 4K/256 request on the same worker
identity; it is not a performance qualification.

## Numerical and output qualification

The final aligned numerical diagnostic completed 9 fixtures: native GPU and
CPU argmax matched on 9/9, native selections matched MLX on 9/9, and common
candidate-set rank-1 matched on 9/9. Pairwise rank agreement was 970/1,068;
log-probability error RMSE was 0.42886. These bounded fixture results do not
establish quality equivalence. The final greedy control completed 3 fixtures;
only 1/3 decoded outputs matched exactly, while emitted counts and normalized
finish reasons matched 3/3. Cross-engine token-ID parity and quality
equivalence remain unestablished. A separate 4K/16 diagnostic located the first native/instrumented-MLX
difference at output 11, after ten mapped tokens in common. Ordinary and
logprob-enabled MLX responses matched text hash, count and finish reason.
Replaying the common 4,106-token prefix from scratch then changed MLX's own
incremental choice. The diagnostic stopped before interpreting score margins.
It establishes preparation sensitivity at that step, with no causal attribution,
near-tie acceptance or quality pass. Ordinary MLX internal token-ID segmentation
is unexposed; the mapped trajectory belongs to the instrumented response.

## Profile attribution (diagnostic only)

The `head-detailed-profile-v1` baseline predates the later GDN projection,
prefill, Q/K and WR1 changes. The observed nonwarm capture contains 32 GPU
passes: one final held-prompt S1 and 31 generated passes. Its warmup capture is
excluded from these shares. Timestamps can perturb scheduling; inclusive
groups overlap their children and must not be summed together. This records
that instrumented baseline, with no production-throughput claim.

| Inclusive group | Observed share |
| --- | ---: |
| GDN | 36.39% |
| MoE | 28.42% |
| HC | 14.96% |
| QSA | 13.21% |
| Head | 6.38% |
| PLE | 0.56% |
| Embedding | 0.03% |

## Qualification boundary

Independent aggregate auditor v2 passed **389 checks** on the completed final
report. Its 54 offline controls passed. Auditor SHA-256:
`e1fd1ea1aff816037f3bc796856c9f3a7a9cb0e42a4e5a8a9778d3dfe48ba3e5`.
The audit checks arithmetic, workload and artifact bindings; it does not reread
model payloads or establish numerical or quality equivalence.

The shared checkpoint has not reached **77–80 tok/s** in these measurements.
The earlier [58.13/60.95 same-checkpoint baseline](QWEN38_SAME_CHECKPOINT_STARK_MLX_SERVE_2026_10_30.md)
is retained unchanged. The earlier [79.52 tok/s Lily comparison](QWEN38_STARK_MLX_SERVE_2026_10_30.md)
used different quantization profiles and remains separate evidence.

Private source/binary archives and validation artifacts remain ignored. This
record and its JSON contain aggregates and hashes, without prompts, generated
text, token vectors, raw scores, probabilities or per-candidate payloads.
