# Qwen3.8 kernel optimization at fixed quantization

This study follows the [sampled generation and BF16 profiling study](QWEN38_SAMPLED_BF16_2026_10_28.md).
Work began on October 28 and continued on October 29. The starting revision is
`d13e5cde2f6f7d73a955aa81b89080f1b7946e38`. MoE, hyperconnections and GDN are
investigated in that order. Rejected candidates are recorded separately from
retained changes.

## Fixed workload and validation

The checkpoint remains `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl` at revision
`0f94842e3682b49acace4daeb4cf2e65a6f94f83`: 55 shards totaling
106,905,155,192 bytes. All 55 stat identities match the preceding complete
SHA-256 verification. No weights or conversion settings change.

Execution remains BF16 activations, F32 reductions/logits/GDN recurrent state
and Q8 attention K/V, with resident experts, MTP and vision disabled, and no
speculation. The host remains an Apple M5 Max with 40 GPU cores and 128 GB
memory, macOS 26.7.1 (25G241), on AC power. Builds use locked Rust 1.98.1;
Python uses the locked converter environment through uv.

Sampled runs use 4,096 input tokens, 256 requested outputs, temperature 1,
top-p 0.95 and seed 42. Top-k 20 and disabled top-k are measured separately.
The first request is warmup; decode counts the following 255 steps. The public
corpus and tokenization are identical to the preceding study. Generation runs,
kernel tests and builds are serialized. Only aggregate results and hashes are
published; runtime records and experimental patches remain private and ignored.

The same preserved starting worker is measured repeatedly because the host
shows substantial between-window drift. No active competing inference or
Darwin background/resource-throttling flags were observed. `pmset` recorded no
thermal or performance warning. These observations do not identify the cause
of the drift. Component comparisons alternate AB/BA within one process and
use distinct weight sets larger than GPU caches. Instrumented GPU intervals
are diagnostic; throughput comes from separate uninstrumented runs.

## MoE trials

The production geometry is 48 layers, 512 resident experts, top-10 routing,
hidden width 2,560, expert width 640 and Q4/group64 expert weights. The existing
gate/up kernel fuses the two projections and SwiGLU; down projection already
combines selected experts and the shared expert.

- Applying the existing small-M masked-nibble technique to decode failed
  exhaustive finite-BF16 word-dot controls: 24,576 mismatches in 1,566,720
  cases. The candidate was rejected before production dispatch or generation.
- Packing independent gate/up row simdgroups into threadgroups preserved all
  81,972 checked BF16 outputs, including tails, group sizes, duplicate routing,
  offset views, canaries, subnormals and cancellation controls.
- Packing two, four or eight rows gave no steady component-level gain. All
  three were rejected. Original MoE source is retained unchanged.

| Paired MoE trial | Original gate / gate+down us/layer | Candidate gate / gate+down us/layer |
| --- | ---: | ---: |
| Four rows, warmed 16 samples | 46.264 / 87.792 | 47.174 / 88.854 |
| Two rows, separate warmed 16-sample window | 104.835 / 203.229 | 105.003 / 203.778 |
| Eight rows, same latter window | 104.835 / 203.229 | 107.870 / 206.958 |

Absolute times differ substantially between windows; acceptance uses each
paired comparison. The unchanged production worker was sampled again after
these trials. Experimental MoE patches and controls are retained privately;
their aggregate receipt SHA-256 is
`b8b228702a621d32a378d803c76c1659715d4850fb918f96a662488bcbb72303`.

## Hyperconnection trials

Each decode token runs 97 reads with hidden width 2,560, four streams and
low rank 320. The original up kernel computes the same scaled BF16 SiLU
activation separately for every output column.

Three routes were tested:

1. The existing small-batch `m1` pair computes the activation in the down
   kernel and reuses it in up. It passes 28 synthetic exact controls but
   changes the full-model teacher logit hash and both seeded sampled output
   hashes. It is not retained. No explicit precision policy changes, but
   source-level mathematical equivalence did not establish numerical parity.
2. Original down, a separate vector activation dispatch, and a cached up body
   copied from the original up kernel restore full-model and sampled hash
   parity. A paired 97-read benchmark finds no mean improvement:
   59.256094 versus 59.256389 us/read. This route is rejected.
3. The existing `m1` down kernel followed by the cached original up body
   restores the two-dispatch graph. The original down/up entry points remain
   available as reference controls. Its component result and full-model
   qualification are recorded below.

The cached up body keeps the original weight staging, dot expressions,
reduction order and BF16 round sites. Only activation computation becomes
cached BF16 loads. The required activation dependency is included in the
barrier before up. No activation tensor is added: the existing scratch buffer
is reused.

The exact controls compare down, injection, inverse RMS, activation and mixed
outputs, check canaries and offset views, and cover production geometry,
injection present/absent, smaller shapes, group counts, zeros and subnormals.
All 28 cases and 67,416 compared bytes match, with aggregate SHA-256
`79877038d7bfca7cbcabbd71dd4ce7fbc530d5dd313d5b32f6297a32b00250db`.

| Paired HC chain | Original us/read | Two-dispatch cache us/read | Change |
| --- | ---: | ---: | ---: |
| Mean, 16 samples | 57.882195 | 54.757597 | -5.40% |
| Median, 16 samples | 57.627363 | 54.749785 | -4.99% |

This chain includes all 97 distinct production-size weight sets, 679,608,320
viewed weight bytes, and optional injection. It warms for two seconds before
16 alternating samples per route. A component gain does not establish an
equal end-to-end gain.

## GDN

There are 36 GDN layers, each with 16 key heads and 48 value heads of width
128. Two changes are measured independently after the HC stage:

- The recurrent wrapper keeps `float(bfloat(o))` in its own thread instead of
  writing a threadgroup BF16 array, synchronizing and reading the same element.
  The BF16 round, F32 state helper and norm reduction are retained. This removes
  one barrier and 256 bytes of threadgroup storage per head.
- Q4 two-row input projection and four-tap causal convolution share a dispatch.
  The existing packed dot helper and BF16 projection round remain. Only the
  first 10,240 of 16,480 projection rows enter convolution; raw QKV/Z/A/B
  outputs are preserved. Each channel has one window writer, and tap order,
  SiLU and BF16 output/window stores match the original convolution. Other
  projection layouts retain the separate path.

| Paired GDN component, 16 samples | Original mean us/layer | Candidate mean us/layer | Change |
| --- | ---: | ---: | ---: |
| Local rounded recurrent output | 17.872397 | 17.666233 | -1.15% |
| Input projection plus convolution | 58.126809 | 54.857205 | -5.62% |

The recurrence comparison uses 36 separate F32 states per route, 113,246,208
payload bytes each. The projection comparison uses 36 distinct weight sets,
854,323,200 viewed weight bytes. Both warm for two seconds and alternate AB/BA.
These percentages apply to their named components, not the complete model.

The local-rounding controls compare every output and F32 state byte over
12 cases of 16 sequential steps: 341,065,728 checked bytes, SHA-256
`9cafa233669f6eefa90f06022cca7a068d4d147ab637190ecc901961697a74b9`.
They include production head geometry, nonzero state, zero Q/K, both gate
activations, saturated gates, offsets and canaries. The test-only original
wrapper has a distinct pipeline name and retains the removed round trip.

Projection/conv controls compare every raw projection, convolution output and
window byte over six cases of 16 steps: 5,612,352 checked bytes, SHA-256
`c6ab0bfc887c22b6e807138b855603e842677211682eb2212e6500be1cf0f67a`.
They include the complete production projection and a smaller case with 191
convolution channels crossing a two-row boundary, nonzero windows, zeros,
signed zeros/cancellation, offsets and canaries. Original generic entry points
provide the reference.

## Sampled generation

Each row below is a separate sequential window. These observations retain the
requested benchmark after each production candidate, including rejected ones.
They are not isolated per-change speedup estimates.

| Window | Warm requests/case | Top-k 20 tok/s | No top-k tok/s |
| --- | ---: | ---: | ---: |
| Starting worker | 4 | 69.8833 | 59.2780 |
| Unchanged worker after MoE trials | 2 | 67.4047 | 48.1533 |
| Existing HC m1 pair, rejected | 4 | 69.4405 | 60.6894 |
| Three-dispatch HC cache, rejected | 4 | 60.5144 | 51.5408 |
| Starting worker, repeat | 4 | 68.9266 | 60.3067 |
| Two-dispatch HC cache | 4 | 65.1575 | 54.8365 |
| HC cache plus local GDN output | 4 | 72.8057 | 63.3321 |
| HC cache plus both GDN changes | 4 | 74.9481 | 64.8264 |

All requests produce 256 outputs and finish at the requested length. Every
retained route matches both starting-worker output hashes exactly. The rejected
existing HC m1 pair is repeatable within itself but changes those hashes.

### Separate-worker ABBA control

A final four-load window runs candidate A, starting worker B, B, A, each with
one warmup and two measured requests per sampling case. Builds and other
inference remain stopped. This produces the following warm medians:

| Load | Top-k 20 tok/s | No top-k tok/s |
| --- | ---: | ---: |
| Candidate A0 | 76.3314 | 66.2175 |
| Starting worker B0 | 73.6158 | 46.7992 |
| Starting worker B1 | 71.3714 | 53.9419 |
| Candidate A1 | 51.4397 | 55.7161 |
| Starting worker, pooled four warm requests | 72.9552 | 48.0785 |
| Candidate, pooled four warm requests | 64.1763 | 61.4601 |

These observations do not establish a net end-to-end gain: the same candidate
moves from 76.33 to 51.44 tok/s between loads. The pooled top-k observation is
12.03% lower and the nucleus observation 27.83% higher, incompatible with a
simple attribution to the measured small kernel gains. A live subsequent
greedy run shows very little paging and no swap-in/out change. Effective worker
QoS and GPU clock state were not measured. No power or scheduling settings
were changed.

A separate final five-request greedy control measures 77.2817 tok/s warm and
matches the prior completion hash. It is a regression control, not an isolated
before/after comparison. A same-loaded-model control is recorded below to
separate kernel scheduling from between-load variation.

### Same-loaded-model control

A private worker build adds a per-request switch between the original kernel
graph and the retained graph. The switch is set only after drain/reset and
before prefill. Both routes share one loaded model, process, weight mappings,
scratch allocation, CPU sampler and seed handling. The original route uses the
original HC pair, separate GDN projection/conv and a distinctly named copy of
the original GDN gated wrapper. The candidate uses the retained kernels.
These controls exist only in the ignored private copy; production has no
benchmark switch, extra request field or environment control.

Each sampling case warms both routes once, then alternates two ABBA blocks:
four measured requests per route. There are 20 requests total, including four
warmups, and 1,020 measured decode steps per route/case. Every request matches
the expected public-production output hash and the selected-route acknowledgment.
An independent source review verifies the switch order and unchanged sampler.

| Sampling | Original graph pooled tok/s | Retained graph pooled tok/s | Observed change |
| --- | ---: | ---: | ---: |
| Top-k 20 | 61.1398 | 62.3204 | +1.93% |
| No top-k | 64.0755 | 64.1106 | +0.05% |

Pooled rate divides 1,020 steps by summed decode duration. Top-k durations are
16.683083 versus 16.367039 seconds; nucleus durations are 15.918712 versus
15.910000 seconds. Top-k pair changes range from -4.63% to +5.80%; one pair
regresses. This is a modest observed top-k improvement, not evidence for a
reliable gain of at least 3%. Nucleus throughput is effectively unchanged.
Absolute rates in this private build do not supersede the separate production
worker measurements. The remaining between-load variation is unresolved.

The private worker SHA-256 is
`8e6b87741a0031d3c2f63b2529c5262c78e1182c5b4396c81f576d12dffe4c54`.
Its 727-input source manifest is
`a3110a036f3566cc3819969a634d99598dcd10767481878672594179be18e04f`;
all inputs and the executable are rechecked after the run. Exactly six private
control files differ from the corresponding production inputs. The aggregate
results receipt is `1a99cc1971d2557a517619a71c3ba6724a80308a1151f33761040900bba341ce`;
the pooled analysis is `b100fe13c5545c038ffeaa08de69e7d52776d0aea637ec27e94d05907d7033c4`.
Private source patches and runtime records remain ignored.

## Numerical and source identities

The retained HC and GDN stages match the untouched pinned Lily oracle's
32 complete F32 teacher readouts at 4K context. The final route also
matches its short-context control. Fresh-state replay and recovery after
rejected input IDs retain the same hashes. These controls qualify their named
inputs; they do not prove equivalence for all possible activations or contexts.

| Identity | SHA-256 |
| --- | --- |
| Config | `2a91155be52ebbb0f6dff20e61dcd8ae7484154a56b4afe067246c59da46f899` |
| Tokenizer | `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3` |
| Native source receipt | `7f5c685f35fc4c5d1509f6379feb1a965cb625f7a1ce76a0b5fe69983561dd30` |
| Public corpus | `daecdf178c6429a721f9e1b5ae379c664adb8f8af55b381535355636ec469955` |
| 4K input IDs, benchmark JSON encoding | `fd1222c74a823b3a6cf5ed3ffd3dd22cb627afde17c21cef8b2150ac9e14b322` |
| Teacher F32 readouts, 4K | `04e8a4ddd7a69f156ade5d021bf474930c01ff737545ea2fdadd23c70561d7ef` |
| Teacher F32 readouts, short context | `01e0d05f93b6f986cd35a2974908a52e84ac76e89eec20aa9dfdea0b095da40d` |
| Sampled 256 outputs, top-k 20 | `621ca90417af420aaf9defdc957f7cc1f57f318c08d908be764834abb382b0d0` |
| Sampled 256 outputs, no top-k | `9c33d3a4ef1fef97562e00f7f88b5dc287aacadb16b50c4739aec28dbea1b614` |
| Starting generator | `f15b471baf1bd929d0d0233f3d9894042276d10e839cc1a9d7ae3148e0264627` |
| Two-dispatch HC generator | `3aac2c7844990e5a7d06da8b2700a480c4518b4e6683bde9a66bd98f7f32dc56` |
| HC plus local-GDN generator | `f5911918a869647f3944d4c86c5939b83694dd9d1a3442b58646d23990724e80` |
| Final generator | `058f6392cae78d1444578e2122f3d576a077c125a7dec1b503b8549d8117ea8c` |
| Final optimized scorer | `4094e491e941b47f06d76dc288962e9db29a38f4a448d61e2ca18339a58a8b25` |

Earlier source-bound F32 and mixed-precision records remain unchanged. These
kernel changes use the same checkpoint and precision policy as the starting
mixed-precision worker.

The private source inventory covers 736 engine/worker/vendored-core files and
locked build inputs against starting commit `d13e5cde`. Its canonical inventory
SHA-256 is `1d2d394f0ccc00638e293f3867935559791e884bd12ff23664272e67ff7dbfb2`.
All 55 checkpoint stat identities were checked again after final builds and
remained unchanged. Rebuilding the final generator produces the same executable
bytes as the measured projection/conv candidate.

## Validation

- Full native CPU/Metal engine command: 665 unit tests passed, 45 ignored;
  seven C++ string integration tests and three doctest checks passed.
- Three ignored real-checkpoint engine tests were run explicitly and passed,
  covering sampled/greedy encoding overlap, callback counts, early cleanup,
  reset replay and rejected-input recovery.
- Vendored core: four default tests passed; the physical-GPU full-vector
  nonfinite-readout control passed explicitly. All three new exact kernel
  controls and all three paired component benchmarks passed explicitly.
- Generator: 51 tests passed, one real-model selector benchmark left ignored;
  scorer: 39 tests passed in each default and optimized configuration.
- Real workers: 13 generator and 12 scorer requests passed, including seeded
  sampling, isolation, streaming/cancellation, natural EOS, full-logit scoring,
  immediate-consecutive cache rules and recovery. Workers loaded serially and
  closed cleanly. The scoring control checks candidate softmax and a consistent
  full-vocabulary log partition; it does not independently sum every vocabulary
  probability.
- Worker Clippy checks pass with warnings denied, including optimized scoring.
  Standalone vendored-core Clippy exposes six pre-existing findings in two lint
  classes (`chunks_exact_to_as_chunks`, `unused_enumerate_index`). It passes
  with only those baseline classes allowed; their affected code is unchanged.
  Root/core formatting and repository-policy checks pass.
- Independent source review found no actionable correctness issue in the
  retained kernels, guards, offsets, barriers or reference controls.

Run kernel controls with an exclusive physical Apple Silicon GPU slot:

```sh
cargo +1.98.1 test --locked --manifest-path vendor/lily-core/Cargo.toml kernels::hc::tests::cached_read_ -- --ignored --nocapture --test-threads=1
cargo +1.98.1 test --locked --manifest-path vendor/lily-core/Cargo.toml kernels::gdn::tests::local_output_rounding_ -- --ignored --nocapture --test-threads=1
cargo +1.98.1 test --locked --manifest-path vendor/lily-core/Cargo.toml kernels::gdn::tests::projection::fused_projection_convolution_ -- --ignored --nocapture --test-threads=1
```

Use `scripts/benchmark_hybrid_context.py` with the pinned checkpoint and corpus,
4,096 input tokens, 256 outputs, stochastic sampling, temperature 1, top-p 0.95,
seed 42, and top-k 20 or 0. Build the generator and optimized scorer separately
as described in [CONTRIBUTING](../../CONTRIBUTING.md).
