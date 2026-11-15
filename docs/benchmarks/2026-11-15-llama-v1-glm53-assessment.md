# llama.cpp 1.0.0 against Stark GLM-5.3

Date: 2026-11-15. Attempts: `20261115-LCPPGLM-01` (source assessment),
`20261115-LCPPGLM-02` (model-free component controls).

## Decision

GLM has a stronger prefill investigation than the retained Lily artifact.
The first target is the existing GPU-resident routing and batched quantized
projection profile, followed by selective MLA/KDA prefill controls. Several
of the useful llama techniques already exist in Stark as opt-in paths. The
question is whether the actual artifact and workload justify enabling or
extending them. It does not justify applying every release change.

The historical same-binary GLM T8 MMA comparison measured 17.13% faster
prefill and 13.66% lower request time on its 96-input/8-output workload. That
is relevant evidence for a noticeable bounded benefit, with the original
source and limitations preserved below. No new GLM full-model speed result
has been measured in this follow-up.

## Scope and current availability

This compares the release inventory from the
[Lily assessment](2026-11-15-llama-v1-flashnext-lily-assessment.md) with two
distinct GLM implementations:

| Implementation | Artifact and execution boundary |
| --- | --- |
| Native Rust/Metal GGUF | `glm5_next` routes to `Forward::Glm`. The documented original artifact is `unsloth/GLM-5.3-Flash-GGUF`, revision `a38483c8cd5df544f53d70fb281afe97369d5ab6`, UD-Q2_K_XL; metadata/tokenizer source `zai-org/GLM-5.3-Flash`, revision `04c4e9e95c5da8862dced7e5056455116f83a7e0`. |
| Optional EXL3/FP8 bridge | Feature-gated `Forward::GlmExl3`, a separately staged MLX/native bridge and different checkpoint. Retained experimental pack: `OpensourceWTF/GLM-5.3-Flash-streaming-repack-exl3-3.05bpw`, revision `50237d5c26ceb98bfd85f779990b0f5e6d7af144`. Its native MTP and retention policy have their own qualification. |

Inspected Stark source: `f9c9720c23ee848fcd64c002450f04b727e32e35`, whose
engine source is unchanged from `cc8e47a824f802e24d442183ebbdd29fd25ecaaa`.
The previous Lily worker is not evidence of GLM execution; the artifact and
backend are different. Installed-app behavior and inherited environment
selectors are not inferred from a source review.

The exact four native GGUF shards total 108,720,071,552 bytes (101.25 GiB).
They were not found in the checked repository model/runtime roots, Stark
Application Support models, Downloads, or the exact Hugging Face cache root.
`~/models` is absent and no external model volume is mounted. Other GLM
repositories in the HF cache do not establish availability of this artifact.
The exact optional EXL3 pack was also not found in these checks. This was a
bounded search of expected locations, not proof that no copy exists anywhere.

Available disk was approximately 182 GiB. No checkpoint was downloaded, no
other task or application was stopped, and no app/default was changed. Missing
weights prevent a matching fresh full-model comparison. Ordinary host activity
is accepted for future measurements as instructed; lack of quiet is not the
reason this comparison is unavailable.

## Native GLM: source comparison

| Mechanism | What Stark actually does | Consequence for llama ideas |
| --- | --- | --- |
| Decode graph | On by code default when the controls are admitted, unless `STARK_GLM53_DECODE_GRAPH=0`. | Avoid describing the whole engine as CPU token-at-a-time; graph execution exists. |
| Expert routing boundary | Without admitted encoder banks, the graph completes to read router logits on CPU, selects expert slices, then builds the expert graph. For the named artifact, this topology has 42 routed-layer completion calls plus final readout. A dated 2026-10-28 profile previously observed 43 completions/token. | Reducing actual CPU/GPU completion boundaries is a more concrete lead than another sigmoid/top-k fusion. The count is conditional source topology plus historical evidence, not a new trace or a claim about every inherited launch environment. |
| GPU bank routing | With admitted banks, sigmoid, selection bias, stable top-k and unbiased normalized mixture execute in the GPU graph, followed by indexed experts. | llama #31615 substantially overlaps this existing route. Its benefit depends on selecting/admitting the route, not merely having a fused router shader. |
| Opt-in complete profile | The wrapper enables resident/encoder banks, GPU greedy, control/KDA/packed-kernel options, overlap and T8 prefill MMA. Most of those selectors are off by code default. | Compare the normal path with the profile first; isolate MMA within a fixed profile before attributing a gain to one flag. No automatic profile injection was found in the inspected standard worker launcher. An inherited environment can still select it. |
| Packed arithmetic | Source GGUF quantization remains packed; native activation, accumulation, KDA state and MLA/indexer storage are F32. | Lily's Q8 KV policy does not apply to GLM. F32 references must be retained; mixed precision remains allowed after numerical and performance qualification. |
| MLA/indexer cache | F32 latent rows, pooled index keys, and tail keys/gates, rather than generic full K/V vectors. | Generic FP8 KV work is not a compatible cache implementation. Quantizing selected streams would be a separate design. |
| MTP | Native GGUF accepts an optional NextN block in the inventory but does not execute it. Enabling native GLM MTP is unsupported; optional EXL3 MTP is separate. | The upstream GLM MTP copy speedup has no reached native counterpart today. |
| Prefix reuse | 1,024-token-grid snapshots restore KDA recurrent/convolution state and incomplete indexer tails; completed attention/index data remain model-owned. | Per-request prefix snapshots are distinct from per-draft rollback snapshots. Do not import #31660's percentage into prefix reuse. |
| Batched prefill | Existing T4/T8 MMA plus optional wider Q5/Q6, grouped expert and batched MLA paths. Many expert formats still use per-token work in a chunk. | Few-row matrix reuse can be relevant during prefill even without MTP. It is not an unexplored idea in this engine. |
| Register KDA prefill | Existing opt-in chronological F32 operator retains state locally across tokens; decode stays on the single-token implementation. | This is the relevant recurrence to evaluate. CUDA GDN/SSM kernels implement different operators. |

Owning sources: [backend selection](../../crates/stark-engine/src/engine/hybrid/mod.rs),
[native GLM](../../crates/stark-engine/src/engine/hybrid/glm.rs),
[GPU router and cache](../../crates/stark-engine/src/engine/hybrid/glm_metal.rs),
[router/KDA shaders](../../crates/stark-engine/src/engine/hybrid/glm_metal.metal),
[prefill routing](../../crates/stark-engine/src/engine/hybrid/glm_prefill_batch.rs),
[greedy orchestration](../../crates/stark-engine/src/engine/hybrid/glm_greedy.rs),
[prefix restoration](../../crates/stark-engine/src/engine/hybrid/glm_prefix.rs),
[experimental profile](../../scripts/with_glm53_optimized.sh), and
[current native documentation](../NATIVE_HYBRID_MODELS.md).

The [dated decode-graph profile](GLM_QWEN_DECODE_GRAPH_2026_10_28.md)
is the provenance for the observed 43-completion count. The
[image-status matrix](../MODEL_IMAGE_STATUS.md) records this native Flash GGUF
route as text-only; the non-mixed original GGUF image loader rejects image
mode. This does not describe the separate unsupported full GLM5.3
`glm_moe_dsa` architecture or establish image qualification for another export.
The optional NextN inventory is preserved rather than claimed to execute;
native draft execution/verifier/rollback is the concrete missing MTP path.

## Ranked native-GLM checks

These are engineering judgments for the named native GGUF artifact, not
predicted speed percentages. The first two deserve priority. No production
default or new arithmetic path is promoted by this source assessment.

| Rank | Check | Worth exploring? | Reason and stop condition |
| ---: | --- | --- | --- |
| 1 | Normal launch versus admitted GPU-bank/complete profile; count exposed command completions | Yes, first full-model diagnostic | The normal CPU routing boundary is a concrete synchronization cost. Existing GPU routing can remove that dependency under its memory/ownership gates. Measure actual selected route and end-to-end time; a shader-level fusion count is insufficient. Reject a memory-heavy configuration if admission fails or it does not clear the busy-host hurdle. |
| 2 | T8 quantized MMA and batched MLA projections | Yes, strongest existing bounded speed evidence | The old T8 same-binary run improved prefill 17.13%. Isolate MMA with other profile flags fixed. Then test MLA batching separately. Avoid transferring the 17% to decode, long prompts, another quantization or today's build. |
| 3 | Register KDA prefill, with chronological F32 arithmetic | Conditional | Source and synthetic controls exist. Measure whether fewer state loads/stores outweigh serial dependencies, register pressure and spills. Test alone against its selected old GPU operator. Stop on numerical/state mismatch or no noticeable complete-request gain. |
| 4 | Grouped expert prefill and wider dense tiles | Low priority unless profiling explains prior failure | Format admission is narrow, and a prior all-selector trial had a routed-expert GPU-stage slowdown despite bounded readout agreement. That does not isolate each selector's effect. Do not repeat the combination merely because CUDA few-row MMQ improved a different model. |
| 5 | Mixed-precision projections or MLA/indexer storage | Separate larger experiment, defer initially | Native F32 gives potential arithmetic/storage scope, but generic FP8 KV is explicitly unsupported for GLM5_NEXT upstream. Define exact streams and references. Weight-dominated model memory and short-context workloads may provide little practical benefit. No mandatory all-F32 production rule is assumed. |
| 6 | Native GLM MTP and coalesced per-draft snapshots | Not a small port; defer | Would first require native draft execution, exact sampling/verifier/rollback, head-state maintenance and cache correctness. A +7–11% improvement to an existing upstream MTP path does not estimate the benefit of implementing that whole subsystem. |
| 7 | More router fusion, cache policy, small-K LoRA dispatch, ggml graph metadata reuse | No direct implementation task now | Router fusion already exists under bank admission; no matching explicit host-expert cache or LoRA K=32–63 path is established; ggml graph metadata is not Stark's allocator. Only reopen a row with a measured matching critical-path cost. |

## Upstream evidence mapped to GLM

| Idea | Source-bound upstream evidence | Judgment for native GLM |
| --- | --- | --- |
| Metal sigmoid/bias/top-k/normalization fusion | [#31615](https://github.com/ggml-org/llama.cpp/pull/31615): GLM 320B.A18B Q4_K_M on M5 Ultra, prefill 787.36→807.48 tok/s at pp512, 684.87→693.41 at pp2048, decode 41.39→42.31 at tg128. | Roughly 1–3% upstream gains. Existing GPU-bank route is already fused; the CPU-routing boundary and bank admission are the important local distinction. |
| Conv rollback copies in one set_rows | [#31660](https://github.com/ggml-org/llama.cpp/pull/31660): GLM MTP greedy on RTX PRO 6000 reports +7–11%, 12 copies→3 set_rows per KDA layer; separate later multi-request GLM results were near-neutral/slower. | Specific to executed MTP rollback. Native GGUF cannot receive that gain without native MTP. Prefix capture is a different event and scope. |
| Preserve fusion groups across command buffers | [#31688](https://github.com/ggml-org/llama.cpp/pull/31688): GLM MTP rounding/reproducibility correction; M5 Max throughput stayed within 1%. | Correctness lesson for scheduling changes, not a speed promise. Stark explicitly encodes its operations. |
| Few-row MMQ/MMVF | [#31533](https://github.com/ggml-org/llama.cpp/pull/31533), [#31959](https://github.com/ggml-org/llama.cpp/pull/31959): CUDA kernels, with large gains in certain speculative-verification or BF16 matrix shapes. | Relevant principle is cross-row weight reuse. Native GLM's existing T4/T8 prefill and optional expert/MLA batches are the actual places to evaluate it; no NVIDIA percentage transfers. |
| K=32–63 matmul dispatch | [#30982](https://github.com/ggml-org/llama.cpp/pull/30982): Metal LoRA rank-32/48 bottleneck. | Do not confuse GLM low-rank architecture names or quantization group sizes with an actual matrix K in this interval. No matching bottleneck established. |
| FP8 KV | [#28392](https://github.com/ggml-org/llama.cpp/pull/28392), [#28413](https://github.com/ggml-org/llama.cpp/pull/28413). [The immutable v1.0.0 context source](https://github.com/ggml-org/llama.cpp/blob/v1.0.0/src/llama-context.cpp#L4502-L4517) explicitly rejects FP8 for MLA and GLM5_NEXT. | Not an upstream-compatible GLM cache port. Any lower-precision latent/indexer experiment belongs to a separate numerical and consumer-kernel design. |
| Quantized state-copy byte count | [#30541](https://github.com/ggml-org/llama.cpp/pull/30541) fixes block-based copy sizing. | Correctness check only; native GLM currently copies F32 state. No speed claim. |
| Expert cache and CPU miss execution | [#30966](https://github.com/ggml-org/llama.cpp/pull/30966), [#31233](https://github.com/ggml-org/llama.cpp/pull/31233), [#31888](https://github.com/ggml-org/llama.cpp/pull/31888): benefits depend on discrete-device uploads/cache misses; selected GLM cache traces improve. | Native bank registration/selected expert buffers are not that explicit CPU-miss cache. Unified memory changes the cost model. Do not add it to accelerate a workload without the corresponding miss path. |
| Scheduler/result reuse and immutable checkpoints | [#32009](https://github.com/ggml-org/llama.cpp/pull/32009), [#32089](https://github.com/ggml-org/llama.cpp/pull/32089), [#30137](https://github.com/ggml-org/llama.cpp/pull/30137). | Inspect actual temporary owners and prefix copies if profiling points there. Multi-slot checkpoint sharing is not an immediate optimization for one retained predecessor. |
| Fused GDN / CUDA SSM scan | [#32078](https://github.com/ggml-org/llama.cpp/pull/32078), [#31818](https://github.com/ggml-org/llama.cpp/pull/31818). | GLM KDA is a different recurrence. Removing dead GDN alternatives is not a new KDA algorithm; Zamba2's CUDA d_state=64 scan result is not GLM evidence. |

## Prior GLM results and reasons not to repeat work blindly

The [2026-11-03 source-bound validation](../archive/runs/2026-11-03-glm53-prefill-merge-validation.md)
used source `cbb66f481c75e16ed8c076290392bcad35df29be`, the pinned UD-Q2_K_XL
artifact, and the same generator binary for MMA off/on. At 96 input tokens,
eight outputs and context 104 on an M5 Max/128 GiB host with macOS 26.7.1,
four accepted requests per arm gave:

| Metric | MMA off | MMA on | Historical change |
| --- | ---: | ---: | ---: |
| Prefill tok/s | 34.8958 | 40.8729 | +17.13% |
| Worker request seconds | 2.9702 | 2.5646 | -13.66% |
| Useful decode tok/s | 34.4880 | 34.3827 | -0.31% |

This was one sequential sweep, with additional requests excluded under that
study's host-screen policy. It is historical bounded evidence, not a fresh
busy-host result or a general default qualification. It already derives from
older llama.cpp Metal MMA code, with source provenance and license retained.
The [2026-11-02 record](GLM53_BATCHED_PREFILL_2026_11_02.md) remains separate.

[Batched matrix documentation](../validation/GLM_BATCHED_MATRICES.md) describes
three additional independent selectors: wide Q5/Q6 tiles, IQ expert grouping,
and batched MLA. Grouped experts support selected IQ2_XS/IQ3_XXS layouts,
up to four rows/job; unsupported formats including IQ4_XS fall back. A positive
selector is not proof of admission. Wider matrix/component results do not
establish an integrated model gain.

The [2026-11-03 batched-matrix research record](../archive/runs/2026-11-03-glm-batched-matrix-research.md)
is the relevant negative native result. With wide scheduling, batched controls,
register KDA, dense-wide, grouped experts and MLA batching together, the
routed-expert GPU-stage median was 11.21 s versus 4.12 s for the matched stage
reference. A separate v2 expert observation was 28.02 s. These are medians of
two post-warmup diagnostic rows on an unscreened host, not full-request
latencies or isolated grouped-expert effects. Ten full-vocabulary readout rows
were bit-exact in the separate bounded v1 check; lifecycle and v2 full-model
parity were not qualified. This is reason to diagnose and isolate selectors,
not evidence that every batching idea must fail.

The earlier KDA segment-tree reassociation failed its 513-token readout gate:
maximum delta about 0.6935 and relative L2 about 0.0313 against a 1e-3 limit.
Matching top-1 selections did not waive that failure. See the
[declined-candidate record](../archive/runs/2026-11-03-glm-kda-segment-tree-declined.md).
The [current register KDA path](../validation/GLM_KDA_PREFILL.md) restores
chronological arithmetic; its existence does not establish full-model speed.

## Optional EXL3/FP8 implementation

This path needs its own ranking and matched artifact. It must not inherit the
native GGUF conclusions, nor Lily's exact-rejection sampler evidence.

- Per-request prefix snapshots already copy mutable KDA/convolution state and
  open DSA-pool tails to CPU, while valid attention/indexer backing stays owned
  by the model. Existing prefix qualification is bound to its source/stage.
- The retained BF16 native profile already uses transition-weighted expert
  retention and direct reads. ClassLRU/CPU-miss ideas are conceptually closer
  here than in native GGUF, but GGUF cache slots, MLX placement and EXL3 row
  access are different. Measure actual misses/reads and dependency stalls
  before implementing another policy. Earlier frequency-first and Q8 attempts
  are EXL3-specific rejected history, not native GGUF results.
- Native MTP already exists in the staged experimental profile. The potential
  #31660 analogue is a narrow audit of per-draft conv-state capture/rollback,
  not a general request-prefix copy rewrite. It is worth inspecting only if
  that work is actually repeated and takes enough critical-path time.
- The native chosen-token/committed-round sampler differs from the native
  GGUF path and from Lily host exact rejection. Preserve its declared
  semantics and qualify acceptance/quality before claiming an MTP speedup.
- [Device-resident MTP hidden rows #31927](https://github.com/ggml-org/llama.cpp/pull/31927)
  is N/A to native GGUF without MTP. At the EXL3 boundary,
  [bridge.zig](../../vendor/glm-exl3-bridge/src/bridge.zig) calls the module's
  `mtpRound` and returns committed token IDs/counts; it does not expose hidden
  rows. The [ABI provenance](../../vendor/glm-exl3-bridge/UPSTREAM.md) describes
  separate full-F32 vocabulary readout copying. No repeated full-hidden host
  copy is visible there. Whether internal hidden rows remain device-resident
  is unknown: the pinned MLX architecture implementation is not present in the
  bridge-owned source. Inspect that exact dependency before proposing a copy
  removal; no upstream percentage is transferred.

The [retained profile record](../archive/runs/2026-11-05-glm-retained-profile.md)
reports a historical 922.66 prefill / 20.30 decode tok/s long sampled observation,
but it failed the then-used external-CPU screen, had unverified clocks, and was
not a new-build performance qualification. Its quality smoke was 22/24, with
the all-pass gate unmet. Those numbers are not compared directly with the
short greedy GGUF table. See the [bridge contract](../validation/GLM_EXL3_BACKEND.md)
for exact stage, sampling and prefix boundaries.

## Fresh controls and next matching comparison

Seven selected model-free controls passed; full-model timing is not run because
the exact weights were not found. No replacement checkpoint was substituted.

| Control | Scope | Result |
| --- | --- | --- |
| Prefill chunk limits/tails | CPU admission/scheduling | 1 passed |
| Bias changes selection but not combine probabilities | CPU routing semantics | 1 passed |
| Graph router corrected selection/probabilities | Metal routing contract | 1 passed |
| Kquant MMA pinned admission geometry | Shape/source policy | 1 passed |
| Actual MMA shapes, F64 reference and failure gates | Metal operator correctness | 1 passed |
| Register KDA ordered/parallel/extreme policies | Metal output/state bits | 1 passed |
| KDA prefix snapshot restoration | Synthetic recurrent/convolution state contract | 1 passed |

These are seven selected cases, not a complete engine suite or a timing study.
They do not prove that the full checkpoint admits a selector or produces a
noticeable gain. The engine test harness requires `nightly-2026-10-01` with
`--features cpu,metal`; worker/scoring release builds remain Rust 1.98.1.
Tests used an isolated Cargo target and serial execution.

The actual test checkout was `d29230d68f795b67c50884b905799551ed6ff2c6`, clean,
rather than the study checkout. Before treating its results as relevant,
the parent independently compared source inputs: engine tree
`a2a83cc1c70f11b072155ed74ecd609b3d2016b2`, macro tree
`84c0a62b612a597848d5e708a1d1706dff388ce5`, Cargo.lock blob
`f431c48f7f67e18ba99c1fd08b7d65df7264bd22`, and Cargo.toml blob
`2f821c683269e634fb41935dcc6d252b7225561b` match the study checkout.
The complete diff for `crates/`, `vendor/`, Cargo manifests/lock,
`rust-toolchain.toml` and `.cargo` is empty. This establishes relevant Rust
source-input equivalence, not binary identity or equivalence of the different
app/converter sources. Logs remain private and their hashes are retained.

The [component aggregate](data/glm53-llama-v1-components-2026-11-15.json)
records exact test names, source trees, toolchain and log hashes. Three cases
execute Metal graphs/kernels with admission/output assertions and no
unavailable-device early return; the other four are policy/reference controls.
The test binary SHA-256 is
`befd0ffc5ad86becceebe80786e403a71819873ffe5caa7f75da1ac33c4e60e1`.
The private receipt SHA-256 is
`62dc06dc25f01f5a0a4d9a766cc15c0b92171c42989bc2a05c151539d0648bdf`.
Toolchain: rustc `1.101.0-nightly (21b707e3f 2026-09-30)`, pinned
`nightly-2026-10-01`; host Mac17,7, macOS 26.7.2 build 25G332. No matching
`STARK_GLM53*` or `STARK_HYBRID*` environment selector was set during testing.
Test durations are not reported as operator performance measurements.

For a later native run, retain the same busy-host policy as the Lily study:
all measured requests remain included, no wait for quiet, and missing host
observations are marked unknown rather than used to discard generation.
Actual memory/shape admission and command ownership guards remain enforced.

Use a fresh source-bound generator, Rust 1.98.1, locked dependencies and a
distinct target. Keep source corpus/tokenizer, artifact hashes, sampler,
precision, context, output counts and request cadence fixed. Clear inherited
diagnostic selectors explicitly; record effective settings and dispatch
counters. A normal launch baseline must not be mislabeled as the complete
experimental wrapper, or vice versa.

Suggested bounded native matrix, after the checkpoint is available:

1. Normal code defaults versus the existing complete profile, measured as a
   whole-profile comparison. Inspect which routing/bank path is reached.
2. Within that fixed profile, MMA off/on only. Use the historical 96-input,
   8-output workload as a separate continuity cell and longer complete-request
   cells at 512/128 and 3,072/128 input/output tokens. Stay within the profile's
   4,096 context cap; copying Lily's 4,096+256 cell would exceed it.
3. Only after the reference passes, compare MLA batch and register KDA
   individually, then supported combinations. Check actual admission counters.
   Grouped expert work requires a new hypothesis explaining its earlier loss.

Use eight balanced paired observations per arm, retain all load conditions,
and require at least 10% whole-request or useful-decode improvement exceeding
the observed same-code range, at least seven/eight paired wins, a confirming
workload, and no greater than 5% median regression in another declared cell.
Use the original source's numerical/state gates, exact prefix restoration,
failure/cancellation/reset controls and artifact-specific MTP/image disposition.
For native GLM this source has no qualified native MTP execution; do not invent
MTP-on performance rows. Image N/A must follow the exact source inventory.

The after comparison must interleave fresh baseline/candidate sessions. Today's
Lily rates, old GLM tables and an unrelated future candidate run cannot form a
causal performance comparison. Keep correctness, measured speed, adoption and
installed-build status separate.

## Outcome and remaining evidence

| Item | Result / disposition |
| --- | --- |
| Native GGUF and optional EXL3 comparison | Completed source assessment, independently reviewed |
| Existing bounded T8 speed evidence | Historical only; original source/workload/host gates preserved |
| Fresh component controls | Seven passed, three with actual Metal execution; no complete-suite or full-model claim |
| Fresh full-model baseline/candidate timing | Resource-deferred: exact pinned payloads not found in checked locations |
| New performance candidate / adoption | No kernel or engine change; no performance/default promotion |
| Highest-value next work | Restore matching artifact availability, measure actual CPU-router versus GPU-bank path and existing T8 MMA/MLA prefill independently |

No fresh number in this follow-up establishes a GLM end-to-end speedup. The
evidence supports prioritizing the above measurements and rejecting several
inapplicable direct ports. The installed build is unchanged, and no earlier
GLM validation or performance record is superseded.
