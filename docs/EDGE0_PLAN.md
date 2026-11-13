# Edge0-145B native support plan

Status: phase 2 component gate passed for local non-commercial evaluation, 2026-11-12.
Compatibility attempt
[`20261112-EDGE0-01`](model-experiments/2026-11-12-edge0-compatibility.md)
records the intake and prerequisites. The
[license and provenance review](archive/runs/2026-11-12-edge0-license-intake.md)
records the owner's local non-commercial evaluation decision. The owner
authorized continuing local branch work while the attempt record remains
unmerged. Commercial/product availability is outside this scope. Source
preparation is tracked separately in
[`20261112-EDGE0-02`](model-experiments/2026-11-12-edge0-source-preparation.md).
The full pinned source set is published locally with every payload hash bound;
native readers/decoders are tracked in
[`20261112-EDGE0-04`](model-experiments/2026-11-12-edge0-block-readers.md).
No Stark model execution is qualified. Follow
[model onboarding](MODEL_ONBOARDING.md), the
[image-default parity gates](IMAGE_DEFAULT_PARITY_PLAN.md) and the
[experiment ledger](model-experiments/README.md).

## Summary

Adding Edge0-145B is feasible. The release is Qwen3.8-Flash-Next (`qwen4exp`)
with its routed experts replaced by latent experts, and Stark already runs that
trunk natively. It is a backend project, not a catalogue entry:

- the weights use a storage layout Stark does not read (a Strata pack plus a
  GGUF skeleton, with no `config.json` and no Safetensors);
- the routed feed-forward path has new arithmetic (project down, run the
  expert at latent width, project up, by expert group);
- three block formats and one table format are new to Stark;
- the upstream runtime for this model is NVIDIA-only in the public tree, so
  there is no Mac reference to compare against.

Recommended first artifact: `Edge0/Edge0-145B-q3k`. The eight binary files listed
below total 103,374,417,600 bytes (96.27 GiB), of which 51.2 GB is the n-gram
table that is read from disk by row. This subtotal includes `labels.bin` and
excludes the small text, tokenizer and license files.

## Pinned intake

| Source | Revision | Notes |
| --- | --- | --- |
| [`Edge0/Edge0-145B-q3k`](https://huggingface.co/Edge0/Edge0-145B-q3k/tree/d7de835cfce7d9bb22a3d5ebb7d1e8565d4ecf74) | `d7de835cfce7d9bb22a3d5ebb7d1e8565d4ecf74` | Q3_K experts; the build upstream measured on Metal |
| [`Edge0/Edge0-145B-NVFP4`](https://huggingface.co/Edge0/Edge0-145B-NVFP4/tree/6aab5f4337b4c7146dd69130dad4c719f83fca03) | `6aab5f4337b4c7146dd69130dad4c719f83fca03` | NVFP4 experts; same skeleton, table and MTP sizes |
| [`Edge0-AI/Edge0`](https://github.com/Edge0-AI/Edge0/tree/b7900cbb322926ccebdcd0c0c5a1280497ecff22) | `b7900cbb322926ccebdcd0c0c5a1280497ecff22` | Apache-2.0; `edge0-strata/` is an MIT fork of Niko1221/Strata (import `f1933b1e4993fecc10d2e9360d85569619a2c42d`, merge base `c62d5e3a15b40ea51c96aae9e8d36928965f53ab`); ggml from llama.cpp `3cf03257f219afbe7334045ff7c6a06ac68c627d` |

Files of the Q3_K folder needed by the default configuration, with the SHA-256
the Hub reports for each:

| File | Bytes | SHA-256 | Content |
| --- | ---: | --- | --- |
| `experts.bin` | 40,076,574,720 | `947d15f3d69ecdb45d602a62bc84803451da779a22294e1db5b3568290b42879` | 48 x 512 routed expert records |
| `ple-fp8.gguf` | 51,200,246,112 | `c7615c59b060277ddf6f84729fa8f7469cfc82c7fb983d6e7eab5a2f7446f170` | N-gram embedding table |
| `model-skeleton.gguf` | 5,991,569,248 | `ab49fbe7590cdcda45d7a2bcae4be0bd42c6a70b951d594a00eb158d7df0b72f` | Attention, recurrence, routers, shared experts, embeddings, readout |
| `latent_dense_q8.bin` | 3,743,416,320 | `fbb25caaa0dcf45ef62f1a4945dcf98a53a166934a3b461b166fbafa831d87ca` | Per-layer down/up projections |
| `dense.bin` | 1,538,625,024 | `f9bcc7f36eac4e59a7b1cb1962e038ba12dc831a41fb193761f1d1e72657fb62` | Pack pool for floating tensors and expert scale vectors |
| `mtp/experts.bin` | 707,788,800 | `09398406be61f1f54c93861f449e48b8df0bfccbc9ec9b2b7636775a6ea9244f` | Draft-layer expert bank |
| `mtp/dense.bin` | 116,099,072 | `c724dc0b0822ada5d2977bf5bde821605feabaa64ea2e0045b67ca656329070a` | Draft-layer dense tensors |
| `labels.bin` | 98,304 | `ce9b4a20b92c5c866e65d84a3884c104986522471cf11cc5f797a153affbcc06` | Projection group of each expert |

Small files (`index.txt`, `native_experts.txt`, `model.json`,
`mtp/dense.txt`, `scales/l0.f32` to `l47.f32`, `tokenizer/`, `LICENSE`,
`NOTICE`, `release-scope.json`, `third-party/Qwen-LICENSE.txt`) are bound by
the repository revision. The preparation helper now pins all 69 default source
paths, sizes and SHA-256 values in
[`edge0_source.py`](../tools/converter/stark_convert/edge0_source.py): eight
binary payloads plus 61 small files, 103,384,750,077 bytes total. Those identities
do not establish a complete local download or native load. `latent_dense_q4.bin`
(1,981,808,640 bytes) is
only needed for the optional Q4 projection choice and is not part of the
default set.

Chat template SHA-256: `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`.
It is byte-identical to the template in the two Flash-Next folders installed on
the intake host (Lily Q4-XL `0f94842e` and MLX-Serve mixed `788c07a8`).

## What the release contains

Read from the skeleton GGUF header, the pack manifests and the pinned source.

| Item | Value | Source |
| --- | --- | --- |
| Architecture | `qwen4exp`, 48 layers, width 2560, vocabulary 248,320, context 262,144 | GGUF metadata |
| Layer mix | 36 recurrent (Gated DeltaNet: state 128, 16 groups, conv kernel 4, inner 6144) and 12 indexed-attention layers, one in every four | GGUF metadata |
| Attention | 24 heads, 2 KV heads, key/value length 256; indexer 4 heads, key length 128, top-k 2048 | GGUF metadata |
| Hyper-connections | 4 streams, low rank 320 | GGUF metadata |
| N-gram embedding | Trigram, 8 heads per n-gram, 16 rows of 160 values per token, 320,001,536 rows, conv kernel 4, layer 1 | GGUF metadata, table header |
| Routing | 512 experts per layer, 10 routed plus 1 shared, expert width 640 | GGUF metadata, `model.json` |
| Latent experts | Input compressed from 2560 to 1792; 8 projection groups per layer; each expert carries one group label | `docs/models/edge0-145b.md`, `latent_moe.hpp` |
| Expert record | 1,630,720 bytes: 640 gate rows and 640 up rows in Q3_K over 1792 inputs, then 1792 down rows in IQ4_NL over 640 inputs | `native_experts.txt`; sizes recomputed from the block formats |
| Projections | Per layer `D` `[8*1792, 2560]` then `U` `[8*2560, 1792]`, Q8_0 rows; 77,987,840 bytes per layer | `latent_dense.hpp`; file size matches exactly |
| Scales | One 1792-value F32 column-scale vector per layer for Q3_K packs; per-expert `scale` and `input_scale` vectors for gate, up and down in the pack pool | `native_expert.hpp`, `index.txt` |
| Skeleton storage | 1,079 tensors: 302 Q8_0, 484 BF16, 292 F32, 1 F16 | GGUF tensor table |
| Table storage | One I8 tensor `[160, 320001536]`, `strata.ple.format = f8_e4m3`, global scale `0.00019931793212890625` | GGUF header |
| Draft layer | One Flash-Next MTP layer; dense tensors in Q8_0, BF16 and F32; a 512-expert bank of 1,382,400-byte records, which matches the fused 1280x2560 plus 2560x640 layout at Q2_0 | `mtp/dense.txt`, `tools/mtp_pack.py`; expert format inferred from size, to be confirmed |
| Tokenizer | GPT-2 BPE with the `qwen35` pre-tokenizer, as `vocab.json`, `merges.txt` and GGUF arrays; `tokenizer/tokenizer.json` is a 541-byte descriptor, not a Hugging Face tokenizer file | Repository files |
| Sampling defaults | top-k 20, top-p 0.95, temperature 1.0 | GGUF metadata |
| Vision | No vision or projector tensors among the 1,079 skeleton tensors and no projector file; `pipeline_tag: text-generation` | GGUF tensor table, file list |

The latent layer, as implemented in the pinned
`edge0-strata/src/latent/latent_moe.cpp`:

1. `xl[t, g] = D_g x[t]` for the projection groups.
2. For each routed expert `e` of token `t`, the expert reads
   `xl[t, label[e]]` and produces a latent output
   `down(silu(gate(a)) * up(a))`.
3. `y = U_{label[e]}` applied to that latent output.
4. The layer output is the router-weighted sum of the ten `y` rows, plus the
   shared expert at full width as in Flash-Next.

Upstream quantizes activations to 8-bit blocks before each of the `D`, expert
and `U` products and uses integer dot products.

## Where the public sources disagree with the announcement

- **Mac availability.** The announcement names Apple M-series as a day-one
  platform. The repository documents Edge0-145B as "Linux or Windows with an
  NVIDIA GPU and CUDA 13", states that its Apple M2 Ultra measurements "do not
  establish macOS release availability", and lists upstream Strata's `metal/`
  engine among the directories removed from the fork. `edge0-infer`'s own GGML
  Metal engine covers only the 35B model.
- **Lineage.** The announcement describes a model trained for devices from the
  outset. The model card, `model.json`, `release-scope.json` and the model page
  identify Qwen3.8-Flash-Next as the base. `release-scope.json` also states
  that no training-source revision was verified.
- **Active parameters.** `model.json` says 6.6 B; the model card says 10.12 B.
- **Repeatability.** `edge0-infer/docs/edge0-145b.md` states that identical
  requests at temperature 0 can produce different answers on the upstream
  engine.

Published scores and throughput are Edge0's own measurements. They are context
and do not qualify Stark.

## Mapping to Stark

| Component | Stark today | Gap |
| --- | --- | --- |
| Trunk graph | `qwen4_exp` in the F32 graph (`crates/stark-engine/src/engine/hybrid/qwen.rs`) and in the mixed-precision backend (`vendor/lily-core/src/qwen4exp/`): Gated DeltaNet, sparse attention with indexer, hyper-connections, n-gram embedding and convolution, routed and shared experts | Field-by-field geometry check against the GGUF metadata; adapter from llama.cpp `blk.N.*` tensor names |
| Trunk storage | MLX affine 4/8-bit Safetensors and typed Uzu artifacts | GGUF v3 with Q8_0, BF16, F32 and F16. The shared `Weights` reader decodes Q8_0 for GLM; the mixed-precision backend has affine kernels only |
| Routed experts | Full-width experts in resident affine Q4 banks | Latent projections by group label, fixed Q3_K plus IQ4_NL records, column and per-expert scales |
| Block decoders | Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ4_XS | IQ4_NL (its codebook is already present for IQ4_XS), Q2_0 (Strata's 64-value 2-bit type), FP8 E4M3 rows, and Q4_0 for the 120B projections |
| N-gram table | Paged Q4 group-32 table and indexed affine ranges | 51.2 GB FP8 table inside a GGUF, with a global scale |
| MTP | Flash-Next one-layer head, one to three drafts, exact rejection sampling | Head stored as a Strata pack; upstream runs four drafts with a restricted draft vocabulary |
| Prefix reuse | Mixed-precision backend: snapshots on a 4,096-token grid. F32 graph: fresh per request | Same state set expected; needs its own inventory and artifact-bound run |
| Tokenizer and template | Hugging Face `tokenizer.json` and a reviewed template | Template identical to Flash-Next. Tokenizer equivalence is unproven: the shipped `vocab.json` and `merges.txt` hashes differ from the installed Flash-Next files |
| Images | Retained BF16 tower for Flash-Next | No tower in this release; N/A record required |
| Catalogue | Discovery through `config.json` | No `config.json`; needs a pinned native entry like GLM |

## Decisions

1. **Artifact.** Start with `Edge0-145B-q3k`. Stark already decodes Q3_K, and
   it is the build upstream ran on Metal. NVFP4 needs FP4 block kernels with no
   benefit on Apple hardware; list it as unavailable with a reason.
2. **Native backend, not Edge0's framework.** For this model `edge0-infer`
   starts `edge0-strata` as a child process, and that engine builds only for
   CUDA at the pinned revision, so there is nothing to run on a Mac. Stark's
   execution path also has no external inference process, and its prefix
   cache, MTP and readiness contracts belong to the worker. The Edge0
   repository is used as the format authority and as oracle source.
3. **Two stages.** Bring the model up first in the F32 graph, which already
   reads GGUF block formats through the shared reader. Then move it to the
   mixed-precision backend for batched prefill, MTP and prefix reuse. If only
   one path is affordable, go straight to the second stage and accept that the
   synthetic fixture is the only reference.
4. **Bytes are retained.** No requantization and no repacking of expert,
   projection, table or skeleton payloads. Q8_0 scales are F16 and are not
   re-encoded as BF16 affine planes.
5. **Activation precision is a measured choice.** Stark uses floating
   activations where upstream uses 8-bit activation blocks. The difference is
   measured before a policy is recorded.

## Work

Each phase ends at a gate. A later phase does not start on an unmet gate.

### Phase 0: records, provenance and license

- Attempt [`20261112-EDGE0-01`](model-experiments/2026-11-12-edge0-compatibility.md)
  (compatibility) is opened from the template and added to the ledger on the
  work branch. Later phases get their own IDs.
- The pinned `Edge0-AI/Edge0` revision is recorded in
  [`provenance/UPSTREAM.md`](../provenance/UPSTREAM.md).
  If any source, table or layout is ported, add the Apache-2.0 and MIT license
  texts beside the existing reference licenses and state the patch provenance.
- Owner decision on the model license. The weights are under the Edge0
  Community Model License 1.0 for Edge0's contributions and the Qwen Community
  License 1.0 for upstream material. Section 4 of the Edge0 license reserves
  model sales, commercial model services and commercial independent coding or
  office assistants, including on-device products. The Qwen license has its own
  commercial permission conditions. Keeping weights out of Stark's distribution
  does not resolve these use restrictions. The prepared folder must keep
  `LICENSE`, `NOTICE`, `release-scope.json` and the Qwen license text, and any
  future catalogue row must link the license. The license intake records the
  exact terms and the owner's local non-commercial evaluation decision.

Gate: attempt record merged; license decision recorded. On 2026-11-12 the owner
selected local non-commercial evaluation only and instructed work to continue
on this branch while the record remains unmerged. This authorizes the local
evaluation sequence; merge, product admission and any commercial permissions
remain separate. The
[license intake](archive/runs/2026-11-12-edge0-license-intake.md#owner-decision-2026-11-12)
records this scope. Phase 1 completed under that instruction.

### Phase 1: source preparation and admission

- Add a pinned choice `edge0-145b-q3k` to `stark_convert.hybrid_prepare` with
  the revision, file list, sizes and SHA-256 values above. Skip assets, launch
  scripts and the Q4 projection file. Reuse the existing resumable staging and
  atomic publication.
- Check before publication: skeleton metadata equals the expected geometry;
  `index.txt` and `native_experts.txt` agree with the file sizes; labels are in
  `0..7`; projection and table sizes equal their formulas; tensors present in
  both the skeleton and the pack pool are byte-equal.
- Write a Stark `config.json` with `model_type: qwen4_exp` and an explicit
  storage block naming this layout, so the existing Lily and external-affine
  admission predicates cannot match it by accident. Write the
  `native-source.json` receipt with native loading reported as unverified.
- Tokenizer, borrowed-reference route (`20261112-EDGE0-02`): derive
  `tokenizer.json` from the pinned Qwen3.8-Flash-Next tokenizer only after proving
  that token IDs, merge order, token types, special IDs and normalization/split
  semantics equal the shipped files and GGUF arrays. Any difference fails this
  route. The pinned release fails this check; retain that result.
- Tokenizer, source-native successor
  ([`20261112-EDGE0-03`](model-experiments/2026-11-12-edge0-native-tokenizer.md)):
  independently represent the released vocabulary, merges, token classes and
  pinned `qwen35` semantics. Qualify it against a tokenizer-only CPU oracle
  from Strata's exact llama.cpp revision before permitting preparation. Require
  exact encode/decode and special-token behavior on the declared controls,
  including Unicode normalization/case-fold boundaries and unused padding.
  An unavailable oracle or mismatch keeps this route disabled. Bind source,
  program, fixture and result hashes under `derived_metadata`; do not substitute
  a modified borrowed tokenizer and relabel the rejected comparison as passing.

Gate: a prepared folder on the intake host whose receipt binds every payload
hash. Binary payloads total 103.4 GB, plus the small metadata files; the host had
317 GiB free at the original intake. Recheck space before download.

Local gate passed: the 69 pinned source files were fully verified and published
with receipt SHA-256
`8dc700fe5cd5f6644ed9f062eb391ce32cb0a09b76a2e19ec44539bf89200a5e`.
See the [preparation record](model-experiments/2026-11-12-edge0-source-preparation.md#published-local-source-phase-1-passed)
for the tokenizer successor, rejected runs and stored-row correction. Normal
native identity/import remains disabled.

### Phase 2: readers and block decoders, without a model

- Readers for the pack index (v3), the native-expert manifest (v6), labels,
  projections and column scales.
- Scalar decoders for IQ4_NL, Q4_0, Q2_0 and FP8 E4M3, with Metal kernels where
  the forward path needs them. Compare against hashes from the pinned ggml C
  decoder, and for the Strata-specific formats against the pinned
  `ple_fp8_parity.cpp` and `tools/mtp_pack.py` definitions.
- Trace and document the scale conventions from `tools/iq_pack.py`,
  `native_expert.cpp` and the Q3_K pack tool: whether the column-scale file
  holds scales or inverse scales, and where per-expert `scale` and
  `input_scale` apply.

Gate: the existing `hybrid::` operator suite passes with the new cases.

Local gate passed in [attempt `20261112-EDGE0-04`](model-experiments/2026-11-12-edge0-block-readers.md):
254 hybrid tests passed with 61 ignored, plus the full native engine and
independent scalar/Metal checks. Readers and decoders enable no backend or
native model admission. The record preserves the unavailable sealed scoring
replay and remaining runtime qualification.

### Phase 3: reference forward in the F32 graph

- Tensor-name adapter, Q8_0 embedding and readout, FP8 table gather, and a
  staged latent stage: projections for the groups in use, selected experts
  through the bounded selected mappings, up-projection and the weighted sum in
  routed order.
- `scripts/edge0_native_fixture.py`: a small synthetic pack with reference
  logits from an independent F64 forward written from the pinned source. If a
  CPU-only build of the pinned Strata verification tools works on macOS arm64,
  use it as a second oracle; record the outcome either way.
- Activation control: floating activations against emulated 8-bit activation
  blocks, on the fixture and on one bounded real layer.
- First full-checkpoint smoke: load, short greedy and seeded stochastic
  generation, state reset. Aggregates and hashes only.

Gate: fixture parity within thresholds fixed in the attempt record; bounded
ordinary-chat smoke passes on the real artifact. `prefix_cache.supported` stays
false and MTP stays unavailable on this path.

### Phase 4: mixed-precision Metal path

- Q8_0 trunk kernels for decode and batched prefill.
- Latent expert kernels: fused projection, gate/up, SwiGLU, down and
  up-projection for decode; prefill grouped by expert. Summing latent outputs
  per group before the up-projection changes the reduction order and is a
  separate, gated candidate.
- Resident banks through the existing registry. Experts are 37.3 GiB and
  projections 3.5 GiB. The listed binaries other than the n-gram table total
  52,174,171,488 bytes (48.59 GiB), including labels and excluding the small
  metadata files.
- Loaded-state memory estimate for automatic planning, and a startup recipe.

Gate: agreement with the phase 3 reference under the recorded numerical gates,
plus matched decode and prefill measurements.

### Phase 5: MTP

- Load the draft layer from `mtp/` and bind it to the existing Flash-Next head
  graph; add the Q2_0 expert kernel once the bank format is confirmed.
- Keep Stark's exact rejection-sampling verification with one to three drafts.
  Four drafts and the restricted draft vocabulary are separate attempts.
- Choose the default by a matched on/off measurement and keep the request
  override.

### Phase 6: prefix reuse

- Complete the state inventory below against the implemented path.
- Reuse the 4,096-token snapshot path of the mixed-precision backend.
- Run `scripts/validate_generation_prefix_cache.py --reuse boundary` for
  exactness and timing on the exact artifact. The validator must learn this
  file set before it can hash it.

| State | Expected for Edge0-145B | Restoration |
| --- | --- | --- |
| Recurrent state and convolution windows, 36 layers | As Flash-Next | Snapshot at the grid boundary |
| Attention K/V and indexer caches, 12 layers | As Flash-Next | Completed prefix rows retained; incomplete pool snapshotted |
| N-gram history and its convolution window | As Flash-Next; table rows are stateless lookups | Snapshot |
| Latent projections, experts, labels and scales | Stateless | None |
| Draft-layer state | As the Flash-Next head | Snapshot, keyed by MTP selection |
| Sampler and scratch | Per request | Reseeded and overwritten |

These are expectations from the Flash-Next inventory, not findings.

### Phase 7: tools, images, admission and catalogue

- Tools: the identical template makes parser reuse a protocol claim only. Run
  `scripts/validate_tool_calling.py` on the exact artifact after extending its
  hashing to the pack files, and record the result separately.
- Images: record N/A with the inventory evidence above. The base family is
  image-capable; pairing a Flash-Next tower with this trunk is unqualified and
  out of scope.
- Memory admission and Beginner recommendations from measured footprints. The
  minimum supported memory is decided by measurement, not by the announced
  48 GB figure.
- Catalogue: one pinned text-only entry for the Q3_K build with its license
  link. NVFP4 rows are unavailable with a reason.

### Phase 8: performance qualification and reference pages

- Matched cold and warm runs, MTP on and off, under `docs/benchmarks/`.
- Update `NATIVE_HYBRID_MODELS.md`, `MODEL_SUPPORT.md`,
  `MODEL_IMAGE_STATUS.md`, `TOOL_CALLING.md` and `MODEL_RECOMMENDATIONS.md`
  only where the evidence supports a capability change.

### Follow-on artifacts, each qualified separately

- `Edge0-120B-q3k`: latent width 1280, 1,164,800-byte records, Q4_0
  projections by default.
- NVFP4 builds.
- Hosts below 64 GiB, where experts cannot all stay resident.
- Edge0 35B and 8B use different architectures and `edge0-infer`'s own engine.
  They are not covered here.

## Risks and open questions

1. No public Mac reference exists for the latent path, and the upstream engine
   is not bit-repeatable. Qualification rests on Stark's own oracle and
   tolerance gates.
2. The model is described as trained at 4 bits with quantized activations in
   the upstream arithmetic. Floating activations may differ from the trained
   forward.
3. The scale conventions and the draft-layer expert format are not yet traced
   to source.
4. Tokenizer equivalence with Flash-Next is unproven.
5. The release was published on the intake date. Revisions may change; every
   step binds the pinned ones.
6. The intake host has 128 GiB. It cannot qualify low-memory behaviour without
   a separate constrained study.
