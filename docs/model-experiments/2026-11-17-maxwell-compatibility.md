# 20261117-MAXWELL-01: Maxwell V1 Flash compatibility prototype

## Identity and decision

| Field | Value |
| --- | --- |
| Registered / executed / owner | 2026-11-17 / 2026-11-17 / Codex; initial tensor naming assumptions recorded retrospectively below |
| Kind / state | Compatibility, memory and precision inventory / completed within metadata scope; native execution blocked |
| Acceptance | Account for every indexed shard and tensor; check backbone attention, mHC and FFN geometry, router dtype, MTP presence and source image evidence; exercise actual admission; identify concrete reuse and blockers |
| Evidence scope | Header structure, static source review and synthetic equation controls; no full-model numerical, lifecycle, quality or timing evidence |
| Promotion / disposition | Diagnostic only / experiment on branch; Maxwell remains unsupported, prefix reuse and MTP unimplemented |
| Stark baseline | `fb7b6771576a89d11f5409f0f23745064f5cc3b2` (origin/main when worktree created) |
| Branch / predecessor / PR | `codex/maxwell-compatibility-prototype` / none / no PR requested or opened |

**Conclusion:** a dedicated Maxwell backend is plausible. Reuse GLM's small-matrix
Sinkhorn/mixing machinery and biased sigmoid router after adapting semantics.
Implement a new block-sparse GQA path and complete its indexer-cache lifecycle.
Do not alias `maxwell_moe` to Qwen or GLM, substitute dense attention at long
contexts, or infer numerical qualification from this probe.

## Source and reproducibility

The checkpoint is [maxwell-model/Maxwell-V1-Flash](https://huggingface.co/maxwell-model/Maxwell-V1-Flash/tree/ca536c4380e0600d6aebdb808d2ce65ec724161b),
revision `ca536c4380e0600d6aebdb808d2ce65ec724161b`. Source code, config, template,
generation metadata, tokenizer config, index and license are pinned to that
revision. See the [public aggregate receipt](../archive/runs/data/2026-11-17-maxwell-compatibility.json)
for their SHA-256 identities and the shard-header manifest hash.

The upstream license is Apache-2.0; its file hash is in the receipt. No upstream
source files or weights are vendored or patched. The probe independently
transcribes a few mathematical equations for synthetic controls; it never imports
or executes the remote Python. Any later upstream-derived implementation needs
copyright/license preservation and its own patch provenance.

```sh
uv run --project tools/converter --locked --python 3.13.13 python \
  scripts/probe_maxwell_compatibility.py \
  --output runtime/maxwell-compatibility-reviewed-20261117
uv run --project tools/converter --locked --python 3.13.13 pytest \
  scripts/test_probe_maxwell_compatibility.py -q
```

The probe downloads small metadata files and two exact HTTP byte ranges per
shard (length prefix and JSON header). It refuses range-ignoring servers before
reading their response body. Header reads are capped at 16 MiB each, with four
concurrent readers and 60-second request timeouts. It checks tensor/index
ownership, complete coverage, contiguous offsets, dtype byte counts and the
index's aggregate size. This validates declared layout, not actual payload bytes,
whole-shard file integrity or model execution. Raw metadata stays in ignored
`runtime/`; public evidence contains aggregate results and hashes only.

Environment: macOS 26.7.2 (25G332), Mac17,7, 128 GiB physical RAM; managed Python
3.13.13 through uv, converter dependencies locked. Lock SHA-256:
`df7711726b1ed7e235ded21b9e6e1cbe130924be4d2c580d5d6bd9005248f8d5`.
No native executable built; Rust/Metal versions, executable hash, peak RSS,
available headroom and paging were not measured for this metadata-only run.

## Checkpoint inventory and attempts

| Trial | Observation | Disposition |
| --- | --- | --- |
| Initial schema probe | Complete 35-shard coverage; 135/222 checks passed, 87 FFN checks failed because the probe assumed `gate/up/down_proj` and `.mlp` names | Invalid naming assumptions, not a defective checkpoint; preserved private initial receipt, same header manifest as final run |
| Corrected schema probe | Source and headers use `.block_sparse_moe.w1/w3/w2`, including dense layers; all 222 grouped checks passed | Structural compatibility inventory only |
| Actual converter admission | `preflight_model` returns `unsupported_architecture` for text, decision and vision-language | Expected native-onboarding blocker; no admission bypass |
| Synthetic equation study | Four seeded cases: Sinkhorn operation sequence has zero float64 difference; no-tie biased selection order matches 4/4; normalization factors differ | Equations only, not calls into Rust, Metal or PyTorch; no precision qualification |

The initial and corrected run consumed the same 4,466,328 header bytes. All
34,270 indexed tensors were accounted for: 36,810,548,414 BF16 elements plus
11,008 F32 elements, totaling 73,621,140,860 declared payload bytes. The F32
elements are the correction biases for 42 backbone MoE layers and one MTP MoE
layer (43 × 256). Preserve their precision during loading/quantization.

Backbone geometry: 45 layers, hidden width 2,048, 32 query heads, four KV heads,
head width 128, four residual streams, 20 Sinkhorn iterations. Three dense FFNs
precede 42 MoE layers: 256 experts, top 12 selected, expert width 512. The source
represents the advertised two shared experts as one ungated width-1,024 shared
FFN. Geometry checks cover every routed expert's three matrices in all 42 layers,
attention/indexer projections and mHC projection geometry. They do not validate
every auxiliary tensor's semantics or the completeness of the MTP decoder.

## Reuse map and concrete blockers

| Area | Existing code | Reuse assessment / required change |
| --- | --- | --- |
| Admission and loading | `tools/converter/stark_convert/model_support.py:313`; `crates/stark-engine/src/engine/hybrid/mod.rs` recognition/dispatch; `hybrid/weights.rs` container readers | Safetensors support does not admit Maxwell. Add a bounded architecture-specific config, tensor mapping, loader and worker adapter; preserve source identities. |
| SwiGLU and experts | `hybrid/glm.rs` dense/expert execution and `hybrid/glm_metal.rs` projections | Candidate building blocks. Map w1=gate, w3=up, w2=down; repack per-expert tensors for selected expert execution. Validate unclipped SiLU and ungated shared contribution; do not inherit GLM clipping/scaling defaults. |
| Router | `hybrid/glm.rs:2459` route; `hybrid/glm_metal.rs:1006` | Closest match: sigmoid, correction bias for selection only, normalize original selected scores with 1e-20. Maxwell has no GLM routed multiplier; use scale 1. Qualify ties, F32 accumulation, BF16 combine boundaries and all 12 selected experts. Qwen's logits/softmax router is different. |
| Hyperconnections | `hybrid/glm.rs:1890` Sinkhorn, `:1922` pre, `:1962` post; Metal counterparts | Four-stream mixing and 20-step normalization are reusable primitives. Maxwell normalizes with `1/(sqrt(mean(x²))+1e-6)` before applying learned projected gates, while GLM uses `1/sqrt(mean(x²)+rms_eps)`. Maxwell projects unnormalized F32 input then scales by its reciprocal; GLM projects normalized input, so reduction/rounding also needs qualification. Add a Maxwell-specific pre-normalization path and tensor mapping. |
| Output contraction | Maxwell `HCHead` in pinned source | Learned per-stream gating must be preserved. Do not replace with the unused simple mean-contraction helper. Validate head normalization separately from per-layer mHC. |
| Attention and selection | Existing GQA, RoPE, normalization and top-k operators; GLM MLA and Qwen pooled selectors | New architecture-specific path required: post-RoPE index Q/K, per-query/per-KV-group token scores, max score within each 128-token block, 16 eligible remote blocks, eight prefix and eight block-aligned local blocks, causal masking/deduplication and sparse GQA. Existing pooled/token selectors implement different algorithms. |
| Precision and quantization | BF16/reference kernels, quantized expert banks | No Maxwell quantization qualified. Preserve F32 biases and source F32 gate/indexer arithmetic; assess other precision choices numerically. Separate every new format/quantization's receipts. |
| Chat/tools | Source template with Maxwell role delimiters, thinking and XML function calls | Similar XML syntax is insufficient for parser admission. Review/render the exact template, typed arguments, retained reasoning, tool results, stops and continuation with source-bound tests before registering a protocol. No tool behavior qualified. |

Paths abbreviated `hybrid/` above are under
`crates/stark-engine/src/engine/hybrid/`. References describe the pinned Stark
baseline, not stable line numbers for future edits.

The synthetic normalization study uses seed 20261117, 8,192 channels, and input
scales 1e-5, 1e-3, 1 and 100. Maximum absolute normalized-input differences are
approximately 3.7904, 1.0445, 2.01e-6 and 4.14e-8. This establishes a concrete
counterexample to direct normalization substitution; it does not measure
checkpoint error. The sparse-selection control shows max pooling and mean
pooling can choose different blocks. Router controls assess selection order
only, excluding ties and combination weights.

## Prefix reuse, MTP and image scope

The backbone needs full attention K/V **and** the historical post-RoPE indexer K
for every layer. The upstream custom cache refuses nonempty legacy-cache restore
because it loses indexer K. Exact prefix restoration appears possible if both
stores and their physical positions/lengths are cropped together, but chunk
boundaries and reduction order need proof and artifact-bound tests. Do not
truncate history to the selected 4,096 attention tokens: future remote selection
can reference any historical block. The 4,096-token support cap limits attended
tokens, not retained history or indexer scanning cost.

Concrete current blocker: no Maxwell backend owns, rewinds or clears this
additional indexer state. Prefix support must remain disabled until its state
hooks cover successful publication, context/scope/options mismatch, divergence,
cancellation, errors, images and restart. Then run the onboarding fresh-versus-
reuse suite with greedy and seeded sampling and expected reused counts.

The config declares one MTP layer, with 799 tensors actually present. Its
decoder uses ordinary GQA without indexer weights, plus MoE and hyperconnections,
embedding/hidden projections, norms and a learned contraction head. The released
forward exposes auxiliary MTP logits using rolled inputs; a usable incremental
speculative decoder still needs draft state, verification and rollback design.
Preserve those weights in preparation. MTP is unimplemented and unqualified;
choose its eventual default by matched performance measurements.

**Image N/A for this source:** `MaxwellMoeForCausalLM` has no vision configuration,
image encoder/processor or image-feature input path, and the complete header
inventory has no vision/visual/image/projector tensors. Vision placeholders in
the Jinja template do not establish image capability. No image-enabled artifact
was created; a future image checkpoint needs separate onboarding.

## Memory, validation and next decision

Declared BF16 weights occupy approximately 68.56 GiB. At batch one, BF16 backbone
K/V plus indexer K require `45 × (2 × 4 × 128 + 128) × 2 = 103,680` bytes/token:
25.3125 GiB at 262,144 tokens. Weight plus cache payload is approximately
93.88 GiB, excluding MTP caches, activations, scratch, transient copies,
allocator/driver overhead and other host allocations. These are analytical byte
counts, not measured allocations or a safe context allowance. Four-bit weights
would ideally occupy roughly 17.14 GiB before metadata and retained high-precision
tensors; no quantized Maxwell artifact was produced or validated. Config capacity
524,288 is not qualification beyond the advertised 256K.

Validation: 14 focused model-free tests passed; Ruff lint/format and repository
policy checks passed. Tests include range refusal, wrong cached ranges, invalid
total length, truncated/oversized range bodies, bounded header sizes, missing or
overlapping tensors, ownership/byte mismatches, memory units and unbounded
geometry refusal. No engine, worker or converter runtime implementation changed,
so native builds, inference, full-model parity, cache lifecycle, agent workflows,
quality, prefill/decode timing and installation were not run or measured.

Recommended next implementation gate: preserve this source and build a
Maxwell-specific CPU reference plus small native component controls for the
normalization, sparse route and cache crop semantics. Only then wire the native
backend, prepare the actual artifact and qualify numerical/cache/tool behavior.
Quantization and optimized sparse Metal execution remain separate gates. The
earlier 1–2 week initial-native / 3–6 week qualified-port estimate remains a rough
planning range; this probe supplies no measured throughput or completion guarantee.
