# 20261112-EDGE0-04: native Edge0 readers and block decoders

## Identity and state

| Field | Value |
| --- | --- |
| Registered / owner | 2026-11-12 / Codex; local non-commercial evaluation only |
| Kind / scope | Phase 2 compatibility and component correctness; native readers and scalar/block operators |
| State / result | Completed within native reader/block component scope; declared local Phase 2 gate passed |
| Promotion | No Edge0 backend, catalogue entry or inference capability enabled by this phase |
| Baseline | `8ab1067e`, completed local source preparation |
| Predecessor | [20261112-EDGE0-02](2026-11-12-edge0-source-preparation.md#published-local-source-phase-1-passed) |
| Governing plan | [Phase 2](../EDGE0_PLAN.md#phase-2-readers-and-block-decoders-without-a-model) |

The Phase 1 local gate passed with all 69 source hashes and atomic publication.
Prepared receipt SHA-256:
`8dc700fe5cd5f6644ed9f062eb391ce32cb0a09b76a2e19ec44539bf89200a5e`.
The phase begins only after that publication. Phase 0 merge remains separate
under the owner's explicit continued-local-work instruction.

## Hypothesis and change

Stark can read the pinned Strata records and decode their block values without
repacking payloads or loading/executing the model. Native metadata readers must
preserve stored/logical geometry and scale conventions. New scalar formats
must agree with independently compiled pinned source references before a
forward graph consumes them.

Implement bounded native readers for pack v3, expert v6, labels, Q8 projections
and column scales. Add IQ4_NL, Q4_0, Strata Q2_0 and FP8 E4M3 scalar decoders,
with Metal decode support where the next forward path requires it. Retain
existing formats and the F32 reference policy. Trace the placement/direction
of column and per-expert scales from source; do not infer it from their names.

## Artifact, sources and environment

- Sole checkpoint target remains
  `Edge0/Edge0-145B-q3k@d7de835cfce7d9bb22a3d5ebb7d1e8565d4ecf74`.
  Source preparation is complete; no model execution has occurred.
- Format authority: `Edge0-AI/Edge0@b7900cbb322926ccebdcd0c0c5a1280497ecff22`.
  GGML decoder authority: llama.cpp
  `3cf03257f219afbe7334045ff7c6a06ac68c627d`. Preserve source path/hash,
  license and port provenance separately from older native validation.
- Reference block fixtures are synthetic, fixed before comparison, unrelated
  to model weights, prompts or inference vectors. Keep raw native captures
  private/ignored; publish aggregate counts and hashes.
- Rust unit builds use pinned `nightly-2026-10-01` and locked dependencies.
  Scoring/worker builds, if later required, use 1.98.1. Use this worktree's
  distinct target directory and serialize native Metal execution.

## Fixed gates and controls

1. Native readers reject unsupported versions/types, unsafe or overflowing
   offsets, inconsistent counts/shapes, out-of-range labels/scales and truncated
   inputs. Preserve 320,001,536 physical versus 320,001,446 logical PLE rows.
   Exercise positive synthetic records and mutations of each invariant.
2. Fix synthetic block cases, source/compiler identities and fixture hashes
   before comparing. Finite scalar dequantization must match the pinned C/source
   F32 reference bit for bit. For nonfinite encodings compare the declared
   classification explicitly; never count a matching error as numerical parity.
3. Cover all IQ4_NL/Q4_0 code positions, every FP8 byte, signed zeros,
   subnormal/normal scale boundaries and multiple Strata Q2_0 bit planes.
   Validate size/stride and malformed-buffer rejection. Source-specific Q2_0
   and FP8 conventions need their own independent authority.
4. Any Metal decoder must match its independently qualified scalar values on
   the declared finite cases; fix metrics/thresholds before running. Dot-product
   accumulation and full-layer activation precision are later gates and do not
   inherit this scalar proof.
5. Run the existing `hybrid::` operator suite with the new cases and the
   applicable repository checks. A compile/setup/resource failure remains
   separate from numerical failure. No loaded-model or speed claim follows.

## Workload and state boundaries

No model forward, sampling, context/output budget, MTP drafts or performance
workload is part of this phase. The new storage operators are stateless.
Backend prefix restoration remains unimplemented/unqualified for Edge0; no
cache capability is enabled. The [planned inventory](../EDGE0_PLAN.md#phase-6-prefix-reuse)
must be traced against the eventual forward path before it can qualify reuse.
Images remain N/A for the source inventory. No tower/head is dropped or
runtime default changed by reader/operator work.

## Attempt history and failures

### Independent reference registration

The pinned CPU reference fixed 4,304 synthetic cases / 599,040 output values
before any Rust comparison. GGML Q4_0, IQ4_NL and Q2_0 use the exact pinned C
dequantizers; Q2_0 also agrees with the pinned Strata Python definition. FP8
uses the attributed decoder subset from `ngram.cpp`; the full parity utility
with table I/O was not built. Raw fixture and build records stay ignored under
`runtime/edge0-blocks-2026-11-12/`. Rust parity is still pending.

### Metal recipe fixed before execution

Direct decoder comparisons use kinds 2, 20 and 42, with 128 blocks per kind:
eight FP16 scale encodings (`0000`, `8000`, `0001`, `03ff`, `0400`, `3400`,
`b800`, `7bff`) and 16 code phases. Every nibble/pair value reaches every
block position. Require all 16,384 finite output values per available Safe and
packed production library to match the qualified scalar decoder bitwise,
including signed zeros and subnormal scale boundaries. This covers the actual
production library as well as the Safe reference; an unavailable optional
packed library retains the existing Safe fallback.

The existing Metal matrix suite adds these three kinds and retains its fixed
F64 dot-product limit: absolute error at most `2e-5 * max(sum(abs(products)), 1)`.
It covers scalar/tiled equality, row tails, separate inputs, batches crossing
tiles and wide checked address arithmetic. A 64-value block needs its own
six-bit address shift; it must not inherit the 32/256-value assumption.

No Rust or Metal comparison has run yet. Record changed fixtures, failed
controls, compile/admission errors and scale findings without replacing prior
results.

## Performance and memory

Not measured. Synthetic reader allocation limits are bounds on tooling, not
model working-set guarantees. No throughput, loaded memory or prefix benefit
is claimed.

## Evidence, decision and limits

The next gate is independent native component parity and the existing operator
suite. Phase 3 may begin only after that gate; the prepared source alone does
not authorize a successful-loading or inference claim.

## Follow-up history

### Native component results

The scalar test reconstructed the registered fixture stream and matched input
SHA-256 `3e096cfdc9860410bbd8ebe065f4ed60b958c7102fb5447f00898c0be60d9797`
before decoding. All 599,040 results matched the independent canonical output
SHA-256 `5eb63d107f4b63981af4d5dbf61665ac4d952618b61246a168d6c84cf0d474f0`:
finite values bitwise including signed zeros, nonfinite classes under the
declared canonicalization. Q2_0's independent Strata Python versus GGML C
comparison had zero mismatches. Reference registration SHA-256
`5b3ceb3419dc32a24d3c9af500df6d4679a5c2daf1f4c5fc64a7f6007a92024f`;
reference report `b77276107849a6e4f54ae4ae005d52e714b53234a0b73173699392629287c092`;
build record `7286b7e72eeee6f7e017b9c807138b3c89811826ca4701a9c035512f30a743ec`.

Native tests used pinned `nightly-2026-10-01`, locked/offline dependencies,
`cpu,metal`, a distinct `target/edge0-phase2-nightly` and one test thread:

| Check | Result |
| --- | --- |
| Focused Edge0 readers/scalars/Metal | 13 passed, 1 ignored |
| Existing hybrid operator suite with new kinds | 254 passed, 61 ignored |
| Prepared-source metadata reader, explicitly included | 1 passed; reads manifests/labels/scales and stats payloads, without model execution |
| Full native engine suite | 1,043 passed, 153 ignored, 275.92 seconds |
| Fresh packed Fast child (`STARK_GLM53_PACKED_FAST_MATH=1`) | 13 passed, 1 ignored; direct finite decode matches in Safe and optional packed libraries |
| Packed Fast matrix F64/tails/batch check | 1 passed at the fixed `2e-5` sum-absolute-products limit |
| Production engine Clippy | Rust 1.98.1, no default features, `metal`, library target, `-D warnings`: passed |
| Nightly Rust formatting, repository policy and whitespace | Passed |

The original unflagged runs used the Safe default; the packed Fast claim comes
from the explicitly flagged fresh children. The full engine run preceded three
equivalent reader lint fixes; the later focused run exercised those changed
readers again. No native model inference or performance workload ran.

An initial strict Clippy command included CPU reference kernels. It reported
26 existing reference precision/chunk lints outside this change and three new
reader lints. The latter were fixed locally; the source-bound CPU coefficient
spellings were preserved. The production feature check above matches the
workers' Metal-only engine dependency and passes. Nightly also emits existing
deprecated/manifest warnings; passing tests do not erase those warnings.

### Scale conventions resolved against active source

At the pinned Edge0 revision, `expert_layout.cpp:590` reads stored column
scale `s` and computes `1/s`. Verify (`verify.cpp:2184`) and prefill
(`prefill.cpp:4996`) apply that inverse once to `D_g x`, before activation
quantization. The Q3_K native path consumes that already-scaled input. A stale
header comment mentions multiplication in the quantizer, but the implementation
and current call paths do not apply it again.

```text
z_g = D_g x
a_g[j] = z_g[j] / s[j]
h_e = silu(W_gate_stored_e a_label[e]) * (W_up_stored_e a_label[e])
y_e = W_down_stored_e h_e
```

The raw `s` file is preserved and the reader returns it without inversion.
The Q3_K/IQ4_NL record has no NVFP4 scale tail; its traced arithmetic adds no
factor from the dense pool's per-expert `.scale` / `.input_scale` vectors.
Those vectors stay preserved as metadata. NVFP4's separate scale tail and
placement do not qualify or change this artifact's path. The public tree lacks
the named Q3_K conversion tool, so that conversion history is not independently
proved by the runtime trace.

Private trace SHA-256:
`902226656fd228ce03f20f192d6409a760b996b973ce6847d7f156bd48ba2f06`;
source identity manifest SHA-256:
`3e1f1934879c40e2e7acd395914eee1a2eced2ccd39b6625b071767939c4082d`.
The FP8 decoder follows Strata's exact E4M3FN convention: NaN byte magnitudes
map to zero, then sign and global scale are applied. Generic FP8 NaN propagation
would be a different path.

### Gate and remaining scope

The local Phase 2 reader/operator gate passes. The normal GGUF admission
allowlist is unchanged; adding these internal decoders does not make Edge0
loadable. Container descriptors are not complete model admission: the eventual
loader must bind source identity, full tensor names/shapes, table format/scale,
mapped-view ownership and model state. No cache or MTP capability is enabled.

The shared hybrid kernels can feed scoring. The documented JevK5 scoring-parity
replay was not run: the required sealed private reference/input/model set was
not present in the named local route, only historical metadata receipts.
Native operator and full engine tests above are available evidence; they do
not replace that model-bound scoring replay or establish production scoring
equivalence. This remains a broader qualification limitation while work stays
on the owner's local evaluation branch.
