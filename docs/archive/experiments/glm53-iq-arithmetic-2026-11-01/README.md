# GLM IQ arithmetic probe

Status: the initial probe is retained below. A subsequent opt-in signed-factor
selector is implemented for qualification; it is disabled by default.

Mode 0 retains the checked grouped32 schedule's existing arithmetic. Mode 1
forms an exactly representable signed integer grid value before multiplying
the input. Mode 2 explicitly calls Safe `fma(input, signed_grid, sum)`.
The production default remains mode 0, including its existing GPU input-bound
check and original multirow4 fallback. The trial factory requires finite host
inputs with absolute value at most `1e20` and an even code offset.

The initial v28 unit rejected mode 1's bit-equality claim with
`IQ exact sign factoring changed bits`. Exact sign factoring in real arithmetic
does not establish identical compiled F32 rounding. A compiler contraction of
the shorter expression is one possible explanation; it has not been proved.
The failed run stopped before complete numerical qualification.

Both trial modes now require the unchanged F64 bound of
`2e-6 * max(sum(abs(weight * input)), 1)` per row and relative RMS error at most
`2e-5` against mode 0. The unit records aggregate bit matches and maximum error
against mode 0, plus mode 1 versus explicit mode 2. Matching mode 2 can support
the contraction hypothesis but does not prove compiler instructions. Existing
retained reference gates are unchanged. The separate ABBA component reports
bit equality as evidence and applies the same relative-RMS gate; the unit
checks the full F64 bound.

The v28 prebuild source bundle was
`40da1e85a08a44c780a1ea7c00d7f0f02b07f22448c1c68a49ab7cf9354e9e44`.
Private evidence remains under `runtime/glm53-optimization-20261101/` in
`prebuild-source-v28.json` and `iq-arithmetic-oracle-v28.log`. A fresh build,
complete oracle run, component timing, and model qualification are required
before any admission decision. No model or private vectors are included here.

## Opt-in follow-up

`STARK_GLM53_IQ_SIGNED_FACTOR=1` selects mode 1 only when both
`STARK_GLM53_IQ_GPU_BOUND=1` and `STARK_GLM53_IQ_TYPED16=1` admit IQ2_XS or
IQ3_XXS with 2048 or 4096 columns and an even code offset. Its separate Safe
pipeline cache retains the GPU input-bound prepass. Odd offsets use the prior
byte path; extreme and nonfinite inputs retain the original multirow4 fallback
and sticky failure behavior. No pipeline descriptor hints are enabled by this
flag.

The dedicated signed-factor prepass test applies the same F64 and relative-RMS
gates and exact extreme-input fallback checks. The ignored graph admission test
interleaves actual even and odd offsets. The new component compares complete
prepass plus matvec for typed mode 0 and signed-factor mode 1, each with and
without test-only 64-thread descriptor hints, using Safe ABBA timing and an
active inference guard. Model qualification of this integration is pending.

Run retained typed bit gates with `STARK_GLM53_IQ_SIGNED_FACTOR=0`. Run
`iq_signed_factor_graph_admits_even_offsets_and_keeps_odd_byte_cache` in a
fresh process with all three flags above set to `1`. The nonignored
`iq_signed_factor_prepass_preserves_f64_offsets_and_exact_extreme_fallback`
and `iq_checked_pipeline_hints_preserve_bits_and_failure` tests do not depend
on environment selectors. The ignored component is
`iq_signed_factor_and_hints_prepass_gpu_component_timings`.
