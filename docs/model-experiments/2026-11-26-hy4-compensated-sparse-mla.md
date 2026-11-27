# 20261126-HY4MLA-02 · Compensated sparse prefill for Hy4-flash

## Hypothesis and scope

Sushi's [Hy4-flash attention](https://github.com/beamivalice/sushi/blob/b7b7975cc017acba1b4c0877d65543c4c5a4a381/docs/hy4-flash-arch.md#r5)
recovers query and softmax precision with several narrow matrix products.
This experiment adapts that arithmetic to Stark's existing indexed-cell
kernel and its original F16 cache. It leaves the selector, stored positions,
prefix state, decode and greedy NextN verification on their existing paths.
Earlier [masked prefill](2026-11-25-hy4-grouped-fragments.md#20261125-hy4frag-16--reuse-tensor-attention-with-the-exact-sparse-selection-mask)
matched the fixed 4K output and missed one of thirteen natural-text top-1
rows. That result motivates a separate precision gate for this experiment.

The experimental Metal 4 specialization scales each F32 query by a power of
two before splitting it into three F16 components. It reverses that scale
before the F32 online softmax. The numerator accumulates two F16 probability
components against the original F16 value rows. The denominator continues to
use the original F32 exponentials. The source is an adaptation of the
arithmetic described in Sushi v1.8.1. Stark's existing Strata-derived shader
and its inspection and licensing records retain their original provenance.

## Synthetic GPU validation

The harness compiled the Hy4 shaders in their production order with Metal
safe math and specialized the selected-cell function constant for both arms.
It dispatched one 1,024-thread group per case on an M5 Max and compared every
output with an independently computed F64 oracle. The synthetic geometry
used 32 heads, 576 key channels, 512 value channels and F16 cache rows.
The eleven cases cover sink-only and padded selections, zero queries,
63/65-cell tails, 2,048/2,051-cell selections and scaled query outliers.
The tolerances were set before dispatch.

| Case group | Maximum absolute error | Maximum RMSE | Test result |
| --- | ---: | ---: | --- |
| Ordinary queries and selection tails | 5.974 × 10⁻⁷ | 7.567 × 10⁻⁸ | Pass |
| Queries with ±131,072 outliers | 1.514 × 10⁻³ | 2.887 × 10⁻⁴ | Pass |
| Cancelling large query components | 7.726 × 10⁻⁵ | 1.175 × 10⁻⁵ | Pass |
| Sink-only and all-padding selections | 0 | 0 | Exact |

Each arm returned 180,224 finite values. The ordinary bounds were
1.2 × 10⁻⁴ absolute and 2 × 10⁻⁵ RMS. The outlier bounds were
3 × 10⁻³ absolute and 5 × 10⁻⁴ RMS. The compensated pipeline used
30,464 B of static threadgroup memory versus 26,240 B in the existing
pipeline. The existing kernel had lower error on the extreme-query case.

## Prototype disposition

Keep the specialization opt-in at `STARK_HY4_PREFILL_COMPENSATED_SPARSE_MLA=1`.
The synthetic test establishes kernel execution and bounded error for this
fixture. Full-model top-1, rendered continuations, image-loaded prefix
behavior and sparse-prefill throughput remain open. The owner-authorized
removal of the local AngelSlim copies to stage Sushi's larger STQ pack made
the full-model gate unavailable in this run.

The Hy4-enabled release worker built with Rust 1.98.1 and locked
dependencies. The generation library suite passed 181 tests, with six
ignored.

| Private evidence | SHA-256 |
| --- | --- |
| Release worker | `a0cac800dec6cf7a034a9a35fd3e58ec1beb5380e052bfa837dfcc07b8fdcee5` |
| Engine source diff on `17e38617e` | `f7f0e9053fb77c358a6a497c30f02f89ef1f4cb38ed0b7d111c3032cc568fa4f` |
| Assembled Metal source used by the harness | `c8544d41e935cc27d87c82e4d01b3f72f962beccad317d02b42d8f5cb0a518c4` |
| Source assembler | `99689ed324bcd967fdee3e02a2b0f87434954fc346d9a202ce4cc83e51af95d2` |
| Validation harness | `dfe26b8e48d6b65a2eb0a416ea30616fc99ada7066016497c212fa69047933dc` |
| Aggregate validation receipt | `f04309d78fa2b333bc085ae31594451f240d8b2ff6b8840c21a08d20c7b497c7` |

## Default policy follow-up

The initial tuned Metal 4 profiles requested this specialization by default. The worker
uses the original selected-cell kernel if Metal rejects the pipeline or
reports a limit below the required 1,024 threads. Set
`STARK_HY4_PREFILL_COMPENSATED_SPARSE_MLA=0` for an explicit control.
The follow-up dispatch of all eleven synthetic cases passed using the same
assembled shader SHA-256 and aggregate receipt as above. The locked Rust
1.98.1 release worker SHA-256 for the proposed policy was
`e5fccfce3394dcb10a56d0b0c72198064a2cd6cf7406fc000f0f4acd325c8e94`.

The local Q3_K weights were removed after the resident OpenCode bracket.
After restoring the exact artifact, the selected-cell stage was 57–60%
slower at 16K and 27K and generated hashes changed at both lengths. A
matched uninstrumented 27K OpenCode pair confirmed slower whole-request
time. The original kernel is again the default. Natural-text quality was
outside this synthetic comparison. [Full-model follow-up](2026-11-26-hy4-post-sushi-controls.md#selected-cell-mla-diagnostics)
