# GLM packed finite-check fusion trial

Status: not admitted to production. The original kernels remain unchanged.
The trial passed its numerical checks and did not establish a complete-model
performance benefit. Host interference prevents attributing the observed rate
differences to the patch.

`rejected.patch` applies to source base `3eca4590`. It adds graph-only packed
GEMV variants that check inputs and row totals, set the existing sticky failure
word, and replace separate pre/post finite scans. Standalone entry points,
quantization, accumulation order, routing and model precision stay unchanged.
The diagnostic `STARK_HYBRID_SEPARATE_FINITE=1` retains separate scans.

| Identity | SHA-256 |
| --- | --- |
| Patch | `7056fa4c730a2111c7019bfcab51847a8cdaae83dc4331a0f38094795c6c87c5` |
| Candidate generator | `b9893aec85c1bef0297e2c23a2d98f2eede7be50d0bc04ff4e6ff14e3aa79f73` |
| Candidate optimized scorer | `190930839f1525722a6f90e2b27c12171a9ddb6af6e5e2bb0595ed319b5d9c3d` |
| Candidate hybrid source bundle | `4ce9181c6b5e9dc361d7540c1209f8ee23edc4653e2c5dc65645c140acc7b16f` |

The bundle hashes the 35 `.rs` and `.metal` files directly under
`crates/stark-engine/src/engine/hybrid`, sorted by relative path. Each entry
contributes UTF-8 path, NUL, bytes, NUL.

The full physical CPU/Metal suite passed: 675 tests, 48 ignored, seven CXX
tests and three documentation tests. The new packed regression checks all 13
admitted formats, finite output bits, row tails, multiple matrices,
NaN/infinity rejection and overflow that a later activation clamp could hide.
Both saved GLM references passed their original 94-position checks. The
generator and optimized scorer passed the small GLM worker integration,
including recovery, request isolation and score-cache invalidation.

The exploratory A/B/B/A medians were 8.58 tok/s for the existing worker and
7.43 for the candidate. Every block failed its host screen. A later A/B/A
window measured 9.79, 9.42 and 5.17 tok/s; only the candidate block passed the
prospective desktop screen. These observations do not qualify an effect size.
The benchmark report retains exclusions and source identities separately.

One possible cost is repeating input finite checks across output rows. No
component experiment established that explanation. A baseline profile still
records 172 submissions and waits over four decode steps: 43 per token.
Bounded execution groups that reduce routing completion boundaries remain a
separate design task under the existing memory-admission limits.

See the [baseline report](../../../benchmarks/GLM53_MLX_BASELINE_2026_11_01.md)
for model provenance, workload, receipts and reproduction. Apply the patch in
a separate checkout and target directory to repeat the trial. It is preserved
as original Stark source with the existing GGML attribution intact.
