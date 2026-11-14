# 20261113-EDGE0-19: singleton generation kernels

| Field | Value |
| --- | --- |
| Date / owner / scope | 2026-11-13 / Codex / local non-commercial evaluation only |
| State / result | Final pruned-source observation complete; 12.09 decode tok/s with singleton specialization |
| Source | Base `b9fe0e01846cd2bd33897fe7ecb0e984c913864d` plus source-manifest-bound local changes |
| Artifact | `Edge0/Edge0-145B-q3k@d7de835cfce7d9bb22a3d5ebb7d1e8565d4ecf74` |
| Plan / predecessor | [Phase 5](../EDGE0_PLAN.md#phase-5-mtp); [attempt 18](2026-11-13-edge0-observed-speed.md) |
| Disposition | Selector one recommended for this local evaluation; default zero and MTP off remain unchanged; strict timing qualification pending |

## Question and implementation

Improve actual prefill and decode speed with MTP off. The retained experiment
specializes token-count-one packed Q8_0, Q3_K and IQ4_NL matrix jobs and fused
Q3_K gate/up SwiGLU jobs. It removes dynamic token accumulators and makes
format selection constant. The 256-column traversal, eight-SIMD-group
reduction order, original weight formats and F32 arithmetic remain intact.
`STARK_EDGE0_SPECIALIZE_SINGLE=1` opts into this experimental path;
other token counts retain the existing batch kernels.

Larger prefill batches, grouped finite checks and shared scratch allocation
did not justify retention in the local observations. Those implementations
and their selectors were removed. The final change contains three engine
files: singleton kernels, dispatch and focused parity tests. No new weight
precision, state precision, context size or inference backend is qualified.

## Workload and evidence

Each worker runs separately with the retained 128-token input, context 512,
greedy seed 42, explicit MTP zero and exactly 16 emitted/sample tokens.
One warmup precedes three measured requests. Prefill is `128/prefill_seconds`;
decode is `15/decode_seconds`, excluding the first token; end-to-end is
`16/parent_request_wall_seconds`. Medians exclude warmup.

The exploratory matched pair ran off then on; the final pruned-source pair
ran on then off. Both pairs enable aggregate native profiling. Within each
pair the worker and source identities match. The historical baseline binary
is separate; a current-checkout binding does not establish its build
provenance. Every attempt records worker, runner, registration and source
identities separately.

| Median observed tok/s | Exploratory off | Exploratory on | Final off | Final on |
| --- | ---: | ---: | ---: | ---: |
| Prefill | 13.14 | 13.85 | 12.78 | 14.71 |
| Decode | 7.08 | 11.55 | 7.35 | 12.09 |
| End-to-end output | 1.33 | 1.52 | 1.33 | 1.59 |

The final pair observes approximately 15% higher prefill, 64% higher decode
and 20% higher end-to-end speed. Final off prefill samples span 12.08–14.06
tok/s. Host activity is uncontrolled; runs proceed without quiet admission or
a full host monitor, retaining paging endpoints without subtraction.
Profiling adds overhead. These short observations do not establish a general
speed guarantee or pass the existing Phase 5 timing qualification gate.

All 36 requests across nine attempts match output SHA-256
`3033ab1a108b72992b627a19aa7fd547c0ae6c72ac2be42c01b77e6022890047`,
sample/emit 16 tokens and stop at length. No token vectors or text are saved.
All workers exit normally and are reaped. Each profiled attempt has four
successful aggregate records, sealed by its hash and record count.

Final worker SHA-256:
`19286a26b7259909f6829d89d0610c617669ce9f47e8985f9ada6ebd6e74f157`.
Final source-manifest SHA-256:
`594e650651796385a8dd03c98ee3870a893f775f6f689e877421ea9406c3a3bf`.

| Private report | SHA-256 |
| --- | --- |
| `single-off-1.report.json` | `765a4d32cbbec6116c615f6c2e7fece8caa6a5f6cc95979904eb709c7de359c9` |
| `single-on-1.report.json` | `ac1aad6dbfee893b46a74f91b347d84b17f668fe0f22049e63eb8814eac478fc` |
| `final-off-1.report.json` | `c020858cd561face1e89b146d3fc860d84cc250ce73f8d40adeb545258ca3e60` |
| `final-on-1.report.json` | `afb9c1abc8a96ea5e3811bb9d67caa3e1c2f93a428a99747d59a96cc29515863` |

## Validation and remaining limits

The final native suite passes 54 tests with 11 ignored. Final generation
verification passes the fixed-work reference checks above. Independent review
recomputes all nine attempts' request rates, medians, registration hashes,
exact output-work checks, profile hashes/counts and normal cleanup.
An intermediate batch-16 inventory test initially retained an eight-token
expected allowance; it was corrected before that experiment was discarded.

The [pinned upstream review](../archive/runs/2026-11-13-edge0-reference-performance-ideas.md)
maps this specialization to reference-engine strategies and identifies fused
expert execution as a further prefill opportunity. No upstream implementation
was copied for this change.

Prefix reuse remains unavailable under the existing concrete state-restoration
blocker. Original artifact restrictions, local non-commercial scope and the
512-token context remain unchanged. Broader workloads, images, product
admission and commercial use are outside this attempt.
