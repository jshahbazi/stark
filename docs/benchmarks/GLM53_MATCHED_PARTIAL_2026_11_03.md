# GLM-5.3 matched-checkpoint comparison: partial study

Date: 2026-11-03. **One of three planned workloads qualified.** The completed
128-input/256-output case favors MLX-Serve on prefill, useful decode, and warm
request time. The native 1024-input block failed host screening; the remaining
blocks and 3840-input case were not run. This is not a completed three-case
comparison.

[Aggregate results and hashes](data/glm53-matched-partial-2026-11-03.json).
[Protocol and pre-measurement amendment](GLM53_WEBSITE_PROTOCOL_2026_11_03.md).

## Completed case

Both engines used the same GLM-5.3-Flash UD-Q2_K_XL GGUF payloads, tokenizer,
128 input token IDs, 256 greedy outputs, and requested context of 384 tokens.
Four independent model-load blocks ran in Stark / MLX-Serve / MLX-Serve / Stark
order. Each block supplied one excluded warmup and five accepted measured
requests: ten measured requests per engine across two independent loads.

| Median over ten measured requests | Stark opt-in profile | MLX-Serve 26.10.2, llama |
| --- | ---: | ---: |
| Prefill tokens/s | 40.8672 | 260.2141 |
| Useful decode tokens/s | 33.0390 | 34.0261 |
| Complete request wall seconds | 10.8613 | 7.9916 |
| Excluded attempts | 16 | 0 |

Stark's median request time was 35.91% longer. MLX-Serve's median prefill rate
was 6.37 times Stark's. These comparisons describe this one workload and the
recorded versions. They do not establish performance at other prompt lengths.

All measured requests emitted 256 outputs and passed the unchanged host screen
with zero pageouts and swapouts. Output hashes repeated within each engine
across both loads. Cross-engine token-ID parity is unavailable: native receipts
hash generated IDs, while external receipts hash response text. The sixteen
excluded native attempts are retained privately; fourteen recorded paging and
three exceeded the external CPU screen, with overlap between reasons. Their
pageout deltas sum to 502. Excluded requests do not enter the medians.

## Artifact and execution identity

- Host: Apple M5 Max, Mac17,7, 128 GiB physical memory, macOS 26.7.1. AC Power
  and `pmset powermode 0` were observed before the study.
- Weights: `unsloth/GLM-5.3-Flash-GGUF` revision
  `a38483c8cd5df544f53d70fb281afe97369d5ab6`; metadata/tokenizer:
  `zai-org/GLM-5.3-Flash` revision
  `04c4e9e95c5da8862dced7e5056455116f83a7e0`. Both engines consumed the same
  four GGUF payloads, without conversion or requantization. Prior full hashes
  and current size, modification-time, and inode guards matched before and
  after each block.
- Native generator: Rust 1.98.1 build source
  `cbb66f481c75e16ed8c076290392bcad35df29be`, binary SHA-256
  `29163c3a12d88cc78003b252903c906c7408ee8b1817bc74d0735d6b5c5e769b`.
  The compiled/runtime source paths match PR #71 head `148792c0` and the
  benchmark checkout. Engine-wide defaults remain unchanged; these measurements
  use the complete opt-in resident-bank/T8 profile, F32 activations and
  accumulation, and its existing admission and scratch caps.
- External executable: installed MLX-Serve 26.10.2, MLX 0.32.3, llama.cpp
  b11429, SHA-256
  `0c7029c3748cd1c86939483903249e3a35ade14eaaa7d41dfcec4ad9823b9b62`.
  It used its embedded `llama` backend, not native MLX execution. This is a new
  version-bound comparison, separate from the earlier 26.10.3 measurements.
- Benchmark tooling source:
  `6cc91370dc2be1f2eaae562edeac89f2eddb5b9f`.
  The frozen corpus was `docs/NATIVE_HYBRID_MODELS.md` from PR #71 head
  `148792c0`, SHA-256
  `536b8626feffc7b6277e0b0de89f897b3559257e765e14adddf1da5f14559240`.
  The completed case's shared input-ID hash was
  `464aeea3636567005a1ed10b75ca5a36f778060f07657d69565168c1976e1981`.

Native requests omitted cache scope and rejected reported nonzero input reuse.
External prefix/tokenizer caches and speculation were disabled, and every
measured external response explicitly reported zero cached prompt tokens.
Effective external settings and all observed native selectors are in the JSON.
External arithmetic precision is not claimed to equal the native F32 policy.

## Timing definitions and host controls

Prefill and decode use each engine's internal phase timers. Useful decode is
255 post-first-token outputs divided by decode seconds. Native wall time
includes its stdio request path; external wall time includes local HTTP.
Model loading, warmups, and quiet-admission waiting are excluded. No estimated
unused forward or other unmeasured overhead was subtracted.

Ten-second quiet admission and per-request CPU, compiler, thermal, monitor, and
paging screens were applied. Fair thermal state is allowed by the policy.
These are observational controls, not physical GPU isolation or proof of an
otherwise idle host. Primary statistics were declared as pooled request
medians before study timing; ranges, maxima, pooled rates, and block medians
remain secondary. Two independent loads per engine do not establish a general
throughput floor.

## Incomplete workloads

| Planned input/output/context | Status |
| --- | --- |
| 128 / 256 / 384 | Complete; four qualified blocks |
| 1024 / 256 / 1280 | First native block failed; other blocks not run |
| 3840 / 256 / 4096 | Not run |

The native 1024/256 block exhausted eighteen attempts over 948.847 seconds.
It supplied a warmup and only two accepted measured requests. All fifteen
excluded attempts recorded pageouts, summing to 1139; three also exceeded
the external CPU screen. Exact output repetition, artifact/source pins, and
worker close/reap checks passed. No rows from this block enter comparison
statistics. Paging is observed but is not attributed to a particular process.
Additional model loads stopped after this failure. Completing the remaining
matrix requires sufficient host memory headroom and new qualified blocks.

## Numerical boundary

The [earlier native validation](../archive/runs/2026-11-03-glm53-prefill-merge-validation.md)
retains its source-bound 512-context, 192-point scorer comparison and F64 kernel
checks. That evidence is distinct from this performance study and does not
establish cross-engine or whole-model quality equivalence.

A planned 4096-context numerical extension ended after 1016.023 seconds before
a paired comparison report was produced. One worker was observed; the
reference-only phase attribution follows the fixed reference-before-candidate
source order. The parent reported `receipt_boundary_invalid` because the report
was absent. The underlying failure cause is unknown. This is not a numerical
mismatch result. Before throughput timing, the protocol was amended to keep
numerical and performance qualification separate; the failed extension remains
visible in the JSON. Long-context numerical equivalence, answer quality, and
long-context stability remain unestablished.

The separate diagnostic pilots are excluded from the study. Public artifacts
contain aggregate metrics and hashes only. Private request receipts, excluded
attempts, and lifecycle evidence remain retained without prompts, generated
text, token vectors, or probability arrays.
