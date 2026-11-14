# Edge0 singleton projection speed observation

Attempt `20261113-EDGE0-19`, 2026-11-13, local non-commercial evaluation only.
Specializing one-token packed projections increased observed median decode
throughput from **7.35 to 12.09 tok/s** in the final binary, an observed 64% gain.
Prefill increased from 12.78 to 14.71 tok/s, an observed 15% gain. Three measured
requests per setting followed one warmup. An earlier build reproduced the decode
gain. These sequential observations do not establish full performance
qualification. MTP remains default off; singleton specialization is opt-in.

## Workload and implementation

The artifact remains
`Edge0/Edge0-145B-q3k@d7de835cfce7d9bb22a3d5ebb7d1e8565d4ecf74`.
Its native preparation receipt SHA-256 remains
`8dc700fe5cd5f6644ed9f062eb391ce32cb0a09b76a2e19ec44539bf89200a5e`.
Original stored formats and F32 activation, accumulation, recurrent-state and
cache policies remain. The worker uses virtual Metal with selected expert-record
bindings, context 512 and unavailable prefix reuse. Payloads were not rehashed.

Each request has 128 input tokens, greedy seed 42, a 16-token output limit and
matching length termination. Decode rate is 15 divided by native decode seconds,
excluding the initial token; prefill is 128 divided by native prefill seconds.
End-to-end rate is 16 divided by parent request wall time, including prefill.
All warmups and measurements have 16 emitted/sampled tokens and zero proposed,
accepted, verified or speculative-round work. Input and output hashes match
across the completed cases. No prompts, generated text or token vectors are
published. The prior [observed-speed record](2026-11-13-edge0-observed-speed.md)
retains its separate binary, host conditions and measurements.

`STARK_EDGE0_SPECIALIZE_SINGLE=1` selects scalar accumulators and format-constant
Q8_0, Q3_K and IQ4_NL kernels only for one-token jobs, including grouped prefill
jobs that contain one token. A separate one-token Q3_K gate/up kernel preserves
the fused activation. Both retain the original 256-column stride, SIMD reduction
and ordered eight-group F32 final sum. Other formats and multi-token jobs retain
their existing kernels. Input finite checks, sticky failure reporting and checked
weight spans remain. The selector defaults off and is fixed once per process.
The [reference-framework review](../archive/runs/2026-11-13-edge0-reference-performance-ideas.md)
maps these choices to pinned upstream sources and identifies further work.
This independently implemented singleton change copies no upstream code.

## Completed observations

Every row below has one warmup and three measured requests. Each setting was run
as a separate sequential worker session. Initial cases have a different build
and profiling recipe from later cases; comparisons across those groups do not
isolate the cause of their absolute throughput differences.

| Follow-up / setting | Prefill tok/s | Decode tok/s | Disposition |
| --- | ---: | ---: | --- |
| `19-FINITE`: prior binary baseline | 10.80 | 5.82 | Reference for next two rows |
| `19-FINITE`: exact-span finite batching | 9.92 | 5.41 | No improvement demonstrated; removed |
| `19-B16`: sixteen-token prefill, finite batching off | 9.73 | 5.91 | No prefill improvement demonstrated; removed |
| `19-HEAP`: automatic scratch heap off | 13.20 | 7.06 | Same-build profiled reference |
| `19-HEAP`: automatic scratch heap on | 13.47 | 7.12 | Small difference; removed |
| `19-SINGLE1`: singleton specialization off | 13.14 | 7.08 | Same-build profiled reference |
| `19-SINGLE1`: singleton specialization on | 13.85 | 11.55 | Earlier-build replication |
| `19-SINGLE2`: final specialization off | 12.78 | 7.35 | Pruned-build reference |
| `19-SINGLE2`: final specialization on | 14.71 | 12.09 | Retained opt-in |

For the singleton comparison, `STARK_EDGE0_PREFILL_BATCH=8`,
`STARK_EDGE0_BATCH_FINITE=0` and `STARK_EDGE0_SCRATCH_HEAP=0` are identical in
both registrations. Only `STARK_EDGE0_SPECIALIZE_SINGLE` changes from `0` to `1`;
each session has its own aggregate profile output. Initial median decode rate
is 1.631x, prefill 1.054x and end-to-end 1.142x (1.330 to 1.519 tok/s).
The final build contains only the three singleton source files. It runs on then
off, reversing the earlier off/on order. Its median ratios are 1.644x decode,
1.151x prefill and 1.202x end-to-end (1.327 to 1.595 tok/s). These are separate
session comparisons, not randomized paired arms or one pooled timing estimate.

The matched profiles corroborate a GPU-side decode reduction. Median cumulative
decode GPU seconds fall from 1.769 to 0.984 in the initial build and 1.690 to
0.945 in the final build. Final decode wait falls from 1.937 to 1.161 seconds;
prefill GPU time falls from 8.789 to 7.555 seconds. Final router F64 decode time
is 0.0064/0.0052 seconds off/on. Categories are inclusive and must not be summed.
Each profile has one warmup and three measured records; medians exclude warmup.

All sessions record global decompressions. The three singleton-off requests
total 500 and singleton-on total 86; compression, pageout and swapout deltas are
zero. Final off/on requests record 197/61 decompressions and 43/0 pageouts; their
compression and swapout deltas are zero. No baseline is subtracted. Final peak
sampled owned RSS is 604,160,000 bytes off and 607,567,872 bytes on. RSS does not
establish physical Metal residency. Every worker exits zero on EOF and is reaped.

## Correctness, bindings and limits

Independent source review found matching block geometry, buffer bounds, finite
obligations and per-token reduction order. Focused native tests cover bitwise
baseline/specialized parity for Q8_0, Q3_K, IQ4_NL and fused Q3_K at columns 512
and 1792, plus poison and truncated-span rejection. The final native run passes
54 tests with 11 ignored; ignored actual-artifact gates are not new passes. The
pruned implementation is freshly built with Rust 1.98.1 and observed in both
settings. Its three changed source hashes and current worker hash match the
final registrations. Earlier source registrations include the discarded
experiments disabled. Registration hashes alone do not prove build provenance.

| Evidence binding | SHA-256 |
| --- | --- |
| Rust 1.98.1 worker, both settings | `ac38244f534ceb39c570ec8a0cf85f6c0ef9febc19ba05a1cdd2cc2d0b39ef94` |
| Registered source manifest | `9d5e4bb52f5168a4a655f8493006217247444d5c1369a5f54f6dc614d57bad40` |
| Comparison runner | `0306eb9371016372e7ad1ebbfb681716b2395fd1a0b6962dab25fe932b6d0628` |
| Off registration | `7c20628555bcfdf915181d7bf86fc4a1ef5d1bbc42e22f3d5aabf60cc5d0b824` |
| On registration | `f9d8e479065dffe40ec2b7a5534d6d1fc8dacaa5755f87a7cc5f47d541f6d13f` |
| Off report | `765a4d32cbbec6116c615f6c2e7fece8caa6a5f6cc95979904eb709c7de359c9` |
| On report | `ac1aad6dbfee893b46a74f91b347d84b17f668fe0f22049e63eb8814eac478fc` |
| Off profile | `a27874d1aa5847ee2b7431f75c2b318b9551ea86f46ab198e787866a4f08fcb9` |
| On profile | `e07cfa5e6c1092e0c075ef9d619ca5faa8afe0d593328d03c34d508d835c5244` |
| Offline singleton audit | `23d10a4e4dc7275614e549daec6bf0d88fafcb389de9fabba7373948bcde7c98` |
| Offline discarded-case audit | `2b080ca5860689df527bd027fce95333270bd2e78041306ba15454a73f1d4a3c` |
| Final Rust 1.98.1 worker, both settings | `19286a26b7259909f6829d89d0610c617669ce9f47e8985f9ada6ebd6e74f157` |
| Final source manifest | `594e650651796385a8dd03c98ee3870a893f775f6f689e877421ea9406c3a3bf` |
| Final off registration | `ff5ab812c7736a21a0a6accbb6340d3ab98dabf3d68a888aadd772c6584b0513` |
| Final on registration | `bec29d992ecd1da1017457b924e01814759b583b23af704a7de4b62f549e2c9a` |
| Final off report | `c020858cd561face1e89b146d3fc860d84cc250ce73f8d40adeb545258ca3e60` |
| Final on report | `afb9c1abc8a96ea5e3811bb9d67caa3e1c2f93a428a99747d59a96cc29515863` |
| Final off profile | `e22d27485403a8271aeca0477288ecdc00ac626860c190c4084904591f46bc39` |
| Final on profile | `1558c257e4ba421330956abd413aa80f7601467950232bc0db939ac322cbe9b0` |
| Offline final audit | `1caeddb1b2ddb244bccb29894481926a100156ca9f04ca1193005d4514b22356` |
| Offline full nine-case audit | `e3f99ab5e411798ba8b44bb42bd1f852feff50ac61fa7031b82129f95c415b8b` |

The offline audits verify report/registration and profile hashes, output/count/
stopping agreement, zero speculative work, teardown and aggregate arithmetic.
Private registrations retain full source manifests and exact binary hashes for
all discarded cases. Their raw receipts and patches remain preserved.

This work adds no host quiet wait or timing eligibility claim. Strict admission,
paging, CPU and GPU screens in the separate qualification workflow remain
unchanged. Small samples, sequential session order and host activity limit causal
and broader-workload conclusions. Larger contexts, prefix restoration, MTP speed,
installed-build behavior and production admission remain outside this result.
