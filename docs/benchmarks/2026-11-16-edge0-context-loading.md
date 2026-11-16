# Edge0 normal local loading and 8K context

Date: 2026-11-16. Attempt
[`20261116-EDGE0-20`](../model-experiments/2026-11-16-edge0-context-loading.md).

The pinned Edge0-145B Q3_K artifact passes bounded generation at an
8,192-token runtime limit through the usual model-path loader. The explicit
`STARK_EDGE0_LOCAL_EVALUATION=1` gate selects the native Metal graph,
expert-record windows and singleton kernels. MTP is unavailable on this route.
This is local, non-commercial evaluation on an Apple M5 Max with 128 GiB RAM
and macOS 26.7.2; no app/catalogue or installed-build admission follows.

| Request | Input / emitted output | Prefill tok/s | Useful decode tok/s | Request wall seconds |
| --- | ---: | ---: | ---: | ---: |
| Initial reference | 128 / 16 | 8.03 | 9.76 | 17.496 |
| Sparse transition | 2,048 / 16 | 13.61 | 11.22 | 151.811 |
| Near capacity | 8,176 / 16 | 13.74 | 10.87 | 596.275 |
| Long reset replay | 2,048 / 16 | 14.10 | 11.14 | 146.603 |
| Short reset replay | 128 / 16 | 14.54 | 11.71 | 10.110 |
| Recovery after rejections | 128 / 16 | 11.28 | 11.03 | 12.733 |

Each row is one request, in the listed order with a rendered-chat smoke between
the long and short reset replays. There are no excluded warmups or pooled
medians. Worker loading took 1.565 seconds and is excluded from request rates.
Greedy sampling uses seed 42, a fixed 16-output limit and context 8,192. The
private corpus is repeated in memory for longer inputs; this tests execution
and state reset, not long-context retrieval or answer quality.

Prefill is input count divided by worker prefill time, ending at the first
sample. Useful decode is emitted tokens after the first divided by worker
decode time. All table rows emitted and sampled 16 tokens with a length stop.
The near-capacity input leaves exactly 16 output slots; its prefill takes
594.871 seconds, about ten minutes. Rates are observations under ordinary host
load. Historical strict screens and rejected measurements remain unchanged.

The initial 128-token output matches the retained reference exactly. Long and
short reset replays match their original hashes/counts/stopping conditions.
An 18-token input rendered through the normal chat template emitted two tokens
and then EOS. Context 8,193 and one-draft MTP requests were rejected, and the
following short generation matched. Readiness reports MTP and prefix reuse
unsupported. The worker closed through stdin EOF, exited zero and was reaped.
Sampled RSS peaked at 1,030,078,464 bytes; this excludes broader mapped/Metal
allocations and does not establish minimum hardware memory.

The original weights, preparation receipt and F32 activation, accumulation,
recurrent-state and cache policy are preserved. Payloads were not rehashed;
the run assumes the previously verified private payloads stayed immutable.
The independent offline audit checks source inventory, artifact bindings,
formulas, replay evidence and cleanup. The
[attempt record](../model-experiments/2026-11-16-edge0-context-loading.md)
contains failures, tests, gate-negative evidence and capability limits.

| Identity | SHA-256 |
| --- | --- |
| Worker | `e66b30c4903990c15b3a5022296f6f7313711e542848a0bb0ae8a9880ec5a335` |
| Source manifest (`da003d3c` plus changes) | `25c61850b981b1cf383312327b66f1f8cda33fafcca2b9014c9b81de76e4273d` |
| Private context report | `169c27aea5bf905edb8173bcc474f5c3e2afb8f19a490f8832d5e8cebffcc1e9` |
| Independent audit | `09af54d122b52a364380679c4dd55e325267dae966ddc064df26f4dd9b57d8de` |

Use the explicit candidate worker and prepared folder in the
[local loading instructions](../NATIVE_HYBRID_MODELS.md#checkpoints-and-preparation).
The [earlier singleton benchmark](2026-11-13-edge0-singleton-speed.md) retains
its separate 512-context comparisons; these longer-context observations do not
replace them or complete the old strict timing gate.
