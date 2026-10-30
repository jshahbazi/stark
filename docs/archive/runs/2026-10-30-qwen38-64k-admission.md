# Qwen3.8 64K admission implementation, 2026-10-30

Status: source implemented; builds, synthetic tests and actual-model 64K
qualification are pending. This record makes no throughput, accuracy or
completed 64K runtime claim. Earlier dated 16K results and failed experiments
remain unchanged.

The standard Uzu Qwen decoder now admits up to 65,536 context tokens, bounded
by the artifact's declared maximum. Generation and scoring workers grant that
request ceiling only when the actually loaded common `LanguageModel` contains
the Qwen decoder variant. Other text, GLM/Flint and vision workers retain their
16,384-token admission. The previous public workflow already had a 16K cap;
the engine's additional 16K clamp was the Qwen capability restriction.

Desktop Chat, local API and Decision mode use the effective worker-advertised
ceiling. Score readiness adds `max_context_tokens`; clients retain a 16K fallback
for old scoring workers. Existing 1 MiB production request bodies, 2..16 scoring
candidates, serialized decisions, exact immediately-consecutive cache keys,
4,096 generation output limit and unsupported-operation gates remain intact.
Only the private full-vocabulary control overlay widens scoring candidates or
body limits.

No shader arithmetic, precision, quantization, cache format or projection
dispatch changes accompany this admission change. Prefill still batches at
most 16 tokens; gathered matmul still splits M into groups of at most eight.
The QSA selector's packed 16-bit compaction counters apply to chunks of at most
16,384 compressed keys. Ratios 1/2/4 at a 65,536-token context use at most
four/two/one chunks; the host admission also bounds visible compressed blocks
to 65,536. Local attention arrays remain 64-token/four-head tiles. This avoids
treating the ratio4 model as proof for every accepted compression ratio.

At the target Q24/KV2/D256, index4/D128, ratio4 geometry, twelve QSA caches
contain 1,056 MiB of logical payload at 64K, versus 264 MiB at 16K. This excludes
weights, allocation overhead, residency, temporary buffers and other state.
Memory availability and allocation behavior require actual measurements.

Prepared controls cover:

- Engine/state admission, exact 65,535/65,536 committed-length boundaries,
  overflow rejection and short-request capacity invariance.
- Independent stable selector expectations through 65,536 compressed blocks,
  ties and chunk transitions; dyadic score-grid controls; final K/V scale/code,
  raw-index and compressed-index writes with guards.
- Analytic attention over high selected block IDs and causal tails of lengths
  0/1/2/3 near 16K/64K boundaries, with poisoned unselected/future rows.
- Opt-in worker admission, loaded-model bounds, unchanged body/candidate limits,
  context-sensitive cache behavior and Python readiness/settings/decision gates.

Root-controlled qualification must first run those source-bound controls, then
allocate the actual model's full 64K state with a short fixed prefix. Exact
short-prefix readout/sample hashes across small and 64K capacities test storage
stride invariance; they do not qualify long-prefix numerical behavior. Matching
long-prefix readouts and sampled generation remain separate gates.

A 65,536-token generation context can contain **65,280 prompt tokens plus 256
sampled outputs**. A full 65,536-token prompt plus 256 outputs would require
65,792 capacity and is outside this admission change. Benchmark both top-k20
and unbounded nucleus without speculation or changed weight quantization, and
record finite behavior, repetitions, source/binary/artifact identities and
aggregate results separately before claiming 64K support or the 77–80 tok/s
target.
