# 20261109-GLMVIS-07: GLM default-image admission

Registered before admission changes. Baseline `fecc27d5`; predecessor:
[GLMVIS-06](2026-11-09-glm-native-image-generation.md).

The first actual installed-GGUF image worker completed native content admission
and reached readiness in 196.30 seconds. Its first control used a malformed
private-probe cache scope, so that failed assertion is not a decoder result.
The corrected probe uses the required opaque 32-character lowercase-hex scope
and records fixed error/count metadata before assertions. The corrected result
and its admission failure are recorded below; following text recovery was not
reached.

A separate candidate text-only diagnostic generated eight tokens, then returned
`resource_budget_exceeded` on its next request after lazy target-bank allocation.
That is an observed admission issue; it does not by itself identify the cause
of any image request failure. Inspect actual available/allocated memory and the
checked request inventory before changing policy. Preserve a concrete host
reserve and peak snapshot/state/scratch accounting, keep any plan bounded to
the exact loaded source/operational profile, and do not relax the generic policy.

The pinned `sha2` 0.10.9 source selects its software backend on aarch64 unless
the assembly feature is enabled; that accounts for a substantial startup cost
while streaming the 109 GB decoder through native content checks. Enable its
CPU-feature-detected backend only for macOS/aarch64, retain the exact library
version and a locked dependency graph, and revalidate identical digests against
the already pinned metadata and complete payloads. Keep the software fallback.
No model arithmetic, activation/cache precision, sampling or state policy may
change as part of hashing acceleration. Startup measurements and source/build
identity remain separate from ordinary text inference measurements.

Status: resource-deferred. Content checking stays mandatory. Neither this investigation
nor the earlier readiness/component results qualify full image generation,
default catalog loading, text cache/performance or either backend's MTP gate.

## Corrected probe and candidate policy

The corrected scope probe reached readiness in 185.79 seconds, generated eight
ordinary text tokens and prepared the 32-pixel fixture into a 112-by-112 canvas
with 16 image tokens. Text output hashes match the separate candidate text-only
control. The actual `generate_image` request then returned
`resource_budget_exceeded`; no image-generation pass is claimed. Available RAM
was 108,584,796,160 bytes before the first text pass and 5,740,347,392 bytes after
lazy target-bank allocation. The private failed-request receipt is
`dcef9fbc32b2559ec5bb733a7a53a1e49402f87d43333e95fc43bc514b2f792f`.

An initial, unadopted candidate separated normal text planning from image-only work. Image
planning retains the complete 1,164,010,176-byte sidecar and 512 MiB transient
allowance. Native GLM planning additionally bills two KDA snapshot copies and
the live/old/replacement host attention states; the earlier inventory omitted
that recurrent snapshot peak. Exact config bytes consumed by the loader are
bound to the pinned finished config, and the mapped decoder must have the four
pinned shard sizes. Only the encoder-bank/resident-bank/wide-encoder/decode-graph
recipe with no bank-group override may select the candidate allocation policy:
checked full request inventory, divisor one, explicit 2 GiB host-memory reserve,
context at most 4,096 and MTP zero. Other sources/recipes retain existing policy.
Image requests select their separately charged image inventory. This is a
candidate to measure, not an allocation reservation or a memory qualification.
The required disk reserve remains 8 GiB.

Independent review rejected adoption: the request inventory omits full remaining
lazy decoder-bank/expert/static residency growth, and a new image can activate
experts not exercised by the first text pass. A successful text request and
observed free RAM do not establish the remaining peak. The size-only text guard
also does not bind exact decoder contents. The host preflight experiment cannot
treat mapped decoder bytes as zero initial allocation. The complete draft is
sealed privately as patch
`8b468bbfb6fc987e70eb1d60436dbe6ad5f9470cbcc42be82e90c49180343bc0`;
runtime and host policy were restored to the conservative baseline. No divisor
or host-reserve relaxation is adopted. The next review must account for all
remaining lazy target/expert/static growth, two snapshots, tower/scratch, cache
lifecycle and a host margin before execution, with exact content-bound selected
descriptors or an explicitly qualified structural contract.

All owned GLM processes closed, and the coordinating agent requested the direct
Qwen handoff before further admission work. Actual image generation, text cache,
matched speed, default loading and MTP remain unqualified. Source investigation
continues while Qwen owns the build/model slot.

## Recovery reconciliation

After the machine restart, the retained worker and helper still match their
sealed identities. Of 1,262 files recorded by the completed worker build,
1,260 match; the only differences are the separately unfinished SHA2 manifest
and lock changes. All four prepared decoder files retain the original
hardlink identities and pinned sizes. Their complete payload hashes were not
repeated during recovery, and no model was loaded. No request or prefix state
survives the restart.

The corrected image failure remains the last completed actual-model attempt.
Neither the restart nor the separately completed hash-vector control closes
its admission or image/cache/performance/MTP gates. Full model execution is held
for review of remaining lazy bank/static residency, complete request peaks and
host margin. The crash cause has not been established.

The private recovery receipt is
`706c3cd31fb93e637d3f4ac940cff7ae2fbb3493fbe5e5105f91fc3eea4ad799`.
The hash-only feature and its bounded digest evidence are recorded separately
before any new worker admission or startup measurement.

## Post-recovery memory review and execution hold

Header-only inspection of the unchanged source inventories the complete
decoder without decoding tensors or loading a model:

| Inventory | Tensor count | Payload bytes | Summed page-rounded Metal spans |
| --- | ---: | ---: | ---: |
| Target expert banks | 126 | 99,033,808,896 | 99,035,873,280 |
| Other target tensors | 1,257 | 6,888,746,232 | 6,909,345,792 |
| Retained predictor | 29 | 2,787,995,776 | 2,788,474,880 |

These span sums can double-count shared pages; actual driver allocations may
also exceed individual spans. They establish an inventory, not measured RSS
or a residency guarantee. The private header/source receipt is
`32b04f2f1bb57fa905ba31a7ea4e1c8e941986cb4f041f97707b37e923e64003`.

The resident recipe budgets all target expert spans, an 8 GiB static reserve
and its encoder-state allowance. At the 4,096-token context cap their source
subtotal is 108,692,065,600 bytes, before the separately loaded image inventory,
host workspace and omitted prefix copies. This already exceeds the earlier
108,584,796,160-byte available-memory baseline. It does not support a safe retry.
The full resident model load remains held; admission policy is unchanged.

One KDA snapshot copies 34 layers of convolution history and recurrent state
into host F32 vectors: 152,633,344 bytes. Two overlapping snapshots require
305,266,688 bytes beyond the live state. Full remaining static growth, attention
copies, tower/scratch and an explicit host margin must also be covered.
Implementing the retained predictor adds its own 2.788 GB payload and request
state; inactive source-head weights do not qualify MTP execution.

### Separate bounded-command investigation

Existing `STARK_GLM53_BOUNDED_GRAPH` execution has a distinct ownership policy
from encoder-bank residency. Each admitted group constructs its own bounded
bank wrappers, completes its graph, drops those wrappers and drains its nested
autorelease pool before the next group. `STARK_GLM53_BANK_GROUP_LAYERS` can
further bound group size using the existing source option. Whole-model encoder
residency must be absent for that candidate; merely limiting encoder groups
while retaining the resident whitelist still budgets the complete model.

This source route is a candidate for a separate memory investigation. Its
default group budget is derived from the recommended device working set and
an 8 GiB static reserve, which alone does not establish current host headroom.
It still requires a complete static/state/image/snapshot/host plan, bounded
driver and paging observations, cancellation cleanup and the existing guards.
No flags, budgets or source policy were changed, and no model was run.

The current optimized chunked prefill requires a single encoder-bank group;
the bounded-command recipe does not inherit that prefill implementation or its
performance qualification. Any future comparison needs all text/image arms
matched to the selected recipe, with fresh cache, arithmetic and lifecycle
checks. The resident recipe's earlier timing cannot be transferred to it.
