# Gemma 4 assistant conversion and native text MTP plan

Status: source metadata and native interface gaps inspected; strict companion
conversion and actual E2B draft execution are implemented on the continuation
branch. Three real proposals, fixed target position, ordinary target-root
agreement and drained cancellation/retry pass. The first numerical gate failed;
the assistant-only eager-BF16 follow-up now matches the unchanged short-input
oracle exactly. Serial verification controls pass, while actual acceptance is
zero of three drafts on that input. Source-matched assistant MLP reductions
subsequently pass chained oracle and transactional controls at 512, 513, 1,024
and 1,026 tokens, with actual three-of-three acceptance for those fixtures.
Exact prefix restoration passes at 1,026. The raw 1,792-token masked selection
fails on an exact cutoff tie. A separately reviewed mathematical-top-k and
canonical-readout policy passes all 15 chained steps and all five input-size
transaction controls, including exact 1,792-token prefix/suffix restoration.
Original raw failures remain preserved. Worker MTP remains unimplemented. Gemma 4 has upstream
MTP assistants, and remains
an open Stark implementation task. Image loading must preserve future text MTP;
image requests keep zero drafts. Follow [model onboarding](MODEL_ONBOARDING.md)
and the [image-default parity gates](IMAGE_DEFAULT_PARITY_PLAN.md).

## Pinned small-profile intake

The official model APIs, configuration files and 5,520-byte safetensors headers
were read on 2026-11-09. HTTP range responses were required before reading the
headers; no weight payload was downloaded during that metadata intake. The
continuation subsequently fetched and converted the exact E2B payload. E4B
remains metadata-only; numerical and serving qualification remain separate.

| Assistant | Revision | Config SHA-256 | Safetensors header SHA-256 |
| --- | --- | --- | --- |
| [E2B](https://huggingface.co/google/gemma-4-E2B-it-assistant/blob/2d874ef7d29f9a30599a1e4b3c1cbc9595f005df/config.json) | `2d874ef7d29f9a30599a1e4b3c1cbc9595f005df` | `7f42f559a6a69ffaeaf6b61a1ece3a562a2ed5ad00b8d30f16917ba5ab1bcbe9` | `6179f5c61f819bf278f6199e454d2fe5b5846dda74a6cd7adb93672c492960a3` |
| [E4B](https://huggingface.co/google/gemma-4-E4B-it-assistant/blob/8d0031ea8c2109e2b1e86bb9368a4539b537f80a/config.json) | `8d0031ea8c2109e2b1e86bb9368a4539b537f80a` | `13cdf5418e5a89cf951717a953cd11cef4e5c2743c4767e9e5dd72802954f822` | `054af9c56f38d4188b39bee058b5cb17cc1b34646c118f89ea0166ff5dbc8515` |

Both declare `Gemma4AssistantForCausalLM` / `gemma4_assistant`, hidden width 256,
four layers (three sliding, one full), four query heads, 512-token local windows,
131,072-token maximum positions, BF16 matrices, 2,048 centroids and intermediate
centroid top-k 32. All four layers declare KV sharing. E2B declares one KV head,
E4B two. Local query head width is 256; global width is 512 with proportional
RoPE at factor 0.25. The backbone widths differ: 1,536 and 2,560.

The actual header inventories include:

- `pre_projection.weight`: `[256, 3072]` / `[256, 5120]` for E2B/E4B.
- `post_projection.weight`: `[1536, 256]` / `[2560, 256]`.
- Own tied `model.embed_tokens.weight`: `[262144, 256]`.
- `masked_embedding.centroids.weight`: `[2048, 256]`, plus an I64
  `token_ordering` tensor of 262,144 entries.
- Query-only attention: local Q/O projections `[1024, 256]` / `[256, 1024]`,
  global Q/O `[2048, 256]` / `[256, 2048]`, learned Q norms, and no K/V projections.
- Four gated MLPs with intermediate width 2,048, source layer scalars and four
  normalization vectors per layer, plus final normalization.

API dtype inventories are 77,731,332 BF16 plus 262,144 I64 values for E2B, and
78,517,764 BF16 plus 262,144 I64 values for E4B. Their declared tensor payloads
are 157,559,816 and 159,132,680 bytes respectively, before headers/alignment.
Full E2B payload hashes and exact target/assistant pairing are recorded in the
continuation; E4B still requires payload intake. Metadata compatibility alone
does not qualify a pair. Initial target revisions
are the independently recorded E2B `3e22461f…` and E4B `4ecc2ba6…` artifacts;
retain their exact precision/materialization policies throughout each comparison.

Google's [architecture](https://ai.google.dev/gemma/docs/mtp/overview) and
[usage guide](https://ai.google.dev/gemma/docs/mtp/mtp) identify assistants for
all five Gemma 4 sizes. The larger 12B/26B/31B assistants need separate metadata,
target execution and qualification; small-profile evidence will not transfer.

## Native interfaces missing today

| Interface | Existing limitation | Required contract |
| --- | --- | --- |
| Target activation handoff | The current dense MTP implementation is Qwen-specific. Gemma text prefill deliberately skips trailing shared-KV query layers when their logits are unused. | Capture the exact source-defined target activation needed by the assistant at every committed handoff/boundary; preserve those prefill optimizations when possible. Validate normalization, BF16 boundaries, shift/position alignment and post-projection feedback against a pinned independent oracle. |
| Cross-model KV views | `TransformerState` sharing refers to owning layers inside the same transformer. Assistant width differs from the backbone, and its query-only layers read backbone KV. | A typed, lifetime-bounded, read-only borrow of the selected owning local/global target caches, carrying loaded-model identity, absolute positions, head geometry, precision, ring offset/length/source origin and capacity. No mutable or unvalidated state escape. Freeze exact source owner selection from the oracle. |
| Assistant graph and readout | `QwenMtp` cannot represent this four-layer projected graph or its ordered clustered vocabulary. | Dedicated assistant config/loader/encoding, target embedding handoff, source projection/norm/scalar arithmetic, own readout embeddings, centroid selection and ordering validation. Keep the complete target distribution for exact rejection sampling. |
| Target verification/commit | Gemma's image/scalar embedding path currently rejects the generic speculative target route. Sliding rings cannot discard overwritten committed rows by scalar rewind. | Text-only candidate verification with exact accepted-prefix commit and rollback/snapshots of every overwritten target/assistant state. Keep image requests excluded. Qualify useful end-to-end speed; serial verification alone does not establish MTP acceleration. |
| Prefix reuse and lifecycle | Current Gemma text snapshots preserve target K/V/ring state, without an assistant handoff/session. | Extend the retained entry with required target activation and assistant state/position. Bind exact head/readout policy and MTP selection to the cache key. Image, cancellation/error, context/scope/model changes clear the complete entry; following text re-enables the selected text policy. |

## Ordered implementation stages

| Stage | Deliverable | Acceptance before advancing |
| --- | --- | --- |
| A: frozen reference and conversion contract | Pin official implementation revision and a CPU oracle; inventory and hash the complete small assistant payloads. Define separate typed companion-artifact identity and exact target binding. Validate every declared tensor, dtype/shape, projection, ordering/index range and norm transform with bounded conversion workspace. | Deterministic conversion plus negative malformed/unsupported/target-mismatch controls. Record source licenses/revisions. No chat capability from a companion-only model. |
| B: handoff and external KV | Implement the typed target-activation and cache-view interfaces above. Preserve source target precision and prefill behavior; add boundary-last-row materialization only where its exact state can be restored. | CPU/Metal synthetic ownership, positions, shared-KV and wrap tests; actual pinned target activation/KV comparison. Establish actual-state accounting and compare MTP-off target numerics/performance before exposing drafting. |
| C: one draft step | Implement the four-layer assistant, pre/post projections and clustered readout with explicit validated target context. | Source-bound next-draft logits/candidates and feedback-feature comparisons, including ring wrap and global/local owner selection. Candidate ordering is validated; untrusted indexes cannot address outside the vocabulary. |
| D: exact text speculation | Integrate bounded multi-draft sampling, target verification, accepted-prefix commit/repair and cancellation draining. Keep source target sampling/rejection policies exact and draft RNG separate. | Greedy equivalence, seeded fresh/cached reproducibility and independent rejection/distribution controls; zero/all/partial acceptance; depth changes; rollback and context edges. Different stochastic algorithms need not emit identical sequences at the same seed. Require actual draft proposals and useful measured speed at matched settings. |
| E: cache, images and default loading | Load target+tower+assistant by the ordinary image-default route. Inventory resident weights, target/ring snapshots, activation handoffs, assistant state and verification/scratch peaks. Integrate complete head-aware prefix reuse and host capability reporting. | Text MTP/cache→real image with zero drafts/proposals/reuse→fresh text MTP→cached text MTP; failures/cancellation/restart and MTP selection switches. Match text-only versus image-loaded speed with MTP off/on. Head stripping or a text-only fallback cannot close this stage. |
| F: E4B and later artifacts | Repeat complete qualification for E4B, then larger targets and each intended precision/format. | Each exact pair has its own artifact/binary/reference hashes, numerical, memory, cache/lifecycle and performance evidence. Release/install status remains separate. |

The [continuation](model-experiments/2026-11-09-gemma-next-image-parity.md)
records exact E2B payload conversion, source/target binding and complete
companion tensor/permutation admission. Three real drafts execute, while the
source numerical gate and subsequent qualification stages remain open. No MTP
speed result has passed. The pinned Transformers 5.15.1 candidate generator and assistant
confirm that all four layers borrow target KV without maintaining predictor KV,
and all proposals within a round use the same absolute rotary position. The
target's final normalized activation and embedding feed the first proposal;
later proposals chain the assistant's post-projected activation. The continuation
contains native four-layer/pre/post-projection and centroid-mask components,
plus a target frame that captures only the final normalized prefill activation.
These compile and execute three real E2B proposals with the image tower resident.
The first source oracle passes root-embedding equality, first-proposal equality,
feedback and centroid bounds, but fails the masked-logit bound. The separate
eager-BF16 score/probability follow-up matches all three numerical planes
exactly, without relaxing a tolerance. See the [continuation attempt](model-experiments/2026-11-09-gemma-next-image-parity.md)
for separate passed and failed controls. The source BF16 committed-query
follow-up and consuming serial target-verification component compile, and the
dedicated committed-query CPU/Metal, serial verification and short-input
source-oracle controls pass. Normal-worker lifecycle/cache and useful-speed
qualification remain open. Drafting positions above 512
are rejected pending a separately qualified extra-row handoff; ordinary target
context support is unchanged. Batched verification/rollback, head-aware caching,
worker/default loading and useful MTP speed remain open. The current
[cache follow-up](benchmarks/2026-11-09-gemma4-warmed-cache.md) addresses normal
text prefix execution while this assistant work remains outstanding.

The G4AP-03 continuation adds a separately retained preaccept 513-row local
handoff and full-accept transactional target batches with physical snapshot
restoration and serial repair. Six synthetic CPU/Metal controls pass. The first
512-token three-step oracle fails masked logits and chained feedback without
relaxing its limits, so actual batch/cache and later long-input controls have
not executed. The public 512-position draft guard remains in force while the
first divergent intermediate is diagnosed. These additions grant no serving
capability or performance qualification.

The assistant-only source GELU, separate BF16 product and source-reference MLP
reductions are implemented and independently reviewed. Per-projection reference
candidates match all four layers across three steps on identical native inputs.
The corrected head passes the unchanged chained oracle and transactional
controls at 512/513/1,024/1,026 tokens. Exact 1,024-boundary restoration and
changed-tail checks pass at 1,026. An actual-mask proof confirms a valid exact
cutoff tie at 1,792; the raw masked-vector comparison remains failed. The
separately versioned and independently reviewed mathematical-top-k/canonical
readout policy passes all chained steps at the unchanged numerical limits.
The 1,792-token batch, rollback, cancellation and 1,024-boundary/768-row-suffix
cache controls now pass. Reproducible source and negative CPU controls live in
[validation tooling](../tools/validation/gemma4_assistant/README.md). Other
assistant linears and ordinary target math retain their existing policy.
Normal-worker MTP, supported sampling, full lifecycle and matched performance
qualification remain open. Serving remains disabled, and the public position
guard remains in force pending bounded worker integration.
