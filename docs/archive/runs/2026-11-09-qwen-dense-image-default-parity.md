# Qwen dense default-image text parity

## 20261109-QDIP-01: preserve the existing dense text predictor

Registered 2026-11-09 after the initial source edits, before executing tests or
model-bound comparisons. This is an implementation and qualification attempt
for IMGPAR-01/02/03 in the [worklist](../../IMAGE_DEFAULT_PARITY_PLAN.md).

| Field | State before validation |
| --- | --- |
| Baseline | Stark `5f702450`, containing merged PR #110 and the standing image-default requirements |
| Candidate | `codex/qwen-dense-image-parity`; executable and patch identities to be sealed before model runs |
| Observed gap | The affine dense text head is explicitly discarded by image conversion; the vision loader and worker do not expose dense text MTP |
| Intended implementation | Preserve a complete declared affine dense head; bound native admission to the admitted target graph and exact tensor set; route text through the existing exact MTP session while the tower is resident |
| Source scope | Existing compatible dense affine predictor, beginning with `Jundot/Qwen3.8-27B-oQ4e-mtp`; unquantized and expert predictor support remain separate implementation items |
| Completion state | Functional dense affine head/cache controls passed on the named 27B artifact; family-wide text speed parity remains open |

Image requests retain zero MTP drafts/proposals and clear retained text state.
Text prefix entries include the selected draft depth. A failed native predictor
operation fences the image wrapper; preflight-invalid requests do not submit
work. Drained cancellation disposes its request session and permits a fresh
request on the same loaded instance. Normal image loading always allocates the tower. A validation-only engine
entry point omits tower allocation after the same full artifact admission and
retains the same decoder/predictor configuration and numeric profile.

## Fixed controls and acceptance

- Model-free conversion tests must preserve all declared head parameters and
  reject missing, extra or undeclared head tensors. Existing headless exports
  remain headless; a declared source without actual head weights is not proof
  of compatible MTP. The MoE expert head must not be interpreted as dense.
- Native admission must reject altered capture layers, predictor operations,
  geometry, unknown fields, malformed packed planes or extra tensors before
  allocation. Legacy artifacts without a head remain accepted.
- Actual-artifact cache checks require exact token/output hashes with correct
  boundary reuse and invalidations. MTP checks require actual proposals, text
  MTP/cache → image zero drafts/proposals/reuse → fresh text MTP → cached text.
- Compare the same admitted target with tower omitted/resident at matching
  sampling, context/output budgets, draft depth and precision. Keep the previous
  qualified text export as a separate arm because its numerical profile and
  normalization materialization differ. No claim of old-text parity follows
  merely from matching the two candidate loading modes.
- Run serialized, one warmup and at least three measured requests per arm,
  alternating loading modes. A short check is a bounded observation. A speed
  qualification requires baseline variability and a predeclared bound; otherwise
  report inconclusive or not measured. Loading/hash verification is excluded
  from request time and recorded separately.
- Existing artifacts stay unchanged. Models, vectors, prompts and responses are
  private; publish only aggregate counts/timings and hashes. Stop on native
  execution failure, paging growth or unavailable memory; do not alter another
  task's process. GPU/build work follows the coordinator's serialized slot.

## Implementation boundary

The prior text graph keeps its existing batched verifier. The image reference
graph does not admit non-flat M-RoPE execution or reference DeltaNet tree
verification. Its text-only fallback executes the real draft head and verifies
one fully accepted target row at a time. After the first rejection it stops
verification, rewinds discarded predictor guesses and repairs committed pairs
with the preceding target residuals. It never enables speculative image input.

This is a correctness foundation. Serial verification does not provide the
optimized text graph's batched-verification acceleration. Image conversion also
selects reference activation/materialization, normalization, attention and
affine projection policies that differ from the earlier text export. Retaining
the head and showing tower-residency equality cannot close that speed gap.

## Exact artifact and execution

| Field | Executed value |
| --- | --- |
| Checkpoint | `Jundot/Qwen3.8-27B-oQ4e-mtp` at `7dd276b77fcb27c982ebd8d8a434539345713fea` |
| Converted format | Schema-3 dense affine version-3 image artifact with complete declared predictor |
| Model file | 17,999,537,280 bytes; 1,639 tensors; SHA-256 `6c64740ac1d09365b0099f868114e5c495bac0d1366c27e842c87a9f3a5db647` |
| Head payload | 323,524,608 bytes, included in the model file |
| Configuration SHA-256 | `b4bb4776365308b3e157e21e5fe7777e5650f328879a3f0a4b9981ef978fa2da` |
| Tokenizer / processor / template SHA-256 | `87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4` / `6a970fd06f30e6943b3e2c14d5d3b42d49b06cf99b99103d56689bef462d90f8` / `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041` |
| Converter sources / lock SHA-256 | `b8ae553051a6a468a09081f3c65beb3e701a2ea61f5ba2cd1253930451155f24` / `df7711726b1ed7e235ded21b9e6e1cbe130924be4d2c580d5d6bd9005248f8d5` |
| Source manifest SHA-256 | `94398c017642c65b666ff2b48383adc765d6928bf9dc98de420782e26a1a5c41` |
| Actual worker SHA-256 | `e41bc8b07681cacb9c6c66d352dd5805cf64baa2285b8287d27edde4a1867d67` |
| Hardware / OS | Apple M5 Max, 40 GPU cores, 128 GiB physical memory; macOS 26.7.2 |
| Arithmetic | Original mixed affine target/head codes, BF16 tower and activation boundaries, reference affine policy, F32 DeltaNet recurrent/convolution snapshots, BF16 attention KV |
| Native loading | Full artifact admission and two selected-file hash passes; about 60–61 seconds per load, excluded from request timings |
| Readiness | Image execution and MTP supported; image qualification remains `unverified`; worker default drafts remain zero |

The new conversion used an eight-GiB disk reserve and preserved every existing
model. Peak RSS, thermal eligibility and paging deltas were not qualified.

## Results and attempt history

| Attempt / control | Result | Evidence and limit |
| --- | --- | --- |
| QDIP-01 complete conversion/head admission | Passed | Same source revision; complete predictor is retained and consumed by native loading |
| QDIP-01 short greedy target controls | 5/5 passed | Eight generated tokens at depths 0, 1 and 3 have identical hashes. Depth 1 proposed/accepted 3 drafts; depth 3 proposed/accepted 5. A real image used zero drafts/proposals; following text resumed depth 3 with the same hash |
| QDIP-03 initial long driver | Invalid setup, rejected before inference | An unknown `progress` field returned `invalid_request`. Receipt retained: `6cb12fbd5e374c2b2a599afdcbee935aee528833c88bcf9ee10a4d62ce9e8ea0` |
| QDIP-03 corrected long driver | 9/9 passed | Fixed criteria were encoded before execution; this public follow-up record was updated after execution. 1,032 input tokens, context 2,048, four output tokens, greedy seed 42; scope and draft depth bound reuse |
| QDIP-03 cache sequence | Exact matched-mode hashes | Cold depth 3 → 1,024-row reuse → switch to depth 0 misses → depth 0 reuses 1,024 → real image with zero drafts/proposals/reuse → fresh depth 3 → 1,024-row reuse |
| QDIP-03 worker cancellation | Passed within bounded scope | Cancellation at the first progress frame clears state, then fresh text MTP succeeds. This is a pre-inference worker cancellation control; it does not prove cancellation during native work |
| QDIP-04 native drained cancellation | Passed | A direct wrapper control cancels after one complete seven-token target/predictor prefill chunk has finished and drained. The same loaded model stays unfenced and creates a successful fresh MTP session; invalid preflight input also remains recoverable |
| Initial ignored-test build | Rejected, corrected | The custom test macro accepts `#[ignore]`, not the reason-bearing attribute form. Corrected form compiled and the three new contract tests passed |
| QDIP-05/06 text speed control | Failed setup, then failed bounded parity | A removed long-chain configuration was correctly refused before inference. A separately bound current text control matches the current decoder/head plan; ABBA timing then measured 61.091 versus 3.394 decode tok/s with identical outputs. See the [benchmark record](../../benchmarks/2026-11-09-qwen-dense-image-text-parity.md) |
| QDIP-02 same-graph tower control | Passed exact-output control; no material decode cost observed | BF16 0.8B: 60.1967/60.1963 tok/s tower off/on. Affine 4-bit 0.8B: 59.9266/59.9866 tok/s. These iterator controls exclude worker/state setup and do not qualify a changed text graph |
| Model-free checks | Passed | 750 final focused Python tests; 134 worker tests with two ignored; 16 native family/header/geometry tests including the three new head contracts; repository policy and formatting |

The short private receipt SHA-256 is
`bd54b8b0daf5ffc6ccfe5f1558c8268b9243e02da1ec8db0f4f18ca2db5cd0c3`.
The corrected long receipt SHA-256 is
`cc461911136ea22ca781872ec55e762035956b1391fa707a1e52d40882866261`.
The direct native cancellation receipt SHA-256 is
`df7177d4d11a8588c7fc03fd935a252e7852670ad7ff7b141d5db8cee2f6cd2b`.
Private receipts retain aggregate hashes, counts and timings, not prompts,
responses, token vectors or probabilities.

Long fresh requests took 185.694–190.036 seconds and exact cached requests
2.733–2.755 seconds. These unscreened functional-run observations include
prefill, output and worker handling; they are not a paired performance study
and do not establish parity against the earlier optimized text export.

## Remaining gates

- **Text performance remains unfinished.** Qualify a fast plain-text graph while
  the image tower is resident, preserving the previous target arithmetic and
  supported batched MTP policy or numerically qualifying a replacement. Account
  for shared weight ownership and memory admission; duplicating every loaded
  decoder is not an established memory solution.
- Run the matched tower-off/on control against the exact admitted target, then
  separately compare with the prior qualified text export. Keep these questions
  separate: resident tower overhead, changed target arithmetic and speculative
  verification performance.
- The existing BF16 0.8B image artifact and packed 0.8B image artifact remain
  separate qualification scopes. The actual packed 0.8B source index declares
  one MTP layer but contains zero MTP-named tensors among 847 entries; this
  export does not contain a compatible executable head. This is not a statement
  about all upstream Qwen3.5 checkpoints.
- Other BF16 sizes, affine widths/layouts, fine-tunes, BF16 predictors, expert
  predictors, stochastic same-mode cache controls, broader image numerics,
  cancellation/device faults and normal host installation need their own gates.

The named implementation is adopted on this branch as a bounded foundation.
This record does not establish family-wide parity, merge, release or installed
behavior.

Independent source review covered canonical head admission, loaded-wrapper
fencing/cancellation, predictor repair in the serial verifier and worker cache
ownership. The cancellation review found that a drained cancelled session must
not permanently fence its loaded model; the implementation and QDIP-04 control
address that case. A final formatted production rebuild reproduced the exact
worker hash used by the model-bound runs above.
