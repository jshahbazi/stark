# 20261110-DENSE-01: capped attention and dense sampling correctness

Registered on 2026-11-10 after implementation and before validation execution.
The baseline is `746e482f` (main after PR 121). The candidate is the
`codex/model-family-correctness` branch; the final source identity and checks
belong in its dated validation record. No model-family performance claim is made.

| Field | Value |
| --- | --- |
| Kind | Correctness, sampling, state and memory |
| Attempt state | Implemented; component/worker controls passed, production artifact qualification pending |
| Evidence scope | Planned component, worker validation and bounded artifact controls |
| Promotion decision | Adopted on branch at component scope |
| Disposition | Unmerged source; no release or installation |
| Canonical validation | [Dense correctness validation](../archive/runs/2026-11-10-dense-attention-sampling-correctness.md) |

## Hypothesis and change

Gemma 2's emitted attention cap was dropped when constructing the attention
kernel. Threading the cap to the existing online attention kernel should apply
`cap * tanh(scaled_qk / cap)` before masking and softmax. Capped calls select this
kernel for every batch size; uncapped dispatch selection stays as before.
Scores and online normalization accumulate in F32, and outputs retain the
configured F32 or BF16 storage. Weight, activation, KV and recurrent-state
precision policies otherwise stay as before. The capped prefill performance
consequence is unmeasured; no quadratic score/probability scratch is added.

The worker should reject temperatures whose F32 reciprocal overflows as a
recoverable request error before native dispatch. The dense GPU sampler should
emit an invalid sentinel for nonfinite raw or temperature-scaled logits or an empty candidate mask;
the stream should report an error and fence state rather than publish a token.
Existing embedding bounds make speculative device-input lookahead safe while
the host waits to inspect that result. All generated work still drains.
Ordinary text exposes its request-state fence after cleanup, and the worker
checks it before publishing a successful reply or retaining any prefix cache.
Submitted prefill/decode owners drain on early errors in text as well as images,
including grammar-backed lookahead submitted before sample resolution.

Dense Qwen MTP and Gemma assistant MTP should intersect top-k with nucleus mass evaluated against the full
vocabulary, matching ordinary dense sampling's existing filter semantics.
Native hybrid sampling retains its separately qualified top-k-first semantics.
This is a filter-contract correction, not a claim that the CPU MTP sampler and
GPU ordinary sampler use identical RNGs or probability rounding.

A dense MTP request without a cache scope cannot retain a prefix. Its fresh
prefill now omits snapshot copying. Cache-enabled prefill and restoration retain
their existing boundary grid and identity checks.

## Fixed validation gates

- F32/BF16 capped attention compares with an independent F64 scaled-tanh-softmax
  oracle: D64/128/256, one- and nine-query batches, zero/nonzero prefixes,
  a prefix above the previous single/two-pass boundary, sliding windows and a
  wrapped ring. Maximum output absolute error limits are `2e-5` for F32 and
  `0.008` for BF16; unused output components must be exactly zero.
- Greedy and stochastic sampling must emit the sentinel for all-NaN,
  all-infinite, mixed finite/NaN, mixed finite/infinite and empty-mask cases;
  a mask admitting exactly one valid token must still return it.
- Resolving a device sentinel must error before adding any token to host history.
- Finite raw logits whose temperature multiplication overflows must emit the
  same sentinel on CPU and Metal (`temperature=1e-38`, three synthetic logits).
- Early failed sample resolution must drain all still-local lookahead
  submissions before releasing their pool. A genuine tiny-model text stream
  with a test-injected completed lookahead sentinel must fence its state on
  both `next()` and Drop-only detection, retain no pending token/prefix and
  reject continuation of that state.
- The existing bad-request worker test must reject `temperature: 1e-40` without
  calling native generation, then successfully process the next valid request.
- Four equal logits, `top_k=2` and `top_p=0.4` must retain two candidates in
  dense MTP and one under the existing hybrid policy. Draft acceptance and the
  bonus draw must remain within the dense candidate set over 64 seeds.
- The existing private dense-MTP toy qualification must compare capture-off and
  capture-on prefill logits and greedy MTP continuation exactly, with zero
  snapshot payload when capture is off and a 1,024-token snapshot when on.

## Artifact scope and limits

Component tests are synthetic and do not qualify a production checkpoint.
Gemma 2 is a text family in the converter; its attention correction does not add
an image bridge or MTP head. Its existing exact attention-prefix restoration
requires no new state for the cap: the cap is immutable loaded-model
configuration, and attention state contains the same KV/ring representation.
Each converted artifact still needs its own model-bound numerical proof.

The snapshot control changes only whether an optional copy is created. It
retains exact-state restoration when the host requests caching. The synthetic
dense-MTP fixture is a correctness control, not production quantization or
image-loaded text qualification. Default image-loaded Qwen paths, actual head
drafts, image recovery, peak memory and speed remain under the image-family
acceptance matrix and require separate receipts.

Toolchains and locks follow `CONTRIBUTING.md`: Rust 1.98.1 for worker builds and
`nightly-2026-10-01` for engine tests and formatting. Root coordinates one build
directory for this worktree and serialized native execution; no downloads or
unrelated workers are part of this attempt. Prompts, responses, token IDs,
vectors and probabilities remain private; public records use aggregate counts
and hashes only.

## Final outcome

Runtime source `b4120152` passes the registered CPU/Metal cap, invalid sampling,
lookahead cleanup/fencing and full-vocabulary filter controls. The synthetic
MTP fixture passes both explicit controls, including capture-off/on and exact
continuation. Release generator tests pass 144 with four ignored; actual dense
same-worker tiny-temperature recovery also passes on the retained Qwen3-4B Q4
artifact. The complete counts, sealed executable/artifact identities, initial
failures and remaining production/image acceptance gates are in the
[aggregate record](../archive/runs/2026-11-10-model-family-triage.md).
No speed, peak-memory or broad Gemma 2 fidelity claim is made.
