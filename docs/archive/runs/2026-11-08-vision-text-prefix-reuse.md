# 20261108-VTP-01: text prefix reuse on vision artifacts

## Identity and acceptance

Registered on 2026-11-08 during implementation, before model-bound execution.
Owner: Codex. Kind: prefix reuse and lifecycle correctness. State: in progress.
Disposition: implemented on `codex/vision-text-prefix-reuse`, awaiting checks.
No installed-build or release claim is made.

The catalog's image-capable download previously selected a worker route that
always recreated decoder state, including for ordinary text. The hypothesis is
that text-only requests can use the admitted decoder's existing exact prefix
restoration without executing the vision tower or retaining image state.

Baseline: `540a21a9` plus catalog-only commits through `f77100f5` (no Rust or
converter changes). Candidate: the branch patch recorded with the results below.
No model weights, kernels, accumulation policy or sampling algorithm are changed.

Fixed correctness gate: successful cached greedy and seeded stochastic results
must exactly match the candidate with caching disabled and the baseline worker;
every advertised reused-token count must match the existing 1,024-token boundary
policy. Scope/context changes, reset, errors, cancellation, image operations and
restart must clear reuse. One retained state, publication after successful reply,
same loaded model identity and no persistence remain required.

## Planned artifact and state inventory

Primary artifact: existing official `Qwen/Qwen3.5-0.8B` schema-2 vision artifact,
revision `2fc06364715b967f1860aea9cf38778875588b17`, BF16 weights and activations,
F32 recurrent state, existing reference accumulation and KV policy unchanged.
Hashes, worker binaries, lockfiles, hardware and OS will be bound by the private
validator receipt. Other artifacts do not inherit its numerical qualification.

The decoder owns attention KV and positions, DeltaNet recurrence and convolution
history. Reuse delegates to its existing non-final prefill boundary snapshot and
restoration, with unchanged chunk grouping and sampler reseeding. Image-prefilled
states also carry image-position/multimodal markers and are excluded. Image
transactions continue to invalidate the worker entry before preparation or use.
The wrapper must enforce model identity, failure fencing and image-state rejection
without exposing a mutable underlying language model.

Gemma 3/4 and other admitted decoder profiles require a separate state-layout
assessment. MTP requires actual retained and qualified draft heads; the image
artifact does not gain them through prefix caching. Concrete dispositions will
be recorded below.

## Workload and limits

Use the model's own template with synthetic agent sessions. All prompts,
completions and token vectors remain in memory; write aggregates and hashes only.
Workers run sequentially, with a private per-worker image staging directory.
Use the validator's greedy and seeded exactness controls, chunk-edge and prefix
divergence cases, and a completed synthetic image transaction where supported.
Do not download another model or alter another task's processes.

Performance is not measured until a separate controlled timing run is recorded.
Functional equality and reused counts do not establish a throughput gain.
Stop a model run on failed startup, timeout, correctness mismatch or resource
rejection; record the failure before changing the recipe or implementation.

## Attempt history

| Trial | Observation | Validity and disposition |
| --- | --- | --- |
| Initial baseline build invocation | `cargo` was absent from the shell PATH; retry uses the installed absolute cargo path | Setup only; no inference ran |
| Initial engine check | Rewind error import used the wrong module path; corrected to the state module | Compile setup failure, before model execution |
| Source review | Capture position zero and rearming a larger boundary required explicit handling; restored-state proof now survives dropping the old snapshot | Fixed before model execution; numerical gate still required |

## Results and promotion

Pending implementation and validation. No full-model timing, memory peak,
quality comparison, additional artifact qualification or installation yet.

The [ledger](../../model-experiments/README.md) indexes this attempt.
