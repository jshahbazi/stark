# Model change and experiment ledger

Use this directory to track attempts to make a model work correctly or run faster.
Record unsuccessful, invalid and resource-deferred attempts alongside successful
ones. A later passing run does not erase the earlier result.

This ledger connects a change to its evidence and shipping decision. Numerical
receipts, benchmark aggregates, source patches and detailed run records retain
their existing canonical locations under `docs/archive/`, `docs/benchmarks/` and
`provenance/`. Private model payloads and evaluation data stay ignored.

## Starting records

| Record | Scope |
| --- | --- |
| [Image-support retrospective](2026-11-08-image-support-retrospective.md) | Conversation-wide findings, model coverage, validation mistakes, review, delivery and remaining work |
| [Qwen image attempts](2026-11-08-qwen-images.md) | BF16 and affine dense Qwen, affine and BF16-source MoE, retained Flash-Next/Lily numerical repairs and resource controls |
| [Gemma image attempts](2026-11-08-gemma-images.md) | Gemma 3 processor/convolution fixes; Gemma 4 efficient, unified, MoE and larger-tower attempts; decoder admission and retained compatibility |

The `IMG-20261108-QI-*`, `IMG-20261108-GI-*` and `IMG-20261108-XI-*` IDs register
historical attempts on 2026-11-08. The registration date is not a claim about
when each experiment ran. Original dates and identities remain in the linked
records. This initial batch covers the image-support conversation and its
explicit prerequisites. It is not an exhaustive backfill of all Stark research.

Existing performance work remains indexed in the
[benchmark index](../benchmarks/README.md). The
[dense Qwen3.8 approach ledger](../benchmarks/2026-11-06-qwen38-27b-approaches.md),
[Flash-Next projection experiment](../benchmarks/2026-11-07-flashnext-moe-projections.md),
[sustained-throughput diagnosis](../benchmarks/2026-11-07-flashnext-sustained-diagnosis.md)
and [cross-backend idle-wake study](../benchmarks/2026-11-08-cross-backend-idle-wake.md)
are related records from other work. Their findings must retain their own
checkpoint, host-screening and source boundaries; they are not new measurements
from this retrospective.

## Record an attempt

1. Before changing code or running a comparison, copy the
   [attempt template](TEMPLATE.md) into a dated Markdown record. Choose a unique
   `WORKSTREAM-YYYYMMDD-AREA-NNN` ID (for example, `IMG-20261108-QI-001`)
   and list the model/checkpoint variants it actually
   targets. State the observed problem, hypothesis, baseline, intended change,
   fixed acceptance limits and stop conditions. Use `not recorded` for missing
   historical details; do not reconstruct them from a different run.
2. Bind the checkpoint revision and artifact hashes, baseline and candidate
   commits/executables, source references, tools, device/OS and precision policy.
   Record conversion, kernel, cache, sampler and MTP choices that can affect the
   result. Separate image/tensor/component checks from full-model execution.
3. Run the smallest useful correctness controls first. For a speed claim, retain
   a matching correctness gate and a controlled end-to-end timing comparison.
   Preserve prompt/input lengths, output budget, warm/cold/reuse mode, seed,
   sampling, draft depth, repetitions, paired order and host/resource screening.
   Publish aggregates and hashes, never prompts, generated text, token vectors,
   probabilities, credentials or raw exception dumps.
4. Append what happened, including every failed gate, rejected arm, invalid setup,
   guard stop and known counterexample. Give each changed candidate or materially
   changed measurement recipe its own attempt or follow-up ID. Keep prior
   receipts immutable and link the predecessor.
5. Record the **outcome** and **promotion decision** separately, with reasons and
   the next test. A component pass, synthetic kernel speedup or unscreened timing
   observation does not establish a production full-model gain. Name each exact
   artifact whose qualification remains missing.
6. Add the entry to this index and link its ID from the implementation PR. Update
   the current model-support/reference page only when the evidence supports a
   capability change. When a new comparison supersedes a benchmark, update its
   current/superseded index without changing the historical measurements.

This is a repository documentation workflow, not an automatic experiment
collector or evidence validator. Reviewers must check the linked receipts and
the claimed scope. A record does not authorize a large model run, a download,
publication or promotion by itself.

## Outcome and promotion vocabulary

| Outcome | Meaning |
| --- | --- |
| Planned / untested | Proposed or implemented route without the named execution evidence |
| Passed | The stated gate passed for the exact recorded inputs and build; name whether it is structural, component, full-model, lifecycle or timing evidence |
| Failed | A valid attempt missed its fixed gate; retain the failing metric and diagnostic limits |
| Invalid setup | The intended experiment was not run with matching inputs/tools/builds; results do not decide model correctness or performance |
| Resource-deferred | Admission or a resource guard prevented completion; this is neither a numerical pass nor a numerical failure |
| Inconclusive | Execution completed but controls, coverage or timing eligibility do not support the intended conclusion |

Use a separate decision such as `promoted within named scope`, `default-off`,
`diagnostic only`, `rejected`, `deferred` or `awaiting qualification`. An
intentionally rejected mutation is a passing **admission negative control**, not
a failed implementation. A counterexample can remain unresolved even when a
bounded feature gate passes. Never turn `implemented`, `merged`, `released` or
`CI passed` into a substitute for model qualification.

## Required evidence boundaries

- A checkpoint name, parameter count or catalog row does not identify the actual
  artifact, quantization, processor/template or selected native path.
- Full logit parity, greedy agreement, image-feature similarity, image-answer
  correctness and lifecycle safety answer different questions.
- A BF16 tower with an affine decoder differs from a quantized tower. A typed
  text-only conversion differs from a retained format containing vision.
- Equal output hashes do not establish throughput. Faster component timing does
  not establish faster requests. Memory admission caps and sampled RSS do not
  establish a full-model working-set guarantee.
- Prefix restoration, cancellation, reset, scope ownership and MTP must be
  assessed for every state-affecting route, following
  [model onboarding](../MODEL_ONBOARDING.md).

Backfill older work by linking its existing canonical record first. Mark missing
comparators or receipts explicitly; do not silently fill gaps with current
results from a different build.
