# 20261116-DFHTTP-01: native Decision HTTP transport and latency profile

## Identity and state

| Field | Value |
| --- | --- |
| Registered date / owner | 2026-11-16 / Codex |
| Kind | Compatibility, lifecycle and performance diagnosis |
| State | Implemented on branch; full checks, actual-artifact execution and installation pending |
| Acceptance | Authenticated buffered decisions use the existing single-worker FIFO; no request content in logs; image and cancellation lifecycle preserved |
| Evidence scope | Model-free HTTP/controller checks; live profiling pending |
| Promotion | HTTP transport adopted on branch; no native arithmetic or performance promotion |
| Related work | [Decider native qualification](2026-11-16-decider-flash-v13.md), [app integration](2026-11-16-decider-flash-app.md) |
| Implementation branch | `codex/decider-http-latency`; no PR requested |

## Hypothesis and change

Decider was available only through the app's private Decision controller and
native frontend. Add `POST /v1/decisions` and compatibility alias
`/v1/systemone`, using the persisted API key, existing HTTP resource admission,
the same FIFO slot as chat/Playground, and the already loaded native worker.
Managed chat capabilities retain their existing route scope. Preserve all
three native question types and one inline PNG/JPEG image; each question still
performs its own forward pass. Active cancellation reaps the worker because
this backend has no cancellation pipe. Queued cancellation leaves it running.

The latency hypothesis is that full-sequence native inference dominates the
reported roughly three-second call. The measurement will separate HTTP wall,
template/token encoding, worker round trip, native state allocation, forward
and readout. This is diagnosis without a speedup candidate or performance gate.

## Artifact, source and environment

Use the existing private `native-v2` artifact for
`perplexity-ai/pplx-decider-flash-v1.3` at upstream revision
`d06fb3f2d97e0a52325a0d463c450cf0338f44fe`. The profiler records config,
weights, tokenizer, template, worker and template-helper SHA-256 identities,
Stark revision/dirty status and OS. No weights, kernels, precision policies or
native protocols change. The retained artifact uses BF16 matrices/activations
and its existing F32 reductions and recurrent state. Exact run bindings and
results will be appended separately after execution; qualification does not
transfer between builds.

## Workload and controls

- Retain the image tower for text measurements. Include one actual inline PNG
  smoke request, separately from text timing; no new vision-quality claim.
- Prefix reuse remains invalid because full-attention representations depend on
  later question tokens. There is no result cache. Generation/MTP are N/A for
  this classification checkpoint with its independent 255-row head.
- Synthetic deterministic unique records at 0, 64 and 256 state-word counts;
  publish exact expanded input counts. No output tokens are generated.
- One loaded model, one HTTP call at a time; exclude load and one warmup per
  length, then measure three repetitions. Report medians and ranges. These are
  bounded observations, not a throughput or general performance qualification.
- Reject competing model processes before loading and each sample; inspect
  per-process Metal counter deltas over one second, allowing named desktop UI
  clients only. Leave foreign processes untouched. Aggregate utilization alone
  does not establish which application is using the GPU.
- Use an ephemeral listener, random in-memory API key and disposable private
  model registry. Never access the installed app's credential or request data.
- Publish only aggregate timing, counts and identities. Inputs, token vectors,
  logits, probabilities and keys stay in memory/private worker pipes.

## Attempt history and validation

| Trial | Observation | Disposition |
| --- | --- | --- |
| Initial broad controller run | Missing native template helper from the test PATH; model import setup failed | Invalid setup; no inference occurred |
| Corrected controller run | Built native template helper with Rust 1.98.1; 368 passed, 1 skipped with installed helper on PATH | Model-free regression evidence |
| Focused new HTTP/Decider checks | 44 passed | Synthetic worker HTTP, schema, image staging and cancellation evidence |
| Existing HTTP/client/media checks | 68 passed | Existing auth, stream and image admission regression evidence |

Repository policy, targeted Ruff and diff checks passed. Full converter checks,
actual-artifact profiling, packaged execution and installation remain pending.
Record those outcomes and any failures in the dated run linked below.
