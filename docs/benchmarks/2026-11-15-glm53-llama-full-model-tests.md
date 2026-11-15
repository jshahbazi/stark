# GLM-5.3 full-model tests of llama-inspired execution paths

Date: 2026-11-15. Attempt `20261115-LCPPGLM-03`.

## Preregistered scope

The user authorized testing the first three recommendations in the
[GLM assessment](2026-11-15-llama-v1-glm53-assessment.md): native CPU-routing
defaults versus the admitted GPU-bank profile, quantized MMA/MLA prefill, and
register KDA prefill. This study restores the exact native UD-Q2_K_XL artifact
and measures the existing execution options. It does not substitute EXL3 or
another quantization. New combinations are tested only when their individual
arms pass the declared controls and show a useful signal.

Model: `unsloth/GLM-5.3-Flash-GGUF` revision
`a38483c8cd5df544f53d70fb281afe97369d5ab6`, four UD-Q2_K_XL shards,
108,720,071,552 bytes. Metadata/tokenizer:
`zai-org/GLM-5.3-Flash` revision
`04c4e9e95c5da8862dced7e5056455116f83a7e0`.
Download uses the pinned existing preparation helper and one ignored staging
directory, atomically renamed after metadata preparation. All four full hashes
must match the pinned LFS identities before model execution.

## Fixed controls and stop rules

- Keep current concurrent host work running. Do not wait for quiet, discard
  busy requests, or retry unfavorable measured observations. Record telemetry
  failures as unavailable. Actual native geometry/memory/resource guards and
  allocation failures remain authoritative; never disable them for a result.
- Use the same source-bound Rust 1.98.1 release worker across arms. Build the
  generator separately from the scorer; dependencies stay locked and targets
  are private to this checkout.
- Preserve source weights/metadata, F32 activation/state policy, greedy
  sampling, input/context/output budgets and frozen corpus bytes. Text-only
  native GLM has no image execution; native MTP is unsupported and remains off.
- Remove inherited inference/capture/profile selectors from the study process.
  Obtain the complete profile environment from the existing wrapper. Keep
  `EXPERT_PREFILL_BATCH`, `PREFILL_DENSE_WIDE`, wide-prefill and unrelated
  candidates off. Restart the worker when changing load-time options.
- Full-vocabulary correctness comparisons use anonymous pipes, RAM only, the
  existing `1e-3 + 1e-3 * abs(reference)` gate, all finite values and five/five
  greedy agreement at each declared prefix. Record exactness separately.
  Validate candidate prefix reuse against a matching fresh run before treating
  it as a usable state-affecting profile.
- Add only aggregate CPU/GPU logical-route invocation markers to the existing
  opt-in profiler. `Counter.calls` counts rows; it is not a dispatch, duration,
  byte-traffic or utilization measurement. Include them in the readout
  diagnostic. All profiling remains disabled in speed arms.
- A profile/load/admission failure is a retained failed or resource-deferred
  attempt. A parity failure stops that candidate's performance claim; no
  relaxed threshold or replacement input is chosen to obtain a pass.

## Ordered matrix

1. **Numerical/route diagnostics:** default and complete profile, MMA off within
   the complete profile, MLA on alone, KDA-register on alone. Freeze prefix
   lengths 129 and 513 for five full-vocabulary readouts each. Existing T8 is
   the reference for MLA/KDA; normal defaults are the macro profile reference.
   Require positive route counters for any claimed selected feature. The direct
   readout helper uses the same model/backend and selectors; it bypasses the
   worker's request-context and memory-admission checks. Timed and prefix-control
   requests must independently pass those worker checks. Loading default GLM's
   source context limit does not preallocate its entire attention capacity;
   diagnostic state grows only for the bounded processed prefix.
2. **Cheap screen:** two balanced A/B pairs, one excluded warmup per workload
   and loaded session, at 96 input/8 output and 512 input/16 output for the
   whole-profile comparison. Isolate each prefill factor at 512/16. Keep the
   complete profile otherwise fixed. MMA off/on changes prefill orchestration
   as well as matrix selection; label it a route comparison, not one kernel.
3. **Confirmation only for survivors:** eight balanced paired observations per
   arm at 512/16 and 3,072/16, plus 512/128 for a claimed general decode gain.
   A general decode claim requires all long-decode observations to reach 128
   sampled tokens; an earlier EOS does not satisfy that workload.
   Total context stays below the complete profile's 4,096-token cap. A
   combination requires independently passing numerical controls and a new
   comparison against the already measured reference.

Each comparison uses four fresh loaded sessions in ABBA order, reversing
workload order in odd sessions. Screening retains one measured request per
workload/session (two observations per arm); confirmation retains four
(eight observations per arm). The pairs are within this session-block design;
they are not eight independently reloaded pairs. Each workload/session also
has one excluded, recorded warmup. Model loading is reported separately from
complete-request wall time.

A screening candidate proceeds only with at least a 5% median useful signal
beyond its observed same-code range, no numerical/admission error and no
greater than 10% loss in another screening cell. This is an inexpensive
selection rule, not the final acceptance criterion. Confirmation requires at
least 10% lower complete-request latency or higher useful decode throughput,
exceeding observed same-code variation, at least seven/eight paired wins, a
confirming workload and no greater than 5% median regression in another
declared confirmation cell. Prefill improvement is recorded separately and is
not presented as a general decode improvement. These are engineering hurdles,
not confidence intervals or model-quality certification.

The off/on configuration comparison is within the same checkpoint and worker.
Later independent runs against old historical numbers cannot establish an
effect. Preserve every setup failure, numerical mismatch, ineligible shape,
telemetry gap and measured regression. Publish counts, hashes, aggregate
errors/timings and memory observations only; no prompt/output/token vectors,
logits or probabilities are persisted.

## Execution record

Acquisition is in progress; full-model inference has not run. The release
worker and anonymous-pipe readout example build with locked dependencies on
Rust 1.98.1. Ten selected CPU/Metal router tests pass under
`nightly-2026-10-01`; thirty-one model-free measurement, decision-gate and
artifact-verifier checks pass. These
validate instrumentation and measurement controls, not checkpoint correctness
or speed. Prior source assessment and model-free records remain unchanged.

The frozen public-source corpus is a byte copy of
`crates/stark-engine/src/engine/hybrid/glm.rs`, 139,830 bytes, SHA-256
`4681f70624b8521889a6881a40be0e55705fb87807e53e380e2b5fac739fcd27`.
Timed inputs use exact raw-token prefixes of that corpus; this measures the
native generation path rather than chat-template or tool-use quality.

| Prepared executable | SHA-256 |
| --- | --- |
| `stark-generate` | `27e59b941503cf5720935fad989da163043b433b658f27ec64a8f99f9f5b482c` |
| `hybrid_prefill_readouts` | `d5cbfc428300e990c68f31021c0faf9c95464930265a482fc860bdb9b3c04506` |

Both release executables include the same disabled-by-default aggregate route
markers. The generator build is `cargo +1.98.1 build --release --locked -p
stark-generate`; the diagnostic example is built separately with `-p
stark-engine --features metal --example hybrid_prefill_readouts`. Neither
build enables the scorer's `optimized` feature. The private build receipt is
`runtime/glm53-llama-v1-tests/build-receipt.json`.

The [serialized campaign](../../scripts/run_glm_llama_v1_campaign.py) runs the
streaming verifier, diagnostics, eligible screens and qualifying confirmation
requests. It records functional success separately from performance acceptance;
an unsuccessful candidate does not discard independent candidates. Private
phase receipts and all measured requests live under
`runtime/glm53-llama-v1-tests/campaign/`; the aggregate controller record is
`campaign.json`. The acquisition dependency checks that the prepared model has
been atomically published and that sealed code, binary and corpus hashes still
match before launching the campaign. It does not wait for quiet host activity.
Any failed attempted phase remains visible in a `completed_with_failures`
campaign outcome; a noise-only comparison is a completed functional test with
no confirmed performance benefit.
