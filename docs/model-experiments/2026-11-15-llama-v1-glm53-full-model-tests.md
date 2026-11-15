# Attempt ID: 20261115-LCPPGLM-03: native GLM-5.3 full-model tests

## Identity and state

| Field | Value |
| --- | --- |
| Attempt ID / registered date / executed date / owner | `20261115-LCPPGLM-03` / 2026-11-15 / not run / Codex |
| Kind | Correctness, kernel and performance |
| Attempt state | Proposed; pinned artifact acquisition is in progress; Rust 1.98.1 release build is ready |
| Result and acceptance criterion | Full-model checks not run; verify the four pinned GGUF shard hashes before inference, then use the numerical and performance gates in the [test plan](../benchmarks/2026-11-15-glm53-llama-full-model-tests.md) |
| Evidence scope | Planned full-model readout, route diagnostics and end-to-end timing; existing model-free controls are component evidence only |
| Promotion decision | No promotion; all execution options remain limited to their existing defaults or opt-in profiles |
| Disposition | Experiment only |
| Predecessor / related attempts / governing issue / PR | [20261115-LCPPGLM-01/02 assessment](../benchmarks/2026-11-15-llama-v1-glm53-assessment.md); [ordered test plan](../benchmarks/2026-11-15-glm53-llama-full-model-tests.md); no implementation PR |

## Hypothesis and change

- Test whether the existing complete GPU-bank profile, Q5/Q6 MMA prefill route, MLA prefill batching and register KDA prefill produce useful full-model outcomes on the exact native GLM artifact.
- Baseline and candidates use the same prepared artifact and source-bound release worker. Compare normal defaults with the complete profile, then isolate MMA-off within that profile, MLA batching alone and register KDA alone. The report defines any later confirmation combinations.
- No kernel, default, precision policy or backend change is part of this registration.
- Full-vocabulary readouts use the existing `1e-3 + 1e-3 * abs(reference)` gate, require finite values and five-of-five greedy agreement at each fixed prefix. The screening and confirmation thresholds are recorded in the linked test plan.
- Aggregate route markers are enabled only for readout diagnostics. The same source-bound Rust 1.98.1 worker build runs every arm; keep marker collection disabled for speed arms. Counters establish logical row invocation, not dispatch, duration or GPU utilization. The inactive marker branch is measurement-only and does not change default routing, precision or kernels.

## Artifact, source and environment

| Field | Baseline | Candidate |
| --- | --- | --- |
| Model family, exact checkpoint and upstream revision | `unsloth/GLM-5.3-Flash-GGUF@a38483c8cd5df544f53d70fb281afe97369d5ab6` | Same pinned artifact |
| Metadata source | `zai-org/GLM-5.3-Flash@04c4e9e95c5da8862dced7e5056455116f83a7e0` | Same pinned metadata |
| Format and quantization | Four `UD-Q2_K_XL` GGUF shards; expected aggregate 108,720,071,552 bytes (101.25 GiB); payload hashes not yet verified | Same |
| Backend and provenance | Stark native Rust/Metal `glm5_next`; preparation via the pinned `hybrid_prepare glm53-q2` source | Same backend; existing execution switches only |
| Stark source and build | Parent commit `d47eae038ba0700b060d02ec208ae05e8784bd5e` plus a nine-line measurement-only Rust patch; Rust 1.98.1 locked release build complete; exact binary hashes are in the test report | Same patched source and worker build across arms; route markers enabled only in readout diagnostics and disabled in speed arms |
| Weight, activation and state precision | Original GGUF quantized weights; F32 activation and recurrent-state policy | Same |
| Hardware and OS | Mac17,7; OS and available headroom to be recorded at execution | Same host and measured session |
| Memory evidence | Not measured | Not measured |

The new [streaming artifact verifier](../../scripts/verify_glm53_llama_weights.py)
checks the exact LFS SHA-256 values, shard sizes, source pins, config/tokenizer
hashes and before/after file stats. Its private success receipt is required
before model execution. The native preparation receipt alone records source
revisions and aggregate size; it does not carry the four payload hashes.

The private build receipt is
`runtime/glm53-llama-v1-tests/build-receipt.json`, SHA-256
`d0dbae9c921ab0ce0870b7cf977791da7f9739bfd4dd46623046b5c0cf084410`.
It binds the parent commit, exact Rust patch, dependency locks, compiler,
hybrid-source bundle and both release binaries. Ten selected CPU/Metal router
tests pass on the patched source under `nightly-2026-10-01`. These tests and
the sixteen measurement-runner checks and four synthetic verifier cases
establish tooling/component behavior; no checkpoint inference has run.

## Workload and controls

- Model capability: native GLM preparation is text-only. Image execution is N/A for this source; native MTP stays off.
- Run the default and candidate readouts at fixed prefix lengths 129 and 513, with five full-vocabulary readouts per prefix. Use anonymous pipes and retain aggregate errors and hashes only.
- Use the same frozen corpus bytes, greedy sampling, prefix/context/output budgets and F32 activation/state policy in every arm. Record the corpus identity and all actual load-time selectors in the private run receipt before execution.
- First screen the complete-profile comparison at 96 input/8 output and 512 input/16 output, with two balanced A/B pairs and one excluded warmup per workload and loaded session. Screen isolated prefill options at 512/16. Only survivors proceed to the eight-pair confirmation matrix in the linked test plan.
- Preserve concurrent host work. Do not wait for quiet, discard busy measured requests or retry unfavorable observations. Record telemetry failures as unavailable. Native memory, geometry and allocation guards remain enabled.
- Keep text MTP off, do not enable wide prefill, expert batching or unrelated selectors, and restart the worker for changed load-time options.
- Validate any state-affecting prefix-reuse profile against a matching fresh run before calling it usable.
- Use the same native release worker for baseline and speed candidates. Enable aggregate route markers only for readout diagnostics, then disable them for all speed arms. The diagnostic observation is not a throughput result and route counts do not establish GPU dispatch count or utilization.
- Preserve acquisition, build, load, parity, admission, telemetry and timing failures. Do not replace an input, weaken a threshold or disable a guard to obtain a passing arm.

## Attempt history and failures

| Trial / source | What was attempted | Observation | Validity and disposition |
| --- | --- | --- | --- |
| Registration, 2026-11-15 | Acquire the pinned native artifact through the existing preparation helper | Acquisition in progress; no full-model execution or timing has occurred | Pending. Continue only after all four pinned LFS hashes pass the verifier |
| Pre-run component controls | Seven synthetic GLM router, chunk-limit, MMA shape/runtime, register-KDA and prefix-state tests; 16 model-free readout-runner tests | All passed. The seven Rust tests ran on an input tree byte-equivalent to the study checkout before its nine-line measurement patch; the private receipt records the d292 test checkout and input-tree hashes | Component evidence only. No model loaded or full-model test run; see `runtime/glm-llama-components-20261115/receipt.json` for the seven Rust cases |
| Artifact verifier fixtures | Synthetic four-shard success, corrupted payload, wrong metadata pin and symlink cases | Four tests passed; success receipt is accepted by `benchmark_hybrid.weight_identity` | Verifier unit evidence only. No GLM weight files were opened or hashed |

## Correctness and lifecycle

- Full-model reference, numerical, prefix-reuse and lifecycle results: not run.
- The seven model-free Rust component tests and 16 model-free readout-runner tests are not promoted to checkpoint correctness. The Rust test executable was built from checkout `d29230d68f795b67c50884b905799551ed6ff2c6`; its private receipt binds engine, macro, lockfile, workspace and vendor input trees to the study checkout before the measurement-only patch. It does not claim identical build binaries or coverage of that patch.
- Four synthetic verifier tests confirm receipt compatibility and rejection of corruption, wrong source pins and symlinks. They exercise no model artifact.
- Numerical comparisons, route counters, cache restoration, cancellation, reset and fresh-versus-reused output equality remain pending the exact artifact and source-bound worker.
- The native loader and runtime resource checks remain enabled. No model data was present during the component tests.

## Performance and memory

| Metric and measurement boundary | Baseline | Candidate | Repeats/dispersion | Eligibility and conclusion |
| --- | --- | --- | --- | --- |
| Prefill / first token | Not measured | Not measured | Not run | No conclusion |
| Useful decode / complete request | Not measured | Not measured | Not run | No conclusion |
| Peak memory / retained payload | Not measured | Not measured | Not run | Resource behavior pending artifact load |
| Readout diagnostic route counts | Not measured | Not measured | Not run | The disabled-by-default markers report logical rows only and do not establish dispatch count or GPU utilization |

## Evidence, decision and limits

- Plan and ordered workload: [GLM full-model test plan](../benchmarks/2026-11-15-glm53-llama-full-model-tests.md).
- Source assessment and ranked ideas: [llama.cpp 1.0.0 GLM assessment](../benchmarks/2026-11-15-llama-v1-glm53-assessment.md).
- Private model-free receipt: `runtime/glm-llama-components-20261115/receipt.json`; no prompt, completion, token vector, logits or probability data is included.
- The historical default-off T8 MMA study and the negative all-selector GPU-stage observation remain bound to their original source/build/workload records. This pending attempt does not replace them or establish an effect for this checkout.
- No candidate/default decision is made. Full-model execution, quality, broad workload performance and installed-build status remain unqualified. Image execution is N/A for this text-only source.
- Next action: finish pinned artifact preparation; run the streaming verifier once; bind exact model, config, tokenizer, native receipt, worker and corpus identities; then follow the registered readout and measurement order.

## Follow-up history

Registration only. Append each completed, failed, invalid or resource-deferred run as a dated follow-up; do not rewrite the planned gates or historical evidence.
