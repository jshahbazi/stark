# Inkling 2 Mini optimization wind-down, 2026-11-23

Optimization of Inkling 2 Mini stopped on the afternoon of 2026-11-23 by owner decision. This record holds the last attempts of the decode and prefill workstreams, which were stopped or reverted and are not measured, the incidents of the last hours, and the state left for anyone who resumes. Sources: the coordinator's account of 2026-11-23 and the workstreams' reports as relayed by the coordinator; statements marked "by the workstream's report" are an agent's account that the coordinator did not verify independently. Host as in the other 2026-11-23 records (128 GiB M5 Max, macOS 26.7.2, 94 W adapter, shared lock). Nothing here is released or installed. The comparison with MLX-Serve and every stopped item's estimate are on the [differences page](inkling-mlx-serve-differences.md).

## Decision

After engine-matrix run 5 ([INKENGINE-05](../benchmarks/2026-11-22-inkling-mini-engine-matrix.md#20261123-inkengine-05--fifth-run-the-committed-tree-through-the-standard-tool)) put the default 8-bit artifact within 2 percent of MLX-Serve's 8-bit pack on prefill and the short standard rows, ahead at 8K and 16K and behind on novel prose (0.91), the owner judged the standing good enough and said the novel-prose gap does not matter: "I don't care too much about prose. It seems fine." Optimization stopped. A three-repeat standard benchmark on the committed build (the `4314ac51` worker) goes ahead to confirm the standing; its result is not in this record yet.

## 20261123-INKROUND-13 · Planner unit tests repaired

| Field | Value |
| --- | --- |
| Attempt ID / date | `20261123-INKROUND-13` / 2026-11-23 |
| Kind | Test infrastructure |
| Result | Passed: 3 of 3 planner unit tests (`high_acceptance_holds_full_depth_despite_noisy_costs`, `high_acceptance_prefers_full_depth`, `low_acceptance_prefers_short_rounds`) with `nightly-2026-10-01` and `--features cpu,metal,inkling-slice` |
| Disposition | Attribute-only fix, committed as `b2a5a88f`; on the final build the coordinator ran 3 of 3 planner tests and 6 of 6 schedule-validation tests, all passing |

**Cause.** The planner tests had never compiled. The engine's test framework requires `#[stark_test]`; the plain `#[test]` attributes in `depth_planner.rs` (3 tests) and `schedule_validation.rs` (6 tests) broke the whole library test build from the planner commit onward. That is also why the commit-check run of `4314ac51` reported the planner test step as failing to build ([INKROUND-12](2026-11-23-inkling-mini-planner-tuning.md#20261123-inkround-12--hold-full-depth-while-acceptance-is-high)). After the attribute change the six `schedule_validation.rs` tests pass as well (6 of 6 on the final build).

## 20261123-INKROUND-14 · Two single-row and verifier launch merges, reverted unmeasured

| Field | Value |
| --- | --- |
| Attempt ID / date | `20261123-INKROUND-14` / 2026-11-23 |
| Kind | Kernel (launch removal) |
| Result | Not measured: the first paired run hung; both merges reverted when optimization stopped |
| Disposition | Reverted; not in the tree |

**Merges tried** (by the workstream's report):

- **A, router join into route selection:** a joined route-weights kernel reading both partial halves of the router projection, on single-row and verifier routes; expected about 48 fewer launches per step and per round.
- **B, NVFP4 up product also writing the down product's inputs:** single row only; expected 44 fewer launches per step.

**What happened.** The first paired run with both merges on hung: the engine process sat idle holding the shared GPU lock from about 15:23 to 15:36 until it was killed. The expected launch savings are estimates; no timing exists for either merge.

## Known issue: `--paired-control` hangs on the current tree

The hang is not caused by the merges. With both merges off, `--paired-control` still hung on the known-good `short_attention` control and was killed by a 150-second alarm, a wrapper in the decode workstream's private batch script added after the 15:23 hang (the harness itself has no deadline); the same harness finished in about 4 seconds earlier in the day on the older tree. Suspects, by the workstream's report: first the logit-agreement block added to `measure_paired_control` on 2026-11-23 (a step, a full-depth overlapped round, then a reset), then the prefill changes of `4314ac51`. That harness code is committed, so `--paired-control` is broken on the current tree until it is bisected. It is a diagnostic: on the same commit the server-level scoreboard, the worker smoke and the worker tests complete normally. That does not prove the worker path free of whatever causes the hang; it has not been bisected.

**If resumed, the workstream's order:** fix or bisect the harness first (drop the logit-agreement block and compare worker output hashes instead); then merge B alone; then merge A; then fold the slot-3 controller norm pair into `ink_verify_attention_mix` on later visits (about 24 fewer launches per step and per round, estimate).

## 20261122-INKFASTPRE-08 · Prefill items reverted unmeasured

| Field | Value |
| --- | --- |
| Attempt ID / date | `20261122-INKFASTPRE-08` (prefill series) / 2026-11-23 |
| Kind | Kernel (prefill passes and diagnostics) |
| Result | Not measured; all reverted when optimization stopped |
| Disposition | Reverted; not in the tree |

| Item | State at the stop |
| --- | --- |
| Remove the expert-input pass for wide prefill: direct gate/up and down kernels in `grouped_nax.metal` that gather token rows through the route order and read the sorted gate/up output in place (expected hash-identical) | A candidate binary compiled; no replay or timing ran |
| Fuse the MLP convolution with the residual add (expected hash-identical) | Built; never measured |
| Plain-store diagnostic for the gathered kernel's fused epilogue | Built; never run |

The gathered NVFP4 kernel's 8-bit paired timing stays at three complete pairs ([INKFASTPRE-07](2026-11-23-inkling-mini-gathered-nvfp4.md)).

## Incidents

| Incident | Effect | Lesson |
| --- | --- | --- |
| The hung paired run held the shared GPU lock for about 13 minutes (15:23 to 15:36) | Other workstreams' jobs waited; the process was killed | A diagnostic that can wait on the GPU needs its own deadline; the harness has none, and only the workstream's private batch wrapper added a 150-second alarm after this hang |
| Teardown of the killed 23 GB engine process overlapped the start of an image timing job | Worker decode read 14 to 17 tok/s for three requests, then recovered; that timing pair was discarded | Wait for a killed model process to exit and release its memory before timing |
| At about 15:55:40 the shared lock directory disappeared while a timing job held it (taken at 15:55:26); a cargo test and a build then ran alongside that job | The timing set was discarded. Cause established: while winding down at about 15:55:30 the decode workstream killed its batch with `pkill`, which also terminated its lock-script wrappers; a wrapper that had already released the lock still had its exit trap armed, and the trap removed the lock directory without checking the owner, deleting the lock a timing job had taken at 15:55:26. Fixed in the coordinator's private lock script (not in the repository): a wrapper releases only while the owner file still names its own process | A lock wrapper must release only the lock it owns, and a timed job should check the lock owner at its end as well as at its start and be discarded if the owner changed |

## Not done when the work stopped

All candidates, with their estimates, are listed on the [differences page](inkling-mlx-serve-differences.md#5-stopped-and-still-open). The coordinator's open gates are tracked there and in [INKAFFINE8-04](2026-11-23-inkling-mini-affine8-default.md).

## Later the same day

| Incident | Effect | Lesson |
| --- | --- | --- |
| The Mac rebooted at about 16:13 during the owner's pause; the session scratch directory under `/private/tmp` came back empty | Helper scripts, the shared lock, build caches and the gated binaries of the `4314ac51` build were lost. A standard run started at 15:58 had already been stopped for the pause (`runtime/benchmark/inkling-2-mini/20261123-155828-inkling-2-mini`, two of twelve arm sessions, unusable). Tooling was recreated under `runtime/session-tools`, build caches under `runtime/build-cache` and binaries under `runtime/final-build` (all ignored) | Keep long-session tooling, build caches and gated binaries under `runtime/`, not in the scratch directory |
| Another session's cargo jobs were running at 18:52 and had finished by 19:10, before the standard run started | None on the result | Check for other sessions' builds before a timed run |

The coordinator then rebuilt and gated the final build and ran the three-repeat standard benchmark: [20261123-INKENGINE-06](../benchmarks/2026-11-23-inkling-mini-standard-benchmark.md).
