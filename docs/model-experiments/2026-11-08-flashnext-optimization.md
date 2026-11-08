# Flash-Next and dense Metal4 optimization attempt ledger

Registered: 2026-11-08. Executed: 2026-11-07–08. This retrospective registers
the complete Isoquant/Flash-Next optimization conversation, through PR 98's
CI repair and merge. The [full report](../benchmarks/2026-11-08-flashnext-optimization-retrospective.md)
contains the numerical results, methods, source transitions and remaining work.
The original dated reports and receipts remain canonical and unchanged.

`FN-LILY`, `FN-OFFICIAL` and `D27-COARSE3` refer to the exact revisions and
execution policies in the report's [artifact table](../benchmarks/2026-11-08-flashnext-optimization-retrospective.md#artifact-and-measurement-boundaries).
They are separate artifacts. The implementation column describes merged source
at main `7dc51834`; it does not verify installation in an app build.

## Model attempts

For readability, report labels `FN-01` through `FN-15` map in order to the full
registered IDs `FN-20261108-001` through `FN-20261108-015` below.
`D27-01` maps to `D27-20261108-001`. Registration date does not replace the
original execution date or source binding.

| Attempt ID | Artifact / change | Implementation decision | Correctness and performance outcome | Evidence |
| --- | --- | --- | --- | --- |
| FN-20261108-001 | FN-LILY: screen one/two/three MTP drafts | Harness retained; recommendation unchanged | Repeatability passed. Pilot timing unqualified: 84/144 arms failed screens. Three drafts lost on short input; no adaptive policy qualified. | [Pilot](../benchmarks/2026-11-07-flashnext-optimization-pilot.md) |
| FN-20261108-002 | FN-LILY: concurrent encoding cutoff 16 to 64 | Candidate excluded; cutoff 16 retained | Matching hashes, including held-out corpus/seed. No demonstrated complete-request benefit; timings unqualified. | [Encoder comparison](../benchmarks/2026-11-07-flashnext-optimization-pilot.md#encoding-policy-comparison) |
| FN-20261108-003 | FN-LILY: separate GPU stage profile | Diagnostic retained | MoE 35.1% of span; gate/up plus shared down 16.8%. Target-selection evidence, no speedup claim. | [Profile](../benchmarks/2026-11-07-flashnext-optimization-pilot.md#separate-stage-diagnostic) |
| FN-20261108-004 | FN-LILY: fixed Q4 gate/up and fixed Q8 shared down | Withdrawn; patch archived | Bounded numerical/prefix checks passed. Small component gain; first whole-model ABBA had one failed screen and no accepted gain. | [Projection study](../benchmarks/2026-11-07-flashnext-moe-projections.md) |
| FN-20261108-005 | FN-LILY: fixed gate/up, ordinary shared down queued first | Withdrawn; patch archived | Bounded controls passed. About 2.2% median paired combined-component gain; no reliable model benefit. | [Components](../benchmarks/2026-11-07-flashnext-moe-projections.md#component-measurements) |
| FN-20261108-006 | FN-LILY: repeated-route fixture and independent scoring replay | Experimental controls archived | Repaired permutation-only fixture; 135,680 BF16 values exact. Scoring digests matched for 993,280 F32 values. Existing draft-head tolerance unchanged. | [Numerical controls](../benchmarks/2026-11-07-flashnext-moe-projections.md#numerical-controls) |
| FN-20261108-007 | FN-LILY: final candidate observed whole-model ABBA | Candidate subsequently withdrawn | Outputs matched. Candidate/baseline 81.23/85.50 tok/s with two drafts; 75.63/80.33 off. Four of 32 arms failed screens; no quiet admission. | [Observed comparison](../benchmarks/2026-11-07-flashnext-moe-projections.md#completed-observed-whole-model-run) |
| FN-20261108-008 | FN-LILY: pause versus consecutive-request cadence | Diagnostic harness retained | Cadence dependence reproduced within unchanged workers; final candidate 95.61 after pause to 89.45 at fourth request. Not a candidate speedup. | [Cadence](../benchmarks/2026-11-07-flashnext-moe-projections.md#cadence-and-slowdown-diagnosis) |
| FN-20261108-009 | FN-LILY: display sleep / lock-screen hypothesis | No production policy change | Display state unconfirmed; all eight control arms ineligible with competing worker/paging. Cause unresolved. | [Display control](../benchmarks/2026-11-07-flashnext-moe-projections.md#cadence-and-slowdown-diagnosis) |
| FN-20261108-010 | FN-LILY: GPU-span/clock diagnosis, including gate-only variant | Opt-in aggregate counters retained; candidate withdrawn | Baseline 95.46 to 82.20 tok/s; about 98% of added verification elapsed inside GPU spans. Exact limiter unknown; 128 diagnostics, 125 screens passed, no quiet admission. | [Sustained diagnosis](../benchmarks/2026-11-07-flashnext-sustained-diagnosis.md) |
| FN-20261108-011 | FN-LILY: one-model, three-route crossover | Private prototype archived; original projection dispatch restored | Small MTP-on changes reverse sign between groups; no reliable gain or confirmed 5% code regression. | [Crossover](../benchmarks/2026-11-07-flashnext-sustained-diagnosis.md#one-model-projection-crossover) |
| FN-20261108-012 | FN-LILY: empty-queue signals during fixed idle | Private prototype archived | Mean first sample 1.0035 to 0.3035 s; 16/16 screens pass. Prototype alone was not production qualification. | [Idle control](../benchmarks/2026-11-07-flashnext-sustained-diagnosis.md#idle-queueresidency-control) |
| FN-20261108-013 | FN-LILY: owner-thread generation queue maintenance | Merged, `753884b8`; integration `640a6a7b` | About 0.78 s avoided after idle; 16/16 matched screens, prefix/EOF controls pass. Post-pause decode unchanged; sustained slowdown open. | [Production fix](../benchmarks/2026-11-07-flashnext-idle-wake-fix.md) |
| FN-20261108-014 | FN-OFFICIAL: generation queue maintenance | Merged, `4fe11fad` | First sample 1.12–1.13 to 0.31–0.32 s; all eight first-response screens pass. Whole campaign retains warm/control failures. | [Cross-backend study](../benchmarks/2026-11-08-cross-backend-idle-wake.md) |
| FN-20261108-015 | FN-LILY: scoring/digest queue maintenance | Merged, `4fe11fad` | Fresh scoring 1.0810 to 0.3028 s; one primary candidate lacks host coverage. Digest/cache controls pass. Does not qualify official-pack scoring. | [Scoring evidence](../benchmarks/2026-11-08-cross-backend-idle-wake.md#results) |
| D27-20261108-001 | D27-COARSE3: native text Metal4 shared-event maintenance | Shared hook merged, `4fe11fad` | Separate 512/64 repeat: first sample 0.71–0.75 to about 0.62 s, 16/16 screens. Prefix/EOF pass. Original 512/256 retains failed screens. | [Dense repeat](../benchmarks/2026-11-08-cross-backend-idle-wake.md#results) |

## Shared tooling and correctness repairs

Report labels `TOOL-01` through `TOOL-08` map in order to full IDs
`TOOL-20261108-001` through `TOOL-20261108-008`.

| Attempt ID | Attempt / defect | Disposition and limits | Evidence |
| --- | --- | --- | --- |
| TOOL-20261108-001 | Quiet-admission / observed modes and GPU coordination | Retained. Request/session/observed protocols stay separate; all arms retained. Coordinate benchmark chats and include teardown in the lease. | [Tooling review](../benchmarks/2026-11-08-flashnext-optimization-retrospective.md#tooling-failures-repairs-and-evidence-controls) |
| TOOL-20261108-002 | Generic failures, lost partial receipts, nested MTP counters, short-call coverage | Retained sanitized stage checkpoints, route parsing and missing-sample classification. Cache-hit controls carry no timing claim. | [Preliminary attempts](../benchmarks/2026-11-08-cross-backend-idle-wake.md#state-lifecycle-and-source-controls) |
| TOOL-20261108-003 | Patch display format, route fixture gap, missing early telemetry source snapshot | Applyable patches and expanded controls archived; original receipts/limitations preserved. | [Patch archive](../archive/experiments/2026-11-07-flashnext-moe/README.md), [pilot](../benchmarks/2026-11-07-flashnext-optimization-pilot.md) |
| TOOL-20261108-004 | Prefix expiry bypass when input arrived during maintenance | Fixed before final Lily ABBA; preliminary sessions retained separately. Deadline includes maintenance. | [Lily controls](../benchmarks/2026-11-07-flashnext-idle-wake-fix.md#state-lifecycle-and-source-controls) |
| TOOL-20261108-005 | Dense `long3` config rejected by strict admission | Invalid fixture selection; used admitted `coarse3` with unchanged weights and separately sealed metadata. Loader not relaxed. | [Setup record](../benchmarks/2026-11-08-cross-backend-idle-wake.md#state-lifecycle-and-source-controls) |
| TOOL-20261108-006 | Exiting worker group `EPERM` masked startup failure | Cleanup repaired, real subprocess regression added; merged `4fe11fad`. Live-group cleanup errors not silently ignored. | [Setup record](../benchmarks/2026-11-08-cross-backend-idle-wake.md#state-lifecycle-and-source-controls) |
| TOOL-20261108-007 | Main merge conflicted with hooks and changed corpus | Preserved images, froze original corpus bytes, recorded separate integration controls at `640a6a7b`. | [Integration](../archive/runs/2026-11-07-flashnext-pr98-integration.md) |
| TOOL-20261108-008 | Linux invalid-PID telemetry result differed from macOS | Fixed validation order and cross-platform tests; merged `dfb2bacf`. All final hosted checks passed. | [CI repair/review](../archive/runs/2026-11-08-pr98-ci-recheck.md) |

These 24 entries include diagnostics and tooling controls; they do not represent
24 independent model-speed candidates. Related IDs may cover the same source
change from different evidence scopes. No measurement or success count is
obtained by summing them.

## Unresolved and unattempted work

| Item | Status / next evidence needed |
| --- | --- |
| Sustained Lily slowdown | Unresolved. Need a discriminating limiter signal/control. The wake fix addresses a different endpoint. |
| Short fresh-scoring coverage | Primary coverage incomplete. New tooling/run may capture short calls; historical missing samples stay unqualified. |
| Long idle / battery costs | Short AC measurements only; longer resource/energy qualification outstanding. |
| Official Flash-Next scoring | Not benchmarked here; scoring entry uses Lily Q4 XL. |
| Other native text Metal4 artifacts | Shared hook implemented, but only dense 27B `coarse3` received model-bound tests in this follow-up. |
| GLM, EXL3, Flint and vision | Discussed/audited only in this campaign; no model-bound wake benefit established. |
| Dense online draft controller | Opening proposal only here; earlier dense ledger records an unimplemented controller. |

## Earlier histories

These records keep their own IDs, dates, models and qualification boundaries.
They are not new measurements from this conversation.

| Scope | Existing record |
| --- | --- |
| Flash-Next F32/Lily/official paths through 2026-10-31 | [142-attempt optimization history](../benchmarks/QWEN38_OPTIMIZATION_HISTORY.md) |
| Dense Qwen3.8-27B packing and MTP | [Approach ledger](../benchmarks/2026-11-06-qwen38-27b-approaches.md), [packed follow-up](../benchmarks/2026-11-06-qwen38-27b-packed-followup.md) |
| GLM-5.3-Flash experiments | [Benchmark index](../benchmarks/README.md), [batched-matrix research](../archive/runs/2026-11-03-glm-batched-matrix-research.md) |
| OLMo 3 / Gemma 4 Q4 | [Source fusion study](../benchmarks/OLMO3_GEMMA4_SOURCE_FUSIONS_2026_11_02.md) |

For future attempts, preserve the distinction between implementation, bounded
correctness, quality, lifecycle and timing qualification. Record failed,
withdrawn, blocked and inconclusive work as well as retained changes, and link
each new entry to immutable source/artifact evidence and its next decision.
