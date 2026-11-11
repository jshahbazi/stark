# Beginner setup choice audit — 2026-11-11

## Scope and source

Reviewed the complete guided setup flow after **Help with code → More model
capacity** selected an older dense Qwen model while a supported Qwen3.8 Flash
Next artifact was already registered locally. Implementation started from
`e091cb3b` on `codex/beginner-installed-models`.

Implementation fingerprint: `90acb0a5159c552275666b4c40efc55cd5236c7c197959d530272e0e3229389d`.
This is SHA-256 of sorted-key JSON mapping repository-relative filenames to
their file SHA-256 values: all `apps/menu-bar/Sources/Beginner*.swift`,
`StarkApp.swift`, and converter modules `desktop.py`, `guided_models.py`,
`guided_installed.py`, `guided_dense_installed.py`, `guided_discovery.py`,
`model_catalog.py`. The dated report is excluded from its own fingerprint.

The audit covers every visible task and model priority, every retry intent,
installed-model reuse, downloads, model switches, navigation, cancellation and
the OpenWork connection gates. It is a source/state-machine audit with
model-free regressions and a metadata-only local-library replay. It does not
establish answer quality, speed or completion of every real OpenWork workflow.

## Findings addressed

| Finding | Corrected behavior |
| --- | --- |
| Installed hybrid models never entered the dense-only catalogue comparison | Supported local dense artifacts and the exact pinned Lily q4-xl Flash Next artifact enter independently of online discovery |
| A source coding tag overrode explicit lighter/capacity selections | Those selections rank their objective before coding-tag tie-breaks, within the hardware/headroom tier |
| Quantized checkpoints lost capacity credit because Hub totals count packed scalars | Supported complete dense configuration geometry supplies logical counts; unknown layouts remain unknown |
| Several recognized coding tag aliases were discarded during catalogue collection | Collection and recommendation normalization recognize the same aliases |
| One metadata failure discarded an entire family of successful candidates | Guided discovery retains successes, marks the result incomplete and retries failures instead of caching them |
| Retry intent silently overrode a later visible priority | Retry updates the visible priority and is consumed once |
| Changing task/priority could expose the previous recommendation through navigation | Changed selections invalidate those offers until refreshed |
| Keeping the current model retained the temporary retry priority | The previous task and priority are restored |
| Rejected install requests could leave setup waiting for an operation that never started | The session enters pending installation only after controller acceptance |
| Invalid startup resource reports left a pending beginner start | The controller reports the error and clears the pending start |
| Identical imports could reappear as a different-model recommendation | Equivalent local artifacts share one offer and exclusion applies to every registered copy |
| Removed installed-only copies could expose a download button | The guide explains the inventory change and offers a refresh |
| Local aliases outside the report's display bounds could invalidate the report | The source name supplies a bounded fallback; valid custom names are preserved |

## Choice coverage

| Choice or condition | Logic followed and checked |
| --- | --- |
| Everyday, text/files, code, unsure | All four visible tasks reach recommendations with the selected priority; code uses source tags, other tasks share resource ranking |
| Balanced, lighter, more capacity | All 12 visible combinations; backend regression also exercises its additional `files` task value |
| Four retry intents | All task × priority × retry combinations pass through the production view request handler |
| Current model ready, restart required, different model | Reuse, explicit restart/switch, failed readiness, and keeping the current model |
| Installed artifact, online candidate, missing local copy | No-download reuse, download/preparation, explicit start, and inventory-change refresh |
| Changing task/priority, Back, sidebar navigation | Old offers cannot be used for a changed selection |
| Install accepted/rejected, cancel/retry, startup failure | Pending states correspond to accepted operations; failures allow recovery |
| API unavailable, missing tool support, unknown context | Connection/start prerequisites remain explicit |
| OpenWork absent/installed/running, approval, client activity | Install/open/connect/approval gates, workspace and model-selection confirmation |
| Discovery partial/offline, invalid/unsupported installed model | Useful candidates survive; limited coverage is disclosed with Advanced available |
| Memory/disk constraints and changed artifact identity | Static physical/GPU fit, installed disk exemption, start-time memory check, and source/config revalidation |

## Local library replay

On this Mac: Apple M5 Max, 18 CPU cores, 128 GiB physical memory, Metal
recommended working set **115,448,725,504 bytes**. The replay copied the four
registered metadata entries into an isolated controller, disabled catalogue
discovery, and ran all 12 visible combinations. It did not start or stop the
installed app or a model worker.

| Task | Balanced first choice | Lighter first choice | More capacity first choice |
| --- | --- | --- | --- |
| Everyday | Installed Qwen3 4B | Installed Qwen3 0.6B | Installed Qwen3.8 Flash Next Lily q4-xl |
| Text/files | Installed Qwen3 4B | Installed Qwen3 0.6B | Installed Qwen3.8 Flash Next Lily q4-xl |
| Code | Installed Qwen3 4B | Installed Qwen3 0.6B | Installed Qwen3.8 Flash Next Lily q4-xl |
| Unsure | Installed Qwen3 4B | Installed Qwen3 0.6B | Installed Qwen3.8 Flash Next Lily q4-xl |

Each combination offered three choices. The coding/capacity result explicitly
had installed IDs and zero download bytes. This replay verifies local discovery
and ranking with offline fallback, not the contents of a changing online scan.

### Flash Next artifact and memory policy

- Source: `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl`.
- Revision: `0f94842e3682b49acace4daeb4cf2e65a6f94f83`.
- Configuration SHA-256: `2a91155be52ebbb0f6dff20e61dcd8ae7484154a56b4afe067246c59da46f899`.
- Prepared safetensors: **106,905,155,192 bytes**.
- Validated paged table tensor planes: **32,000,153,600 bytes**.
- Physical planning estimate at 32,768 tokens: **112 GiB**; separate GPU estimate: **83 GiB**.
- Admission retains 2 GiB physical reserve and the existing 85% GPU-budget cap.

The planner reads bounded metadata/headers, validates the known paged-table
shapes and dtypes, and charges that table to physical memory rather than
resident GPU allocations. It includes conservative cache growth, recurrent
state, two prefix snapshots, optional MTP draft state and an 8 GiB workspace
allowance. Estimates are planning allowances, not measured allocation peaks.
Capacity is a declared lower bound from routed expert matrices, not a guessed
model total or a quality score.

The allocation references are the existing Lily wrapper and the vendored core's
`session_bytes`, `new_state`, `new_scratch_with_capacity` and paged table loader.
This change adds no model/backend, precision policy or generation-state path.
Existing generation prefix reuse remains in place; its snapshots are included
in planning. Other hybrid artifacts still need their own planner and are
disclosed as unassessed instead of inheriting this artifact's plan.

## Validation

- Focused converter recommendation, discovery, local-planning and catalogue suites: **286 passed**.
- Broad converter suite: **6,565 passed, 90 skipped**, with the required template helper built. The final bounded local-name fallback and invalid-inventory disclosure were checked in the focused suite after that broad run started.
- Native beginner session/view tests: **13 passed**.
- Native controller tests: **137 passed**.
- Native window/Host tests: **7 passed**.
- Repository policy check, converter Ruff and changed-file whitespace checks passed.
- Built the pinned Rust 1.98.1 template helper required by the converter suite.

An initial broad run found an unrelated LocalAPI test synchronization race:
the HTTP response can finish before server cleanup releases the active client.
It passed six isolated reruns; delaying cleanup reproduced its assertion
failure without production changes. No LocalAPI code or tests were changed.

No app was installed or relaunched during this audit. Model loading, measured
memory peaks and a fresh interactive OpenWork session remain separate runtime
validation.

## Follow-up: persistent backend pipe stall

After installing the app, the guide stayed on **Checking models** for several
minutes. The earlier controller tests injected decoded events and the packaged
replay exercised Python directly; neither checked delivery through the live
Swift pipe reader.

The production readability handler used
`FileHandle.readData(ofLength: 65_536)`. On this platform that call waits for
the requested byte count or EOF. A recommendation response smaller than 64 KiB
could remain unread while the persistent backend stayed alive. A live packaged
backend check with the registered library delivered its 4,563-byte report in
5.36 seconds, selecting the installed Flash Next, while its stdin/stdout stayed
open. The native reader prevented that completed result from reaching the view.

The handler now uses one bounded `Darwin.read` call, retrying interrupted reads,
and feeds the existing framed-event receiver immediately. Tests exercise the
production handler through real persistent pipes: a small recommendation clears
the spinner before writer closure, fragmented/coalesced events retain correct
framing, and EOF removes the handler.

Follow-up source begins at `9c9388a8` with these changed-file identities:

| File | SHA-256 |
| --- | --- |
| `apps/menu-bar/Sources/StarkApp.swift` | `1dadd0db9e5065cd7a124f7f6ff2b15ba67b0826f1b7e249bbd27335ecb551f3` |
| `scripts/test_menubar_controller.py` | `94b5949184bcfd8ebc551b0829b767c646e9f32991423f63a44265fd935295bc` |

Follow-up checks: **140 controller tests**, **13 beginner tests** and **7 window
tests** passed. Recommendation/backend code and inference state paths did not
change in this transport fix.
