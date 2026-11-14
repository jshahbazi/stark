# Guided recommendation corner-case review

Date: 2026-11-13. Reviewed branch: `codex/competitive-guided-alternatives`,
including the installed-RAM policy at `0a0c090b` and the subsequent main merge
at `9fc1b393`. This record supplements the [capacity-policy checks](2026-11-13-guided-capacity-recommendations.md).

## Reproduced issues and fixes

1. **Paged startup reminder used total memory.** An installed paged recipe can
   have a 112 GiB total plan and an 83 GiB startup working set. With 98 GiB
   available, startup permits the model but Swift's reminder previously
   compared against 112 GiB. The report now parses the effective startup
   requirement, validates it and uses it for the reminder. Older reports that
   omit the field retain the total-memory fallback. Tests cover the exact
   startup boundary, one byte below it and malformed values.
2. **Speed alternatives could repeat one parent or retain distant scores.**
   On a sufficiently large synthetic Mac, three Flash Next exports could fill
   the three offer slots. Speed could also retain the screenshot's 73.7 vs
   51.0/48.4 gap. Both guide modes now apply the same 10-point distance and
   distinct-parent shortlist after their own eligibility and sort policies.
   Speed's payload budget, score floor and primary ordering remain unchanged.
   Tests include separately reviewed export repositories sharing one parent.
3. **Registered seeds bypassed task priority.** Registered small models were
   yielded before the task-ranked window, spending candidate slots first.
   Guided planning now sorts those seeds with discovered candidates. Tests
   put 35 registered unscored sources ahead of a scored parent and verify
   independent General/Coding seed order.
4. **Overlapping queries reduced the priority window.** Two queries returning
   the same 160 popular sources could spend 300 raw positions before reaching
   a stronger source. Repositories now contribute one position across queries.
   Pinned/native seeds preserve their exact identity when a head listing
   describes another revision or format; a separate regression checks this.
5. **Producer and consumer result bounds disagreed.** Nineteen families could
   produce 570 downloadable rows, or 5,700 accumulated unavailable rows, while
   the desktop rejected more than 450 rows. Discovery now returns only
   downloadable rows, preserving incomplete/failure signals. A shared bound
   derives the maximum from family count and page size: currently 570 rows.
6. **Subprocess deadline was shorter than the scan envelope.** Nineteen
   ten-second families with three workers permit seven waves, while the
   desktop timeout was 65 seconds. The shared bound now allows the wave
   budget plus ten seconds for startup/serialization: currently 80 seconds.
   The 2 MiB serialized-response limit remains. A pipe-backed test accepts the
   full row count; its writer runs concurrently to avoid assumptions about
   platform pipe capacity.

## Validation

- The full model-free converter suite during the review passed:
  **7,758 passed, 95 skipped**. Following the fixes, the affected guided,
  recommendation, catalogue, cache and discovery suites were rerun:
  **520 passed**.
- Additional preparation tests distinguish installed-RAM recommendations from
  the 2 GiB preparation allowance and full worker-start requirement, including
  missing/bool input and the one-byte preparation boundary.
- Swift report tests cover effective startup memory, legacy report fallback,
  invalid startup metadata and continued low-memory offer visibility.
- 80 app/setup/build/window harness checks passed. The full menu-bar Swift
  source typecheck passed with no diagnostics. Ruff lint/format, repository
  policy and diff checks passed.

The review used synthetic metadata and isolated native UI harnesses. No model
weights were downloaded or model workers started; no installed app was replaced.
These checks establish the reviewed selection, discovery and startup-boundary
contracts. They do not qualify model answer quality or runtime performance.
