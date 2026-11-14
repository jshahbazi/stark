# Model download wake recovery and named failures

Date: 2026-11-14. Source base: `4bcaba76` on `origin/main`; the implementation
and this record are committed together on `codex/resume-model-downloads`.

## Behavior

- The menu bar controller forwards macOS sleep/wake notifications to its active
  backend, removing observers when the runtime terminates or the app shuts down.
- Sleep interrupts an active downloading child and retains its installation,
  exact checked command/revision, cache and owned native staging. Wake signals
  the waiting operation to retry immediately. Already-running conversion and
  inference are not selected for termination by this handler.
- Typed transport failures and HTTP 408, 429, 500, 502, 503 and 504 responses
  before conversion keep the installation active. Retry delays increase from
  5 seconds to a 60-second maximum. Cancel, Stop and quit interrupt the wait.
- Native preparation reports allowlisted failure categories over its private
  progress pipe, matching generic conversion. No exception text is sent to the UI.
- A buffered conversion-start frame after sleep selected a download does not
  prevent recovery. If the child published valid output just before termination,
  normal installation validation can admit that output without another child.
- Model-related failure text includes the requested model's readable short name.
  Request correlation and guided-run rollback retain the failed model's identity
  even when another model is loaded or selected afterward.

## Validation

- Repository policy check, Ruff on changed converter modules/tests, and
  `git diff --check`: passed.
- Menu bar controller harness: 154 checks passed, no failures or skips. Includes
  synthetic sleep/wake notification forwarding and observer removal, waiting
  installation ownership, stale failure fencing and model-name selection.
- Focused converter/native/backend validation: 113 tests passed. Includes real
  synthetic child-process interruption/retry, prompt wake recovery, cancellation
  during backoff, retry delay cap, late conversion progress, completed output
  reuse, and native partial staging reuse.
- Final standalone `test_desktop.py` run: 361 passed, 1 skipped, 1 failed because
  a synthetic worker was absent after setup. The failed test and a prior
  readiness failure both passed on isolated rerun (2 passed). An earlier broad
  run had five varying worker-readiness failures; the selected rerun covering
  those cases passed 34 tests. These runs do not establish a consistently clean
  broad backend suite. No fixture timeouts or unrelated inference code were
  changed to suppress these failures.
- Native template helper built with Rust 1.98.1 and locked dependencies before
  backend tests. No engine, worker inference or model precision paths changed.

## Limits

Sleep/wake was injected through notifications and backend commands; physical
lid-close recovery with a real model download is not qualified by these tests.
No large model was downloaded, and the installed production app was not changed.
Automatic recovery applies to an active installation in the current app session;
it does not schedule automatic installation after relaunch. Opaque transfer
failures without a sleep notification or typed transient cause remain terminal.
