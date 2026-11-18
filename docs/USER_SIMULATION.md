# Manual desktop user simulation

Invoke `$stark-user-test` in Codex to have it operate Stark, OpenWork, OpenCode
Desktop and Pi as a user building a small project. The repository skill lives
at [`.agents/skills/stark-user-test`](../.agents/skills/stark-user-test/SKILL.md).
An agent with computer-use access drives the interfaces. This workflow is
explicitly invoked: no scheduler, CI job, background service or live test starts
from installing or discovering the skill. The helper alone prepares fixtures
and verifies artifacts; it does not drive the desktop.

Each client builds its own standard-library Python expense ledger, extends it in
the same conversation, handles a failed command and bad inputs, then exercises
cancellation and connection/session recovery. Independent checks execute the
generated program against fresh varied inputs. A connection-only chat cannot
pass this workflow. The default run includes all three apps; an explicitly
requested subset is reported as that subset.

## Scope and prerequisites

- A Mac with Stark, official signed OpenWork and OpenCode Desktop, compatible Pi,
  and Codex computer-use access. Missing apps are blocked results. Run the helper
  with the repository's pinned uv-managed Python 3.13.13; it uses no extra packages.
- One running Stark model with tool support and adequate API context. OpenWork's
  normal initial context can exceed 16,384 tokens; follow the current
  [menu bar guide](../apps/menu-bar/README.md) and avoid reducing the client's
  context merely to get a pass. Use the current image/MTP settings and record them.
- A fresh disposable project and conversation per client. Never point a task at
  an existing project. Complete decisions and desktop interaction serially.
- Use managed Connections for OpenWork and OpenCode Desktop. A standalone
  OpenCode CLI with a copied key exercises a different connection. Pi uses its
  existing Advanced Connections setup and Terminal launch with Pi's Keychain
  credential command.
  See [local API connection details](LOCAL_API.md).

Observe initial connection and working-folder settings without exporting chat,
credentials or environment data. Restore any changed choices during cleanup.
Synthetic project files remain private under ignored `runtime/user-simulation/`.
The helper launches generated Python with a reduced environment, bounded output,
timeouts and process-group cleanup. This is not an OS sandbox. The task uses
only standard-library local tools and requires no network or package installation.

## Prepare and identify a run

From the checkout containing the skill:

```sh
uv run --no-project --python 3.13.13 python scripts/stark_user_test.py prepare
```

The command prints an absolute run directory. Its `openwork`, `opencode` and `pi`
subfolders contain only `expenses.json` and the initial authored `BRIEF.md`.
Use that absolute directory as `RUN` in the examples below. `CLIENT` is one of
`openwork`, `opencode` or `pi`. An explicit single-client request can use
`prepare --clients pi` (or a list). Do not reuse a completed run for a new build
or model. Preparation records harness/oracle hashes. The identity command, run
after opening the apps, binds the actual running bundle paths reported by macOS,
application executable hashes, and Stark's bundled generation/template workers,
bundled Python, Python backend and native libraries. It does not assume `/Applications`; source
installs under `~/Applications` are supported. These hashes do not establish the
installed engine's source revision.

For each client, inspect Stark's running model and the client's selected model.
Record the actual model identifier, context, client version and visible settings:

```sh
uv run --no-project --python 3.13.13 python scripts/stark_user_test.py identity --run "$RUN" --client "$CLIENT" --model owner/model --context 32768 --mtp off --images loaded --client-version 1.2.3
```

Replace the example values with observations. MTP accepts `on`, `off`,
`unavailable`, or `unknown`; images accepts `loaded`, `text-only`, or `unknown`.
Unknown is an evidence limit. Supply `--artifact-sha256` only when the exact
loaded artifact's SHA-256 is already established; otherwise it remains null.
A catalog name is not a weight digest. Model/context/setting identity changes
require a new run. Do not silently change the model or a supported default.
Identity capture requires Stark and the selected desktop client to be running;
missing or ambiguous application locations cannot satisfy build identity. Pi's
CLI path is resolved from the helper's PATH; confirm it is the same executable
selected by Stark's Connections setup. Verification rechecks running app paths
and fingerprints. Final reporting rehashes the bound paths so cleanup may close
the apps without invalidating otherwise unchanged build evidence.

## Drive the real client

For every observed step, use:

```sh
uv run --no-project --python 3.13.13 python scripts/stark_user_test.py observe --run "$RUN" --client "$CLIENT" --step connection --outcome pass
```

Use `fail` for observed wrong behavior, `blocked` when a prerequisite prevents
the check. Never enter pass for an unobserved action. Entries retain aggregate
history and timestamps. This is an operator evidence record; the helper cannot
attest that a UI action happened. There is no arbitrary diagnostic-text field.

1. **Connect and select.** Open the client through its real user flow, choose its
   own run subfolder, start a fresh conversation and select Stark's exact running
   model. OpenWork and OpenCode use Stark's managed Connect/Allow controls; a new
   OpenCode engine may need Allow. Pi uses **Advanced → Connect to... → Apps → Pi**,
   **Change Working Folder…**, and **Open Pi**. The Open button is disabled
   until setup matches the running model. Do not display or script-read
   the master key. If the OS requires personal credential approval, let the user
   complete it. Record `connection` and `model` after checking both interfaces.
2. **Build.** Read the current authored brief through the helper (below), submit
   it in the actual client, and let that client write code, run tools and finish.
   Approve local project operations as needed. Observe progressive output and
   actual file/test tool rounds. Confirm Stark's request readout becomes active
   for this client and returns idle; record `stark_activity` and `build_tools`.
   Inspect the visible provider/model again after the task. A fallback provider,
   stalled final turn, or claimed test without execution fails the relevant step.
   Run independent `verify --stage build`.
3. **Revise in the same conversation.** Submit the `revise` brief. Observe file
   edits, test execution, completion and continued Stark activity. Record
   `revise_tools`, then independently verify `revise`. The new filter/CSV behavior
   must preserve earlier valid JSON behavior.
4. **Failed tool and recovery.** Submit the `recover` brief. Observe the deliberately
   missing-input command actually fail, then subsequent test execution and a
   successful valid-input run in the same conversation. Record `failed_tool` and
   `recover_tools`; independently verify `recover`. A blocked tool permission is
   not an executed failed-command recovery.
5. **Cancel and recover.** Ask the client to explain the ledger implementation at
   length without editing files. While generation is active in Stark, use the
   client's Stop/Escape control. Confirm output stops and Stark returns idle within
   two minutes; record `cancel`. Submit a short follow-up asking it to read
   `expenses.json`, run the ledger and report completion. Observe a successful
   tool round and Stark returning idle; record `cancel_recovery`. Verify `recover`
   again if project files changed. If the generation finished before cancellation,
   retry once with a longer explanation; an unobserved cancel remains blocked.
6. **Connection/session lifecycle.** For OpenWork/OpenCode, Disconnect only the
   target client in Stark. In a fresh task/session attempt a request to Stark;
   require a visible connection/authentication failure without another provider
   succeeding. Record `disconnect_denied`. Reconnect through Stark, handle any
   required app reload/engine approval, then record `reconnect`. Start a fresh
   task on the same test folder, select the same model, ask it to read and run
   the existing ledger, and observe successful tools plus Stark activity; record
   `reconnect_recovery`. Other clients' authorizations must remain intact.
   For Pi, exit only its test session, then reopen through the same documented
   Stark Pi flow and project; record `session_restart`. In the fresh session
   read and run the ledger, observe Stark activity and successful tools, then
   record `restart_recovery`. Pi has no managed Disconnect/revocation test.
7. **Cleanup.** If tool code changed the final artifact after its last verification,
   repeat `verify --stage recover` while the apps are still open. Stop only test
   activity, close test sessions, restore changed
   connection/working-folder choices and confirm Stark idle. Do not stop a shared
   OpenCode engine, delete unrelated sessions, rotate credentials or erase the
   run evidence. Record `cleanup`.

Get a brief and run independent verification with:

```sh
uv run --no-project --python 3.13.13 python scripts/stark_user_test.py brief --run "$RUN" --client "$CLIENT" --stage build
uv run --no-project --python 3.13.13 python scripts/stark_user_test.py verify --run "$RUN" --client "$CLIENT" --stage build
```

Replace `build` with `revise` or `recover` at the appropriate step. Each `verify`
requires recorded identity, connection, model, Stark activity and the current
tool step. Stages must pass in order. Retesting an earlier stage supersedes
later results. Verification checks single-file `ledger.py`, varied hidden data,
integer totals, empty input, exact filtering, escaped/sorted CSV and rejection
of malformed data without input mutation. Neither agent-written tests nor the
agent's final prose decide project acceptance. Public sample input must remain
unchanged. Raw generated stdout/stderr are discarded, including on failure.

If acceptance fails, supply only failed check IDs to the client, with at most
two repair turns per stage. The operator must not fix generated code or expose
the oracle/reference implementation. Allow at most ten minutes per task turn
and two minutes per cancellation/recovery check unless the user changes the
budget. On timeout, cancel, confirm idle and record the outcome; leave blocked
dependent stages incomplete and continue the next available client. Resume a
partially completed run only with unchanged identities, helpers and settings.

## Report and interpret

```sh
uv run --no-project --python 3.13.13 python scripts/stark_user_test.py report --run "$RUN"
```

This writes private `report.md` and `summary.json`; `run.json` retains aggregate
attempt history. Exit codes are 0 for passed, 1 for failed and 2 for incomplete.
Every selected client's required UI observation and all three project stages
must pass; current program and input hashes must still match. Missing/blocked
checks, a changed harness or changed installed applications prevent a full pass.
Observed failures retain the `failed` outcome even when identities change;
`evidence_valid`, `harness_unchanged` and `installed_unchanged` report identity
validity separately. Unchanged missing identities never count as valid evidence.
UI observations remain distinct from independently executed project checks.
The run establishes one task on the recorded versions/settings; an unknown
artifact digest limits artifact-specific conclusions. It does not qualify
arbitrary models, broad coding quality, cache parity, performance or app signing.

Reports contain aggregate outcomes, versions, model IDs, hashes and counts.
Keep prompts/responses, tool payloads, native artifacts, client databases,
credentials, exception dumps and screenshots out of published records. Do not
add any runtime output to Git. If a dated result is needed, publish only reviewed
aggregate evidence under `docs/archive/runs/` and preserve any previous results.

## Model-free helper checks

These validate the harness without launching apps, contacting Stark or loading
a model. Running them does not complete a desktop user simulation:

```sh
uv run --no-project --python 3.13.13 python -m unittest discover -s scripts -p 'test_user_simulation*.py'
```
