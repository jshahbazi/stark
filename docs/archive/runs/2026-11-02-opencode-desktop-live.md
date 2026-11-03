# OpenCode Desktop managed client: desktop run — 2026-11-02

This record covers the first desktop run of the OpenCode Desktop managed
client added in the [implementation record](2026-11-01-opencode-managed-client.md).
The user drove the Stark and OpenCode windows; the checks below read process
state, public configuration, Keychain item attributes and aggregate fields of
the client database. No prompt, response, session title or credential value
was read, saved or published.

## Environment

- Host: Apple M5 Max, 128 GiB, macOS 26.7.1.
- Source: branch `claude/opencode-managed-client-de8248` at `5e8d362a`, which
  merges `main` at `850832f2` into the implementation. The merge also extends
  the prefix-cache reset to `client_prefix` revocations (see below).
- Stark test package, built with `scripts/build_menubar.py` and Rust 1.98.1
  workers from that source, then copied and re-signed:
  - bundle identifier `com.starkengine.menubar.validation.opencode20261102`,
    version `0.0.1`, which selects its own development Keychain namespace and
    broker socket;
  - Info keys `StarkOpenWorkTestProfile` (a private, owner-only profile folder)
    and `StarkOpenCodeTestApplication` (the unpacked release below);
  - Developer ID Team `S5BT2PRN5B`, hardened runtime, secure timestamp,
    37 signed runtime images; not notarized;
  - local API on port 18140. The installed `/Applications/Stark.app` was not
    modified.
- Connector: bundled `stark-openwork` `a5534db0526e08c9b9b4df12825c9da7117cae86`.
- Client: official OpenCode Desktop 2.2.5, unpacked from the asset pinned in
  the implementation record; all four pinned SHA-256 values matched and
  Gatekeeper accepted the app (Notarized Developer ID, Team `5NZ4Q7NXJ4`).
  Stark launched it with `OPENCODE_DESKTOP_TEST_ROOT` inside the test profile.
- Model: `alibiserikbay/JevK5` (Stark ID `0dbfc3c58fb64720b3142500958537fd`),
  converted output 8,413,483,320 bytes, SHA-256
  `ea3fd57de724a9cc34cce999f42e78f30245e11bdf18938377bd367d4bbf0dc3`, in Chat
  mode.

| Artifact | SHA-256 |
| --- | --- |
| Signed test app executable | `27ceaa63dfd3d4a7442bff6aa0524613642aa8995e89f6882cea1767f7a73893` |
| Signed generation worker | `7aad4b186ea4b83cece0821b8c4e9e8c096d85ec9a8575518298127832b14ccc` |
| Unsigned generation worker | `c338b1efea3bd647bb175d42b2b35b9686889f333026d523dcb37afc797c38b4` |

## Results

| Step | Observed result |
| --- | --- |
| First **Connect** | Stark opened the test OpenCode, verified its staged `opencode-cli` engine (`serve --service`, configuration directory matching the running app), approved it and wrote one connector entry into the test profile. About 15 seconds later the engine exited: OpenCode's managed service port 49374 was already held by this account's existing OpenCode engine. This was a test-setup conflict, not a Stark failure; the test profile was given port 49375 with `opencode service set port` |
| Live configuration reload | With the entry set aside, OpenCode started a fresh engine. **Reconnect** wrote the entry while that engine was running. The engine logged `config.updated` in the same millisecond and loaded the Stark connector 0.16 seconds later, without restarting. Stark appeared in OpenCode's model picker |
| Chat | One turn on `stark` / `alibiserikbay/JevK5` succeeded: 8,896 input and 8 output tokens |
| New engine and **Allow** | `opencode service restart`, run with the staged binary and the test profile's locations, started a new engine instance. Stark held it as pending; after the user chose **Allow**, the approval item changed and the next turn succeeded. Session totals: two successful turns, 20,092 input and 13 output tokens |
| **Disconnect** | The approval item was rewritten; within four seconds the engine logged provider and model updates, and the user's next message failed. No further turn succeeded |
| Cleanup | The test engine, test OpenCode and test Stark were stopped. The account's own OpenCode configuration and service registration had unchanged SHA-256 values and modification times, and its existing engine was not touched |

The client database showed no message rows for the attempt made before
**Allow**, so how OpenCode presented that refusal was not recorded.

## Findings

- **Configuration reload.** A running OpenCode 2.2.5 engine picks up a newly
  written plugin entry. This answers the open question in the implementation
  record. The restart instructions in Connection Details and the docs remain as
  a fallback.
- **Engine restarts.** When the engine was stopped while OpenCode was still
  open, OpenCode started a new one. The restart instructions therefore say to
  quit OpenCode before `opencode service stop`.
- **No ancestry needed.** The engine restarted from the command line inherited
  the test profile's locations rather than OpenCode's own environment. Stark
  accepted it as a pending engine because it ran from the staging folder as
  `serve --service` with the configuration the running app uses.
- **Fixed service port.** Two OpenCode Desktop profiles under one account
  cannot run engines at the same time without a port override. Isolated
  validation profiles need one; normal use does not.
- **Service environment.** `opencode service set` can also store environment
  variables for the engine. Stark checks the engine's actual launch environment,
  so loader, inspector and interpreter variables are still rejected and a
  different configuration directory does not match. This was not exercised.

## Prefix-cache interaction

The prefix-reuse work merged into `main` before this run. A `client_prefix`
revocation now also rotates every prefix-cache scope whose client ID starts
with that prefix. Other clients and the master key keep their scopes. A new
test covers this.

## Checks after the merge

| Check | Result |
| --- | --- |
| Menu bar scripts: desktop connections, connector installer, controller, build, API credentials | 160 passed |
| Focused backend tests including prefix cache and tool prefix safety | 520 passed, 1 skipped |
| Full offline converter suite | 5,853 passed, 89 skipped |
| App type-check, including with `STARK_CONNECTION_VALIDATION` | Passed |
| Ruff check and format on changed Python | Passed |

## Privacy

Only the aggregate fields above were read from the client database
(model ID, token counts, outcome, message counts and times), through a
read-only connection. During setup, the disposable test engine's per-launch
loopback password was displayed once in the operator session; that engine was
stopped and the value is not recorded here. No Stark API key or capability was
read, copied or used.

## Not covered

- Tool calls, **Deny**, port changes and Stark restarts with OpenCode open.
- OpenWork and OpenCode connected at the same time in one desktop run.
- The production bundle identifier, the installed app and a notarized build.
- One session and one model; not a performance measurement.
