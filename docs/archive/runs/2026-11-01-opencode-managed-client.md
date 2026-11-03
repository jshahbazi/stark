# OpenCode Desktop managed client — 2026-11-01

This record covers adding OpenCode Desktop as a second managed client next to
OpenWork, on branch `claude/opencode-managed-client-de8248`, based on Stark
`6c396202`. The implementation is in `35dcb332` (backend revocation scope)
and `96c65050` (menu bar app, tests and reference docs). It holds public
identities, hashes and aggregate results only.

## Decision

The user chose to keep OpenWork and add OpenCode Desktop. For OpenCode's
detached shared engine, the trust rule is "approve each new OpenCode engine
once". The standalone and Homebrew `opencode` CLIs stay on the copied-API-key
path. OpenWork's behaviour, approval item, socket path and connector entry are
unchanged.

## Pinned OpenCode release

Source: the official release asset `opencode-desktop-mac-arm64.zip`, version
2.2.5.

| Item | Value |
| --- | --- |
| Asset SHA-256 | `9ae5c967d1adf7a4a831ba5366c29b3c4d2692b410174fce0b69c510935b7830` |
| App bundle ID, team | `ai.opencode.desktop`, `5NZ4Q7NXJ4` (Anomaly Innovations, Inc.) |
| App signature | Developer ID, hardened runtime, notarized; Gatekeeper accepted |
| `Contents/MacOS/OpenCode` SHA-256 | `b9aeb1e1fcd6b6c2a4818c2a9d79b4d2c8bcc9489368259e784d2350f7fbf01b` |
| `Contents/Resources/app.asar` SHA-256 | `a95223726f8524440019cba5f7c24056912d4a09aa5c8bf67e2db27cfc8a6161` |
| Engine | `Contents/Resources/opencode-cli`, signing identifier `opencode-cli`, version file `2.2.5` |
| Engine SHA-256 | `4365edf9f61588d323bfde9e8fbee7ec90b564e3e92ee5714801178e61f56f84` |
| Engine entitlements | `allow-jit`, `disable-library-validation` |

Both designated requirements have the shape Stark already uses: Apple anchor,
the identifier, the Developer ID intermediate and leaf markers, and leaf
`subject.OU = 5NZ4Q7NXJ4`.

## Release inspection

Read-only inspection of the unpacked 2.2.5 bundle (main-process JavaScript and
engine strings) established:

- **Staging path.** Desktop copies the engine to
  `<userData>/cli/<version>-<arch>/opencode-cli` and reuses an existing copy.
  For the production build `userData` is
  `~/Library/Application Support/ai.opencode.desktop`, not
  `~/Library/Application Support/OpenCode` as first assumed. A running 2.2.5
  engine on the test Mac used that path. The stage name is sanitized to
  `[a-zA-Z0-9._-]`.
- **Detached service.** Desktop spawns the copy as `serve --service` with
  `detached: true` and `unref()`. The engine inherits Desktop's environment. A
  registration file (`$XDG_STATE_HOME/opencode/service.json`, default
  `~/.local/state/opencode/service.json`) lets later Desktop launches and the
  terminal CLI reuse a ready engine of the same version. A version mismatch
  replaces it. The CLI has `opencode service start|restart|status|stop`.
- **Test isolation.** `OPENCODE_DESKTOP_TEST_ROOT` moves Desktop's `userData`
  (and therefore the staged engine) under `<root>/desktop` and sets
  `XDG_DATA_HOME`, `XDG_CONFIG_HOME`, `XDG_CACHE_HOME` and `XDG_STATE_HOME`
  under the root. That isolates the engine's configuration and service
  registration. Stark's validation profile uses it.
- **Configuration.** The global configuration directory is
  `OPENCODE_CONFIG_DIR` when set, otherwise `$XDG_CONFIG_HOME/opencode`. A
  `plugins` entry may be a string or `{package, options}`. `file://` targets
  are resolved to directories, which matches the entry Stark already writes.
- **Plugin API.** The engine still provides `provider.transform`,
  `provider.reload` and `session.hook("http.request", ...)`, which the
  connector uses.
- **Configuration reload.** The engine contains
  `ConfigPluginSource.watchConfiguredSources` and reloads providers and agents
  on `config.updated`. This suggests a new plugin entry is picked up without
  restarting the engine. It has not been confirmed in a live run (see below).

## Connector

The bundled connector stays at `jshahbazi/stark-openwork`
`1ca4c4586de8052c5a3bcb95455ab67e0b1fd95c`. A local clone of that revision
reproduced all six SHA-256 values in
`apps/menu-bar/Resources/OpenWorkConnector.lock.json`. Its source sends only
the `status` and `connection` operations to the configured socket. It checks
the expected loopback base URL and the OpenCode plugin APIs, and contains no
OpenWork app, team or sidecar check, so it is client-agnostic. Its package
name and placeholder key text still say "openwork"; they are not shown as
identities and need no change.

## Implementation

- `DesktopClient` (`openWork`, `openCode`) carries each client's display name,
  bundle ID, team ID, engine signing identifier and backend client-ID prefix.
  `DesktopApplicationIdentity` replaces `OpenWorkApplicationIdentity`.
- One broker socket and one connector entry are shared. `DesktopConnections`
  owns the broker. For each peer it tries every running verified client app,
  identifies the client and dispatches to that client's controller. The socket
  path, ownership marker and entry format are unchanged.
- `DesktopClientCapabilities` holds the grant, renewal, acknowledgement and
  revocation logic that `OpenWorkConnectionController` previously contained.
  Both controllers use it.
- OpenCode peer rule:
  - the audit token's code must satisfy the `opencode-cli` / `5NZ4Q7NXJ4`
    requirement;
  - its executable must resolve to `opencode-cli` directly inside one staging
    folder (`[A-Za-z0-9._-]{1,64}`) under the real account home's
    `Library/Application Support/ai.opencode.desktop/cli`;
  - its first two arguments must be `serve --service`, as Desktop launches it;
  - its launch arguments and environment must pass the existing checks. OpenCode
    additionally rejects non-empty `DYLD_*`, `BUN_BE_BUN`, `BUN_INSPECT*` and
    `OPENCODE_DESKTOP_REMOTE_DEBUGGING_PORT`, and invalid `HOME` or
    `XDG_CONFIG_HOME` paths;
  - a verified OpenCode Desktop app must be running, and the engine's resolved
    configuration directory must equal the one resolved from that app's launch
    environment. The engine inherits Desktop's environment, so this binds the
    peer to the configuration Desktop hands its own engine.

  There is no ancestry requirement. OpenWork's launch checks are unchanged.
- Approval: `OpenCodeConnectionController` approves the engine it verified
  during **Connect**. Other verified engines get `not_connected` for `status`
  and `.unapproved` for `connection`, and appear in Connection Details with
  **Allow** and **Deny**. Approvals are stored in the Keychain item account
  `opencode-approval-v1` as
  `opencode:5NZ4Q7NXJ4:ai.opencode.desktop:v1:enabled:<pid.sec.usec>,...` (at
  most eight, dead engines pruned on save) or `...:v1:revoked`. Denials are
  kept in memory only.
- Client IDs are `opencode.<pid>.<sec>.<usec>`; OpenWork's remain
  `openwork.<pid>.<sec>.<usec>`.
- Backend: `revoke_api_clients` accepts an optional `client_prefix` (same
  grammar as `client_id`, exclusive with it). OpenWork's Disconnect now sends
  `openwork.` and OpenCode's sends `opencode.`. A command with only
  `api_session` still revokes every client. The shared admission table grows
  from 16 to 24 entries so OpenWork's unchanged cap of 16 and OpenCode's 8 fit.
- Install target: the engine's `OPENCODE_CONFIG_DIR`, otherwise
  `$XDG_CONFIG_HOME/opencode`, otherwise `$HOME/.config/opencode` from the
  engine's environment, falling back to the account home.
- Test profile: the existing `StarkOpenWorkTestProfile` key, plus a new
  `StarkOpenCodeTestApplication` key. OpenCode is launched with
  `OPENCODE_DESKTOP_TEST_ROOT=<profile>/opencode-desktop`, and only engines
  under that root's `desktop/cli` are accepted.

## Independent review

A separate reviewer read the uncommitted diff against `HEAD`. It confirmed:

- the OpenWork refactor kept OpenWork's approval item, socket path, connector
  entry, client IDs, launch checks and grant, renewal and disconnect paths;
- peers cannot be routed to the wrong client;
- each controller ignores the other's acknowledgements;
- each Disconnect revokes only its own prefix;
- the backend rejects ambiguous or malformed prefixes.

It reported five findings, all fixed before commit:

| Severity | Finding | Fix |
| --- | --- | --- |
| Medium | Connect approved the newest staged engine, which a same-user process could start with its own configuration | Engines must run as `serve --service` and read the configuration directory resolved from the running Desktop app; Connect only considers those engines |
| Low | `BUN_BE_BUN=1` turns the compiled engine into the plain Bun interpreter | Rejected for OpenCode peers |
| Low | Swift caps (16 plus 8) exceeded the backend's 16-entry table | Backend table raised to 24 |
| Low | Connect during the startup Keychain read could overwrite saved engine approvals | Connect waits until approvals are loaded |
| Low | After a port change during Connect, Keychain recorded the approval but memory did not | OpenCode updates its in-memory approval before reporting the port change |

The last finding also exists in OpenWork's unchanged connect path, where it
only affects a yes/no flag. It is left as is to keep OpenWork's behaviour
unchanged.

## Interaction with the prefix-reuse branch

Branch `claude/eager-maxwell-613814` (`cc942e25`, `8c0e9e37`, `c2cac138`,
unmerged) keys a prefix-KV cache scope on each managed client grant
(`client_id`) and resets that scope on revocation. This change adds
prefix-scoped revocation to `revoke_api_clients`. When the two land together,
the prefix branch's revocation reset must also cover `client_prefix`
revocations, resetting every scope whose client ID starts with the prefix.
Client IDs stay unique per engine instance for both clients.

## Component checks

| Check | Result |
| --- | --- |
| `scripts/test_desktop_connections.py` | 27 native scenarios passed (17 existing, 10 new: OpenCode approval store, pending Allow, Deny, disabled, corrupt and still-loading approval, scoped revoke and pending retry, pruning and the eight-engine cap with live child processes, staging-path rule, config-directory precedence, launch-environment rejections and `serve --service` detection, coordinator routing and socket lifecycle) |
| `scripts/test_openwork_connector_installer.py` | 26 native scenarios passed (new: shared client entry) |
| The two scripts above plus `scripts/test_menubar_controller.py`, `scripts/test_build_menubar.py`, `scripts/test_api_credentials.py` | 157 passed |
| `tools/converter/tests/test_local_api_clients.py`, `test_desktop_api_clients.py`, `test_local_api.py`, `test_desktop.py` | 246 passed, including prefix revocation with live streams, invalid and ambiguous scopes, wire validation and the 24-entry table |
| Native scenario repeatability | Ten timing-sensitive scenarios passed in three repeated rounds |
| Optimized app build and validation-flag type-check | Passed |
| Full offline converter suite | 5,312 passed, 88 skipped |
| Ruff check and format on changed Python | Passed (one pre-existing format finding in `scripts/test_menubar_controller.py` is unchanged) |
| Repository policy check, `git diff --check` | Passed |
| App type-check (`swiftc -typecheck` over `apps/menu-bar/Sources`) | Passed |

No production Keychain item, Stark installation or OpenCode profile was read or
modified by these checks.

## Not yet validated

No desktop run has qualified the connection with the real OpenCode Desktop
app. Two questions remain:

- whether a fresh connector install takes effect in a running engine (the
  configuration watcher suggests it does), or the engine must be restarted;
- the full Connect, Allow, chat, Disconnect and reconnect flow against a
  signed Stark build.

Connection Details and the documentation tell the user how to restart the
engine (`opencode service stop`, then reopen OpenCode and choose **Allow**). A
live run needs a validation Stark build with its own Keychain namespace and an
OpenCode test root. Record it separately when it is done.
