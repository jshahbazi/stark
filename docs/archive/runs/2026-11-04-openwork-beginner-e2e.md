# Fresh OpenWork Beginner setup validation

Date: 2026-11-04, America/New_York (2026-11-05 UTC).

## Scope and isolation

The live test used the production Beginner views, session state, interface
preferences, EngineController, connection broker and packaged Python/native
runtime. A private regular-app delegate supplied a disposable data directory,
unique bundle/Keychain namespace and OpenWork profile. Its single-window routing
differs from the production menu-bar delegate; production window routing is
covered separately by the native window checks. The driver did not advance
steps, approve connections, start models or confirm client selections.

All setup actions were performed through the native interfaces. The official
OpenWork app was downloaded privately before testing; its own model picker,
local-workspace chooser, engine and connector were used. This exercises fresh
client configuration with an already available official application, rather
than the browser download or dragging an app into Applications.

Each fresh run began with an empty Stark model library, empty OpenWork profile
and unset approval. Model-host source caches could be reused. No production
Stark library, preferences, API key or OpenWork profile was used. Only aggregate
results, versions and hashes are recorded here; request/result content, token
vectors, credentials and exception dumps are excluded.

## Environment and client

| Item | Value |
| --- | --- |
| Host | Apple M5 Max, arm64, 128 GiB physical memory |
| macOS | 26.7.1 |
| Swift build target | arm64 macOS 26.4 |
| OpenWork | Official 0.18.64 arm64 release |
| OpenCode bundled by OpenWork | 2.2.0 |
| Client identity | `com.differentai.openwork`, Team `F5DJWB4CCV` |
| Client verification | Strict/deep signature check and official release digest matched |
| OpenWork archive SHA-256 | `5c5c9ca59e91c38564d122b1c8ed17fcb6285ee61ad1523abb8220e7f7ac0ee0` |

The client came from the [official download page](https://openworklabs.com/download)
and [0.18.64 release](https://github.com/different-ai/openwork/releases/tag/v0.18.64).
This host does not qualify model performance or memory fit on a smaller Mac.

## Complete fresh run at f2937a80

The first source-bound run compiled production source
`f2937a80f9caa4451a16f6aa295a8dcb9878fa34` with observer-only instrumentation.

- Beginner was selected after the first actual empty inventory. Switching to
  Advanced showed the existing model manager; returning retained the task choice.
- Guided Steps recommended and installed the pinned Qwen3 4B recipe, then
  started it only after an explicit click.
- The operator created a disposable local workspace in OpenWork, authorized the
  connection, completely quit/reopened the client, selected Stark and the running
  model, and confirmed both choices. At the observed checkpoints, Finish was
  disabled until the required checks and confirmations were present.
- A small arithmetic task completed with the expected result. A second task
  completed a real `read` tool call and returned the expected fixture heading.
- Try a different model offered Qwen3 0.6B. Installing it retained the selected,
  loaded 4B model. Only the explicit Switch action changed the running model.
- OpenWork retained its workspace and authorization. The new model was selected
  in its picker and confirmed in Guided Steps before finishing again.
- The same two task checks succeeded on 0.6B. A separate read-only audit of the
  client's disposable database confirmed a completed `read` call and nonempty
  expected final result for each model, without exporting their content.
- Keep current model returned to Ready. Switching back to the installed 4B model
  required no additional install and required model-selection confirmation again.

The aggregate contains two installs, three explicit model starts, eight completed
API requests (four per model) and no engine/guide error events. Observed request
inputs reached 15,933 tokens on 4B and 15,924 on 0.6B. Both ran with 32,768 local
and API budgets while retaining a separate declared native capacity of 40,960.
Completed setup stayed Ready during generation. An independent audit passed
17 checks across the final frozen state transitions.

The native Ready screen was also visually inspected. Both test applications
were quit. The disposable client profile, preferences and exact validation
Keychain items were removed after recording the aggregate audit.

## Review fixes and final local checks

Independent review and the live test led to these fixes:

- Start admission now returns acceptance to the guide; a rejected action cannot
  leave a pending start. Busy/missing-model rejection preserves current state.
- Open OpenWork and Connect share profile-aware launch configuration, including
  failing closed when an isolated profile cannot be created.
- Same-model generation preserves completed setup while stop, changed model,
  revocation, expired discovery and changed client scope still invalidate it.
- The sidebar describes a retained model as loaded while another is preparing.
- Each Host snapshot queries current-model discovery directly. It cannot reuse
  the previous model's cached periodic observation during a switch.

Production source through `f45b0c85` and the regression at `033f2165` passed 186
combined app/controller/native checks. The new actual-Host regression also failed
when the previous cached-discovery expression was restored only in its temporary
compiled fixture, confirming that it catches the reviewed timing issue.

The full converter rerun passed **5,984 tests, 89 skipped**. Ruff, formatting,
repository policy and whitespace checks passed. The complete production app
package built and passed its resource/provenance/worker/ad-hoc verification at
`f45b0c85`; test-only changes at `033f2165` do not change that package's sources.
The packaging metadata marks the checkout dirty because the window regression
fixture was being edited. The production source revision and exact packaged
source/resource hashes remain recorded separately.

## Complete replay with the final discovery fix

A new empty library, empty client profile and unique validation namespace
replayed the complete flow at `033f2165fc34db360610bc6e8aff7db4797d6f72`.
The production Swift code matches `f45b0c85`; the newer commit adds its regression.
All 18 production Swift source hashes in the final driver manifest were checked
against the committed `033f2165` files with zero mismatches.
Both models again reached Ready, each completed the arithmetic check and a real
read-tool round trip, and the client result audit confirmed both expected results.
The new run recorded two installs, two starts and eight completed API requests.
Maximum observed inputs were 15,926 tokens for 4B and 15,924 for 0.6B, within the
same 32,768 local/API budgets and separate 40,960 native capacity.

The final driver records the Host's current-model check separately from the
periodic observation. Two sampled transitions retained a true cached observation
while the actual Host correctly returned false. The immediate post-switch view
showed that it was waiting for the new model, with Finish disabled. After fresh
discovery and the operator's model selection/confirmation, setup completed again.
The corrected loaded-model sidebar copy was also observed during preparation.
The final independent audit passed 19 checks: every sampled Ready state had a
verified current-model discovery, both stale-cache samples remained at Connect
with the new model unconfirmed, and one validation host process served the run.

An intervening replay was interrupted by native automation focus/window access
and a test-driver restart. Its partial record and cleanup receipt remain separate;
it is not counted as a completed setup or task run. The successful final wrapper
used a distinct app/window/executable identity. The actual Beginner views,
controller, runtime and client authorization were unchanged by that wrapper fix.

Both final test applications were quit after the frozen aggregate and client
audit were saved. The final profile, defaults domain/file and exact three test
Keychain accounts were then verified absent. The test API port was closed and
bindable. Models, app bundles and aggregate receipts remain ignored and private.

The [hosted development bundle build](https://github.com/jshahbazi/stark/actions/runs/41944434378)
completed successfully for `033f2165`. This is separate from subsequent PR checks.

## Source-bound receipts

Private artifacts remain ignored under `runtime/e2e-beginner-2026-11-04/` and
`artifacts/`. The original component qualification record remains separate in
[the earlier run](2026-11-04-beginner-guided-setup.md).

| Receipt | SHA-256 |
| --- | --- |
| First completed aggregate | `289736c892d03d0bd2fab5deeb141824f604fe2b86c311ab9eecfa20865defe3` |
| First driver build manifest | `a327415d1a72e1f257f403168044cf39db1f1cb013048f00b4d78a8dfafb48c0` |
| Independent 17-check audit | `a6c9648ef0494c187dcd4b37ab46fba3c9ab094c4102420b8fd97e35abc4dda6` |
| Read-only client result audit | `4cd40a84ab897d4e068d7f6a2ec7ef19d650d216f5f36ca4a87b30726a74929e` |
| First cleanup receipt | `3772459a1dcb3d2a2c8321de3c9637b2f1d0f62b7e657a0e6cb0046a99e1f54b` |
| Production packaging receipt, outer `Stark.app.manifest.json` | `c9acfd29c2a7fe0dd54b2ede346b1c634113154bdb07c8e3208bdcd1829f40b7` |
| Inner `Resources/menu-bar-manifest.json` | `5294df79e5efcff12aad82b15bfac62df07389a0de6ca1c26812e56a703de7fa` |
| Final completed aggregate | `49b8a02ccc801d70396341ee7c89b9115fc591338e7031a1b7f1658f39ad18fe` |
| Final driver build manifest | `e07a04c3ca61b0a0d1dca6922a72be7dddfcd3045ca8b4c9d7d73508cf13e5a5` |
| Final read-only client result audit | `00300d6fa58d7794d7e867aba68dba3ac385fd0e59f5086000ead45e114b164c` |
| Final independent 19-check audit | `9678028f4eacffb929b421e6782ab111ac4fa0bb8d448680f22da4387ef94c5d` |
| Final cleanup receipt | `be96d0a0d7b13e605c3eddfa0b5fc0a6cd70aef2d79870fa7df92efa003a2d10` |

These bounded tasks qualify the tested setup and tool round trip. They do not
establish general agent quality, every OpenWork tool, long-session reliability,
model fit on every supported Mac, or a signed/notarized Stark release.
