# Managed OpenWork connection validation — 2026-10-31

This record covers the helperless connection developed on
`codex/openwork-secure-connection`, based on Stark `b0f889f7`. It is separate
from the [earlier Keychain-helper experiment](OPENWORK_KEYCHAIN_E2E_2026_10_31.md).
Only aggregate results, public identities and hashes belong here.

## Implementation and source review

The connector is maintained in the separate first-party
[stark-openwork repository](https://github.com/jshahbazi/stark-openwork).
Stark bundles six files at revision
`1ca4c4586de8052c5a3bcb95455ab67e0b1fd95c`, checked against the SHA-256 hashes
in `apps/menu-bar/Resources/OpenWorkConnector.lock.json`. The normal build
accepts this exact source without a development override. The connector has
no dependencies and does not execute package-install scripts.

Connection Details installs public native OpenCode plugin metadata and saves
approval in a separate Keychain item. The unchanged official OpenWork runtime
obtains a renewable bearer capability through Stark's owner-private socket.
Stark does not export its persistent API key to OpenWork. The verified app and
its plugins share a trust boundary; bearer use itself is not kernel-bound to a
process. No validation credential helper participates in this connection.

A distinct reviewer checked the native peer, approval, broker, controller,
installer, backend admission, connector and builder paths. Findings resolved
before source signoff included absolute transport deadlines, acknowledgement
and revocation recovery, stale-port completion fencing, package/export
agreement, isolated-profile matching, background-hook failure recovery,
default-HTTP-port normalization and accurate lease/recovery wording.

## Component checks

| Check | Result |
| --- | --- |
| New native connection runner | 17 scenarios passed; approval and combined failure recovery, admission acknowledgement, revocation, session fencing, bounded frames and deadlines, safe paths, and zero net FD growth across 20 broker restarts |
| Native installer runner | 25 scenarios passed; JSONC preservation, conflicts, unsafe files and concurrent edits |
| Existing Swift controller | 111 checks passed |
| Menu bar builder | 35 tests passed |
| Focused Python API/desktop checks | 196 passed |
| Isolated API credential runner | 2 passed; no production Keychain namespace |
| Separate connector | 11 Node tests passed on Node 24.15.0 |
| Full locked offline converter suite | 5,161 passed, 88 skipped in 161.85 seconds |
| Repository policy, Ruff, formatting, whitespace | Passed |

Rust source is unchanged by this connection change; reused generation/scoring workers came
from the existing Rust 1.98.1 locked build.

## Packaged environment

- Client: official, unmodified OpenWork **0.18.63**, bundled OpenCode **2.0.26**.
- Official source tag commit: `9add0d2b8374d79537588c2303be5021e8605d48`.
- App identity: `com.differentai.openwork`, Apple Team `F5DJWB4CCV`; engine
  sidecar `opencode2`.
- Stark test bundle: `com.starkengine.menubar.validation.helperless20261031`.
  Its separate development Keychain namespace and owner-private persistent
  OpenWork profile leave the installed app's credential and normal client
  profile untouched.
- Packages use the existing builder/signing/install process with Developer ID,
  hardened runtime, secure timestamps and 37 signed runtime Mach-O images.
  They are not notarized. Production-identifier and notarized upgrades remain
  outside this evidence.

| Package | Host SHA-256 | Signed generation worker SHA-256 |
| --- | --- | --- |
| C, 0.11.10 | `4c32500634daf81531944b836adf1950ea8aee2f469b267f574a2a90eff9ec83` | `899eb761b061aa94726234742f886032603070cb4c8476bdf058d9b8faf70809` |
| D, 0.11.11, read-only privacy instrumentation | `83f4eb1ed32fb8e1d7233994cb4c3884c1185098890440cedac1d0689f7be54c` | `38197d3bd5213c00f11d4c8cd06c4d7d9aeb27c540ef0f22601d20cb0f68fef4` |
| E, 0.11.12, ordinary build without privacy instrumentation | `dbb4943d89403ed9cdcec9158a575ee4115da9572e5c0886238ebb1049c22c99` | `c9c37887d1b78fd5476a93654792c00bce04c2a93c850230ae1811678598f697` |

Package C established the initial workflow. Package D adds explicitly compiled
test-only instrumentation that remembers credentials only in Stark's memory
and scans bounded regular files in the disposable client profile on shutdown.
It emits only counts and sanitized failure categories. It does not participate
in authentication or provide credentials to a helper. The ordinary product
build excludes that instrumentation.

## Actual desktop results

Using package C, the real Connect to OpenWork button installed a single public
plugin entry. Its options contained only the expected loopback URL, socket
path and ownership marker. Reopening OpenWork loaded the connector and exposed
Stark's running `qwen3-dense` model in the normal model picker. The synthetic
chat completed with a matching expected marker: one assistant completion,
19,203 input tokens and 33 output tokens, stop finish and no client error.
The model was Qwen/Qwen3-0.6B with unquantized BF16 weights. No API key was
copied, no manual configuration was entered, and no credential helper ran.

The initial package B attempt refused installation because the isolated
profile was not matched through Electron's parent environment. Matching the
verified native descendant's public config directory fixed that test-only
isolation issue in C; the failure did not modify the normal client profile.

After manual unlock, the remaining desktop checks used the same official app
with a persistent disposable profile. No credential helper, copied key,
terminal setup command or manual JSON edit participated in these connections.
The test bundle supplied public profile-location selectors to isolate client
state; the normal app discovers the user's existing local workspace.

| Workflow | Observed result |
| --- | --- |
| D chat and follow-up through OpenWork's streaming adapter | Both expected synthetic markers matched; two stop/succeeded assistant turns; aggregate 37,912 input and 13 output tokens |
| D bounded tool task | Two distinct completed client-executed reads in separate assistant turns; full linked-path and result-content checks; subsequent correct computed sum; label-format miss recorded below |
| D Disconnect during generation | OpenWork task failed with HTTP 499; Stark showed disconnected; explicit reconnect then a clean chat succeeded |
| D Stark quit/relaunch while OpenWork stayed open | Saved model selection showed unavailable while no model was loaded, then became available after the same model was loaded; chat succeeded without Connect or client-setting changes |
| D to ordinary E signed upgrade while OpenWork stayed open | Chat succeeded without Connect or client-setting changes; saved endpoint and approval retained |
| E bounded tool replay | Two distinct completed client-executed reads in separate turns with correct paths and returned fixture content; final numeric checker failed |
| E OpenWork Stop | Task outcome interrupted; a later request could run |
| E stop/replacement during inference | Old request failed; the replacement model loaded; subsequent recovery chat succeeded after restoring the qualified model |
| E wrong bearer, native HTTP health request | 401 `invalid_api_key` using a synthetic invalid bearer |
| E occupied port | Applying occupied 18136 explained the conflict, retained actual endpoint 18135 and model readiness, and a fresh OpenWork chat succeeded through 18135; saved port restored to 18135 |

One earlier model-change attempt finished before the replacement action and
does not establish cancellation. The later stop/replacement check occurred
while inference was active. Its client task failed without a surfaced numeric
HTTP status; component tests separately verify `model_changed` and its status.

The final recovery chat returned its expected marker without errors, with
19,211 input and 8 output tokens. The public native configuration SHA-256
remained `eb5ebb0fca2cc8338826ebf1ce252ed930d23e1fa64f13f7e91594fa81ca451e`
across relaunch and upgrade. It contained one plugin with exactly
`expectedBaseURL`, `socketPath` and `starkConnector` options. Metadata-only
Keychain queries, explicitly disabling return of item data, found identical
creation and modification metadata before and after E. Native credential
tests separately verify byte-for-byte stored-key reuse and storage failures.

## Model behavior and artifact identity

The tool model was `mlx-community/Qwen3-4B-4bit`, revision
`4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25`, imported locally as
`qwen3-4b-4bit`. Precision was MLX affine 4-bit/group-64: 580 tensors and
2,263,417,416 weight bytes. Declared context/output limits were 40,960/4,096.
These capacities do not establish full-context performance qualification.

| Artifact | SHA-256 |
| --- | --- |
| Weights | `7c2af9a45e5b0d196f06a5cbfd6fc5cc95adf29df79885d0540f2206fb3d40d8` |
| Config | `5b505594bf00ef9682b373c6a292b88f94f1e1ceab449784c12accb835806c7f` |
| Tokenizer | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| Qualified upstream template | `87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5` |

The D task allowed exactly one read in each of two separate assistant turns.
The first fixture identified the otherwise-unknown second filename. Both
values were absent from the prompt. A hardened metadata-only checker verified
two unique call IDs, both `executed: false` (client execution), completed
states, exact linked paths, fixture values in successful `state.content`, and
the correct final computed numeric result. The assistant tool counts were
`[1, 1, 0]`, with two tool-call finishes and one stop; aggregate usage was
57,551 input and 228 output tokens. A distinct reviewer independently replayed
all seven fixed boolean checks successfully. The requested output label did
not match, including after whitespace normalization; strict formatting failed.

An initial less explicit task produced two calls together, one tool error and
an unsatisfactory final result. E's variant completed the same two client
reads and returned both fixture values, but failed the final-number checker;
its aggregate usage was 57,543 input and 206 output tokens. Merely containing
the expected number did not count as arithmetic success. These results
qualify a narrow result-use workflow and expose model limitations. They do not
establish reliable general agent behavior.

The checker bound HTTP replies to 8 MiB, used direct loopback HTTP with proxies
and redirects disabled, inspected only completed tool-result content for
fixture matches, and failed oversized final text. It emitted fixed booleans,
counts, known error categories and token totals. It retained no bodies, tool
arguments, output text or credentials. Checker SHA-256:
`cd222a515658d1eab7e81fd23cb19cdd47d04cd88b514379f7996ee656a9dc3a`.

## Privacy and cleanup

Test-only D instrumentation tracked the persistent key and every capability
issued during each host lifetime in RAM. It scanned enumerated owner regular
files in the disposable client profile for literal credential bytes. Both scans
completed within their bounded file/byte/work limits:

| Scan | Regular files | Bytes | Distinct secrets | Matching files | Skipped nonregular entries |
| --- | ---: | ---: | ---: | ---: | ---: |
| Before D relaunch | 11,057 | 194,526,435 | 3 | 0 | 12 |
| Before E upgrade | 11,063 | 195,125,575 | 2 | 0 | 12 |

This evidence covers literal bytes in the enumerated regular files. E excludes
the instrumentation and has no separate raw-key/capability scan claim. The
reviewed authentication and connector paths are shared by both builds. Client
plugins remain trusted code with access to request memory.

After validation, both apps and their sidecars were closed. The original
two-model list was restored by removing only the temporary import registration;
the converted artifact files were retained. The disposable OpenWork profile,
including sessions, fixtures, auth/config state, logs and caches, was deleted.
The test socket and its empty private parent directory were absent. The
isolated UserDefaults domain was deleted. The temporary installed cleanup
app was also removed after successful cleanup; original package E remains intact.

Out-of-bundle cleanup was refused with `errSecInvalidOwnerEdit`. A one-shot
cleanup host at the exact isolated signed app identity and installed path
successfully deleted only the test bearer and approval items; metadata-only
queries verified both absent. It requested no item data, changed no ACLs, and
never participated in the client connection. Production credentials and the
normal OpenWork profile were not accessed.

Retained evidence reports contain only aggregates and public build/model hashes;
validation source, reproducible build outputs and model artifacts remain ignored.
Production-identifier upgrades, notarized
distribution and arbitrary OpenWork plugins remain outside this validation.
Goose's historical CLI compatibility check remains separate; its desktop GUI
was not tested by this managed-connection run.

The connector branch `codex/helperless-connection` was published at the pinned
revision. The builder's normal network retrieval path fetched all six files
from that revision and verified their locked hashes with `source_mode: download`
and `development_override: false`. The ordinary full package E build used the
same hash-verified source locally, also without a development override.
