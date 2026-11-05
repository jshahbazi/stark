# Beginner Guided Steps implementation and local validation

Date: 2026-11-04 (America/New_York).
Feature commit: `53dae9b31ab04c5dc660bbe02414f0f7fd41d80e`.
Base: `c7b89c8b556f61c8bf8fa7db19dfbd5605392973`.
Branch: `codex/beginner-guided-setup`.

## Delivered behavior

- A separately persisted Beginner/Advanced interface preference. Initial model
  inventory selects Beginner for an empty library and Advanced for an existing
  library. Chat/Decision selection remains a separate preference.
- Native Guided Steps for task choice, a source-bound starting model, actual
  download/preparation progress, explicit start or switch, and managed OpenWork
  setup. Advanced retains the existing model browser, source import, Playground,
  connection controls and options.
- Try a different model from the ready step and Beginner chat. Recommendations,
  candidate progress and installation results have separate operation identities.
  Choosing, preparing or canceling a candidate preserves the loaded worker and
  conversation. Requests may wait during installation because the backend remains
  serialized. A matched explicit start acknowledgement clears the old chat.
- OpenWork installation remains a visit to its official download page followed
  by manual installation. Workspace and model selection are explicitly confirmed
  by the user in OpenWork. Stark does not choose a folder or execute its tools.
- Readiness requires a running Chat model with declared capacity and admitted
  tool support, local API availability, saved authorization, a recent verified
  connector status advertising the actual API model ID and context, and the
  user's workspace/model confirmations. The five-second connector poll refreshes
  a thirty-second observation; an HTTP request or capability lease is not needed
  to finish setup. Confirmations are invalidated on model/client/profile/session
  changes and revocation.

Current behavior is documented in the [menu bar guide](../../../apps/menu-bar/README.md#beginner-and-advanced-setup).

## Starting-model and context policy

| Recipe | Immutable source | Metadata route |
| --- | --- | --- |
| Qwen3 4B, 4-bit | `mlx-community/Qwen3-4B-4bit@4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25` | `hf-generic` with its own pinned sidecars |
| Qwen3 0.6B, BF16 | `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca` | Existing Qwen3 catalog route, with the complete pinned own-repository source selection |

The manifests, licenses and exact source-file identities were checked on
2026-11-04. Receipt matching requires the complete expected source selection,
metadata policy, source-manifest hash and converted-config binding. Native model
admission remains mandatory. A short-lived backend offer is required to install
or start a guided recipe; names and catalog popularity do not grant eligibility.

Both recipes currently require at least 16 GiB physical memory. Runtime planning
estimates are 10 GiB and 9 GiB respectively, derived from selected weights, BF16
KV backing for the initial context plus a 1,024-token suffix, and a 3 GiB
workspace reserve. These are conservative enrollment estimates, not measured
per-Mac fit or quality guarantees. Storage checks cover both the installation
volume and download cache. Resources are checked again before installation and
after releasing the previous worker for an explicitly requested switch.

Guided startup applies a 32,768-token budget to local chat and the API. The raw
native capacity stays separate; `api_context_tokens` reports the actual served
budget. An already-running starter with a smaller or larger policy requires an
explicit restart to apply guided settings. Advanced retains its native-capacity
controls, and a later ordinary Run restores its ordinary API policy.

This budget is informed by the preserved
[OpenWork Keychain qualification](OPENWORK_KEYCHAIN_E2E_2026_10_31.md): 16,384-token
attempts failed, while accepted client runs served 40,960 tokens. Existing
source-bound client records observed initial inputs of approximately 17,000 to
19,211 tokens. Those records were not changed or relabeled as a new 32K client run.

## Local checks

| Check | Result |
| --- | --- |
| Repository policy | Passed |
| Ruff and format, converter plus changed Python scripts | Passed |
| Whitespace | Passed |
| Full model-free converter suite | 5,984 passed, 89 skipped |
| Combined native/controller/window/beginner/connection/credential/app-build checks | 184 tests passed |
| Full Swift source typecheck | Passed, arm64 macOS 26.4 target |
| Native generation worker | Rust 1.98.1, release, locked, passed |
| Native scoring worker | Rust 1.98.1, release, locked, `optimized`, separate Cargo invocation, passed |
| Native template helper | Rust 1.98.1, release, locked, passed |
| Local app package | Ad-hoc build from clean feature commit, passed builder checks |

The workers were built in this checkout's own `runtime/cargo-target` directory.
No Rust inputs changed from the base commit. Worker build receipts preserve the
exact commands, Rust input digest and executable hashes; packaging records the
current worker/converter overlays separately from the inherited CLI snapshot.
The app's bundled `desktop.py` and `guided_models.py` were checked byte-for-byte
against the committed sources.

The broad suite initially exposed four fixture/recovery regressions: a synthetic
status event without a message, and image fixtures constructed without the new
converter termination lock. These were fixed and the complete suite rerun. A
separate review also caught early-rejection recovery, initial status-handshake
readiness, mode-window switching, confirmation scope, and both directions of a
guided context mismatch. The final checks include regressions for those paths.

## Source-bound native model checks

A disposable model library exercised the real guided protocol and converter
routes. No production library, API credentials, OpenWork profile or installed
application was changed.

- Two fresh conversions using the exact recipe source policies.
- Three native loads and five real chat requests: 4B, 0.6B, then 4B again.
- Both models completed a 20,023-input-token synthetic prefix under the 32,768
  budget, longer than the largest preserved stock-client input above.
- Both declared `qwen3-hermes` tool parsing. This is protocol support, not a
  general-purpose agent competence result.
- Installing the 0.6B candidate preserved the loaded 4B worker and conversation.
- Reusing 4B retained its installed ID and did not run the converter again.

Only aggregate results and hashes are retained here. Prompts, responses, token
vectors, probabilities, credentials and exception dumps are not included.

| Evidence | SHA-256 |
| --- | --- |
| Final private aggregate | `ff86677abd654be339d7eb5ad66fc6fdd07c6cc5b0c7038853eaf2f26859759e` |
| Qualification harness | `30ab4d85c1036441833629191ff6d2433a1d00571ddea4af1b744ae141272e0b` |
| Desktop controller | `9e268794965b787c71adcf2d53ad62df4109a7cb836ce21fb7b0f197dda0ef8b` |
| Guided recipes | `dd200774638682149a1e896f29783a50fed7eca510153caf1d3978215e5d89f2` |
| Generation worker before packaging/signing | `3413696a9c98be45f1476b67e0fd9d83b394e54136b99eca016564cdfa10608d` |
| Template helper before packaging/signing | `1057fa450134f579285e014846a12e60525769dd1b0a4e9e57116ee2a389fffb` |
| Final app manifest | `36fff6b3efe3ccebc14f92b2fe777dcf6cd95e69635c6ca32eba3d2673707374` |

Earlier 4K qualification attempts and the intermediate failed source-route run
remain separate private records. The passing 32K aggregate identifies the final
controller and recipe hashes above.

## Native interface checks and limits

An isolated validation app compiled the actual native guide, state and interface
picker without a backend, production preferences or Keychain access. Accessibility
interactions verified task selection, recommendation, ready, Try a different
model, and Beginner/Advanced switching with retained state. It was quit after
inspection and removed its disposable preferences.

Offscreen native renders showed nonblank task, recommendation, ready and retry
bodies fitting at 900 by 740 points. Desktop capture was unavailable. The native
segmented control rendered blank in the offscreen cache, so its pixels were not
qualified; its labels and selection were verified through accessibility.

The final app package passed imports, converter provenance, worker self-checks,
connector pinning, resource checks and ad-hoc seal checks. Packaging itself did
not load a model. The source-bound model checks above are separate evidence.

A fresh real OpenWork desktop task at 32K was not run. Native large-prefix tests
support budget feasibility and prior client records establish the source
integration, but neither proves a new complete client session or general task
success. No hosted CI, production release, notarization or deployment is claimed.
