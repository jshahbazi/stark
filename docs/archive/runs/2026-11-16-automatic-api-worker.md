# 20261116-PIAUTO-01: automatic worker selection for connected apps

## Identity and decision

- Registered and executed: 2026-11-16; owner: Codex.
- Kind: compatibility, lifecycle and cache ownership.
- State/result: completed; bounded functional checks passed.
- Disposition: adopted on `codex/automatic-api-worker`; signed development build
  `20261116210119608069` installed at `~/Applications/Stark.app`. No release or
  merge qualification is claimed.
- Related work: [Pi ledger](../../model-experiments/2026-11-13-pi-connections.md).
  This is a separate attempt in the [model experiment ledger](../../model-experiments/README.md).

## Trigger and implementation

An existing Pi configuration requested generation while the Playground was in
Decision mode. Stark had loaded a scoring worker, advertised no chat model, and
validated Pi's output limit against stale generation limits before checking
readiness. The user saw the guard's generic HTTP 400.

Stark now selects generation for authenticated discovery/completions and scoring
for Playground decisions. Preparation reserves the operation slot and loads one
worker at a time. The Playground selection remains unchanged. Active work is
preserved with `429/model_busy`; failed preparation uses
`503/runtime_load_failed`. Missing readiness precedes output-limit validation.
Stop/model replacement fence startup and stale cleanup cannot close a replacement.

Pi recognizes a fixed allowlist of error codes and emits fixed guidance. Error
bodies have an 8 KiB limit and a one-second read deadline; raw upstream messages
are discarded. The public Pi resource revision is
`069e37e2be473eeb13da9702da3ae3b6ba77f4c2` from `stark-agent-clients`.

## Artifact, source and environment

- Baseline source: `3d17fd688652fda4ff69f6119e98c26392a94ff7`.
- Candidate desktop module SHA-256:
  `ab467a57d1a8a7cc20e06050eedda1a3ffa823f33805f104be6adbf70bc6520e`.
  The installed manifest records that base revision, dirty source hashes and the
  signed build number; manifest SHA-256:
  `e6f7f7af7c58ebd8fdc8c4c1259ae10c5d23922758d72f5cffe34e2d6f4aea57`.
- Artifact: local `qwen3-4b-4bit`, Qwen3 4B, 4-bit, displayed source
  `mlx-community/Qwen3-4B-4bit` revision
  `4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25`.
- Artifact hashes: config
  `5b505594bf00ef9682b373c6a292b88f94f1e1ceab449784c12accb835806c7f`;
  tokenizer `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`;
  weights `7c2af9a45e5b0d196f06a5cbfd6fc5cc95adf29df79885d0540f2206fb3d40d8`.
- Installed signed worker hashes: generation
  `b567c246d4a43aedeb5c4b30a1d634aa48bc4cb07d7f0356cd1b9ff5c7ebe2c4`;
  scoring `a244449c7fb058ce6637f447e40da1ece417e76c1aabd406af4b3e0a5e20b76f`;
  template `cda4e60838de4d01d62514cc18b576531b48563356e9476179a15454f731e5cd`.
- Native workers were reused from the installed app and signed during packaging.
  The signing step changes their file hashes. No Rust/kernel or precision policy
  was changed. Internal activation/accumulation/KV precisions were not independently
  inventoried by this lifecycle check.
- Environment: Apple silicon arm64, macOS 26.7.2, uv-managed Python 3.13.13,
  installed Pi 1.5.2. Hardware utilization and paging were not benchmarked.

## Controls and results

The native check loads scoring, requests generation, repeats the generation on
the same worker, performs a decision, discovers generation capabilities, and
requests generation again. It verifies that replaced worker processes exit,
the Playground mode remains Decision, consecutive generation retains its worker,
and all three greedy response messages match. The synthetic generation fixture
has 16 prompt tokens, an eight-token output limit, temperature zero and two
completion tokens. Three responses matched; response-message SHA-256:
`9619a98f5bcf3d1505ff346dad57f8fad3dd5ee19e56b434cd12b2a76cbb1e1d`.

The installed-app check starts a scoring worker through the Decision Playground,
then launches a fresh installed Pi process from the Planetable working directory.
Saved provider/model defaults and the normal Keychain command resolve authentication.
The CLI receives an explicit refreshed guard, disables other extensions, tools,
MCP, context files and session saving, and performs a synthetic text request.
Exit status is zero; the Playground remains Ready in Decision mode. This does
not qualify an unrestricted Pi tool-execution session or other models.

| Validation | Result |
| --- | --- |
| Desktop lifecycle and API client suites | 392 passed, 1 skipped |
| Local API, clients and media suites | 68 passed |
| Focused final automatic selection/Stop/cancellation controls | 7 passed |
| Native controller harness | 156 checks passed, including automatic-mode preservation |
| Native window lifecycle suite | 14 passed |
| Pi connection and resource tests | 1 and 13 passed |
| Pi integration repository | 72 passed |
| Repository policy, changed Python Ruff checks and signed app build | Passed |

## Cache scope and limits

An actual worker replacement destroys decision and generation native caches.
Generation reload resets API principal scopes; caches are never transferred
between worker protocols. Immediately consecutive generation keeps its worker and
existing exact-prefix rules. This short fixture reports zero reused tokens; it
does not establish long-prefix numerical parity. Synthetic tests cover cancelled
admission, wrong-model preservation, busy decisions, failed load cleanup and Stop
fencing. Stop can terminate loading; HTTP disconnect during native startup is
checked again after readiness and does not itself interrupt that startup.

This Qwen3 artifact is text-only; image qualification is N/A here. Existing image
and MTP execution paths are unchanged, and their artifact qualification is not
expanded by this record. Throughput, loading latency, quality and memory peaks
are not measured. Switching protocols adds loading latency. Discovery now may
prepare a worker; a short client discovery deadline can require retrying after Ready.

Private aggregate receipt hashes, retained under ignored
`runtime/automatic-api-worker/`:

- Pre-install native lifecycle:
  `9b7008178d1dda338893f1ea54f85a47c55773f0f9ed195cc013e291b603974f`.
- Installed signed native lifecycle:
  `9bbc9902f274a0f7d6a4ba0751a02f16bcd5d9640ad7a2a386dce2486cca9281`.
- Installed Pi:
  `e3cd4eddb35e7aebf80fb554422f54d5058858c751af228ced8ecdede6da6220`.

## Retained failed checks

The initial worktree test invocation could not locate the native template helper;
an unsupported environment-variable override did not resolve it. Those failures
are invalid setup evidence. The matching existing helper was then selected through
PATH and the suites passed. The first window run retained the old requirement
that Decision mode hide the running model. Updating that expectation to the new
contract produced the passing rerun. A one-off install wrapper initially lacked
the scripts import path and made no installation change; the corrected wrapper
installed the verified signed candidate.
