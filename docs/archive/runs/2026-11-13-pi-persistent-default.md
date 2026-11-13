# 20261113-PIGUI-05: persistent manual Pi connection

Registered retrospectively and executed on 2026-11-13, owner Codex. This completed
compatibility attempt passed and was adopted on the feature branch with a signed
local installation. Governing request: normal command-line Pi must remain
connected after closing the GUI-launched process. Predecessor:
[PIGUI-01–04](../../model-experiments/2026-11-13-pi-connections.md);
implementation [PR #155](https://github.com/jshahbazi/stark/pull/155).

The isolated GUI launcher did not configure subsequent ordinary `pi` launches.
Connections now also saves the `stark-local` provider, required guard path and
default provider/model in Pi's standard user configuration. Other providers,
extensions and unrelated settings are preserved. The key remains in macOS
Keychain; Pi's supported request-time command resolver obtains it directly.
Existing stored Stark authentication is detected as a conflict rather than
silently overriding that resolver. This continues to give Pi the master key;
managed identity-bound grants are a separate milestone.

## Delivered identities

- Stark source: `5b069261ce551032f12943ad6d900845d2c601a6`, clean at build time.
- Integration: `e01f7735ac46c741d2e94435871d42e6d3240609`, eight public
  source/license files verified against the owning Git blobs.
- Installed signed app: `/Applications/Stark.app`, build `20261113171433476153`.
- Embedded manifest SHA-256:
  `6adeed98225f2bc54722da37f11493e2e73ecd3707110f475581a30fa67d73f1`.
- All 21 Swift sources and 19 resources matched their recorded hashes. Candidate
  and installed manifests matched; deep/strict signature verification passed.

The installed app's actual generation-worker arguments selected
`runtime/desktop-agent-validation/qwen3-4b-4bit` in the original Stark checkout,
with prefix caching enabled. The four files at that path were rehashed:

| Artifact field | Identity |
| --- | --- |
| Source | `mlx-community/Qwen3-4B-4bit`, revision `4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25` |
| Weights | `7c2af9a45e5b0d196f06a5cbfd6fc5cc95adf29df79885d0540f2206fb3d40d8` |
| Config | `5b505594bf00ef9682b373c6a292b88f94f1e1ceab449784c12accb835806c7f` |
| Tokenizer | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| Conversion metadata | `bfa4c305b1bf64b0fc213552dc1f729ace85ef76a521e803e8ca2d34916aac1c` |

Weights/config/tokenizer match the prior recorded artifact; the conversion hash
is recorded separately here. Existing Rust/Metal workers were reused as explicit
build inputs; their signed identities are in the embedded manifest. No backend,
kernel, precision or state-restoration path was added. Activation/accumulation/KV
policy, hardware inventory, peak headroom and template bytes were not independently
requalified in this connection-only run. The source is text-only; image checks are
N/A. Numerical equivalence and reuse counters were not measured. Earlier cache
and precision results retain their original identities.

## Validation

All 70 client tests passed with zero skipped. The latest native controller/window
run passed 15 checks. Packaging/distribution/build tests passed 84 checks, with a
final 38-check package/distribution rerun after the version-gate pin changed.
Repository policy, Ruff check/format and whitespace checks passed.

Independent review found and corrected stored-auth precedence, failed lock-marker
cleanup and the missing version gate on ordinary Pi startup. The reviewer then
reported no remaining source blockers. Recovery tests cover an interrupted partial
commit, concurrent-edit refusal and reclaiming only marked locks of exited Stark
helpers. Existing/foreign/live/unmarked locks are preserved. The narrow crash gap
between lock-directory and ownership-marker creation requires manual inspection.
The first persistence candidate built successfully but was superseded during
source review before native qualification because normal startup lacked the
version gate. This was a review finding, not an observed model failure. The final
gate rejects unsupported Pi versions before provider registration.

The installed app's Advanced Connections setup completed and reported that normal
`pi` would use Stark. The model reached Ready. A separate private harness then
started the actual installed Pi 1.5.2 twice, with its standard user profile and
saved defaults. Both fresh processes selected `stark-local/qwen3-4b-4bit` and
completed authenticated native text generation through the real Keychain command.
Neither process inherited `STARK_API_KEY` or `PI_CODING_AGENT_DIR`, and neither
received provider, model or API-key flags. Credentials and client streams were
processed only in memory; no credential value or transcript was recorded.

The acceptance controls were fixed in the private harness before execution:
two sequential fresh client processes, exact saved provider/model, no inherited
credential/profile override, a nonempty successful assistant turn on each, a
90-second deadline per launch and bounded in-memory client streams. The backend
advertised context 40,960; the saved provider caps output at 1,024. Input counts,
seed/sampling values and actual prefix reuse were not recorded. This is not a
timing or task-quality experiment. The one native attempt passed; there were no
failed/discarded native arms. Temporary workspaces and owned client processes
were cleaned; the user's running Stark model was left available.

Aggregate [receipt](data/2026-11-13-pi-persistent-default.json) SHA-256:
`f0263ab8d38792139e0ac6d69804689e477a9b50af6e2cfb695fff1bceea3325`.
Private harness SHA-256:
`cc970b235adf79d900ce1451205ea1cf21ab4e96069f224b531d0737a7a0688c`.

This is bounded default-selection and authenticated-restart evidence. The live
harness used RPC, offline mode, no saved session and disabled tools/context
resources; it did not automate Terminal's TUI. Ordinary interactive Pi retains
its existing resources, session, retry and compaction policy. The earlier task
and lifecycle matrices retain their original build identities. No engine,
precision, numerical-parity or performance change is claimed.
Performance, peak memory, cancellation, context exhaustion, model changes and
cross-session cache isolation were not measured here. The remaining acceptance
is the user's ordinary interactive CLI workflow. This attempt does not establish
managed authentication or qualify other artifacts or Pi versions.
