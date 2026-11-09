# 20261109-ADM-01/02/03: resource ceilings and installed-app recovery

## Identity and acceptance

Memory and lifecycle work, 2026-11-09, Codex. These shared-infrastructure attempts
target the installed `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl` artifact at
revision `0f94842e` and the existing converted `Qwen/Qwen3-0.6B-MLX-4bit`
artifact at revision `173234aa840d113125e9f2271100ddbaf16c9620` for small native
controls. They do not change tensor arithmetic or precision. Related records:
[initial correction and evidence](../archive/runs/2026-11-09-playground-resource-admission.md)
and [attempt index](README.md).

| Attempt | State, criterion and evidence scope | Disposition |
| --- | --- | --- |
| 20261109-ADM-01 | Completed; source audit, model-free checks and short native Qwen control passed. The 64 MiB content ceiling and generation RAM fractions were removed; Lily discounts resident allocations. Registered retrospectively from its existing immutable run record. | Adopted on branch at `b9e904fe`; not installed in that attempt. |
| 20261109-ADM-02 | Native controls passed; follow-up audit found Decision's quarter-RAM gate, stale Auto allowance retention and a cumulative progress-frame cutoff. Generation and scoring now use one current-headroom policy; Auto follows increased capacity; late progress continues beyond the former byte total; resource overflow still rejects cleanly. | Adopted on branch; packaging pending. |
| 20261109-ADM-03 | Passed signed-build installation and short native controls. Signing identity, generator/scorer features and pinned connector were preserved; installed bundle/overlay hashes verified; the actual Flash Next text artifact completed two Auto requests and a bounded follow-up on one worker. | Signed development build installed and opened; Flash Next Ready observed. |

## Audit boundary and remaining limits

The text Playground/API path has no separate fixed 512/4,096 output-token
ceiling, message-count ceiling or 16,384 context ceiling. Legacy startup
fallback values are replaced by current readiness metadata. Logical content,
token storage and transport allowances derive from available memory; configured
model context, protocol integer ranges and Metal buffer/index bounds remain
enforced. Template execution, JSON structure, image decoding and HTTP queueing
also retain their separately documented resource/security limits.

The remaining memory estimates are conservative admission evidence. Generic
finite prefix reuse and some non-Lily hybrid inventories may include retained
allocations when comparing with current RAM. Exact discounts for those paths
need allocation accounting beyond this patch; a fixed token ceiling is not
being introduced to hide that limitation. Lily's actual-resident scratch path
is the model-specific correction tested here. Current available memory is not
a reservation or a process RSS guarantee.

## Sources, artifacts and planned controls

Baseline for the follow-up is `b9e904fe`. Rust remains 1.98.1, converter Python
3.13.13 through locked uv, and nightly formatting uses `nightly-2026-10-01`.
Generation must build with `metal`; scoring builds separately with `optimized`
to prevent feature unification. The existing CLI runtime and pinned connector
must pass the builder's manifest verification. Fresh native build hashes and
the signed delivered hashes are separate identities.

The installed predecessor is signed stable 0.18.8 at `1e421519`. The new local
build uses the same Developer ID team and bundle identifier, a fresh numeric
build identifier and the development channel. A persistent bundle backup must
precede replacement. Models, Keychain items and settings remain outside the
bundle; replacing the running app ends its memory-only chat.

Before the Flash Next check, the old app/worker will be stopped by the
authorized installation workflow. Only one test model worker will run at a
time. Use the installed template and tokenizer, greedy seed 42, default MTP
when advertised, Auto output with the declared context, two short synthetic
requests and a subsequent bounded request for recovery. Retain prompts, output
text, token vectors and probabilities only in memory. Stop the owned test
worker on cancellation, timeout or failed protocol/memory controls. Publish
aggregate token counts, finish reasons, process-retention booleans and hashes.
Sampling and workload choices are functional controls, not quality references.
The validation guard samples at 10 Hz and stops the owned worker if system
available RAM falls below 2 GiB, additional swap exceeds 1 GiB or worker RSS
exceeds 120 GiB. These are validation stop conditions, not production caps.

Performance, full-window numerical quality and matched speed are not measured.
Detailed build, test, memory and installation results will be appended after
execution, including failed or deferred controls.

## Native candidate validation before packaging

- Rust 1.98.1: generation 135 passed and two benchmarks ignored; resource budget
  six passed; scoring and digest 32 each; GGUF protocol one; template two unit
  and 19 protocol tests. The added regressions exercise the shared memory
  boundary, increased Auto allowance and progress after the former byte cutoff.
- Locked release builds succeeded separately for `stark-generate` (`metal`),
  `stark-score` (`optimized`) and `stark-template`; all are arm64. Fixed
  invalid-argument checks and the template self-test passed.
- Pre-signing SHA-256: generator
  `76538bb3c3589010926cb1705cde9e458ff583fb7336791db19b32875cd0f124`;
  scorer `7497714b9fbedfa8c1cd4db081dc225ba4b018220e03d8c5de03032d66a749a5`;
  template `84400ce69a958528402e3a2d6c3dc2e2ba641a0504b9f9e62d0251c749c2cfe4`.
- Cargo lock SHA-256 is
  `85c73ea6bdb1ece383aca9502820095be4dba8818d6d7e348caca938507e18d6`;
  converter lock SHA-256 is
  `df7711726b1ed7e235ded21b9e6e1cbe130924be4d2c580d5d6bd9005248f8d5`.
- A Python chat/desktop/resource regression run during compilation produced
  420 passes, one skip and two fake-worker startup failures before the request
  under test. Both were the same startup assertion as the prior record, in
  different cases. Diagnostic isolated reruns are pending; no request-admission
  failure was observed in those cases.

## Completed installation and artifact-bound control

The follow-up implementation is `de60be71f1f1858bc3017fee5e040ef80f2b724e`.
Release Clippy with warnings denied, Rust formatting and repository policy checks
passed. The two Python cases each passed three normal isolated repetitions
(six passes); readiness took 128–257 ms and all fields validated. A controlled
2.2-second fake-worker delay reproduced both failures at their two-second
startup deadline. Compilation contention in the earlier run remains inferred,
because that run did not capture startup events. No product deadline was
changed to accommodate the fixture.

| Installed identity | Value |
| --- | --- |
| Channel / application version / build | Development / `0.0.0` / `20261109182050927057` |
| Source | Clean `de60be71f1f1858bc3017fee5e040ef80f2b724e` |
| Signature | Developer ID team `S5BT2PRN5B`; hardened runtime and secure timestamp; deep strict verification passed before and after installation |
| Manifest SHA-256 | `10a5c689c83df8b3238cf819b54c91afe742e5f1720891ba9437d4d3d4c9b407` |
| Signed generation worker SHA-256 | `9e6a9df5ba7e228bc8ef942fdc54b5576c60af0c0479f5d36d77f0338f7502aa` |
| Signed scoring worker SHA-256 | `07d84455a5ea470130057b7c5fd8df96521371c094a2f179cdd22fd516814bf3` |
| Signed template worker SHA-256 | `8b4e9e370c9bc2213ba178e437eb7c33dbb13af0d9d3b720893da1510b163233` |
| Overlay verification | All 459 signed overlay file hashes matched the installed files |

The previous 0.18.8 bundle was archived outside the destination and its ZIP
integrity check passed. The installer staged and verified the candidate before
atomic replacement in the user's Applications folder. The new app and embedded
backend launched from that installed path. Its selected Flash Next model was
started through the app and its Ready state was observed. The installed native
template self-test also passed. This is a local signed build, not a new
notarized stable release.

A later process inventory found the app and desktop backend running without a
generation worker. Reopening the app's main menu showed Stopped, with no error
displayed. Whether this followed user interaction or another event was not
established. A prior attempt to refresh the hidden menu failed in the computer
use capture service; reopening the menu restored observation. Source review
found no intentional idle/lock/sleep unload timer. The initial Ready observation
and short request checks therefore do not establish indefinite idle retention.
The user subsequently confirmed manually restarting during verification,
consistent with that stopped snapshot. A spontaneous idle exit was not
reproduced.

The rollback archive contains 110,437,094 bytes, SHA-256
`5d75f7ff3995c23e81b9148cffa12afd267a066008e17aa9b3d2255523c738f4`.
Host observations for this run were Apple M5 Max with 128 GiB RAM, macOS 26.7.2
and Xcode 26.4.1 build 17E202. Driver identity was not separately recorded.

### Exact Flash Next artifact

The model is `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl` at
`0f94842e3682b49acace4daeb4cf2e65a6f94f83`, derived from
`Qwen/Qwen3.8-Flash-Next` at `f5d08274bafd880402bd16f5e3e6c514136ec06c`.
The installed metadata and all 55 shard sizes match the existing guided
profile, totaling 106,905,155,192 shard bytes. Payload bytes were not freshly
rehashed. Stored weights retain Q4 GDN and Q8 attention/shared/head/embedding
groups, group-64 affine defaults, group-32 n-gram embeddings and Q4 draft
weights. Runtime precision remains BF16 activations, F32 reductions/logits/GDN
state and Q8 K/V.

| Artifact metadata | SHA-256 |
| --- | --- |
| `config.json` | `2a91155be52ebbb0f6dff20e61dcd8ae7484154a56b4afe067246c59da46f899` |
| `tokenizer.json` | `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3` |
| `chat_template.jinja` | `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041` |
| `native-source.json` | `7f5c685f35fc4c5d1509f6379feb1a965cb625f7a1ce76a0b5fe69983561dd30` |
| `model.safetensors.index.json` | `f7e35f06cb23dba09dea6197545111c092459f001715beead877c5f84959ccd4` |

### Native request and memory result

The signed candidate worker was exercised before replacement; its hash matches
the installed worker exactly. The bundled template and model tokenizer rendered
the private synthetic fixture. Three sequential requests used greedy seed 42
and two MTP drafts. Both Auto requests admitted the artifact's declared
262,144-token context, with 18 input tokens and 262,126 allowed output tokens.
Both completed on EOS with ten sampled tokens and nine emitted tokens. The
bounded 4,096-context, 16-output follow-up also completed on EOS with the same
counts. All three output hashes matched, and the same worker stayed alive.

At 10 Hz sampling, minimum system available memory was 24.118 GiB, peak worker
RSS was 74.515 GiB and additional swap was zero. No validation guard triggered.
RSS is not physical footprint or a hard allocation reservation; sampling can
miss brief peaks. The entire functional run took 15.053 seconds, including
startup and cleanup; it is not a throughput measurement.

Private aggregate receipt SHA-256:
`d5a85a96690151c65f829d1b9688ee1a616edb8d5a8a61d09faee934b339552c`.
Full-window execution and long-output quality remain untested: the admitted
window was not filled. The artifact itself declares 262,144 positions; the
backend does not impose a separate clamp at that value. Other artifact/model
paths retain the source-review limitations documented above.

## Follow-up history

- 2026-11-09: independent follow-up audit completed. The native-prefix and
  non-Lily conservative inventory findings remain source-review limitations;
  the three confirmed policy ceilings are being corrected before packaging.
