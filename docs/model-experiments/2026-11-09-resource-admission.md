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
| 20261109-ADM-03 | Planned installed-build/lifecycle check. Acceptance: preserve signing identity, installed generator/scorer feature sets and pinned connector; verify the installed bundle and embedded hashes; complete short synthetic Auto requests on the actual Flash Next text artifact with aggregate-only results and same-worker recovery. | Build/install requested; pending. |

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

## Follow-up history

- 2026-11-09: independent follow-up audit completed. The native-prefix and
  non-Lily conservative inventory findings remain source-review limitations;
  the three confirmed policy ceilings are being corrected before packaging.
