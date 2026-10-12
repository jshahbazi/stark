# Converter hardening validation and review

Date: 2026-10-12. Branch: `codex/converter-hardening`, based on `origin/main` at `6e19d962`. Converter version: 0.9.1. This is a tool hardening packet; adapter coverage, tensor transformations, the public native ledger and prior performance evidence are unchanged.

## Local validation

| Check | Result |
| --- | --- |
| Complete offline model-free converter suite | 3,679 passed, 87 skipped in 36.72 seconds |
| Ruff lint | Passed |
| Ruff format | 62 files checked, passed |
| Converter lock consistency | Passed; 27 locked packages |
| Repository policy checks | Passed |
| Whitespace validation | Passed |
| Clean sdist-to-wheel build and isolated installation | Passed |
| Explicit wheel license and complete packaged provenance | Passed |
| Exact checkout/installed source and lock identity | Passed |
| Installed CLI version/help/capability probe | Passed |
| Local sharded catalog and anonymous generic conversion, offline | Passed |
| Installed two-repeat benchmark with exact output parity | Passed |
| Python byte compilation, including capsule cache files | Passed |
| Missing local policy and incomplete download failures | Passed; no output publication |

The 87 skips are opt-in private metadata parity checks whose `STARK_CONVERT_REFERENCE_ARTIFACTS` inputs are absent. Frozen numerical fixtures and public model-free contracts run normally. No new real model weights were downloaded. The clean-install script removes its tiny synthetic model fixtures, environments and distributions on exit.

Source and dependency identities for this validation:

```text
converter_sources_sha256 63827e103f1f1da61d3d0a24df1c32191f624e38d65e82bd800b880fd317bb38
converter_lock_sha256    d064b7fb928392342ceae7f40da0cfcb67ea146e1289f589ea3521232e13d9c8
reference revision       831accb8252a6b6606c378048caeb44178f1a050
engine upstream revision 8d01581474760bd1fc7f4a547ff1d02f7dada437
```

## Independent review

Two independent read-only agents reviewed packaging/provenance and CLI/benchmark behavior. Their findings were integrated before closeout:

- Packaging previously depended on checkout-only lock/build inputs. The wheel now embeds and verifies those inputs; the actual built artifact is compared with the checkout.
- Editable installations intentionally remain checkout-bound, with that boundary documented.
- Capsule Python files can acquire interpreter caches during installation. Both package inventories now ignore caches consistently while rejecting unaccounted non-cache files; the installed test explicitly compiles bytecode and verifies its presence.
- The fresh-process launcher changed from `uv` to the current installed Python. Schema 3 identifies this protocol, and all historical timing packets remain separate.
- Failed final report writes can retain a previous running report. A targeted test verifies an unsuccessful command and a complete retained report; documentation explicitly excludes it from passing evidence.
- The replacement plan now distinguishes implemented local packaging from deferred publication and engine compatibility policy.

Review closeout found no remaining blockers in this packet. Tests additionally reject missing, boolean, negative, nonfinite and overflowing benchmark metrics; completed measurements survive later errors/interruption, and report replacement preserves the previous complete JSON on storage failure.

## Limits

Validation was local on macOS/Apple silicon with managed Python 3.13.13. The existing Linux repository CI job now runs the same installed-package check; hosted results are recorded by CI separately. This packet grants no new native loading, inference, classification quality or Lalamo performance qualification. Further model support, total-process resource limits, crash durability guarantees and publication policy remain separate work.
