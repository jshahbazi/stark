# Distribution size reduction validation

Date: 2026-11-16. Source baseline: `3d17fd68`; branch:
`codex/slim-distribution`. This record describes the packaging changes committed
with it. Tests ran on Apple silicon, macOS 26.7.2, uv 0.12.24 and Python 3.13.13.

## Changes

- The DMG retains real top-level and installer-contained app directories for
  existing updaters and App Translocation. Corresponding files share HFS+ hard
  links. Mounted-image validation checks both inventories and shared file
  identities before publication; new receipts require the same receiving check.
- Both distribution and local app builders remove Python headers, IDLE,
  Tkinter/turtle, Tcl/Tk libraries and data, and the unused shared Python library.
  Remaining Mach-O dependencies are inspected after pruning and desktop package
  installation. A dependency on a removed library fails the build.
- Derived app runtime manifests retain the parent manifest hash and record
  pruned paths. Existing licensing notices remain included.

## Measurements

These measurements use an isolated copy of the installed Stark 0.21.5 snapshot,
not a newly built or notarized release. The installed app was not changed.

| Measurement | Before | After |
| --- | ---: | ---: |
| Python tree, sum of file sizes | 120,799,334 bytes | 91,931,305 bytes |
| Payload-only compressed HFS+/UDZO DMG | 240,390,191 bytes | 108,442,021 bytes |

Python pruning removed 28,868,029 bytes (27.5 MiB). The DMG comparison used two
independent app copies before, then the pruned app with shared hard links after.
The compressed image was 54.9% smaller. Installer executable/artwork overhead,
fresh compilation, release signing and notarization were outside this size
comparison; the published asset size must be measured separately.

Snapshot identities (SHA-256):

- App executable: `0b4a63dad1661cbe63b69f5d14ea1a83a111b56dd4db4ddc7962548e045d642f`.
- Python interpreter: `cb2526474f06a5778c48e836e133d843c2e4f5d09db8c59e4590e070c082f44b`.

## Verification

- Repository policy checks, scoped Ruff checks/formatting and scoped
  `git diff --check` passed.
- 211 tests passed across `test_build_distribution`, `test_build_dmg`,
  `test_build_menubar`, `test_app_update`, `test_install_app`,
  `test_sign_distribution`, `test_install` and `test_stark_cli`, using
  `uv run --project tools/converter --locked --python 3.13.13 python -m unittest`.
- A follow-up receiving-verification check passed with both new shared-payload
  receipts and legacy receipts for independent copies.
- The full converter suite passed: 7,820 passed and 95 skipped in 335.57 seconds,
  using `HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false uv run --project
  tools/converter --locked --python 3.13.13 pytest tools/converter/tests -q`.
- The native DMG test signs fixture bundles ad hoc, creates and mounts a real
  compressed HFS+ image, verifies shared file identities and both code seals,
  copies both app locations through Foundation, unmounts the image, and verifies
  the independent staged copies and their signatures. The existing native
  handoff test also quits a running fixture app and activates its replacement.
- On the isolated pruned snapshot, real native dependency inspection, core
  package imports, desktop/text/image imports, native template rendering, and
  converter/chat/desktop `--help` entrypoints passed from outside the checkout.

No models were loaded. These checks do not establish Developer ID notarization
or Gatekeeper App Translocation behavior for a fresh release. Release construction
continues to require the existing signature, ticket and Gatekeeper checks.
