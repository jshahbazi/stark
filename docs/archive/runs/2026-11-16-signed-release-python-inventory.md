# Signed release Python inventory correction

## Source and failure

- Failed action: [Signed release run 43983245993](https://github.com/jshahbazi/stark/actions/runs/43983245993).
- Action source: `1f69d9ed9253d28e97fb28a7cd1031e2b9134579`.
- Fix base: `377459f4e533eaf762bceb5befd51fed580f9a4b`.
- The combined build, unsigned archive verification and credential import passed.
  Signing stopped with `python_library_missing`; cleanup passed and publishing
  was skipped.
- The distribution builder deliberately removes unused shared `libpython` and
  verifies that remaining native images do not depend on removed libraries.
  The signing inventory still required the removed library.

## Change and validation

Removed the shared-library presence requirement. Required workers and the
interpreter remain mandatory native images; every discovered runtime Mach-O
image still receives an individual signature and verification. The signing
fixture now matches the pruned runtime, including the nested app runtime.
Additional cases cover older bundles with shared `libpython` and rejection of
non-native required executables.

The revised fixture reproduced `python_library_missing` before the fix. After
the fix, the following checks passed locally with managed Python 3.13.13:

```sh
uv run --no-project --python 3.13.13 python -m unittest scripts/test_build_distribution.py scripts/test_build_menubar.py scripts/test_sign_distribution.py scripts/test_apple_credentials.py scripts/test_build_dmg.py scripts/test_release_version.py scripts/test_app_update.py
uv run --project tools/converter --locked --python 3.13.13 ruff check scripts/sign_distribution.py scripts/test_sign_distribution.py
uv run --project tools/converter --locked --python 3.13.13 ruff format --check scripts/sign_distribution.py scripts/test_sign_distribution.py
uv run --locked --python 3.12.13 --script scripts/ci/check_repository.py
git diff --check
```

The unit suite passed 169 tests. Signing and notarization coverage uses stubbed
Apple tools; this run does not establish live certificate signing, Apple
acceptance, release publication or installed-build status. The signed workflow
only runs on `main`; live confirmation requires the fix there and a new run.
