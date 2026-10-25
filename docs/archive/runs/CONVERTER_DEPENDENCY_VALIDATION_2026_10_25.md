# Converter dependency refresh validation — 2026-10-25

This run qualifies a converter lockfile change on top of `33ff111c9e1e50e9318cde6a80a88daef679d874`. GitHub advisories published on 2026-10-23 (`GHSA-567q-5qvw-pf92`, `GHSA-9vhv-jxww-xfm5`, `GHSA-v4x8-3wr5-3wq8`, `GHSA-2g95-wg33-c5x2`, `GHSA-crv4-6mhj-j5jh` and `GHSA-jqc4-5cvw-7w85`) match filelock 4.1.0, which the converter lock carried as a transitive dependency. The OSV lockfile gate failed on `main` without any repository change.

## Change

`tools/converter/uv.lock` moves filelock from 4.1.0 to 4.2.1, the first patched release, with its registry hashes and sizes. No other locked package, constraint or converter source changed. `uv lock --check` reports the lock consistent with `pyproject.toml`. The converter build-system requirement remains setuptools 83.0.0; it is not part of the locked runtime.

The same refresh moves the main tools lock to filelock 4.2.1 and setuptools 84.1.0, and the vLLM benchmark lock to setuptools 84.1.0 for `GHSA-xmj3-9gf7-h52w`. Those projects do not ship in the release.

## Automated checks

- The complete locked converter suite passed: 4,105 passed and 88 skipped in 102.74 seconds, using `HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false uv run --project tools/converter --locked --python 3.13.13 pytest tools/converter/tests -q`.
- Ruff lint and format checks passed for all 80 converter Python files.
- The locked environment imported filelock 4.2.1 under Python 3.13.13.
- Repository policy checks and their 52 self-tests passed with `scripts/ci/check_repository.py` under Python 3.12.13.

No model was converted or loaded, no native worker was rebuilt, and no distribution was built or released. Native loading, conversion parity and release bundle verification for this lock remain unqualified until the next release build runs its own checks.

## Runtime and evidence identity

The run used macOS 26.7.1 on arm64 hardware model `Mac17,7`, uv 0.12.24 and the converter's Python 3.13.13.

| Artifact | SHA-256 |
| --- | --- |
| Previous `tools/converter/uv.lock` | `a7836107110aadcb25688772547d518b4067d4856df76724bd07bc470e5cce9c` |
| Candidate `tools/converter/uv.lock` | `df7711726b1ed7e235ded21b9e6e1cbe130924be4d2c580d5d6bd9005248f8d5` |
