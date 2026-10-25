# CLI chat validation — 2026-10-21

New validation for `codex/cli-chat`; prior engine qualification receipts are unchanged.

- Converter suite: 3,804 passed, 88 skipped (`uv run --project tools/converter --locked pytest -c tools/converter/pyproject.toml tools/converter/tests -q`).
- Launcher contracts: 15 passed. Distribution builder contracts: 14 passed. Ruff and `git diff --check` passed.
- Local source-launcher smoke: two generation replies, reset acknowledged, exit status zero. Transcript content was not persisted. This checks interactive operation, not answer quality or performance.
- Model: converted Qwen/Qwen3-0.6B BF16, source revision `c1899de289a04d12100db370d81485cdf75e47ca`; conversion-receipt weights SHA-256 `87c3344a80b7376392cc5c44d5369680b68936e6930bb8ac476ca65a136f64c7`.
- Existing native worker SHA-256: `d28aee32a56a49f0386c2d61831975c6a179fc93eaf262b77c49dbd727a44c0d`. The smoke used this existing release binary; no claim is made that it was rebuilt from this change.
- Full source bundle installation was attempted and blocked during Cargo resolution: locked `rustls-webpki 0.103.16` was unavailable from the configured registry. `Cargo.lock` and the installed bundle were unchanged. Relocated-bundle chat import/help checks were added but could not run against a newly built bundle.

The checkout launcher is available immediately with `./scripts/stark chat --model-path /absolute/path/to/converted/model`. The installed `stark` command gains chat after a successful updated bundle installation.
