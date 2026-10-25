# Stark

- This is a standalone Uzu-derived Rust/Metal engine. Keep host application integration in separate repositories.
- Preserve upstream copyright and licenses. Record source revisions and patch provenance.
- Preserve source-bound validation results and upstream provenance. Record new validation runs separately when tooling changes.
- Do not log prompts, responses, token vectors, decision probabilities, credentials, or exception dumps. Keep native validation artifacts private and publish aggregate results and hashes only.
- Serialize complete decisions against a single loaded model. Cache only identical immediately consecutive token sequences with the same context size. Clear on failures and restarts; never persist the cache.
- Models, binaries, generated distributions, private evaluation data, and runtime artifacts stay ignored.
- Keep dependencies locked. Use Rust 1.98.1 for scoring builds and `uv` for Python tools.
- Write documentation in Markdown without emojis. Do not use shell heredocs.
- Use Conventional Commits, `git ai-commit codex`, and author `Codex <noreply@github.com>`.
- Push authorized work to its branch. Do not open a PR unless explicitly requested.
- Do not enable inherited upstream cloud/signing/release infrastructure. New distribution infrastructure requires a separate scoped implementation.
- Toolchains, local checks and test layout are in `CONTRIBUTING.md`; the documentation index is `docs/README.md`. Reference pages under `docs/` describe current behaviour and change with the code; dated validation records go in the pull request or under `docs/archive/runs/`, benchmarks under `docs/benchmarks/`.
