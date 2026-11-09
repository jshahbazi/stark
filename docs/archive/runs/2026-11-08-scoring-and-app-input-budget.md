# Scoring and app-input budget follow-up

Date: 2026-11-08

## Source and scope

Baseline: `cba6459126dd93b6aea3540b391acda884af1e44`, PR #99.

Runtime-source manifest SHA-256: `14c0c3501f63930cb9936bf19041f1a0a6de752e2ff29ab2a83fabe1dc2b04fc` (1,075 files). It uses the source-file selection and binary-hash manifest encoding documented in the [preceding PR review](2026-11-08-context-budget-pr-review.md). Earlier validation records retain their own source and artifact bindings.

The scorer shares the enlarged request budget, so fresh native scoring now checks memory admission before state creation or prefill. Generic scoring counts state once, every retained prompt-chunk scratch pool, an additional inventory for the final partial/split pool, bounded command overhead and one final vocabulary/readback allowance. It reserves no output position: input may exactly fill context. Hybrid scoring uses the loaded family's requested-span profile. Resource rejection clears cached logits and preserves the worker for retry.

The private app command pipe now applies the same transient JSON headroom policy as HTTP. It checks before accumulating large frames and again before decoding. A rejected frame drains through its newline, allowing smaller commands and Stop to continue. Framing budgets, structural admission and fixed content-free errors remain in force.

## Validation

- Full converter suite: 6,699 passed, 96 skipped.
- Pinned-nightly CPU/grammar engine suite: 554 passed, 30 ignored; seven C++ string tests, two profile tests and three doctests passed.
- Release default and optimized scorer suites: 65 tests passed in each configuration. Release Clippy with warnings denied passed in both configurations.
- Generation worker: 132 passed, two ignored; shared resource-budget tests: five passed. Rebuilt generator and release Clippy passed.
- Focused desktop-input, HTTP-media and resource-policy tests: 34 passed. Decision lifecycle/retry tests: 13 passed.
- Scoped Rust/Python formatting, Python lint, repository policy and diff checks passed.

On physical Apple silicon, the rebuilt default scorer completed two requests each on a dense Qwen3 GGUF Q8 artifact and a retained Qwen3.5 2B Q4 artifact. Each prompt had one input token and selected context one; all replies reported no truncation. Both processes remained available for the repeated request. These checks exercise exact-fill scoring and fresh-state admission, with no probability, prompt, response or token-vector logging. They do not establish numerical parity or long-prompt qualification.

Release default scorer SHA-256: `50040a8985ec48b90ffc3da008ce8f3ef5432516a67f254c27e3613de72bf2db`.

Rebuilt generator SHA-256: `f5399db7e74b1aa057f96f67f7fbdc526b621d186d5163b3d032157617ca0f91`. Its undeclared-context synthetic ShortConv HTTP smoke was repeated on these sources: two 200 responses, unchanged undeclared capacity and the same worker/PID, with the finite allowance and EOS-only completion described in the preceding record.

The retained Qwen3.5 file hashes are recorded in the preceding review. The dense artifact manifest binds its config, tokenizer and converted weight file (three files), with SHA-256 `2f717d2234906f32104c25c5f438ddf61c3d4c5605b6bd13f24c84e53b736499`. It sorts filenames and hashes `filename + NUL + binary SHA-256(file) + LF` for each selected file.

## Limits

Memory admission is conservative headroom estimation, not a reservation or hard RSS ceiling. Scoring can reject inputs under pressure before reaching the declared context maximum. Parser admission protects decoding in the owning host; frontend serialization and transport remain bounded by their existing wire budget. This run does not qualify a filled context, a long-prompt scoring workload or a trained undeclared-context scoring artifact. The installed app was not updated.
