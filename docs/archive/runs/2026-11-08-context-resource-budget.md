# Context and Resource Budget Validation

Date: 2026-11-08

## Scope

This record captures focused validation of configured request budgets, model context checks, generated output beyond the former 4,096-token ceiling, and recovery after an oversized-context request. It does not qualify full-window generation quality or performance.

Source baseline: commit `78090aae66e47afc8cb5a470a8d784f3e3226ac8` with the working-tree patch hash `f2b1c88664aa589dd18df4cb2d66cdd7ac8917423af073034d26b1dbbb81fcb3` captured before this record was added. The binary hashes below identify the tested artifacts.

Final runtime-source manifest SHA-256: `01743baca6defda1b820da925321dfa690892ae5a45743a3b054a1390296e21d` (849 files). The manifest sorts repository-relative files under `crates/`, `apps/menu-bar/Sources/` and `tools/converter/stark_convert/`, plus the root Cargo manifest/lock and converter project manifest/lock. Each entry hashes `path + NUL + binary SHA-256(file) + LF`; the aggregate is SHA-256 over those entries. This excludes the validation record itself and ignored models/binaries.

## Rust tests

With Rust 1.98.1, `cargo test -p stark-generate -p stark-score --locked` passed: 120 generation tests, 25 scoring tests, and 25 scoring digest probe tests; two generation benchmarks were ignored because they require private evaluation data and exclusive GPU access. After the prepared image output-budget change, `cargo test -p stark-generate --lib --locked` passed again: 120 passed and 2 ignored.

The generation tests include a synthetic backend case that emits 4,097 tokens successfully, exercising generation beyond the former global 4,096-token cap. This checks the configured worker path and synthetic backend only; it makes no claim about model quality over a long generation.

## Native worker smoke

An existing local Qwen 3.5 2B Q4 model was used; no artifacts were downloaded. The worker reported a model context capacity of 262,144 tokens and a resource-budgeted readiness maximum of 262,143 generated tokens. A synthetic text request with a 16 MiB resource budget and explicit `max_new_tokens=4097` completed with 4,097 sampled tokens and `finish_reason=length`.

Worker binary SHA-256: `9df7432d852a1e70adfd71f8e1e3f75d86de11400de8ee579bd23c5f0c05c718`.

## HTTP to native worker recovery smoke

The Desktop bridge, local HTTP API, tokenizer, and native worker were run together with an isolated temporary data directory. The existing model was registered in memory for this process only. API context was set to 4,096 tokens and the environment resource budget was 64 MiB. The first request body contained 1,155,080 bytes of repeated synthetic text. The native tokenizer counted 330,011 input tokens, and the API returned HTTP 400 with `context_length_exceeded` after tokenization. The loaded worker PID remained unchanged. A following small request returned HTTP 200 with 16 prompt tokens, 8 completion tokens, and `finish_reason=stop`; the worker PID remained unchanged.

No prompt or completion content was recorded. The API used a private random key that was not printed. Model configuration and tokenizer hashes are included only to bind the local run:

- Worker binary SHA-256: `9df7432d852a1e70adfd71f8e1e3f75d86de11400de8ee579bd23c5f0c05c718`
- Model configuration SHA-256: `ca579bbcbd1b3e13bbedbcd3c17c2266ebdc8c357a07e8c7706580ddfb327c8e`
- Tokenizer artifact SHA-256: `87a7830d63fcf43bf241e3f8ceef92de4fb14ce7571b586c0e6597758b72732`

## Automatic output through HTTP

A later release worker build, SHA-256 `9318df2c953bcf78b489d57e75544189ec05ed2260470b0d4390e8c5ce596520`, was checked through the actual local HTTP API, Desktop bridge, native template helper and the same local model. No output-token limit was supplied. With an API context of 4,096 tokens, the native tokenizer reported 18 input tokens and the host selected 4,078 output tokens. HTTP returned 200 with one completion token, and the loaded worker was preserved. These are admission and recovery checks, not throughput measurements.

## Final regression checks

- Rust 1.98.1 worker, scoring, template and shared-budget tests passed. Template subprocess tests include text above 1 MiB, escaped transport, resource rejection and recovery. Formatting and Clippy with warnings denied passed.
- The full converter run reached 6,475 passes and 90 skips. Its only failure was an old unknown-context playground test assuming an implicit finite output allowance. That test now specifies 32 output tokens. The subsequent run of impacted regression files passed 971 tests with one skip, including the corrected test and both additional tool parsers.
- Native template controls also passed at the 64 MiB developer override: 114 tests, including hostile expansion, cancellation, fuel, sandbox and deadline controls. Smaller allocator boundary fixtures explicitly use the 1 MiB development budget.
- Swift controller checks passed 133 cases; window checks passed seven tests. Native worker startup isolation passed six cases. Python lint, formatting and repository policy checks passed.

The prepared-image protocol now permits a generation output budget different from the initial preparation budget when the actual image-plus-text input and the requested output fit the same prepared context. Synthetic image tests cover this path; this run did not qualify a real image artifact.

Buffered native generation now uses bounded text/progress callbacks as it generates. This can add decoding overhead; no matched performance comparison was performed.

## Limits

These checks establish request-size admission, tokenized context rejection, bounded output behavior, and worker recovery for the tested local configuration. They do not establish broad backend compatibility, full-context memory behavior, long-generation quality, or a production workload envelope.
