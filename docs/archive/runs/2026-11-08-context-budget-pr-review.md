# Context-budget PR review

Date: 2026-11-08

## Source and scope

Baseline: `fc7e83fa2a8d689a328a33487318a8c8fe3e3735`, PR #99.

Runtime-source manifest SHA-256: `b68b4b660963c19b9067c07b727efa2bfec63e39dd14f83e59b787f866fe850f` (1,075 files). The manifest sorts tracked and nonignored new files under `crates/`, `apps/menu-bar/Sources/`, `tools/converter/stark_convert/` and `vendor/lily-core/`, plus the root Cargo manifest/lock and converter project manifest/lock. Each entry is `path + NUL + binary SHA-256(file) + LF`; the aggregate is SHA-256 over those entries. Generated artifacts and this record are excluded.

Independent reviews found two gaps in the initial PR: finite-context generation could grow its cache beyond available RAM, and mixed-Unicode JSON could exceed the HTTP parser's three-times-wire-size memory estimate. The correction also makes hybrid inventory/index-limit failures recoverable so a smaller request can use the same worker.

Finite Auto now uses advertised native memory planning with the declared context as its upper bound. Explicit text requests also undergo allocation admission. Generic sparse finite contexts count their requested token span; dense and undeclared contexts count their full selected allocation. The separate GLM, Qwen, Flint and retained Lily paths inventory their actual cache geometry, growth overlap and workspace policies. EXL3 keeps its existing full-context device phase admission. Inference precision, kernels and template execution protections are unchanged.

Large HTTP bodies reserve twelve times their raw size as transient parsing headroom. Admission accounts for concurrent readers atomically and releases that reservation after parsing. Raw-body reservations remain held through completion.

## Checks

- Full converter suite: 6,693 passed, 96 skipped.
- Pinned-nightly CPU/grammar engine suite: 553 passed, 30 ignored; seven C++ string tests, two profile tests and three doctests passed.
- Rust generation/scoring/resource tests: 198 passed, two ignored, including hybrid resource-error recovery.
- Retained Lily inventory: two focused pure tests passed.
- Swift controller: 134 checks passed.
- Release generator build, release Clippy with warnings denied, scoped Rust formatting, Python lint/format, repository policy and diff checks passed.

## Native execution

Release generator SHA-256: `b5d8f7189b9f84d45ddab900be58f9136220d0c047af32c2ce03c7c4371f94f4`.

On physical Apple silicon with Rust 1.98.1 and Native Metal:

- The retained Qwen3.5 2B Q4 artifact accepted two host Auto requests with no context or output settings. Both plans used the declared 262,144-token context, with 17 input tokens and 262,127 output slots. Each request stopped after one completion token and retained the same worker. Model discovery continued to report the declared capacity.
- The private synthetic undeclared-context ShortConv fixture from the earlier validation record accepted two HTTP requests with only messages. Both returned 200 in the same worker/PID. Six input tokens received a finite allowance of 16,777,222 total tokens, with 16,777,216 output slots; both stopped on EOS without completion tokens. Discovery kept the context undeclared.

The selected allowances were not filled. No prompt, response or token vector was recorded.

Retained Qwen3.5 artifact SHA-256 values:

| File | SHA-256 |
| --- | --- |
| Config | `ca579bbcbd1b3e13bbedbcd3c17c2266ebdc8c357a07e8c7706580ddfb327c8e` |
| Safetensors | `a9e01fbf845a1d2af4727f4394b244815abd26b25d186bc49730249fb16c840d` |
| Tokenizer | `87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4` |

## Limits

Admission uses conservative estimates and current headroom, not a memory reservation or hard RSS ceiling. Finite retained states can be charged conservatively again and rejected under pressure. Other applications can consume memory after admission. Existing cancellation and request deadlines remain active. This run does not qualify a filled model window, a trained undeclared-context artifact, unknown-context Dense MTP, or full-window GLM/Lily performance. Earlier source-bound records remain separate evidence. The installed app was not updated.
