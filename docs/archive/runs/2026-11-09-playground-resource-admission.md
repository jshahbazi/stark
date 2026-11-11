# Playground resource admission correction

Date: 2026-11-09

## Source and scope

Baseline: `1e421519`.

Runtime source manifest SHA-256: `a562280a556f1cf45200f22101adafebf5173fef9d4e73f9611f728e805a8f7d` (1,285 tracked paths). The manifest covers `crates/`, `vendor/`, `apps/menu-bar/Sources/`, `tools/converter/stark_convert/`, the root Cargo manifest/lock and the converter project manifest/lock. Entries sort repository-relative paths and hash `path + NUL + binary SHA-256(content) + LF`; symbolic-link content is its literal target. This excludes this record, tests and ignored runtime artifacts.

The reported error occurred in the Playground with Qwen 3.8 Flash Next Lily. Read-only process metadata identified installed app version 0.18.8 and the `qwen38-flash-next-lily-q4-xl-0f94842e` artifact. No original prompt, response or token vector was read or retained. The original request was not replayed.

Source inspection found conflicting automatic-planning and generation memory policies, a fixed 64 MiB logical-content ceiling in Rust/Python/Swift, and Lily estimates charging already-resident allocations against current remaining RAM. The patch removes the fixed content ceiling, uses one available-memory allowance for generation planning and execution, and derives Lily's incremental requirement from actual resident scratch geometry. Model-loading profiles remain unchanged.

Generation's remaining-memory allowance is current available RAM minus a 256 MiB transient serialization/host reserve. Lily counts cache and decode-scratch replacement peaks, prefill growth, request workspace and up to two new snapshots. The original cold inventory and Metal individual-buffer checks remain available. Other backends may still include retained allocations conservatively. The process-start content allowance remains proportional to available RAM, with a 16 MiB floor and checked allocation-address-space bounds.

## Model-free validation

- Rust 1.98.1 generation tests: 133 passed, two private-data/GPU benchmarks ignored. Scoring and scoring-digest suites: 32 passed each; GGUF protocol integration: one passed.
- Metal-disabled generation suite: 91 passed. Existing dead-code warnings remain in that configuration.
- Lily request-memory tests: four passed, including retained small requests after larger prefills and context-growth overlap.
- Shared resource-budget tests: five passed. Native template tests: two unit tests and 19 protocol tests passed.
- Swift controller harness: 139 passed, including runtime allowances above 64 MiB, inconsistent metadata and integer overflow.
- Native generation-process startup/isolation checks: six passed with `STARK_REQUEST_BUDGET_MIB=128`.
- Release generation build, Clippy with warnings denied, Rust formatting, changed Python lint/format and repository policy checks passed.

The first full converter run had 6,797 passes, 96 skips and three template-expansion boundary failures. Those tests assumed the former bounded automatic allowance; they now explicitly select a 1 MiB native test budget. A second full run had 6,799 passes, 96 skips and one fake-worker startup assertion failure before the request under test. The fake worker has a two-second readiness timeout and performs no resource admission. Concurrent compilation is a plausible cause; the failure did not capture startup timing, so that cause is unconfirmed.

With compilation complete and no further product changes, the isolated desktop, template-runtime and resource-budget regression run passed: 450 passed, one skipped. This includes the startup case and corrected expansion boundaries.

## Native generation smoke

The release worker was exercised through the Python worker client with an existing converted Qwen 3 0.6B 4-bit artifact and an explicit 128 MiB content budget. Two sequential requests used automatic planning for a 512-token context. Both counted 15 input tokens, planned 497 output tokens and completed with 11 output tokens and `finish_reason=eos`. The same worker remained loaded. Only aggregate results were printed.

- Worker SHA-256: `8b44d80d8f8debc5bbc711d15a6993e99fd5a97e7bab8abf6b33f564111786de`
- Checkpoint: `Qwen/Qwen3-0.6B-MLX-4bit`, revision `173234aa840d113125e9f2271100ddbaf16c9620`
- Base model: `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca`
- Configuration SHA-256: `60d4fb79e35aa0a6a72e7d71e53cd701df9f446930aa37d7e85390213194388b`
- Tokenizer SHA-256: `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`

This validates native short-request admission, budgets above the former ceiling and worker retention for that artifact. Flash Next's incremental estimate has static review and synthetic geometry coverage; this run did not load another Flash Next instance or qualify its full-window memory, numerical behavior or performance. The 256 MiB reserve is an admission margin, not a hard RSS guarantee. The installed app was not replaced or restarted.
