# Chat terminator and desktop metrics validation

Validated on 2026-10-25 with Apple M5 Max, arm64 macOS 26.7.1 and Rust 1.98.1. This run covers the changes accompanying this document on base revision `7f56e28b88fe4baec6998728a6447944696dadd2`. Earlier qualification remains bound to its original source and tooling.

## Cause and changes

The installed JevK5 configuration declared generation stop ID `248044` (`<|endoftext|>`) and chat codec EOS `<|im_end|>` (tokenizer ID `248046`). The native worker used only the generation stop list. It now validates and merges both declarations at model load, including for existing installations. Terminal tokens are excluded before decoding and streaming, preserving other special tokens required by structured model output.

The desktop bridge previously forwarded completion rates only with built-in chat replies. Local API requests now emit a separate content-free metrics event, consumed by Swift independently of a pending chat message. Successful built-in completions use the same event. Existing rate definitions remain unchanged; an unmeasurable duration still yields no rate.

## Results

- Native JevK5: six exact-output checks, three streaming and three non-streaming, with greedy sampling, a 1024-token context and a 128-token output limit. No caller stop sequence or stop ID was supplied.
- Six of six checks stopped on the chat EOS, excluded the terminator, matched the expected exact output, maintained streaming/final consistency where streaming was enabled, and returned finite positive input and output rates.
- Worker readiness advertised both model stop IDs.
- Rust worker tests without Metal: 26 passed, including stop merging, deduplication, invalid metadata, terminal exclusion, streaming and timing checks.
- Desktop tests: 40 passed, including API streaming/non-streaming rates, content-free metrics events, and no completion metrics after failure, disconnect or stale generation.
- Native Metal worker built with locked dependencies using `cargo build --locked -p stark-generate`.
- All Swift sources passed `xcrun swiftc -parse-as-library -typecheck apps/menu-bar/Sources/*.swift`.
- Ruff passed for the changed Python files; `git diff --check` passed.

The native check consumed request and response content only in memory and private pipes. Its local harness is retained under ignored `artifacts/validation/2026-10-25-terminal-metrics/`; only aggregates and hashes are published here.

## Artifact hashes

| Artifact | SHA-256 |
| --- | --- |
| Native debug worker | `c1b08db33833c3e370c2cf3420e8e7875b17427d1d4bc509a8311d69cd34094f` |
| Private validation harness | `f3a4685e1c605de9a05f4ff8688d9b9ece1925f7adef12671cd82f1ec9a7e2a5` |
| Installed model configuration | `9e8e47e82f80cc37a0b318115df950f183a5e9823e57e36829647113201d357a` |
| Installed model tokenizer | `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523` |
| Stop resolution source | `2ff17dd325758bf129d7d95597bbe3cae76b52f3e6ddf316b4504d90f98698d5` |
| Native backend source | `d9eca31071d433490b8f96453141045157cad40c3d28d408e4c554675dde4dae` |
| Swift app source | `6bcd7a9f296acc2b9f887af9de628235d44e5a17fda75b6dfbd7809555b6cac1` |
| Desktop bridge source | `f0e1148fba8a0e850f1bc7e9120b0b8269d4bb01ba546784163a18da570f73ba` |

## Limits

This is a bounded functional check, not broader model or performance qualification. The native checks exercised the new worker directly; desktop rate forwarding used synthetic worker fixtures. Swift typechecking verifies compilation, not live visual rendering. The installed application was not rebuilt or replaced during this validation and needs a build containing the worker, Python bridge and Swift changes to expose both fixes.
