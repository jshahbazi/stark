# Final Lily projection-reuse adoption sources

Attempt: `20261116-LILY-REUSE-01`. Base:
`60bfa00a8f1759b7af73e224ff8786181a18d371`.
The [run record](../../runs/2026-11-16-lily-projection-reuse.md) and
[aggregate evidence](../../runs/data/2026-11-16-lily-projection-reuse.json)
bind every source and receipt hash, result and limit.

| Source | Execution |
| --- | --- |
| `integration.patch` | Exact four-file production kernel/test change relative to the base; default group-64 dispatch, original prototype shader math, expanded component controls and corrected diagnostic label |
| `qualify.py` | Original host-wrapper lifecycle attempt; resource admission prevented any worker from starting |
| `qualify-v2.py` | Passing direct-native image/cache lifecycle; first seeded-arm attempt stopped because its cancellation request omitted streaming |
| `qualify-v3.py` | Passing final/prototype/baseline comparisons, actual streamed cancellation/recovery, original natural-prompt replays and whole-book replay |
| `publish.py` | Integrity-checked aggregate-only publication and byte-identical source copying |

The harnesses expect their original ignored location,
`runtime/lily-projection-adoption/`, and the unchanged private identity/template
helpers from the earlier studies. This archive is an immutable record, not the
execution location. Prompt input uses private stdin with PTY echo disabled;
source texts, responses and token vectors stay in RAM. The public book is
fetched and hash-checked in RAM. No model or evaluation payload is redistributed.

Apply the patch independently to the base in a separate checkout. Use a unique
build target for that checkout, and Rust 1.98.1 with locked dependencies:

```sh
cargo +1.98.1 build --release --locked -p stark-generate
cargo +1.98.1 test --release --locked --manifest-path vendor/lily-core/Cargo.toml -- --test-threads=1
cargo +1.98.1 test --release --locked --manifest-path vendor/lily-core/Cargo.toml projection_reuse_matches_ordinary_bytes -- --ignored --nocapture --test-threads=1
cargo +1.98.1 test --release --locked --manifest-path vendor/lily-core/Cargo.toml decode_rows_gemv_matches_ordinary_bytes -- --ignored --test-threads=1
cargo +1.98.1 test --release --locked --manifest-path vendor/lily-core/Cargo.toml fused_gdn_projection_rows_match_decode_and_all_rollback_prefixes -- --ignored --test-threads=1
cargo +1.98.1 test --release --locked -p stark-generate -- --test-threads=1
cargo +nightly-2026-10-01 test --locked -p stark-engine --features cpu,metal -- --test-threads=1
```

The final implementation requires no experimental environment selector. The
sealed prototype uses its historical selector only in its own comparison
process. New kernels are original Apache-2.0 Stark wrappers around pinned Lily
arithmetic; upstream source/licensing remains intact. No llama.cpp code is
copied. Earlier source archives and performance decisions remain unchanged.
