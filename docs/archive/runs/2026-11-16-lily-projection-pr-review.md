# Projection reuse: independent review and clean PR preparation

Date: 2026-11-16. Attempt: `20261116-LILY-REUSE-02`.
State: completed; approved for PR publication. This is a source-port check, not a
new model or performance comparison.

## Scope and fixed checks

The user requested a distinct double check before opening a PR. The original
implementation remains on `codex/flashnext-lily-mtp-default` at `44c8940c`.
That branch also contains older app/catalog changes, so the PR is prepared on
`codex/lily-projection-reuse`, based on main
`ab7bc60e8339df133cba6df667bbb4b6822dcfd7`.

The clean branch takes only the study's net change from baseline `cc8e47a8`
through `44c8940c`: verifier projection reuse, documentation and the complete
source-bound benchmark history. The unrelated catalog feature and
already-merged image-retry implementation are excluded. Two desktop tests and
general reference-page edits from the retired MTP policy study are also omitted;
their historical evidence remains on the original branch and in archived run
records. No retired automatic one-draft implementation is introduced.

Before opening the PR, require:

- An independent source review with no unresolved correctness blocker.
- Exact Rust/Metal execution-source and dependency-lock equality to the
  previously validated implementation; preserve all 232 existing data/archive
  artifacts and verify the final aggregate/source/receipt hashes.
- A fresh locked Rust 1.98.1 generator build and the physical 240-case
  projection test in separate targets owned by the clean checkout.
- Repository policy, formatting and whitespace checks. Desktop behavior is
  outside this final diff, so its two proposed test additions are excluded.

## Review and evidence scope

Independent review covers dispatch limits, Q4's 16-lane reduction, Q8's SIMD32
reduction and expression order, math mode, buffer offsets, one writer per
output, fallback layouts and verifier-only reachability. Snapshot contents,
prefix keys and restoration remain unchanged. Two drafts remain the enabled
recommendation; omitted native selectors remain off.

The [original final-build validation](2026-11-16-lily-projection-reuse.md)
retains its own source/build hashes, 56 native checks and complete prompt
replays. Those model runs are not repeated or relabeled as a new PR-build
execution. The historical M5/pinned-Lily speed result stays 5.40% decode and
2.39% less request time for the whole-book case. Host-wrapper memory admission
was separately deferred; no host memory guard was changed.

## Results

An independent code reviewer found no correctness blocker. A separate scope
and evidence audit confirmed:

- All 1,213 Rust/Metal, manifest, lockfile and toolchain files match original
  `44c8940c` byte-for-byte.
- All 232 prior files under `docs/benchmarks/data` and
  `docs/archive/experiments` match base `60bfa00a`; including the older run-data
  directory extends that comparison to 234 unchanged files.
- The published aggregate SHA, five archived source hashes, nine private
  receipt hashes and 18 + 25 + 5 + 8 = 56 native check counts are consistent.

The original checkout's physical projection control was rerun successfully.
The clean PR checkout then built its own fresh locked release generator and
physical component test, using separate checkout-owned targets. The latter
passes all 240 cases / 10,436,400 bytes with zero unequal bytes. Repository
policy, changed Rust formatting and whitespace checks pass. No kernel fix was
needed. Build hashes are recorded below; these are new build identities,
not replacements for the original model-validation identities.

| Fresh PR build | SHA-256 |
| --- | --- |
| Release generator | `2fcd40ec135306979965685c688ecda12f5c4cf7239c910c8b01f4e1812ab8ac` |
| Physical core test executable | `9f5398bc7b26931c2f0f2545fc237b40735b737cc3d6c53fdeda846261977a3c` |

## Benchmark CLI privacy correction

Review found that the bundled baseline CLI copied its input corpus into the
output directory despite its aggregate-only promise. This was a publication
blocker for that tool. The current CLI now captures the corpus once in RAM
and passes it through an optional `corpus_bytes` argument to the paired
harness. All matrix cells receive the same bytes even if the source file is
changed or removed; existing callers retain their ordinary file-input behavior.
Runtime and startup failures emit only safe categories/types, without exception
messages or dumps.

Four new model-free privacy controls are in the existing CI-covered
`scripts/test_benchmark_qwen38_mtp.py`. All 27 tests pass, covering the RAM
override, stable eight-cell inputs after source deletion, absence of corpus
text from artifacts/output, and runtime/startup error redaction. This changes
benchmark tooling only; no model or performance rerun is claimed.

Historical privacy correction: the archived v2 runner copied its synthetic
source-code corpus into each run's ignored output as `frozen-corpus.rs`; the
assessment's original claim that no prompts were retained was too broad.
That copy stayed in ignored runtime storage. No corpus text, generated
completion, token vector or probability was published. The archived runner,
aggregate data and historical results remain byte-exact. The assessment now
records this dated correction rather than changing the measurements.

## Fresh native build commands

Commands, with `CARGO_TARGET_DIR` set respectively to ignored
`runtime/projection-pr-review/generator-target` and `core-target` in this
checkout:

```sh
cargo +1.98.1 build --release --locked -p stark-generate
cargo +1.98.1 test --release --locked --manifest-path vendor/lily-core/Cargo.toml projection_reuse_matches_ordinary_bytes -- --ignored --nocapture --test-threads=1
```

No model/performance replay, merge, installation or release is part of this
follow-up. CI results must be checked separately on the PR head.
