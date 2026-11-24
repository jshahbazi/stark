# Inkling 2 Mini worker-schedule captures and checks

Status: tooling. No new pinned capture has been taken with this packet, so nothing here compares Stark with the pinned reference at a new prompt length or context. The Stark-only checks below have run on uncommitted builds on 2026-11-22. Their results are recorded in [20261122-INKDEFAULT-01](../../../model-experiments/2026-11-22-inkling-mini-default-fast-paths.md) and [20261122-INKLONGPOS-01](../../../model-experiments/2026-11-22-inkling-mini-long-position-gain.md). This packet qualifies no artifact or build by itself.

This packet extends the [2026-11-21 Ollama capture protocol](../2026-11-21-inkling-oracle/README.md) to caller-chosen prompt lengths and contexts. It was written for two opt-in selectors that never shipped. Their behaviour is now the default on the original M5 Max worker, with no environment variable:

- The worker's source prefill and multi-row source verifier admit every non-empty text prompt (floor 1 token; it was 512). Since [20261122-INKFASTPRE-03](../../../model-experiments/2026-11-23-inkling-mini-prefill-round3.md), packed bulk commands of 16 rows or more run the source kernels on the default path; shorter commands use the lane or serial kernels. The 65-row source-kernel minimum now applies only to the `--exact-prefill` reference.
- The source global-attention workspace is sized by context, from the default 4,352 keys up to 32,768, and source prefill and the source verifier are admitted above 4,352.

The `STARK_INKLING_SOURCE_PREFILL=0` kill switch was removed later the same day ([20261122-INKFUSE-02](../../../model-experiments/2026-11-22-inkling-mini-launch-fusion.md#20261122-inkfuse-02--second-session-paired-timing-and-single-row-fusions)), so the source path cannot be disabled by environment. Since [20261122-INKFASTPRE-02](../../../model-experiments/2026-11-22-inkling-mini-unguarded-mxfp8-prefill.md) the default prefill is not bit-exact against the pinned reference; pass `--exact-prefill` to the subcommands below for the bit-exact reference kernels. A comparison against the earlier 512-token floor and 4,352 context limit needs a build of `f1666090` or earlier.

The archived directory and its scripts are unchanged. They remain bound to their recorded results.

## Provenance

The captures reuse the pinned stock-schedule runner without rebuilding it. Its SHA-256 is `64293fa71651074e08703bdcd33d66c870b1ea23967f04f9da17bd6f095693b5`, as recorded in [the packed-prefill record](../../../model-experiments/2026-11-21-inkling-mini-packed-prefill.md). It was built from Ollama revision `6949a42ad24450cf567d7ce1ff7d278ebb3eb5d8` with the archived `instrument.py`, `capture.go` and `full_target.py`, plus private prefill and expert trace extensions that stay inactive without a `PrefillTrace` request field. It loads the imported manifest `bd9c92c3d4c4c1a6c6904dea6a51e8503cceb0f77f8073c2c9e5c63976e0aa4d` and the release Metal v4 libraries beside it.

The runner already accepts any `Start`, any chunk list and any `--ctx-size`. It prefills the chunks in order and then evaluates four singleton rows from `Start`. No source change is needed for the new shapes.

| File | Purpose | Origin |
|---|---|---|
| [schedule.py](schedule.py) | Case list, chunk schedule and replay boundaries | New |
| [make_requests.py](make_requests.py) | Private requests and self-check token files | Parameterised copy of `request_text` in [workloads.py](../2026-11-20-inkling-mini/workloads.py) |
| [capture.py](capture.py) | Runs the pinned runner per case and records hashes | New; follows the archived reproduce loop |
| [compare_digests.py](compare_digests.py) | Compares two Stark reports and ignores timings | New |
| [test_schedule.py](test_schedule.py) | Model-free schedule tests | New |

The comments in `schedule.py` still describe the two selectors and a floor of 69, and its `MAX_VERIFY_ROWS` is 8 where the engine's is 4. The larger value only makes `check_case` demand more context room; the case list and chunk schedule are unaffected.

The request generator keeps the workloads file's message headers, odd-window synthetic records and exact-length padding. It uses as many records as the length requires. Prompts below 512 tokens use one short user turn, because the full headers alone exceed the shortest lengths. With the release `tokenizer.json`, SHA-256 `7f332727f9a3f093dd452a39cc978e6757627e5dad1b0673aa4e75174f236dd6`, and `tokenizers` 0.23.3, a dry run produced these token-sequence hashes. Each hash covers the little-endian `u32` IDs of N + 3 tokens.

| Case | N | Context | Capture chunks | Token SHA-256 |
|---|---|---|---|---|
| s68 | 68 | 4,352 | none, self-check only | `decbf4f6b484ac096143b75c5766afafa91bb58e17a32c7f249c9f338b40a1f1` |
| s69 | 69 | 4,352 | 65, 3 | `05802c3dd4712e483bc356538c9596322d4dfad77e56d48ca68fe90c888b5a11` |
| s73 | 73 | 4,352 | 69, 3 | `699b1a59240efb93cac1d68bde0e4626fe7fa2ac9e2afe5a9a69266e30831e94` |
| s130 | 130 | 4,352 | 126, 3 | `9fa84f771154a85ac24d702fe64dc5cf1a87e22954b8ea6c3eae31e05dd8be18` |
| s511 | 511 | 4,352 | 507, 3 | `b8b3d37bb2dea69788b7ec0c975041f67579ee87782348d719cffd6fd6642d8b` |
| r4110 | 4,110 | 4,352 | 2 × 2,048, 10, 3 | `0ad360044a4b0d3552091be1dda41158685145a884e409275e6d9037ec4d202a` |
| r4150 | 4,150 | 4,352 | 2 × 2,048, 50, 3 | `9dafa17998a585677c2126d5a23c7ef31d17844e7d12d677178f998378648625` |
| l4350 | 4,350 | 9,216 | 2 × 2,048, 250, 3 | `e19e34f99ce3c6f99ffd587a670e5a1647bf5ed8fe5f94fe0a0c86acb7804648` |
| l8188 | 8,188 | 9,216 | 3 × 2,048, 2,040, 3 | `341bfc64e41ffa6343d6b62f76bdef250dfc75cc7b79937289b78b467c61cb14` |
| l8800 | 8,800 | 9,216 | 4 × 2,048, 604, 3 | `aaf186957a21d96c335503f1974bf366c4bc48be7676407a9cd938dbbdb3b99b` |
| l16000 | 16,000 | 16,384 | 7 × 2,048, 1,660, 3 | `e36586d2876e586fc87324038f87b57ab62904ce100b3cc2c7ab011128730b59` |

`Start` is N − 1 for every case, so row 0 is the generation seed and rows 1–3 are teacher rows. s68 has a 64-row bulk command and s69 a 65-row one; under the exact reference the 65-row minimum separates them (incumbent packed kernels against source kernels), while on the default path both use the source kernels. `capture.py` skips s68 because it was planned as a below-floor control; it now serves only the self-checks. For l8800, the 604-row bulk tail, the seed and all teacher rows lie past position 8,192, where global attention log scaling begins. l4350 and l8188 place the verifier's rows across positions 4,352 and 8,192. r4110 and r4150 end the bulk with a 10-row command, which takes the per-row lane path, and a 50-row command, which takes the incumbent packed kernels under the exact reference and the source kernels on the default path. These remainders already occurred at 4,352 under the earlier 512-token floor, for example at about 2,053 to 2,116 tokens, and they are more common now that the workspace is sized by context. l16000 uses the requested 16,384 context. The worker's automatic context planner may choose another size on a given machine.

## Capture and check conventions

- Keep requests, logs, token files and captures in an ignored private directory, such as `runtime/inkling-schedule-oracle-2026-11-22`. Publish only aggregates and hashes.
- Run one GPU job at a time. Do not overlap a capture with a Stark check.
- The pinned runner exports a cache window hard-coded to 4,352 keys in `capture.go`. Above that extent the exported `keys` and `values` tensors in `row-0.safetensors` are a rotated tail window. The full-target check does not read them. Do not feed these captures to the archived `merge.py` or the two-layer `inkling_slice <shard> <reference>` check.
- The archived runs found that chunk width can change routed expert results. Use the worker chunk schedule above. If a long capture runs out of memory, record it as resource-deferred instead of narrowing the chunks.

## Stark diagnostics

The `inkling_slice` example has four subcommands for this packet, implemented in [`schedule_validation.rs`](../../../../crates/stark-engine/src/engine/inkling_slice/schedule_validation.rs). Existing subcommands and their outputs are unchanged.

| Subcommand | Arguments | Needs a capture | Accepted prompts |
|---|---|---|---|
| `--full-target-schedule` | target, capture directory, context | Yes; N is read from `row-0` | N ≥ 5 and N + 3 ≤ context |
| `--schedule-replay` | target, directory with `tokens.safetensors`, N, context | No | N ≥ 5 and N + 3 ≤ context |
| `--measure-verify-transaction-schedule` | target, draft, directory, N, context | No; checks the seed against `row-0` when present | N ≥ 5 and N + 100 ≤ context |
| `--measure-source-dspark-depth-schedule` | target, draft, directory, N, context | No; checks the seed and first teacher row when present | N ≥ 5, N + 40 ≤ context and a source workspace covering the context |

With the source path active, each command submits bulk commands of at most 2,048 rows over the first N − 4 tokens, a three-row tail and the seed step. Otherwise it ingests the first N − 1 tokens. `--full-target-schedule` and `--schedule-replay` choose the precision scope with `worker_admits_source`, a copy of the worker's length rule, which now admits every non-empty prompt. Reports state that copy's result as `worker_source_admitted` and the session's effective path as `source_text_active`. Both fields are expected to be true for every case above. The copy is not the worker's code. The verifier command refuses to run unless the source path is active. The DSpark command forces the source precision scope, as the archived 4K command does.

Every report includes `prefill_commands`, the command lengths actually submitted, and `context_workspace_additional_bytes`, the global workspace above the default 4,352 keys (0 at or below 4,352; with the explicit mask freed on the default path since INKFASTPRE-03, 39,845,888 at 9,216 and 232,783,872 at 32,768, and by the same per-key formula 98,566,144 at 16,384; the earlier mask-inclusive figures were 188,801,024, 467,034,112 and 1,102,995,456). The verifier report also includes `verifier_additional_metal_bytes`, the device allocation change when the source verifier is enabled.

Each command restores snapshots taken at the end of bulk commands at 4,096 and 8,192 and at the worker's prefix-reuse boundary. It replays the rest of the schedule and compares the seed logits and the committed state bit for bit (`boundary_replays`). The verifier command includes the draft state. It also checks every rejection width from the seed (`rejection_controls`) and from the pre-seed snapshot (`crossing_controls`), an abandoned transaction and the rejection of a source-mode snapshot in legacy mode. It then runs speculative greedy rounds at depths 1 to 3, which must equal 96 serial greedy tokens and the committed target state. Verified positions run from N to N + 94 (`first_verified_position`, `last_verified_position`).

Exit status 1 from `--full-target-schedule` means at least one of `all_logits_exact`, `exact_snapshot_replay` and `exact_boundary_replays` is false, or the command failed. Exit status 1 from `--schedule-replay` covers the two replay fields. The verifier command exits 1 on a rejection or crossing control mismatch, a speculative round that changes serial output or committed state, `seed_matches_pinned` false or a boundary replay mismatch. The DSpark command exits 1 on a pinned mismatch, a packed tap mismatch or a changed serial replay. Any other error also exits 1 with a one-line label on standard error.

## Commands for AC power

Set `STARK` to this checkout, then set the variables below. Keep each output private until you have reduced it to aggregates. No environment selector is needed.

```sh
umask 077
cd "$STARK"
SLICE="$STARK/target/release/examples/inkling_slice"
TOOLS="$STARK/docs/archive/experiments/2026-11-22-inkling-schedule-oracle"
PRIVATE="$STARK/runtime/inkling-schedule-oracle-2026-11-22"
cargo +1.98.1 build --release --locked -p stark-engine --no-default-features \
    --features inkling-slice --example inkling_slice
```

`$TARGET`, `$DRAFT` and `$TOKENIZER` are the original target shard, `mtp.safetensors` and `tokenizer.json`. `$RUNNER` is the pinned stock-schedule runner and `$STORE` is the original Ollama store. `$STOCK` is one existing stock-schedule 4K capture directory with Start 4095.

### 1. Requests and self-check tokens

```sh
(cd "$TOOLS" && python3 -m unittest -q test_schedule)
(cd "$TOOLS" && uv run --offline --no-project --with tokenizers==0.23.3 \
    python make_requests.py "$TOKENIZER" "$PRIVATE")
```

### 2. Captures

```sh
(cd "$TOOLS" && python3 capture.py "$RUNNER" "$STORE" "$PRIVATE")
```

Pass `--cases s69,s73` to capture a subset. Each case writes `capture.json` and reports whether the runner's tokens equal the release tokenizer preview. If they differ, rerun the self-checks on the captured tokens. The stock 4K capture took about one minute including load. Expect a few minutes per long case. The 16K case is untested and may exceed memory.

### 3. Checks without a capture

These compare Stark with itself. They can run before step 2. Run `compare_digests.py` on each pair.

Workspace capacity on the stock capture. The 9,216-token run must report `source_text_active` true and exit 0. Its rows and cache fingerprints must equal the 4,352 run, ignoring `context` and the workspace byte count.

```sh
"$SLICE" --full-target-schedule "$TARGET" "$STOCK" 4352 > "$PRIVATE/stock-4352.json"
"$SLICE" --full-target-schedule "$TARGET" "$STOCK" 9216 > "$PRIVATE/stock-9216.json"
python3 "$TOOLS/compare_digests.py" "$PRIVATE/stock-4352.json" "$PRIVATE/stock-9216.json" \
    --ignore '^context$' --ignore 'context_workspace_additional_bytes'
```

`--full-target-schedule` on `$STOCK` at 4,352 should also reproduce the `native_logits_sha256` and `prefill_native_cache_sha256` values that `--full-target-batched` reports for the same capture.

Short-prompt replay. Both runs must report `worker_source_admitted` and `source_text_active` true and exit 0. s68 submits a 64-row bulk command and s69 a 65-row command; with `--exact-prefill` they fall on either side of the reference's 65-row source-kernel minimum, and on the default path both use the source kernels. There is no below-floor comparison any more.

```sh
"$SLICE" --schedule-replay "$TARGET" "$PRIVATE/s68" 68 4352 > "$PRIVATE/s68-replay.json"
"$SLICE" --schedule-replay "$TARGET" "$PRIVATE/s69" 69 4352 > "$PRIVATE/s69-replay.json"
```

There is no environment negative control any more; the source path is always active on a worker that loaded it.

Drafting on equals drafting off, with rollback, prefix replay and the verifier rows across 4,352 and 8,192. Each command must exit 0.

```sh
for case in s68:68:4352 s69:69:4352 s73:73:4352 s130:130:4352 s511:511:4352 r4110:4110:4352 r4150:4150:4352 \
        l4350:4350:9216 l8188:8188:9216 l8800:8800:9216 l16000:16000:16384; do
    name=${case%%:*} rest=${case#*:}
    prompt=${rest%%:*} context=${rest#*:}
    "$SLICE" --measure-verify-transaction-schedule \
        "$TARGET" "$DRAFT" "$PRIVATE/$name" $prompt $context > "$PRIVATE/$name-txn.json"
done
```

The same token files also serve other contexts, for example l8800 at 16,384 or l16000 at 32,768, provided N + 100 fits.

### 4. Checks against the new captures

These bit-exact expectations apply to the reference kernels, so each command passes `--exact-prefill`. Each must exit 0. `source_text_active` must be true. Every row's logits, hidden state and 48 visit partials must report zero unequal values. `boundary_replays` must cover 4,096 and 8,192 for l8800 and l16000. The default (inexact) prefill is judged against the captures with `--prefill-tolerance` and its noise-floor gate instead ([INKFASTPRE-02](../../../model-experiments/2026-11-22-inkling-mini-unguarded-mxfp8-prefill.md)).

```sh
for case in s69:4352 s73:4352 s130:4352 s511:4352 r4110:4352 r4150:4352 l4350:9216 l8188:9216 l8800:9216 l16000:16384; do
    name=${case%%:*} context=${case#*:}
    "$SLICE" --full-target-schedule --exact-prefill "$TARGET" "$PRIVATE/$name" $context \
        > "$PRIVATE/$name-full.json"
done
```

Rerun the step 3 verifier loop with `--exact-prefill`. Every report with a capture must now show `seed_matches_pinned` true. Then run the source DSpark depth check. `pinned_unequal_logits` must be `[0, 0]`.

```sh
for case in s69:69:4352 s511:511:4352 l8800:8800:9216 l16000:16000:16384; do
    name=${case%%:*} rest=${case#*:}
    prompt=${rest%%:*} context=${rest#*:}
    "$SLICE" --measure-source-dspark-depth-schedule \
        "$TARGET" "$DRAFT" "$PRIVATE/$name" $prompt $context > "$PRIVATE/$name-dspark.json"
done
```

## Limits

- These are engine checks. The worker's admission lives in `stark-generate`; `worker_admits_source` is a copy of its length rule, not the worker. A worker-level check needs requests through `stark-generate`.
- Prompts of 1 to 4 tokens are admitted by the worker (the bulk is empty and the tail shrinks) but none of these subcommands accepts them. They are unchecked here.
- The source DSpark command forces the source precision scope, as the archived 4K command does. It refuses contexts that the allocated workspace does not cover.
- Without captures, steps 3 and 4 establish self-consistency (drafting against serial decode, snapshot replay, context-size neutrality), not agreement with the pinned reference.
- Record every run, including failures, in `docs/model-experiments/` as separate attempts. Bind the build and capture hashes. Link them from this packet only through a new dated record.
