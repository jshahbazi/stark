# 20261122-INKDEFAULT-01 · Default fast paths and wider source admission for Inkling 2 Mini

## Identity and state

| Field | Value |
| --- | --- |
| Attempt ID / registered date / executed date | `20261122-INKDEFAULT-01` / 2026-11-22 / 2026-11-22 |
| Kind | Code adoption with correctness checks: decode selectors made unconditional, source prefill and verifier admission widened, default drafting |
| Attempt state | Completed for the checks listed. The long-context failure it exposed is recorded and fixed under [20261122-INKLONGPOS-01](2026-11-22-inkling-mini-long-position-gain.md). |
| Result and acceptance criterion | Passed: stock 4K full-target parity unchanged, the 4K transaction check and Stark self-consistency at N = 68 to 8,800. Failed at N = 16,000 on this build. Parity with the pinned reference at the new lengths and contexts is not checked. |
| Evidence scope | Full-model numerical (four pinned stock 4K captures) and full-model self-consistency (drafting against serial decode, snapshot replay); model-free tests. No end-to-end worker or desktop run and no timing. |
| Promotion decision | Adopted as the branch default in the working tree, awaiting qualification at new lengths against a reference |
| Disposition | Adopted on branch, uncommitted on top of `f1666090`. Not merged, released or installed. |
| Predecessor / related attempts / PR | [Opt-in decode attempts 01–78](2026-11-21-inkling-mini-decode-60.md) and [79–110](2026-11-22-inkling-mini-decode-80.md); [default one-row router](2026-11-21-inkling-mini-decode-default.md); [default fast prefill](2026-11-21-inkling-mini-prefill-default.md); [20261122-INKLONGPOS-01](2026-11-22-inkling-mini-long-position-gain.md); [20261122-INKFUSE-01](2026-11-22-inkling-mini-launch-fusion.md); [20261122-INKENGINE-01](../benchmarks/2026-11-22-inkling-mini-engine-matrix.md). No PR. |

## Change

Twenty-six `STARK_INKLING_EXPERIMENT_*` environment selectors were removed:

- The 24 retained decode selectors now describe unconditional behaviour: `NVFP4_VECTOR_BLOCK`, `PARALLEL_ROUTE`, `CERTIFIED_SAMPLE`, `SOURCE_VERIFY`, `SHARED_LOCAL`, `SHARED_GLOBAL`, `EXACT_RUN_WIDTH`, `VERIFY_HALF_INGRESS`, `DRAFT_SHORTLIST`, `DRAFT_INT8`, `VERIFY_FP4_LOOKUP`, `VERIFY_NATIVE_ROUTER`, `VERIFY_COMPACT_RUNS`, `VERIFY_FUSED_FINISH`, `VERIFY_FUSED_ATTENTION`, `VERIFY_SHORT_ROUTER`, `VERIFY_LEAN_DECODE`, `VERIFY_OVERLAP`, `VERIFY_FUSED_CACHE`, `VERIFY_SINGLE_DRAFT`, `VERIFY_COMBINED_SHORT`, `VERIFY_QKV_BATCH`, `VERIFY_MIX_ROWS` and `VERIFY_MXFP8_VECTOR`.
- `NVFP4_ROWS64` is unconditional: the 64 × 32 grouped NVFP4 prefill tile always registers.
- `NVFP4_SHARED_SCALES` was removed without being enabled. It is mutually exclusive with the 64 × 32 tile and its measured gain was smaller ([run tiles](2026-11-21-inkling-mini-nvfp4-run-tiles.md), [shared scales](2026-11-21-inkling-mini-nvfp4-router.md)). Its registration function remains in the tree as unused code.

The byte-pair FP4 coefficient table takes precedence over the lean (affine) decoder, as in the retained configuration; the affine decoder is compiled out. The remaining environment reads are the diagnostics `STARK_INKLING_EXPERIMENT_PROFILE_VERIFY` and `STARK_INKLING_EXPERIMENT_MXFP8_TENSOR_AUDIT` and the `STARK_INKLING_SOURCE_PREFILL=0` kill switch, which is not yet removed.

Two admission behaviours change. They never shipped as selectors.

| Behaviour | Before (`f1666090`) | Now |
| --- | --- | --- |
| Source prefill and multi-row source verifier, input length | At least 512 tokens | At least 1 token |
| Source prefill and verifier, context | At most 4,352 | Up to 32,768, with the source global-attention workspace sized by context |
| Global workspace keys | 4,352 | Context, clamped to 4,352–32,768 |
| Extra bytes charged at admission | None | 188,801,024 at context 9,216; 467,034,112 at 16,384; 1,102,995,456 at 32,768; 0 at or below 4,352 |

The load-time source reserve still covers the default 4,352 keys. Only a source-path request charges the extra workspace, and the automatic context planner includes it. Bulk commands shorter than 65 rows still run the incumbent packed (16 rows or more), lane or serial kernels inside the source-scoped request.

Default drafting also changes. A text request that names no MTP setting, or `mtp: true` without a count, resolves to three proposals greedy and two sampled when the source verifier and DSpark are loaded. This applies on the worker, API, chat CLI and desktop paths. Image requests stay at zero drafts and explicit user settings override. The worker advertises MTP only with the multi-row source verifier. If the verifier is unavailable at a request's context, a drafted request decodes serially; the legacy one-draft route is unreachable from the worker.

Source: [worker admission and drafting](../../crates/stark-generate/src/inkling.rs), [packed prefill admission](../../crates/stark-engine/src/engine/inkling_slice/packed_prefill.rs), [global workspace sizing](../../crates/stark-engine/src/engine/inkling_slice/packed_global_probe.rs), [converter MTP resolution](../../tools/converter/stark_convert/mtp.py).

## Artifact, source and environment

| Field | Value |
| --- | --- |
| Checkpoint | Original fixed-three-pass `thinkingmachines/Inkling-2-Mini` at `6e7c8248f75265bdd4fbab5273304243b5d8c2cb`; target shard and resident DSpark shard SHA-256 as in the [default-router record](2026-11-21-inkling-mini-decode-default.md#artifact-and-run-controls) |
| Reference | Pinned stock-schedule Ollama captures `g4096-01-retry`, `g4096-02`, `s4096-00` and `s4096-01` ([capture protocol](../archive/experiments/2026-11-21-inkling-oracle/README.md)). No pinned capture exists for any other prompt length than 4 and 4,096 tokens. |
| Source | Uncommitted working tree on top of `f1666090`. The defaults-only tree state was not saved as a patch; no patch hash is recorded. |
| Builds | Defaults-only: `stark-generate` SHA-256 beginning `2da8222d57cf5080`, `inkling_slice` example beginning `b45cbce0d2dbff70`. Full hashes are not recorded here. Locked Rust 1.98.1. |
| Precision | Unchanged from the retained opt-in configuration: original NVFP4 and MXFP8 experts, source BF16 rounding points, F32 accumulation as in the qualified kernels |
| Host | 128 GiB Apple M5 Max (Mac17,7), macOS 26.7.2 (25G332), AC power, `pmset` powermode 2 |
| Memory | Admission charges computed from the workspace formula above. Actual Metal allocation at the larger workspaces was not measured. |

## Correctness checks on the defaults-only build

| Check | Inputs | Result |
| --- | --- | --- |
| `--full-target-batched` at context 4,352 | The four pinned stock 4K captures | Exit 0 on all four. Logits and hidden state had zero unequal elements on every row. The only nonzero intermediate fields were single `down_quantized` elements: `g4096-02` row 2 physical 4, and `g4096-01-retry` row 0 physical 5 and row 3 physical 9. These are identical to the earlier session's reports for the same captures; those reports are private and their hashes are not recorded here. |
| `--measure-verify-transaction` | `g4096-02`, 4,096 tokens | Exit 0 |
| `--measure-verify-transaction-schedule` (new) | Self-check tokens, context 4,352: N = 68, 69, 73, 130, 511, 4,110, 4,150 | Pass |
| Same | Context 9,216: N = 4,350, 8,188, 8,800 | Pass |
| Same | Context 16,384: N = 16,000 | Fail: committed state differed from serial decode while emitted tokens matched. See [20261122-INKLONGPOS-01](2026-11-22-inkling-mini-long-position-gain.md). |
| `--full-target-schedule` (new) | `g4096-02` capture at context 9,216 against the same at 4,352 | 2,785 fields compared, 0 differing, ignoring the context value and the workspace byte count. `source_text_active` true; additional workspace 188,801,024 bytes. |
| Model-free tests | Same tree | Worker library 165 passed. Converter suite 7,985 passed, 96 skipped. Repository policy check passed. |

The transaction-schedule check runs drafting at depths 1 to 3 and requires 96 greedy outputs and the committed state to equal serial decode. It also checks every rejection width from the seed and from the pre-seed snapshot, an abandoned transaction, mode isolation and replay from the 4,096, 8,192 and worker prefix boundaries. These are Stark-against-Stark checks. They do not compare with the reference.

## New tooling

- [`schedule_validation.rs`](../../crates/stark-engine/src/engine/inkling_slice/schedule_validation.rs) reproduces the worker's cold text schedule for an N-token prompt at any context and reports aggregates and SHA-256 fingerprints only.
- The [`inkling_slice` example](../../crates/stark-engine/examples/inkling_slice.rs) gains four subcommands: `--full-target-schedule`, `--schedule-replay`, `--measure-verify-transaction-schedule` and `--measure-source-dspark-depth-schedule`.
- The [worker-schedule capture packet](../archive/experiments/2026-11-22-inkling-schedule-oracle/README.md) holds the case list, request generator, capture driver, digest comparison and commands. Its README was written for two opt-in selectors and a floor of 69; it was revised to the default behaviour in this change. No capture has been taken with it.

## Prefix reuse, images and lifecycle

The worker's prefix-snapshot boundary on the source path keeps the four-token suffix and 2,048-row bulk alignment, so inputs below 2,052 tokens have no boundary. The transaction-schedule check replays snapshots at 4,096, 8,192 and the worker boundary with exact seed logits and committed state on the passing cases. A worker-level prefix restore at the new lengths and contexts was not run. Image requests keep zero drafts and the existing text-prefix invalidation; no image request was run on this build. Cancellation and failure clearing paths are unchanged in code and were not exercised.

## Performance

Not measured in this attempt. The first-look engine matrix ([20261122-INKENGINE-01](../benchmarks/2026-11-22-inkling-mini-engine-matrix.md)) includes this defaults-only build as its baseline arm, one invocation per arm.

## Limits and open items

- Parity with the pinned reference at prompt lengths other than 4 and 4,096 and at contexts above 4,352 is unchecked, because no capture exists. The capture packet is ready but has not run.
- A latent gap in the previous default: prompts whose bulk remainder after 2,048-row commands is 1 to 64 rows, for example about 2,053 to 2,116 tokens, already mixed incumbent kernels into a source-scoped request at context 4,352 and were never checked against the reference. Cases r4110 and r4150 cover that shape for self-consistency only.
- Prompts of 1 to 4 tokens are admitted but none of the schedule subcommands accepts them, so they are unchecked.
- Actual Metal allocation at the larger workspaces was not measured. No end-to-end worker, API, chat CLI or desktop run was made on this build.
- The kill switch `STARK_INKLING_SOURCE_PREFILL` remains.
- Performance playbook: not updated with this record; any update is a separate change.

Keep prompts, completions, token vectors and probabilities out of this document; the private reports stay ignored.

## Follow-up history

- 2026-11-22: the `STARK_INKLING_SOURCE_PREFILL` kill switch was removed under [20261122-INKFUSE-02](2026-11-22-inkling-mini-launch-fusion.md#20261122-inkfuse-02--second-session-paired-timing-and-single-row-fusions). The statements above that it remains describe the defaults-only build.
- 2026-11-22 to 11-23: the default prefill changed under [20261122-INKFASTPRE-01](2026-11-22-inkling-mini-exact-fast-prefill.md) (exact) and [20261122-INKFASTPRE-02](2026-11-22-inkling-mini-unguarded-mxfp8-prefill.md) (unguarded MXFP8, no longer bit-exact against the pinned reference). The parity results above bind to the defaults-only build.
- 2026-11-23: [20261122-INKFASTPRE-03](2026-11-23-inkling-mini-prefill-round3.md) freed the explicit global mask on the default path, so the extra workspace charged at admission fell to 39,845,888 bytes at 9,216 and 232,783,872 at 32,768. The figures in the admission table above describe the defaults-only build. Bulk commands of 16 rows or more now use the source kernels; the 65-row threshold applies only to the exact reference.
