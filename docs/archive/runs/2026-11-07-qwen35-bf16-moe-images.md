# Qwen3.5 BF16 MoE image implementation

Date: 2026-11-07. Base source: `e2358ada2e0f6f2f34fb6fc1a870f74456988294`.
Branch: `codex/qwen-moe-bf16-images`.

This record describes structural implementation and bounded synthetic controls.
The full official BF16 35B artifact was not downloaded, converted or loaded.
No full-model quality, numerical parity, worker lifecycle, throughput or memory
qualification is claimed for that artifact. Native checks are coordinated by
the integration checkout; this worktree creates no Cargo target.

## Immutable source evidence

The source is
[`Qwen/Qwen3.5-35B-A3B@59d61f3ce65a6d9863b86d2e96597125219dc754`](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/tree/59d61f3ce65a6d9863b86d2e96597125219dc754).
Only metadata and two HTTP 206 ranges of at most 1 MiB each were read. Aggregate
source transfer was below 3 MiB; no payload was retained locally. The source
config establishes 40 layers, model width 2,048, 256 experts, top-8 routing,
512-wide routed/shared experts and the 27-layer vision tower.

| Source component | Bytes read or header bytes | SHA-256 |
| --- | --- | --- |
| `config.json` | 3,544 | `5e4d7f74fec2f360eb9cfbfcd6ec0c4c76e684d3a11caaed259d9fd9bfbc7944` |
| `model.safetensors.index.json` | 187,463 | `d8d0b7ca4e61ae107e3e87a3ff21136b3ac7c789e64bb24267227ca804e04205` |
| Shard 6 header, excluding length prefix | 688 | `29b1d47c04f7ba5d0db2cc3f1f5c5da1734458c9827718dc3431b73e6351d6e4` |
| Shard 12 header, excluding length prefix | 1,328 | `05cb22d118a5fc2c655dc7c686617c20a9d770c4fcd9e4ee663524af813a09a2` |

Layer-0 `experts.gate_up_proj` is BF16 `[256, 1024, 2048]` and
`experts.down_proj` is BF16 `[256, 2048, 512]`. The source index binds all
expert/shared/router names. These header facts do not establish payload hashes.

The primary operation reference is Transformers v5.8.0, commit
[`049d2bf1220747b6d39e2a978b9f5fe0defa1dca`](https://github.com/huggingface/transformers/blob/049d2bf1220747b6d39e2a978b9f5fe0defa1dca/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py),
`modeling_qwen3_5_moe.py` SHA-256
`efc5f0085cc7a7b0f334d864f2ca1f0a257e6393e257393e329f160f6cc3f97a`.
It defines gate-before-up fused banks, F32 softmax followed by selected-score
normalization and a BF16 weight cast, expert-ID iteration with BF16 products and
partial sums, and a sigmoid-gated shared expert. Existing upstream license and
provenance records remain applicable; no Transformers runtime is imported.

## Implementation and bounded checks

Version 5 binds full BF16 expert banks separately from affine version 4. Native
admission reconstructs exact MoE geometry and full-precision header metadata.
It rejects changed operations, folded norms, incorrect BF16 flags, malformed
routing counts, expert width/index overflow, packed expert storage and missing
shared-gate tensors. Converter tests exercise source-backed fused row views,
per-expert gate/up swapping, source manifest binding, all-BF16 output, mixed
dtype rejection, unknown aliases, missing/incorrect banks and atomic failure.
Preflight and catalog tests keep BF16 MoE image-only and require source-bound
processing metadata. Runtime recognition alone never grants qualification.

The existing BF16 reference linear rejected gathered banks and the router's F32
storage. The new indexed dot keeps the reference eight four-lane F32 registers
and fixed reduction while loading selected bank rows directly. The router's F32
buffer contains BF16-rounded values. Synthetic native controls compare indexed
and contiguous dots bitwise over multiple rows and noncontiguous IDs, exercise
invalid IDs and geometry, and run full banks with the actual
`Qwen35ReferenceBF16` profile and source-ordered BF16 accumulation.

A whole-Qwen MoE synthetic component additionally loads the actual reference
profile and exercises router BF16/F32 storage, three selected experts with mixed
signs, four input rows and the separate sigmoid-gated shared branch against an
independent scalar BF16-boundary calculation. It is a bounded component control,
not a substitute for official trained weights or image-worker execution.

Local converter/preflight/catalog/geometry and existing Qwen regression checks:
512 passed. Repository policy, scoped Ruff, Rust formatting and diff whitespace
checks passed. Native tests were authored but not compiled or executed in this
worktree; the integration owner runs them in its existing warmed build.

The indexed dot allocates only `[rows, selected_outputs]` F32 accumulation and
BF16/F32 output, independent of bank row count and inner width. RoutedExperts
still queues its existing route/index/activation/output scratch per selected
expert in a command buffer. No temporary weight matrices multiply that retained
scratch by model or expert width. This is allocation accounting from source,
not a measured 35B peak-memory result.

A CPU-only allocation regression uses test-only pool accounting to count every
scratch request while encoding 32 rows with top-3 experts in one command buffer.
The 96-route synthetic graph must request less than 512 KiB in aggregate. A
materialized gather would require over 2 MiB of weight matrices alone for that
fixture. This bound excludes page rounding and allocator overhead; the test is
authored for the integration native run and has not run in this worktree.

Image prefix reuse remains disabled. Attention positions/KV, DeltaNet F32
recurrence, convolution ring/history and image-dependent M-RoPE state lack a
complete image-backend snapshot/restore contract. Rewinding token counts cannot
recover overwritten ring state. Fresh state is required for text and image
requests through this artifact. Affine v3/v4, Gemma and ordinary text policies
retain their existing selection.
