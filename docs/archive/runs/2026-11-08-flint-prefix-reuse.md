# Flint generation-prefix implementation

Date: 2026-11-08. Implementation base: `1678e1a7`, derived from Stark's
`b9a91a94` native engine. This record describes a new state-restoration path.
The [November 7 tool qualification](2026-11-07-glm-flint-tool-qualification.md)
and the bounded MaxxBench pilots remain tied to their original workers and
fresh-state behavior.

## Disposition

Implemented; compilation, artifact qualification and timing measurements are
pending the coordinated hardware window. No model-bound correctness, speedup,
maximum-context operation or measured-memory claim follows from this record.
The benchmark remains paused until the runtime path is qualified.

## Artifact and execution policy

The intended qualification artifact is `RecursiveSI/flint-int4` at
`3b93f9053b530488405ae9fca51d4900d6ce21f0`. Its prepared config SHA-256 is
`06271655d67c1ce5cfc08ccc771faaddcc1a1c9719b3b2c499ddae1c1aa8cef0`;
the complete prepared-artifact fingerprint from the MaxxBench inventory is
`4c84481ae7657c35954de9d4e4e28ae0670febe162f83c30d550d3264b5e4764`.
These identities must be checked again before and after the artifact run.
Worker source, binary hash, validation-harness hash and native toolchain will
be recorded with that run rather than inferred from the implementation base.

The implementation preserves original affine INT4/BF16-scale weights and
eight-bit embeddings. Matrix operations use the admitted native Metal path;
attention KV storage, host normalization, attention accumulation and snapshot
payloads remain F32. No precision change, new matrix kernel, prompt change or
sampling change is introduced.

The configured context remains the artifact's declared 262,144-token maximum.
Qualification may use shorter synthetic histories to exercise restoration;
that does not establish behavior at a filled maximum window. MTP is unavailable
on Flint. The worker reseeds greedy/stochastic request sampling through its
existing path.

## State inventory and restoration

| State | Ownership and restoration |
| --- | --- |
| Full-attention KV | Per-layer host `VecDeque<Vec<f32>>`; every prefix row is copied into the snapshot and copied into reset state. |
| Sliding-attention KV | The ordered, most recent `min(position, 513)` rows are copied for each sliding layer, preserving both eviction state and iteration order. |
| Absolute position | Restored with the copied attention rows; RoPE uses that same absolute next position. |
| Attention sink and routing | Immutable model weights/configuration; no mutable recurrence or sink history is read across tokens. |
| Matrix scratch | Native matrix methods return completed host vectors before capture. Scratch is overwritten by subsequent operations and is not part of autoregressive state. |
| Sampler | Owned by the worker request and cleared/reseeded through the existing generation lifecycle. |

Capture occurs after a complete non-final prefix on a 1,024-token grid. The
scalar suffix uses the same per-token kernels, precision and reduction order
as fresh execution. Snapshots contain complete state and do not alias KV rows
that a later request could overwrite. The model's private ownership token
rejects snapshots from another loaded instance. Restore checks the boundary,
context, layer count, row widths, finite values and full/sliding row counts
before changing state. Failed restore clears the working state; reset and
failed token execution clear pending capture and captured state.

The existing worker owns one immediately preceding scoped entry. It matches
the prompt prefix, context, loaded instance, host cache scope and relevant
options, excludes final chunks and decoded rows, and publishes only after a
successful terminal reply has been written and flushed. Errors, cancellation,
images, scope/context changes, reset, expiry and restart retain their existing
invalidation rules. Snapshot data and token history are never persisted,
logged or included in protocol replies.

## Memory accounting

Each attention row pair uses `4 KV heads × 128 dimensions × 2 × 4` = 4,096
payload bytes. With 14 full-attention layers and 35 sliding layers:

```text
snapshot(B) = (14 × B + 35 × min(B, 513)) × 4,096
```

At the last reusable boundary `B = 261,120`, the snapshot payload is
15,047,208,960 bytes. Live KV at the complete 262,144-token configured context
is 15,105,929,216 bytes. A live cache plus the previous and pending snapshots
has a payload bound of 45,200,347,136 bytes, about 42.10 GiB. The artifact's
weight files total 22,568,912,600 bytes; file size is not a process-residency
measurement. Vector/deque metadata, allocator overhead, copies in flight,
matrix work and host processes add to those payload bounds.

The source's checked 32 GiB live-KV guard remains. Capture allocates only rows
already populated at its actual boundary. It does not allocate a maximum-size
snapshot at load time and does not reduce context to satisfy a memory guess.
The complete copy was chosen to preserve existing reset/ownership behavior;
its latency and observed peak memory still require measurement.

## Required verification

Drafted model-free coverage includes complete-token capture; independent
full/sliding copies after eviction and later mutation; cross-instance,
position, layer, shape and finite-value rejection; reset/native-error clearing;
and a CPU scalar full-prefill bit comparison. The generic worker suite covers
scope/context matching, final-chunk boundaries, publication, cancellation,
expiry, images, reset and restart behavior.

Before the shared compile window, the repository policy check,
`cargo +nightly-2026-10-01 fmt -p stark-engine --check` and `git diff --check`
passed. Those checks do not type-check or execute the Rust changes. Engine
compilation and the drafted tests remain pending and must not be reported as
passing until run.

The coordinated artifact run must compare successful greedy and seeded outputs
between the cached candidate, the candidate with reuse disabled and the matching
fresh reference worker. It must assert actual reused counts across growing,
repeated, shorter and diverging prompts and after reset/restart. Equal failures
are not equivalence. Timing must be measured separately with matching prompts,
output budgets and sampling; report copy cost, reused counts and observed memory
without inferring speedup from the implementation.
