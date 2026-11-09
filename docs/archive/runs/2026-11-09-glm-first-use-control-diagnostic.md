# 20261109-GLMMEM-15: tiny first-use Control/Q5 diagnostic

State: source and injected controls pass; the sealed test binary builds and one
corrected tiny native run passes. The initial runner-filter attempt remains
failed. Full GLM execution remains held.
This follows the [first-use audit](2026-11-09-glm-first-use-pressure-audit.md)
and [upstream memory review](2026-11-09-glm-metal-upstream-memory-review.md).

## Fixed scope

One ignored `cfg(test)` control initializes the existing weight library/Packed0
pipeline, allocates a synthetic Control (hidden 256, four streams, 20 Sinkhorn
iterations), encodes `Control::pre`, then encodes a 13×256 Q5_K projection in the
**same unsubmitted graph**. The packed buffer is 2,288 owned bytes. No model,
GGUF or other file is mapped. Two warm graphs repeat the same operations.

The test asserts that both this geometry and the actual first 8,192×4,096 Q5_K
projection choose the same Safe block family and 64-thread dispatch. All
`STARK_GLM53_`, `STARK_HYBRID_` and `STARK_QWEN38_` selectors must be absent.
The driver sets only the explicit test gate and disables backtraces. A one-byte
owner start gate keeps the child before any Metal API until the supervisor has
captured a successful live process/service sample.

The numerical check independently decodes the packed bytes on CPU and uses F64
products on the actual Control output. It verifies the projection, finite
outputs and nonempty input. It does not establish mathematical Control parity.
The weight library includes existing `cfg(test)` shader fixtures; its exact
source/binary is sealed separately from the serving worker. Persistent OS shader
cache state is unknown. This is a pressure discriminator with no timing claim.

## Supervision and source review

The separate supervisor derives from reviewed functional10 source `9860befa…`.
Hard ceilings are 1 GiB each for own RSS, own physical footprint, host wired
growth and positive compiler-service RSS growth; compressor growth must remain
below 256 MiB. Entry/runtime available floors are 64/32 GiB. Swap used must be
zero, with no Swapins/Swapouts growth. Deadline is 60 seconds. Pageouts remain
diagnostic and every performance eligibility flag is false.

Each timestamped sample records every `MTLCompilerService` PID, executable and
RSS, their sum, per-PID peaks and newly seen PIDs. The enforced growth is the sum
of each existing PID's positive RSS increase plus the complete RSS of new PIDs.
Exiting baseline services cannot hide another service's increase. Net growth is
recorded separately. This is sampled service RSS, not physical footprint or a
guarantee against unsampled peaks. Services are never signaled.

Only the exact owned `Popen` can be cancelled, terminated or killed. Constructor
and cleanup failures still drain it. Post-drain host/service observation lasts
one second. Success requires the exact eleven phases, successful native exit,
positive process samples, no violation, drained/joined supervisor and positive
post-drain samples. The production admission policy and mapping defaults do not
change.

Twenty-one injected CPU controls pass, including hard-limit boundaries, global
Pageouts as diagnostic, telemetry/foreign/deadline stop, owner cleanup and the
service accounting case where a baseline 200 MiB service exits while a new
1,100 MiB service appears. Net 900 MiB would fit; positive 1,100 MiB correctly
fails. These injected controls spawn no child and execute no Metal. The actual
native run below separately exercises successful real-child start/drain;
earlier functional10 owner-termination evidence remains separately bound.

## Immutable registration

All paths below are private and ignored under
`runtime/glm-image-source-2026-11-09/`.

| Evidence | SHA-256 |
| --- | --- |
| `first-use15-source-registration01.json` | `2e4fd0d4dd69aeabcb0f3ecc41e6407be539431f585679132a45d7c265679138` |
| 1,267-file source graph | `39715e1e825f8291d5fa2974b3ccdb94019f8177a30e041ddc5c14bfb5cd05be` |
| `first-use15-native-review01.patch` | `7309edbe083879bb636e94a2e62e9804497b1b41264cbb08390cacd0efd9bc27` |
| `first-use15-supervisor-review02.patch` | `cc7236acd2a3641e466ce08429a1d4e65ab998d08e4bf14855c728b14ddb94c6` |
| `first_use_guard15.py` | `125d1a4084290aaf1e4f0d35463ab314b5c5b71a44dd4940790d7c2736ab0dc5` |
| `run_first_use15.py` | `d5f6a29a83147810884df56ae702418fb3a161b8d23bbad531d720c41a085621` |
| `first-use15-injected-cpu-receipt02.json` | `876c72e95c74d02ee78e969f4db5ae0ae1f6f20aab59318172d0c5a489c6cbc7` |

Package formatting, repository policy and `git diff --check` pass. The proposed
build reuses only this checkout's own `target-bounded09-engine`, after preserving
the old sealed `bcea1d7a…` binary through an APFS clone/hash. It uses two jobs,
locked/offline dependencies and `--no-run`, with an 8 GiB disk reserve. No
cross-checkout target reuse or full model execution is authorized by this record.

## Build and invalid runner attempt

The approved same-checkout build passed in 89.36 seconds with two jobs and
`--no-run`. All 1,267 registered files remained byte-identical. The new binary
is `8e7abaeeca0f052d56ae0e0e9afee04099379cb4b6ec29c00662fc4d65653afa`;
build receipt `57d883fd80fefba8f215219b794bb19fce807f5519a284db831029d5238d2eba`
and build seal `2157e54902b5b07cb2f39257510f81bed7936eaf970cef3fcfe6ceb89552abf2`.
The old binary remains archived and hash-verified.

The first driver omitted the `stark_engine::` crate prefix from the exact test
filter. It ran zero tests and produced zero phase markers. The driver correctly
failed its phase contract; its owned child exited and supervisor drained/joined.
No Metal phase ran. Failed receipt
`44cd32a2708f1492f242bd2be1e0be34d60035fa6b789210fc9467a72c7ca0ca`
is retained.

New driver `run_first_use15_v2.py` adds the crate prefix, an exact-one `--list`
metadata preflight and a separate execution-registration binding/receipt path.
Listing passed with exactly one identifier; no test closure ran during listing.
All original graph files, the original build seal and native binary are unchanged.
The new execution registration is
`5ea8d20344939292848433f516c06ec666a2ee9e22b180a52a25a86ceb93343e`,
driver `c7b5bcc5e1eff710b570cca8746e69131b9858a8d6723d1de0432ba7ee57307f`,
and list receipt `d9a2105a04bc59e78a2631962fd4067683e7cf282af1eeb5954c9fcff2635c60`.
No new Rust build or supervisor-policy change was involved.

## Corrected one-run result

After independent review and a distinct one-run grant, all eleven expected
phases passed, including first graph completion, both warm repetitions, the F64
projection checks and test-buffer owner drain. Native exit was zero. Receipt
`first-use15-native-receipt02.json` hashes to
`f9706505aec9701d27555ff0dfc7856b809b323a496e669b73acf54633b8b058`.

| Sampled observation | Bytes |
| --- | ---: |
| Peak own RSS | 19,677,184 |
| Peak own physical footprint | 8,946,216 |
| Peak host wired growth | 475,136 |
| Peak total compiler-service RSS | 262,701,056 |
| Peak positive compiler-service RSS growth | 59,310,080 |
| Minimum available host memory | 106,102,259,712 |
| Peak native Metal allocated bytes | 1,867,776 |
| Native Metal bytes after test-buffer owner drain | 720,896 |

There were two positive process samples and four post-drain samples, with eleven
compiler-service PIDs observed in total, including one new PID. Pageouts,
Swapins, Swapouts, swap-used and compressor growth were all zero. No hard limit
or foreign-worker guard fired; child and supervisor drained/joined, and the
exclusive slot was released immediately. Executor/PSO singletons remained
process-owned at the test-buffer drain marker.

The first graph completed at 0.653 seconds from the test's phase clock, including
the owner start gate. These phase intervals are diagnostic observations, not a
performance benchmark. Persistent shader-cache state is unknown and two process
samples cannot bound brief unsampled compiler peaks. This no-file first-use path
did not reproduce pressure. Cold/non-test source, GGUF-backed views, later
formats/operators and full-model behavior remain unresolved. No full-model
retry, default change, cache qualification or performance clearance follows.
