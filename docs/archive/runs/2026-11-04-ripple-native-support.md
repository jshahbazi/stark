# Ripple native implementation validation, 2026-11-04

## Source and host

Architecture implementation: `b1de057f`. The bounded dense-view cache is
`4cb209cd`; catalog and inventory display fixes are `a32a9ee3` and `259c1065`.
The source is on `codex/native-ripple`.

The physical test host is an Apple M5 Max MacBook Pro (`Mac17,7`) with
18 CPU cores, 40 GPU cores and 137,438,953,472 bytes of unified memory
(128 GiB), running macOS 26.7.1, build `25G241`. Metal 4 and the Xcode Metal
toolchain were available. Production workers were built separately with
Rust 1.98.1 and locked dependencies; the engine suite used
`nightly-2026-10-01` as required by the repository's custom test harness.

Architecture equations and reference Python sources are pinned to
`ricursiveAI/Ripple-65B-Preview` revision
`e27e171792f303d74c048184e8259ba663cfa0d5`. Native inference executes Rust
and Stark Metal kernels. The publisher's Python runs only in the independent
reference validators. See [provenance](../../../provenance/UPSTREAM.md) and
[current Ripple behavior](../../RIPPLE.md).

## Model-free and small-model checks

| Check | Observed result |
| --- | --- |
| Repository policy | Passed |
| Rust formatting | Passed |
| Converter suite | 5,959 passed, 89 skipped |
| Full physical CPU/Metal engine suite | 767 passed, 78 ignored; seven integration tests and documentation tests passed |
| Generator tests | 112 passed, two ignored |
| Optimized scorer targets | All passed: target counts 1, 25 and 25; the scoring cases occur in both library and binary targets |
| Distribution notice tests | 23 passed |
| Changed Python lint checks | Passed |
| Signed app bundle | Built successfully; includes Ripple's license and the native workers |
| Final cache policy checks | 24 Metal weight tests passed, one ignored; all six Ripple tests passed |
| Swift inventory/controller checks | Five focused tests passed |
| Split-reference regression checks | Four passed |

The complete engine suite and the small F32/BF16/Q8 numerical table below are
bound to `b1de057f`. The final cache-only revision is covered by the targeted
Metal weight/Ripple tests, full-checkpoint readout comparison and signed-worker
checks below.

All six new native Ripple tests passed in the physical Metal suite. They cover
an independent F64 full-forward oracle, unequal hidden/query widths, grouped
heads, value mixing, gates, clipping, sliding eviction, GGUF names and metadata,
reset/ingest, invalid-token recovery and malformed-source rejection. Loading
the released maximum-context metadata does not preallocate that context.

The private source-bound fixtures execute the publisher's unchanged Python
with PyTorch 2.14.1 and Transformers 5.15.1. The publisher's config records
Transformers 4.57.6; the validation runtime version is recorded separately.
Each fixture uses five layers, 17 token positions and a four-position sliding
window, with a global NoPE layer, nonunit norms, nonzero learned value mixes,
explicit embedding scaling and clipping exercised on both signs. Reference
execution is F32, including when stored parameters are BF16 or Q8_0.

| Stored weights | Maximum absolute logit error, native Metal vs reference | Relative L2 error | Top-token agreement |
| --- | ---: | ---: | ---: |
| F32 | 0.0000035763 | 0.0000011685 | 17/17 |
| BF16, promoted for F32 reference execution | 0.0000039935 | 0.0000012331 | 17/17 |
| Q8_0, independently decoded for F32 reference execution | 0.0000046492 | 0.0000010405 | 17/17 |

All three pass the fixed `2e-5 + 2e-5 * abs(reference)` elementwise tolerance.
Reset replay is exact, invalid tokens are rejected and subsequent replay starts
from cleared state in every fixture. The streamed Q8 publisher-layer helper
also passes its small-model self-check.

The Q8 fixture passes end-to-end worker checks: the tokenizer and EOS union,
greedy reference readouts, request isolation, invalid-request recovery,
candidate-score parity, and immediate-consecutive scoring-cache hit and
invalidation rules. The raw optimized scorer's maximum absolute candidate
logit error is `0.0000017434`. Tests use supplemental private tokenizer
metadata and preserve the fixture's model configuration and weight bytes.
The signed app bundle's generator and scorer pass the same worker checks,
including the same maximum candidate-logit error and clean shutdown.

Plain-chat and tool/reasoning-history source templates render with the native
MiniJinja evaluator. Preserving and rendering the template does not qualify
Ripple's structured tool parser or agent-task performance.

## Build identities

Raw production generator SHA-256:
`43f79486ad523136df3cc379d3faca8fe46f8b231698f245bc1a6ac8b9b8fbd9`.

Raw optimized scorer SHA-256:
`7dc273133f25b32abb8a89ee23a7bfbabd9e358cd8031fddbc5e3141f8e68842`.

Native Metal parity example SHA-256:
`67dae58e8ed4b6b4d12a3a00dae478dce9f6052ef9e7d2214498f6496fdde930`.

Those raw hashes identify the initial architecture baseline. Final cached
generator SHA-256:
`222d75eade7e381c169b91601333cc38c17220bb50cc7e8b537a101bb27ea4c6`.
Final cached optimized scorer SHA-256:
`cde06f4c1c10d7e3e8fc4ffd07409cd44210bdc0593f5ffbd915d89573a7a7b9`.
Final cached parity example SHA-256:
`c8c7a5e0fa6e2c8159ad50cf18350fd641dd58c75036e41f36631f7f4b19e5d7`.

App signing changes executable container signatures. The final bundled
generator SHA-256 is
`461de6c37355341f600a2ae79cf540cd6e6c3a7c418abf04ad87aed647e6f53d`;
the final bundled scorer SHA-256 is
`eb96289e287bdfa0f81e6bbf73225468c3aad47dfac5d216c7f98bc1219ee402`.

Private receipts and source fixtures remain under the isolated worktree's
ignored `runtime/` directory, and executable outputs under `target/`. The full
checkpoint is in the original checkout's ignored `models/` directory. Only
aggregate metrics and hashes are published here. Existing upstream and
other-model validation records remain unchanged.

## Full 65B Q8 checkpoint

The two AtomicChat shards total 68,840,113,568 bytes. Their actual SHA-256
hashes match the pinned Hugging Face LFS identities:

| Shard | Bytes | SHA-256 |
| --- | ---: | --- |
| `Ripple-65B-Preview-Q8_0-00001-of-00002.gguf` | 44,906,597,984 | `1fdbb76b6ffcc416b120d74257e2e4412f8b7bb99aac315d9b7088e47adf12b5` |
| `Ripple-65B-Preview-Q8_0-00002-of-00002.gguf` | 23,933,515,584 | `48c6dea463d63c471d7f6d225cd52b1c54f7fbfa56c7f03ae1a7aae3f16fa9a7` |

The prepared configuration SHA-256 is
`b429b1908e2d5857f51b7710d4dd80a6db78e0616cf795ad1bf8eb9eef9c8233`.
The exact recipe and prepared-source identity checks pass. Files were moved
into the checkout's ignored `models/ripple-65b-preview-q8` directory without
copying the weights.

The first full-reference attempt exposed a validator assumption: secondary
GGUF shards omit global architecture metadata. The native loader already
handled that layout. Validator commit `3a4f9b5f` adds complete split-inventory
checks and records its own source hashes. The failed receipt remains private
and unchanged; the separate v2 run is the full reference below.

The independent PyTorch oracle dequantizes one layer at a time and executes
all 80 unchanged publisher decoder layers on CPU, with eight input positions
and F32 arithmetic. Its model-forward stage took 50.785 seconds; content
hashing is additional. The reference helper SHA-256 is
`876ec3679410f269c0f0a5c4562a6b2191f8b26e703c1658571a615b60e0b074`.

| Full checkpoint comparison | Result |
| --- | ---: |
| Vocabulary readouts | 8 × 100,352 |
| Maximum absolute logit error | 0.0002365112 |
| Mean absolute logit error | 0.0000441712 |
| Relative L2 error | 0.0000030137 |
| Top-token agreement | 8/8 |
| Fixed absolute/relative tolerance | `0.001 + 0.001 * abs(reference)` |
| Reset replay, invalid-token rejection and failure recovery | Passed |

Both the initial 8 GiB-view-cache baseline and the final retained-view policy
pass. They produce the same complete native readout hash:
`8e0ec13b55615d6fb046cafe486e5ffcd3f2260c80ca278b02426217341e4405`.
The independent reference readout hash is
`c47d203efdd3f0041dd4a805cddfc16e6cf2b145f01f2702d03ad0fc46ce7e77`.

The final signed generator also passes two elementary real-model reply checks,
an intervening-request replay, invalid-request rejection and exact recovery.
All four successful reply identities match the baseline. This is a bounded
functional smoke check, not a task-quality evaluation.

Peak reported Metal allocations during that check were 68,295,262,208 bytes
(63.605 GiB); peak worker RSS was 153,501,696 bytes. These are distinct
measurements of a file-backed, shared-memory implementation and should not be
summed as a physical-memory total. System-wide swap-out increased by
10,567,680 bytes during the interval; that counter is not attributable to the
worker alone. The full model completed on the 128 GiB host without allocation
failure. Large-context operation and quality remain unqualified.

The [timing and memory report](../../benchmarks/RIPPLE_65B_NATIVE_2026_11_04.md)
records the retained-view policy, observed shared-load rates and failed
quiet-machine admission separately. The longer unscreened run has exact
replays and a two-run warm median of 1.126 decode tok/s; short elementary
responses and two-step diagnostics are reported with their own scopes.

## Installed app check

The final signed development bundle was installed into `~/Applications/Stark.app`
using the existing installer, preserving the tested worker bytes. The prior
app was copied to an ignored local backup before replacement. The prepared
model was imported through **Manage Models > Add from Source > Local Folder**.
Its weights remain in the selected `models/ripple-65b-preview-q8` directory;
registration did not copy them.

The installed app displays the Ripple family, Q8_0 storage, 68.84 GB size and
pinned source revision. Selecting **Run** produced the **Running** row and
**Stop** control. Ripple was left selected and running. No model conversation
content was recorded by the UI check. Full text generation was separately
verified through the identical signed worker as described above.
