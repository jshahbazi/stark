# Remaining Qwen image-loaded text controls

Attempts [QNIP-01/02/03](../archive/runs/2026-11-09-qwen-remaining-image-text-parity.md)
extend the shared-weight fast text path beyond the qualified Jundot 27B head.
Three exact **headless converted artifacts** pass the registered short text-speed
comparison. Recovered complete receipts qualify the BF16 and dense affine 0.8B
headless lifecycle matrices; the MoE lifecycle rerun and complete BF16 predictor
qualification remain open. These measurements do not establish support for every model size,
quantization or source head.

## Matched short comparison

Apple M5 Max, 128 GiB unified memory, macOS 26.7.2 / 25G332. Candidate worker
`4010b86e06e729e7b71c4b352e771672f1136d6be6693650866fcf0211d7e0e5`
was built with Rust 1.98.1, locked dependencies, release, debug information off
and incremental compilation off. Source seal
`88a08d0efd2c9123f5ae388fd71620f59eb8646ec88500a284e948a700566279`
binds the implementation used here; subsequent BF16 expert-head/header changes
are outside this binary and these results.

Both arms use identical decoder payload bytes, canonical ordinary-text
configuration and Legacy numerical policy. The image-loaded arm retains its
admitted reference image graph and tower, sharing immutable raw matrix buffers
with its fast text graph. Text controls are private APFS parameter views of the
same files: vision descriptors are removed while original payload offsets and
opaque trailing bytes remain. They are validation controls, not normal
gapless converter exports. Every decoder transform, dtype, shape and scalar
policy was checked against the exact source's ordinary text plan. A separate
older-worker text timing has not been run for these three artifacts.

Four alternating ABBA loads, one excluded warmup and three measured requests per
load give six measured requests per arm. Input is 16 tokens, context 2,048,
greedy seed 42, MTP0, fresh unscoped requests. Dense output budget is 64; official
BF16 emits 62 before valid EOS. MoE fixes a 16-token budget before timing. Every
measured output hash and actual length matches across its artifact's arms.

| Exact artifact | Text decode tok/s | Images loaded tok/s | Change | Text / image first token ms | Text / image request ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Official Qwen3.5-0.8B BF16, older headless image export | 303.618 | 304.108 | +0.16% | 10.222 / 9.441 | 218.599 / 217.038 |
| MLX Qwen3.5-0.8B 4-bit, headless source | 659.888 | 661.392 | +0.23% | 7.841 / 8.098 | 105.355 / 105.307 |
| MLX Qwen3.5-35B-A3B 4-bit, headless source | 39.988 | 39.960 | −0.07% | 119.318 / 119.016 | 519.989 / 520.113 |

All decode, first-token and request medians pass the strict five-percent loss
gate. The registered bound is the larger of five percent and twice the baseline
relative range; five percent applies to all three. Decode ranges are respectively
302.144–304.179 / 303.851–304.359, 659.288–661.425 / 660.943–662.691, and
39.967–40.027 / 39.932–39.988 tok/s. This quick workload does not establish a
speedup or long-context/sustained performance.

The GPU was exclusive by agent coordination; every measured row has zero
pageout, swap-in and swap-out delta and eligible thermal status. Loading/startup
is excluded and was not separately timed. Later global free-disk loss stopped
all subsequent native work; it does not supply missing cache or MTP evidence.

## Storage and correctness boundaries

Actual engine tests load each complete image artifact and create independent
text/image states. Shared matrix bytes are 1,503,789,056 for BF16, 422,940,672 for
dense 4-bit, and 19,495,956,480 for affine MoE. MoE's shared expert-bank subset is
18,119,393,280 bytes; it is not counted again. All three have zero private matrix
copies under this policy. These are parameter allocation counters, not process
RSS or request/scratch/kernel memory. Raw storage remains shared; profile-specific
controls and mutable state remain independent.

Three canonical configuration/payload checks, three actual loaded state/storage
checks, four existing predictor-contract tests and four loader sharing/isolation
tests pass on sealed engine executable
`433b81169a9268d21f29636c503a12c02667347e81949c7b131e26e09282169e`.
BF16's initial cache matrix passes eleven cases, then an invalid test expectation
requests reuse across a changed scope. The worker correctly reports a miss.
That recipe is preserved; its corrected full rerun was pending at that stage.

Subsequent recovered version-four receipts pass 30/30 cases for each exact
headless 0.8B artifact on the same frozen worker, including real-image
transitions and the two-suffix-token boundary guard. The affine MoE matrix has
no completed version-four receipt. See the [recovery record](../archive/runs/2026-11-09-qwen-remaining-image-text-parity.md#crash-recovery-and-bounded-continuation)
for receipt hashes and separate actual BF16-head results. Those later results
do not change the binary, workload or measurements in the table above.

The official BF16 source contains 15 predictor tensors, so the older headless
export is **not** proof of MTP absence. BF16 predictor retention, raw norm-gain
policy, schema upgrade and strict native contracts are implemented separately;
actual head execution/speed is still unqualified. The two exact affine sources
contain no predictor tensors despite a config declaration. Official BF16 MoE
has 785 predictor tensors, including per-expert matrices; its full payload is
absent and is a separate blocker.

## Source-bound receipts

Private files stay under ignored runtime storage. Published hashes identify
aggregate records without exposing prompts, completions or token vectors.

| Record | SHA-256 |
| --- | --- |
| Exact source/control derivation | `d4124cef744a5ae535e8c7b9924f756d351590c5273efe552655ba949e205afd` |
| BF16 short timing | `16250bce8e0fe9367d9cb5d6facfba657c5ac769ad0f9b46c5517deff176140d` |
| Dense affine short timing | `1026dcde59e85d02954468e8ed09fb6cd36e7ef28ab107959eae84ad351ee704` |
| Affine MoE short timing | `b43b7a541480db5f5bd2e803a7a5355b19a8f42f6525f623c8a167db4a064964` |
| First BF16 timing recipe, rejected valid EOS length assumption | `5b6f02ac6a067887e6071a8f9f6531aa65b063e9aa82e9f60bb4699c76afc2f2` |
| First BF16 cache recipe, invalid scope expectation after eleven passing cases | `e01781db132c46f92486887f0cf60f657f4c38d6a71eaa8153cbaddc7e2d576f` |

Artifact payload hashes: BF16
`7c5960e7225b3b36844fa397a85c5e20d281d732e382475062f552c0737c79d4`,
dense affine
`7e127dfb8c78f22b94edd35e52717f50e3d70398fe3668abd2290d088b5b1343`,
affine MoE
`2f762b2629aaefc2fea6eb37ff6caaa4747fdc23a040e36f331fa4a056c6cad7`.

No result here supersedes the separate qualified 27B or retained Lily records.
Eight-bit dense, other sizes, BF16 heads, official BF16 MoE, actual ddalcu,
installed-release admission and complete lifecycle matrices retain their own
qualification gates.
