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
actual head execution/speed was unqualified at this original run. The continuation
below updates its cache and speed evidence while retaining a failed numerical
head gate. The two exact affine sources
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

## Official BF16 head: ordinary MTP0 continuation

This section supersedes the earlier pending actual-head cache and short-speed
claims for the exact official Qwen3.5-0.8B derivative. It does not supersede the
headless table or qualify the entire model family. Source `e4144be5`, worker
`1e592b4816b159f230a64d71555dac842a0dcd7d59e68169f95a4e67a908add1`,
image payload `31934dd3d9a4121f91b48519475ba51b7ef926ace338d43a833899c40ead98b4`
and its exact text control are bound in the [continuation record](../archive/runs/2026-11-09-qwen-remaining-image-text-parity.md#final-qualification-continuation).
The head remains loaded; MTP0 now selects the ordinary optimized target stream
and target-only snapshot. Positive draft depths keep predictor sessions, with
exact depth/graph misses on a representation switch. The prior predictor-session
MTP0 run was about half as fast; it remains a separately recorded failed attempt.

All timings use the registered context 2,048, 16-token input, greedy seed 42,
64-output budget and ABBA loads. Valid EOS produces 62 output tokens. One warmup
and three measured requests per load give six measured requests per arm/depth.
Loading is excluded; the worker's prefill duration is a first-token proxy.

| Matched comparison | Baseline decode tok/s | Candidate decode tok/s | Change | Baseline / candidate first-token proxy ms | Baseline / candidate request ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Image-loaded headless → headed, MTP0 | 299.031 | 294.968 | −1.36% | 10.225 / 9.540 | 222.159 / 223.176 |
| Headed text-only → image-loaded, MTP0 | 299.377 | 299.810 | +0.14% | 10.481 / 10.113 | 222.205 / 220.910 |
| Headed text-only → image-loaded, MTP3 | 149.586 | 153.860 | +2.86% | 12.175 / 11.062 | 428.730 / 414.884 |

Every strict five-percent median gate passes. Decode ranges remain broad, with
registered variability bounds of 12.89%, 16.93% and 10.90%, respectively. These
quick controls establish bounded parity and removal of inactive-predictor cost;
they do not establish a speedup. MTP3 proposes 468 and accepts 222 drafts in
each loading arm's six measured requests. MTP0 proposes zero. Same-mode output
hashes match across arms; greedy MTP0/MTP3 hashes also happen to match on this
workload. MTP3 is slower than ordinary MTP0 on this small model.

The loading-mode run records guarded startup separately: 0.422–0.465 seconds
for its text controls and 6.388–6.585 seconds for admitted image artifacts. Peak sampled
RSS across its four loads is 2,126,823,424 bytes. All measured paging deltas are
zero and thermal screens are eligible. A hard preflight and continuous watch
reject foreign model workers; every accepted load has no competing worker and
no watchdog stop. An earlier contaminated partial attempt is preserved and
excluded, rather than pooled into these measurements.

Actual lifecycle checks pass **34/34 image-loaded** and **28/28 text-only**:
nonperiodic 1,792-token prompts, seeded MTP0 and MTP3 fresh/cached equality,
0/1/2/3 switches, exact 1,024-row reuse, scope/context/reset/error/cancellation,
and real image zero-MTP execution followed by fresh/repeated text. A native tiny
hybrid test additionally exercises multi-token ordinary-stream greedy/seeded
snapshot restoration. These are bounded cases, not all contexts or workloads.

Additional controls on the same worker pass **48/48 image-loaded** and
**26/26 text-only** cases: MTP0/MTP3 shrinking/growing/divergent prefixes,
1,024/1,025/1,026-token suffix guards, and primed text prefixes followed by
failed/cancelled/successful images. Each following text request is fresh and
can cache again. Paging remains zero; these are correctness cases without a
new timing claim.

The independent pinned MLX common-input head replay **fails** its unchanged
normalized-hidden maximum-absolute-error limit: 0.25/0.125/0.140625 exceed
0.0625, despite sub-one-percent relative RMS, passing logit limits and all
top-one matches. A test-only reference FC makes fusion exact, while full head
errors `0.25/0.1875/0.1875` still fail. Common-input local norms and residuals
are exact; attention/MLP arithmetic remains under diagnosis. No candidate
arithmetic is adopted. Cache/speed passes do not close that numerical gate or
imply original-training equivalence.

A further test-only all-head-reference-linear candidate is also rejected:
hidden errors `0.1875/0.25/0.15625` exceed the same limit and stage two's top-1
match fails. This yields no new speed or production qualification.

| Continuation aggregate receipt | SHA-256 |
| --- | --- |
| Headed/headless MTP0 ABBA | `88880038813cdfc7b85eb1cb6fa3ff43910cd89d78d85a221a4c246f8abf580f` |
| Headed loading-mode MTP0/3 ABBA | `a5154274e9e9d4f9830de4c4693460424b667d8f8a194bd99308c52c8f7d9d7c` |
| Image-loaded lifecycle | `4a0ca99e3a1f1954278b6dcd546ba43da9f001aed874e9b3c12961a68063163b` |
| Text-only lifecycle | `3c36e91dc62ec388f492a8ac877f357c1b85fb7541733e785ae52aa97f1af963` |
| Ordinary hybrid toy | `6412ef65e68fb418b406be8decb88723ab93f52004096b24b9db300bbb36431f` |
| Failed independent numerical gates | `3626be445f0ec47d288fb127fc707fabcb25da62daac34beabc13511c8810dc2` |
| Additional image cache/image recovery, 48 cases | `aeb8a3034c56bc34ffdcf172be105906e9f8951d87fd50100c2e5fdd06c72012` |
| Additional text cache geometry, 26 cases | `cadb6fe0a8f910c5ab674f9cce943644431bb092d1efd38ecdc7c969dc24225f` |
| Failed test-only causal FC numerical gates | `e2acf10016bb868fbd674456bf2a9e44399ad8e4f4ecd10c6b76a39e5cb705b9` |
| Rejected test-only all-head-linear numerical gates | `868cbd232975bd209d5c5cc1ad6c46dfc9c1750ef5b73dff3796ea3a018e2b71` |

The corrected affine MoE lifecycle continuation was resource-deferred when an
external installed worker reloaded. A later quiet attempt reaches readiness
but stops on one global Pageout with zero swap and no foreign worker; no cache
case completes and no memory-limit failure is established. The [retained ddalcu
memory plan](../archive/runs/2026-11-09-ddalcu-default-image-memory-plan.md) is
source/header-only and its full run remains held. Official BF16 MoE payloads
are absent; its expert-predictor component test does not qualify full execution.

A separately preregistered functional-only MoE attempt also remains unqualified:
its first 1,792-token prefill crosses the retained 64 GiB available-memory floor
before a cache case completes. Sampled RSS/footprint peak at 36.752/37.054 GB;
Pageouts, swap and compressor growth are zero. All workers close, no retry is
made, and no timing from that functional policy is performance eligible. See
receipt `31b0c7bed7a4362e56669a061b56ba097e54675853ac00bb64b37f3b3ebeab92`
and the continuation record for the complete hard-limit measurements.

The subsequent head-only BF16 sigmoid-materialization candidate improves the
common-input MLP error but still fails complete-head hidden limits
(`0.25/0.125/0.1875`). It supplies no speed or production qualification; gate
receipt `4ed2abf75b92c6b3058fb4e47db2acee99ba5064dc257bfb6e66b7b07af2df0b`.
Frozen-worker source audit rules out ordinary per-token full recurrent-state
histories. Command-heap attribution is pending a tiny operator diagnostic;
dispatch counts alone do not establish the memory stop's cause. The MoE cache
matrix remains open with zero completed cases and no further retry granted.
