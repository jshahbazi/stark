# AngelSlim Hy4-flash Q3_K multi-query MLA prefill

## 20261125-HY4MLA-01 · Share contiguous latent cache tiles across queries

The [final resident prefill profile](2026-11-25-hy4-decode-cache-construction.md#20261125-hy4nextn-05--remaining-4k-prefill-budget)
measured 1.880 s for MLA inside an 8.251 s 4K request. The existing path
often reloads a contiguous latent tile for several neighboring queries.
The opt-in Metal 4 kernel loads it for 16 query rows at once and keeps a
64-key tile on that shared path. The Q and probability operands are F16
with F32 scores, running softmax and output. Each row keeps its original
absolute causal or sliding window and attention sink.

The kernel applies to all sliding blocks and to the initial indexed
rows whose selector already includes every causal key. On this artifact
the latter ends at position 2,047. Sparse rows after that boundary
use the existing selected-cell kernel. IndexPool construction and
selection still run and keep the same history. Target decode and
NextN verification continue with their original kernels. The new file
records the pinned llama.cpp layout and MIT provenance.

The [pinned Q3_K artifact and original projector](2026-11-25-hy4-native-q3-k.md#artifact-and-runtime-identities)
ran on the same 128 GiB M5 Max at 8,192 context, with a 40 GB arena,
a single 4,096-row prefill chunk, grouped F16 routed projections,
packed dense projections, resident embeddings, retention and depth-two
greedy NextN. The three arms used the same binary, zero prefix reuse
and stage sampling off. All six outputs matched all 64 llama.cpp
reference IDs and accepted 41 of 43 drafts.

| Arm in run order | First prefill tok/s | Warm prefill tok/s | First decode tok/s | Warm decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| Multi-query first | 534.83 | 484.70 | 28.64 | 41.61 |
| Control | 472.19 | 423.85 | 28.38 | 40.66 |
| Multi-query last | 507.35 | 447.36 | 28.12 | 39.32 |

First-use prefill improved 7–13% and warm prefill 6–14% against the
intervening control. A separate stage sample covered all 42 blocks and
timed MLA at 0.910 s. That is 0.970 s below the earlier 1.880 s profile
on the preceding build. Other stages moved between those instrumented
runs. Use the uninstrumented table to judge end-to-end throughput.
With this MLA split, routed experts and dense projections account for
the largest remaining prefill time.

Stark base `296b16e64d90a6bc4bba2175eb72d54b5ac01415`. The source patch
SHA-256 is `401bfaded6f8efeb2a3c995b0d84f3f161929f38dc5901e6a21671e2fdec1a33`.
The full Metal source passed the offline Metal 4 syntax check. Locked
offline Rust 1.98.1 release binary SHA-256
`21b44b54d6d0a296466f76ca2dfebb04a832bc9058bb23e2b3603284d264e744`.
Harness SHA-256 `2f70e4874bc67ed379bb82a6799c046227533980038dd141bf7c396262ae75da`.

| Receipt | SHA-256 |
| --- | --- |
| Multi-query first | `6487fd2331699ce52cc4ee3d3aed64e598049ecddf40441db533cde55789f420` |
| Control | `2fb98e81f1328f15c576aaba8bbb76ce5f72e119846033dce823a3664caf58ef` |
| Multi-query last | `67e0f2f3ef936d7c42216b671d068a5dda29da382d025ae95a035dbfab65c00d` |
| Multi-query stage sample | `b964aeebbbc47c22236c0f5b801011d882cda69dc708047823ab31bff8144e71` |

## Final validation

The same binary agreed on all 13 natural-text top-1 rows, with six actual
next tokens correct in each arm. It passed both real image cases and
exact text-prefix hits before and after the image and cancellation
transitions. At a 34,816-position capacity, depth-two generation reused
8,192 tokens and matched the fresh continuation exactly. The changed
request context reused zero tokens and matched the same continuation.
Keep `STARK_HY4_PREFILL_TENSOR_MLA=1` opt-in. The standard natural-text
quality result and requested throughput targets remain open.

| Validation receipt | SHA-256 |
| --- | --- |
| Natural-text top-1 | `a760a6aae76e6b12c619470b98114b2c425caf70cc59d56d333f613e5c1bd2e8` |
| Images and cancellation | `e60fe22ea41aea4555d625577f5e641981b77459653425c9b407c3ba6cc643a7` |
| Long prefix | `a71d587a5f033fb7b206c180df15846e0982c5e7252a30ea71626cae8a2de5f2` |
