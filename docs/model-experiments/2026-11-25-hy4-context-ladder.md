# AngelSlim Hy4-flash Q3_K context ladder and decode residency

These experiments use the verified [AngelSlim Q3_K source artifact](2026-11-25-hy4-native-q3-k.md#artifact-and-runtime-identities) and its BF16 projector on an Apple M5 Max with 128 GiB. The starting Stark tree is `717a551be682e60c7c627b780ce18a88ecbcd429`. Rust builds use 1.98.1, release, offline, locked, and `stark-generate --features hy4`. The [preceding 4K work](2026-11-25-hy4-dense-prefill-expert-retention.md) identifies attention projections and routed expert compute as the largest prefill stages and expert reads as the main warm decode regression.

## 20261125-HY4LADDER-01 · Local llama.cpp reference ladder

AngelSlim's [model card](https://huggingface.co/AngelSlim/Hy4-flash-GGUF/blob/main/README.md) reports Q3_K at 743.2 tok/s prefill at 4K and 50.34, 49.47, and 47.37 tok/s decode at 4K, 16K, and 32K, with up to 70 tok/s on code. The card identifies WorkBuddy scenarios. The specific inputs, output caps and aggregation behind those rates are unavailable. The [Mac guide](https://huggingface.co/AngelSlim/Hy4-flash-GGUF/blob/main/README_MAC-Metal.md) describes NextN depth two and recommends a 112 GiB Metal wired limit. This local reference used the machine's existing automatic wired limit, AC power, depth two, greedy sampling, 512 batch, 256 microbatch, F16 KV, the patched llama.cpp server and the original projector. It allocated a 34,816-position window to give each prompt room for decoding.

The first row replays the previous repeated-text fixture. The coding ladder pins a fixed instruction and takes exact token lengths from a source archive at the starting Stark commit. The shorter coding fixture uses the same task with an empty archive. All rows use fresh prompts and disable prefix reuse. Prefill rates divide input tokens by the reported prompt phase. Decode rates divide post-first tokens by reported decode time. Reference and native worker phase boundaries follow their respective engine clocks. These are matched-input diagnostics with one reference observation per row.

| Input | Output cap | Patched llama.cpp prefill tok/s | Patched llama.cpp decode tok/s | Accepted / proposed |
| --- | ---: | ---: | ---: | ---: |
| 4,096 repeated text | 64 | 564.94 | 58.90 | 41 / 42 |
| 4,096 code archive | 192 | 419.35 | 43.75 | 110 / 162 |
| 16,384 code archive | 192 | 371.77 | 42.09 | 113 / 154 |
| 32,768 code archive | 192 | 339.25 | 39.11 | 106 / 167 |
| 81 coding request | 192 | 212.09 | 43.29 | 108 / 165 |

The reference process completed every output cap with zero reused input. The raw-ID fixture hashes, output hashes and counts are bound in the private aggregate receipt. Reference executable SHA-256 `b9aad7086abc83bb351abdfa3a41e3f8fde73529fbe8c4a9006d2ee23de0b9bf`. Harness SHA-256 `bfc5b6cfa3c5f31eb7761c1ca6071c3e3dc38bfae60c18bb04c2586f7ddb25fd`, receipt SHA-256 `3adc30696d703bcd880b1c646e496f997b37662099c8cefaadb155e18e3eddd2`. The repeated-text fixture accepted 98% of drafts and the coding fixtures accepted 63–73%. Keep the code generation target tied to a coding workload and report acceptance alongside its rate.

## 20261125-HY4LADDER-02 · Load-time context sizing

Native context used the shared memory-window fit and a ceiling of 34,816. The plan prices full trunk KV and index caches, pooled index rows and scores, and the NextN history and cache. Fixed allowances cover the peak prefill workspace, draft expert pool and largest supported image. At a 4,096-row prefill chunk, the context-dependent estimate is about 2.65 GB at 34,816 positions. The worker allocates the selected capacity at load and keeps it stable through requests. The 1,048,576-position GGUF limit describes the checkpoint.

Live admission rejected the first explicit 36 GB arena at this context after the reference warmed the file cache. That trial produced no timings. Native loading succeeded with the ordinary 27,996,717,056-byte streamed arena, 4,096-row chunks, grouped F16 routed operands, repeat-demand retention and the BF16 projector. Packed dense projections and cached decode reads were disabled. System memory pressure stayed normal in the observed samples and swap held at 4,342.62 MiB across the native ladder.

| Input | Full output | Prefill tok/s | Decode tok/s | Accepted / proposed | Decode catch-up s | Target host cuts s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 repeated text | 64 | 423.79 | 24.59 | 41 / 43 | 0.301 | 0.842 |
| 16,384 code archive | 192 | 336.85 | 24.05 | 112 / 156 | 1.505 | 1.951 |
| 32,768 code archive | 192 | 332.04 | 20.26 | 111 / 158 | 3.142 | 1.789 |
| 81 coding request | 192 | 72.12 | 34.53 | 117 / 145 | 0.000 | 1.921 |

Every request had zero reuse. The 4K fixture matched all 64 llama.cpp token IDs. The three coding outputs diverged from their corresponding llama.cpp outputs. A standard comparison still requires matched chat rendering and the natural-text quality gate. The 32K request spent 3.142 of 9.429 decode seconds in NextN catch-up, excluding the final partial chunk integrated in the first full draft step. Hypothetically removing those measured 3.142 seconds alone raises the 32K result to about 30.4 tok/s. Reducing the cost of cache construction and reducing routed-expert stalls are both necessary for the target. Moving catch-up into the prefill clock would give a different phase denominator.

Native release SHA-256 `39c1bfd0bb6eb8f7511f709a7cdd272c1e321e8cf798216f9be507c7288f746b`, source diff SHA-256 `72387b0c79b789c182a1c057c07d3e47b092914095bb394204f07d9f3c865e84`, corrected native harness SHA-256 `5d11e594251c84d1a9149e054e2dff7cebe549091bceb02a930beabfc07a6854`, native ladder receipt SHA-256 `374f476a370314f3ae123a69001945bc55d6c33c509ffb297405e1d00a8eaeb2`, 36 GB refusal SHA-256 `ca6f704c53f1ed75a29a913e76ec12b8685004a207b975577b9d3caa816a422e`. The first native harness invocation tried startup before acquiring the run lock and emitted only an exception class. The corrected harness starts the worker inside the lock and emits fixed startup codes.

## 20261125-HY4LADDER-03 · Packed Q8_0 attention projections and decode file caching

Used the native release and 34,816 context above. Every worker ran the 4K fixture twice with the projector loaded and zero reused input. The first request follows process load with the existing system file cache. The repeat warms the same loaded worker. A separate worker was needed for each option. Pressure remained normal and no profiler was enabled.

On the packed projection arm, every trunk Q8_0 attention projection with eligible row geometry reads its original packed weights into SIMD-group fragments. Metal tensor operations use F16 operands, F32 accumulation and F32 output. Indexer F32 weights and all decode projections keep their existing arithmetic. The implementation reuses the fragment coordinates from Stark's Inkling prefill kernel with its pinned MLX MIT provenance. No second weight copy is needed. The cached decode arm let macOS retain file-backed pages for streamed decode reads from the exact same inode with read-ahead disabled on that descriptor. The prompt and initial fill kept their normal I/O path.

| Arm in run order | First prefill tok/s | Warm prefill tok/s | First decode tok/s | Warm decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| Control start | 359.71 | 356.21 | 23.50 | 28.84 |
| Packed Q8_0 attention | 400.07 | 405.51 | 23.16 | 28.29 |
| Decode file caching | 387.33 | 361.74 | 22.07 | 31.33 |
| Control end | 384.41 | 353.95 | 27.57 | 32.76 |

All eight requests matched the 64 reference output IDs and accepted 41 of 43 proposals. Packed projection throughput beat both controls, by 4–11% first and 14–15% warm. Retain it opt-in as `STARK_HY4_PREFILL_DENSE_PACKED=1`. Decode file caching beat the first warm control by about 9% and fell below the end control by about 4%. Its first request was slower than both. Warm host-cut seconds were 0.727, 0.742, 0.504 and 0.402 in run order. The decode result remains within bracket drift. The cached-read switch was removed.

| Receipt | SHA-256 |
| --- | --- |
| Control start | `35328c1ce60e3be2559b53115215f602c84de51133505c39cf1c3fa34f6ba583` |
| Packed | `a4e4f6851f225aa4d7509369c11c4ca450d2b8bb1ccba2a99377f3a9f6d1110b` |
| Cached decode | `92f7dd809c4a09bc06c588ff3b8360684d669ee62f082d87acdc97c7dc759e81` |
| Control end | `443e40a8f7e9e8fb08cd7399686251355f5a7f65ced9d85c6f45fe6ac497c1b8` |

The packed projection agreed with the F32 projection on 13 of 13
teacher-forced natural-text top-1 rows, with 6 observed next tokens correct
in each arm. In the packed path, depth-two generation at a 34,816-position
capacity reused 8,192 input tokens and matched the fresh continuation
exactly. The same continuation under a different request context reused zero
tokens and also matched. These checks used the screening binary and cached
expert reads for both arms. Quality receipt SHA-256
`6e948d14428e2c9130dc7ff1a52fa84274911a5fbf564544aa76eb400949930e`, long-prefix
receipt SHA-256 `54355f8ead66e8630ab6fb9e2dbecce191030fe6ea056bad3058ce94f331786d`.

## 20261125-HY4LADDER-04 · Final build check

The final release binary removed the cached-read candidate. A fresh 4K
request at 34,816 positions with grouped routed half and packed attention
returned all 64 reference IDs, accepted 41 of 43 NextN proposals and
measured 479.18 tok/s prefill and 27.33 tok/s decode. This single observation
checks the delivery build. The bracket above supplies the paired estimate.
With 1,024-row chunks, the same binary passed both real projector inputs,
the exact text-MTP prefix hit, invalid-image and cancellation recovery.
The standard benchmark plan recognizes the artifacts, helpers, requested
capacity and planned 8,800-ID case. It reports the missing pinned llmprobe
checkout and the draft status of the profile.

Final executable SHA-256 `a247da3ad1e4eb1e5988e709734e162eda3914218d2e775466ffb8a04e446b25`.
4K receipt SHA-256 `fdf50b70157436df3ed8cddaa82e7335eec43393e458a28801ae4e6d5a8e2ba0`.
Image and cache receipt SHA-256 `0e9e7440616bcc116322b35211086bf0e548e45948596a323c8d8c0b03998926`.

## Next measured work

The 4K prefill and decode targets remain open. The context and short coding
speeds remain below the requested targets. Test batched NextN cache projections
and resident token embeddings against equal-work decode and draft acceptance,
and direct transfers into NextN history against total prefill time. Test wider
shared-latent MLA in the target verifier with the same sparse cell lists.
Keep the live working-set rejection in the expert-capacity sweep. The 34,816
limit allocates that capacity up front. Request-sized physical growth needs its
own cache-restoration gate. The standard profile remains draft pending a
pinned llmprobe client, matched rendering and drafting, and a full
natural-text quality result.
