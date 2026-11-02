# GLM-5.3 MTP Fixed-Prefix Cache v7c

Date: 2026-11-02. Status: **bounded component feasibility only**. V7c measured one unscreened proof per fixed-prefix mode. It did not execute speculative generation or a two-token verifier, and it does not establish throughput gain, runtime admission, or production readiness. The experiment added no runtime promotion, cap change, or broader cache policy.

## Candidate and source lineage

V7c is a test-only continuation of v7b. V7a was never compiled or model-run. V7b's no-run build failed with `E0308`/`E0277` because quantization-kind metadata used `usize` where the engine expected `u32`; v7b had no model run. V7c changed the `GEOMETRIES` kind type to `u32`. It retained the v7b driver and cache policy.

The candidate passed 98 model-free parser fixtures (9 accepted and 89 rejected forms), nine independent arithmetic-reference cases, formatting, and three serialized native CPU units covering planner admission, full-F32-bit comparison, and TLS poll-policy restoration. The release `--no-run` build passed on `nightly-2026-10-01` with two build jobs in 103.055 seconds. Compiler output was processed in capped memory; raw compiler streams were not retained. These checks validate the test probe and its bounded plan, not a production feature.

## One-step component observations

Both model attempts used the same three-case corpus with 59 prompt rows at context 256 and up to eight teacher-forced continuation tokens per case. Each had 24 comparisons, 23 agreements, identical continuation hashes, zero EOS cases, and zero rollback calls. The 23/24 rate is an observed teacher-forced comparison result; it does not establish generation-time acceptance. Both attempts passed initial and post-preload alignment, cached/uncached full-row parity, ownership-release, source, binary, metadata, weight, and memory guards. Each mode was a single unscreened proof, so there is no host-screened performance qualification.

| Fixed FFN prefix | Head wall (ms) | Head GPU (ms) | Mapped FFN wall (ms) | Mapped FFN GPU (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 0 experts | 17.067811 | 2.574248 | 13.999174 | 1.337168 |
| 192 experts | 8.121092 | 2.517950 | 5.429328 | 1.296038 |

These are bounded one-step components from one proof per mode. The 192-expert run used `fixed_prefix_single_ffn_blocking` with a blocking final wait and explicit FFN concurrency `1`; packed Fast Math was disabled. It does not measure speculative acceptance in generation, verifier cost, catch-up, rollback, or coordination.

## Cache and memory bounds

The prefix-192 run prebound 576 views, registering 1,758,461,952 bytes; preload took 487.31025 ms. The warm profile recorded 375 cached hits among 576 plane references, leaving 201 ephemeral references, and no fully cached heads. Its warm cached-expert histogram across bins 0–8 was `[0,0,1,0,7,4,9,3,0]`. The static cache grew from 609 entries / 6,573,162,496 bytes to 1,185 entries / 8,331,624,448 bytes, leaving 258,310,144 bytes of the existing 8 GiB static reserve. The cap remained 109,676,289,225 bytes. These observations do not change the cap or admit a broader cache policy.

The prefix-192 proof reported 48 selected calls and 48 owner-release checks, 39 post-preload cache owner-release checks, and zero ephemeral selected live bytes at return. Cached views remained retained and accounted for. Its sampled Metal allocation peak was 107,769,348,096 bytes. The guard's projected production-overlap reservation was 109,304,375,616 bytes, including a 536,870,912-byte overlap allowance, leaving a reported 371,913,609-byte margin under the recorded cap. This is a guarded projection, not runtime memory admission. Exact gauges and counts are in `cache-v7c.json`.

## Break-even boundary

For a one-draft cycle, a necessary condition is `H + V2 + C < (1 + p) × D`, where `p` is draft acceptance, `D` is matched target time, `H` is draft-head time, `V2` is two-token verification time, and `C` is unmeasured coordination. Using observed teacher-forced agreement `23/24` as an illustrative proxy for `p`, with rounded same-mode `D=31.215 ms` and `H=8.121 ms`, gives a necessary `V2 + C < 53.008375 ms` budget. The separately measured T2 component is approximately 63.98 ms wall / 56.26 ms GPU, so its wall figure exceeds that illustrative budget by about 10.972 ms before coordination.

The T2 measurement has an unmatched span, run, and recipe. The arithmetic is only a necessary budget comparison; it is not a composed benchmark, throughput estimate, or prediction that the components can be combined at those costs. No MTP integration or speculative speedup claim follows.

## Source and receipt provenance

The nine-file source freeze is based on `379e5c79d1cfae249e947740732b783b1a6fec82`, with freeze receipt SHA-256 `374d508d2d46a7571590d258ac039f0060c4f92a8b44dc5d477475afc7726e3a`. The canonical full-context [source patch](cache-v7c.patch) has SHA-256 `1a6793e39680cad489661b3c912c9a0083649726bf4d31cccd6e2b01c56ba831` (105,425 bytes). It applies with default-context `git apply`, reconstructs all nine frozen source hashes, and reverse-checks cleanly. The separate one-line v7b-to-v7c patch remains private; v7c is represented publicly by the complete base-379 patch.

The sealed test binary hash is `27696e170a8f987f18abbe3b14772a6dfc615dada8de0260388bf8ad5c93ea23`; the driver hash is `39ebd844d75bbb11f6c0c9e0594e66867d170a89e237c940c813566e0d44dac4`; the guarded launcher hash is `72c9ea738375fec289f6e22b260b2ba8561468368263e364e4043041569946e0`. The prefix-0 result/guard hashes are `e2cab58ded7f6a6c69cbe26be1a35777089445dab9cdf8dc72d697602fef5d4e` and `da00cfd191d73a82b39134e5059334c424a923a257d8e2fa010916f4f27a5a0d`; the prefix-192 result/guard hashes are `8bbf0ceb91a88de2290576ff9e733e6b4d9dfac878a195b09285deaf056fa97e` and `257c5b50c34715c642c334523a75064f1a318df6db32ed7bb0629bfdc1a7c187`. The private build, CPU-unit, parser, arithmetic-reference, source-freeze, and patch-proof receipt hashes and all nine source file hashes are recorded in `cache-v7c.json`.

The source freeze records llama.cpp revision `1b6935b227b9f9aa89deea8016c0e208f547446f` and vLLM revision `3b6eeb060621eba88deb74a146039291a1afbc42` as pinned architecture references. Guarded metadata and prior GGUF content hashes matched before and after both attempts; the guards used the prior content receipt with size, modification-time, and inode checks, not a fresh weight rehash. Runtime prompts, responses, token IDs, vectors, logits, probability arrays, and raw process output were not persisted. The public patch contains source and authored fixed test fixtures only.
