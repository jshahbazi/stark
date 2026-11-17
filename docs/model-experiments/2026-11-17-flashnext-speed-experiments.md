# Flash Next prefill and decode experiments

**November 17, 2026 · Lily Q4-XL on a physical M5 Max, 128 GiB**

These attempts follow the [installed-worker speed review](../benchmarks/2026-11-17-flashnext-speed-review.md). The baseline is the 4,096-token prefill grid and the existing grouped MoE and MTP verifier kernels. One sealed experimental worker runs the baseline and changes one option at a time. Runtime flags are private experimental controls. The production defaults are preserved.

| Attempt | Candidate | Pre-run correctness and speed criteria |
| --- | --- | --- |
| 20261117-FLASHSPEED-02 | Set `STARK_LILY_PREFILL_CHUNK=1024` or `2048` at worker launch. The engine and worker share the chosen boundary. | Confirm reported geometry, exact reuse counts and same-grid cold/resumed output identity. Compare cold prefill and growing follow-up first-token time at each length. Treat a cross-grid output change separately from cache correctness. |
| 20261117-FLASHSPEED-03 | Set `STARK_FLASHNEXT_PREFILL_GROUPED_SWIGLU=1`. Fuse routed Q4 gate/up and SwiGLU in the grouped prefill tile. | Require exact BF16 intermediate/activation component behavior and same full-model emitted output at matched settings. Measure cold prefill and observe the rerouting threshold. |
| 20261117-FLASHSPEED-04 | Set `STARK_LILY_VERIFY_HC_DOWN_REUSE=1`. Reuse Q8 HC down weights across exact verifier rows. | Exact component intermediate and output bytes. Compare equal-output, equal MTP-work generations at short and long contexts. Record component timing separately. |
| 20261117-FLASHSPEED-05 | Instrument full, partial, rejected, no-draft and interrupted MTP round finishes to inform a GDN state replay design. | The profile must contain aggregate counts/time only. Assess exposure and future rollback design independently of a speed claim. |

Repeat timing rows with a single model resident. Exclude explicit warmups. Retain host observations and variation alongside group medians. Small net effects or effects within the observed variation need a separate paired confirmation before any default changes. A fresh input uses an unscoped request. A reuse input follows its identical scoped prefix with the same model, context size and MTP setting. Full-model runs require a complete artifact and sufficient headroom for one worker. The installed service is stopped only for that campaign and restarted afterward.

## Source and component checks

The experimental native source is `3e956d960c631beacf573053a6a2fa0ba0c99b3e` on Stark main `2cc7345369d3ae34d7f293b2a6ac20db87bc21cb`, with vendor revision `e22d58fa7cc488493fb1b5d810a198bd29730aeb`. The native release executable SHA-256 is `bdf851bedf895ab09859cbdc40cab583a9976ddaf06f01d98ad3872dcd8f7dbe`. Built with Rust 1.98.1, locked dependencies, Xcode 26.4.1 (17E202), macOS 26.7.2 (25G332), on AC power. See [vendor provenance](../../vendor/lily-core/UPSTREAM.md).

All full-model arms load `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl@0f94842e3682b49acace4daeb4cf2e65a6f94f83` with the vision tower and draft head present. The image manifest binds 55 weight files, 106,905,155,192 bytes, to artifact SHA-256 `18c7bcc1e97d1fefdd361472c2daa59d81d89fcd6f3a61983d995a032b44bfb4`. Its config hash is `2a91155be52ebbb0f6dff20e61dcd8ae7484154a56b4afe067246c59da46f899`. The native retained-image loader validates the manifest at startup. The runner checks weight and metadata file identities before and after each arm. It excludes the generated image digest cache, which native startup intentionally rewrites. No image was sent in the kernel screening.

- **Grouped MoE.** The physical Metal test matched all 276,224 output bytes in six cases, including both 32×4 and 64×4 tiles, model geometry, partial row tiles, expert offsets and output guards. The candidate preserves both BF16 projection roundings. It avoids writing the two projection buffers and retains their allocations for this experiment.
- **HC down.** The physical Metal test covered 128 cases and 699,360 output bytes in each switch setting. All bytes agreed with the reference, including the exact verifier's down, injection, inverse RMS and activation outputs. In a separate synthetic benchmark with 24 distinct weight banks and 16 alternating pairs after warmup, the median paired speedups were 1.066× for two rows (12/16 wins), 1.115× for three (16/16) and 1.107× for four (16/16). The exercise streams 85,094,400 logical weight and normalization bytes per read sequence and isolates the HC down path.

## GDN replay design

The current exact verifier already accepts an absent intermediate-state capture tensor. Its Lily Metal recurrence reads the old F32 column once and writes the final column once. A candidate can write that final column into a separate state buffer and retain the pre-round state without a forward copy. To recover an accepted prefix, replay the same recurrence using stored normalized key, decay and update terms. An initial proof can retain BF16 post-convolution QKV, A and B and call the current kernel over the accepted prefix. A compact version can save about 33 KB per layer per row if the normalized key is shared by the value heads, with exact F32 update terms and unchanged contraction. [Verifier recurrence](../../vendor/lily-core/src/kernels/metal/gdn.metal) · [Host finish path](../../vendor/lily-core/src/qwen4exp/spec.rs)

The pre-round buffer must survive until the output queue drains. A response can be truncated after a full or partial accept and then roll back again. Each replay starts from the same immutable state and applies the accepted inputs plus the pending token. Reusing a `Tensor` clone as a checkpoint would alias the writable storage. The rollback must also preserve convolution windows, n-gram history, KV position and draft-head state. The official affine pack uses its own ValueKey recurrence and requires a separate exactness test.

In two repeats of the 4,096-token sampled prompt, all 96 output tokens matched. Each request had 24 fully accepted, six partially accepted and ten rejected drafted rounds, with one zero-draft tail. The 60% full-accept rate motivates trying a final-state swap. The other 40% of drafted rounds would still need a replay. The aggregate finish counter measured 39–49 ms per request and covers the finish step. The proposed change also targets writes inside verification. No state replay kernel was run.

## Native decode screening

The baseline and HC arms loaded one worker per arm. Both used two exact drafts, temperature 1, top-k 20, top-p 0.95 and seed 42 on identical fixture tokens. After one warmup per context, each arm measured two 96-token completions at each length. Emitted-token and text hashes, finish status, draft counts, accepted counts and verifier rows all matched. Decode rate counts the emitted tokens after the first, divided by native first-to-last-sample time.

| Input tokens | Baseline decode tok/s | HC candidate decode tok/s | Baseline prefill seconds | HC candidate prefill seconds |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 39.97, 44.17 | 90.37, 79.32 | 3.244, 4.158 | 1.753, 1.892 |
| 32,768 | 45.58, 55.33 | 69.86, 68.70 | 28.630, 29.185 | 22.122, 22.738 |

The earlier 4,096-token baseline smoke measured 91.58 tok/s after warmup. The measured baseline profile's 4,096-token GPU prefill span also moved from 1.637 s in its first request to 4.325 s in its second. HC reuse is specific to batched verification and cannot explain the prefill difference. The sequential full-model arms establish bounded output and MTP-work agreement. This timing comparison is inconclusive because of large host drift. The Metal component result supports keeping the default-off prototype for a later counterbalanced run.

The initial decode fixture emitted 26 tokens before EOS at its 96-token cap. That run is functional evidence. The following fixture hit the cap, but the harness first classified a generated `.stark-image-validation.json` rewrite as an artifact change. The guard now excludes that disposable digest file and retains the manifest and source weight checks. Separate unframed cache and prefill probes stopped on immediate EOS. They are excluded from the timing comparison. The corrected harness includes an assistant frame and accepts valid single-sample EOS in prefill-only measurements.

## Growing prompts and cold prefill

The framed cache suite used 40,960 context tokens, greedy draws, MTP disabled and 12 output tokens. It grew a 7,700-token prompt to 8,700 and a 12,288-token prompt to 13,288. Each block captured a prefix, appended 1,000 tokens before the assistant frame, repeated the longer prompt and ran that prompt unscoped. Each grid had one warmup and one measured block at each length, with a separate loaded worker. All 24 measured rows matched their corresponding input and output hashes and generation counts across grids. Each same-grid resumed output also matched the unscoped fresh output.

| Longer input | Chunk | Appended turn cached tokens | Appended turn seconds | Fresh seconds | Identical repeat seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8,700 | 4,096 | 4,096 | 3.184 | 6.015 | 0.499 |
| 8,700 | 2,048 | 6,144 | 1.573 | 5.379 | 0.383 |
| 8,700 | 1,024 | 7,168 | 1.424 | 7.203 | 0.451 |
| 13,288 | 4,096 | 8,192 | 3.761 | 8.307 | 0.788 |
| 13,288 | 2,048 | 10,240 | 1.905 | 8.079 | 0.650 |
| 13,288 | 1,024 | 11,264 | 1.611 | 9.569 | 0.720 |

The 2,048 grid saved 2,048 tokens on both growing turns. Its native first-token time was about half the 4,096 grid's in these observations. The 1,024 grid saved a further 1,024 tokens and took longer on both fresh controls. Exact-repeat boundaries were identical across grids at these lengths. The absolute elapsed values are single observations in sequential runs with background paging. Within each 2,048 block, appending reached the first token in 24–29% of the fresh control time, compared with 45–53% on 4,096. The 2,048 grid is the first candidate for a counterbalanced long-session check with native MTP, natural agent transcripts, the 32K case, and cancellation and image transitions.

The grouped MoE arms used the 4,096-token grid and unscoped greedy text with MTP disabled. All input and output hashes and token counts agreed across arms. After one warmup at each length, the 4,096-token native prefill median was 2.390 s baseline and 2.114 s fused (11.6% lower). At 8,700 tokens it was 4.872 s baseline and 5.133 s fused (5.4% higher). Each median uses two measurements. The longer cell had observed paging and consecutive launches can differ materially on this host. The run supports keeping the exact, default-off prototype. A duration or tile-based dispatch policy requires a repeatable crossover on the same host conditions and separate profiling of the projection pair.

The cache policy still retains the immediately preceding request. The mixed-4/8-bit pack's grouped 64-row tile was reviewed separately. Its existing 32-row allocation bounds fit both maps, and the tile changes the tensor geometry feeding the same official BF16 SwiGLU and combine. That checkpoint needs its own projection and full-logit comparison. Its local text artifact was inspected without execution. Neither mixed-pack path participated in this campaign.

The original signed Stark app was reopened after the campaign and the same Lily model reached Ready. The experiment uses a local release worker and all three selectors are opt-in. [Runner](../archive/experiments/2026-11-17-flashnext-speed/benchmark.py) · [Ordered counts, timings and hashes](../benchmarks/data/flashnext-speed-experiments-2026-11-17.json)
