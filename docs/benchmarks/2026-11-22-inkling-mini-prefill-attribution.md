# Inkling 2 Mini prefill attribution: MLX-Serve against Stark

Attempt `20261122-INKPREATTR-01`, 2026-11-22, about 18:46 to 20:40 local time. A diagnostic timing study on the original checkpoint: where MLX-Serve's prefill lead over Stark comes from, and why Stark's worker is slower than its own engine diagnostic. No code changed. Record in the [experiment ledger](../model-experiments/README.md).

**Findings.**

- Engine against engine, almost the whole gap is per-token work, not attention growth. With fits measured in the same window, the linear term accounts for about 2.3 s of the gap at 8,800 tokens and the quadratic term for about 0.15 s.
- Stark's worker is about 19 percent slower than its engine diagnostic. About half of that is draft work inside prefill and about half is GPU clock state: the engine diagnostic's uninstrumented pass runs on a GPU that has just been idle, while the worker runs requests back to back. Snapshot, workspace resizing and host time add almost nothing.
- MLX-Serve runs this checkpoint with no draft head, so it does no draft work in prefill.

## Identity and conditions

| Field | Value |
| --- | --- |
| Attempt ID / date | `20261122-INKPREATTR-01` / 2026-11-22 |
| Kind | Performance diagnosis (attribution), no candidate |
| Result | Completed; attributions marked measured or inferred below |
| Model | Original `thinkingmachines/Inkling-2-Mini` at `6e7c8248f75265bdd4fbab5273304243b5d8c2cb` in both engines |
| Stark | Worker `stark-generate` `73a0fa2e1a5191e1087f2a7152d72b5e280466e2d8e06a5802c8adc80672e76b` and engine diagnostic `inkling_slice` `a92cc9ee00136e3b8bd2f31381b4e8a971ad82f112001eccc9e28028c07c946a`: the exact fast-prefill build of [INKFASTPRE-01](../model-experiments/2026-11-22-inkling-mini-exact-fast-prefill.md), before the unguarded MXFP8 default |
| MLX-Serve | `v26.11.4-pre-release.1`, binary SHA-256 beginning `05983e1278d6`, launched with the matrix harness's prefill arguments (`--ctx-size 32768 --max-concurrent 1 --no-pld --metrics --warmup-full --log-file off --prefix-cache-entries 0`) plus `--prefill-trace`, and `--prefill-chunk` for the sweep; both flags are in the binary's own flag list |
| Inputs | llmprobe was not installed for this study, so unseen text was generated deterministically from the system word list with a fresh seed per trial. Engine-only runs used the private oracle token files at 8,800 and 16,000 and prefixes of the 16,000 file below that. Oracle tokens against the generated corpus: 8.40 against 8.09 s at 8,800 and 14.43 against 13.63 s at 16,000, within noise. |
| Requests | Output cap 8, zero cached tokens, exactly N prompt tokens. Stark drafting either the engine default (three proposals, source verifier) or off (`mtp_drafts` 0). Stark contexts as the harness chooses: 4,352 up to 4,104 tokens, then 9,216 and 16,384. |
| Trials | At least one discarded warmup per set (two for the paired rounds and the chunk sweep), then three to six measured trials. The engine diagnostic ran `--profile-schedule` with two repeats in the early batch and one in the paired sets; each repeat gives one uninstrumented and one instrumented pass. |
| Host | 128 GiB Apple M5 Max (Mac17,7), macOS 26.7.2 (25G332). AC power, but the 140 W adapter negotiated only 94 W and the battery fell from 81 to 72 percent while reporting "charging". OS thermal state 1 (Fair). Metal traces show the GPU moving between Maximum, Medium and Minimum performance states; even fast runs spent about a third of their time at Medium. Every model or GPU run went through the shared lock. |
| Receipts | Private scripts, receipts and Metal System Traces in the session scratch directory and `runtime/prefill-attribution-2026-11-22/` (ignored); hashes not recorded here |

Identical requests varied by ±10 to 15 percent. Between 19:08 and 19:18 every Stark number was about 1.4 times slower (engine-only 8.40 s at 8,800). The first request after an idle period was consistently faster (MLX-Serve 3.1 to 3.5 s against 3.8 to 4.4 s back to back). Only the time-paired comparisons below should be used.

## Time against length

Medians of six back-to-back MLX-Serve trials (20:20 to 20:36) and three Stark trials in the same window; the engine column is the uninstrumented pass, with the instrumented pass in brackets.

| N | Stark worker, drafting on | Stark worker, drafting off | Stark engine only | MLX-Serve |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 0.937 s | 0.842 s | 0.740 (0.658) s | 0.384 s |
| 2,048 | 1.819 s | 1.615 s | 1.305 (1.234) s | 0.846 s |
| 4,096 | 3.549 s | 3.227 s | 2.752 (3.066) s | 1.614 s |
| 8,800 | 7.749 s | 7.136 s | 6.168 (7.377) s | 3.770 s |
| 16,000 | 13.13 s | 12.11 s | 12.05 (12.39) s | 7.413 s |

Fits of time against prompt length:

| Engine | Per token | Per token squared | At 8,800 tokens |
| --- | ---: | ---: | --- |
| MLX-Serve | 0.382 ms | 5.07 ns | 3.36 s linear plus 0.39 s quadratic; intercept 0.002 s. An earlier set at 18:46 gave 0.390 ms and 4.78 ns. |
| Stark engine only | 0.640 ms | 7.03 ns | |

The worker fits come out with a negative quadratic term because of clock drift and per-request overhead; they are not used.

## Side by side, paired in time

Three rounds of MLX-Serve, then the Stark worker, then the Stark engine, 19:24 to 19:39. Medians of round medians with ranges.

| Tokens | MLX-Serve | Stark, drafting on | Stark, drafting off | Stark engine, uninstrumented | Stark engine, instrumented |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 1.59 (1.59–1.78) s | 3.32 (3.28–3.62) s | 3.08 (3.04–3.33) s | 2.71 (2.58–3.07) s | 3.14 s |
| 8,800 | 3.95 (3.76–4.30) s | 7.43 (7.22–7.78) s | 6.70 (6.59–7.33) s | 6.13 (5.53–6.21) s | 6.71 s |

With 12 s of idle before each 8,800-token request: MLX-Serve 3.12, 3.12, 3.91 and 4.31 s; Stark with drafting 6.27 s median; without 5.45 s median.

Traced at 8,800 tokens, MLX-Serve used 3,238 ms of GPU (8,192-row chunk 2,900 ms, 608-row chunk 338 ms). Stark with drafting off used about 5,315 ms (2,048-row commands 1,154, 1,188, 1,218 and 1,253 ms, then 442 and 59 ms and a 22 ms seed), although the GPU sat at Maximum more of the time in Stark's trace. Per row in large chunks that is 0.354 ms for MLX-Serve against 0.56 to 0.61 ms for Stark, about 1.65 times.

## MLX-Serve details

- **Draft head:** the startup log says `mtp.safetensors` is skipped because the index does not name it, so MLX-Serve runs this checkpoint without drafting work in prefill. A `--no-mtp` control was therefore moot.
- **Chunk size at 8,800 tokens** (two sessions per size in opposite orders, three trials each after two warmups):

| `--prefill-chunk` | Chunks | Back to back, median | First request after idle |
| ---: | ---: | ---: | ---: |
| 1,024 | 9 | 4.36 s (4.34–4.57) | 3.52–3.57 s |
| 2,048 | 5 | 4.14 s (4.05–4.33) | 3.31–3.38 s |
| 4,096 | 3 | 4.00 s (3.87–4.08) | 3.18–3.24 s |
| 8,192 (default) | 2 | 3.85 s (3.80–3.88) | 3.12–3.73 s |
| 16,384 | 1 | about 4.1 s (3.73–4.38, noisy) | 3.09–3.18 s |

  At 4,096 tokens: 1,024-row chunks 1.74 s, 2,048-row chunks 1.63 to 1.67 s, one chunk 1.50 to 1.64 s. Large chunks are worth about 0.3 s (7 percent) to MLX-Serve at 8,800 tokens against Stark's 2,048-row size, and about 0.5 s against 1,024. Applying that to Stark is an inference.
- **Fused kernels:** with `MLX_SERVE_INKLING2_FUSED=0`, 8,800 tokens take 26.95 to 27.04 s (7 times slower) and 4,096 take 4.82 to 4.84 s (3 times). The unfused path is strongly quadratic.
- **Server phase times** (`[prefill-trace]`): chunked forward 363–375 ms at 1,024, 3,787–3,819 ms at 8,800 and 7,349–7,489 ms at 16,000, with final evaluation 3 to 25 ms and last token 14 to 18 ms.
- **Metal System Trace** of one 8,800-token request after idle (3.30 s): prefill window 3,274 ms with 3,238 ms of GPU busy time (98.9 percent) in 1,002 command buffers, host gaps about 27 ms, GPU at Maximum for 2,164 ms and Medium for 1,074 ms. MLX-Serve labels no command buffers and the shader profiler tables were empty, so kernel-level attribution for MLX-Serve was not possible, only per chunk.

## Where Stark's worker spends the extra time against its engine diagnostic

| Item | Evidence | Size at 8,800 tokens |
| --- | --- | --- |
| Draft-tap commits inside prefill | Measured. At 2,048 tokens with the GPU at Maximum, the 2,044-row bulk command took 1,288 ms with drafting against 1,155 ms without (+11.5 percent GPU time). At 8,800, drafting on was slower than off by 0.45 to 0.84 s back to back and 0.75 to 0.8 s after 12 s idle. | About 0.6 s |
| GPU clock state | Inferred. The diagnostic's uninstrumented pass is the first GPU work after a 5 to 15 s model load; its second pass in the same process was 0.53 to 1.2 s slower. Instrumentation itself cost about 0.05 s in an earlier record. With drafting off and matched conditions the worker equals the engine command for command (2,048 tokens: 1,154.6 ms bulk, 49.6 ms tail, 18.6 ms seed against 1,187.5 and 55.2 ms), and idle-preceded worker runs match the engine's first pass (5.44 to 5.46 s at 8,800). | About 0.6 s |
| Host gaps between commands | Measured: about 14 to 15 ms per bulk boundary with drafting on, 1.5 to 2.5 ms off; 73 against 29 ms of idle in prefill | About 0.05 s |
| Seed step and first sample | Measured: 19 to 24 ms GPU plus about 5 ms gap; the diagnostic excludes the seed | About 0.03 s |
| Tail commands | Measured: same schedule as the diagnostic (4 × 2,048, 604, 3, then the seed); the 3-row tail costs 50 to 59 ms in both | 0 |
| Prefix snapshot | Measured, not on the benchmark path (completions set no cache scope): no difference detectable within noise | About 0 |
| Workspace resizing | Measured, not on the benchmark path: no difference detectable at 8,800; 64-token fillers that resized took 20 to 90 ms longer | About 0 |

The engine-only drafting profile (`--profile-schedule-draft`) could not run: the measured `inkling_slice` predates that subcommand and every attempt exited 1.

## Ranked remaining excess against MLX-Serve

The engine-level gap is about 2.2 to 2.4 s at 8,800 tokens; the worker-level gap is about 3.5 s.

1. Per-token expert work (inferred as the largest). NVFP4 dot, routing, quantization and epilogue are about 2.5 s and MXFP8 about 1.1 s of the 5.85 s INKFASTPRE-01 reference profile, while MLX-Serve's whole prefill is 3.24 s of GPU. That the gap is per token is measured (0.640 against 0.382 ms per token).
2. Dense attention projections, about 1.19 s in the reference profile (inferred second; cannot be split from the linear gap without MLX-Serve kernel times).
3. Draft work in the worker's prefill (measured): 0.5 to 0.8 s at 8,800, 0.25 to 0.3 s at 4,096.
4. Chunk size (measured on MLX-Serve, inferred for Stark): about 0.3 s at 8,800.
5. Attention growth (measured fits, attribution inferred): 7.0 against 5.1 ns per token squared, about 0.15 s at 8,800 and about 0.5 s at 16,000.
6. Host gaps, seed and first sample (measured): about 0.05 to 0.1 s.
7. Clock state is not a Stark-specific excess, but it explains why worker and diagnostic numbers taken at different moments disagree by about 0.6 s.

The MXFP8 share has since fallen under [INKFASTPRE-02](../model-experiments/2026-11-22-inkling-mini-unguarded-mxfp8-prefill.md); this study's figures bind to the exact build.

## Limits

- Kernel-level attribution for MLX-Serve was not possible.
- The noisy, power-limited host makes every unpaired number unreliable; only the time-paired tables carry the comparisons.
- The clock-state share is inferred from first-pass and idle-preceded runs, not measured with power metrics.
- Draft work in prefill was not measured on the engine diagnostic.
