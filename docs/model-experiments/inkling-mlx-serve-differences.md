# Inkling 2 Mini: Stark against MLX-Serve, what worked, what did not, what is open

This is a maintained list, not a dated record. It collects every known difference between Stark and MLX-Serve for Inkling 2 Mini, every change Stark made or tried during the 2026-11-22 to 2026-11-23 work, and what is still open, so that nothing has to be rediscovered. The dated records stay the evidence: every entry links its record or names its private receipt. Where this page and a record disagree, the record is right. When an item changes, update its entry here and add the evidence to a dated record.

Conventions:

- **MLX-Serve** means `ddalcu/mlx-serve` at `2fe9be834a601c70ef1a597563026340f0e722fe` (release binary `v26.11.4-pre-release.1`, SHA-256 beginning `05983e1278d6`). Its file and line references are relative to `src/src/` of the pinned source and come from the [source comparison][mlxsrc], which read the source but not the MLX library underneath it (`lib/mlx-src` was not in the download). Statements about MLX's own kernels are marked inferred.
- **Stark** paths are under `crates/stark-engine/src/engine/inkling_slice/` unless given in full.
- **Levels:** *engine* is the `inkling_slice` diagnostic on restored state; *worker* is a `stark-generate` request; *server* is the llmprobe client through the gateway. Figures from different levels are not interchangeable.
- **Status:** adopted (default in the tree), rejected (tested and dropped), in progress (being built or measured), queued (decided, not started), not started (idea only), not applicable.
- **Artifacts:** "original" is the fixed-three-pass `thinkingmachines/Inkling-2-Mini` at `6e7c8248f75265bdd4fbab5273304243b5d8c2cb`. "Stark 8-bit" is Stark's own affine 8-bit dense artifact ([INKAFFINE8-03][a8]). "MLX 8-bit" is the author's published pack `ddalcu/Inkling-2-Mini-MLX-8bit` at `cb7fa052a9bbafbbeb91750edc67046e7bacd4f1`; its quantized tensors are byte-identical to Stark's quantizer output ([INKAFFINE8-02][a8]), but its layout and loader differ.
- **Commits.** The work of 2026-11-22 to 11-23 was committed and pushed on 2026-11-23 in five commits on top of `f1666090`: `61501bf5` (converter, 8-bit artifact), `2be13769` (client drafting defaults), `2a1d72a2` (engine and worker), `bf126020` (benchmark tool), `87b2bb17` (documentation). "Adopted" below means default in that tree unless an entry says it is pending. Nothing is released or installed.
- **Verification of the committed tree** (coordinator, before committing `2a1d72a2`; private receipts in `runtime/commit-check/`, summary SHA-256 prefix `8cb9cd12821508c4`): clean build with no warnings (`inkling_slice` `f47ac2f4ba86…`, `stark-generate` `a7e1fee1c05f…`); 167 worker tests, 8,010 converter tests and 124 benchmark tests passed; the exact reference exact on `g4096-02` and `s4096-00`; drafting on equals off at 73, 4,150, 8,800 and 16,000 tokens on both artifacts; a repeated 8,800-token replay with identical hashes; the worker smoke on both artifacts.

## 1. Where things stand

### Owner decisions

| Date | Decision |
| --- | --- |
| 2026-11-22 | Improvements ship as defaults: no opt-in flags, selectors or off-switches. Bit-exactness with the Ollama reference is not required; validate numerically. Do not be conservative. |
| 2026-11-22 | Unguarded MXFP8 prefill may be the default if it passes self-consistency and a noise-floor comparison (it did, [INKFASTPRE-02][fp2]). |
| 2026-11-22 | Build the engine side of a Stark 8-bit dense artifact ([INKAFFINE8-03][a8]). Re-download and keep the MLX 8-bit pack. |
| 2026-11-22 to 11-23 | Formalize the session's benchmark as the standard Stark benchmark ([`docs/BENCHMARK.md`](../BENCHMARK.md)); the open choices were settled on 2026-11-23 (all-blocks-agree standing, the quality bar, burst reported only, block layout, no image exception, artifact-based arm names). |
| 2026-11-23 | Download and read MLX-Serve's source at the pinned commit for comparison and kernel timing. |
| 2026-11-23 | One user is the goal; the 16-user burst is out of scope. |
| 2026-11-23 | Make the Stark 8-bit artifact the default. Commit the branch. |
| 2026-11-23 | Keep this list. |

### Latest scoreboard (server level)

llmprobe 0.6.18 with the aggregate receipt patch, this 128 GiB M5 Max, reasoning `off` (effort level 0 on both engines; both still think), each engine's default drafting, one invocation per arm in the legacy layout. Run 4 (`runtime/inkling-engine-matrix/20261123-060520-both`, manifest SHA-256 `72a056bf5f4e91b6a512a2574cb129ce8457e53ca7a534c39e145ab692ca4804`) was taken on a quiet, cool host with other agents paused; Stark worker binary beginning `0ba839d4d3be` (`morning1-build`, built at 06:05 on 2026-11-23 from the working tree after the overnight pause; its `inkling_slice` begins `229178093f3f`): short-context verifier attention and the fused prefill router, 2,048-row commands, fixed depth 3. It is **not a standard result**: one repeat, legacy layout. Both Stark arms have natural-text quality results attached and pass the bar (section 7). The run predates the committed tree's depth planner, wide commands, BF16 prefill attention and any change to expert-input quantization.

| Scenario, tok/s | Stark original | Stark 8-bit | MLX original | MLX 8-bit |
| --- | ---: | ---: | ---: | ---: |
| Initial coding | 78.9 | 86.9 | 86.7 | 101.0 |
| Sustained coding | 89.9 | 88.9 | 82.5 | 97.8 |
| Novel prose | 58.4 | 67.8 | 73.3 | 91.6 |
| About 512 tokens | 76.1 | 89.7 | 82.4 | 97.1 |
| About 4K | 112.0 | 115.4 | 89.2 | 107.6 |
| About 8K | 100.1 | 110.8 | 100.1 | 99.8 |
| About 16K | 95.1 | 96.2 | 94.6 | 103.8 |
| Prefill, 8,768 repeated-sentence tokens | 1,634 | 1,780 | 2,214 | 2,220 |
| Prefill, exact 8,800 IDs | 1,879 | 1,892 | 2,574 | 2,486 |

Reading it: on matched 8-bit artifacts Stark leads at about 4K and 8K and trails on short coding, novel prose, 512 and 16K, and on prefill by about 25 percent. MLX-Serve forces decode to 192 tokens while Stark stops where the model stops, so decode rows do not cover identical work. Worker-reported round times at depth 3 in run 4: Stark 8-bit 26.9 ms (coding), 26.6 (sustained), 27.4 (prose), 24.2 (a 27-token prompt); original 29.6, 27.3, 33.4, 27.0. Worker and client rates agreed within 1.5 percent.

History of the server-level scoreboard (tok/s, Stark original / MLX original / MLX 8-bit unless noted; all one repeat):

| Run | Initial coding | Novel prose | About 4K | Exact 8,800 prefill | Notes |
| --- | --- | --- | --- | --- | --- |
| Before this session (another worktree) | 34.8 against 97.6 | | | 389 against 1,898 | Mismatched reasoning modes and a forced length; Stark fast paths not admitted outside 512 to 4,352 tokens. Not an engine comparison. |
| Run 1, 2026-11-22 16:11 ([INKENGINE-01][eng]) | 78.0 / 80.5 / not run | 58.6 / 71.8 / not run | 106.5 / 84.2 / not run | not run | MLX 8-bit shards had been deleted |
| Run 2, 17:53 ([INKENGINE-02][eng]) | 79.7 / 79.3 / 93.3 | 62.0 / 71.7 / 89.0 | 98.2 / 85.1 / 113.1 | 1,194 / 2,236 / 2,225 | Exact fast prefill build |
| Run 3, 23:24 ([INKENGINE-03][eng]) | 75.4 / 81.8 / 90.8 | 64.2 / 72.7 / 88.4 | 108.2 / 85.8 / 102.5 | 1,707 / 2,088 / 1,897 | Stark 8-bit arm refused (`stark_images_not_loaded`); taken while agents were loading the GPU |
| Run 4, 2026-11-23 06:05 ([INKENGINE-04][eng]) | 78.9 / 86.7 / 101.0 (Stark 8-bit 86.9) | 58.4 / 73.3 / 91.6 (67.8) | 112.0 / 89.2 / 107.6 (115.4) | 1,879 / 2,574 / 2,486 (1,892) | Quiet host; table above |

## 2. Every known difference

Status refers to Stark. "Effect" is Stark's measured change where one exists; estimates say who made them.

### 2.1 Speculative decoding policy

| # | MLX-Serve | Stark | Status and effect |
| --- | --- | --- | --- |
| S1 | Drafter proposes 7 tokens per round (`draft.zig:1-8`); a per-round **confidence cut** keeps drafts while the running product of draft confidences is at least `min_sure` = clamp(1.34·√(verify-row cost / plain-step cost), 0.3, 0.8), costs measured live (`plugins/inkling2/plugin.zig:290-342, 425-435`); zero drafts makes a plain step | Fixed depth up to the requested MTP count (3) every round; proposals are not read back before verification so draft and verify overlap | **Rejected** for Stark after trace simulation ([INKROUND-06][round]): the best confidence cut is 0.3 to 3 percent below the best fixed depth on every input, because reading confidences first removes the draft/verify overlap (1 to 3.5 ms per round) and the 3.6 to 4.1 ms draft command runs even when the cut picks zero. Revisit if the draft command becomes much cheaper. |
| S2 | Engine **backoff**: starts with 4 plain rounds, compares each drafted round's value (tokens × measured plain-step time) with its wall time, and skips drafting for a doubling period up to 32 rounds when it loses (`engines/mlx/spec/dflash/policy.zig:169-224`, `engines/mlx/spec/dspark.zig:92-93, 131-150, 208-215`) | No backoff | **Rejected** for Stark ([INKROUND-06][round]): on Stark a drafted round costs about 1.29 times a plain step (20.5 against 15.9 ms at depth 1), so even prose at about 35 percent first-position acceptance gains from drafting (no drafting 56.5 against 71.0 tok/s at depth 1). The backoff's test fires on most rejected rounds and costs 10 to 15 percent. |
| S3 | Depth is a policy outcome, which is why MLX-Serve's tokens per step is low on prose (1.19) and high at long context | Stark's history planner (`depth_planner.rs`: running acceptance and cost estimates, EMA 0.03 and 0.1, hysteresis 1.03, probe one row deeper every 16 rounds, at least one draft) chooses 1 to 3 drafts per round | **Adopted** in `2a1d72a2` (floor 1, capped by the requested count) after the worker smoke and the commit-check gates passed; its tuning for 4K and 8K is **in progress** ([INKROUND-06][round]). Simulated within 1.2 percent of the best fixed depth on every trace; +8 percent on 8-bit prose and +6 percent on original prose against fixed 3; −1.5 to −3 percent at 4K and 8K; level on code. |
| S4 | Verify width up to 7 drafts (8 rows) | `MAX_VERIFY_ROWS = 4` (3 drafts) wired into dense, attention, mix, norm, gate, save and rollback, logits rows and draft commit | **Not started.** Estimate (decode agent, assumption that the 0.88 conditional acceptance holds): 4 drafts about +8 percent and 5 drafts about +12 percent at about 8K, where 74 percent of anchors accept all 3. Several days of kernel work; long-context traffic only. |
| S5 | Verify forward appends nothing; commit appends accepted rows (`kernels.zig:572-611, 579-587`); no rollback | Transactional verifier with exact rollback and snapshot; commit is a separate command (about 0.45 ms per round) | Not applicable as a difference in output. Folding the commit into the next draft command is **queued** (decode agent estimate 0.3 to 0.45 ms per round). |
| S6 | Draft evaluated with a blocking `evalMany` before the verify graph is built (`plugin.zig:395`); no overlap | Verify encoded while the draft runs (overlapped) | Stark advantage; keep. It is also why the confidence cut does not pay on Stark. |
| S7 | Drafter context K/V cached across rounds (`draft.zig:48-62, 184-230`) | Incremental draft cache (`dspark.rs:504-590`) | Not a difference. |
| S8 | Draft head: bundled DSpark drafter with a 3-launch Markov RNN per step | Same DSpark head; INT8 vocabulary bank (440 MiB) built from the BF16 head | Draft command 3.0 to 3.2 ms, almost all the fixed block forward (3 BF16 draft layers about 500 MB plus the INT8 head about 460 MB); each extra proposal row about 0.1 ms ([INKROUND-01][round]). |
| S9 | MLX-Serve does not load the draft head for this checkpoint at all in the attribution session's launch (startup log skips `mtp.safetensors`), so it does no draft work in prefill | Stark drafts by default and commits draft cache rows during prefill | Stark's packed draft commit made the cost small (section 3). See C1 in section 7 for a conflict about its size. |
| S10 | Tokens per step: MLX-Serve 1.18 to 3.43 depending on scenario | Stark 2.0 to 3.31 at fixed depth 3 | Policy outcome, not verify width ([coordinator finding](#7-measurement-rules-learned-and-open-questions)). |

### 2.2 Decode step structure and fusion (one row and verify rows)

| # | MLX-Serve | Stark | Status and effect |
| --- | --- | --- | --- |
| D1 | About 12 to 14 launches per first-pass visit and 17 to 19 per later-pass visit, about 0.75k per token (inferred count, `kernels.zig:985-1002`) | 1,907 launches per single-row token before fusion, 1,368 after INKFUSE-01, 992 after INKFUSE-02 | **Adopted** ([INKFUSE-01 and -02][fuse]): single-row step 22.86 → 19.36 ms (engine, 0.844, 6/6 pairs). Launch count alone is about 25 percent lower in MLX-Serve, not 2 times. |
| D2 | One `pre` kernel per visit: depth-block softmax mix plus weighted RMS norm of all q/k/v inputs (`kernels.zig:190-214`, `pre.metal`) | Mix plus q/k/v norms fused (`ink_verify_mix_qkv_norm`); later-pass controller chain 17 → 5 → 4 launches | **Adopted** ([INKFUSE-01/02][fuse]). |
| D3 | `prep`: k/v short convs, L2 clips, v norm, gate sigmoid, bias row in one launch (`kernels.zig:310-345`) | K/V convolution, normalization and cache write plus Q/R normalization in one launch (`ink_verify_heads_pack_one`) | **Adopted** ([INKFUSE-02][fuse]). |
| D4 | Projections joined: q, gate, r and sink as one; k and v as one on sliding first-pass visits (`plugin.zig:858-860`, `layers.zig:270-287`) | Six attention projections plus two dynamic-conv projections in one launch (`ink_matvec_source_visit`; on 8-bit `ink_a8_matvec_visit`) | **Adopted** ([INKFUSE-02][fuse], [INKAFFINE8-03][a8]). |
| D5 | Output gate folded into the output projection (`oproj`, `kernels.zig:366-378`) | Gate in `ink_attention_gate`, then a separate output projection | **Not started.** Part of the 0.5 to 1.5 ms controller and tail estimate (source-comparison agent, at 5 to 7 µs per dependent launch). |
| D6 | `mid`: attention conv, residual or block boundary, depth mix and MLP norm in one launch | `ink_verify_attention_mix` does the same, with deferred history writes | **Adopted** ([INKFUSE-01][fuse]). |
| D7 | `ups` fuses SwiGLU into the gate/up product; `downs` fuses the down product, weighted pick-order sum, MLP conv and residual (`downs.metal:36-52`) | Separate up, activation quantization, down, `ink_expert_sum`, conv and `ink_after_experts` launches on the single row; the expert sum, MLP conv and residual are one launch since INKFUSE-01 | Partly adopted. The rest **not started** (same 0.5 to 1.5 ms estimate). |
| D8 | Verify rows: 4 to 8 rows read K/V once per KV head (`relsplit` and `relcomb`) | Shared local and global verifier attention were admitted only at position ≥ 1,024; below that every row ran its own attention (288 launches per row) | **Adopted** ([INKROUND-03][round]): batched short-context verifier attention (`verify_short.metal`) whenever position + rows < 1,024. Four-row verify −1.71 ms (8-bit) and −1.60 ms (original), 12/12 pairs each. |
| D9 | Plain steps are pipelined: the next forward is `async_eval`ed and its token resolved at the start of the next tick (`engines/mlx/generate/pipeline.zig:75-90`) | One command per step with commit and wait | **Not started.** Estimate (source-comparison agent) 0.3 to 0.8 ms per plain step. Worker-loop study: host work is about 0.2 ms and the commit about 0.45 ms per round; GPU idle between rounds about 0.6 ms ([INKROUND-05][round]). |
| D10 | Cross-visit fusion not present either | — | **Rejected as not worth building** ([INKFUSE-02][fuse]): race-free only on about 18 local first-pass visits per round; verifier launches measured free. |

### 2.3 Expert products

| # | MLX-Serve | Stark | Status and effect |
| --- | --- | --- | --- |
| E1 | Decode `ups.metal`: one SIMD group per interleaved gate/up row pair, 16 inputs per lane per sweep read as `float(x[k+i])`, nibble and scale decode by bit tricks into half (`ups.metal:26-37`, `moe_fp4_e2m1x2_header.metal:1-4`, `moe_fp4_nv_scale.metal:1-3`); one output row per SIMD group by design (`kernels.zig:41-44`) | `ink_nvfp4_up` / `_vector_up` with byte-pair FP4 coefficient table, K16 F32 accumulation | Stark's multi-row-per-SIMD-group single-row variants **rejected** ([INKROUND-02][round]): six bitwise-identical variants were 6.75 to 11.5 ms against 4.80 ms, register pressure and occupancy. A half-buffer variant was written and never timed. |
| E2 | Verify: `ups`/`downs` widen the grid by rows with no expert deduplication (`kernels.zig:914-953`) | Run-grouped verifier NVFP4 shares weights across rows routed to the same expert | Stark design differs; verify expert stage 11.74 ms at four rows on s73 (20 distinct experts per visit) against a measured 8.29 ms gather ceiling ([INKROUND-01][round]). Shrinking it is the larger decode lever; **not started**. |
| E3 | Prefill NVFP4: `gather_qmm_nax` double-buffered schedule, BM 96 × BN 64 × BK 64 at mean rows per expert ≥ 48 and K ≥ 1,024 (`gather_qmm_nax.zig:28-39`); weight tile dequantized to BF16 in threadgroup memory, next K step's packed words prefetched into registers; activations read from device as `vec4` register fragments (`gather_qmm_nax_header.metal:543-651, 612-640`); MPP 16×32×16 with `relaxed_precision` | `grouped_fast.metal`: 64 × 64 tiles of 32 × 32 SIMD-group fragments, BK 32, vectorized double-buffered decode, BF16 store with fused epilogues | Stark's kernel **adopted** ([INKFASTPRE-01][fp1]). A port of the MLX-Serve design (`grouped_nax.metal`) **rejected**, not faster at BM 64, 96 or 128 ([INKFASTPRE-03][fp3]); left in the tree disabled. Untested differences: register-fragment activation loads, half scale decode, the planner. `relaxed_precision` is set in both; not a difference. |
| E4 | Gate and up in one gathered call over the interleaved bank (N 2,304), down reads the SwiGLU output already sorted; un-sort fused into `sum.metal` with the weighted sum, conv and residual | Same two products; fused epilogues | Not a material difference after INKFASTPRE-01. |
| E5 | Sorted copy of the expert input rows: MLX `argsort` plus `take`, about 300 MB per visit at 8,192 rows (`layers.zig:448-467`) | Run-grouped ingress without a full sorted copy | Not a Stark cost. |
| E6 | — | Expert ceiling: production NVFP4 product = decode plus tensor work back to back (9.2 ms per visit against 4.7 tensor-only and 4.8 decode-only); weights pre-decoded to BF16 in device memory are slower (9.8 ms) | Measured ([INKFASTPRE-04][fp4]). Overlapping decode with tensor work is **not started** (prefill agent: producer and consumer SIMD groups, deeper pipeline or less threadgroup memory). |
| E7 | — | Weight-stationary NVFP4 (`grouped_stationary.metal`): each weight block decoded once per up to 256 rows of an expert | **Rejected at 2,048-row commands** (gate/up 1,368 → 1,683 ms, wall 4.34 → 4.99 s, hash-identical; occupancy) ([INKFASTPRE-04][fp4]). Its test at 8,192-row commands (about 512 rows per expert) is **queued**. |
| E8 | MXFP8 prefill (4 visits) through stock MLX `gather_qmm(mode "mxfp8")` (`gather_qmm_nax.zig:43-44`) | Unguarded tensor MXFP8 since INKFASTPRE-02 (was certified with source-order replay) | **Adopted** ([INKFASTPRE-02][fp2]): 1.169× (4K) and 1.174× (8,800) on the engine. |
| E9 | No NVFP4 guard or replay | NVFP4 per-K-step guard and replay removed | **Adopted** ([INKFASTPRE-01][fp1]): 1.325× at 4K, still bit-exact; the guard never changed a result on nine captures. |

### 2.4 Dense projections

| # | MLX-Serve | Stark | Status and effect |
| --- | --- | --- | --- |
| P1 | Decode at one row: MLX affine `quantized_matmul` (qmv) on the 8-bit pack; BF16 `x @ W.T` on the original (`sdk/bases/blocks/linear.zig:39-75`, `qmatmul.zig:90-195`); verify rows 2 to 6 use a split-K verify kernel (`qmatmul.zig:169-172`) | 8-bit: `affine8.metal`, dequantize inside the unchanged K-lane order; dense 2.90 ms at one row (1,373 MB, 473 GB/s) | Stark near 80 to 90 percent of the measured 513 to 586 GB/s ceilings at one row; no gap expected (inferred). At four rows Stark's 8-bit dense is compute- or latency-bound (4.27 against about 2.4 ms); a four-row kernel is **not started**. |
| P2 | Prefill on the 8-bit pack at 2,048 rows and above: dequantize each weight to a BF16 transient, then MLX's BF16 steel or NAX GEMM (`prefill_dq_gemm.zig:15-31, 68-107`); the 608-row tail uses affine `quantized_matmul`; original uses `x @ W.T` | `dense_fast.metal`: BF16 tensor operands from device, one F32 accumulation, 128 × 64 tiles; the 8-bit artifact keeps a resident BF16 copy for prefill | **Adopted** ([INKFASTPRE-03][fp3]): dense stage about 1,290 → 687 ms at 8,800 tokens, about 14 TFLOPS. The earlier 175 ms FLOP floor was wrong: a decomposition on synthetic operands at 8,192 rows measured about 2.7 ms per projection in production against about 1.8 ms tensor-only and about 2.1 ms loads, so dense runs at about 1.5 times its tensor floor and the lever is overlapping loads with tensor steps ([INKFASTPRE-06][fp4]). Not started. |
| P3 | Projections joined (q, gate, r and sink; k and v on sliding first pass) | Not joined in prefill | **Queued** (prefill agent). Source-comparison estimate for the whole dense gap: 300 to 450 ms of 687 ms. |
| P4 | — | Weight-stationary dense (`ink_dense_stationary`) and three variants with both operands staged in threadgroup memory (128 × 64 BK 64, 256 × 64 BK 32, 256 × 128 BK 32) | All **rejected** ([INKFASTPRE-04][fp4]): stationary 713 against 670 ms; staged 1,123, 1,220 and 1,115 against 967 ms at 8,192-row commands. Operand reuse is not the limit; the coordinator's tile-traffic model was wrong. |

### 2.5 Attention

| # | MLX-Serve | Stark | Status and effect |
| --- | --- | --- | --- |
| A1 | Prefill chunks over 32 rows use `relattn.metal`, a half-precision flash kernel ported from mlx-lm: half q fragments, half K/V staged in threadgroup memory, half probabilities before P·V, F32 scores and accumulator; 32 queries per threadgroup, 16 keys per step; stops at the causal diagonal and starts at the window; no mask tensor (`kernels.zig:420-433`, `relattn.metal:32-129`) | Local: source relative attention with BF16 relative table, now with BF16 query operands; global: MLX NAX-derived kernel, now BF16 with F32 scores and accumulation, query tile 1,024 with causal skip; the exact reference keeps F32 | BF16 query operands in local attention and BF16 global attention **adopted** in `2a1d72a2` ([INKFASTPRE-05][fp4]): natural-text gate passed on eight inputs (accuracy change +0.19 points, 99.28 percent agreement on confident rows); paired 1.018× with a wide range. Source-comparison estimate for the whole attention gap was 250 to 400 ms of 533 ms at 8,800; the measured gain is far smaller. Half-precision (not BF16) staging as in `relattn` is not started. |
| A2 | No mask tensor | Explicit F32 global mask (up to 957 MiB at 32K) | **Adopted**: mask-free global kernel (`global_relative.metal`, MIT derivative of MLX `attention_nax`) bit-identical ([INKFASTPRE-02][fp2]); mask freed on the default path, workspace 1.27 GB → 0.27 GB at 32K ([INKFASTPRE-03][fp3]). |
| A3 | Causal skip | Global mask-free kernel specialized with `do_causal` | **Adopted, no gain** ([INKFASTPRE-04][fp4]): hash-identical, 311 against 301 ms (noise). Global keys are the whole prefix plus the query tile, so little lies past the diagonal. The committed default uses a global query tile of 1,024 with the causal skip, also hash-identical ([INKFASTPRE-05][fp4]). |
| A4 | Decode: MLX `fast::scaled_dot_product_attention` with sinks and an array bias at one row (`kernels.zig:618`) | Source-order sharded attention (32 or 64 shards) | Not expected to be a gap (source-comparison agent, low confidence). |
| A5 | Long-position gain (global layers, position ≥ 8,192) | Single-rounding `ink_long_position_gain` everywhere | **Adopted** after a defect ([INKLONGPOS-01][lp]): the verifier rounded twice and diverged from serial above about 13,000 positions. |

### 2.6 Routing

| # | MLX-Serve | Stark | Status and effect |
| --- | --- | --- | --- |
| R1 | Decode router: MLX F32 matmul on first-pass visits (`kernels.zig:835`); fused into the `norm` kernel on later passes (`kernels.zig:827-834`, `norm.metal:30-36`); stays exact F32 | Native M16 × N8 tensor router on the single row and verifier | A 4-experts-per-SIMD-group F32 matvec replacement **rejected** ([INKROUND-04][round]): −0.06 ms per step, +0.56 ms at four rows (0/12 wins); KL 0.003 to 0.009 against the old router. The instrumented 1.11 ms router figure was a timestamp-split artifact. |
| R2 | Prefill router on first-pass visits: MLX F32 matmul over more than 32 rows, which MLX-Serve's own notes call TF32 on M5 at about 1e-3 (`docs/gotchas/engine-mlx.md:1605-1611`); message-gate projections likewise (`kernels.zig:225-229`) | Source split-K F32 router; `ink_prefill_router_tiles` shares weight fragments across four 16-row tiles with identical arithmetic | Router tiles and fused selection **adopted, no measurable gain** (1.006×, hash-identical) ([INKFASTPRE-04][fp4]). A fixed-split (width-independent) router **rejected** by the gate: 2 of 1,768,704 selections changed, worst row 104.6 percent against an 87.3 percent control maximum ([INKFASTPRE-03][fp3]). A TF32-class prefill router is **not started** (estimate 200 to 400 ms together with expert-input passes, source-comparison agent). |
| R3 | `topk`: sigmoid plus bias, rank by counting, weights sigmoid / sum × scale, one threadgroup (`topk.metal`) | Prefill selection fused into `ink_singleton_route_weights_parallel` (one dispatch for select, LogAddExp, negate, softmax and scale) | **Adopted, no gain** (1.001× at 4K, 1.005× at 8,800) ([INKFASTPRE-04][fp4]). Routing time is not in the selection chain. |
| R4 | Router width does not depend on command width | Source split-K changes at 1,024 rows; the one remaining width-class dependence | Known. Caused the cache on/off fault in section 2.8. |

### 2.7 Quantization and precision

| # | MLX-Serve | Stark | Status and effect |
| --- | --- | --- | --- |
| Q1 | **No expert-input quantization** on any path: decode reads BF16 inputs into F32 registers (`ups.metal:26`, `downs.metal:31`); prefill NVFP4 loads BF16 activations as register fragments; no `mlx_quantize` call at run time. This changes the model's semantics against the publisher's four-over-six activation quantization; MLX-Serve has no reference check of its own. | Publisher's NVFP4 activation quantization (`INK_EXPERT_INPUTS_QUANTIZED 1` in `experts.metal`, Rust `QUANTIZE_EXPERT_INPUTS`); MXFP8 keeps its own activation quantization under both settings | **Pending, not in the committed default** ([INKROUND-07][round]). With quantization off: decode about −0.8 ms (−0.78 measured) per step and −0.72 to −0.81 ms per verify round on both artifacts (engine, 12 pairs, 11 to 12 wins). Joint build (original): pooled next-token accuracy 73.76 percent against the exact reference's 73.64, agreement 99.38 percent at 0.5 confidence and 99.93 at 0.9, mean KL 0.0100 (today's quantized default: 0.0129). Waiting on the 8-bit quality half and paired prefill timing before the define flips. The exact reference stays quantized. |
| Q2 | Half-precision prefill attention (A1) | BF16 query operands (local) and BF16 global attention with F32 scores and accumulation | Adopted (A1). |
| Q3 | TF32-class prefill router and message-gate projections (R2) | F32 | Not started. |
| Q4 | Decode router exact F32 | Native tensor router | Not a precision difference in practice. |
| Q5 | Opt-in `MLX_SERVE_F16_PREFILL` casts affine banks with biases to fp16; never NVFP4 or MXFP8 (`experts.zig:67-83`) | — | Not applicable. |
| Q6 | MXFP8 via stock MLX | Unguarded MXFP8 (E8) | Adopted. |
| Q7 | Dense prefill via BF16 GEMM with F32 accumulation (inferred) | `dense_fast` F32 accumulation over full K | Adopted (P2); this change carries most of the default's distance from the exact reference (mean row L2 15.2 percent on synthetic captures, inside the gate). |
| Q8 | Width-dependent arithmetic: one row uses `oproj` and TAIL, 2 to 8 rows other kernels; committed state comes from the multi-row forward; width depends on measured timing, so output is not reproducible run to run (source reading of the earlier web analysis) | Drafting on equals drafting off bit for bit; snapshot replay exact; run-to-run identical | Stark property, kept by every adopted change. |

### 2.8 Prefill scheduling and chunking

| # | MLX-Serve | Stark | Status and effect |
| --- | --- | --- | --- |
| C1 | Default chunk 8,192 rows (`engines/mlx/generate.zig:106`), tails under 512 merged; 8,800 tokens run as 8,192 + 608 in two lazy forwards; larger chunks are worth about 0.3 s (7 percent) to MLX-Serve at 8,800 against 2,048 ([INKPREATTR-01][attr]) | 2,048-row commands until 2026-11-23; width API `Session::set_batched_prefill_width` for 2,048, 4,096 or 8,192 rows with reserves 1,344, 2,317 and 4,263 MiB; snapshots stay aligned to 2,048 | Engine: 8,192 against 2,048 rows 1.033× (8/8), 4,096 1.021× (5/8) ([INKFASTPRE-03/04][fp3]). **Adopted in the worker** ([INKWIDTH-01][width]): `source_command_rows` picks the widest width memory admits. Side effect: cancellation is checked between commands, so a cancel on a long prompt can take about 3.5 s instead of about 1 s. |
| C2 | Unscoped and scoped requests see the same chunks | Unscoped requests ran their whole bulk in 8,192-row chunks from position 0, merging the remainder after the last 2,048 boundary into a wide command | **Fault found and fixed** ([INKWIDTH-02][width]): cache on against cache off differed in 34 of 37 validator cases on both artifacts (router rounding differs across the 1,024-row class). `unscoped_source_split` makes unscoped requests split at the same boundary. Re-validation **queued**. |
| C3 | — | Source kernels from 16 rows on the default path (65 for the exact reference) | **Adopted** ([INKFASTPRE-03][fp3]). |
| C4 | — | Tail fold (merging the short tail into the bulk) | Code in the tree, **parked** (`SOURCE_TAIL_ROWS = MIN_ROWS`, does nothing). |
| C5 | Blocking eval every 4 layer visits for long sequences (earlier web reading, approximate) | Blocking wait per command | Worker overhead estimate 0.3 to 0.5 s (source-comparison agent); measured worker extras per request: snapshot 9 to 13 ms, 3-row tail 56 to 66 ms, seed 24 to 28 ms, first-request `set_context` 32 to 37 ms ([INKFASTPRE-04][fp4]). |
| C6 | No draft work in prefill | Packed draft-cache commit per command | **Adopted** ([INKFASTPRE-03][fp3]). |

### 2.9 Host loop and pipelining

| # | MLX-Serve | Stark | Status and effect |
| --- | --- | --- | --- |
| H1 | Lazy MLX graph per step; MLX decides command-buffer splitting (not read) | One Metal command per step or round | See D9. |
| H2 | Drafted rounds synchronous: draft eval, verify eval, lazy commit | Draft and verify overlapped; commit a separate command | Stark measured host work 0.2 ms per round, commit 0.45 ms, between-round 0.01 ms; round time flat across 0 to 3 accepted drafts; prompt-to-prompt differences up to 5 ms come from distinct experts per round ([INKROUND-05][round]). |

### 2.10 Artifacts and packs

| # | MLX-Serve | Stark | Status and effect |
| --- | --- | --- | --- |
| K1 | MLX 8-bit pack: attention, embedding and head affine 8-bit, group 64; experts original bytes; router upcast to F32; four shards with mlx-lm names | Stark 8-bit artifact: the same 178 tensors quantized by a byte-identical quantizer, Stark names, BF16 router, one file, BF16 copies kept resident | **Default by owner decision of 2026-11-23**; the converter and engine support are committed (`61501bf5`, `2a1d72a2`), the catalog and preparation default is done ([INKAFFINE8-04](2026-11-23-inkling-mini-affine8-default.md)); prefix-reuse exactness is shown and validator rule conformance is open ([INKAFFINE8-01 to -03][a8]). Engine: step 18.44 → 15.14 ms, verify rounds 2.9 to 3.4 ms faster. Memory 22.76 GB against 21.09 GB (no saving, copies resident). Pooled natural-text quality passes the bar (section 7). |
| K2 | Inkling Small is a separate plugin (`plugins/inkling/`) that runs serially; Inkling 2 Mini is `plugins/inkling2/` | — | The earlier web-based reading confused the two; corrected by the source comparison. |
| K3 | MLX-Serve 8-bit pack loading needs all four shards | Harness shard preflight | Adopted in the harness after the first run lost its 8-bit arm ([INKENGINE-02][eng]). |
| K4 | — | Stark 8-bit artifact must load images to be timed by the standard benchmark | The arm was refused twice (`stark_images_not_loaded`, runs 3 and `20261123-000711-stark`) until the converter recognized the affine8 layout as the Inkling image family. **Fixed** (coordinator change in `inkling_prepare.py`, `vision_runtime.py`, `inkling_affine8_artifact.py`). |

### 2.11 Measurement

| # | MLX-Serve | Stark | Status |
| --- | --- | --- | --- |
| M1 | Published decode only: llmprobe `--bench-only`, warmup discarded, median of 3, its own code prompt; M4 Max, M5 Ultra and an RTX box; no M5 Max table, no Inkling 2 Mini row and no prefill figure (`benchmarks.md:1-9`) | Every figure on this page is Stark's own measurement on this M5 Max | The often-quoted 116 tok/s is MLX-Serve's launch claim on M4 Max with its 8-bit pack; it was not reproduced here (best MLX 8-bit cell 113.1 in run 2, 107.6 in run 4). |
| M2 | Decode forced to 192 tokens (`ignore_eos`, `min_tokens`) | Stops where the model stops; the gateway rejects the override | Decode rows compare different amounts of work. |
| M3 | Accepts `reasoning_effort: "medium"` | The Inkling template rejected it; "off" renders effort level 0 and the model may still think | Match the mode; check identical input token counts. |

## 3. What worked

Every change that landed, newest work last within each group. "Engine" figures are paired ABBA on the `inkling_slice` diagnostic unless stated.

| Change | Measured effect | Level | Record |
| --- | --- | --- | --- |
| 26 decode selectors made unconditional, source prefill and verifier admitted from 1 token to 32,768 context, drafting by default (3 greedy, 2 sampled) | No timing of its own; server prefill and short decode rose sharply against the pre-session comparison | Correctness, server | [INKDEFAULT-01][def] |
| Long-position gain rounded once in serial and verifier | Correctness fix above about 13,000 positions | Correctness | [INKLONGPOS-01][lp] |
| Verifier and single-row launch fusion, two sessions | Single-row step 22.86 → 19.36 ms (0.844, 6/6); verifier flat | Engine | [INKFUSE-01/02][fuse] |
| `STARK_INKLING_SOURCE_PREFILL` kill switch removed | — | Code | [INKFUSE-02][fuse] |
| NVFP4 guard and replay removed; fused 64 × 64 NVFP4 kernels; faster exact MXFP8 replay | 1,085 → 1,634 tok/s at 4K (1.506×), 1,034 → 1,455 at 8,800 (1.476×), bit-exact | Engine | [INKFASTPRE-01][fp1] |
| Unguarded MXFP8 and mask-free global attention; noise-floor gate; `--exact-prefill` reference | 1.169× (4K), 1.174× (8,800) | Engine | [INKFASTPRE-02][fp2] |
| Dense projection kernel, packed draft commit, mask freed, source kernels from 16 rows | 1.107× (4K) and 1.130× (8,800); independent re-check 1.108× and 1.111×; packed commit 5,722 → 4,679 ms with drafting on | Engine | [INKFASTPRE-03][fp3] |
| Prefill router tiles and fused selection | 1.006× (no gain), hash-identical | Engine | [INKFASTPRE-04][fp4] |
| Global causal skip | No measurable gain, hash-identical | Engine | [INKFASTPRE-04][fp4] |
| Width API and worker width selection | Engine 8,192 rows 1.033×; worker uses the widest admitted width | Engine, worker | [INKFASTPRE-04][fp4], [INKWIDTH-01][width] |
| Unscoped prefill split fix | Restores cache on = cache off | Correctness | [INKWIDTH-02][width] |
| Stark 8-bit dense artifact and decode kernels | Step 18.44 → 15.14 ms (73 tokens), 19.24 → 16.02 (4,150); verify rounds −2.9 to −3.4 ms; server run 4: initial coding 78.9 → 86.9, about 8K 100.1 → 110.8 tok/s | Engine, server | [INKAFFINE8-03][a8], [INKENGINE-04][eng] |
| Batched short-context verifier attention | Four-row verify −1.71 ms (8-bit), −1.60 ms (original), 12/12 | Engine | [INKROUND-03][round] |
| Converter recognizes the 8-bit artifact as the Inkling image family | Stark 8-bit arm can be timed with images loaded | Harness | coordinator change, [INKENGINE-04][eng] |
| Benchmark harness: power-mode parse, shard preflight, failed arm does not abort, `--arms`, `--stark-arm-model`, then the standard benchmark tool and profile | Runs complete; first-run failure modes removed | Harness | [INKENGINE-02][eng], [`docs/BENCHMARK.md`](../BENCHMARK.md) |
| Natural-text quality probe (`--quality-compare`) and the quality bar | Default prefill and Stark 8-bit both pass | Quality | [INKROUND-08][round] |
| History depth planner as the worker default (floor 1, capped by the requested count) | Simulated +8 percent on 8-bit prose and +6 percent on original prose against fixed depth 3, level on code, −1.5 to −3 percent at 4K and 8K (tuning in progress) | Trace simulation, worker gates | [INKROUND-06][round] |
| BF16 query operands in local prefill attention; BF16 global attention with F32 scores and accumulation; global query tile 1,024 with causal skip | Paired 1.018× (wide range); natural-text gate passed on eight inputs (+0.19 points accuracy, 99.28 percent confident agreement) | Engine, quality | [INKFASTPRE-05][fp4] |
| Commit-check gates on the committed tree | Exact reference exact; drafting on equals off at 73 to 16,000 on both artifacts; worker smoke on both | Correctness | `runtime/commit-check/` (summary `8cb9cd12821508c4`) |

## 4. What did not work

Every rejected, reverted, failed, invalid or inconclusive attempt from this work, with its numbers. Check here before trying something again. Earlier Inkling attempts (INKDECODE-01 to -110) are in the [decode toward 60](2026-11-21-inkling-mini-decode-60.md) and [decode toward 80](2026-11-22-inkling-mini-decode-80.md) records and the [speed history](speed-inkling.md).

### Kernels and arithmetic

| Attempt | Numbers | Why | Record |
| --- | --- | --- | --- |
| Shared NVFP4 weight scales (`NVFP4_SHARED_SCALES`) | Gain smaller than the 64 × 32 tile; mutually exclusive with it | Not enabled when selectors were removed; code left unused | [INKDEFAULT-01][def] |
| Unguarded MXFP8 under the old INKPREFILL-07 gate | Failed on 4 of 9 inputs (mean row L2 up to 5.17 percent, worst row 16.0) | The gate compared against an exact incumbent; replaced by the noise-floor gate, under which it passed | [INKFASTPRE-01][fp1], [INKFASTPRE-02][fp2] |
| NVFP4 BK 64, with or without double buffering | Slower | Too much threadgroup memory | [INKFASTPRE-01][fp1] |
| NVFP4 128 × 32 and 128-row tiles | Slower | — (a later retry with an overlap fix is listed as an idea) | [INKFASTPRE-01][fp1] |
| Eight-row and row-major MXFP8 replay | Slower; removed | — | [INKFASTPRE-01][fp1] |
| Fused NVFP4 path on verifier rows | Broke the rollback proof | Verifier commands excluded from the prefill fast path | [INKFASTPRE-01][fp1] |
| MLX-Serve-design NVFP4 prefill kernel (`grouped_nax.metal`) | Gate/up 1,514 to 1,796 ms against 1,477 to 1,611 for the incumbent at 8,800 | Not faster at BM 64, 96 or 128; cause unexplained | [INKFASTPRE-03][fp3] |
| Fixed-split (width-independent) prefill router | 2 of 1,768,704 selections changed; worst row 104.6 percent against an 87.3 percent control maximum, mean 19.2 against 18.6 | Failed the noise-floor gate | [INKFASTPRE-03][fp3] |
| First split-consistency check mixing widths 333 to 1,500 rows | Failed for the exact path too | Source arithmetic switches at 1,024 and 2,048 rows; check rewritten by width class | [INKFASTPRE-02][fp2] |
| Weight-stationary dense | 713 against 670 ms | Activations re-read per 64-column tile dominate, not weights | [INKFASTPRE-04][fp4] |
| Weight-stationary NVFP4 at 2,048-row commands | Gate/up 1,683 against 1,368 ms; wall 4.99 against 4.34 s | Occupancy: about 128 rows per expert fill half of a 256-row tile. 8,192-row test queued. | [INKFASTPRE-04][fp4] |
| Three staged dense variants at 8,192 rows | 1,123 / 1,220 / 1,115 against 967 ms | Staging costs more than it saves; larger tiles do not help | [INKFASTPRE-04][fp4] |
| NVFP4 weights pre-decoded to BF16 in device memory | 9.8 against 9.2 ms per visit | Feeding weights from device costs as much as decoding them | [INKFASTPRE-04][fp4] |
| Global causal skip | 311 against 301 ms | Little work past the diagonal; kept because free and hash-identical | [INKFASTPRE-04][fp4] |
| Prefill fused selection and router tiles | 1.001× to 1.006× | Routing time is not in the selection chain; kept, hash-identical | [INKFASTPRE-04][fp4] |
| Single-row NVFP4 multi-channel variants (six) | 6.75 to 11.5 ms against 4.80 | Register pressure and occupancy; half-buffer variant never timed | [INKROUND-02][round] |
| Decode router F32 matvec rewrite | −0.06 ms per step, +0.14 / +0.38 / +0.56 ms at 2 / 3 / 4 rows | Instrumented 1.11 ms router cost was an artifact of splitting the encoder | [INKROUND-04][round] |
| Cross-visit fusion; down-activation quantization folded into the down kernel; verifier down epilogue into the expert finish | Not built | Expected about zero: verifier launches measured free; 576-fold redundant quantization in the hottest kernel | [INKFUSE-02][fuse] |

### Policies

| Attempt | Numbers | Why | Record |
| --- | --- | --- | --- |
| Confidence cut (MLX-Serve rule) on Stark | 0.3 to 3 percent below the best fixed depth | Removes draft/verify overlap; draft command runs even when zero rows are kept | [INKROUND-06][round] |
| Plain-step backoff (MLX-Serve rule) on Stark | 10 to 15 percent below the best fixed depth on prose | A drafted round costs only 1.29 times a plain step on Stark | [INKROUND-06][round] |
| First adaptive-depth planner version | +8 percent against fixed 3 on 8-bit prose, but lost to fixed 1 on prose and fixed 2 on explain | Tuned (slower estimates, hysteresis, probing); the tuned planner is in progress | [INKROUND-06][round] |
| Interleaved real-arm depth runs (`policy-*`, `adapt*`, `chatdepth*`) | Same fixed depth moved by up to 25 percent between arms in one session (8-bit explain d1 71.5 against d2 54.5) | Too noisy to decide; replaced by trace simulation with same-session costs | [INKROUND-06][round] |
| Depth-run harness | Rounds committed tokens past end of sampling | Invalidated the prose, 8-bit explain and s73 runs; fixed | [INKROUND-06][round] |
| Raw-text continuation depth data | Fixed depth 3 or 2 best depending on input | Distribution unlike chat traffic; kept as reference only | [INKROUND-06][round] |

### Coordinator hypotheses that were wrong

| Hypothesis | What was found | Record |
| --- | --- | --- |
| Launch count is the verifier's main lever | Verifier round time stayed flat when 120 to 250 launches were removed. Later, batching per-row attention launches did help (1.7 ms at four rows), so the lever is per-row launches, not total count | [INKFUSE-02][fuse], [INKROUND-03][round] |
| MLX-Serve's higher tokens per round comes from wider verification | Undercut by the reasoning-mode mismatch in the earlier comparison; the source shows tokens per step is a policy outcome | [mlxsrc] |
| Drafting adds about 0.5 s to an 8,800-token prefill | Measured in one session it is 10 to 70 ms; the earlier figure compared different power states (see conflict C1, section 7) | [INKFASTPRE-04][fp4] |
| The worker loop loses 2.5 to 3 ms per round | Host work about 0.2 ms plus 0.45 ms commit; prompt differences come from distinct experts on varied text | [INKROUND-05][round] |
| A tile-shape traffic model for dense prefill (more work per staged byte is faster) | Three staged variants, one with twice the work per staged element, were no faster | [INKFASTPRE-04][fp4] |
| Stage floors from instrumented timings | Splitting the encoder at stage boundaries inflates small kernels; only paired uninstrumented timing counts | [INKROUND-04][round] |
| A 175 ms FLOP floor for the dense prefill projections (55 TFLOPS on 9.5 TFLOP) | A decomposition on synthetic operands at 8,192 rows put the tensor-only floor at about 1.8 ms per projection against about 2.7 ms in production; dense is at about 1.5 times its floor, not 4 times | [INKFASTPRE-06][fp4] |

### Defects, invalid runs and incidents

| Item | Outcome | Record |
| --- | --- | --- |
| Verifier long-position double rounding | Drafted state diverged from serial above about 13,000 positions; fixed | [INKLONGPOS-01][lp] |
| INKFUSE-01 novel-prose regression (58.6 against 65.3 tok/s) | Did not reproduce; one pair slower on untouched stages pointed to host state | [INKFUSE-02][fuse] |
| Stale `attention` and `trace.mlp_conv` probes after fusion | Repaired | [INKFUSE-02][fuse] |
| Engine-matrix run 1 8-bit arm | Pack shards deleted by another session's cleanup; MLX-Serve exited 1 | [INKENGINE-02][eng] |
| Stark 8-bit arm in run 3 and run `20261123-000711-stark` | Refused, images not loaded; fixed by the converter change | [INKENGINE-03/04][eng] |
| Unscoped wide prefill | Cache on against off differed in 34 of 37 cases; fixed, re-validation queued | [INKWIDTH-02][width] |
| Decode-only unquantized-input quality run | Failed: diagnostic control required a verifier activation kernel absent without a draft; fixed, re-run queued | [INKROUND-07][round] |
| Compile errors from concurrent edits (`grouped_pilot.rs` private `register`; `decode_stages.rs` ambiguous integer) | Fixed | [INKFASTPRE-04][fp4] |
| Data volume filled (167 MB free) at about 08:25 on 2026-11-23 | A build failed; duplicate build caches and old raw traces deleted; no source truncated | coordinator notes |
| Busy-host guard tripped by brief cargo invocations outside the lock | Benchmark refused to start several times | coordinator notes |
| Joint target-plus-draft state hash under `--exact-prefill` differed from `fused2-build` | Resolved: the INKFASTPRE-01 build, the round-3 final and the re-check build agree; the intermediate fusion build was the outlier | [INKFASTPRE-03][fp3] ("Joint draft-state hash, resolved") |
| `s4096-01` reported not exact under `--exact-prefill` by the 8-bit workstream | Resolved: that run used the source-holdout capture of the same name, taken on a 512-row schedule; all four stock captures are exact | [INKFASTPRE-03][fp3] ("All four stock captures exact") |

## 5. In progress and queued

As last reported on 2026-11-23. "Owner" is the workstream that holds it.

| Item | Owner | State | What decides it |
| --- | --- | --- | --- |
| Tune the history depth planner for 4K and 8K (it is the committed default; simulated 1.5 to 3 percent below fixed depth 3 there) | Decode workstream | In progress | Trace simulation with same-session costs, then a worker run |
| `INK_EXPERT_INPUTS_QUANTIZED` = 0 as the default | Coordinator | Decode timing and the original-artifact quality half done | Stark 8-bit quality half and paired prefill timing; then fold the remaining activation pass into the products |
| Decode-only natural-text comparison for unquantized inputs | Decode workstream | Re-run queued after a diagnostic fix | Quality bar |
| Prefix-reuse re-validation of the wide-command worker, and a control on the 2,048-row worker | Coordinator | Queued | Validator passes; the control decides whether 5 reuse-rule mismatches predate wide commands (section 7) |
| Stark 8-bit artifact as the default: validator rule conformance and the remaining qualification gates | Coordinator | Open | The catalog and preparation default is done ([INKAFFINE8-04](2026-11-23-inkling-mini-affine8-default.md)). Prefix-reuse exactness is shown; the worker still advertises 1,024-token reuse chunks while it snapshots every 2,048 rows, and the validator does not model default drafting. Matched image-off/on text speed, tool calling, the app end to end and a standard benchmark run are not done |
| Standard benchmark run (three repeats, block layout) | Benchmark workstream | Queued | First standard result |
| CI running the benchmark tests | Coordinator | `.github/workflows/ci.yml` untouched | — |
| Overlap dense loads with tensor steps (dense runs at about 1.5 times its tensor floor) | Prefill workstream | Identified | Paired ABBA at 8,800 |
| Joined dense projections in prefill | Prefill workstream | Queued | ABBA |
| Elementwise fusion in prefill (about 430 ms of passes) | Prefill workstream | Queued | ABBA |
| Register-fragment activation loads in `--expert-ceiling` | Prefill workstream | Queued | Kernel-level timing |
| Weight-stationary NVFP4 at 8,192-row commands | Prefill workstream | Queued | Stage profile against the incumbent |
| MLX-Serve kernel timing harness on the `--expert-ceiling` capture | Prefill workstream | Not run | — |
| Fold the commit into the next round's draft command | Decode workstream | Queued after the planner | Worker loop timing (0.3 to 0.45 ms expected) |
| Launch-count merges on the one- and two-row paths | Decode workstream | Queued | Paired engine timing |

## 6. Ideas identified and not started

| Idea | Estimate | Estimated by |
| --- | --- | --- |
| Pipelined plain steps (encode step N+1 while N runs) | 0.3 to 0.8 ms per plain step | Source-comparison agent |
| Remaining expert tail and controller fusion (output gate in the projection, SwiGLU in up, one input kernel per visit) | 0.5 to 1.5 ms per token | Source-comparison agent |
| Shrink the four-row verify expert stage (7.9 ms of dots on s73) toward the 8.29 ms gather ceiling for the whole stage | Larger than any host change (not quantified) | Decode workstream |
| Wider verification (4 or 5 drafts) for long context | +8 to +12 percent at about 8K, assuming 0.88 conditional acceptance holds | Decode workstream |
| Overlap NVFP4 decode with tensor work (producer and consumer SIMD groups, deeper pipeline, less threadgroup memory); retry taller row tiles with the overlap fix | NVFP4 products 2,085 ms against an estimated 1,050 ms floor at 8,800 | Prefill workstream (floor is a FLOP estimate) |
| TF32-class prefill router and dropping expert-input passes in prefill | 200 to 400 ms at 8,800 (router 371 plus quantization 136 ms) | Source-comparison agent |
| Non-blocking prefill command submission | 0.3 to 0.5 s worker overhead at 8,800 | Source-comparison agent; a second packed workspace would be needed |
| Four-row 8-bit dense kernel | 4.27 against about 2.4 ms bandwidth floor at four rows | 8-bit workstream |
| Affine draft vocabulary head instead of the 440 MiB INT8 bank; drop resident BF16 copies of the 8-bit artifact | Memory, about 1.5 GB | 8-bit workstream |
| Confidence cut, if the draft command becomes much cheaper | — | Decode workstream |
| Isolate the 14 to 15 ms host gap between prefill commands seen with drafting on | Small at 8,800 | Prefill workstream |

## 7. Measurement rules learned and open questions

### Rules

- **Pair and lock.** On this host every build and GPU job runs behind one lock script. Time only in base, candidate, candidate, base process pairs inside one lock hold, at least four pairs, and report the median of per-pair ratios. The same configuration moved by up to 25 percent between arms in one session; absolute times drifted 20 to 30 percent over long sessions; the first request after load is consistently faster. ([INKFASTPRE-01][fp1], [INKPREATTR-01][attr])
- **Power.** The 140 W adapter negotiates 94 W (20 V, 4.69 A) throughout and the battery drains under GPU load while "charging" (100 to 87 percent in about 90 minutes). The owner has been told the likely fix is the MagSafe 3 cable. It has not been shown to limit speed.
- **Instrumented sub-stages are upper bounds.** Splitting the encoder to time small stages inflates them (the router's 1.11 ms). Judge only by paired uninstrumented timing. ([INKROUND-04][round])
- **Synthetic captures do not measure quality.** The oracle and capture prompts are near-uniform: the model predicts their next token 0 to 2 times in 34 (only the 73-token prompt is natural, 18 of 34). Flip rates and draft acceptance on them say nothing about quality or real-traffic speed; they remain valid for exactness and self-consistency. Use natural text (`runtime/inkling-natural-text-2026-11-22/`, repository prose and source, 300 to 8,000 tokens) and chat-template inputs for acceptance. ([INKROUND-08][round])
- **The quality bar** ([`docs/BENCHMARK.md`](../BENCHMARK.md)): pooled next-token accuracy not more than 2.0 points below the reference and at least 98 percent top-1 agreement where the reference's top-1 probability is at least 0.5. Pooled over 2,580 rows: default prefill against exact prefill 73.41 against 73.64 percent, 99.28 percent confident agreement; Stark 8-bit against original 73.41 against 73.41 percent, 98.92 percent. Both pass.
- **Noise-floor gate** for inexact changes against an exact incumbent: candidate inside the span of single-ulp perturbation trials and its mean at most twice the control median. A sensitivity comparison, not a quality measure. ([INKFASTPRE-02][fp2])
- **Match the reasoning mode and rendering**; check identical input token counts across arms; record that MLX-Serve forces 192 decode tokens. ([INKENGINE-01][eng])
- **Interleaved real arms are too noisy for policy decisions**; simulate policies on traces with costs measured in the same session at the same positions, then confirm once at worker level. ([INKROUND-06][round])
- **Read what the other engine loads and claims**: MLX-Serve skips the draft head here; its published numbers are decode only on other machines. ([mlxsrc])
- **A server run with one repeat in the legacy layout is not a standard result**; the standard needs three blocks, block layout and an attached passing quality result. ([`docs/BENCHMARK.md`](../BENCHMARK.md))

### Open questions

- **Prefix-reuse rule mismatches.** The validator reported 5 reuse-rule mismatches believed to predate wide commands: the worker advertises 1,024-token reuse chunks while source prefill aligns to 2,048 rows with a 4-token suffix, and it reuses a snapshot across a change of drafting mode where the validator's rule expects none. A control on the 2,048-row worker is queued; if confirmed, this is a qualification gap in the Inkling port.
- **Power adapter.** Whether the 94 W negotiation limits sustained speed.
- **CI.** The new benchmark tests are not in the CI workflow.
- **Why MLX-Serve's per-row NVFP4 expert cost is lower**, after a structural port was not faster.
- **MLX library internals** (`quantized_matmul`, `gather_qmm`, steel GEMM, `scaled_dot_product_attention`) were not available to read.
- **The 4,096-row prefill pair** at round 4's 4K timing was interrupted; there is no clean 4K ratio for the router tiles.

### Conflicts in the sources

- **C1, size of draft work in prefill.** Prefill round 3 measured the packed draft commit at 5,722 → 4,679 ms (two alternating runs in one loaded target, drafting on), and the attribution study put drafting at 0.45 to 0.84 s back to back. The coordinator's later finding is that drafting adds only 10 to 70 ms in one session (round 4, after the packed commit: 5,320 against 5,252 ms, GPU 5,241 against 5,229) and that the earlier 0.5 s figure compared different power states. Both can hold if the packed commit removed most of the cost; neither run isolated that, so the size of the packed commit's gain is uncertain.
- **C2, the router's cost.** The decode stage table attributed 1.11 ms per step to the router; the paired rewrite saved 0.06 ms. The stage table overstates small stages.

[def]: 2026-11-22-inkling-mini-default-fast-paths.md
[lp]: 2026-11-22-inkling-mini-long-position-gain.md
[fuse]: 2026-11-22-inkling-mini-launch-fusion.md
[fp1]: 2026-11-22-inkling-mini-exact-fast-prefill.md
[fp2]: 2026-11-22-inkling-mini-unguarded-mxfp8-prefill.md
[fp3]: 2026-11-23-inkling-mini-prefill-round3.md
[fp4]: 2026-11-23-inkling-mini-prefill-round4.md
[a8]: 2026-11-22-inkling-mini-affine8.md
[round]: 2026-11-23-inkling-mini-decode-round.md
[width]: 2026-11-23-inkling-mini-wide-prefill-worker.md
[eng]: ../benchmarks/2026-11-22-inkling-mini-engine-matrix.md
[attr]: ../benchmarks/2026-11-22-inkling-mini-prefill-attribution.md
[mlxsrc]: ../benchmarks/2026-11-23-inkling-mini-mlx-serve-source.md
