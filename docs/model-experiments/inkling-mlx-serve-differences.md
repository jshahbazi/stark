# Inkling 2 Mini: Stark against MLX-Serve, what worked, what did not, what is open

This is a maintained list, not a dated record. It collects every known difference between Stark and MLX-Serve for Inkling 2 Mini, every change Stark made or tried during the 2026-11-22 to 2026-11-23 work, and what is still open, so that nothing has to be rediscovered. The dated records stay the evidence: every entry links its record or names its private receipt. Where this page and a record disagree, the record is right. When an item changes, update its entry here and add the evidence to a dated record.

Conventions:

- **MLX-Serve** means `ddalcu/mlx-serve` at `2fe9be834a601c70ef1a597563026340f0e722fe` (release binary `v26.11.4-pre-release.1`, SHA-256 beginning `05983e1278d6`). Its file and line references are relative to `src/src/` of the pinned source and come from the [source comparison][mlxsrc], which read the source but not the MLX library underneath it (`lib/mlx-src` was not in the download). Statements about MLX's own kernels are marked inferred.
- **Stark** paths are under `crates/stark-engine/src/engine/inkling_slice/` unless given in full.
- **Levels:** *engine* is the `inkling_slice` diagnostic on restored state; *worker* is a `stark-generate` request; *server* is the llmprobe client through the gateway. Figures from different levels are not interchangeable.
- **Status:** adopted (default in the tree), rejected (tested and dropped), in progress (being built or measured), stopped (work ended with the wind-down of 2026-11-23; see section 6), not started (idea only), not applicable.
- **Artifacts:** "original" is the fixed-three-pass `thinkingmachines/Inkling-2-Mini` at `6e7c8248f75265bdd4fbab5273304243b5d8c2cb`. "Stark 8-bit" is Stark's own affine 8-bit dense artifact ([INKAFFINE8-03][a8]). "MLX 8-bit" is the author's published pack `ddalcu/Inkling-2-Mini-MLX-8bit` at `cb7fa052a9bbafbbeb91750edc67046e7bacd4f1`; its quantized tensors are byte-identical to Stark's quantizer output ([INKAFFINE8-02][a8]), but its layout and loader differ.
- **Commits.** The work of 2026-11-22 to 11-23 was committed and pushed on 2026-11-23 in five commits on top of `f1666090`: `61501bf5` (converter, 8-bit artifact), `2be13769` (client drafting defaults), `2a1d72a2` (engine and worker), `bf126020` (benchmark tool), `87b2bb17` (documentation). Later commits: `5a37dd37` (documentation), `8b550c9d` (NVFP4 expert inputs unquantized by default), `222a9436` (the Stark 8-bit artifact as the default model), `1206760e` (planner tuning), `4314ac51` (gathered NVFP4 prefill kernel for wide commands, planner hold rule). "Adopted" below means default in the tree at `4314ac51` unless an entry says otherwise. Nothing is released or installed.
- **Verification of `4314ac51`** (coordinator; `runtime/commit-check-3/`, summary SHA-256 prefix `4d038aef3e5d6fc0`): clean build with no warnings, 167 worker tests, the exact reference exact on `g4096-02` and `s4096-00`, drafting on equals off at 73, 4,150, 8,800 and 16,000 tokens on both artifacts, a repeated replay with identical hashes, the worker smoke on both artifacts. The planner unit tests failed to build in that run (missing `cpu` feature) and have never run.
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
| 2026-11-23 afternoon | After run 5 the owner judged the standing good enough and said the novel-prose gap does not matter ("I don't care too much about prose. It seems fine."). Optimization stopped ([wind-down][wind]). |
| 2026-11-23 afternoon | A three-repeat standard benchmark runs on the committed build (the `4314ac51` worker) to confirm the standing. |

### Latest scoreboard (server level)

**The standard result** ([INKENGINE-06][std]): three repeats in the block layout through the standard tool, 19:10 to 19:50 local on 2026-11-23, on the final build (worker `f86d94b877b3…`, built from the code committed as `7e73e180`, arithmetic hash-identical to `4314ac51`), natural-text quality attached and passing on both Stark arms. The tool reports the run eligible with no deviations and all four arms as standard results. Run directory `runtime/benchmark/inkling-2-mini/20261123-191052-inkling-2-mini`, manifest SHA-256 `7c049a5d63dd30ecdaad177af9922ef49df4eb334590f2de42d41cbb45f2060c`. Host notes: adapter 94 W of 140 W, thermal state 1 throughout. The burst was not run (out of scope).

| Standard row | Stark / MLX-Serve, 8-bit pair (default) | Standing | Stark / MLX-Serve, original pair | Standing |
| --- | --- | --- | --- | --- |
| Initial coding | 1.02 (1.00 to 1.10) | unresolved | 1.07 (1.00 to 1.07) | ahead |
| Sustained coding | 1.06 (0.94 to 1.08) | unresolved | 1.11 (1.08 to 1.12) | ahead |
| Novel prose | 0.86 (0.84 to 0.93) | behind | 0.96 (0.96 to 0.99) | behind |
| About 512 tokens | 1.00 (0.91 to 1.12) | unresolved | 1.06 (0.90 to 1.08) | unresolved |
| About 4K | 1.03 (1.03 to 1.36) | ahead | 1.18 (1.09 to 1.28) | ahead |
| About 8K | 1.04 (1.02 to 1.11) | ahead | 1.06 (0.97 to 1.09) | unresolved |
| About 16K | 1.03 (0.99 to 1.08) | unresolved | 1.07 (0.93 to 1.10) | unresolved |
| Repeated-sentence prefill, 8,768 tokens | 1.13 (1.07 to 1.14) | ahead | 1.02 (0.92 to 1.07) | unresolved |
| Exact 8,800-ID prefill | 0.96 (0.96 to 1.02) | unresolved | 0.98 (0.93 to 0.99) | behind |

Paired ratio is the median of the three per-block ratios with its range; a standing is ahead or behind only when all three blocks agree. Reading it: on the default artifact Stark is level to ahead everywhere except novel prose, which is behind by about 14 percent; that is not a sweep, and five of the nine standard rows on the 8-bit pair are unresolved. MLX-Serve 8-bit's repeated-sentence prefill rate was lower in this run than in the single-repeat runs, so the exact 8,800-ID row (unresolved at 0.96) is the matched prefill cell to read. MLX-Serve forces 192 decode tokens while Stark stops at end of sequence, and MLX-Serve reused 9 cached tokens on the novel-prose cell, which is defined as cold. Medians, ranges, tokens per step and supplementary rows are in the [record][std]. This is the current comparison; runs 1 to 5 below were single repeats and are history.

### Previous scoreboard, run 5 (single repeat)

Run 5 ([INKENGINE-05][eng], `runtime/benchmark/inkling-2-mini/20261123-150914-inkling-2-mini`, 15:09 local on 2026-11-23, manifest SHA-256 `19ef5dce32f0431adf2e000baaf5db4de0762bdb01f4385f9e165a591ef4578c`): the first four-arm run through the standard tool, on the committed tree `4314ac51` (Stark worker beginning `a966b11554ae`), llmprobe 0.6.18, this 128 GiB M5 Max, reasoning `off` requested (both engines still think), each engine's default drafting, block layout with one block. It is **not a standard result**: one repeat, and no quality result attached to the run directory (the natural-text results for this build's arithmetic pass the bar, [INKROUND-09][quant]). Process thermal state was 0 before the first arm's probe session (Stark original) and 1 for everything after it, so the Stark original decode rows ran on a cooler host; the matched 8-bit pair ran in the same state.

| Scenario, tok/s | Stark original | Stark 8-bit | MLX original | MLX 8-bit | Stark / MLX, original | Stark / MLX, 8-bit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Initial coding | 83.3 | 84.9 | 80.6 | 86.7 | 1.03 | 0.98 |
| Sustained coding | 87.8 | 99.4 | 79.2 | 97.4 | 1.11 | 1.02 |
| Novel prose | 63.0 | 75.0 | 62.0 | 82.2 | 1.02 | 0.91 |
| About 512 tokens | 75.3 | 87.7 | 76.3 | 89.2 | 0.99 | 0.98 |
| About 4K | 99.1 | 108.7 | 95.8 | 109.2 | 1.03 | 0.99 |
| About 8K | 98.6 | 105.1 | 91.1 | 89.6 | 1.08 | 1.17 |
| About 16K | 101.8 | 98.4 | 85.0 | 91.2 | 1.20 | 1.08 |
| Prefill, 8,768 repeated-sentence tokens | 2,010 | 1,912 | 1,985 | 1,909 | 1.01 | 1.00 |
| Prefill, exact 8,800 IDs | 2,040 | 2,138 | 2,088 | 2,160 | 0.98 | 0.99 |

Reading it: on the matched 8-bit pair, which is the default artifact, Stark is within 2 percent of MLX-Serve on initial coding, sustained coding, 512, 4K and prefill, ahead at 8K and 16K, and behind on novel prose by 9 percent. Novel prose is the one clear gap left: about 25 ms per Stark round for 1.90 tokens against about 14.5 ms per MLX-Serve step for 1.19. In run 4 the same pair stood at 0.86 on initial coding, 0.74 on novel prose and 0.76 to 0.80 on prefill. Absolute rates of both engines moved between the runs (MLX-Serve 8-bit prefill 2,220 to 1,909, initial coding 101.0 to 86.7), so only ratios within a run compare. With one repeat none of these ratios is a standing; a three-repeat run with quality attached is still owed. MLX-Serve forces decode to 192 tokens while Stark stops where the model stops, so decode rows do not cover identical work. The run also has supplementary counting-ceiling rows (single measurements, reported only): on the 8-bit pair Stark / MLX-Serve is 0.83 at about 512 tokens (83.0 against 99.5), 1.08 at 4K (Stark reused 2,048 cached tokens there by the scenario's design), 0.89 at 8K (96.6 against 108.3) and 0.98 at 16K, so short and mid-length decode is not settled until the three-repeat run.

### Previous scoreboard, run 4

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

History of the server-level scoreboard (tok/s, Stark original / MLX original / MLX 8-bit unless noted; runs 1 to 5 one repeat each, the standard run three):

| Run | Initial coding | Novel prose | About 4K | Exact 8,800 prefill | Notes |
| --- | --- | --- | --- | --- | --- |
| Before this session (another worktree) | 34.8 against 97.6 | | | 389 against 1,898 | Mismatched reasoning modes and a forced length; Stark fast paths not admitted outside 512 to 4,352 tokens. Not an engine comparison. |
| Run 1, 2026-11-22 16:11 ([INKENGINE-01][eng]) | 78.0 / 80.5 / not run | 58.6 / 71.8 / not run | 106.5 / 84.2 / not run | not run | MLX 8-bit shards had been deleted |
| Run 2, 17:53 ([INKENGINE-02][eng]) | 79.7 / 79.3 / 93.3 | 62.0 / 71.7 / 89.0 | 98.2 / 85.1 / 113.1 | 1,194 / 2,236 / 2,225 | Exact fast prefill build |
| Run 3, 23:24 ([INKENGINE-03][eng]) | 75.4 / 81.8 / 90.8 | 64.2 / 72.7 / 88.4 | 108.2 / 85.8 / 102.5 | 1,707 / 2,088 / 1,897 | Stark 8-bit arm refused (`stark_images_not_loaded`); taken while agents were loading the GPU |
| Run 4, 2026-11-23 06:05 ([INKENGINE-04][eng]) | 78.9 / 86.7 / 101.0 (Stark 8-bit 86.9) | 58.4 / 73.3 / 91.6 (67.8) | 112.0 / 89.2 / 107.6 (115.4) | 1,879 / 2,574 / 2,486 (1,892) | Quiet host; previous table above |
| Run 5, 2026-11-23 15:09 ([INKENGINE-05][eng]) | 83.3 / 80.6 / 86.7 (Stark 8-bit 84.9) | 63.0 / 62.0 / 82.2 (75.0) | 99.1 / 95.8 / 109.2 (108.7) | 2,040 / 2,088 / 2,160 (2,138) | Committed tree `4314ac51` through the standard tool; run-5 table above |
| Standard run, 2026-11-23 19:10 ([INKENGINE-06][std]) | 81.2 / 76.1 / 84.5 (Stark 8-bit 85.1) | 68.1 / 69.4 / 84.4 (74.0) | 99.9 / 84.1 / 92.2 (106.5) | 1,985 / 2,030 / 2,092 (2,014) | Three repeats, medians; standings in the latest table above |

## 2. Every known difference

Status refers to Stark. "Effect" is Stark's measured change where one exists; estimates say who made them.

### 2.1 Speculative decoding policy

| # | MLX-Serve | Stark | Status and effect |
| --- | --- | --- | --- |
| S1 | Drafter proposes 7 tokens per round (`draft.zig:1-8`); a per-round **confidence cut** keeps drafts while the running product of draft confidences is at least `min_sure` = clamp(1.34·√(verify-row cost / plain-step cost), 0.3, 0.8), costs measured live (`plugins/inkling2/plugin.zig:290-342, 425-435`); zero drafts makes a plain step | Fixed depth up to the requested MTP count (3) every round; proposals are not read back before verification so draft and verify overlap | **Rejected** for Stark after trace simulation ([INKROUND-06][round]): the best confidence cut is 0.3 to 3 percent below the best fixed depth on every input, because reading confidences first removes the draft/verify overlap (1 to 3.5 ms per round) and the 3.6 to 4.1 ms draft command runs even when the cut picks zero. Revisit if the draft command becomes much cheaper. |
| S2 | Engine **backoff**: starts with 4 plain rounds, compares each drafted round's value (tokens × measured plain-step time) with its wall time, and skips drafting for a doubling period up to 32 rounds when it loses (`engines/mlx/spec/dflash/policy.zig:169-224`, `engines/mlx/spec/dspark.zig:92-93, 131-150, 208-215`) | No backoff | **Rejected** for Stark ([INKROUND-06][round]): on Stark a drafted round costs about 1.29 times a plain step (20.5 against 15.9 ms at depth 1), so even prose at about 35 percent first-position acceptance gains from drafting (no drafting 56.5 against 71.0 tok/s at depth 1). The backoff's test fires on most rejected rounds and costs 10 to 15 percent. |
| S3 | Depth is a policy outcome, which is why MLX-Serve's tokens per step is low on prose (1.19) and high at long context | Stark's history planner (`depth_planner.rs`: running acceptance and cost estimates, EMA 0.03 and 0.1, hysteresis 1.03, probe one row deeper every 16 rounds, at least one draft) chooses 1 to 3 drafts per round; since `1206760e` prior 0.8 / 0.75 / 0.75, acceptance weight 0.05, probe every 32 rounds; since `4314ac51` it holds three proposals while every per-position acceptance estimate is at least 0.70 | **Adopted** in `2a1d72a2`, tuned in `1206760e` and `4314ac51` ([INKROUND-06][round], [INKROUND-11 and -12][plan]). Trace replays: within 1.5 percent of the best fixed depth on all 14 runs (mean 0.998). Worker ABBA against fixed depth 3 on the 8-bit artifact (tuned settings, before the hold rule): prose 1.088, explain 1.008, p4000 0.995, c4000 0.991, code 0.986, c8000 0.976, mean 1.007. The hold rule removed the 4K and 8K loss in noisy trace replays and has no worker measurement; the code loss remains. Planner unit tests have never run. |
| S4 | Verify width up to 7 drafts (8 rows) | `MAX_VERIFY_ROWS = 4` (3 drafts) wired into dense, attention, mix, norm, gate, save and rollback, logits rows and draft commit | **Not started.** Estimate (decode agent, assumption that the 0.88 conditional acceptance holds): 4 drafts about +8 percent and 5 drafts about +12 percent at about 8K, where 74 percent of anchors accept all 3. Several days of kernel work; long-context traffic only. |
| S5 | Verify forward appends nothing; commit appends accepted rows (`kernels.zig:572-611, 579-587`); no rollback | Transactional verifier with exact rollback and snapshot; commit is a separate command (about 0.45 ms per round) | Not applicable as a difference in output. Folding the commit into the next draft command is **stopped**, a candidate if work resumes (section 6) (estimate 0.3 to 0.45 ms per round). |
| S6 | Draft evaluated with a blocking `evalMany` before the verify graph is built (`plugin.zig:395`); no overlap | Verify encoded while the draft runs (overlapped) | Stark advantage; keep. It is also why the confidence cut does not pay on Stark. |
| S7 | Drafter context K/V cached across rounds (`draft.zig:48-62, 184-230`) | Incremental draft cache (`dspark.rs:504-590`) | Not a difference. |
| S8 | Draft head: bundled DSpark drafter with a 3-launch Markov RNN per step | Same DSpark head; INT8 vocabulary bank (440 MiB) built from the BF16 head | Draft command 3.0 to 3.2 ms, almost all the fixed block forward (3 BF16 draft layers about 500 MB plus the INT8 head about 460 MB); each extra proposal row about 0.1 ms ([INKROUND-01][round]). |
| S9 | MLX-Serve does not load the draft head for this checkpoint at all in the attribution session's launch (startup log skips `mtp.safetensors`), so it does no draft work in prefill | Stark drafts by default and commits draft cache rows during prefill | Stark's packed draft commit made the cost small (section 3). See C1 in section 7 for a conflict about its size. |
| S10 | Tokens per step: MLX-Serve 1.18 to 3.43 depending on scenario; in run 5 on the 8-bit pack 1.43 on initial coding, 1.19 on novel prose, 2.82 at about 4K | Stark 2.0 to 3.31 at fixed depth 3 (runs 2 to 4); in run 5 with the planner, 8-bit 2.18 tokens per round on initial coding (acceptance 0.55), 1.90 on novel prose (0.41), 3.10 at about 4K (0.71) | Policy outcome, not verify width ([coordinator finding](#7-measurement-rules-learned-and-open-questions)). |

### 2.2 Decode step structure and fusion (one row and verify rows)

| # | MLX-Serve | Stark | Status and effect |
| --- | --- | --- | --- |
| D1 | About 12 to 14 launches per first-pass visit and 17 to 19 per later-pass visit, about 0.75k per token (inferred count, `kernels.zig:985-1002`) | 1,907 launches per single-row token before fusion, 1,368 after INKFUSE-01, 992 after INKFUSE-02 | **Adopted** ([INKFUSE-01 and -02][fuse]): single-row step 22.86 → 19.36 ms (engine, 0.844, 6/6 pairs). Launch count alone is about 25 percent lower in MLX-Serve, not 2 times. |
| D2 | One `pre` kernel per visit: depth-block softmax mix plus weighted RMS norm of all q/k/v inputs (`kernels.zig:190-214`, `pre.metal`) | Mix plus q/k/v norms fused (`ink_verify_mix_qkv_norm`); later-pass controller chain 17 → 5 → 4 launches | **Adopted** ([INKFUSE-01/02][fuse]). |
| D3 | `prep`: k/v short convs, L2 clips, v norm, gate sigmoid, bias row in one launch (`kernels.zig:310-345`) | K/V convolution, normalization and cache write plus Q/R normalization in one launch (`ink_verify_heads_pack_one`) | **Adopted** ([INKFUSE-02][fuse]). |
| D4 | Projections joined: q, gate, r and sink as one; k and v as one on sliding first-pass visits (`plugin.zig:858-860`, `layers.zig:270-287`) | Six attention projections plus two dynamic-conv projections in one launch (`ink_matvec_source_visit`; on 8-bit `ink_a8_matvec_visit`) | **Adopted** ([INKFUSE-02][fuse], [INKAFFINE8-03][a8]). |
| D5 | Output gate folded into the output projection (`oproj`, `kernels.zig:366-378`) | Gate in `ink_attention_gate`, then a separate output projection | **Not started.** Part of the 0.5 to 1.5 ms controller and tail estimate (source-comparison agent, at 5 to 7 µs per dependent launch). |
| D6 | `mid`: attention conv, residual or block boundary, depth mix and MLP norm in one launch | `ink_verify_attention_mix` does the same, with deferred history writes | **Adopted** ([INKFUSE-01][fuse]). |
| D7 | `ups` fuses SwiGLU into the gate/up product; `downs` fuses the down product, weighted pick-order sum, MLP conv and residual (`downs.metal:36-52`) | Separate up, activation quantization, down, `ink_expert_sum`, conv and `ink_after_experts` launches on the single row; the expert sum, MLP conv and residual are one launch since INKFUSE-01 | Partly adopted. Removing the expert-input pass by reading BF16 rows inside the products was **rejected**: bit-exact but 0.15 ms slower per step and up to 1.2 ms per four-row round, because the widening repeats per output channel ([INKROUND-10][quant]). Two single-row and verifier launch merges were tried at the wind-down and reverted unmeasured after their paired run hung ([wind-down][wind]); the merges are **stopped**, a candidate if work resumes (section 6) (estimate 0.5 to 1.5 ms, source-comparison agent). |
| D8 | Verify rows: 4 to 8 rows read K/V once per KV head (`relsplit` and `relcomb`) | Shared local and global verifier attention were admitted only at position ≥ 1,024; below that every row ran its own attention (288 launches per row) | **Adopted** ([INKROUND-03][round]): batched short-context verifier attention (`verify_short.metal`) whenever position + rows < 1,024. Four-row verify −1.71 ms (8-bit) and −1.60 ms (original), 12/12 pairs each. |
| D9 | Plain steps are pipelined: the next forward is `async_eval`ed and its token resolved at the start of the next tick (`engines/mlx/generate/pipeline.zig:75-90`) | One command per step with commit and wait | **Stopped**, a candidate if work resumes (section 6), together with the commit fold. Estimate (source-comparison agent) 0.3 to 0.8 ms per plain step. Worker-loop study: host work is about 0.2 ms and the commit about 0.45 ms per round; GPU idle between rounds about 0.6 ms ([INKROUND-05][round]). |
| D10 | Cross-visit fusion not present either | — | **Rejected as not worth building** ([INKFUSE-02][fuse]): race-free only on about 18 local first-pass visits per round; verifier launches measured free. |

### 2.3 Expert products

| # | MLX-Serve | Stark | Status and effect |
| --- | --- | --- | --- |
| E1 | Decode `ups.metal`: one SIMD group per interleaved gate/up row pair, 16 inputs per lane per sweep read as `float(x[k+i])`, nibble and scale decode by bit tricks into half (`ups.metal:26-37`, `moe_fp4_e2m1x2_header.metal:1-4`, `moe_fp4_nv_scale.metal:1-3`); one output row per SIMD group by design (`kernels.zig:41-44`) | `ink_nvfp4_up` / `_vector_up` with byte-pair FP4 coefficient table, K16 F32 accumulation | Stark's multi-row-per-SIMD-group single-row variants **rejected** ([INKROUND-02][round]): six bitwise-identical variants were 6.75 to 11.5 ms against 4.80 ms, register pressure and occupancy. A half-buffer variant was written and never timed. |
| E2 | Verify: `ups`/`downs` widen the grid by rows with no expert deduplication (`kernels.zig:914-953`) | Run-grouped verifier NVFP4 shares weights across rows routed to the same expert | Stark design differs; verify expert stage 11.74 ms at four rows on s73 (20 distinct experts per visit) against a measured 8.29 ms gather ceiling ([INKROUND-01][round]). Shrinking it is the larger decode lever; **not started**. |
| E3 | Prefill NVFP4: `gather_qmm_nax` double-buffered schedule, BM 96 × BN 64 × BK 64 at mean rows per expert ≥ 48 and K ≥ 1,024 (`gather_qmm_nax.zig:28-39`); weight tile dequantized to BF16 in threadgroup memory, next K step's packed words prefetched into registers; activations read from device as `vec4` register fragments (`gather_qmm_nax_header.metal:543-651, 612-640`); MPP 16×32×16 with `relaxed_precision` | `grouped_fast.metal`: 64 × 64 tiles of 32 × 32 SIMD-group fragments, BK 32, vectorized double-buffered decode, BF16 store with fused epilogues | For commands of 768 rows or more, Stark now uses `grouped_nax.metal`: the MLX-Serve tile structure (96 × 64 on 192 threads with 128 loading, register-filled 16 × 32 × 16 operations with vector activation loads from device, double-buffered dequantized weights, one barrier per K step, full unrolling) with Stark's decode and fused epilogues; `grouped_fast.metal` below 768 rows | **Adopted** in `4314ac51` ([INKFASTPRE-07][nax]): bit-identical, replays hash-identical on both artifacts, paired 8,800-token prefill on the original 1.130× (3,895 → 3,346 ms, 8/8); 8-bit three complete pairs 1.126 to 1.140 before a pause cut the run. The first MLX-Serve-structured port (round 3, tensor views on both operands) stays **rejected** ([INKFASTPRE-03][fp3]), and the first build of the new design was 6 times slower until its loops were fully unrolled. On a captured visit MLX-Serve's bare product is still about 6 ms faster on gate/up (13.4 against 19.4 to 22.5 ms); the fused epilogue's share is not measured (the plain-store diagnostic was built and never run before the wind-down). Half-precision scale decode untested. `relaxed_precision` is set in both; not a difference. |
| E4 | Gate and up in one gathered call over the interleaved bank (N 2,304), down reads the SwiGLU output already sorted; un-sort fused into `sum.metal` with the weighted sum, conv and residual | Same two products; fused epilogues | Not a material difference after INKFASTPRE-01. |
| E5 | Sorted copy of the expert input rows: MLX `argsort` plus `take`, about 300 MB per visit at 8,192 rows (`layers.zig:448-467`) | Run-grouped ingress without a full sorted copy | Not a Stark cost. |
| E6 | — | Expert ceiling: production NVFP4 product = decode plus tensor work back to back (9.2 ms per visit against 4.7 tensor-only and 4.8 decode-only); weights pre-decoded to BF16 in device memory are slower (9.8 ms) | Measured ([INKFASTPRE-04][fp4]). The gathered kernel of E3 interleaves decode and tensor work in the same SIMD groups as MLX-Serve does, which is where its gain came from ([INKFASTPRE-07][nax]). |
| E7 | — | Weight-stationary NVFP4 (`grouped_stationary.metal`): each weight block decoded once per up to 256 rows of an expert | **Rejected at 2,048-row commands** (gate/up 1,368 → 1,683 ms, wall 4.34 → 4.99 s, hash-identical; occupancy) ([INKFASTPRE-04][fp4]). Its test at 8,192-row commands (about 512 rows per expert) was not run; the gathered kernel (E3) took that slot. |
| E8 | MXFP8 prefill (4 visits) through stock MLX `gather_qmm(mode "mxfp8")` (`gather_qmm_nax.zig:43-44`) | Unguarded tensor MXFP8 since INKFASTPRE-02 (was certified with source-order replay) | **Adopted** ([INKFASTPRE-02][fp2]): 1.169× (4K) and 1.174× (8,800) on the engine. |
| E9 | No NVFP4 guard or replay | NVFP4 per-K-step guard and replay removed | **Adopted** ([INKFASTPRE-01][fp1]): 1.325× at 4K, still bit-exact; the guard never changed a result on nine captures. |

### 2.4 Dense projections

| # | MLX-Serve | Stark | Status and effect |
| --- | --- | --- | --- |
| P1 | Decode at one row: MLX affine `quantized_matmul` (qmv) on the 8-bit pack; BF16 `x @ W.T` on the original (`sdk/bases/blocks/linear.zig:39-75`, `qmatmul.zig:90-195`); verify rows 2 to 6 use a split-K verify kernel (`qmatmul.zig:169-172`) | 8-bit: `affine8.metal`, dequantize inside the unchanged K-lane order; dense 2.90 ms at one row (1,373 MB, 473 GB/s) | Stark near 80 to 90 percent of the measured 513 to 586 GB/s ceilings at one row; no gap expected (inferred). At four rows Stark's 8-bit dense is compute- or latency-bound (4.27 against about 2.4 ms); a four-row kernel is **stopped**, a candidate if work resumes (section 6). |
| P2 | Prefill on the 8-bit pack at 2,048 rows and above: dequantize each weight to a BF16 transient, then MLX's BF16 steel or NAX GEMM (`prefill_dq_gemm.zig:15-31, 68-107`); the 608-row tail uses affine `quantized_matmul`; original uses `x @ W.T` | `dense_fast.metal`: BF16 tensor operands from device, one F32 accumulation, 128 × 64 tiles; the 8-bit artifact keeps a resident BF16 copy for prefill | **Adopted** ([INKFASTPRE-03][fp3]): dense stage about 1,290 → 687 ms at 8,800 tokens, about 14 TFLOPS. The earlier 175 ms FLOP floor was wrong: a decomposition on synthetic operands at 8,192 rows measured about 2.7 ms per projection in production against about 1.8 ms tensor-only and about 2.1 ms loads, so dense runs at about 1.5 times its tensor floor and the lever is overlapping loads with tensor steps, worth about 150 ms at 8,800 tokens if perfect (estimate) ([INKFASTPRE-06][fp4]). **Stopped**, a candidate if work resumes (section 6). |
| P3 | Projections joined (q, gate, r and sink; k and v on sliding first pass) | Not joined in prefill | **Not started.** The dense decomposition says joining projections does not touch the real limit (load and tensor overlap); the source-comparison estimate of 300 to 450 ms for the whole dense gap predates it. |
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
| Q1 | **No expert-input quantization** on any path: decode reads BF16 inputs into F32 registers (`ups.metal:26`, `downs.metal:31`); prefill NVFP4 loads BF16 activations as register fragments; no `mlx_quantize` call at run time. This changes the model's semantics against the publisher's four-over-six activation quantization; MLX-Serve has no reference check of its own. | Publisher's NVFP4 activation quantization (`INK_EXPERT_INPUTS_QUANTIZED 1` in `experts.metal`, Rust `QUANTIZE_EXPERT_INPUTS`); MXFP8 keeps its own activation quantization under both settings | **Adopted** in `8b550c9d` (`INK_EXPERT_INPUTS_QUANTIZED 0`; the exact reference still quantizes) ([INKROUND-09][quant]). Decode about −0.8 ms per step and −0.67 to −0.81 ms per verify round on both artifacts (12 pairs). Pooled natural text, 2,580 rows against the exact quantized original: original 73.76 percent next-token accuracy against 73.64, 99.38 percent confident agreement, mean KL 0.0100; 8-bit 73.99 percent, 99.43 percent, KL 0.0116 (on the final build that the standard run timed: original 73.57 and 99.22, 8-bit 73.84 and 99.28, both passing). Prefill level within noise (4,150 tokens 1.009×; 8,800 inside a 0.81 to 1.19 range). The activation pass still runs as a copy; removing it from wide prefill was built as a candidate and reverted unmeasured at the wind-down ([wind-down][wind]), and the decode-side fold was rejected (D7). |
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
| C1 | Default chunk 8,192 rows (`engines/mlx/generate.zig:106`), tails under 512 merged; 8,800 tokens run as 8,192 + 608 in two lazy forwards; larger chunks are worth about 0.3 s (7 percent) to MLX-Serve at 8,800 against 2,048 ([INKPREATTR-01][attr]) | 2,048-row commands until 2026-11-23; width API `Session::set_batched_prefill_width` for 2,048, 4,096 or 8,192 rows with reserves 1,344, 2,317 and 4,263 MiB; snapshots stay aligned to 2,048 | Engine: 8,192 against 2,048 rows 1.033× (8/8) in the first screen and 1.052× (8/8, hash-identical) paired on the current default; 4,096 1.021× (5/8) ([INKFASTPRE-03/04][fp3], [INKFASTPRE-04 follow-up][fp4]). Wide commands also enable the gathered NVFP4 kernel (E3). **Adopted in the worker** ([INKWIDTH-01][width]): `source_command_rows` picks the widest width memory admits. Side effect: cancellation is checked between commands, so a cancel on a long prompt can take about 3.5 s instead of about 1 s. |
| C2 | Unscoped and scoped requests see the same chunks | Unscoped requests ran their whole bulk in 8,192-row chunks from position 0, merging the remainder after the last 2,048 boundary into a wide command | **Fault found and fixed** ([INKWIDTH-02][width]): cache on against cache off differed in 34 of 37 validator cases on both artifacts (router rounding differs across the 1,024-row class). `unscoped_source_split` makes unscoped requests split at the same boundary. Revalidated for exactness on the wide-command worker: cache on equals off in 37 of 37 cases on both artifacts ([INKAFFINE8-04](2026-11-23-inkling-mini-affine8-default.md)); the validator's reuse-count rule then matched in 15 of 37, a gap that predated wide commands; `7e73e180` closed it (45 of 45 on both artifacts). |
| C3 | — | Source kernels from 16 rows on the default path (65 for the exact reference) | **Adopted** ([INKFASTPRE-03][fp3]). |
| C4 | — | Tail fold (merging the short tail into the bulk) | Code in the tree, **parked** (`SOURCE_TAIL_ROWS = MIN_ROWS`, does nothing); **stopped**, a candidate if work resumes (section 6). |
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
| K1 | MLX 8-bit pack: attention, embedding and head affine 8-bit, group 64; experts original bytes; router upcast to F32; four shards with mlx-lm names | Stark 8-bit artifact: the same 178 tensors quantized by a byte-identical quantizer, Stark names, BF16 router, one file, BF16 copies kept resident | **Default by owner decision of 2026-11-23**; the converter and engine support are committed (`61501bf5`, `2a1d72a2`), the catalog and preparation default is done ([INKAFFINE8-04](2026-11-23-inkling-mini-affine8-default.md)); prefix-reuse exactness and rule conformance (45 of 45 after `7e73e180`), matched image-on against image-off text speed and tool calling are met; the app end to end is not done ([INKAFFINE8-01 to -03][a8]). Standard result on this pair in [INKENGINE-06][std]. Engine: step 18.44 → 15.14 ms, verify rounds 2.9 to 3.4 ms faster. Memory 22.76 GB against 21.09 GB (no saving, copies resident). Pooled natural-text quality passes the bar (section 7). |
| K2 | Inkling Small is a separate plugin (`plugins/inkling/`) that runs serially; Inkling 2 Mini is `plugins/inkling2/` | — | The earlier web-based reading confused the two; corrected by the source comparison. |
| K3 | MLX-Serve 8-bit pack loading needs all four shards | Harness shard preflight | Adopted in the harness after the first run lost its 8-bit arm ([INKENGINE-02][eng]). |
| K4 | — | Stark 8-bit artifact must load images to be timed by the standard benchmark | The arm was refused twice (`stark_images_not_loaded`, runs 3 and `20261123-000711-stark`) until the converter recognized the affine8 layout as the Inkling image family. **Fixed** (coordinator change in `inkling_prepare.py`, `vision_runtime.py`, `inkling_affine8_artifact.py`). |

### 2.11 Measurement

| # | MLX-Serve | Stark | Status |
| --- | --- | --- | --- |
| M1 | Published decode only: llmprobe `--bench-only`, warmup discarded, median of 3, its own code prompt; M4 Max, M5 Ultra and an RTX box; no M5 Max table, no Inkling 2 Mini row and no prefill figure (`benchmarks.md:1-9`) | Every figure on this page is Stark's own measurement on this M5 Max | The often-quoted 116 tok/s is MLX-Serve's launch claim on M4 Max with its 8-bit pack; it was not reproduced here (best MLX 8-bit standard-row cell 113.1 in run 2, 107.6 in run 4, 109.2 in run 5). |
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
| History depth planner as the worker default (floor 1, capped by the requested count) | Simulated +8 percent on 8-bit prose and +6 percent on original prose against fixed depth 3, level on code, −1.5 to −3 percent at 4K and 8K; tuned later (row below) | Trace simulation, worker gates | [INKROUND-06][round] |
| BF16 query operands in local prefill attention; BF16 global attention with F32 scores and accumulation; global query tile 1,024 with causal skip | Paired 1.018× (wide range); natural-text gate passed on eight inputs (+0.19 points accuracy, 99.28 percent confident agreement) | Engine, quality | [INKFASTPRE-05][fp4] |
| Commit-check gates on the committed tree | Exact reference exact; drafting on equals off at 73 to 16,000 on both artifacts; worker smoke on both | Correctness | `runtime/commit-check/` (summary `8cb9cd12821508c4`) |
| NVFP4 expert inputs unquantized by default (`8b550c9d`) | Decode about −0.8 ms per step and per verify round on both artifacts; prefill level; natural text: original 73.76 against 73.64 percent, 8-bit 73.99 percent, both above 99.3 percent confident agreement | Engine, quality | [INKROUND-09][quant] |
| Stark 8-bit artifact as the default model choice (`222a9436`) | Offline preparation reproduced the pinned target hash; worker smoke passed | Compatibility | [INKAFFINE8-04](2026-11-23-inkling-mini-affine8-default.md) |
| Planner tuning (`1206760e`) and hold rule (`4314ac51`) | Trace replays within 1.5 percent of the best fixed depth on 14 runs; worker ABBA against fixed 3 mean 1.007 (prose 1.088, c8000 0.976) before the hold rule; the hold rule's worker effect is not measured | Trace simulation, worker | [INKROUND-11 and -12][plan] |
| Gathered NVFP4 prefill kernel for commands of 768 rows or more (`4314ac51`) | 1.130× paired at 8,800 tokens on the original (3,895 → 3,346 ms, 8/8); 8-bit 1.126 to 1.140 on three complete pairs; bit-identical | Engine | [INKFASTPRE-07][nax] |
| Paired command width on the current default | 8,192 against 2,048 rows 1.052× (8/8), hash-identical | Engine | [INKFASTPRE-04 follow-up][fp4] |
| Prefix-reuse grid advertised as 2,048-token chunks with a 4-token tail; validator models default drafting (`7e73e180`) | Validator 45 of 45 on both artifacts (was 15 of 37 on its reuse-count rule) | Lifecycle | [INKAFFINE8-04](2026-11-23-inkling-mini-affine8-default.md) |
| Planner and schedule-validation unit tests compiling (`b2a5a88f`) | 3 of 3 and 6 of 6 pass | Tests | [wind-down][wind] |
| The first standard result | Three repeats, quality attached, all four arms standard results; default pair ahead on three rows, behind on novel prose, unresolved on five | Server | [INKENGINE-06][std] |

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
| First MLX-Serve-structured NVFP4 prefill kernel (round 3 `grouped_nax.metal`) | Gate/up 1,514 to 1,796 ms against 1,477 to 1,611 for the incumbent at 8,800 | Not faster at BM 64, 96 or 128. It used tensor views on device and threadgroup memory for both operands; the second design, which fills register fragments with vector activation loads, keeps only dequantized weights in threadgroup memory and unrolls fully, was 20 to 35 percent faster per visit and is now the default for wide commands ([INKFASTPRE-07][nax]) | [INKFASTPRE-03][fp3] |
| First build of the second gathered kernel | Gate/up 140 against 24 ms, down 72 against 13 ms per visit, bit-identical | Fixed-size fragment loops did not unroll, so accumulators and fragments went to the stack; fixed with full unrolling | [INKFASTPRE-07][nax] |
| Fold the expert-input pass into the decode and verifier products | Bit-exact; single step 0.15 ms slower, verifier 0.34 / 0.83 / 1.2 ms slower at 2 / 3 / 4 rows, every pair slower | Widening BF16 to half inside the products repeats per output channel; kept off as a diagnostic control | [INKROUND-10][quant] |
| Fixed-split (width-independent) prefill router | 2 of 1,768,704 selections changed; worst row 104.6 percent against an 87.3 percent control maximum, mean 19.2 against 18.6 | Failed the noise-floor gate | [INKFASTPRE-03][fp3] |
| First split-consistency check mixing widths 333 to 1,500 rows | Failed for the exact path too | Source arithmetic switches at 1,024 and 2,048 rows; check rewritten by width class | [INKFASTPRE-02][fp2] |
| Weight-stationary dense | 713 against 670 ms | Activations re-read per 64-column tile dominate, not weights | [INKFASTPRE-04][fp4] |
| Weight-stationary NVFP4 at 2,048-row commands | Gate/up 1,683 against 1,368 ms; wall 4.99 against 4.34 s | Occupancy: about 128 rows per expert fill half of a 256-row tile. Its 8,192-row test was not run; the gathered kernel took that slot | [INKFASTPRE-04][fp4] |
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
| First adaptive-depth planner version | +8 percent against fixed 3 on 8-bit prose, but lost to fixed 1 on prose and fixed 2 on explain | Tuned (slower estimates, hysteresis, probing) | [INKROUND-06][round] |
| Planner as first committed (prior 0.7 / 0.6 / 0.6) | Started long-context requests at depth 2 and took most of a 256-token window to return to 3; worst trace replay 0.968 (original c4000) | Retuned in `1206760e` | [INKROUND-11][plan] |
| Tuned planner under a strict no-loss rule | Worker ABBA c8000 0.976, code 0.986 against fixed depth 3 | Measured-cost noise talks it down from depth 3 (reproduced with 10 percent noise in replay); committed anyway for the prose gain, then the hold rule | [INKROUND-11 and -12][plan] |
| Hold rule combined with hysteresis 1.08 or 1.10; hysteresis alone at 1.10 to 1.15; hold at 0.75 or 0.78 | Without noise 8-bit explain 0.952; original s73 0.956 to 0.968; up to 10 percent lost on original c4000; no better than 0.70 | Side effects; only the 0.70 hold was adopted, and the code loss (0.984) remains | [INKROUND-12][plan] |
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
| Unscoped wide prefill | Cache on against off differed in 34 of 37 cases; fixed, and cache on equals off in 37 of 37 on the fixed worker | [INKWIDTH-02][width], [INKAFFINE8-04](2026-11-23-inkling-mini-affine8-default.md) |
| Decode-only unquantized-input quality run | Failed: diagnostic control required a verifier activation kernel absent without a draft; fixed. Superseded by the pooled joint comparison on both artifacts that decided the default | [INKROUND-07][round], [INKROUND-09][quant] |
| Output hashes differed between two planner worker builds | First blamed on the gathered prefill kernel; the decode workstream then read the define embedded in each binary: the 09:52 build was quantized, the 10:26 builds unquantized. Expected change, not a defect | [INKROUND-09][quant] |
| `crates/stark-generate/src/inkling.rs` briefly held fixed depth (about 10:24 to 10:27) after a killed comparison run | Restored and checked with `cmp`; `1206760e` was committed after the restore | [INKROUND-11][plan] |
| Planner unit tests | Never compiled until the wind-down: the plain `#[test]` attributes in `depth_planner.rs` and `schedule_validation.rs` broke the library test build (the framework needs `#[stark_test]`), and the gate run also lacked the `cpu` feature. Attribute fix: 3 of 3 planner tests pass (fix uncommitted at the time of writing) | [INKROUND-12][plan], [INKROUND-13][wind] |
| `Path` not in scope in `grouped_pilot.rs` | Broke shared builds briefly; fixed | [INKFASTPRE-07][nax] |
| Compile errors from concurrent edits (`grouped_pilot.rs` private `register`; `decode_stages.rs` ambiguous integer) | Fixed | [INKFASTPRE-04][fp4] |
| Data volume filled (167 MB free) at about 08:25 on 2026-11-23 | A build failed; duplicate build caches and old raw traces deleted; no source truncated | coordinator notes |
| Busy-host guard tripped by brief cargo invocations outside the lock | Benchmark refused to start several times | coordinator notes |
| Joint target-plus-draft state hash under `--exact-prefill` differed from `fused2-build` | Resolved: the INKFASTPRE-01 build, the round-3 final and the re-check build agree; the intermediate fusion build was the outlier | [INKFASTPRE-03][fp3] ("Joint draft-state hash, resolved") |
| `--paired-control` hangs on the current tree | A paired run of two launch merges hung and held the shared GPU lock for about 13 minutes (15:23 to 15:36) until killed; with both merges off it still hung on the known-good `short_attention` control and was killed by a 150-second alarm in the decode workstream's private batch wrapper (the harness itself has no deadline), where it had taken about 4 seconds on the older tree. Suspects (by the decode workstream's report): the logit-agreement block added to `measure_paired_control` that day, then the `4314ac51` prefill changes. Not bisected; the diagnostic is broken on the committed tree. The scoreboard, worker smoke and worker tests complete on the same commit, which does not prove the worker path free of the cause | [wind-down][wind] |
| Teardown of the killed 23 GB engine process overlapped an image timing job | Worker decode read 14 to 17 tok/s for three requests, then recovered; the pair was discarded | [wind-down][wind] |
| The shared lock directory disappeared at about 15:55:40 while a timing job held it; a cargo test and a build then ran alongside the job | Set discarded. Cause established: while winding down at about 15:55:30 the decode workstream killed its batch with `pkill`, which also terminated its lock-script wrappers; a wrapper that had already released the lock still had its exit trap armed, and the trap removed the lock directory without checking the owner, deleting the lock a timing job had taken at 15:55:26. Fixed in the coordinator's private lock script (not in the repository): a wrapper releases only while the owner file still names its own process | [wind-down][wind] |
| The Mac rebooted at about 16:13 during the owner's pause; the session scratch directory came back empty | Helper scripts, the shared lock, build caches and the gated `4314ac51` binaries were lost; a standard run started at 15:58 had already been stopped for the pause after two of twelve arm sessions and is unusable. Tooling recreated under `runtime/` and the final build rebuilt and gated | [wind-down][wind] |
| `s4096-01` reported not exact under `--exact-prefill` by the 8-bit workstream | Resolved: that run used the source-holdout capture of the same name, taken on a 512-row schedule; all four stock captures are exact | [INKFASTPRE-03][fp3] ("All four stock captures exact") |

## 5. Stopped and still open

Optimization stopped on the afternoon of 2026-11-23 by owner decision: after run 5 ([INKENGINE-05][eng], one repeat) put the default 8-bit pair level on prefill, within 2 percent on the short standard rows and ahead at 8K and 16K, the owner judged the standing good enough and set aside the novel-prose gap (0.91). Nothing is in progress in the decode or prefill workstreams. Their last attempts were reverted unmeasured ([wind-down][wind]): two launch merges (router join into route selection, about 48 fewer launches per step and per round; NVFP4 up writing the down inputs, 44 fewer per step), whose first paired run hung; and three prefill items (removing the expert-input pass for wide prefill, fusing the MLP convolution with the residual add, the plain-store epilogue diagnostic). The scope stands: one user is the goal; the 16-user burst is out of scope.

Status of the coordinator's items at the end of the work (details in [INKAFFINE8-04](2026-11-23-inkling-mini-affine8-default.md)). Nothing is released or installed, and the adapter still negotiates 94 W:

| Item | State | What decides it |
| --- | --- | --- |
| Three-repeat standard benchmark run on the committed build | **Done**: [INKENGINE-06][std], eligible, all four arms standard results | — |
| Prefix-reuse rule conformance (advertised 1,024-token reuse chunks against 2,048-row snapshots; the validator's drafting expectation) | **Met** after `7e73e180`: 45 of 45 validator cases on both artifacts | Validator passes on the current default, both artifacts ([INKAFFINE8-04](2026-11-23-inkling-mini-affine8-default.md)) |
| Matched image-on against image-off text speed on the 8-bit artifact | **Met**: two clean off, on, on, off sets on the final build, outputs identical, image-on over image-off decode 1.002 and 0.995, prefill 1.018 and 1.000 (the earlier 0.93 to 0.97 interim came from contaminated sets and is superseded) | [INKAFFINE8-04](2026-11-23-inkling-mini-affine8-default.md) follow-up |
| Planner unit-test attribute fix | **Done**: `b2a5a88f`; planner 3 of 3 and schedule validation 6 of 6 pass on the final build | — |
| Tool calling on the 8-bit artifact | **Met**: all four `validate_tool_calling.py` checks pass | — |
| CI running the benchmark tests | `.github/workflows/ci.yml` untouched | — |
| The app end to end for the 8-bit default (catalog install, upgrade of an existing original install, image check, chat) | Not done | [INKAFFINE8-04](2026-11-23-inkling-mini-affine8-default.md) |
| `--paired-control` hang | Diagnostic broken on the committed tree; not bisected | Bisect, or replace the logit-agreement block with worker output-hash comparison |

## 6. Candidates if work resumes

Estimates are the ones each item carried when it stopped; none was measured.

| Candidate | Estimate | Estimated by |
| --- | --- | --- |
| Fix or bisect `--paired-control` first (drop the logit-agreement block, compare worker output hashes instead) | Prerequisite for paired decode timing | Decode workstream |
| NVFP4 up product also writing the down product's inputs (merge B), alone first | 44 fewer launches per step | Decode workstream |
| Router join folded into route selection (merge A) | About 48 fewer launches per step and per round | Decode workstream |
| Fold the slot-3 controller norm pair into `ink_verify_attention_mix` on later visits | About 24 fewer launches per step and per round | Decode workstream |
| Pipelined plain steps and folding the commit into the next draft command | 0.3 to 0.8 ms per plain step; 0.3 to 0.45 ms per round | Source-comparison agent; worker-loop study |
| The draft command's fixed cost (about 3 ms per round, almost all the fixed block forward) | Matters most on low-acceptance text such as prose | Decode workstream |
| A four-row 8-bit dense kernel | 4.27 ms against about 2.4 ms at four rows | 8-bit workstream |
| Remove the expert-input pass from wide prefill (direct kernels in `grouped_nax.metal` reading token rows through the route order) | About 136 ms instrumented at 8,800 tokens, expected hash-identical; a candidate binary had compiled | Prefill workstream |
| Fuse the MLP convolution with the residual add in prefill | Expected hash-identical; built, never measured | Prefill workstream |
| Plain-store diagnostic for the gathered kernel's epilogue; then the rest of the NVFP4 gap to MLX-Serve's bare product (about 6 ms per visit on gate/up); half-precision scale decode | Not quantified | Prefill workstream |
| Paired 8-bit timing of the gathered kernel with four complete pairs | Three pairs exist (1.126 to 1.140) | Prefill workstream |
| Tail fold; elementwise fusion in prefill (about 430 ms of passes); overlap dense loads with tensor steps | About 150 ms if overlap were perfect (estimate); the others not quantified | Prefill workstream |
| Remaining expert tail and controller fusion (output gate in the projection, SwiGLU in up, one input kernel per visit) | 0.5 to 1.5 ms per token | Source-comparison agent |
| Shrink the four-row verify expert stage (7.9 ms of dots on s73) toward the 8.29 ms gather ceiling for the whole stage | Not quantified | Decode workstream |
| Wider verification (4 or 5 drafts) for long context | +8 to +12 percent at about 8K, assuming 0.88 conditional acceptance holds | Decode workstream |
| A planner rule that removes the code loss (0.984) without the side effects of stiffer hysteresis | Not quantified | Decode workstream |
| TF32-class prefill router | Part of 200 to 400 ms together with the expert-input passes at 8,800 | Source-comparison agent |
| Half-precision (not BF16) prefill attention staging as in `relattn` | Not quantified; BF16 returned 1.018× | Source-comparison agent |
| Non-blocking prefill command submission | 0.3 to 0.5 s worker overhead at 8,800 | Source-comparison agent; a second packed workspace would be needed |
| Affine draft vocabulary head instead of the 440 MiB INT8 bank; drop resident BF16 copies of the 8-bit artifact | Memory, about 1.5 GB | 8-bit workstream |
| Confidence cut, if the draft command becomes much cheaper | — | Decode workstream |
| Isolate the 14 to 15 ms host gap between prefill commands seen with drafting on | Small at 8,800 | Prefill workstream |

## 7. Measurement rules learned and open questions

### Rules

- **Pair and lock.** On this host every build and GPU job runs behind one lock script. Time only in base, candidate, candidate, base process pairs inside one lock hold, at least four pairs, and report the median of per-pair ratios. The same configuration moved by up to 25 percent between arms in one session; absolute times drifted 20 to 30 percent over long sessions; the first request after load is consistently faster. ([INKFASTPRE-01][fp1], [INKPREATTR-01][attr])
- **Keep session tooling under `runtime/`.** A reboot at about 16:13 emptied the scratch directory and took the lock script, build caches and gated binaries; tooling now lives in `runtime/session-tools`, `runtime/build-cache` and `runtime/final-build`. Check for other sessions' builds before a timed run ([wind-down][wind]).
- **Lock ownership.** A lock wrapper must release only the lock it owns, and a timed job should check the lock owner at its end as well as at its start and be discarded if the owner changed. On 2026-11-23 a killed wrapper's armed exit trap deleted a lock that a timing job had just taken, and a build ran alongside the job ([wind-down][wind]).
- **Kill a model process and wait for it to exit before timing.** The teardown of a killed 23 GB engine process overlapped an image timing job and held worker decode at 14 to 17 tok/s for three requests ([wind-down][wind]).
- **Give GPU diagnostics a deadline.** A hung `--paired-control` run held the shared lock for 13 minutes before it was killed ([wind-down][wind]).
- **Power.** The 140 W adapter negotiates 94 W (20 V, 4.69 A) throughout and the battery drains under GPU load while "charging" (100 to 87 percent in about 90 minutes). The owner has been told the likely fix is the MagSafe 3 cable. It has not been shown to limit speed.
- **Instrumented sub-stages are upper bounds.** Splitting the encoder to time small stages inflates them (the router's 1.11 ms). Judge only by paired uninstrumented timing. ([INKROUND-04][round])
- **Synthetic captures do not measure quality.** The oracle and capture prompts are near-uniform: the model predicts their next token 0 to 2 times in 34 (only the 73-token prompt is natural, 18 of 34). Flip rates and draft acceptance on them say nothing about quality or real-traffic speed; they remain valid for exactness and self-consistency. Use natural text (`runtime/inkling-natural-text-2026-11-22/`, repository prose and source, 300 to 8,000 tokens) and chat-template inputs for acceptance. ([INKROUND-08][round])
- **The quality bar** ([`docs/BENCHMARK.md`](../BENCHMARK.md)): pooled next-token accuracy not more than 2.0 points below the reference and at least 98 percent top-1 agreement where the reference's top-1 probability is at least 0.5. Pooled over 2,580 rows: default prefill against exact prefill 73.41 against 73.64 percent, 99.28 percent confident agreement; Stark 8-bit against original 73.41 against 73.41 percent, 98.92 percent. Both pass. After the expert-input default changed (`8b550c9d`), against the exact quantized original reference: original 73.76 percent with 99.38 percent confident agreement, 8-bit 73.99 percent with 99.43 percent ([INKROUND-09][quant]). On the final build that the standard run timed: original 73.57 percent with 99.22, 8-bit 73.84 percent with 99.28; these final-build figures belong to the committed code (the earlier probe's tree is not identified; see INKROUND-09's follow-up).
- **Noise-floor gate** for inexact changes against an exact incumbent: candidate inside the span of single-ulp perturbation trials and its mean at most twice the control median. A sensitivity comparison, not a quality measure. ([INKFASTPRE-02][fp2])
- **Match the reasoning mode and rendering**; check identical input token counts across arms; record that MLX-Serve forces 192 decode tokens. ([INKENGINE-01][eng])
- **Interleaved real arms are too noisy for policy decisions**; simulate policies on traces with costs measured in the same session at the same positions, then confirm once at worker level. ([INKROUND-06][round])
- **Read what the other engine loads and claims**: MLX-Serve skips the draft head here; its published numbers are decode only on other machines. ([mlxsrc])
- **A server run with one repeat in the legacy layout is not a standard result**; the standard needs three blocks, block layout and an attached passing quality result. ([`docs/BENCHMARK.md`](../BENCHMARK.md))

### Open questions

- **Prefix-reuse rule conformance.** Resolved by `7e73e180`: the worker advertises 2,048-token reuse chunks with a 4-token tail and the validator models default drafting; 45 of 45 cases pass on both artifacts ([INKAFFINE8-04](2026-11-23-inkling-mini-affine8-default.md)).
- **The `--paired-control` hang.** Broken on the committed tree since the logit-agreement block or the `4314ac51` prefill changes (suspects by the decode workstream's report); not bisected ([wind-down][wind]).
- **Power adapter.** Whether the 94 W negotiation limits sustained speed.
- **CI.** The new benchmark tests are not in the CI workflow.
- **The remaining NVFP4 gap to MLX-Serve.** The second structural port closed most of it ([INKFASTPRE-07][nax]), and at server level prefill is level in run 5 (1.00 and 0.99 on the 8-bit pair, one repeat); MLX-Serve's bare product is still about 6 ms per visit faster on gate/up, and how much of that is Stark's fused epilogue is not measured.
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
[quant]: 2026-11-23-inkling-mini-expert-inputs-default.md
[plan]: 2026-11-23-inkling-mini-planner-tuning.md
[nax]: 2026-11-23-inkling-mini-gathered-nvfp4.md
[wind]: 2026-11-23-inkling-mini-wind-down.md
[std]: ../benchmarks/2026-11-23-inkling-mini-standard-benchmark.md
