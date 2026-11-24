# 20261122-INKFASTPRE-03 · Inkling 2 Mini prefill round 3: dense kernel, packed draft commit, mask removal and width independence

## Identity and state

| Field | Value |
| --- | --- |
| Attempt ID / registered date / executed date | `20261122-INKFASTPRE-03` (series registered 2026-11-22) / registered 2026-11-23 / executed 2026-11-23 |
| Kind | Kernel and precision change (prefill dense projections), draft-cache commit, workspace memory, command geometry; two rejected kernel attempts |
| Attempt state | Completed |
| Result and acceptance criterion | Passed: Stark self-consistency, the exact reference under `--exact-prefill`, and the [noise-floor gate](2026-11-22-inkling-mini-unguarded-mxfp8-prefill.md#why-the-gate-changed). Paired engine-diagnostic timing about 1.11× over the previous default. The fixed-split router failed the gate; the MLX-Serve-design NVFP4 kernel was not faster. One open item: a joint target-plus-draft state hash difference under `--exact-prefill` (below). |
| Evidence scope | Full-model numerical against pinned captures with the perturbation control, full-model self-consistency, row-independence checks, paired engine-diagnostic timing, workspace accounting. No worker timing; no natural-text comparison of default against exact prefill. |
| Promotion decision | Landed steps promoted to the default prefill path within the source-path scope; the command width stays at 2,048 rows |
| Disposition | Adopted on branch, uncommitted on top of `f1666090`. Not merged, released or installed. |
| Predecessor / related attempts / PR | [20261122-INKFASTPRE-01](2026-11-22-inkling-mini-exact-fast-prefill.md), [20261122-INKFASTPRE-02](2026-11-22-inkling-mini-unguarded-mxfp8-prefill.md), [prefill attribution](../benchmarks/2026-11-22-inkling-mini-prefill-attribution.md), [20261122-INKAFFINE8-03](2026-11-22-inkling-mini-affine8.md#20261122-inkaffine8-03--stark-8-bit-engine-artifact), which was merged into the same tree before the independent re-check. No PR. |

## Landed changes

1. **Dense attention projections (`dense_fast.metal`).** BF16 operands are read directly as tensor operands with one F32 accumulation over the full K, in 128 × 64 tiles of 32 × 32 SIMD-group blocks. This replaces the reference's 16 × 32 single-SIMD-group kernel, its split-K choice and the thin tilings for every source projection. The instrumented dense stage at 8,800 tokens went from about 1,290 to 687 ms; a 64-row tile measured 693 ms against 678 ms for the 128-row tile. The q, gate, relative and sink projections were not concatenated.
2. **Packed draft-cache commit in prefill (`draft_commit.metal`).** With drafting on, prefill previously encoded the draft commit once per row (mix, norm, two matrix-vector projections, RoPE and a copy per draft layer). It is now one packed pass per command: a row-batched mix and value store, the existing `ink_packed_norm` and `ink_draft_rope` over all rows, `ink_dense_fast` for the K and V projections, and the final feature mix for the last row only, reusing existing packed scratch. In one loaded target at 8,800 tokens, two alternating runs per arm, wall time went from 5,722 to 4,679 ms and the instrumented "other" stage from 1,112 to 465 ms. Drafting now costs about 0.3 to 0.5 s against the no-draft profile (4,190 ms, one run). Target state is unchanged because draft cache rows do not feed the target. The exact reference keeps the per-row commit.
3. **Explicit global mask freed on the default path.** The mask-free kernel of INKFASTPRE-02 no longer needs it; the exact reference allocates it on first use and a resize releases it. The global workspace now counts only K/V staging (8,192 bytes per key):

| Context | Global workspace before | After | Extra charged at admission, before | After |
| ---: | ---: | ---: | ---: | ---: |
| 4,352 | 168,927,232 | 35,651,584 | 0 | 0 |
| 9,216 | 357,728,256 | 75,497,472 | 188,801,024 | 39,845,888 |
| 16,384 | not reported | 134,217,728 (computed) | 467,034,112 | 98,566,144 (computed) |
| 32,768 | 1,271,922,688 | 268,435,456 | 1,102,995,456 | 232,783,872 |

   The 16,384 figures are computed from the per-key formula, not reported by the source. The fixed 127 MiB mask at 4,352 keys could also come out of the 1,344 MiB load-time reserve; that change is not recorded as made.
4. **Every packed command of 16 rows or more on the source kernels.** The 65-row threshold remains only for the exact reference. `Session::SOURCE_MIN_BULK_ROWS` is now 16 and the worker's workspace accounting tests were updated to the new figures (by the coordinating session, before its re-check).

With the dense kernel, the MXFP8 path and mask-free global attention, the default path is width-independent except the router's 1,024-row split boundary and commands under 16 rows (the three-row tail and the seed, which use the lane and singleton kernels by design). The permanent `--split-consistency` check compares mixed splits inside two width classes (1,025 to 2,048 rows: `2048`, `1200`, `1100,1900`, `1777,1234,1489,1500`; 16 to 1,024 rows: `1024`, `16,1000,64,65,500,355`, `333,17,674`) and is bit-identical on l8800, b12000 and l16000 for the default path; the exact control differs, as expected. A router selection probe (re-select with the exact router and count rows with a different expert set) is part of `--prefill-tolerance`.

## Measured, not changed: command width

At 8,800 tokens, 8,192-row commands against 2,048-row commands measured 1.033× paired (1.018 to 1.042), 8/8 wins, with identical state hashes across the two widths, which directly confirms width independence. Additional Metal allocation of batched prefill: 1,133 MiB at 2,048 rows, 2,106 MiB at 4,096 and 4,052 MiB at 8,192. The default stayed at 2,048 rows. Switching would need the engine's batched prefill reserve to rise from 1,344 MiB to about 4.3 GiB for 8,192 rows; prefix snapshots stay aligned to 2,048 rows at any width. Removing the blocking waits between commands was not attempted (it would need a second packed workspace).

## Rejected attempts

| Attempt | Numbers | Disposition |
| --- | --- | --- |
| NVFP4 product modelled on MLX-Serve's `gather_qmm_nax` (`grouped_nax.metal`): only decoded weights in threadgroup memory, double-buffered at BK 64; activations read from device memory; next K step's packed words fetched into registers before the tensor work and decoded after it; one barrier per step; epilogue on the accumulator registers. Structure taken from a reading of `ddalcu/mlx-serve` at `2fe9be834a601c70ef1a597563026340f0e722fe` (Apache-2.0, itself adapted from MLX); no code copied, source revision credited in the file. | At 8,800 tokens, gate/up and down GPU ms: current `grouped_fast` BM 64 1,477–1,611 / 720–794; new BM 64 1,514–1,736 / 740–850; BM 96 1,769–1,796 / 799–824; BM 128 1,699–1,789 / 789–844 | Not faster. Left in the tree disabled (`FAST_NAX_DESIGN = false`). Why MLX-Serve's per-row expert cost is lower remains unexplained; untested differences are register-operand fragments with explicit vector loads, half-precision scale decode and MLX-Serve's tile planner, which was not found. |
| Fixed-split router: one K pass at every command width, to remove the 1,024-row width dependence | Changed 2 of 1,768,704 expert selections against the exact router on the same inputs. Failed the noise-floor gate on the 512-row holdouts: worst row 104.6% against a control maximum of 87.3%, mean 19.2% against a maximum control trial mean of 18.6% | Rejected; the router keeps the source split-K choice (two-way split for commands of at most 1,024 rows, one pass above) |

## Artifact, source and environment

| Field | Value |
| --- | --- |
| Checkpoint | Original `thinkingmachines/Inkling-2-Mini` at `6e7c8248f75265bdd4fbab5273304243b5d8c2cb`, hashes as in the [default-router record](2026-11-21-inkling-mini-decode-default.md#artifact-and-run-controls) |
| Baseline (previous default) | INKFASTPRE-02 build, `inkling_slice` `60c51ea41b47a8943c081d6705f38df01249dd7b06c2b0e591135bd5236a3d69` |
| Agent's final build | `inkling_slice` `6ab9a824f4e73532b8ee54820564c86f3ab6b68745f214ddac8e916fd6d8219d` |
| Re-check build (merged tree, including INKAFFINE8-03) | `inkling_slice` `42faee01586f533c291d164ad71daa0133e5215df344bb6bdebb6ec71869ed41`, `stark-generate` `70316ca5e658b42c8c3354eadbdb582c91e4d8e8296cee0ddb8657902eeae416` |
| Source | Uncommitted working tree on top of `f1666090`; no patch hash recorded. New: `dense_fast.metal`, `draft_commit.metal`, `grouped_nax.metal`. |
| Precision policy | Prefill dense projections: BF16 tensor operands, single F32 accumulation over full K, BF16 store (inexact against the pinned reference). MXFP8 as in INKFASTPRE-02. Router: source split-K. Exact reference kernels behind `--exact-prefill`. |
| Host | 128 GiB Apple M5 Max (Mac17,7), macOS 26.7.2 (25G332), AC power, shared GPU lock; the host was noisy during the re-check |

## Correctness

| Check | Agent's final build | Independent re-check, merged build |
| --- | --- | --- |
| Noise-floor gate (12 trials, 9 inputs) | Pass: mean row L2 15.2%, worst row 64.8%, top-1 flips 25.0%; control spans 0–18.6%, 0–87.3%, 0–30.6%; control median 10.3% | Identical figures; pass; router rows with different experts 0 of 1,768,704. Receipt `tolerance.json` SHA-256 `136dca269f0eadf0a893a9f38ec98509573ea55c1d80093f8bc4b25c79a39633` |
| Drafting on equals off, `--measure-verify-transaction-schedule` | Pass at 73, 4,150, 8,800 and 16,000 | Pass at 73, 4,150, 8,800 and 16,000; 8,800-token replay repeated with identical hashes |
| Snapshot and boundary replays, determinism, split check | Pass | Not repeated separately |
| `--exact-prefill --full-target-schedule` | Exact on the four stock captures, cache and rows equal to the INKFASTPRE-01 build | Exact on `g4096-02` and `s4096-00` |
| `--exact-prefill --full-target-batched` | Zero unequal logits | Not run |
| `--exact-prefill --measure-verify-transaction` | 1,424 report fields identical apart from timing | Not run |
| Worker library tests | Not run (the agent could not edit the worker) | 165 passed, 0 failed, after updating the worker's workspace tests and `SOURCE_MIN_BULK_ROWS` |

The default now sits above the control median (15.2% mean against 10.3%), where MXFP8 alone sat at 2.38%. It is still inside the gate (span and at most twice the median). The gate's flip-rate and agreement figures are computed on synthetic captures; see the natural-text limit below.

**Joint draft-state hash, resolved.** Under `--exact-prefill` the joint target-plus-draft state hash (`serial_state_sha256`) of the re-check build differs from the intermediate `fused2-build` at 73, 4,150 and 8,800 tokens, while the seed logits, serial teacher and every other hash match and draft acceptance counts are identical. A follow-up run of the INKFASTPRE-01 build (`prefill-build`, where the exact path is the only path, so without the option) settled which side moved: at 4,150 tokens the INKFASTPRE-01 build, the agent's final binary and the re-check build all give `79af626da35a…`, and at 73 tokens the INKFASTPRE-01 build and the re-check build both give `4a1623f6796b…`; only `fused2-build` differs (`f04286d46658…` and `be812b734fa7…`). The exact reference therefore reproduces the original joint state, as the agent reported, and the outlier is the intermediate fusion build. What in that intermediate build changed the committed draft state was not investigated; target state, seed and teacher hashes and acceptance counts were identical in all builds. Private receipts for these comparisons are in `runtime/review3/` (scripts `review3.sh` to `review7.sh`; `summary7.txt` SHA-256 prefix `173bcfe46de2da60`).

**All four stock captures exact.** The follow-up also ran `--exact-prefill --full-target-schedule` on the two stock captures the first re-check skipped (`g4096-01-retry` and `s4096-01`): both exact in logits, top tokens, snapshot and boundary replays. The same command on the source-holdout capture of the same name (`source-holdouts/s4096-01`, the directory the 8-bit agent used) is not logit-exact, with all top tokens matching: that capture was taken with a 512-row command schedule and the worker schedule replays it in 2,048-row commands, a different width class, so exactness is not expected there. The noise-floor run replays the holdouts on their own schedule and is exact on all of them.

## Performance

All engine-only, drafting off, ABBA alternation under one lock.

| Comparison | 4,096 tokens | 8,800 tokens |
| --- | --- | --- |
| Agent: final against INKFASTPRE-02 | 1,916 → 2,126 tok/s, 1.107× (1.003–1.115), 8/8 | 1,847 → 2,103 tok/s (4,184 ms), 1.130× (1.087–1.148), 8/8 |
| Agent: dense step alone against INKFASTPRE-02 | 1.119×, 7/8 | 1.122×, 8/8 |
| Re-check: merged build against INKFASTPRE-02 | 1,920 → 2,130 tok/s, 1.108×, 4/4 | 1,852 → 1,930 tok/s median, 1.111×, 3/4 (noisy host) |
| Re-check: merged build against agent's final | 1.001× at 4,150, identical state and seed-logit hashes | 0.991×, identical state and seed-logit hashes |

The last row shows that merging the 8-bit decode work did not cost prefill speed. Against the original exact build of INKFASTPRE-01 (about 1,085 and 1,034 tok/s in that session), the agent's figures are about 2.0× at both lengths; that is a cross-session total and indicative only. Re-check pair receipts: `paired-4k.jsonl` `a3af365b…`, `paired-8800.jsonl` `7f95e767…`, `paired-final4-4150.jsonl` `54788bc9…`, `paired-final4-8800.jsonl` `8764d00b…`.

Instrumented GPU time on the default path at 8,800 tokens after the last step, 4,742 ms in total (uninstrumented wall 4,190 ms): NVFP4 dot products 2,085 ms (44%), dense attention 687 ms (14%), relative and cache attention 659 ms (local 247, global 286), other 427 ms, routing 371 ms, MXFP8 339 ms, attention and expert inputs 395 ms, NVFP4 quantization 136 ms.

## Quality limit: synthetic captures

The pinned captures and oracle prompts are synthetic and nearly uniform. On the 4,150-token input the model's top-1 equals the actual next token 0 times in 34 rows (field `teacher_next_token_top1_original` of the 8-bit quality receipt `runtime/inkling-affine8/runs/a8-quality-r4150.json`, SHA-256 prefix `ae7396d93ec1a9d9`; the other padded inputs score 0 to 2 of 34, and only the natural 73-token prompt scores 18 of 34). Agreement and flip rates on these inputs, including the noise-floor control's flip rates, therefore say little about quality on real text. They remain valid for exactness and self-consistency, and the gate remains a sensitivity comparison. A comparison of default against exact prefill on natural text has not been measured; a generalized probe (larger scoring window, confidence-stratified agreement, KL divergence) is being built.

## Not done

- No worker timing of this build; the draft-commit gain is measured in the engine only. The 14 to 15 ms host gap between commands with drafting on (seen in the attribution trace) was not isolated; the agent's per-command measure stayed at about 7 to 8 ms with or without the packed commit.
- `cargo test --lib` still does not compile on stable in this crate, so the split-pattern unit test does not run.
- One build failed while another agent was mid-edit; it was retried.
- The source tree has moved on since this record's builds: at the time of writing it contains a load-time selectable command width (2,048, 4,096 or 8,192 rows with matching reserves). That later change is not part of this attempt and is not evaluated here.

## Follow-up history

- 2026-11-23: the joint-hash item in the identity table is resolved; see "Joint draft-state hash, resolved" and "All four stock captures exact" under Correctness.
- 2026-11-23: the natural-text comparison of default against exact prefill was measured with `--quality-compare` on ten natural inputs, 258 rows each, pooled over 2,580 rows: next-token accuracy 73.41 percent (default) against 73.64 (exact), 99.28 percent top-1 agreement where the exact reference is at least 50 percent confident. It passes the quality bar of [`docs/BENCHMARK.md`](../BENCHMARK.md). See [INKROUND-08](2026-11-23-inkling-mini-decode-round.md#20261123-inkround-08--natural-text-quality-probe-and-the-synthetic-capture-finding).
- 2026-11-23: the landed changes are in commit `2a1d72a2`. Later prefill work is [INKFASTPRE-04 to -06](2026-11-23-inkling-mini-prefill-round4.md); the comparison with MLX-Serve is on the [differences page](inkling-mlx-serve-differences.md).
- 2026-11-23: the MLX-Serve-structured NVFP4 kernel rejected above was replaced by a second design in the same file, `grouped_nax.metal`, which is bit-identical, 1.130× on paired 8,800-token prefill and the default for commands of 768 rows or more: [INKFASTPRE-07](2026-11-23-inkling-mini-gathered-nvfp4.md). The rejection recorded here stands for the first design.
