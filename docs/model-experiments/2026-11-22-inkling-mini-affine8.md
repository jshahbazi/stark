# Inkling 2 Mini 8-bit affine dense weights

These attempts concern a different artifact from the original checkpoint that the other Inkling 2 Mini records qualify: the original target with its dense attention projections, embedding and head quantized to MLX 8-bit affine form, as in the author's published MLX-Serve pack. Nothing here qualifies that artifact in Stark, and no result transfers to or from the original checkpoint.

## 20261122-INKAFFINE8-01 · Feasibility of 8-bit dense weights

| Field | Value |
| --- | --- |
| Attempt ID / registered date / executed date | `20261122-INKAFFINE8-01` / 2026-11-22 / 2026-11-22 |
| Kind | Compatibility and performance feasibility (source and file reading, arithmetic estimates) |
| Attempt state | Completed as an assessment. No files edited, no cargo or GPU work. |
| Result | Planned/untested: a 10 to 17 percent decode gain is estimated, not measured (the feasibility report's headline said 12 to 17; its body and the analysis below say 10 to 17) |
| Evidence scope | Structural (pack file inspection and two dequantization error checks), source review of MLX-Serve, byte-count arithmetic |
| Promotion decision | Deferred; recommends Stark quantizing the original itself (option b below) |
| Disposition | Experiment only |
| Related | [INKAFFINE8-02](#20261122-inkaffine8-02--reproduce-the-published-pack-tensors-byte-for-byte); [engine matrix run 2](../benchmarks/2026-11-22-inkling-mini-engine-matrix.md#20261122-inkengine-02--second-run-with-the-author-8-bit-pack) |

### The pack's format

The author pack `ddalcu/Inkling-2-Mini-MLX-8bit`, pinned locally at Hugging Face revision `cb7fa052a9bbafbbeb91750edc67046e7bacd4f1`, is Apache-2.0 with `base_model: thinkingmachines/Inkling-2-Mini` (also Apache-2.0), file format `mlx`, about 18.7 GB, built by `tests/convert_inkling2.py --bits 8` in MLX-Serve.

- **8-bit tensors:** in all 24 layers `self_attn.{q,k,v,o,gate,r,sink}_proj`, plus `k/v_sconv.weight_proj` in the four global layers, plus `embed_tokens` and a separate untied `lm_head`, each [201024, …].
- **Layout:** codes are `U32` [out, in/4], four unsigned 8-bit codes per word, little-endian along the input dimension. `scales` and `biases` are BF16 [out, in/64]. The value is code × scale + bias, one scale and bias per row per group of 64. About half the scales are negative: MLX's affine mode maps the larger-magnitude edge exactly.
- **Error checks against the original:** layer-0 `sink_proj` relative L2 0.79 percent (maximum error 9.8e-4), `k_proj` 0.68 percent.
- **Unchanged content:** expert banks are the original bytes re-typed (NVFP4 on layers 0 to 19, MXFP8 on 20 to 23); router weights are upcast to F32, bit-exact; message gates, `rel_logits_proj`, convolutions, norms and the BF16 vision tower are unchanged.
- **Names and files:** mlx-lm naming over four shards and an index, against the original's single file with `model.llm.*` names.
- **Metadata:** `tokenizer.json`, `chat_template.jinja` and `processor_config.json` match Stark's pinned hashes. `config.json` does not (`2b3281…` against `8216dd…`): it renames the model type to `inkling2_loop` and adds `quantization`. There is no `hf_quant_config.json`. The index hash begins `6280a7`.
- **Draft head:** `mtp.safetensors` was byte-identical to the original by `cmp` in this attempt.

### How MLX-Serve runs it (source review, partly inferred)

Codes, scales and biases stay separate (`sdk/bases/blocks/linear.zig`); dense and affine weights go through `qmatmulBits`. That function's body was not read, so the vector-kernel description is inferred. `kernels/oproj.metal` fuses the attention gate and the 8-bit output projection, one SIMD group per output row, F32 accumulation. Multi-row inputs are split into one-row pieces so each row's bits match a one-row call; the inference is that MLX-Serve does not share dense weight reads across verifier rows, which Stark already does.

### Budget estimate

Per token, the dense weights read are about 2.58 GB in BF16 (48 visits of about 26.9 M parameters) and the readout 0.926 GB. At 8 bits a weight costs 1.0625 bytes, about 1.64 GB less per token. The dense stage takes 5.86 to 5.93 ms at three rows against 5.84 ms at one, so it is bound by weight reads at one to four rows. At unchanged effective bandwidth (about 430 to 442 GB/s), dense would save about 2.7 ms and the readout about 1.0 ms per step or round, realistically 2.5 to 3.7 ms because the small k, v, r and sink matrices may become launch-bound (a lossless BF16 repack that cut reads 24 percent got slower). With a round of about 25 to 32 ms (inferred) that is +10 to 17 percent. Confidence medium-low: the byte arithmetic is solid, the round time and small-kernel efficiency are not measured. The novel-prose gap to the 8-bit pack (62 against 89 tok/s in run 2) is a drafting gap that 8-bit weights would not close.

Memory: about 1.5 GB less (embedding, head and dense from 3.14 to 1.67 GB). Prefill: expected neutral to slightly worse, because Stark's prefill kernels would need 8-bit tiles.

### Options

- **(a) Load the author's pack:** new hash set and shard list, a multi-shard tensor map, name mapping, affine buffers, new affine one-row and two- to four-row kernels for dense and readout, an 8-bit prefill kernel, and a rebuilt or replaced INT8 draft vocabulary bank (it is built from the BF16 head). Inherits a foreign layout, the F32 router and a four-shard trust root. About 8 to 10 steps.
- **(b) Stark quantizes the pinned original itself (recommended):** derive the 8-bit tensors in the converter from the already-pinned original, keep Stark's names, the BF16 router and one file, and choose a kernel-friendly layout. Implement MLX's quantizer exactly so the bytes can be checked against the author's pack. About 7 to 9 steps.

### Qualification plan proposed

An F32 reference on the dequantized weights for this artifact, plus a separate comparison with the original to bound quantization loss; teacher-forced KLD and top-1 agreement at short, 4K and 16K; the same against MLX-Serve on the pack; drafting on equal to off; restored prefix equal to fresh; images with drafting off and text after an image; matched text speed with images loaded. Proposed acceptance: KLD at or below the pack's published noise floor (about 0.004), top-1 agreement at least 97 percent and no worse than the pack, perplexity within 0.1 percent, exact drafting and prefix equality, and speed adoption only above the playbook gate.

Risks named: small kernels may not reach the needed bandwidth (saving toward 2 ms, +6 to 8 percent); prefill and draft-bank rework could cost more than decode gains; and the bit-exact original-checkpoint path is given up for this artifact.

## 20261122-INKAFFINE8-02 · Reproduce the published pack tensors byte for byte

| Field | Value |
| --- | --- |
| Attempt ID / registered date / executed date | `20261122-INKAFFINE8-02` / 2026-11-22 / 2026-11-22 |
| Kind | Conversion correctness (quantizer reproduction) |
| Attempt state | Completed |
| Result | Passed: all 178 affine tensors in the pack are byte-identical in codes, scales and biases to the new quantizer's output from the original checkpoint |
| Evidence scope | Structural and numerical comparison of every quantized tensor; model-free tests. No model execution. |
| Promotion decision | Converter building block only; not wired into conversion |
| Disposition | New untracked files under `tools/converter/`; not committed. `inkling_prepare.py`, the catalog, artifact identities and `crates/` are unchanged. |

### Result

178 tensors (24 layers × 7, 4 global layers × 2, plus embedding and head): 1,572,323,328 codes and 24,567,552 scales and as many biases, zero differing elements. The name mapping was confirmed in both directions. Private aggregate `runtime/inkling-affine8/verify.jsonl` (178 records), SHA-256 `56d7695de9aac0fe27cc6823ffd83c6be043c1d16570ca7decc220a3b0384f27`. Run on the M5 Max in five batches under the shared lock.

### Files

- [`tools/converter/stark_convert/mlx_affine.py`](../../tools/converter/stark_convert/mlx_affine.py): the quantizer (`quantize_rows`, chunked `iter_quantized_bf16`, `dequantize_rows`, rounding and BF16 helpers, U32 little-endian packing).
- [`tools/converter/stark_convert/inkling_affine8.py`](../../tools/converter/stark_convert/inkling_affine8.py): name mapping and the `verify` and `inventory` commands, which print per-tensor aggregates only.
- [`tools/converter/tests/test_mlx_affine.py`](../../tools/converter/tests/test_mlx_affine.py): 14 model-free tests.

### Algorithm

Per row, per group of 64, in IEEE float32 from exactly widened BF16: scale = max((max − min) / 255, 1e-7); the edge with the larger magnitude (ties pick the maximum) sets the sign; q0 = round_half_away(edge / scale); if q0 ≠ 0 then scale = edge / q0 and bias = edge, otherwise bias = 0; code = min(round_half_away((w − bias) / scale), 255). Codes use the unrounded float32 scale and bias; only the stored planes are rounded to BF16 (nearest, ties to even). There is no lower clamp, as in the kernel. This is the GPU kernel `affine_quantize` in MLX's `mlx/backend/metal/kernels/quantized.h`, which `mx.quantize` runs by default; the CPU fallback in `mlx/ops.cpp` does not match the pack.

Counterfactuals on layer 0 (7 tensors, 26.9 M codes) show the details matter: half-to-even rounding changes 111,685 codes, codes from BF16-rounded scale and bias change 4,732,255, a reciprocal multiply instead of division changes 10,822, and the implementation as written changes none.

### Everything else in the pack

Of the pack's 1,559 tensors (the original has 1,244), all are accounted for and compared value by value: 680 renamed only with identical bytes (norms, `rel_logits_proj`, clip scales, queries, message-gate down projections, the router bias as `e_score_correction_bias`, the whole vision tower); 120 BF16 to F32, bit-exact (router and message-gate projections); 88 short convolutions transposed [C, 1, 4] to [C, 4, 1] with values unchanged; 48 expert code tensors re-typed U8 to U32; 48 expert block-scale tensors re-typed to U8; 40 new `global_scale` tensors on layers 0 to 19 replacing 80 `*_amax` and `*_outer_scale_denominator` tensors, equal to `amax * (2688 / denominator)` in float32 bit for bit on layers 0, 7 and 19 (the other operation order is within 1 ulp on all 20); and `message_input.query` assembled from the two cross-pass query tensors with identical values.

### Provenance and licences

- MLX (https://github.com/ml-explore/mlx), MIT, Copyright (c) 2023 Apple Inc.: `mlx/ops.cpp` and `mlx/backend/metal/kernels/quantized.h`, read over the web at `main`, which resolved to `58d91ee81a86dd9710123ec42e7ff91f6d1d21b1` at read time. Not cloned, nothing executed.
- MLX-Serve (https://github.com/ddalcu/mlx-serve), Apache-2.0: `tests/convert_inkling2.py` at `2fe9be834a601c70ef1a597563026340f0e722fe`. The rename rules and quantized-tensor pattern are transcribed as data. The script pins no MLX version, so which MLX build produced the pack is unknown; byte equality shows its kernel arithmetic matches.
- Model: `thinkingmachines/Inkling-2-Mini`, Apache-2.0. Pack: `ddalcu/Inkling-2-Mini-MLX-8bit` at `cb7fa052a9bbafbbeb91750edc67046e7bacd4f1`, Apache-2.0.
- Both new source files carry these notes in their docstrings.

### Tests and limits

- `test_mlx_affine.py`: 14 passed (rounding ties, zero and constant groups, both edge signs and a magnitude tie, clamping at 255, rejected shapes and dtypes, agreement with a scalar transcription on random BF16 rows, packing, chunked output, the rename map, and an end-to-end verify on synthetic files that detects injected differences). Full converter suite at the time: 7,999 passed, 96 skipped; ruff clean on the new files. The repository policy check was not run by that author on the new files.
- Not compared in this attempt: `mtp.safetensors`, `config.json`, tokenizer and template files (INKAFFINE8-01 reports `mtp.safetensors` byte-identical by `cmp` and the tokenizer, template and processor hashes matching).
- Assumed rather than shown by the published source, though consistent with byte equality: correctly rounded float32 division in the GPU kernel and round-to-nearest-even for the BF16 cast.

## 20261122-INKAFFINE8-03 · Stark 8-bit engine artifact

| Field | Value |
| --- | --- |
| Attempt ID / registered date / executed date | `20261122-INKAFFINE8-03` / registered 2026-11-23 / executed 2026-11-23 |
| Kind | New artifact and decode kernels: compatibility, correctness, memory and performance |
| Attempt state | Completed through engine, worker smoke and converter; qualification incomplete |
| Result | Self-consistency passed exactly on the 8-bit artifact. Engine timing: 3.3 ms faster per single-row step and 2.9 to 3.4 ms faster per verify round than the original artifact, prefill not slower. Quality against the original: 94.9 percent top-1 agreement on natural text (coordinator's re-check). |
| Evidence scope | Kernel screen and kernel correctness against a dequantized reference; full-model self-consistency; paired engine timing; one worker smoke; teacher-forced comparison with the original on synthetic and natural text. No image qualification, no prefix-reuse validation script, no worker throughput study. |
| Promotion decision | Not the default anywhere and not qualified for installation; diagnostic and opt-in artifact only |
| Disposition | Code adopted on branch (uncommitted on top of `f1666090`); the artifact exists only in an ignored runtime directory. Nothing in the catalog, desktop or app selects it. Not released or installed. |
| Related | [INKAFFINE8-01](#20261122-inkaffine8-01--feasibility-of-8-bit-dense-weights), [INKAFFINE8-02](#20261122-inkaffine8-02--reproduce-the-published-pack-tensors-byte-for-byte), [20261122-INKFASTPRE-03](2026-11-23-inkling-mini-prefill-round3.md) (merged into the same tree) |

### Phase 1: kernel screen

`--measure-affine8-dense` on s73 at context 4,352: 20 alternating pairs, GPU time per command, real state restored after prefill and four decode steps, all 48 visits, weights from the converter run on the original checkpoint. The 8-bit kernels won all 20 pairs in every cell; 64 to 512 threads per group made no difference, so 128 stays.

| Rows | Dense BF16 → 8-bit (pair-median saving) | Readout BF16 → 8-bit (saving) | Dense plus readout saved |
| ---: | --- | --- | --- |
| 1 | 4.75 → 2.63 ms (2.11, range 2.03–2.31) | 2.05 → 0.82 ms (1.23) | 3.35 ms (3.25–3.73) |
| 2 | 5.57 → 3.43 ms (2.14) | 2.04 → 0.82 ms (1.22) | 3.36 ms (3.33–3.42) |
| 3 | 5.58 → 3.71 ms (1.87) | 2.04 → 0.82 ms (1.22) | 3.08 ms (3.07–3.13) |
| 4 | 5.60 → 4.23 ms (1.38) | 2.05 → 0.84 ms (1.21) | 2.59 ms (2.53–2.64) |

The measured saving matches the INKAFFINE8-01 estimate of 2.5 to 3.7 ms. At four rows the 8-bit dense stage is limited by compute or latency rather than bandwidth (4.2 ms against 2.6 ms at one row).

Kernel correctness against the dequantized reference: the 8-bit dense kernels are bit-identical to the BF16 kernels run on the BF16 copy of the effective weights at one to four rows (0 of 5.2 M values differ). Against an FP64 product on the effective weights, relative L2 is 0.00166 with 61 to 272 values not equal to the BF16 rounding of the FP64 result, the same as the BF16 kernel on the copy; against unrounded FP32 dequantization 0.0019. Readout against FP64: relative L2 8.3e-8 with top-1 agreeing on every row (the BF16 kernel on the copy gives 3.2e-7). GPU dequantization equals the load-time CPU copy on all 1.57 G values, and the 8-bit embedding lookup equals the BF16 lookup on the copy. Quantization loss against the original BF16 weights: relative L2 about 0.005 for dense projections and 0.0055 to 0.0065 for the readout.

### The artifact

- **Builder:** [`tools/converter/stark_convert/inkling_affine8_artifact.py`](../../tools/converter/stark_convert/inkling_affine8_artifact.py) with tests in [`tools/converter/tests/test_inkling_affine8_artifact.py`](../../tools/converter/tests/test_inkling_affine8_artifact.py) (18 tests pass together with the MLX quantizer tests). It builds from the pinned original, revision `6e7c8248…`, target SHA-256 `65e60f51…`, in about 17 s.
- **Content:** the 178 MLX-selected tensors become `NAME.affine8_codes` (U8), `NAME.affine8_scales` and `NAME.affine8_biases` (BF16, group 64) under Stark's names, with codes from `mlx_affine.quantize_rows` (INKAFFINE8-02). Every other target tensor and `mtp.safetensors` are copied byte for byte. The configuration carries a `stark_inkling_affine8` binding.
- **Identity:** target 18,052,013,648 bytes, SHA-256 `c9e14848ad017467dc0deb91c6aa67022e181dbcdfe02a1b04024434bf9acddd`, in the ignored `runtime/inkling-affine8/artifact/`. `affine8-provenance.json` records the original revision and hashes, the MLX algorithm and revision `58d91ee81a86dd9710123ec42e7ff91f6d1d21b1` (MIT), the MLX-Serve tensor selection at `2fe9be834a601c70ef1a597563026340f0e722fe` (Apache-2.0) and the model licence (Apache-2.0); a `native-source.json` receipt names the layout `inkling2-mini-affine8-dense`.
- **Worker:** `crates/stark-generate/src/inkling_artifact.rs` has `Layout {Original, Affine8}` chosen from the configuration binding, with the 8-bit target hash and size pinned; the original artifact's checks are unchanged. `inkling.rs` adds 1.67 GB to memory admission for the 8-bit layout and reports the loaded artifact's hash as the image identity.

### One set of effective weights

Each weight is `bf16(fma(code, scale, bias))` computed in FP32. At load Stark builds a BF16 copy on the CPU, which prefill and the draft head read unchanged. The decode kernels (`affine8.metal`) dequantize to the same BF16 values inside the unchanged K-lane order and FP32 block sums, which is why every dense output equals the BF16 kernel's output on the copy. The readout has its own kernel whose per-row reduction does not depend on row count; single-row, verify and prefill-final readouts all use it.

Decode paths switch through a lookup from each BF16 copy to its codes (`affine8.rs`): the single-row projection and the fused visit kernel (which gained an 8-bit twin), the two- to four-row verifier projections, the readout (which also replaces the verifier's per-row cap calls), and the text and draft-token embedding lookups. The draft head still builds its INT8 vocabulary bank and fold from the BF16 copy of the 8-bit head; the 440 MiB bank stays because per-group affine weights cannot feed its INT8 tensor product, and an affine replacement was not tried. The BF16 copies of head and embedding also stay resident.

### Self-consistency (8-bit artifact)

- Drafting on equals drafting off (`--measure-verify-transaction-schedule`) at s73, r4150, l8800 (context 9,216) and l16000 (16,384): 9/9 rejection and crossing controls, depths 1 to 3 equal serial output and state, exact boundary replays at 4,096, 8,192 and 14,336. **Independent re-check** on the merged build: pass at 73, 4,150 and 8,800.
- Restored prefix: `--schedule-replay` exact at s73, r4150 and l8800. A repeated s73 run gave identical seed, teacher and state hashes.
- Worker smoke (new `stark-generate`, SHA-256 prefix `cbb6fba4`, prefix cache and vision on, a prompt of about 3K tokens): greedy output with MTP on equals MTP off; prefix reuse took 2,048 tokens and the follow-up equals the cold request; a synthetic image request completed with drafting off; the following text request reused 0 tokens, equals cold output and drafted (51 proposed, 13 accepted). This is a smoke, not image or prefix qualification.
- The original artifact on the same build, compared with `fused2-build` under `--exact-prefill`: s73 transaction schedule identical in every hash and control; r4150 schedule replay identical. The r4150 joint target-plus-draft state hash differs reproducibly while target state, seed, teacher, controls and boundary replays match; see [INKFASTPRE-03](2026-11-23-inkling-mini-prefill-round3.md#correctness), where a follow-up resolved it: the current tree reproduces the original exact build's joint state and `fused2-build`, the baseline used here, is the outlier. `--full-target-schedule --exact-prefill` on `s4096-01` reported `all_logits_exact` false for both the new and the baseline build, with an identical cache hash. The same follow-up explains it: the run used the source-holdout capture of that name, taken with a 512-row schedule, which the worker schedule replays in a different width class; the stock capture `s4096-01` is exact, as are the other three stock captures.

### Quality against the original artifact

**Synthetic captures (the agent's run).** Teacher-forced decode over the last 32 prompt tokens plus the teacher tokens, 34 or 35 rows per prompt: relative logit L2 median 0.062 to 0.123 and maximum 0.27 to 1.27, top-1 agreement 24 to 30 of 34 or 35 (70 to 88 percent) across s73, s511, r4150, l8800, l16000, `s4096-01` and `g4096-01`. Against the pinned captures over four rows each, the original artifact's own relative L2 to the capture is 0.03 to 0.38 with 8/8 top-1, the 8-bit artifact's 0.04 to 0.47 with 7/8.

These prompts are synthetic and nearly uniform: on the 4,150-token input the model's top-1 equals the actual next token 0 times in 34 (field `teacher_next_token_top1_original` of `runtime/inkling-affine8/runs/a8-quality-r4150.json`, SHA-256 prefix `ae7396d93ec1a9d9`; the other padded inputs score 0 to 2 of 34, and only the natural 73-token prompt scores 18 of 34). Agreement on them says little about real quality.

**Natural text (coordinator's re-check).** Ten inputs built from repository prose and source, 300 to 8,000 tokens (contexts 4,352 and 9,216), 35 teacher-forced rows each, 350 rows in total. Receipts `runtime/review3/natural-a8-*.json`; summary SHA-256 `df3d2cf0933a847e8bfbb5c8f234304b0e39c444e8400e6a98f9d276f1253105`.

| Measure | Result |
| --- | --- |
| Top-1 agreement, 8-bit against original | 332 of 350 (94.9%) |
| Next-token accuracy against the actual text | 8-bit 256 of 340; original 252 of 340 |
| Relative logit L2 per input | Median 0.055 to 0.106; maximum 0.143 to 0.493 |

Per input, top-1 agreement ranged from 31 to 35 of 35. No KL divergence or confidence-stratified agreement was measured; a generalized probe is pending. The acceptance criteria proposed in INKAFFINE8-01 (KLD, perplexity, top-1 at least 97 percent) have not been evaluated as such; 94.9 percent is below the proposed 97 percent top-1 figure, which was itself proposed without a natural-text baseline.

Draft acceptance on the 8-bit artifact varied by prompt: lower than the original at l16000 (26 of 69 against 39 of 55 at depth 1, single samples) and similar on the worker prompt (59 of 105 against 58 of 109).

### Performance and memory

Paired engine timing, both artifacts loaded side by side, wall ms, alternating order, same restored state:

| | s73: original → 8-bit (saving) | r4150: original → 8-bit (saving) |
| --- | --- | --- |
| Single-row step | 18.44 → 15.14 (3.34; 20/20 pairs, range 2.96–3.72) | 19.24 → 16.02 (3.29; 15/16 pairs) |
| Two-row round | 21.11 → 17.82 (3.33) | 21.70 → 18.38 (3.37) |
| Three-row round | 23.77 → 20.51 (3.27) | 24.42 → 21.18 (3.32) |
| Four-row round | 26.34 → 23.26 (3.01) | 26.92 → 23.97 (2.94) |

Every round pair favoured the 8-bit artifact. Prefill: s73 165 against 171 ms (7/7 pairs faster for 8-bit); r4150 2,449 against 2,510 ms median (5/7 pairs, wide spread from the power state). No worker throughput or engine-matrix run on the 8-bit artifact exists, and the drafted tok/s gain at the observed acceptance is not measured.

Memory: the affine kernel tensors add 1,668,243,456 bytes on top of the BF16 copies; Metal allocation after load was 21.09 GB for the original and 22.76 GB for the 8-bit artifact. The 1.5 GB saving estimated in INKAFFINE8-01 is not realised, because the BF16 copies stay resident. Load: one worker start each, 8.6 s for 8-bit and 10.8 s for the original (single samples).

### Identities

Agent binaries (SHA-256 prefixes): phase-1 example `f6055147`, phase-2 examples `29b4788d` and `1d8cf002`, `stark-generate` `cbb6fba4`, baseline `a7817f3b`. Re-check build: `inkling_slice` `42faee01586f533c291d164ad71daa0133e5215df344bb6bdebb6ec71869ed41`, `stark-generate` `70316ca5e658b42c8c3354eadbdb582c91e4d8e8296cee0ddb8657902eeae416`. Private results in `runtime/inkling-affine8/` (hashes not recorded here) and `runtime/review3/`.

### Not done

- Image qualification and matched text performance with images loaded, beyond the smoke.
- Prefix-reuse qualification through `validate_generation_prefix_cache.py`.
- Catalog, desktop and app integration; nothing selects the artifact.
- KL divergence, perplexity and confidence-stratified agreement on natural text.
- Four-row dense kernel work, an affine draft vocabulary head, and dropping the resident BF16 copies.
- Failed or invalid runs by the agent: a quality run that panicked on a row index (fixed), a hung debugger session (killed), a schedule-replay probe with a wrong token length, and one empty shell heredoc (no effect).

### Follow-up history

- 2026-11-23, natural-text quality pooled over 2,580 rows (ten inputs, 258 rows each, `--quality-compare`, original default as the reference): next-token accuracy 73.41 percent for the 8-bit default against 73.41 for the original, 98.92 percent top-1 agreement where the reference is at least 50 percent confident; per-input KL mean 0.0067 to 0.0225. It passes the quality bar of [`docs/BENCHMARK.md`](../BENCHMARK.md), which settles the KL and confidence-stratified items listed under Not done for this corpus ([INKROUND-08](2026-11-23-inkling-mini-decode-round.md#20261123-inkround-08--natural-text-quality-probe-and-the-synthetic-capture-finding)).
- 2026-11-23, images: the converter now recognizes the 8-bit target hash and prepared layout as the Inkling image family (coordinator change in `inkling_prepare.py`, `vision_runtime.py` and `inkling_affine8_artifact.py`). Before it, the benchmark refused the Stark 8-bit arm twice with `stark_images_not_loaded`; after it, the client's local image check passes. This is not image qualification.
- 2026-11-23, server timing: engine-matrix run 4 timed the Stark 8-bit arm with images loaded, one repeat in the legacy layout (initial coding 86.9, novel prose 67.8, about 8K 110.8 tok/s against 78.9, 58.4 and 100.1 for the original on the same worker binary); see [INKENGINE-04](../benchmarks/2026-11-22-inkling-mini-engine-matrix.md#20261123-inkengine-04--fourth-run-with-the-stark-8-bit-artifact). Not a standard result.
- 2026-11-23, commits: converter and artifact builder in `61501bf5`, engine and worker in `2a1d72a2`; the committed tree passed drafting on equals off at 73 to 16,000 tokens and the worker smoke on both artifacts (`runtime/commit-check/`).
- 2026-11-23, decision: the owner made the 8-bit artifact the default.
- 2026-11-23, default: catalog and preparation integration done in [20261123-INKAFFINE8-04](2026-11-23-inkling-mini-affine8-default.md); `inkling2-mini` now publishes this artifact and `inkling2-mini-original` the original. Prefix-reuse exactness is shown on this artifact (output with the prefix cache on identical to the same build without it in 37 of 37 validator cases, invalidations as documented); validator rule conformance is open. Not installed.
