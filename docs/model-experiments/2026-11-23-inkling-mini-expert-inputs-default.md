# Inkling 2 Mini: unquantized NVFP4 expert inputs as the default, and the rejected fold

Two attempts on 2026-11-23 on the 128 GiB Apple M5 Max (Mac17,7), macOS 26.7.2 (25G332), AC power with the adapter negotiating 94 W, all GPU work behind the shared lock. They cover the original checkpoint (`thinkingmachines/Inkling-2-Mini` at `6e7c8248f75265bdd4fbab5273304243b5d8c2cb`) and the Stark 8-bit dense artifact (target SHA-256 `c9e14848ad017467dc0deb91c6aa67022e181dbcdfe02a1b04024434bf9acddd`); each result names its artifact and none transfers to the other. Nothing here is released or installed. The measurements before the decision are in [INKROUND-07](2026-11-23-inkling-mini-decode-round.md#20261123-inkround-07--unquantized-nvfp4-expert-inputs); the comparison with MLX-Serve, which never quantizes these inputs, is on the [differences page](inkling-mlx-serve-differences.md).

## 20261123-INKROUND-09 · Read NVFP4 expert inputs unquantized by default

| Field | Value |
| --- | --- |
| Attempt ID / registered date / executed date | `20261123-INKROUND-09` / 2026-11-23 / 2026-11-23 |
| Kind | Precision (a semantic change to the publisher's NVFP4 activation quantization) with speed |
| Attempt state | Completed |
| Result and acceptance criterion | Passed: pooled natural-text quality on both artifacts against the exact, quantized reference meets the bar of [`docs/BENCHMARK.md`](../BENCHMARK.md) (next-token accuracy not more than 2.0 points below, at least 98 percent top-1 agreement where the reference is at least 50 percent confident); drafting on equals off and the worker smoke on both artifacts. Decode about 0.8 ms faster per step and per verify round. Prefill level within noise. |
| Evidence scope | Full-model natural-text quality (2,580 teacher-forced rows per artifact), full-model self-consistency, worker smoke, paired engine timing. No server-level or standard benchmark run on this change alone. |
| Promotion decision | Promoted to the default on both artifacts; the exact prefill reference still quantizes |
| Disposition | Committed as `8b550c9d` (2026-11-23). Not released or installed. |
| Predecessor / related | [INKROUND-07](2026-11-23-inkling-mini-decode-round.md#20261123-inkround-07--unquantized-nvfp4-expert-inputs) (the measurements); [INKROUND-10](#20261123-inkround-10--fold-the-expert-input-pass-into-the-product-kernels) (rejected fold) |

**Change.** `#define INK_EXPERT_INPUTS_QUANTIZED 0` in `experts.metal` (mirrored by `QUANTIZE_EXPERT_INPUTS` in `mod.rs`). Single-row decode, the verifier and default prefill pass each NVFP4 expert input as its binary16 value with a unit row scale (the row amax is set to 1,536 so the products' outer factor reduces to expert_max / denominator); the dot kernels are unchanged. The activation pass still runs as a copy, so the launch count is unchanged. MXFP8 experts (layers 20 to 23) keep their own activation quantization. The exact prefill reference compiles and uses the quantized activation (`ink_nvfp4_activation_quantized`, `ink_verify_half_activation_quantized`, selected by `nvfp4_activation(runtime)`), so `--exact-prefill` results are unchanged.

**Quality, natural text.** Ten inputs from repository prose and source at 300 to 8,000 tokens, 258 teacher-forced rows each, pooled over 2,580 rows; reference: the exact, quantized prefill path on the original artifact; candidate: the unquantized default on each artifact (`--quality-compare`, build `quant0-build`, `inkling_slice` SHA-256 `a5d192f012828a6eaa69f35ab0bad7399e1360e11751b63266568baf3a747d63`). Recomputed for this record from the per-input receipts `runtime/quant0/qc-orig-*.json` and `qc-a8-*.json` (summaries SHA-256 prefixes `9c99b618904701fc` and `01eb31b6010d8e98`):

| Candidate | Next-token accuracy, candidate (reference 73.64) | Top-1 agreement overall | At reference p ≥ 0.5 | At p ≥ 0.9 | Mean KL from the reference |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original artifact, unquantized | 73.76% | 95.70% | 99.38% | 99.93% | 0.0100 |
| 8-bit artifact, unquantized | 73.99% | 95.35% | 99.43% | 99.93% | 0.0116 |

Both pass. The 8-bit row compares across artifacts (8-bit candidate against the original's exact reference), so it bounds the combined effect of 8-bit dense weights and unquantized inputs; it is not a property of either change alone. For scale, the quantized default against the same reference had mean KL 0.0129 ([INKROUND-07](2026-11-23-inkling-mini-decode-round.md#20261123-inkround-07--unquantized-nvfp4-expert-inputs)).

**Speed.** Decode, `--paired-control expert_inputs`, 12 ABBA pairs on s73: single step 14.13 → 13.35 ms on the 8-bit artifact (11/12) and 17.44 → 16.65 ms on the original (12/12); verify rounds 0.67 to 0.81 ms faster at two to four rows (12/12 each) (receipts `ei-pair-a8-s73`, `ei-pair-orig-s73` from INKROUND-07). Prefill, engine-only ABBA, four pairs, quantized build (`quant1`, SHA-256 prefix `7b9b01409013f5a2`) against unquantized (`quant0`, `5ec075e053f2673a`), original artifact, 2,048-row commands (`runtime/quant0/paired-4150.jsonl`, `paired-8800.jsonl`; summary prefix `df0082927968e9dd`):

| Prompt | Quantized median | Unquantized median | Median of per-pair ratios | Range |
| --- | ---: | ---: | ---: | --- |
| 4,150 tokens | 1,992 ms | 1,977 ms | 1.009 (this record's recomputation; the commit body says 1.007) | 1.005 to 1.011, 4/4 |
| 8,800 tokens | 4,070 ms | 4,139 ms | 0.965 | 0.814 to 1.189, 2/4 |

The commit body calls 8,800 "level". The recomputed median is 0.965 with a range of 0.81 to 1.19, which is host noise rather than a measured slowdown or gain; read it as no measurable prefill change.

**Correctness of the default** (coordinator, before committing; receipts in `runtime/quant0/` and `runtime/commit-check-2/`, summary SHA-256 prefix `c4b2f06c4538c897`; `inkling_slice` `139e3ae8021d…`, `stark-generate` `46278ca04842…`): define read as 0; clean build with no warnings; 167 worker tests; the exact reference exact on `g4096-02` and `s4096-00`; drafting on equals off at 73, 4,150, 8,800 and 16,000 tokens on both artifacts; a repeated 8,800-token replay with identical hashes; the worker smoke on both artifacts. The same gate run recorded a failed format check (`fmt-check exit=1`); whether formatting was corrected before the commit is not recorded in the receipts.

**Not done.** No fresh prefix-reuse validator run on this default ([INKAFFINE8-04](2026-11-23-inkling-mini-affine8-default.md) records exactness on the earlier worker and the open rule-conformance gap). No server-level timing of this change alone; it is included in the four-arm scoreboard run on `4314ac51`, which was running when this record was written.

**Output hashes changed, as expected.** Worker builds before and after the flip give different output hashes on every input. The decode workstream first attributed this to the gathered prefill kernel and then corrected itself by reading the define embedded in each binary: `worker-planner-2` (built 09:52, runs `abba2` and `abba3`) embeds `INK_EXPERT_INPUTS_QUANTIZED 1`, and `worker-fixed-3` and `worker-planner-3` (built 10:26, runs `abbb1` to `abbb4`) embed `0`. The older planner comparison is therefore not confounded: both of its builds were quantized.

## 20261123-INKROUND-10 · Fold the expert-input pass into the product kernels

| Field | Value |
| --- | --- |
| Attempt ID / date | `20261123-INKROUND-10` / 2026-11-23 |
| Kind | Kernel (launch removal) |
| Result | Rejected: bit-exact but slower on both artifacts |
| Disposition | Kernels kept in the tree, off by default; reachable only as the diagnostic control `direct_expert_inputs`. Committed in that state in `4314ac51`. |

**Change tried.** With unquantized inputs the activation pass only widens BF16 to binary16, so the single-row and verifier NVFP4 products read the BF16 rows directly and widen each value themselves (`ink_dx_up_bf`, `ink_dx_down_bf` in `decode_experts.metal`; an `INK_VERIFY_BF16_INPUT` variant `ink_verify_nvfp4_bf_dot` in `verify_nvfp4.metal`; a direct branch in `Layer::nvfp4`; `GroupedExperts::encode_projected` in `grouped_pilot.rs`). This removes two activation launches per NVFP4 visit, 88 per single-row step, and two per visit per verifier round. Default prefill and the exact reference were not changed.

**Result.** Paired on s73, 12 pairs per artifact: logits bitwise identical on the step and all four verifier rows; the single step 0.15 ms slower (8-bit 0.163, original 0.143); the verifier 0.34 ms slower at two rows, 0.83 at three and 1.2 at four, with every pair slower. The kernel screen agrees: zero unequal values, and the direct path is 0.22 ms slower than the activation pass plus the incumbent products. Cause: widening BF16 to half inside the products repeats the conversion for every output channel, which costs more than the removed launches save. Receipts in `runtime/inkling-affine8/runs/`: `direct-screen-a8.json` (SHA-256 prefix `7935b61b7dd39fb2`), `direct-pair-a8-s73.json` (`af77d227a9bec1dd`), `direct-pair-orig-s73.json` (`e118e998937c8bb7`).

**Prefill side, not tried yet.** Removing the expert-input pass from default prefill (fold into ingress or the product kernel; expected hash-identical; about 136 ms of instrumented time at 8,800 tokens) is queued with the prefill workstream.
