# Inkling 2 Mini decode round, 2026-11-23: stage table, short-context attention, router, worker loop, depth policy and expert inputs

These attempts ran on 2026-11-22 evening to 2026-11-23 on the 128 GiB Apple M5 Max (Mac17,7), macOS 26.7.2 (25G332), AC power with the adapter negotiating 94 W, all GPU work behind the shared lock. They cover both the original checkpoint (`thinkingmachines/Inkling-2-Mini` at `6e7c8248f75265bdd4fbab5273304243b5d8c2cb`) and the Stark 8-bit dense artifact ([INKAFFINE8-03](2026-11-22-inkling-mini-affine8.md#20261122-inkaffine8-03--stark-8-bit-engine-artifact)); every result names its artifact. All code is uncommitted on top of `f1666090`; nothing is installed. Private receipts are under `runtime/inkling-affine8/runs/` (ignored); names are given, contents stay private. The [MLX-Serve differences page](inkling-mlx-serve-differences.md) summarizes where each result stands.

Status labels: **adopted** (default in the tree), **rejected**, **not final** (still being gated when last reported), **measured** (no code change).

## 20261123-INKROUND-01 · Decode stage table, measured ceilings and the worker-faithful round budget

Measured. Engine builds `stage-slice-2` and `-3` (SHA-256 prefixes `034a2ddbf7cf`, `d63e5aa70f0f`), s73, context 4,352. GPU ms from timestamped encoder splits, which add about 2 ms of gap and inflate small stages (see INKROUND-04); uninstrumented command totals are the reliable figures.

8-bit artifact, uninstrumented command GPU 14.07 / 16.51 / 18.45 / 21.67 ms at 1 / 2 / 3 / 4 rows; 991 / 1,751 / 2,043 / 2,383 launches.

| Stage, 8-bit artifact | 1 row | 4 rows |
| --- | --- | --- |
| Dense projections | 2.90 ms, 96 launches, 1,373 MB, 473 GB/s | 4.27 ms, 321 GB/s |
| Experts | 7.83 ms, 336 launches, 1,868 MB, 238 GB/s | 11.74 ms, 4,605 MB (20 distinct experts per visit), 392 GB/s |
| Attention | 0.84 ms, 192 launches | 2.92 ms, 1,152 launches |
| Attention inputs (controllers) | 0.88 ms, 150 launches, 96 GB/s | 0.90 + 0.92 ms |
| Expert inputs | 0.94 ms, 144 launches | — |
| Readout | 0.92 ms, 490 MB, 531 GB/s | readout and other 0.89 ms |

Original artifact: the same stages except dense 4.87 / 5.56 / 5.66 / 5.80 ms (2,584 MB) and readout 2.16 ms (922 MB); uninstrumented 17.38 / 19.83 / 21.74 / 24.68 ms.

Ceilings, read-only and read-plus-sum agreeing: streaming one buffer 571 to 586 GB/s; the exact selected expert slices gathered once per plane per visit: 1 row 1,839 MB in 3.59 ms (513 GB/s), 4 rows 4,577 MB in 8.29 ms (552 GB/s). Experts are far from the ceiling (7.83 against 3.59 ms at one row; 11.74 against 8.29 at four). 8-bit dense is near the ceiling at one row and compute- or latency-bound at four. Attention and controllers are latency-bound. Up plus down alone for the 44 NVFP4 visits is 4.80 ms in isolation; the finer one-row breakdown (`stage2-a8-s73`) gives router 1.11 ms, route select 0.30, activation quantization 1.05 (88 launches), NVFP4 dots 4.96 (up 3.1, down 1.85), MXFP8 0.64; at four rows dots 7.9, MXFP8 1.48, routing 1.65, quantization 1.01. The router figure was later shown to be overstated (INKROUND-04).

Round budget (engine replay of `collect_output`, 128 greedy tokens, s73, output equals serial in every arm). 8-bit, serial step 15.24 ms: depth 1 round 20.13 ms (draft GPU 3.00, verify GPU 16.27, commit 0.15, verify encode 0.52, readback and sampling 0.30, other idle 0.73), 79.4 tok/s; depth 2 22.87 ms, 85.0 tok/s; depth 3 25.75 ms, 83.4 tok/s. Original, serial 18.51 ms: depth 3 29.08 ms, 101.9 tok/s. Outside the verify command a round costs about 4.1 ms: the draft command (3.0 to 3.2 ms, almost all the fixed draft block forward: 3 BF16 draft layers of about 500 MB plus the INT8 head of about 460 MB; each extra proposal row about 0.1 ms) plus about 1 ms of host, idle and commit. The two artifacts generated different text on this prompt, so the acceptance figures here are not comparable. Receipts: `stage-a8-s73`, `stage-orig-s73`, `stage2-a8-s73`, `budget-a8-s73`, `budget-orig-s73`.

## 20261123-INKROUND-02 · Single-row NVFP4 expert kernels with several channels per SIMD group

Rejected. Six variants of the single-row up and down kernels with several output channels per SIMD group and register-resident activations, all bitwise identical to the incumbent (half activations included), measured 6.75 to 11.5 ms against the incumbent's 4.80 ms for the 44 NVFP4 visits. Register pressure and occupancy dominate; this agrees with MLX-Serve's one-row-per-SIMD-group choice. A half-buffer variant (`ink_dx_up_hb`) was written and never timed. The variants stay in `decode_experts.metal` for the screen only, but they are registered at every M5 session load, which costs one library compile. Receipt: `dx-variants-s73` (build `stage-slice-4`, prefix `dd6f368366b3`).

## 20261123-INKROUND-03 · Batched short-context verifier attention

Adopted. The shared local and global verifier attention kernels were admitted only at position ≥ 1,024, so every short-context round, the scoreboard case, fell back to per-row attention: a K/V pack, relative table, parts and gate per row per visit, 288 launches per row. New `verify_short.metal` (`ink_verify_short_parts`, `ink_verify_short_gate`) runs all rows in one launch each with the single-row 32-shard short-path arithmetic unchanged; dispatched from `encode_lanes` in `prefill.rs` through `VerifyWorkspace::encode_shared_short` whenever position + rows < 1,024. No selector; a diagnostic-only field turns it off for the paired screen.

Exactness: `--measure-verify-transaction-schedule` on the 8-bit artifact at s73 reproduced every hash of the previous build (teacher `d4db2d199ad1`, state `b5342e5977c2`, 9/9 controls, exact replays, depths 1 to 3 equal serial). The coordinator's gate runs also passed on the original at 73 and 4,150 and on the 8-bit artifact at 73 and 8,800.

Paired timing, `--paired-control … short_attention`, 12 ABBA pairs, s73, GPU ms off → on:

| Artifact | Verify 2 rows | 3 rows | 4 rows | Draft+verify wall at 4 rows |
| --- | --- | --- | --- | --- |
| 8-bit | 16.68 → 16.01 (−0.67) | 19.57 → 18.42 (−1.17) | 22.50 → 20.76 (−1.71) | 25.95 → 24.22 |
| Original | −0.62 | −1.17 | −1.60 | 29.58 → 27.88 |

12/12 pairs faster in every 8-bit cell, 11 to 12 of 12 on the original; the single step, which the change cannot affect, stayed at 14.16 against 14.17 ms. Rounds at position ≥ 1,024 are unaffected. Not final: the transaction schedule at 16,000 on both artifacts, replays, a determinism repeat and the worker smoke were still owed when last reported. Receipts: `pair-short-a8-s73`, `pair-short-orig-s73`, `short-txn-a8-s73`; build `short-slice-2` (prefix `f3afafdb0153`).

## 20261123-INKROUND-04 · Decode router rewrite

Rejected and reverted. The instrumented stage table put the single-row router projection at 1.11 ms per step (25 µs per visit on the native M16 × N8 tensor router). The candidate replaced it with a 4-experts-per-SIMD-group F32 matvec shared by the single row and the verifier. Paired ABBA, 12 pairs, s73, GPU ms old → new:

| Artifact | Single step | Verify 2 / 3 / 4 rows |
| --- | --- | --- |
| 8-bit | 14.13 → 14.07 (−0.06, 12/12) | +0.14 / +0.38 / +0.56 (0/12 wins) |
| Original | −0.05 | +0.14 / +0.36 / +0.56 |

Exactness of the transaction held on both artifacts. The new router changed routing arithmetic: on natural inputs, 258 rows each, KL 0.003 to 0.009 and top-1 agreement 0.957 to 0.996 (0.992 to 0.996 where p ≥ 0.5). Finding: the 1.11 ms figure was a timestamp-split artifact; splitting the encoder at stage boundaries inflates tiny kernels, so instrumented sub-stage numbers are upper bounds and only paired uninstrumented timing decides. The candidate is saved outside the tree. Receipts: `router-pair-*`, `router-txn-*`, `router-qc-*`.

## 20261123-INKROUND-05 · Worker-loop study

Measured. `collect_output` in `crates/stark-generate/src/inkling.rs` gained round-time statistics (round wall from start to commit end, split by accepted count; in-round host time; time between rounds); output unchanged, no gate run on that instrumentation. The real worker, streamed and greedy as the server runs it, on three template-rendered chat inputs (code, explain, prose; 768-token cap), two repeats, both artifacts, fixed depth 3 (build `worker-rounds-1`, prefix `5185d78a82db`):

| Artifact, input | Verify wall | Commit wall | In-round host | Between rounds | tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8-bit, code | 24.0 ms | 0.44 ms | 0.22 ms | 0.007 ms | 104.6 (second repeat 96.7) |
| 8-bit, explain | 27.6 | 0.46 | 0.19 | 0.007 | 75.6 |
| 8-bit, prose | 29.3 | 0.45 | 0.16 | 0.007 | 58.7 |
| Original, code | 27.3 | 0.45 | 0.22 | — | 91.8 |
| Original, explain | 31.1 | 0.48 | 0.19 | — | 68.8 |
| Original, prose | 29.5 | 0.41 | 0.16 | — | 58.0 |

Round time by accepted count (0 / 1 / 2 / 3) on the 8-bit artifact is flat within a prompt (code 24.3 / 24.5 / 24.7 / 25.1 ms; prose 29.8 / 30.0 / 30.1 / 31.0), so a rejection costs nothing extra. Between prompts it differs by up to 5 ms because the four verifier rows route to more distinct experts on varied text (8 at one row, 13.5 at two, 20 at four on s73). The coordinator's earlier hypothesis of a 2.5 to 3 ms worker-loop overhead per round was wrong. Remaining worker-loop saving: folding the commit into the next round's draft command, about 0.3 to 0.45 ms (queued). Receipts: `wloop-artifact.result.json`, `wloop-prepared-inkling.result.json`.

## 20261123-INKROUND-06 · Depth policy: history planner, confidence cut and backoff

Not final. The planner was being landed as the worker default when last reported.

**Planner.** `depth_planner.rs`: per-round depth 1 to 3 from running acceptance and cost estimates (settings EMA 0.03, cost EMA 0.1, hysteresis 1.03, probes one row deeper every 16 rounds, at least one draft, capped by the requested MTP count). Output is identical by construction; it changes speed only. Its unit tests had not been run when last reported.

**Invalid and superseded runs.** A first planner version beat fixed depth 3 on prose by 8 percent (8-bit) but lost to fixed 1 on prose and fixed 2 on explain; it was tuned. The `adapt*` and `chatdepth*` runs were not interleaved and are superseded. Interleaved real-arm runs (`policy-*`) were too noisy to decide: the same fixed depth moved by up to 25 percent between arms in one session (8-bit explain d1 71.5 against d2 54.5, the reverse of every other measurement). A harness bug committed tokens past end of sampling and invalidated the prose, 8-bit explain and s73 runs; it was fixed. Raw-text continuation depth data (`depth-*`) favour depth 3 on some inputs and 2 on others and are kept as reference only, because their distribution is unlike chat traffic. Template-rendered chat inputs show lower acceptance: on the 8-bit artifact chat-prose d1 75.3, d2 70.1, d3 62.4 tok/s; chat-code 78.1, 89.4, 91.1; chat-explain 81.9, 79.1, 76.0 (non-interleaved, superseded as a decision basis).

**Method used for the decision.** The draft head reports each proposal's confidence (the winner's softmax weight over the rescored candidates, from `ink_draft_winner`, read with `Session::proposal_confidences`; proposals unchanged). `--draft-trace` runs the target at every anchor of the greedy output (768 tokens on chat inputs, 256 at 4K and 8K), records accepted prefix lengths and confidences, and measures interleaved round costs at the same positions: plain step, overlapped draft plus verify at depths 1 to 3, and the proposal command alone followed by a non-overlapped verify (a confidence-gated round must read confidences before choosing rows). Each policy is simulated on the trace with those costs. Confidences stay in memory; only aggregate tok/s is written. Receipts `trace3-*` (chat code, prose, explain, s73, c4000, p4000, c8000; both artifacts).

Simulated tok/s, 8-bit artifact:

| Input | Best fixed depth | History planner | Best confidence cut (threshold) | Cut plus backoff | Fixed 3 plus backoff |
| --- | ---: | ---: | ---: | ---: | ---: |
| Code | 102.7 (d3) | 102.7 | 102.4 (0.45) | 95.3 | 93.5 |
| Prose | 71.0 (d1) | 71.0 | 69.7 (0.45) | 61.3 | 60.0 |
| Explain | 79.7 (d2) | 79.1 | 76.8 (0.45) | 73.8 | 72.1 |
| s73 | 91.9 (d3) | 92.1 | 90.5 (0.35) | 86.4 | 89.0 |
| c4000 | 114.5 (d3) | 112.8 | 112.6 | 110.2 | 111.4 |
| p4000 | 119.3 (d3) | 119.3 | 116.5 | 112.7 | 113.4 |
| c8000 | 123.6 (d3) | 121.2 | 120.6 | 118.0 | 121.6 |

Original artifact, planner against best fixed: code 88.4 / 87.9, prose 61.5 / 60.8, explain 72.1 / 71.6, s73 87.7 / 89.5, c4000 105.1 / 108.6, p4000 111.6 / 111.0, c8000 119.4 / 120.1. Against fixed depth 3, the current default, the planner gains about 8 percent on 8-bit prose (71.0 against 65.8) and 6 percent on original prose, costs 1.5 to 3 percent at 4K and 8K, and is level on code. Earlier simulation on 8 traces put it within 1.2 percent of the best fixed depth everywhere. Real interleaved runs on code: 8-bit d1 82.5 / d2 89.3 / d3 92.0 / planner 91.7; original 72.5 / 80.5 / 82.4 / 82.4; explain, original 67.7 / 71.3 / 70.1 / 70.4.

**Why MLX-Serve's rules do not transfer.** (a) On Stark a drafted round is cheap relative to a plain step (depth 1 20.5 ms against a 15.9 ms step, 1.29 times), so even prose at about 35 percent first-position acceptance gains from drafting (no drafting 56.5 against 71.0 tok/s at depth 1). The backoff's per-round test, (accepted + 1) × plain-step cost < round cost, fires on most rejected rounds and switches drafting off while it is paying, costing 10 to 15 percent. (b) The confidence cut needs the proposals on the host before choosing rows, which removes the draft and verify overlap and adds about 1 to 3.5 ms per round; the 3.6 to 4.1 ms draft command runs even when the cut keeps zero rows. Both conclusions depend on Stark's cost ratios; if the draft command becomes much cheaper, the cut deserves another look.

**Wider verification (estimate, not measured).** At c8000 on the 8-bit artifact 74 percent of anchors accept all three drafts and P(third accepted | second accepted) is 0.88. Assuming that rate continues and each extra row costs about 3 ms: four drafts about 135 tok/s (+8 percent), five about 140 (+12 percent); c4000 and p4000 are similar. `MAX_VERIFY_ROWS = 4` is wired into the dense matvec kernels, shared attention, row mix, norm and gate kernels, the save and rollback workspace, logits rows and the draft commit; several days of work, long-context traffic only. Not started.

**Landing.** The planner was switched on in `collect_output` (floor 1, capped by the requested MTP count) with these steps queued: worker build; worker smoke on both artifacts (MTP on equals off, prefix reuse equals cold, image then text); worker loop timing on the chat inputs; the owed short-attention gates. The decode workstream would revert the switch if any step failed. The outcome was not reported when this record was written.

## 20261123-INKROUND-07 · Unquantized NVFP4 expert inputs

Not final. MLX-Serve does not apply the publisher's NVFP4 activation quantization to the expert inputs on any path (source reading). Stark now has one switch: `#define INK_EXPERT_INPUTS_QUANTIZED 1` in `experts.metal`, mirrored at compile time by `QUANTIZE_EXPERT_INPUTS` in `mod.rs`. With 0, `ink_nvfp4_activation` and `ink_verify_half_activation` write each input as its binary16 value and set the row amax to 1,536, so the products' outer factor reduces to expert_max / denominator; dot kernels are unchanged, single-row decode and verifier rows stay identical to each other, and packed prefill ingress converts the values to BF16 exactly. MXFP8 experts keep their activation quantization under both settings. The exact reference is always quantized. The launch itself still runs as a cheap copy.

Decode-only paired timing, `--paired-control expert_inputs`, 12 ABBA pairs, s73, GPU ms quantized → unquantized: 8-bit single step 14.13 → 13.35 (−0.78, 11/12), verify 2 / 3 / 4 rows −0.77 / −0.81 / −0.80 (12/12 each); original single step 17.44 → 16.65 (−0.79, 12/12), verify −0.72 / −0.72 / −0.67 (12/12). Receipts `ei-pair-a8-s73`, `ei-pair-orig-s73`.

Joint quality (coordinator, build `quant0-build`, `inkling_slice` prefix `a5d192f01282`, original artifact): the exact reference stays exact on `g4096-02`; drafting on equals off at 73 and 4,150 (and 73 on the 8-bit artifact); pooled natural-text next-token accuracy 73.76 percent against the exact reference's 73.64; top-1 agreement 95.70 percent overall, 99.38 at reference confidence ≥ 0.5, 99.93 at ≥ 0.9; mean KL 0.0100, against 0.0129 for today's quantized default. The decode-only natural-text comparison failed its first run (the diagnostic control required a verifier activation kernel that sessions without a draft lack); fixed and queued. Still owed before the define flips: the 8-bit quality half and paired prefill timing.

## 20261123-INKROUND-08 · Natural-text quality probe and the synthetic-capture finding

Measured. `--quality-compare REF CAND DIR PROMPT CONTEXT WINDOW` (each side `default:PATH`, `exact:PATH` or `off-CONTROL:PATH`) teacher-forces 258 rows per input and reports next-token accuracy of each side, top-1 agreement overall and on rows where the reference's top-1 probability is at least 0.5 and 0.9, KL mean and maximum, and mean absolute change in log-probability of the actual next token. Inputs: ten natural texts built from repository prose (p) and source (c) at 300, 1,000, 2,000, 4,000 and 8,000 tokens (`runtime/inkling-natural-text-2026-11-22/`).

| Comparison | Per-input range | Pooled over 2,580 rows |
| --- | --- | --- |
| Original: exact prefill (ref) against default prefill | Agreement 0.922 to 0.988; 0.982 to 1.000 at p ≥ 0.5; 1.000 at p ≥ 0.9; KL mean 0.0066 to 0.0208 | Next-token accuracy: default 73.41, exact 73.64 percent; 99.28 percent agreement at p ≥ 0.5 |
| Original default (ref) against 8-bit default | Agreement 0.911 to 0.981; 0.979 to 1.000 at p ≥ 0.5; KL mean 0.0067 to 0.0225 | 8-bit 73.41, original 73.41 percent; 98.92 percent at p ≥ 0.5 |

Both pass the adopted bar of [`docs/BENCHMARK.md`](../BENCHMARK.md) (pooled accuracy not more than 2.0 points below, at least 98 percent confident agreement). Single 258-row inputs differed by up to 2.3 points, which is why the bar is pooled. The 8-bit artifact differs from the original by about as much as default prefill differs from exact prefill. Receipts `qc-a8-*`, `qc-prefill-*` (SHA-256 prefixes in the decode workstream's report); the coordinator's earlier 35-row natural-text check is in [INKAFFINE8-03](2026-11-22-inkling-mini-affine8.md#quality-against-the-original-artifact).

**Synthetic captures.** The oracle and capture prompts used for exactness gates are synthetic and nearly uniform: the model predicts their next token 0 to 2 times in 34 rows; only the 73-token prompt is natural (18 of 34). Agreement and flip rates on them, including the noise-floor control's, are not quality measures, and draft acceptance on them is not representative of chat traffic. They remain valid for exactness and self-consistency.

## Other items from this round

- New diagnostic subcommands, all off the inference path: `--decode-stages`, `--round-budget`, `--paired-control` (controls `short_attention`, `router`, `expert_inputs`), `--quality-compare`, `--decode-experts`, `--draft-trace`, `--depth-policies`; files `decode_probe.metal`, `router_mpp_reference.metal`; `span_dispatches` in `PrefillTiming`.
- A compile error in `decode_stages.rs` (ambiguous integer type) briefly broke shared builds and was fixed.

## Follow-up history

2026-11-23, after the commit of the branch (sources: the coordinator's report and its verification receipts in `runtime/commit-check/`, summary SHA-256 prefix `8cb9cd12821508c4`):

- **INKROUND-03 gates.** The committed tree (`2a1d72a2`) passed drafting on equals off at 73, 4,150, 8,800 and 16,000 tokens on both artifacts, a repeated 8,800-token replay with identical hashes and the worker smoke on both artifacts. The batched short-context verifier attention is adopted with its owed gates closed.
- **INKROUND-06 landed.** The history depth planner is the committed default in `2a1d72a2` (floor 1, capped by the requested count) and passed those gates. Its tuning for 4K and 8K, where the simulation put it 1.5 to 3 percent below fixed depth 3, is in progress and not final. The confidence cut and the backoff stay rejected on Stark's cost ratios.
- **INKROUND-07 unchanged.** Unquantized expert inputs: decode about 0.8 ms faster per step and per round on both artifacts (12 ABBA pairs), joint quality passed on the original artifact. The 8-bit quality half, the paired prefill timing and the default flip are still pending; the committed default keeps `INK_EXPERT_INPUTS_QUANTIZED 1`.
- Owner decisions the same day: the 16-user burst is out of scope; the Stark 8-bit artifact becomes the default (integration starting, not done).
