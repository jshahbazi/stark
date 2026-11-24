# Inkling 2 Mini DFlash 2 and DSpark on the same greedy traces

Attempt `20261124-INKDRAFT-01`. Apple M5 Max, 40 GPU cores, 128 GiB unified memory, macOS 26.7.2. This engine diagnostic tests how much of a known target continuation each drafter can propose and how much target verification costs. The reference worker is the original-checkpoint arm of the [November 23 standard benchmark](../benchmarks/2026-11-23-inkling-mini-standard-benchmark.md).

## Trace and artifact identity

The four requests are block zero, first measured initial coding, first measured sustained coding, first measured novel prose and the 4K context coding rung. The recovery tool rebuilt the tagged prompts using the original llmprobe corpus builder, rendered them with Stark's template, and matched the native input hashes. It then ran the archived worker with its default drafting and compared each 192-token greedy output to the corresponding output hash. The native capture used packed source prefill through prompt length minus four, split at 2,048 rows, then three singleton steps and the final prompt token. The sustained label identifies that benchmark's prompt. This replay does not reproduce its thermal sequence.

| Component | Identity |
| --- | --- |
| Target | `thinkingmachines/Inkling-2-Mini` at `6e7c8248f75265bdd4fbab5273304243b5d8c2cb`, original NVFP4/MXFP8 expert banks and BF16 shared input/output tables. Target SHA-256 `65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1` |
| Archived Stark worker | `f86d94b877b3034db82843480669112ecd97306ad09ffbdf7ff8c5addabcc2e3`, image-loaded worker used for historical output recovery |
| Native DSpark | Publisher `mtp.safetensors`, SHA-256 `7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2` |
| DFlash 2 | [IncoAI checkpoint](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2/tree/4f9088ea66c21f2b99ed657d1941c306b03e84d2), BF16, SHA-256 `217980ab00d116693fa5946df7bbfd967caf5fac1f529afd541fbd470ddb54ee` |
| Reference implementations | [z-lab/dflash MLX and PyTorch](https://github.com/z-lab/dflash/tree/07ebd93db9f472af339b644bb70221ad8428328a/dflash), [Inkling SGLang adapter](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2/blob/4f9088ea66c21f2b99ed657d1941c306b03e84d2/sglang_inkling_dflash2.py), [Splash 1.7.2 family embedding rule](https://github.com/incoai/splash/blob/2afc7337aacc044722fa95e4cb3da75c281a19a8/runtime/model/Inkling2LoopTarget.cpp#L453) |

Each candidate block receives the known anchor. Its context ends immediately before that anchor. DFlash 2 reads the target partial residual entering visits 5, 23 and 41 and the final normalized output at 48. It RMS-normalizes the target token embeddings for the anchor and seven masks, scores the unpadded vocabulary of 200,058 tokens and chains the top-16 candidates with the checkpoint's post-correction softcap. Each tested width uses the first one through seven predictions from the eight-row draft block. Each width walks its own rounds on the common continuation. A round emits the anchor plus the consecutive correct proposals, clipped to the remaining output budget. The aggregate uses total emitted tokens divided by total rounds.

## Tokens per round

| Trace | Input IDs | DSpark, 3 | DFlash 2, 3 | DSpark, 7 | DFlash 2, 7 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial coding | 71 | 2.40 | 2.87 | 2.74 | 3.43 |
| Sustained coding | 72 | 2.34 | 2.70 | 2.59 | 3.25 |
| Novel prose | 40 | 2.00 | 2.63 | 2.11 | 3.10 |
| 4K context coding | 4,019 | 3.10 | 3.20 | 3.62 | 4.57 |
| All 768 output tokens | | 2.40 | 2.83 | 2.67 | 3.51 |

| Proposals | DSpark rounds | DFlash 2 rounds | DSpark tokens/round | DFlash 2 tokens/round |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 449 | 424 | 1.71 | 1.81 |
| 2 | 361 | 323 | 2.13 | 2.38 |
| 3 | 320 | 271 | 2.40 | 2.83 |
| 4 | 305 | 245 | 2.52 | 3.13 |
| 5 | 296 | 233 | 2.59 | 3.30 |
| 6 | 290 | 225 | 2.65 | 3.41 |
| 7 | 288 | 219 | 2.67 | 3.51 |

DSpark used Stark's resident INT8 proposal head. Both drafters verified against the same original target. DFlash 2 used MLX 0.31.2 with BF16 weights and activations, a BF16 shared output projection and F32 selector corrections following [Splash's Metal arithmetic](https://github.com/incoai/splash/blob/2afc7337aacc044722fa95e4cb3da75c281a19a8/runtime/metal/kernels/decode/sampling.metal#L1328). The earlier BF16-selector screen required 225 rounds at seven proposals. The F32-selector run required 219.

## Stark verifier cost by width

Median milliseconds of host wall time for target verification. Each cell has 24 timed observations at six positions in its trace, with four rotated repeats per position after warmup. A row is an input to the target. Four rows verify an anchor and three proposals. Stark permits up to four rows in one transaction. Widths five through eight use two successive calls on a fully accepted teacher block, with a singleton step where one row remains. The clock excludes draft generation, diagnostic checks and transaction commit. The singleton step includes its native readout and state update.

| Trace | 1 row | 2 rows | 3 rows | 4 rows | 5 rows | 6 rows | 7 rows | 8 rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Initial coding | 17.79 | 19.24 | 21.72 | 23.78 | 41.39 | 43.43 | 45.57 | 47.44 |
| Sustained coding | 17.67 | 19.31 | 21.56 | 23.84 | 41.39 | 43.42 | 45.72 | 47.44 |
| Novel prose | 17.89 | 19.41 | 21.50 | 23.73 | 41.52 | 43.04 | 45.21 | 47.10 |
| 4K context coding | 18.80 | 20.26 | 22.71 | 25.41 | 44.47 | 45.91 | 48.47 | 51.07 |

The native target matched the historical greedy token at all 768 prefixes. Two repeat captures reproduced all four private fixtures byte for byte. At 16 sampled anchors, native DSpark depth two with shared KV and depth three each agreed with the corresponding prefix of the seven-proposal call in all 16 cases. All 192 width-by-position warmup controls matched serial logits and committed target and draft state exactly. The 768 timed samples ran on AC power with nominal thermal state. Width-sweep executable SHA-256 `4f4791e33e7aa76534c7ff9d143f4280ea871f11fbb129c07b1c4660a71c2972`.

The independent PyTorch 2.11 CPU control produced the same seven-token block as MLX at 15 of 16 sampled anchors. Only prose at output offset 95 differed. A full prose check found agreement on 1,287 of 1,316 comparable proposed positions and 181 of 191 full blocks. PyTorch required 61 rounds at seven proposals on prose and MLX required 62. This difference sets the numerical scope of the replay. The four-prompt sample and greedy policy also limit transfer to other workloads or samplers.

An initial singleton-prefill capture diverged from the historical worker and was excluded. The scored capture used the recovered historical teacher and the source prefill schedule above. The initial width pilot stopped at verifier initialization and produced no timing. The depth replay counts a final singleton round when the output budget leaves one token. An independent cursor walk on the private candidate rows reproduced the reported round counts.

## Verifying actual proposals

Six timing blocks replayed each candidate stream through the native verifier. Drafter order alternated by block and trace. The first three blocks measured depth three first. The last three measured depth seven first. Each depth began from the same prompt state after a serial reference pass. For each arm, the table sums times across four traces within each block, takes the median across blocks and then divides by logical rounds or emitted tokens. Verify covers the target call. Commit covers accepting the prefix and discarding speculative state. The clocks exclude drafter execution, the host's sampling and acceptance loop, and diagnostic checks. A terminal singleton includes its native state update in verify.

| Drafter | Proposals | Rounds | Target calls | Verify ms/round | Verify ms/output token | Verify + commit ms/output token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DSpark | 3 | 320 | 320 | 24.73 | 10.30 | 10.57 |
| DFlash 2 | 3 | 271 | 271 | 24.38 | 8.60 | 8.85 |
| DSpark | 7 | 288 | 330 | 28.65 | 10.74 | 11.00 |
| DFlash 2 | 7 | 219 | 279 | 31.09 | 8.87 | 9.12 |

DFlash 2 at three proposals used 16.5% less verifier time per token than DSpark at three. At seven proposals it used 17.5% less than DSpark at seven. DFlash 2 needed a second target call in 60 of its 219 seven-proposal rounds. DSpark needed one in 42 of 288. Early rejection avoids that second call when the first four rows settle the round.

| Trace | DSpark (3), ms/token | DFlash 2 (3), ms/token | DSpark (7), ms/token | DFlash 2 (7), ms/token |
| --- | ---: | ---: | ---: | ---: |
| Initial coding | 9.88 | 8.32 | 9.96 | 8.60 |
| Sustained coding | 10.17 | 8.83 | 10.39 | 8.85 |
| Novel prose | 12.85 | 9.13 | 13.15 | 9.04 |
| 4K context coding | 8.21 | 7.99 | 9.18 | 8.52 |

Seven DFlash 2 proposals saved 52 draft rounds and required eight extra target calls. If generating a draft block costs the same at either width and stages run sequentially, seven breaks even when the work saved per skipped draft round exceeds **3.77 ms**. This estimate is the median of the six paired differences in total verifier-plus-commit time divided by 52. The individual thresholds ranged from 1.65 to 13.77 ms. A live implementation can have extra costs for capturing new features, updating draft KV and rolling back speculative state.

All 48 native verifier replay processes ran on AC. Peak thermal state was nominal for 19 processes and fair for 29. Across the six blocks, DFlash 2 took 8.43–9.05 ms/output token at three proposals and 8.67–9.98 at seven. The [aggregate receipt](../benchmarks/data/inkling-mini-dflash2-traces-2026-11-24.json) includes each block's times, thermal state, hashes and both native binary identities. Every replay reproduced the independent round count and all 192 committed-row logits per trace. Target and DSpark state matched at sampled boundaries and at the end.

## Scope and disposition

This is a completed component benchmark against the original artifact. The diagnostic loaded the text target, maintained DSpark target taps in both replay arms, and accessed private offline DFlash 2 candidates. Production text serving with DFlash 2, its per-round draft time, image-loaded MTP and cross-request prefix restoration remain unqualified. Testing prefix restoration for DFlash 2 requires an integrated owner for its context features and draft cache. Each measured walk started from a fresh prefill.

## Reproduction

Run the [fixture recovery tool](../archive/experiments/2026-11-24-inkling-dflash2/recover_inputs.py) with the pinned llmprobe checkout, the original arm's block-zero `native.jsonl`, the prepared original target, a private output directory and the archived worker. Input recovery stops when a token hash differs. Use Rust 1.98.1 to build the `inkling_slice` release example with `--locked --no-default-features --features inkling-slice`.

For each recovered input, run `--capture-dspark-trace TARGET DSpark INPUT TRACE`. Run the [MLX evaluator](../archive/experiments/2026-11-24-inkling-dflash2/dflash2.py) on that trace with the target weights, the pinned draft and z-lab checkout, and a private candidate fixture. The [PyTorch control](../archive/experiments/2026-11-24-inkling-dflash2/torch_control.py) takes the pinned upstream `dflash/model.py`, draft config, draft weights, target weights and the same trace. Keep input, feature and candidate tensors in an ignored private directory. Both drafters are evaluated as a single text lane.

For the full-block width sweep, run `--measure-trace-verify-widths TARGET DSpark TRACE`. For actual candidates, run `--measure-trace-candidate-rounds TARGET DSpark TRACE CANDIDATES` once per arm and block. The default depth order is three then seven. Append `--seven-first` for the reverse order. Supply the native trace as CANDIDATES for DSpark, or the private MLX candidate file for DFlash 2. Save the aggregate JSON in private trace subdirectories named `coding`, `sustained`, `prose` and `context4k`, using `actual{block}-{dspark|dflash}.json` for block numbers 0 through 5. Run the [receipt summarizer](../archive/experiments/2026-11-24-inkling-dflash2/summarize_walks.py) with the parent directory to reproduce the pooled table.
