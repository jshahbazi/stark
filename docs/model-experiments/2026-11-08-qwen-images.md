# Qwen image experiments: retained attempt ledger

Recorded 2026-11-08. This retrospective preserves distinct outcomes from the
October BF16 baseline and November image-family integration. It describes
source-bound local evidence. It does not establish universal Qwen support,
installed-app capability, full larger-decoder parity, or image speedups.

IDs use the retrospective registration date. Actual October 26 baseline and
November 7–8 integration run dates remain in their original evidence records.

An ID identifies an attempt or independently useful control. **Pass** means its
stated gate passed; **component** means a bounded subgraph/control passed;
**fail** preserves an evaluated rejection; **invalid** means the setup cannot
qualify the intended result; **deferred** means resources prevented evaluation;
**unverified** means implementation/preparation exists without that execution
evidence. Subsequent success does not change an earlier outcome.

## Model and artifact coverage

| Exact source / artifact | Coverage retained | Remaining qualification |
| --- | --- | --- |
| Official `Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17`; schema-2 BF16 weights `7c5960e7…` | CPU/Metal tower exact; three full-model logit cases exact; developer worker/API functional checks | Results belong to recorded build/device/OS; installed packages need their own check |
| `mlx-community/Qwen3.5-0.8B-MLX-4bit@5d894f8cc4ef3e6c88537bf3746ed262f549da6a`; dense affine-v3 artifact `7e127dfb…` | Two complete 248,320-logit cases exact on production packed-reference encoder | Broader shapes/checkpoints and throughput |
| `Jundot/Qwen3.8-27B-oQ4e-mtp@7dd276b77fcb27c982ebd8d8a434539345713fea`; dense affine-v3 weights `dff4809e…` | All 333 retained BF16 visual tensors exact; 60 tower stages exact; actual worker lifecycle passes | Full 27B decoder numerical parity, quality and performance |
| `mlx-community/Qwen3.5-35B-A3B-4bit@1e20fd8d42056f870933bf98ca6211024744f7ec`; affine MoE-v4 weights `2f762b26…` | 333 visual tensors and 60 tower stages exact; actual worker lifecycle passes | Full quantized-MoE decoder numerical parity, quality and performance |
| Official `Qwen/Qwen3.5-35B-A3B@59d61f3ce65a6d9863b86d2e96597125219dc754`; BF16-source MoE-v5 | Metadata/header inspection, converter and native synthetic components | Official payload not downloaded/converted/loaded; full-model execution/resource qualification absent |
| `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl@0f94842e3682b49acace4daeb4cf2e65a6f94f83`; original format, schema-6 artifact `18c7bcc1…` | Two source-bound tower fixtures exact through 27 blocks; guarded worker lifecycle passes | Retained-decoder numerical parity, larger grids, quality and throughput |
| Same Lily source through typed Uzu **text** conversion | Separate text route; vision and MTP omitted | Cannot inherit schema-6 image capability |
| `ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit@788c07a8725642bccf13a0a3b8a21954ab84b4f2` | Retained image preparation supported | Image execution unverified; PipeNetwork `qwen38-mixed` remains text-only |

Full hashes and source details are in the linked records; prefixes above are
labels, not admission identifiers. See [Qwen families][families],
[BF16 MoE][bfmoe], [original BF16 execution][bf16], and
[native format distinctions](../NATIVE_HYBRID_MODELS.md#retained-lily-image-artifacts).

## BF16 baseline and affine-family attempts

RMS below is relative RMS. Paired metrics refer to the two frozen cases unless
specified. The unchanged full-model gate requires RMS ≤0.02, cosine ≥0.999 and
matching top-1. The original tower patch gate is ≤0.005, later stages ≤0.02.

| ID | Change / identity and comparison | Outcome and finding | Disposition / next test |
| --- | --- | --- | --- |
| IMG-20261108-QI-001 | Initial BF16 reference construction blanket-cast model to BF16 | **Invalid:** truncated 15/16 explicitly F32 rotary frequencies. Earlier traces/SDPA comparisons retained separately. [Preprocessing][preprocess] | Correct reference construction; assert F32 buffers |
| IMG-20261108-QI-039 | Original JPEG decoder versus the retained Pillow comparison budget | **Fail / repair:** subsampled inputs missed the retained budget. Production JPEG decoding was replaced with exact-locked pure-Rust `jpeg-decoder 0.3.2`, default features off; production `image` enables PNG only. The source paragraph does not give a numeric miss. [Preprocessing][preprocess] | Retain the codec/dependency boundary; fixture checks do not qualify every model's end-to-end JPEG requests |
| IMG-20261108-QI-002 | Native BF16 tower before source-compatible arithmetic; baseline `a02ed037` | **Fail:** normal square merger CPU/Metal RMS 0.04872362/0.05060871; rectangle 0.04055180/0.04028932. Small synthetic towers passed despite official-weight failure. [Preprocessing][preprocess] | Preserve failure; localize same-input operators |
| IMG-20261108-QI-003 | Corrected BF16 tower: Welford/affine, source contraction, rotary/softmax and GELU schedules | **Pass:** CPU and Metal each 60/60 checkpoints exact across four fixtures. Independent actual HF load matches extracted normal-case projector. [Tower parity][tower] | Retain as exact-artifact baseline; qualify decoder separately |
| IMG-20261108-QI-004 | BF16 full decoder and worker; worker SHA256 `5fc92516…`; source manifest `a96c7fa7…` | **Pass:** 744,960 logits across three fixtures exact; boundary case crosses 1,024-token prefill. Eight normalized QA answers match reference; both score 6/8, with counting/OCR misses retained. [Execution][bf16] | Preserve quality limitations and build/device/OS tuple |
| IMG-20261108-QI-005 | BF16 source class/interpreter controls | **Invalid:** installed child-model substitution and Python 3.13 fixture AST mismatch rejected. Corrected pinned child classes and Python 3.12 reference replay qualify IMG-20261108-QI-004. [Execution][bf16] | Never compare against substituted reference identities |
| IMG-20261108-QI-006 | Dense-v3/MoE-v4 conversion and geometry | **Pass:** 27B weights 17,676,008,168 B; MoE weights 20,394,001,296 B. MoE streaming conversion 36.33 s, 8 MiB workspace, 585,678,848 B peak RSS; conversion observation only. [Families][families] | Execute admitted artifacts; do not infer decoder parity |
| IMG-20261108-QI-007 | Initial 27B automatic text check, before `224a112b` correction | **Fail:** BF16 reference MLP gate accepted advisory Int8 activation preference; automatic check failed before images. [Families][families] | Reject result; correct declared output precision |
| IMG-20261108-QI-037 | 27B BF16 MLP gate correction at `224a112b`, engine SHA256 `4ee7629d…` | **Pass:** forced declared BF16 output; multirow MXU regression and eight-step text continuation pass. [Families][families] | Retain precision correction; image qualification follows |
| IMG-20261108-QI-008 | Actual 27B, quantized MoE35 and retained Lily worker at `7f8b8379`, SHA256 `41baf746…` | **Pass, functional:** distinct color cases, cancellation after streamed delta, cleanup, text recovery and fresh image. [Families][families] | Numerical and performance qualification remain separate |
| IMG-20261108-QI-009 | Original-checkpoint CPU tower references versus converted 27B/MoE35; native `224a112b` | **Component:** all 333 visual tensors per artifact match bitwise, including independent Conv3D transpose; all 60 stages per artifact exact. [Families][families] | Qualifies retained towers only |
| IMG-20261108-QI-010 | Packed 0.8B full decoder at `7f8b8379` versus F32 affine-weight computation then single BF16 cast | **Fail:** RMS 0.054535/0.030942; same top prediction insufficient. First-layer RMS ≈0.0018 accumulates. [Families][families] | Freeze reference; isolate GDN/MLP projections |
| IMG-20261108-QI-011 | Independent stock-MLX policy control | **Component:** 131,072 dequantized weights exact under single-cast BF16 policy; regular MLX matmul agrees with Torch; quantized matmul differs. [Families][families] | No alternate full-model reference promoted |
| IMG-20261108-QI-012 | Single-partition F32 GEMM plus explicit affine GEMV boundary; `2eb2ad96` | **Fail:** projection controls reduced to three BF16 mismatches; full RMS 0.054045/0.022619 still exceeds gate. [Families][families] | Remove remaining contraction discrepancy |
| IMG-20261108-QI-013 | Independent scalar packed-dot witness; exact-artifact test override at `202269cd`, SHA256 `70ad27c3…` | **Component:** all frozen projections and both complete logit readouts exact. [Families][families] | Establish contraction contract; production encoder still required |
| IMG-20261108-QI-014 | Production SIMD source-order affine encoder; `2529098a`, test SHA256 `7c8e5ede…` | **Pass:** both 248,320-logit cases exact; 32 production/scalar projection comparisons exact; seven focused encoder tests pass. Receipt `7fd2dc2d…`. [Production qualification][families] | Keep source-arithmetic markers authoritative; qualify larger decoders independently |
| IMG-20261108-QI-015 | Fresh production release worker `5b8d4025`, SHA256 `5985412d…`; actual 27B/MoE35 | **Pass, functional:** both lifecycle replays pass; receipts `3893a10d…`/`927fdb52…`. Sampled peak worker RSS 18,115,510,272/21,685,420,032 B. [Release replay][families] | No quiet-host speed comparison or large-decoder parity claim |
| IMG-20261108-QI-016 | BF16-source MoE-v5 full-bank indexed dots, router boundaries, selected experts/shared branch | **Component:** 512 focused Python checks; integration native indexed/full-bank, whole-MoE and cumulative scratch assertions pass. Official source inspected using <3 MiB metadata/ranges. [BF16 MoE][bfmoe], [integration][integration] | Full official 35B artifact remains **unverified**; requested scratch excludes allocator overhead |

The production affine encoder retains effective BF16 coefficients and the
source's F32 reduction order without a dequantized bank. It bounds work/output
before allocation, supports gathered expert rows and preserves explicit source
arithmetic markers. Small exact dense controls do not numerically qualify the
27B or either 35B profile. BF16-source MoE-v5 remains a distinct artifact contract
from quantized MoE-v4.

## Retained Lily numerical attempts

All rows use the unchanged original BF16 visual payloads. The independent CPU
producer binds Transformers `2e12b325…`, Torch 2.14.1 and upstream Lily
`e22d58fa…`, with F32 rotary buffers. Two frozen inputs are checked at 0, 1 and
27 blocks. Tower RMS ≤0.02/cosine ≥0.999 and the tighter attention gate
RMS ≤0.001/cosine ≥0.999999 were never widened. [Detailed numerical record][lily]
contains full seals and manifests for every source below.

| ID | Attempt / integrated source | Outcome and decisive aggregate | Disposition / next control |
| --- | --- | --- | --- |
| IMG-20261108-QI-017 | Baseline tower `c2d7d6cd`; reproduced with operators at `7cb13f39` | **Fail:** full-depth RMS 0.0252454230/0.0188444247; attention same-input RMS 0.00423718/0.00408923 dominates other operators | Compare materialization/normalization policies [Numerical evidence][lily] |
| IMG-20261108-QI-018 | CPU attention policy oracle | **Component:** normalized BF16 probabilities yield RMS 0.0001280236/0 versus retained online 0.0042373480/0.0040892320 | Native compiler/reduction still needs replay [Numerical evidence][lily] |
| IMG-20261108-QI-019 | Two-pass tiled normalized attention, `621326da` | **Fail:** attention RMS 0.00334685/0.00330653; tower 0.02805142/0.02207323 | CPU control reproduces missing scaled-score BF16 boundary [Numerical evidence][lily] |
| IMG-20261108-QI-020 | Float-cast stage tracing, `43fc445c` | **Invalid for production qualification:** observation changes compiled arithmetic; 2,461 ordinary/observed BF16 mismatches, RMS 0.01284323. Wrapper also retained only last trace | Require hard ordinary/observed equality [Numerical evidence][lily] |
| IMG-20261108-QI-021 | Integer BF16 rounding barriers, `202269cd` | **Component; tower fail:** attention gate and observer equality pass; full-depth RMS 0.02625438/0.01731256 | CPU 326,400-pattern rounding control exact; localize remaining sequence [Numerical evidence][lily] |
| IMG-20261108-QI-022 | Seventeen-stage sequence, `ec8db81c`; full-depth norm captures | **Component diagnostic:** observers exact; case-0 RMS grows through norm, attention and GELU to depth-one merger 0.002451595 | Isolate linears/nonlinearities on identical source inputs [Numerical evidence][lily] |
| IMG-20261108-QI-023 | Independent four-/32-stream contraction witness, `36414711` | **Component:** all 14 linears exact; production NAX has source-bit mismatches. Same-input norms retain small errors | Promote only measured dimension pairs with allocation bounds [Numerical evidence][lily] |
| IMG-20261108-QI-024 | Production contraction-only `42efacbf` | **Fail:** all 14 production linears exact; tower RMS 0.02534985/0.01829176. Ten independent canonical norm/GELU controls exact | Qualify Welford/activation selection [Numerical evidence][lily] |
| IMG-20261108-QI-025 | Production Welford + GELU `42f328a2` | **Fail:** all linears/nonlinearities exact; each GELU matches 65,280 finite BF16 inputs. Case 1 exact at every depth; case 0 full-depth RMS 0.024297598 | First discrepancy is position addition [Numerical evidence][lily] |
| IMG-20261108-QI-026 | Position controls; witness `5067716c` | **Component:** sequential uncontracted sum with both BF16 boundaries exact. Removing blend boundary causes 25,196 mismatches; pairwise summation causes two | Select measured safe body; witness alone does not fix production [Numerical evidence][lily] |
| IMG-20261108-QI-027 | Production position `2529098a` | **Fail:** position/first four sequence stages exact; full-depth case-0 RMS 0.02349485; rotary now first discrepancy | Capture rotary F32 stages [Numerical evidence][lily] |
| IMG-20261108-QI-028 | Safe uncontracted rotary witness `a1649a6a` | **Fail to close gate:** case-0 mismatches 38 →17; RMS 0.00001410644 | Not promoted; inspect frequencies and trig separately [Numerical evidence][lily] |
| IMG-20261108-QI-029 | Frequency/trig diagnostics `1d0fe455`, `c31fc114` | **Component diagnostic:** 7/18 inverse frequencies differ ≤2 ULP; native trig at exact angles still leaves eight BF16 mismatches | Correct both construction and trig [Numerical evidence][lily] |
| IMG-20261108-QI-038 | Source-compatible host rotary witness `96b6da7d` | **Component:** Rust F32 powf, block-major angles, pinned SLEEF schedule and BF16 application exact for both cases | Promote measured rotary schedule with bounded live coefficient buffers [Numerical evidence][lily] |
| IMG-20261108-QI-030 | Production source-compatible rotary `e7717364` | **Fail at tower gate:** production rotary exact; tower case-0 RMS 0.02741351; attention now first discrepancy | Retain component repair; preserve buffers through completion and inspect attention order [Numerical evidence][lily] |
| IMG-20261108-QI-031 | Independent seven-stage attention-order witness `7e537d9d` | **Component:** all source-order controls exact; ordinary path still differs at 131 BF16 outputs | Map proven reduction tree to production SIMD [Numerical evidence][lily] |
| IMG-20261108-QI-032 | Three-pass streaming attention candidate `94ce9e1a`, sealed `9b1c1e44` | **Pass, tower component:** six depth comparisons exact across 499,200 values; contractions, nonlinearities, position/rotary, attention and observers exact. Test SHA256 `4d6631c7…` | Larger grids/full retained decoder remain unverified; benchmark separately [Numerical evidence][lily] |

IMG-20261108-QI-032 uses fixed register state and no N×N production tensor. Geometry checks
bound the H1152/head72/16-head path to admitted grids; other geometries retain
their existing route. The source-compatible schedules include extra attention
passes and arithmetic work. Exactness supplies no evidence of a speed gain.
Its 4.740-second test wrapper includes reference work and is not throughput.
The 54 `full_norm` diagnostics intentionally measure the legacy baseline;
production Welford evidence comes from its separate selected-path controls.

## Resource attempts, integration and retained gaps

| ID | Attempt / identity | Outcome and consequence | Next test |
| --- | --- | --- | --- |
| IMG-20261108-QI-033 | Fresh Lily worker `3898d16d`, SHA256 `ad208012…`; admission 102,611,992,576 B available | **Deferred:** unrelated worker reduced availability below static 86,215,737,056-B bound before large allocation. Receipt `2e4a1a89…`; only owned processes reaped. [Resource evidence][families] | Recheck resources before separately authorized retry; no correctness result |
| IMG-20261108-QI-034 | Same sealed worker, explicit Lily retry; artifact `18c7bcc1…` | **Pass, functional:** full lifecycle, cancellation, cleanup and recoveries; receipt `17d457e5…`. Peak sampled RSS 61,299,671,040 B, minimum available 30,880,858,112 B; no guard fired. [Retry evidence][families] | Retained decoder parity/quiet-host throughput still open |
| IMG-20261108-QI-035 | Converter helper missing/stale in integrated suite | **Invalid:** absent helper or helper predating merged template code cannot establish regression success. Separate receipts retained. [Review][review] | Verify live module path and sealed helper before rerun |
| IMG-20261108-QI-036 | Final integrated native `f8293f87`, converter `a612db68` | **Pass, regression:** 931 native tests, 130 ignored, six filtered; 6,604 converter tests, 96 skipped. Final worker SHA256 `a49cb062…`. [Review][review] | Ignored model tests require their own receipts; no evidence transfer from regression counts |

The final numerical implementation is `e5dade4392b3fe01d076dd90a76a3f309fad5df9`;
later admission/main synchronization and formatting are recorded separately.
PR #97 merged as `b9a91a94`; this retrospective does not relabel earlier
model-bound receipts as runs on the final `a49cb062…` worker. The pre-PR record
binds that worker's final seal; Qwen lifecycle receipts retain their actual
earlier worker identities. [Integration][integration], [review][review]

All image paths clear/bypass prefix reuse and disable image MTP. BF16 MoE-v5
does not have a complete image-state snapshot/restore contract. Preparation,
structural admission, source-bit components, lifecycle, task quality and
performance remain separate evidence categories. Sampled RSS and wrapper
elapsed times cannot establish allocator ceilings or comparative throughput.

The original Lily schema-6 manifest binds source/index/shards, external n-gram
tables when present, config, tokenizer, processor and template. It cannot be
replaced by a text receipt. No manifest grants image capability implicitly.
The separate MLX-Serve retained format has preparation support without recorded
image execution; the original PipeNetwork text loader has no image bridge.

Next meaningful work is artifact-specific: larger-decoder numerical references,
broader retained-tower grids, full official BF16 MoE resource admission, then
paired quiet-host image performance measurements. This ledger records no new
model runs. Raw prompts, responses, vectors, images and private logs remain
outside tracked documentation. New evidence should append a new attempt ID and
immutable source/artifact/binary/reference identity without replacing failures.

[families]: ../archive/runs/2026-11-07-qwen-image-families.md
[bfmoe]: ../archive/runs/2026-11-07-qwen35-bf16-moe-images.md
[lily]: ../archive/runs/2026-11-07-retained-lily-vision-numerics.md
[integration]: ../archive/runs/2026-11-07-image-family-integration.md
[review]: ../archive/runs/2026-11-07-image-family-pr-review.md
[bf16]: ../archive/runs/2026-10-26-qwen35-image-execution.md
[tower]: ../archive/runs/2026-10-26-qwen35-vision-parity.md
[preprocess]: ../archive/runs/2026-10-26-qwen35-native-vision-preprocessing.md
