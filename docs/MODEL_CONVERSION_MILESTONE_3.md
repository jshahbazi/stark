# Milestone 3: independent MLX Qwen3 conversion

Validation date: 2026-10-11. This is the first compressed-format packet under Phase 4, with scoped G4 evidence for two exact MLX affine Qwen3 tuples. Complete Phase 4 and full Lalamo replacement remain open. Work began on `codex/conversion-milestone-3` from merged `main` at `41db29b7`.

The independent converter now implements the 12 pinned `Qwen/Qwen3-{0.6B,1.7B,4B,8B,14B,32B}-MLX-{4bit,8bit}` profiles. Total implementation coverage is **45 of 110 built-ins plus four extensions**, or 49 identities. Two new BF16 load-only qualifications bring the exact qualification ledger to 13; the prior 11 records are preserved. The remaining 65 built-ins are unimplemented, and 36 implemented identities have no supplied exact native qualification. There is no runtime Lalamo fallback or dependency.

See the [independent technical review](MODEL_CONVERSION_MILESTONE_3_REVIEW.md), [conversion plan](MODEL_CONVERSION_PLAN.md), and [usage](../tools/converter/README.md). Historical prototype and milestone 1/2 results retain their original source hashes and outcomes.

## Format and implementation

The new [compressed helper](../tools/converter/stark_convert/compressed.py) and [Qwen3 planner](../tools/converter/stark_convert/qwen3.py) repack existing MLX affine checkpoints. They do not compute new quantization. Supported bit widths are 4 and 8; group sizes are 16, 32, 64, and 128. Logical input widths must be divisible by both the group size and packing factor. BF16/F16/F32 coefficient conversions use the existing independently tested cast semantics. Native qualification remains BF16 only.

| Plane | Hugging Face source | Stark output |
| --- | --- | --- |
| Packed weights | Little-endian I32/U32 `[rows, columns / (32 / bits)]`, low code first | Identical bytes as U8 `[rows, columns * bits / 8]` |
| Input/tied embedding scales and affine biases | Float `[vocab, model_dim / group_size]` | Target float dtype, unchanged physical orientation, `input_output` spec |
| Projection/readout scales and affine biases | Float `[output, input / group_size]` | Target float dtype, transpose to `[groups, round_up(output, 4)]`, zero padding, `output_input` spec |
| Normalization scales | BF16/F16/F32 vectors | Existing F32 widening |

QKV rows fuse in Q, K, V order; MLP rows fuse in up, gate order. Every compressed matrix consumes its weight, scale, and affine-bias triplet, including duplicate tied-head checks. Padding applies after complete row fusion. Tiled coefficient export handles shard and workspace boundaries without a full matrix allocation. Packed weights stream directly. Random coefficient writes trigger the existing completed-file hash path; full initial/final source hashing and publication checks remain mandatory.

Separate matrices may be dense or compressed. A fused group must have one representation and a uniform coefficient dtype. Missing planes, wrong shapes/dtypes, unexpected tensors, contradictory quantization declarations, unsupported modes/options/groups, and mixed fused layouts fail explicitly. Catalog entries bind the declared bit width. Dense catalog profiles cannot route packed checkpoints, and a 4-bit profile cannot accept an 8-bit source. Reviewed `qkv_bias` and `use_qk_norm` annotations are checked against accepted Qwen3 semantics.

The source inspection used actual pinned reference loaders/exporters and native owning code:

- Lalamo `model_import/loaders/huggingface.py` for I32 unpacking, row fusion, matrix/embedding layouts and dtype casts; `compressed/mlx.py`, `compressed/quant_params.py`, and `compressed/utils/packing.py` for byte order, dequantization and exported padding.
- Stark `encodable_block/weight_matrix.rs`, `embedding_table.rs`, `linear/matmul.rs`, and the Metal matmul/embedding kernels for ScaleBias, packed shapes, coefficient strides and groups. Positive group acceptance by the loader alone is insufficient for compute-route acceptance. The initial group set matches inspected GEMV/GEMM contracts.

The compressed inventory also contains Llama, SmolLM3, Gemma, LFM, Qwen3.5, Llamba and 1-bit rows. Their adapters and other schemes remain separate work. MLX helper availability does not grant those rows conversion or execution support. No Rust, Metal, native worker, Cargo dependency, or inherited infrastructure was changed. `stark-convert` is now 0.4.0; dependency versions are unchanged.

## Exact real artifacts and native loading

Both real checkpoints use affine group size 128 and BF16 output. Their tokenizer, chat-template and generation files resolve from `Qwen/Qwen3-0.6B` at `c1899de289a04d12100db370d81485cdf75e47ca`; the primary compressed revision is independently pinned in each manifest. The external reference runner uses those same explicit primary/external pins offline.

| Registered identity | Primary revision | Tensors | Safetensors bytes | Exact reference parity | Named native build load-only |
| --- | --- | ---: | ---: | --- | --- |
| `Qwen/Qwen3-0.6B-MLX-4bit` | `173234aa840d113125e9f2271100ddbaf16c9620` | 452 | 316,959,744 | Passed | Passed |
| `Qwen/Qwen3-0.6B-MLX-8bit` | `e53d3ae02ebc79eac17f1fa9e23a2b65ce227893` | 452 | 614,952,000 | Passed | Passed |

Parity compares every name, shape, dtype, payload hash, specification, logical metadata field, complete root/nested config, and normalized tokenizer. Whole-file reference hashes can differ with serialization ordering. Each artifact loaded with the unchanged optimized Rust 1.98.1 worker, emitted readiness, and exited successfully on stdin EOF with **zero native scoring calls**. This establishes construction and complete parameter consumption by the named build. No scoring, generation, judge evaluation, Jev/JevK5 run, or inference-speed/quality qualification was performed.

`--require-native-support` matches registered identity, BF16 dtype and all source/artifact hashes, then returns the stored complete named-build receipt. It provides no qualification for another binary, revision, dtype, group, or config. F16/F32 remain conversion-only and runtime-blocked by converter policy. Ten other compressed Qwen3 identities and all unmeasured groups remain unqualified.

## Synthetic and repository validation

The final integrated converter suite passed **1,310 tests with 36 skipped**, using 13 supplied private metadata references. Without those references, the model-free count is 1,297 passing cases and 49 unsupplied-reference skips. The suite includes:

- 288 frozen actual pinned-loader/exporter/decompress cases: 4/8 bits, four groups, all nine source/target float pairs, both layouts, and single/fused matrices. Every 4/8-bit code, signed high-bit words, independently varied row/group coefficients, global 17-row padding, shard boundaries and a 96-byte workspace are covered. Packed/spec hashes and binary-exact synthetic dequantized values use a predeclared tolerance of zero.
- 72 complete planner dtype/group combinations, tied/untied heads, mixed dense/packed separate matrices, projection biases, fused padding and 16–4,096-byte workspaces, malformed planes/configs/operations and source consumption.
- 18 transposed coefficient cast edge cases tied to frozen reference bit patterns, including signed zero, NaNs, infinities, subnormals and rounding boundaries.
- 12 complete synthetic CLI artifacts independently compared with externally generated pinned reference outputs. All nine float pairs, four groups, both bit widths, metadata, cross-shard layouts, tied/untied embeddings and dense/packed combinations passed exact full artifact parity. The first harness attempt stopped before conversion because it lacked explicit synthetic external-metadata resolution; its aggregate is preserved separately. The corrected harness supplies every dependency locally.
- End-to-end compressed catalog selection, both backends, immutable external pinning and offline replay, both bit-width mismatches, dense-profile packed rejection, private staging cleanup on parameter-writer failure, and inherited mutation/publication/qualification/import-isolation regressions.

Ruff lint/format passed for 39 files, the converter lock check passed, repository policy checks passed, and 17 repository CI-policy tests passed. The locked optimized Rust 1.98.1 build succeeded with the existing libLLVM debug-strip warning and unchanged worker hash. Hosted CI execution is not claimed. No inherited workflow was enabled and no new dependency advisory scan is claimed.

## Complete-command performance

Objectives were recorded before timing: exact artifact parity, unchanged source/tooling, transformation workspace at most 8 MiB, sampled candidate process-tree RSS below 1 GiB, candidate complete-command median no slower than pinned Lalamo, and candidate median sampled RSS at most half the reference. Both exact tuples passed every objective.

The protocol ran three repetitions per backend and tuple in fresh direct Python processes, serially with other model-dependent work paused. Source files were warmed before each command, and backend order alternated. Setup/download, warming and parity comparison were excluded. Complete command time includes startup, source verification, planning/metadata, writing, hashing and receipts. Lalamo's external command has weaker source verification/receipts, so these ratios compare complete commands rather than identical isolated transformations.

Host: Apple silicon arm64, 18 CPUs, 128 GiB RAM, Darwin 25.6.0, managed Python 3.13.13. Converter dependencies retain NumPy 2.5.3, ml-dtypes 0.6.0, tokenizers 0.23.3, huggingface-hub 2.2.0 and psutil 7.2.2.

| BF16 output tuple | Streaming median [range] | Lalamo median [range] | Observed complete-command speedup | Median tree RSS: streaming / Lalamo |
| --- | --- | --- | ---: | --- |
| Qwen3-0.6B MLX 4-bit, group 128 | 1.011 s [1.006–1.099] | 1.796 s [1.785–1.900] | 1.78× | 188.3 / 2,703.2 MiB |
| Qwen3-0.6B MLX 8-bit, group 128 | 1.506 s [1.320–1.521] | 1.818 s [1.802–1.921] | 1.21× | 188.5 / 2,805.2 MiB |

Sampled RSS fell by 93.0% and 93.3% respectively. Each of the 12 timed outputs passed complete reference parity outside measurement, and all six final-source candidate outputs matched the stored native qualification. RSS samples are spaced 10 ms apart, may miss peaks, and may count shared pages more than once. The workspace bound excludes headers/catalogs, tokenizer normalization, hashing and interpreter overhead; it is not a process-memory ceiling.

These are descriptive warm-cache conversion results for two small checkpoint tuples on this host. They do not establish latency parity for other models, cold storage, inference, or all compressed formats. Milestone 2's eight failed dense latency hypotheses remain open and unchanged.

## Source-bound hashes

| Input or evidence | SHA256 or revision |
| --- | --- |
| Lalamo reference source | `831accb8252a6b6606c378048caeb44178f1a050` |
| Stark Uzu base | `8d01581474760bd1fc7f4a547ff1d02f7dada437` |
| Final converter sources/configuration | `42a883a195b095b5812a0bbb5502b85ef8efa5840b4181f6d347fa11d30fea0c` |
| Converter dependency lock | `d230bd2dc0681a13a081876eb7677bd22b7d1f029bf2d046cc0bf2e50156e658` |
| Catalog | `4db2607b2a5ad06766d0f3667d99706c413e5649dffdaf631775f638c1d424c1` |
| Thirteen exact native qualification records | `61b8b1ae7fc5abadbe7eb1e469ac24b3cf6ca175b4d1f7b85fc25c2b10b8501d` |
| Complete two-tuple benchmark | `03b7bab9fd26503dcfae901ae7d3fb399aa1a2a627c2d2bea0fae1674bcf319c` |
| Benchmark coordinator | `14484f3ba5629f0fe54f217265abafc8e915ba330f1e32662f8412a3d8f8239c` |
| External reference runner with explicit source pins | `d6ce08e6aac94509f00eac81da4868477a876ed3d90d91486a2a7f7056adc071` |
| Initial qualification summary and final tooling | `739f5c2b090597df67906a184a5ab743a37e9590bf31268681d4a5c7d1301e06` |
| Corrected twelve-case full synthetic aggregate | `d4efc149a4f62831d40ca008337eada086b6295e3d3c86ef4dac5c6df3f1aeb9` |
| Preserved initial synthetic harness attempt | `8cfe8f20e3598b798aad6e50731d226dff036e8ba9baab3512484499e469fe31` |
| Frozen 288-case MLX numerical/export fixture | `6e90f3262ce84f7e0f791dd250fc05078929f1a0c26b7dfaf196b01d50dc5d15` |
| MLX 4-bit source manifest | `9b868e70c1cf5d09bf845fd267c1aff071b6bb2d43e935702d2f7fd3eed8d9e7` |
| MLX 8-bit source manifest | `86f32a8bc00e0943aa92dca65fccd1d556be0c6e7c40c2c688ac92336b20020d` |
| MLX 4-bit weights | `cbd6d43eb6a40e56066c5498a72fee0c8e8c7aa316a29da7c6cdc79284611337` |
| MLX 8-bit weights | `c279f5ca586407745d1faa75ecd35de67dac93d692bea9d00f08025ba80989f0` |
| MLX 4-bit complete config | `713e0ad35c89bfdb8bc28eb5bfffecbc223ecd52256756f466e89fa9de585269` |
| MLX 8-bit complete config | `dfa5763dc220e871b86a6119536f5c1c2862a5137a473eaf9d812be3cc53924e` |
| Shared normalized tokenizer | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| Unchanged native worker | `61bb81a83fa368b7c8f647469ac468048cf6868d973ed74e7d9d62633bfba96d` |
| Native source manifest | `0cc926bcf73db3921eab03f127185626b7f4311b73c62dad79b407c642d0c9af` |
| Cargo lock | `65444f3e0e0707c43521539e43c95365cfc6a51ec6f9e90180bc43bc8fb4603e` |
| Rust toolchain file | `72da9f9e503464ba7393560f6f51a69fb06be72633d52ae7725fa0ba59dfda22` |

The initial real qualifications bound converter source `7e7a458262cd43224f990b78d2ed9c7352bad0904dda25bfafe74e5422a89593`. Adding qualification records and a stricter per-source packed-byte validation changed the converter hash. Those initial artifacts and receipts are preserved; six fresh final-source benchmark conversions reproduced the exact qualified artifacts. No older evidence was silently refreshed.

Models, private manifests, operational config/tokenizer files and permitted aggregate/hash receipts stay in ignored owner-only storage. Public documentation contains public identities/revisions, aggregate outcomes and hashes. Prohibited execution content was not captured. Lalamo's MIT notice and adapted source provenance remain preserved.

## Remaining compressed work

The next dependency-ready packet is reuse of the reviewed MLX helper for separately inspected dense Llama/SmolLM/Gemma text profiles, with their own exact metadata and native qualifications. Other compressed schemes need actual loader/spec/packing inspection and numerical protocols. State-space/hybrid/multimodal architectures retain their separate adapter dependencies. Full G4, complete registry replacement, arbitrary Hugging Face support, strict process-memory enforcement, distribution and execution quality remain open.
