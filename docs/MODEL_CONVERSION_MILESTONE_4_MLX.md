# Milestone 4 continuation: Qwen3.5 text and LFM2 affine MLX

This packet continues the [conversion plan](MODEL_CONVERSION_PLAN.md) from `1b18197c` on `codex/conversion-milestone-4`. Version 0.7.0 independently converts existing affine MLX 4/8-bit Qwen3.5 text and LFM2 checkpoints. The runtime never imports or invokes Lalamo, JAX or PyTorch. Lalamo remains an external reference at `831accb8252a6b6606c378048caeb44178f1a050`.

## Reusable architecture and checkpoint coverage

The catalog gains ten Qwen3.5 text MLX profiles and eight LFM2 MLX profiles, for **81/110 implemented built-ins plus four extensions**. That is 42 dense and 39 MLX built-ins, with 29 built-ins still planned. Extensions preserve their previous identities and metadata rules.

- Qwen3.5: `mlx-community/Qwen3.5-{0.8B,2B,4B,9B}-MLX-{4bit,8bit}` and `mlx-community/Qwen3.5-27B-{4bit,8bit}`.
- LFM2: `mlx-community/LFM2-{350M,700M,1.2B,2.6B}-{4bit,8bit}`.

Adapters remain keyed by **architecture and input format**. Catalog identities bind metadata source rules and bit width. Exact native qualifications bind immutable source manifests, config/tokenizer/weights hashes and a named build. An uncatalogued future checkpoint using these accepted layouts can use `--metadata-profile hf-generic`; it needs self-contained metadata and retains an unqualified native status. No new catalog row is required. Known identities cannot bypass their catalog rules through the generic profile.

Both adapters use the unchanged sequential affine helpers for groups 16/32/64/128, little-endian I32/U32 packed words, complete weight/scale/affine-bias planes and BF16/F16/F32 parameter casts. They repack an existing checkpoint without computing new quantization. Individual unfused projections can stay dense; mixed representation or coefficient dtype inside one fusion fails. Tied duplicate heads are compared in every plane. Unsupported dimensions, missing/extra planes, mixed namespaces, contradictory bit/group metadata and source mutations fail before publication.

Qwen3.5 applies all gated-attention and DeltaNet projection permutations equally to packed weights, scales and biases. It preserves MLX's baked RMSNorm weights, requiring the primary `quantization` field instead of inferring these semantics from alias-only `quantization_config`. Nested wrapper fields retain top-level precedence. The real MLX wrapper uses `language_model.model.*` and `vision_tower.*`; vision omission uses the exact namespace associated with the selected decoder root and rejects mixed aliases. Declared vision/MTP payloads remain in the complete source hash/signature set even when omitted from text output.

LFM2 derives its convolution layout from actual embedding planes, following the pinned loader. Both 3D convolution aliases remain contiguous after singleton removal. Kernel-first flat 2D inputs use a new bounded transpose-and-F32 widening operation, including portable F16 NaN handling, exact BF16 bit widening and completed-file hashing after random tile writes. Dense embedding variants retain dense convolution layout rules.

## Implementation validation

Parallel workers owned the two family adapters and their independently generated reference fixtures. The coordinator owned catalog/CLI dispatch, generic integration, shared transpose widening, version/lock and documentation. An independent reviewer authored no production implementation, fixture generator or qualification/benchmark tooling. [Technical review](MODEL_CONVERSION_MILESTONE_4_MLX_REVIEW.md) records its disposition against final evidence.

The synthetic reference uses the actual pinned Hugging Face loader, config importer and full `Decoder.export`; every emitted tensor name/shape/dtype/payload hash/spec and decoder config is compared. There are **738 complete synthetic exports**: 432 Qwen3.5 and 306 LFM2. They cover both bit widths, all four groups, all nine source/output precision pairs, projection layouts, tied/untied heads, biases, convolution/topology variants, independently dense projections and cross-shard inputs. Separate IEEE cast-edge and malformed-input tests remain. Normal tests import no reference framework.

The integrated suite passed **2,807 tests / 60 skipped** with all twenty-five private metadata reference views. Skips are unqualified profiles without independently generated real metadata reference artifacts. After appending the four native records, the qualification/CLI/generic/provenance subset passed another **173 tests**. Ruff check/format, the locked 27-package dependency check, repository policy, seventeen policy unit tests and whitespace checks passed. The lock changes only the editable converter version, 0.6.0 to 0.7.0. A cached Rust 1.98.1 locked release/optimized build passed with the existing debug-strip tool warning; native source and binary hashes remain unchanged. Hosted CI was not run for this packet.

The first integrated run exposed two obsolete negative fixtures that named newly implemented MLX profiles. They now exercise the retained Qwen3.5 MoE block. Initial real validation also rejected Qwen's inspected MLX tensor namespaces; the failure and original fixture bindings remain private, separately bound from the final implementation and measurements.

## Exact real qualification

Each selected source is affine group 64, converted with BF16 matrix parameters and F32 normalization/recurrent/convolution parameters. Every output tensor/spec/config/tokenizer matches its independently generated pinned reference. Whole weights-file hashes differ because serialization order/header layout differs; all payload hashes are equal. LFM2 outputs 250 tensors, Qwen3.5 text 424. Each conversion retains the 8 MiB transformation-workspace bound.

| Checkpoint | Immutable primary revision | Source-manifest SHA-256 |
| --- | --- | --- |
| `mlx-community/LFM2-350M-4bit` | `18dc72abf3b2337f9123cfd6eeeb58dfa7947066` | `c26912c3f4fc1d5ddef5fdc82e3cede6d6571a6e4cce95acfcf194f20254a2b7` |
| `mlx-community/LFM2-350M-8bit` | `3b5014a1966c54313045fd0a23a098a220083b1e` | `d61aefdc2f77a7d29c40fb935132fdae04edae1bda5179968bdf18f2c62f5a3e` |
| `mlx-community/Qwen3.5-0.8B-MLX-4bit` | `5d894f8cc4ef3e6c88537bf3746ed262f549da6a` | `e71eec7c6849a003f53678bfea91acfb02ccf1d8ba55dae0ccc8f8ea4dc97c97` |
| `mlx-community/Qwen3.5-0.8B-MLX-8bit` | `45917392c34ba9b095b192a5acfad126ab764ed9` | `4b5905088cfc093b0d1ab031033ecd57f26c41f3bf41569f03f84f5cf8ed387a` |

Qwen3.5 metadata is selected from its corresponding official tokenizer repository, `Qwen/Qwen3.5-0.8B` at `2fc06364715b967f1860aea9cf38778875588b17`, and the official `Qwen/Qwen3.5-27B` generation repository at `fc05daec18b0a78c049392ed2e771dde82bdf654`, with the existing reviewed overrides. LFM2 uses its own tokenizer/Jinja and reviewed static generation defaults. All external revisions and role/content hashes are pinned in the private source manifests. The [native ledger](../tools/converter/stark_convert/native_qualifications.json) contains all four exact candidate weights/config/tokenizer hashes and named builds. The prior twenty-one records are unchanged, for **twenty-five exact qualifications and sixty implemented-unqualified identities**.

| Selected qualification evidence | SHA-256 |
| --- | --- |
| LFM2-350M 4-bit | `630c1d3673dca2ac070e68617d1a7f9669506932bc5ebe1c26d29677df951574` |
| LFM2-350M 8-bit | `2bee9c3ac7f2e7692cc00f5b3b10c70af25fb2d7cb7ac19063ef04c5b85ae7b9` |
| Qwen3.5-0.8B 4-bit | `eca97fef80ba2924543696be334e339de56c4d1ff58d18305ef9d1b9879c198c` |
| Qwen3.5-0.8B 8-bit | `40376227ff24d8b1f2c457985386cf258702645c206cbebbd82a670398e770ac` |

The named native target is Rust 1.98.1, release, default + optimized. Its worker hash is `61bb81a83fa368b7c8f647469ac468048cf6868d973ed74e7d9d62633bfba96d`; native source-manifest hash `0cc926bcf73db3921eab03f127185626b7f4311b73c62dad79b407c642d0c9af`, Cargo lock `65444f3e0e0707c43521539e43c95365cfc6a51ec6f9e90180bc43bc8fb4603e`, and Rust-toolchain file `72da9f9e503464ba7393560f6f51a69fb06be72633d52ae7725fa0ba59dfda22` remain unchanged. Readiness/EOF checks made zero scoring calls and retain permitted aggregates only.

## Preregistered live Lalamo benchmark

The new run contains **forty serial fresh complete commands**: four exact group-64 BF16 tuples, two backends and five repeats. Every selected source file is warmed immediately before each command. Backend order alternates by repeat and rotates its starting order between tuples. Model/test workloads were paused during measurement; no command was rerun and no observation was removed.

Candidate timing includes startup/imports, complete initial/final source hashing and signature/selection checks, metadata normalization, planning/export/output hashing, publication, receipt and exact `--require-native-support` matching. The external clean pinned Lalamo runner includes startup/imports and complete conversion, with weaker source/receipt verification. Setup/downloads, warming, parity comparisons and native readiness checks are outside timed commands. There is no old converter latency baseline for these previously unsupported architectures/formats.

Preregistered hard gates are every payload/spec/config/tokenizer parity, exact candidate source/artifact/qualification match, transform workspace at most 8 MiB and observed candidate process-tree RSS below 1 GiB. Hypotheses require candidate median command time no higher than Lalamo and median sampled RSS at most half Lalamo. All forty parities and twenty candidate hard-gate checks passed; both hypotheses pass for all four tuples.

| MLX checkpoint | Stark median, s | Lalamo median, s | Lower time | Median sampled RSS, MiB, Stark / Lalamo | Lower RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| LFM2-350M 4-bit | 0.532 | 1.853 | 71.3% | 113.0 / 1411.4 | 92.0% |
| LFM2-350M 8-bit | 0.623 | 1.581 | 60.6% | 113.2 / 1527.2 | 92.6% |
| Qwen3.5-0.8B 4-bit text | 0.997 | 2.059 | 51.6% | 203.5 / 4012.3 | 94.9% |
| Qwen3.5-0.8B 8-bit text | 1.393 | 1.940 | 28.2% | 203.5 / 4155.3 | 95.1% |

| MLX checkpoint | Stark range, s | Lalamo range, s |
| --- | ---: | ---: |
| LFM2-350M 4-bit | 0.507–0.679 | 1.764–2.714 |
| LFM2-350M 8-bit | 0.604–0.697 | 1.545–2.083 |
| Qwen3.5-0.8B 4-bit text | 0.964–1.065 | 2.042–6.228 |
| Qwen3.5-0.8B 8-bit text | 1.382–1.415 | 1.925–3.256 |

All observations, including slower reference observations, are retained. Sampled process-tree RSS uses 10 ms observations; it can miss peaks or double-count shared pages and is not an enforced total-memory ceiling. The workspace bound covers transformation buffers separately. Observed timing ranges are separated in this run, but five warm-file repeats on one host do not establish cold-cache, other-host, whole-family, future-checkpoint, execution-quality or inference-speed claims.

The run occurred **2026-10-12 13:28:56.712–13:30:48.889 UTC**, on arm64 macOS 26.7.1, eighteen logical CPUs, 128 GiB RAM and Python 3.13.13. Candidate and reference use the same executable hash; reference source/lock, source manifests, candidate/reference frozen artifacts, qualification evidence and native build were checked before and after the run. Every measured result is persisted before acceptance checks, and failed-command handling retains permitted timing/RSS/sample-count/category aggregates without diagnostic content. Timeout, nonzero exit and malformed-receipt retention were checked synthetically before freeze. No measured command failed.

## New source and evidence bindings

| Item | SHA-256 |
| --- | --- |
| Final converter sources | `075cb04034d9b811a8a336a245374b6f67f4c447b865b8e739dff649dad6cf9a` |
| Converter lock | `96fd35c0cfdf3895915d48e7a9203bc9416c4aa3515d40cc466ade96f66d755d` |
| Catalog, 81 built-ins + four extensions | `cc7b6a5ba1c8d1d171cabd8e2ce9763bfae8f11f8e7ddf4051783f928ced7187` |
| Native ledger, twenty-five records | `901743a8d946aa942b04f7c6493c1ee79be9ef17bb11b4bb546ce2ec86e69f33` |
| Preregistered objectives | `d023e06b95d39b62d597b0ab737197287875ad0d4c3e1433fb792222252e804e` |
| Final benchmark freeze | `7512a2eb38ca179bfb004c9a484f228e7605e606f5c73e59b1e3943d5ad6f104` |
| Forty-command report | `39b626bcd2762acae4b4a9c38522da6ad0ca95f1b80c9bf954e22efa339b6de5` |
| Benchmark harness | `70b2e185974b3612507e02ccb2a29b6500875939ac4672150d1f720fb722b413` |
| Unchanged external reference runner | `e785c2272366f523f79237be847549f76ea07bdc44b20cf69b7d76b6e38ad010` |
| Reference preparation | `beb83bfb36b5078fa84f9aa00a4d658f376f6b652cf70a81bfacd60dd5fc38b7` |
| Native qualification coordinator | `fb9ec1bcf1c3779d53fe35bf1e07ef2037cabbc2953a1d6b3d730d8dbbf72f7d` |
| Original Qwen3.5 324 exports | `bc3a209b4884e33cb58934b2fbc9195dd5c0a83b4643b2d92c94890bfbbcdc83` |
| Additional Qwen3.5 108 MLX-VL exports | `4e1ceb3f7268dbcef78f78d869745db4e32d54af6738faf14f0a985493dae814` |
| LFM2 306 exports | `14759692f1d42889f564fe8094d7be5e049a872f80f1e103ab85d5a13d4a74bf` |
| Initial closed Qwen-prefix attempt aggregate | `9c02a0fb63224b9250d8d9835a4409cdaa6004cd5d12899ffe0946ab17201789` |

Final load-only qualifications bind the pre-ledger converter source `21b091e92a2caafed15b52c7f1c4b884760f084f509f1726b6663c7793749770`. The benchmark separately binds the appended twenty-five-record ledger and final source. Initial successful LFM qualification/output evidence and the closed Qwen-prefix failure remain privately preserved; final qualification reran all four after the inspected namespace update. The original 324-case Qwen fixture is unchanged, with the new 108-case fixture and generator recorded separately. Earlier milestone results are never reassigned to newer code.

## Acceptance boundaries and remaining work

This packet expands scoped G4/G5 conversion coverage. Native evidence is readiness followed by stdin EOF, with **zero scoring/generation calls**. It establishes exact artifact construction with the recorded build; execution correctness, quality and inference latency remain unverified. Other bit/group/config/checkpoint combinations are implemented-unqualified. F16/F32 output is conversion-only and runtime-blocked. Generic routes remain unqualified, with additional runtime blocks for unsupported dimensions/configs.

Qwen3.5 MoE, later Qwen/LFM generations, other compressed schemes, remaining state-space/classifier/TTS/NeMo rows and complete replacement accounting remain open. The dense Qwen3.5 latency objective and user-deferred SmolLM3 8-bit latency retain their existing evidence and status. The next architecture work is the individually inspected Mamba/LLaMBA/state-space packet.

All real model inputs, operational source manifests, native artifacts and reference outputs remain ignored/private. Public evidence contains public source identities, permitted aggregates and hashes. No prompts, responses, token vectors, decision probabilities, credentials or exception dumps are recorded. No Jev/JevK5 execution, host integration, distribution, cloud/signing/release infrastructure or dependency update occurred.

Adapted reference sources include `lalamo/model_import/model_configs/huggingface/{qwen3_5,lfm2}.py`, `model_import/loaders/huggingface.py`, Qwen/LFM model specs and decoder/layer exports. [Lalamo's MIT license](../tools/converter/LICENSE.lalamo), Mirai copyright and Stark's Uzu base revision `8d01581474760bd1fc7f4a547ff1d02f7dada437` remain preserved. Earlier qualification/benchmark documents retain their original source bindings.
