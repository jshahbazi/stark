# Milestone 4 MLX continuation: independent technical review

Reviewed on 2026-10-12 against committed baseline `1b18197c`, the final converter-source manifest below, and [the scoped packet](MODEL_CONVERSION_MILESTONE_4_MLX.md). The reviewer authored no production code, public tests, export fixtures, qualifier, benchmark harness or native ledger entries. This document is the reviewer's only repository edit.

## Disposition

**Accepted for the scoped conversion and exact named-build load-only packet. No blocking finding remains.** The implementation adds eighteen pinned affine MLX profiles: ten dense-text Qwen3.5 and eight LFM2. Four selected group-64 checkpoints converted with BF16 matrix parameters have exact artifact qualifications. The prior twenty-one native records remain unchanged.

This accepts the recorded construction, source verification, transformation-workspace and benchmark evidence. Native validation is readiness followed by stdin EOF, with zero scoring or generation calls. Execution correctness, model quality and inference latency are unverified. Other model sizes, configurations, groups and precision combinations remain implemented-unqualified; full G4/G5 and complete Lalamo replacement remain open.

## Implementation and contract inspection

The reviewer inspected the pinned reference's actual config importers, Hugging Face matrix/attention/DeltaNet/short-convolution loader, decoder-root dispatch, affine export and parameter-plane layout. Enum declarations alone were not used as proof of an input format. The production converter's shared affine helpers are unchanged, and its dependency lock changes only the editable package version from 0.6.0 to 0.7.0.

Qwen3.5's contiguous row views apply each gated-attention and DeltaNet permutation equally to packed weights, scales and affine biases. The views retain the original file identities/signatures; the plan retains every original input. All three DeltaNet layouts and both gated-attention representations are compared with actual full decoder exports. Mixed packed/dense representations within one fusion are rejected; independent projections can remain dense. Every plane of a duplicate tied head is hash-compared before omission.

The pinned Qwen config importer selects baked RMSNorm semantics through its primary `quantization` field. Review identified the ambiguity of accepting alias-only `quantization_config`; the final planner rejects it. Identical duplicate descriptions are accepted, disagreement is rejected, and normalized wrapper config retains top-level precedence. The Qwen loader uses its standard convolution layout without LFM's transpose rule; the full reference cases cover channel-first 2D and both singleton 3D representations.

Initial real validation rejected the inspected MLX-VL namespace. The final adapter associates `language_model.model.*` exclusively with declared `vision_tower.*` omission; the other accepted decoder roots use declared `model.visual.*`. An explicitly declared text/vision wrapper is required. Cross-wrapper visual aliases, multiple decoder roots, undeclared MTP and alternative MTP prefixes fail. New, separately bound actual VL exports cover the physical namespace, nested heads and outer-only quantization metadata. Accepted auxiliary payloads remain subject to complete source verification.

LFM2 derives convolution permutation from the actual embedding planes, matching its specific pinned loader branch. The two singleton 3D shapes and kernel-first 2D shape have full reference comparisons, including dense embedding and independently dense projection variants. Kernel-first inputs use the new bounded F32 transpose. Its tile allocation accounts for source reads, cast/NaN scratch and contiguous output; BF16 widening preserves bits, F16 follows the frozen portable NaN policy, and F32 preserves payload bits. Random output writes trigger completed-file hashing. Tiny-workspace, IEEE edge, digest and invalid-operation preflight checks passed.

CLI dispatch normalizes Qwen wrapper config before checking catalog bit width. An eighteen-identity catalog diff was independently checked: ten Qwen3.5 and eight LFM2 rows change; the four extensions are unchanged. Qwen3.5 MoE and later Qwen/LFM generation rows remain planned. Missing/extra/malformed planes, incompatible packing, conflicting config, fusion coefficient dtype and unsupported schemes fail closed.

The future-checkpoint route remains architecture/input-format based. `hf-generic` can reuse either new affine adapter with self-contained metadata and schema-2 offline replay. Known identities retain their catalog source and bit-width rules; they cannot use generic metadata to bypass a planned row or external metadata requirement. Generic conversion does not grant native qualification. Existing template, stop metadata, optional-sidecar selection and source-role validation remain in force.

## Independent validation

After the family implementations settled, reviewer reruns passed **590 LFM2/dense/cast cases** and **616 Qwen3.5/dense/generic cases**. A further CLI/metadata rerun passed **132 cases / 85 skipped** without private reference views. These overlapping subsets are not summed into a unique test count. The coordinator's integrated final suite passed **2,807 cases / 60 skipped**, followed by **173 qualification/CLI/generic/provenance cases** after ledger append. Two obsolete unsupported-profile negative fixtures were corrected to exercise the retained MoE block.

The independent reviewer also changed an omitted synthetic auxiliary payload after the writer's signature check, keeping file size and restoring modification time. Final complete-content verification rejected publication and removed staging/output. This checks the omission and generic-publication boundary beyond a simple timestamp mutation.

The frozen synthetic oracle has **738 actual full decoder exports**: the preserved 324 Qwen affine cases, 108 separately generated MLX-VL cases and 306 LFM2 cases. Fixture payload/spec/config comparisons passed. The reviewer rehashed their private generators and the original/final recipe copies, preserving the initial Qwen fixture's original source binding.

For all four real checkpoints, the reviewer independently recomputed every emitted tensor name/shape/dtype/payload hash, matrix spec, logical JSON config and tokenizer comparison against the retained pinned reference outputs. All match the stored proof. Different serialization/header ordering and JSON formatting explain whole-file hash differences between backends; each candidate's exact whole-file hashes remain individually bound in the ledger. All input source contents, immutable manifest revisions, qualification hashes and named binary/build inputs were rechecked.

The four appended ledger rows match those independently checked source/artifact/native proofs. All prior twenty-one rows and the ledger schema match committed baseline `1b18197c`; the ledger has twenty-five distinct repository identities. Each real native proof reports readiness/EOF success and zero native score calls with the unchanged recorded build.

## Benchmark review

The preregistered benchmark contains **forty serial fresh Python commands**, five candidate and five live pinned Lalamo observations for each of four selected checkpoints. Every selected source file is warmed before each command; backend order alternates and its initial order rotates across checkpoints. Download/setup, warming, parity checks and native readiness are excluded. Candidate timing includes complete source verification, metadata, output construction/hashing, exact qualification lookup and receipt work. Lalamo's weaker source/receipt verification remains disclosed.

Review required preserving a measured observation before acceptance checks could reject it. The final harness persists timing/RSS/receipt aggregates before source/artifact/native/parity checks and retains safe aggregate failed-command records for timeouts, nonzero exits and receipt failures. Failures receive status/category fields without diagnostics. No timed observation failed, was removed or was replaced by a favorable rerun.

The reviewer independently checked all forty backend/repeat/order records and parity results. All twenty candidate capabilities report `qualified_tuple`; their entire qualification objects equal the exact ledger rows. Every candidate passes the frozen whole-artifact, source/tooling, 8 MiB transformation-workspace and sampled RSS checks. The report, freeze, current/post-run tooling, objectives, runner and unchanged native build hashes match.

| Selected checkpoint | Candidate median [range], seconds | Lalamo median [range], seconds | Lower median time | Lower median sampled tree RSS |
| --- | --- | --- | --- | --- |
| LFM2-350M 4-bit | 0.532 [0.507–0.679] | 1.853 [1.764–2.714] | 71.28% | 91.99% |
| LFM2-350M 8-bit | 0.623 [0.604–0.697] | 1.581 [1.545–2.083] | 60.58% | 92.59% |
| Qwen3.5-0.8B 4-bit | 0.997 [0.964–1.065] | 2.059 [2.042–6.228] | 51.58% | 94.93% |
| Qwen3.5-0.8B 8-bit | 1.393 [1.382–1.415] | 1.940 [1.925–3.256] | 28.21% | 95.10% |

Medians, ranges, speed ratios and hypothesis booleans were recomputed from all observations. All four raw latency and memory hypotheses pass, with disjoint observed timing ranges. These are descriptive five-repeat warm-cache results on one arm64 macOS 26.7.1 host with 18 logical CPUs, 128 GiB RAM and Python 3.13.13. They establish no statistical significance, other-host/cold-cache result or family-wide timing. RSS is sampled every 10 ms, can miss peaks or count shared pages repeatedly, and its acceptance threshold is checked after the command rather than enforced as an OS limit.

## Reviewed source bindings

| Artifact | SHA-256 |
| --- | --- |
| Final converter-source manifest | `075cb04034d9b811a8a336a245374b6f67f4c447b865b8e739dff649dad6cf9a` |
| Converter lock | `96fd35c0cfdf3895915d48e7a9203bc9416c4aa3515d40cc466ade96f66d755d` |
| Catalog | `cc7b6a5ba1c8d1d171cabd8e2ce9763bfae8f11f8e7ddf4051783f928ced7187` |
| Twenty-five-record native ledger | `901743a8d946aa942b04f7c6493c1ee79be9ef17bb11b4bb546ce2ec86e69f33` |
| Qualifier before ledger append | `fb9ec1bcf1c3779d53fe35bf1e07ef2037cabbc2953a1d6b3d730d8dbbf72f7d` |
| Benchmark objectives | `d023e06b95d39b62d597b0ab737197287875ad0d4c3e1433fb792222252e804e` |
| Final benchmark harness | `70b2e185974b3612507e02ccb2a29b6500875939ac4672150d1f720fb722b413` |
| Final benchmark freeze | `7512a2eb38ca179bfb004c9a484f228e7605e606f5c73e59b1e3943d5ad6f104` |
| Final benchmark report | `39b626bcd2762acae4b4a9c38522da6ad0ca95f1b80c9bf954e22efa339b6de5` |
| Preserved 324-case Qwen affine fixture | `bc3a209b4884e33cb58934b2fbc9195dd5c0a83b4643b2d92c94890bfbbcdc83` |
| Separate 108-case Qwen MLX-VL fixture | `4e1ceb3f7268dbcef78f78d869745db4e32d54af6738faf14f0a985493dae814` |
| 306-case LFM2 affine fixture | `14759692f1d42889f564fe8094d7be5e049a872f80f1e103ab85d5a13d4a74bf` |
| Preserved initial closed-failure attempt | `9c02a0fb63224b9250d8d9835a4409cdaa6004cd5d12899ffe0946ab17201789` |

Final real qualifications retain pre-ledger converter-source binding `21b091e92a2caafed15b52c7f1c4b884760f084f509f1726b6663c7793749770`; the subsequent freeze/timed receipts bind the final ledger-containing manifest above. Native source, binary, Cargo lock and Rust-toolchain bindings are unchanged and recorded in the packet and ledger. The external reference is clean at `831accb8252a6b6606c378048caeb44178f1a050`. Private manifests, inputs, native artifacts and reference outputs remain ignored; public evidence contains permitted aggregates, source identities and hashes.

The dense Qwen3.5 latency objective, deferred SmolLM3 8-bit latency, remaining twenty-nine built-ins, other schemes and broader execution qualification retain separate gates. This packet supplies no Jev/JevK5 execution result or host-application integration.
