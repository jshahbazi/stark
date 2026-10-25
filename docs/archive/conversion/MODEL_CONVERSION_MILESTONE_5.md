# Milestone 5: LLaMBA and the Mamba2 mixer

This packet continues the [replacement plan](MODEL_CONVERSION_PLAN.md) from `ade81001` on `codex/conversion-milestone-4`. Version 0.8.0 independently converts the two LLaMBA identities in pinned Lalamo revision `831accb8252a6b6606c378048caeb44178f1a050`: `cartesia-ai/Llamba-1B` and `cartesia-ai/Llamba-1B-4bit-mlx`. The installed converter never imports or invokes Lalamo, JAX or PyTorch.

Artifact lifecycle update, 2026-10-12: after this packet's validation, review and commit, the user directed removal of completed conversion model weights. The [separate cleanup record](MODEL_CONVERSION_ARTIFACT_CLEANUP.md) documents deletion of this packet's input and frozen output weights. Retention statements below describe the original validation state; small evidence records and original hashes remain retained.

## Architecture, format and source contracts

The catalog now implements **83/110 built-ins plus four extensions**, comprising 43 dense and 40 MLX built-ins. Twenty-seven built-ins remain planned. This packet adds a reusable `llamba` decoder adapter and its Mamba2 mixer helper. There is no separate Mamba1 or general Mamba architecture row in the pinned registry; those architectures are not claimed here. Extensions and previous reference/qualification results retain their original bindings.

The planner handles the canonical backbone decoder and the alternating SSM/MLP decoder, complete declared layer order, tied/untied embeddings, accepted projection/MLP/convolution biases, norm semantics, projection fusion and 2D/3D convolution aliases. Config dimensions, head/group divisibility, vocabulary padding, activation, tensor names/shapes/dtypes and full source consumption are validated. Missing recurrent gate or skip parameters fail instead of retaining empty or nondeterministic reference initializer values. Unknown variants, partial planes, mixed representation within a fusion and contradictory fields fail explicitly. Independent dense embeddings/projections can coexist with packed matrices.

Outer dense matrices support independently implemented BF16/F16/F32 casts. Norm, convolution, recurrence skip weights and gate bias stay F32. Mamba2 uses the pinned LLaMBA identity activation and fixed recurrence conventions. Native kernel sizes below two are runtime-blocked. Other configurations retain unverified native status until exact source/artifact/build qualification.

The registered MLX profile is 4-bit. Its primary repository contains no model config. The external dense LLaMBA config supplies structural dimensions; the catalog supplies the bit width, while complete packed/scales/affine-bias planes establish group size and layout. Every matrix must agree with the accepted groups 16/32/64/128. The real checkpoint uses group 128 and retains a dense embedding. Repacking uses the existing bounded affine helper and does not compute new quantization.

The real alternating MLX layout contains a separately quantized `head.linear` even though the selected config declares tied embeddings. The pinned decoder uses the dense embedding for both directions. This adapter validates the complete auxiliary head triplet and retains all source signatures/hashes, then deliberately omits that unused approximate head. Its values are not asserted equal to the dense embedding. This exception requires the alternating layout, declared tied embeddings and an explicit MLX representation. If both embedding and duplicate head are packed, all three planes must match. Dense duplicate heads are compared; backbone dense-embedding/packed-head variants and unknown or partial head namespaces fail.

`SourceSnapshot.config_path` now selects the declared primary or external model-config record. The MLX physical root comes from selected primary weights rather than an invented primary config or the external repository path. Exact deterministic catalog sidecar/weight descriptors are validated on live checks and schema-1 replay, including repository, filename and role. External model-config paths must lie outside the primary checkpoint by both lexical and resolved routes. Unselected primary config decoys cannot override the external config. Full initial/final content verification, original signatures, private manifests, atomic publication and sanitized errors remain in force.

Registered LLaMBA metadata uses its pinned fixed EOS property and the official Llama tokenizer/generation source rules. Official Hugging Face access succeeded for `meta-llama/Llama-3.2-1B-Instruct` at `9213176726f574b556790deb65791e0c5aa438b6`; no alternate transport was needed. An uncatalogued future checkpoint can use `--metadata-profile hf-generic` with `model_type=llamba` and self-contained config/tokenizer/template/stop metadata. Generic MLX requires explicit quantization; generic metadata keeps its own stop settings and remains native-unqualified. Known dense identities cannot bypass their compressed profile boundary.

## Parallel implementation and review

Separate workers owned the Mamba2 helper and LLaMBA decoder/oracles. The coordinator owned source/CLI/catalog/metadata/generic integration and version/lock changes. An independent reviewer authored no production code, tests, generators, qualifier, benchmark harness or native ledger. [Technical review](MODEL_CONVERSION_MILESTONE_5_REVIEW.md) records the final disposition.

The synthetic oracle includes **108 actual full decoder exports** and **72 isolated actual Mamba2 exports** from the pinned config importer, loader and export implementation. Both use all nine source/target precision pairs. Full decoder cases cover both physical layouts, dense/MLX, tied/untied/bias variants, convolution aliases, group/expansion variants and cross-shard inputs. Shared numerical, malformed-input, source-mutation and transactional publication regressions remain. Normal tests import no reference frameworks. Final fixture, generator and recipe copies are retained with matching hashes. Earlier provisional decoder-fixture generations were not fully archived before formatting/negative-test extensions; their reported aggregates remain separate and are not used as final source-bound proof.

Independent review reproduced and resolved a dense-profile packing bypass and an external model-config redirect to an unselected primary file. Regression tests include consistently rehashed role/repository/path forgeries, both symlink directions, normalized path aliases, restored-timestamp mutations, config-less remote roots, explicit generic metadata and offline replay. The first full integrated run passed **3,300 tests / 62 skipped**, using the prior twenty-five real metadata views; final qualification and validation are recorded separately below.

The final integrated suite passed **3,303 tests / 60 skipped** in 37.82 s with all twenty-seven private real metadata reference views and the final native ledger. Skips are implemented identities without supplied independently generated real metadata reference artifacts. Adding the external-config case exposed an older metadata test's assumption that every primary source owns `config.json`; the test now reads the declared model-config repository/file. Production conversion and independent full metadata parity had already passed. Ruff check/format, the unchanged 27-package locked dependency check, repository policy, seventeen policy unit tests and whitespace checks passed. The lock changes only the editable package version from 0.7.0 to 0.8.0. Hosted CI was not run for this packet.

## Exact real qualification

| Checkpoint | Immutable primary revision | Source-manifest SHA-256 |
| --- | --- | --- |
| `cartesia-ai/Llamba-1B` | `212c94fc91c28f607542232da347693dde248063` | `18c089adce1c1c64625ba0e525831d9b7805cca9ae03be28dcd180b5f9c67e91` |
| `cartesia-ai/Llamba-1B-4bit-mlx` | `24fcab78470f75b4f9082fc4caa74baf1699ee7e` | `8a1620e749d037b3046bdd92d7f6226c47da9c9a04846687492b0e6b0a6cc1bd` |

The MLX model config selects the dense repository at the revision above. Both select official Llama metadata at `9213176726f574b556790deb65791e0c5aa438b6`. Source files retain immutable revisions, content hashes, file roles and signatures. Independent review recomputed every complete candidate/reference tensor/spec/config/tokenizer comparison and every source/artifact/native binding before timing. Both candidates remain stored alongside their frozen reference weights.

Dense input contains 178 F32 tensors / 5,620,639,744 payload bytes, producing 162 tensors / 2,811,504,640 payload bytes. MLX input contains 341 tensors / 1,843,111,936 payload bytes, producing 290 tensors / 1,134,340,096 payload bytes. Both target BF16 outer matrix parameters with F32 norm, convolution, recurrence skip and gate parameters. The MLX tuple is exactly 4-bit/group-128. Both use the same inspected native configuration: sixteen layers, model dimension 2048, thirty-two heads/groups, head and state dimensions 64, kernel size four and identity activation.

Whole weights-file hashes differ between candidate and reference because of serialization order/header layout; every tensor payload/spec and logical config/tokenizer matches. Exact candidate weights are `6c7a0ae43a9c4361c24287642616334c9aaa532ff78b8f6ada9f4045f4525920` (dense) and `dd5efd9e44d4b5f223e8d10caa40aaf84ad9aa7d905e9310eef633a5fa985142` (MLX). Both root configs hash to `dbae5c4611302c0b318fd5aeda93beb61bb2eab1669c9aab3271f62ee4b07b42`; normalized tokenizers hash to `6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b`.

The [native ledger](../../../tools/converter/stark_convert/native_qualifications.json) appends the two exact BF16 tuples; its prior twenty-five records and schema are unchanged. There are **twenty-seven exact named-build load-only qualifications and sixty implemented-unqualified identities**. Native evidence is readiness/EOF with zero score calls, Rust 1.98.1 release/default + optimized. Worker hash `61bb81a83fa368b7c8f647469ac468048cf6868d973ed74e7d9d62633bfba96d`, native source `0cc926bcf73db3921eab03f127185626b7f4311b73c62dad79b407c642d0c9af`, Cargo lock `65444f3e0e0707c43521539e43c95365cfc6a51ec6f9e90180bc43bc8fb4603e` and Rust-toolchain file `72da9f9e503464ba7393560f6f51a69fb06be72633d52ae7725fa0ba59dfda22` remain unchanged.

| Qualification evidence | SHA-256 |
| --- | --- |
| Dense native/full-parity proof | `a5f579613f2c7bad2d7441a7f92ab7fa99049223c189a67eb981aa21574c78a2` |
| MLX native/full-parity proof | `c3f084c84571c738809121328de335ae35f032d32967cecffc40233dca758031` |
| Independent dense artifact/source/build audit | `3c25f18869224aa30bb8f01a7c3337de21ebed468476b368fa00b7930e6fbc23` |
| Independent MLX artifact/source/build audit | `fa3df5ea84b0d2b5e7869960b529c8b1944c304e0d4069fede30311502a44b88` |

## Preregistered live Lalamo benchmark

The new run contains **twenty serial fresh complete commands**: two exact BF16 tuples, two backends and five repeats. Selected source files are warmed before each command. Backend order alternates by repeat and reverses its starting order across tuples. Coordinated model/test work was paused during measurement; pre-existing user processes were not stopped. No observation was dropped or replaced by a favorable rerun.

Candidate timing includes startup/imports, complete initial/final source hashing, descriptor/signature/selection checks, metadata, planning/export/output hashing, publication, receipt and exact `--require-native-support` matching. The clean pinned Lalamo runner includes startup/imports and complete conversion with weaker source/receipt verification. Download/setup, source warming, every-output parity and native readiness checks are outside timed commands. There is no previous converter baseline for these previously unsupported profiles.

Hard gates are complete payload/spec/config/tokenizer parity, exact candidate source/artifact/native qualification, an 8 MiB transformation workspace and observed sampled candidate RSS below 1 GiB. Hypotheses require candidate median time no higher than Lalamo and median sampled RSS at most half Lalamo. Every measured result is persisted before acceptance checks; failed/interrupted output and permitted elapsed/RSS/sample-count/category aggregates are retained. Each frozen artifact, native proof, independent audit and source/tool binding is checked before and after the run.

| Exact checkpoint | Stark median, s | Lalamo median, s | Median time result | Median sampled tree RSS, MiB, Stark / Lalamo | Lower RSS |
| --- | ---: | ---: | --- | ---: | ---: |
| LLaMBA-1B 4-bit MLX, group 128 | 2.142 | 2.225 | 3.73% lower; raw median passes | 165.0 / 4606.5 | 96.42% |
| LLaMBA-1B, F32 source | 5.791 | 2.105 | 175.15% higher; raw median fails | 167.0 / 5817.6 | 97.13% |

| Exact checkpoint | Stark range, s | Lalamo range, s |
| --- | ---: | ---: |
| LLaMBA-1B 4-bit MLX | 2.133–2.262 | 2.205–2.852 |
| LLaMBA-1B, F32 source | 5.714–5.819 | 2.035–2.544 |

All twenty output parities and ten candidate source/artifact/qualification/workspace/RSS gates passed. Both memory hypotheses pass. **Dense conversion is 2.75 times as slow in this run, so its complete-command latency objective remains open.** MLX's passing median has overlapping timing ranges and does not establish a robust speed advantage. This comparison includes stronger candidate verification and does not isolate the cause of the dense slowdown.

RSS is sampled every 10 ms, can miss peaks or double-count shared pages, and is an observation rather than an enforced process-memory ceiling. The transformation-workspace bound is separate. Five warm-file repeats on one host establish these descriptive measurements, without cold-cache, other-host, broader model/family, inference-speed or execution-quality claims.

The run occurred **2026-10-12 14:23:55.539–14:26:25.466 UTC** on arm64 macOS 26.7.1, eighteen logical CPUs, 128 GiB RAM and Python 3.13.13. Both backends use the same interpreter executable hash. All four frozen candidate/reference weights are retained. Timed temporary outputs were removed only after their complete recorded hash/parity checks; no timed command failed. The earlier untimed filesystem failure remains separately recorded and is not a timed observation.

## New source and evidence bindings

| Item | SHA-256 |
| --- | --- |
| Final converter sources | `0612178540e4c5e3af5aeef556225787ca512549a10971a0108141229a3a06f5` |
| Converter lock | `5401164a38a3ffd8e6bf4f8252d6d583e798347e9ecf9931b725378e317526f2` |
| Catalog, 83 built-ins + four extensions | `17cef5bb200a2a40ad7fe359fe8f7cae27a40733fd80540cc8ed644cca4aab28` |
| Twenty-seven-record native ledger | `a08a969c68ec20447e960771af0e1cffe6dabc19f6bdb7c513c9853df19fe607` |
| Preregistered objectives | `ea68f5f96a539197d303b39eece5f606021e098e0488ac2a72a191b9a8848809` |
| Final benchmark freeze | `1954c51ff06f31f59e83a06049d1eb247e63d2048c5cf90df834936b91f59e78` |
| Twenty-command report | `f9a0aa1175e5f4b836d3686374d929ee2b9878e28afa1f8abd7cf8bb270c0691` |
| Benchmark harness | `951b32bc1ecfed3caeb1754ca98af1b1212f53bfc180db503572a818c4b01bf1` |
| Unchanged external reference runner | `e785c2272366f523f79237be847549f76ea07bdc44b20cf69b7d76b6e38ad010` |
| Successful reference preparation | `92706b2775f12319ae20a324859797e444ddb8d2adb9c292e12db190910149ed` |
| Native qualification coordinator | `bdb1b8103662007a143a6db769b7ab3041ee499397c4fed1153de911bdc9e9f6` |
| Final 108-case decoder fixture | `e169254d4c33eb7146234e8e573555e0871de0e816b82ee0d912453339051721` |
| Seventy-two-case mixer fixture | `d6a9705ae28fc2fae0c601ef5ff58320b26e6f4d6f4975f05e12014fb962b919` |
| Private verified storage consolidation audit | `4a4f1f2cdd49a231d8823f12ba218f1b9a7241e890479a28fded1f563b513855` |

Final native/full-parity qualification proofs retain pre-ledger converter binding `3fed0870712a8dd9080eff4394350eb082b15b43234ba01cb8d2629e66108e86`. The subsequent freeze and timed receipts bind the final twenty-seven-record ledger and final converter sources above. Earlier committed milestone evidence is never reassigned to this implementation. Successful retry diagnostics contained only safe numeric errno/phase/resource aggregates; benchmark runner and pinned Lalamo source remained unchanged.

## Resource preparation and evidence retention

The first untimed reference export returned a sanitized filesystem error. Its aggregate failure, source/tool bindings and partial-output observations remain private. The cause is not established from that category alone; it produced **zero timed runs**. Only that incomplete current-packet output was removed after recording the failure.

The user increased local disk capacity before the successful retry. A separate private storage audit verified two completed historical generated outputs had identical full SHA-256 and were absent from all twenty-seven raw source manifests inspected. One duplicate pair was consolidated with a hardlink, preserving both artifact paths and contents and modifying zero raw inputs. The private audit records old/new file identities and hashes; prior source-bound validation content and receipts remain unchanged. All complete current-packet candidate and frozen reference weights are retained. Timed temporary copies are removed only after their measurements, hashes and parity are recorded; failed/interrupted outputs are retained.

## Acceptance boundaries and remaining work

This packet extends scoped G5 conversion coverage. Exact native evidence is construction/readiness followed by stdin EOF, with **zero scoring/generation calls**. Execution correctness, quality and inference latency are separate, unverified dimensions. F16/F32 output is conversion-only and runtime-blocked. Generic routes and other bit/group/config/checkpoint combinations retain their individual unqualified status.

Remaining work includes classifier roots, conversion-only MoE accounting, later state-space variants, multimodal/audio/TTS/NeMo origins, other packed schemes and the remaining twenty-seven pinned rows. The dense Qwen3.5 latency objective and user-deferred SmolLM3 8-bit latency keep their existing results and status. Complete independent replacement remains open.

All real weights, configs, tokenizers, source manifests, reference outputs and native artifacts remain ignored/private. Public evidence contains public identities, permitted aggregates and hashes only. Native qualification and benchmark artifacts contain no prompts, responses, token vectors, decision probabilities, credentials or exception dumps. Setup helpers emitted diagnostic tracebacks for two missing remote file names and one inventory-shape error, without credential or model/tokenizer content. Real artifact audit and preparation/reference/qualification/benchmark tooling records sanitized categories and suppresses diagnostic dumps. No Jev/JevK5 execution, host integration, distribution or cloud/signing/release infrastructure changes occurred.

Adapted reference sources include `lalamo/model_import/model_configs/huggingface/llamba.py`, `model_import/model_specs/llamba.py`, `model_import/loaders/huggingface.py`, `modules/token_mixers/mamba.py` and decoder/layer exports. [Lalamo's MIT license](../../../tools/converter/LICENSE.lalamo), Mirai copyright and Stark's Uzu base revision `8d01581474760bd1fc7f4a547ff1d02f7dada437` remain preserved.
