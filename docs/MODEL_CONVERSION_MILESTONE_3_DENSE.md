# Milestone 3 continuation: MLX Llama, SmolLM3 and Gemma text

Validation date: 2026-10-11. This packet extends the independently reviewed [Qwen3 MLX packet](MODEL_CONVERSION_MILESTONE_3.md) on `codex/conversion-milestone-3`, starting from `e40dc0ce`. Prior source-bound results are preserved. See the [independent review](MODEL_CONVERSION_MILESTONE_3_DENSE_REVIEW.md), [plan](MODEL_CONVERSION_PLAN.md), and [converter usage](../tools/converter/README.md).

## Scope and behavior

Nine additional pinned built-ins now have independent MLX conversion: five Llama profiles, two SmolLM3 profiles and two Gemma3 text profiles. Coverage is **54 of 110 built-ins plus four extensions**, or 58 implemented identities, including 21 MLX profiles. Six new exact BF16 tuples bring the qualification ledger to 19; the prior 13 records are preserved unchanged. Fifty-six built-ins remain unimplemented and 39 implemented identities lack exact native qualification. Full Phase 4/G4 and complete Lalamo replacement remain open.

| Family | Added registered profiles |
| --- | --- |
| Llama | `mlx-community/Llama-3.1-8B-Instruct-4bit`, `mlx-community/Llama-3.2-{1B,3B}-Instruct-{4bit,8bit}` |
| SmolLM3 | `mlx-community/SmolLM3-3B-{4bit,8bit}` |
| Gemma3 text | `mlx-community/gemma-3-1b-it-{4bit,8bit}` |

The [Llama](../tools/converter/stark_convert/adapters/dense.py), [SmolLM3](../tools/converter/stark_convert/adapters/granite.py), and [Gemma](../tools/converter/stark_convert/adapters/gemma.py) planners reuse the unchanged, reviewed [compressed helper](../tools/converter/stark_convert/compressed.py). Existing affine checkpoints are repacked without new quantization. Little-endian I32/U32 weights retain their bytes as U8. Projection coefficient planes cast to the requested dtype, fuse in Q/K/V or up/gate order, transpose to group/output orientation and pad the complete output count to four. Input/tied embedding planes retain their physical input/output orientation. Accepted bits are 4/8, groups are 16/32/64/128, and logical input widths must be divisible by the group size and packing factor.

Each matrix consumes its complete weight/scale/affine-bias triplet. Every accepted duplicate tied-head plane must match. Dense and compressed matrices can coexist separately; fused groups must share representation and coefficient dtype. Full source consumption, malformed-layout rejection, unchanged-source checks, bounded workspace, private staging and atomic no-replace publication remain mandatory. The CLI binds every compressed identity to its bit width and rejects packed sources under dense profiles. No Lalamo, JAX, PyTorch or Transformers import/invocation was added to converter runtime.

Family semantics remain explicit:

- Llama retains its accepted Llama3 RoPE, tied/untied embeddings and bias restrictions. Qwen2/Mistral quantization stays unsupported despite sharing the dense planner.
- SmolLM3 retains the pinned per-layer RoPE-mask interpretation, its omitted tied-embedding default, accepted full-attention annotations, training metadata and optional projection biases. Granite quantization remains unsupported because its scalar transforms need a separate compressed contract.
- Gemma3 text retains runtime square-root embedding scaling, normalization scale offsets/upcast mode, attention scaling, local/global windows and accepted RoPE variants. Packed weights are not rescaled. Gemma2 quantization, multimodal Gemma 4B/27B and newer architectures stay unsupported.

The companion package is 0.5.0. Dependency versions, shared MLX writer, Rust/Metal formats, native worker, scoring engine and inherited infrastructure are unchanged. BF16/F16/F32 conversion remains available; F16/F32 are runtime-blocked by converter policy. Named native qualification is BF16 load-only.

## Exact real checkpoint qualification

All six selected tuples use affine group size 64 and BF16 output. Complete external comparisons include every tensor name, shape, dtype, payload hash, spec/logical metadata field, root/nested config and normalized tokenizer. Header serialization differences do not change payload/spec parity.

| Registered identity suffix under `mlx-community/` | Immutable primary revision | Output tensors | Safetensors bytes | Exact parity / named-build load-only |
| --- | --- | ---: | ---: | --- |
| `Llama-3.2-1B-Instruct-4bit` | `08231374eeacb049a0eade7922910865b8fce912` | 228 | 695,420,616 | Passed / passed |
| `Llama-3.2-1B-Instruct-8bit` | `d48cdf0a4ea22d893b7c63a99d6a693e24822795` | 228 | 1,313,294,064 | Passed / passed |
| `SmolLM3-3B-4bit` | `d3a7e0594d6642dbcfb7d149bed8b0bdf49f95ce` | 508 | 1,730,352,904 | Passed / passed |
| `SmolLM3-3B-8bit` | `316f091e34982bc6eaf7f6cc1db82bdb77ac2103` | 508 | 3,267,827,584 | Passed / passed |
| `gemma-3-1b-it-4bit` | `2d44e83dc9e80843d22fb941d3d699a0b1351aa6` | 472 | 562,979,272 | Passed / passed |
| `gemma-3-1b-it-8bit` | `7b963136f21d05ca8b367c93a9c47a7944ad281a` | 472 | 1,062,855,208 | Passed / passed |

Every artifact constructed successfully with the unchanged optimized Rust 1.98.1 worker and exited on stdin EOF after readiness, with **zero native scoring calls**. This qualifies loading/complete parameter consumption by that named build. No scoring, generation, quality, inference speed or Jev/JevK5 experiment was performed. Other Llama sizes, groups, source revisions, configs and dtypes remain unqualified. `--require-native-support` requires the exact stored identity/source/artifact/build match; catalog probes do not grant qualification.

Metadata uses registered source rules: primary MLX tokenizer files, official Llama/Gemma generation files and original SmolLM3 chat-template/generation files. Every logical source is pinned before conversion. Model weights, operational metadata and replay manifests remain private.

## Verified public transport for Llama metadata

The initial Llama preparations failed because the official generation file was gated. An authorized Chrome inspection showed the signed-in official page still required its access form. No license/contact form, account, credential or official cache was changed.

An explicit public search found an exact copy at [project-free-llama revision `66da35cf6e4fcbdf48ecdba53c252715932d9a8b`](https://huggingface.co/project-free-llama/Llama-3.2-1B-Instruct/blob/66da35cf6e4fcbdf48ecdba53c252715932d9a8b/generation_config.json). The [official public repository metadata](https://huggingface.co/api/models/meta-llama/Llama-3.2-1B-Instruct/revision/9213176726f574b556790deb65791e0c5aa438b6?blobs=true) binds `generation_config.json` to a 189-byte Git blob at canonical revision `9213176726f574b556790deb65791e0c5aa438b6`. Computing Git's blob hash over the actual public mirror bytes gives the same `75ae08310d6d23df373ee2644b497192b3cce6d8`; content SHA256 is `88effbb63300dbbc7390143fbbdd9d9fa50587b37e8bfd16c8c90d4970a74a36`. Three additional public mirrors independently matched.

The private preparation wrapper verifies the canonical and mirror pins/advertised blob, recomputes Git-blob and SHA256 values, and returns the normal public-mirror cache path for this exact file. Standard source manifests retain the canonical FileSpec identity and actual transport path. Separate owner-only receipts bind canonical and transport identities/revisions, content hashes, script/evidence hashes and private manifest hash. The canonical cache is untouched. The external reference runner consumes those same manifest paths and rechecks external revision/content hashes.

This is explicit preparation of byte-verified local source content for offline replay. The runtime resolver and catalog gained no automatic mirror route, metadata override or framework fallback. Normal remote resolution can still require official access. Unverified or merely similar mirror metadata receives no canonical qualification. Initial access failures and the source recovery evidence remain separately preserved.

## Synthetic acceptance

The new frozen references use the actual pinned `load_huggingface_decoder` and export paths:

- 144 Llama/SmolLM3 complete decoder exports: two families, both bits, four groups and all nine source/target float pairs. Every output tensor descriptor/hash, complete decoder config and spec is compared, using a 96-byte writer workspace.
- Gemma text coverage includes 72 bit/group/dtype combinations, complete frozen exports, tied-plane duplicates, cross-shard signed words, optional biases, separate dense matrices, global output padding and local/global attention/RoPE variants.
- Twenty-one complete synthetic CLI artifacts matched independently generated pinned Lalamo tensors/specs/config/tokenizer: eight Llama/SmolLM3 cases and thirteen Gemma cases. They cover Llama3.1/3.2 RoPE factors, omitted Smol tied defaults and biases, all float pairs, linear/YARN/window annotations, dense embeddings/readouts, and padding. Source/tooling stayed unchanged and CLI workspace stayed within 4,096 bytes.
- End-to-end catalog/bit selection and offline external replay cover all four MLX adapters and both backends. Existing shared MLX frozen integer/affine/cast regressions, malformed-plane/config/fusion checks, private failure cleanup, full source checking and exact qualification selection remain active.

The first supplemental CLI harness attempts used 96 bytes below the CLI's 4,096-byte minimum and stopped before publication. Those failed aggregates are retained. Corrected runs passed without production changes. The public fixtures contain synthetic hashes/config/specs, never real checkpoint payloads or native execution content.

## Validation and complete-command measurements

Final integrated validation passed **1,625 tests with 39 skipped**, using 19 supplied private metadata references. The model-free portion contains 1,606 cases with 58 unsupplied-reference skips. Ruff lint/format passed for 41 files, the lock check and repository policy passed, 17 CI-policy tests passed, and actionlint passed with the existing disabled-job expression exception. The locked optimized Rust 1.98.1 build succeeded with the existing debug-strip/libLLVM warning and unchanged native worker hash. Hosted execution and a new advisory scan are not claimed; dependency versions are unchanged.

The predeclared hard checks were full artifact parity, unchanged source/tooling, transformation workspace at most 8 MiB and sampled candidate tree RSS below 1 GiB. Performance hypotheses were candidate median complete time no slower than pinned Lalamo and candidate median tree RSS at most half the reference. All six tuples passed hard checks and memory hypotheses; two passed the raw latency hypothesis.

Three repetitions per backend/tuple ran in fresh direct Python processes, serially with other model conversions paused, warmed selected files and alternating backend order. Setup/downloads, warming and artifact comparisons were outside timing. Complete commands include startup, verification, planning/metadata, writing, output hashes, final checks and receipts. The external reference rehashes selected external metadata and performs weaker primary-source/receipt verification. Ratios compare complete commands with those different contracts.

Host: Apple silicon arm64, 18 CPUs, 128 GiB RAM, Darwin 25.6.0, managed Python 3.13.13; NumPy 2.5.3, ml-dtypes 0.6.0, tokenizers 0.23.3, huggingface-hub 2.2.0 and psutil 7.2.2. RSS is sampled every 10 ms, can miss peaks and can count shared pages more than once. Workspace excludes catalog/header/tokenizer/hash/interpreter costs and imposes no whole-process ceiling.

| Group-64 BF16 tuple | Stark median [range], seconds | Lalamo median [range], seconds | Median tree RSS, Stark / Lalamo MiB | Raw latency hypothesis |
| --- | --- | --- | --- | --- |
| Llama3.2-1B 4-bit | 1.845 [1.667–1.872] | 2.292 [2.137–2.349] | 279.0 / 4,720.7 | Passed, 1.24× ratio |
| Llama3.2-1B 8-bit | 2.233 [2.195–2.609] | 2.137 [1.985–2.392] | 167.7 / 4,985.0 | Failed; ranges overlap |
| SmolLM3-3B 4-bit | 3.979 [3.804–4.188] | 2.809 [2.800–3.067] | 273.8 / 5,981.6 | Failed |
| SmolLM3-3B 8-bit | 6.019 [6.006–6.778] | 3.544 [3.070–3.572] | 277.8 / 7,273.4 | Failed |
| Gemma3-1B 4-bit | 1.991 [1.922–2.004] | 2.444 [2.263–2.446] | 472.6 / 5,102.7 | Passed, 1.23× ratio |
| Gemma3-1B 8-bit | 2.655 [2.647–2.691] | 2.171 [2.171–2.194] | 467.9 / 6,256.9 | Failed |

Median sampled RSS reductions are 90.7–96.6%. Every one of the 36 timed outputs passed exact external comparison outside measurement. All 18 final-source candidate outputs matched exact stored native qualifications. These warm-cache descriptive conversion results are bounded to the six tuples, source state and host. The four slower median outcomes remain open latency work. No inference-speed, quality or general model-size claim follows from this benchmark. Prior Qwen3/dense measurements remain tied to their earlier sources.

## Evidence binding

Final measurements bind converter source `779ec68a7a1c55edff0438ebe60afdca7194e3e6046679491cac06bcc6656aa1` and lock `06d5e2a227ce9480adaa69501a6d7552c68c95e2d37dc0bd830d650ac98fae55`. The initial synthetic and Smol/Gemma qualifications retain source `d07bf3bbadab099c090a30630dab251c40065bdc5903382cde1696326689d9e5`. The four-record ledger update produced source `cfde8302d7532be773c19953f9ee45ab1cada07d8c20b5f7961f7f2aaa0485f9`, binding the initial four-tuple timing batch and Llama qualification. Adding the two Llama records produced the final source. Earlier outcomes were not silently rebound.

| Input/evidence | SHA256 or revision |
| --- | --- |
| Pinned Lalamo source | `831accb8252a6b6606c378048caeb44178f1a050` |
| Stark Uzu base | `8d01581474760bd1fc7f4a547ff1d02f7dada437` |
| Catalog | `d855e4c02ada48cca0062e7c273d3d7372b67b4ba70379ef3cf275acd9b78a3a` |
| Nineteen exact native qualification records | `26074c55ae93a5ef0a7aba15a676668ba4f0d4fa76466e23c2076b3515b8beac` |
| Final six-tuple benchmark | `576a22a4e1b5a90054ada0056bfab425f4b71d47de3657163c290a819770674d` |
| Six initial qualification summary / final tooling | `9c863fa6f27cbacfce1ba47f8710897e2abdef7b8484e4f39c69c10ab06a6067` |
| Preserved initial four-tuple timing batch | `a226d597954df6fb2a39177ca074945844fa99c26a6c4b6df77db10aae70edd9` |
| Final benchmark coordinator | `1882623bfe69612feea28abbb7183e748bb840021cd201e83df0cd1c7bdbbae4` |
| External reference runner with actual manifest metadata paths | `e785c2272366f523f79237be847549f76ea07bdc44b20cf69b7d76b6e38ad010` |
| Llama/Smol frozen decoder exports | `f891c31a24442bc2db061bd60150c723613b22cc7bc79bfd1a7e5021b9d5697c` |
| Gemma frozen exports | `d350a885d8e52da04d28cc645e8bf493a9bd5dca42ae191c4bca4c13f08695a4` |
| Eight complete Llama/Smol synthetic comparisons | `c23b6e683c8b3b8aba746a88aaf5d1b66f09faa1b373a1b135a6e672a762feb7` |
| Thirteen complete Gemma synthetic comparisons | `a98d7c9821a6e429f54c7009ecd933e7129eec2b32986801e18d9a06c5b07e19` |
| Exact public mirror evidence | `3c9fb4f70658a7d8a2d9377b0b6348277825fc1d752494cd51f7a03ea27e047c` |
| Llama metadata recovery preparation | `f4a2d0d3dde276fb22796c5ad4795cf3f4fa9df15add93fb991440d41d215bfd` |
| Private recovery script | `96713ed2c0818907370ad472e9d60b39242a337cc50c16dee767ce772b5979c2` |
| Unchanged native worker | `61bb81a83fa368b7c8f647469ac468048cf6868d973ed74e7d9d62633bfba96d` |
| Native source manifest | `0cc926bcf73db3921eab03f127185626b7f4311b73c62dad79b407c642d0c9af` |
| Cargo lock | `65444f3e0e0707c43521539e43c95365cfc6a51ec6f9e90180bc43bc8fb4603e` |
| Rust toolchain file | `72da9f9e503464ba7393560f6f51a69fb06be72633d52ae7725fa0ba59dfda22` |

Remaining work includes other compressed schemes, specialized/multimodal architectures, unqualified sizes/groups/dtypes, strict process-memory enforcement, execution/quality and broad complete-command latency parity. No model distribution or inherited cloud/signing/release infrastructure was enabled.
