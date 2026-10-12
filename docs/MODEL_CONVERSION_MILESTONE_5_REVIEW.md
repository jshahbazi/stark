# Milestone 5 technical review

Review baseline: `ade8100117bdf4d5c3f5115bcab52f65466d867e` on `codex/conversion-milestone-4`. The reviewer did not author the adapter, fixtures, fixture generators, source resolver, native qualifications, or benchmark harness.

Post-review artifact lifecycle note, 2026-10-12: the user subsequently directed removal of completed conversion model weights. The [cleanup record](MODEL_CONVERSION_ARTIFACT_CLEANUP.md) records that later action. Artifact-retention statements in this review describe its original inspection state; the reviewed hashes, results and verdict remain historical and unchanged.

## Scope

This packet implements the two LLaMBA identities in Lalamo revision `831accb8252a6b6606c378048caeb44178f1a050`: `cartesia-ai/Llamba-1B` and `cartesia-ai/Llamba-1B-4bit-mlx`. It adds reusable LLaMBA decoder and Mamba2 mixer planning. It does not implement Mamba1, later state-space families, generation quality validation, or complete Lalamo replacement.

The pinned MLX identity selects its model config from the dense LLaMBA repository. The primary MLX repository has no config sidecar. Its catalog bit width and validated tensor planes establish the packed representation; logical matrix dimensions come from the selected external config. Generic checkpoints require their own config and explicit quantization and stop metadata.

## Verdict

**Accepted for the scoped LLaMBA conversion, source validation, bounded workspace, and two exact named-build load-only qualifications. No blocking findings remain.** The dense F32-source latency hypothesis failed. The MLX median was only 3.73% lower with overlapping ranges; broader conversion-speed or statistical significance claims are not established. Complete Lalamo replacement, Mamba1 and execution quality remain open.

## Independent checks completed

- Read the pinned LLaMBA ConfigMap, `HFLlambaConfig`, decoder layouts, `load_mamba2`, convolution import, and current native Mamba2 construction paths.
- Ran all 192 Mamba2 helper tests independently. The frozen fixture has eight actual mixer exports across nine source/target precision combinations, with every source and output tensor hash checked. The private generator's recorded hash matched its actual bytes.
- Ran a focused source/metadata/generic/registry/qualification check: **687 passed, 87 skipped** because the private real metadata reference directory was not selected. After the final auxiliary-head restriction, independently reran all **261 LLaMBA decoder tests**. Both decoder layouts, tied and untied embeddings, configured biases, convolution aliases/shapes, and dense and MLX precision paths matched frozen actual full-decoder exports.
- Checked that convolution, recurrence skip weights, gate biases and normalization export as F32, with bounded shared projection conversion. Missing recurrence parameters fail instead of retaining the pinned importer's empty-template values.
- Independently checked that registered LLaMBA uses the pinned fixed EOS set, and generic LLaMBA retains its explicitly supplied metadata.
- Inspected the tied packed-head rule: lossily packed auxiliary `head.linear` is omitted only for the alternating layout with a dense embedding and declared tied semantics. Its complete representation is validated and original source signatures and content checks remain bound. Both packed embedding/head planes require exact equality. Partial planes, unknown extras and the unverified backbone lossy-head variant fail.

## Findings before the evidence freeze

Two concrete integration issues were independently reproduced and reported before timed runs:

1. The dense catalog identity could accept explicitly packed MLX inputs and produce MLX output. The catalog format boundary needs a fail-closed check; generic explicit quantization remains a separate supported route.
2. A consistently rehashed source manifest could redirect the declared external model config to an unselected primary config, and the planner consumed a changed normalization value. Exact repository/filename/role descriptors need an additional physical route check for external model config. This guard enforces selected source routing; handwritten manifests are not cryptographic attestations of remote repository contents.

Both findings are resolved. The registered dense format now rejects explicit packed MLX configuration. The external config guard checks both normalized absolute paths and resolved paths; a primary path that resolves outside root and an external alias resolving into root both fail. Independently rerun tests cover live assertions and consistently rehashed schema-1 replay.

The coordinator's final integrated run passed **3,303 tests / 60 skipped** after the real metadata oracle helper was corrected to select the declared model-config source rather than assume a primary config. Production source/metadata routing already passed complete real parity. The skipped checks need additional private real metadata reference views; these are not additional native qualifications. Earlier runs and provisional fixture generation had different bindings and are not presented as validation of the final code.

## Real artifacts and native loading

The reviewer independently hashed all candidate and reference tensors, compared names, shapes, dtypes, payload hashes and logical matrix specs, compared complete root config and tokenizer, verified full candidate artifact hashes against their private qualification receipts, replayed and content-checked each selected source manifest, and recomputed the named native executable/source/build bindings.

| Exact source | Exported tensors | Qualification proof SHA-256 | Independent audit SHA-256 |
| --- | ---: | --- | --- |
| `cartesia-ai/Llamba-1B` | 162 | `a5f579613f2c7bad2d7441a7f92ab7fa99049223c189a67eb981aa21574c78a2` | `3c25f18869224aa30bb8f01a7c3337de21ebed468476b368fa00b7930e6fbc23` |
| `cartesia-ai/Llamba-1B-4bit-mlx` | 290 | `c3f084c84571c738809121328de335ae35f032d32967cecffc40233dca758031` | `fa3df5ea84b0d2b5e7869960b529c8b1944c304e0d4069fede30311502a44b88` |

Both exact BF16 outputs passed complete parity and native readiness followed by stdin EOF with zero scoring calls. The readiness procedure was inspected independently; its executable and native source/build hashes matched the recorded evidence. This proves loading on that named build. Execution behavior and quality remain untested.

All four complete candidate/reference weight artifacts remain retained privately. Original selected sources, private manifests, qualification receipts, independent audit records, config and tokenizer remain retained. One failed untimed reference preparation is preserved separately as `filesystem_error`, with zero timed observations; specific failure causality is unverified. The storage policy was settled before freezing timed inputs.

## Frozen synthetic provenance

| Oracle | Fixture SHA-256 | Private generator SHA-256 |
| --- | --- | --- |
| Eight Mamba2 configurations, 72 precision exports | `d6a9705ae28fc2fae0c601ef5ff58320b26e6f4d6f4975f05e12014fb962b919` | `fc6b9476585d54413a2cba144851fe0d2e205cee0562263243a33e156bef769a` |
| Twelve LLaMBA configurations, 108 full-decoder precision exports | `e169254d4c33eb7146234e8e573555e0871de0e816b82ee0d912453339051721` | `7fae1907b8a6202aae7ea40b4ceae3053e007725590ec0aaf88d7c55a89b5121` |

The generators invoke actual pinned Lalamo import/export paths, with synthetic finite inputs. They do not generate model responses.

## Independent benchmark audit

The final audit recomputed summaries from every raw observation and verified the complete alternating backend/repeat order for **20 serial fresh commands**: five runs per backend for each exact tuple. It checked every persisted full-parity result, all ten exact candidate qualification records against the final ledger, every candidate's output/source/toolchain hashes and workspace/RSS gate, and every native build object. Every successful measurement was recorded before its artifact/parity/memory gates. No outlier was removed or favorably rerun.

The reviewer rehashed all four retained weight artifacts and their config/tokenizer files, independently recomputed tensor and matrix-spec manifests, replayed both selected source manifests with full content verification, and checked the final converter, native binary/source/build, clean pinned reference checkout/interpreter/lock, harness, objectives, frozen inputs, qualification proofs and independent audit records. All bindings remained unchanged. The prior twenty-five native qualification objects are JSON-identical to baseline; only the two scoped records were appended.

| Exact input | Stark median (s), range | Lalamo median (s), range | Time change | Sampled RSS reduction | Raw latency hypothesis |
| --- | --- | --- | ---: | ---: | --- |
| LLaMBA-1B MLX 4-bit, group 128 | 2.142352, 2.132834–2.262480 | 2.225409, 2.204905–2.852188 | 3.73% lower | 96.42% | Passed; ranges overlap |
| LLaMBA-1B dense F32 source | 5.791280, 5.713934–5.819341 | 2.104777, 2.034701–2.543896 | 175.15% higher | 97.13% | **Failed** |

Both outputs target BF16. Both hard correctness/qualification/workspace/source gates and the memory hypotheses passed. Each candidate used at most 8 MiB transform workspace and sampled tree RSS below 1 GiB. Ten-millisecond RSS sampling can miss peaks or double-count shared pages; it does not enforce a total process memory cap. Native readiness, setup, downloading, source warming and parity verification are excluded from elapsed command measurement. Candidate complete commands include full source validation and receipt qualification; pinned Lalamo has weaker source/receipt verification. These results apply to the selected warm-cache inputs and host.

## Final evidence bindings

| Binding | SHA-256 |
| --- | --- |
| Final converter sources, including appended ledger | `0612178540e4c5e3af5aeef556225787ca512549a10971a0108141229a3a06f5` |
| Converter lock | `5401164a38a3ffd8e6bf4f8252d6d583e798347e9ecf9931b725378e317526f2` |
| Final 27-record native ledger | `a08a969c68ec20447e960771af0e1cffe6dabc19f6bdb7c513c9853df19fe607` |
| Final private benchmark harness | `951b32bc1ecfed3caeb1754ca98af1b1212f53bfc180db503572a818c4b01bf1` |
| Preregistered settled objectives | `ea68f5f96a539197d303b39eece5f606021e098e0488ac2a72a191b9a8848809` |
| Frozen serial-run inputs | `1954c51ff06f31f59e83a06049d1eb247e63d2048c5cf90df834936b91f59e78` |
| Complete 20-command private result report | `f9a0aa1175e5f4b836d3686374d929ee2b9878e28afa1f8abd7cf8bb270c0691` |

The audit records preserve the pre-ledger qualification toolchain identity separately from the final benchmark package identity. Qualification proves the exact outputs on the named unchanged native build; appending the records does not retroactively rewrite those initial proofs.

Setup inventory/source-probe helpers emitted static tracebacks containing no model payload, tokenizer content, prompts or credentials. That setup logging limitation is acknowledged; subsequent qualification, benchmark and independent artifact audits suppress exception dumps and emit fixed aggregate error categories. Intermediate unpromoted fixture revisions were not fully archived; the final 108-export oracle, source recipe and generator are preserved with their own bindings. Neither limitation is represented as a passing historical evidence check.
