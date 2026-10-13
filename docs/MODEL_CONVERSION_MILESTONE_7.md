# Milestone 7: resume Lalamo conversion parity

Version 0.10.0 resumes architecture coverage after the tool-hardening and README packets, from `88d5ea24` on `codex/converter-parity`. The pinned comparison remains Lalamo `831accb8252a6b6606c378048caeb44178f1a050`. The installed converter has no Lalamo, JAX or PyTorch dependency or runtime fallback.

## Coverage and implementation

Seventeen additional built-ins bring implemented coverage to **102/110 plus four separate Stark extensions**. There are 52 full-precision language profiles, including two MoE profiles, 48 affine MLX profiles and two ModernBERT classifiers. Architecture adapters remain reusable through `hf-generic`; registered identities separately bind metadata and format rules.

| Added packet | Built-ins | Implementation scope |
| --- | ---: | --- |
| LFM2.5 | 7 | Existing strict LFM2 attention/short-convolution layout; three full-precision and four MLX 4/8-bit identities; pinned sampling and Thinking metadata. |
| Gemma3 wrappers | 6 | Full-precision and MLX 4/8-bit text-decoder exports for 4B/27B wrappers; declared vision/projector tensors validated before omission and retained in source verification. |
| Later dense Qwen | 2 | Qwen3.6/3.8 27B config, rotary alias and gate-field validation; Qwen3.8 three-level reasoning-effort metadata. |
| Qwen MoE | 2 | Distinct `qwen3_5_moe` adapter for Qwen3.5/3.6 35B-A3B full-precision batched routed/shared experts, router and shared gate. |

Gemma wrapper conversion matches Lalamo's text-only output. It does not export a vision encoder or qualify image execution. All original tensor files remain signature/content-bound, including omitted tensors.

The two later dense Qwen 27B sources declare a `swish` output gate, while the pinned Lalamo exporter emits sigmoid-gate semantics. Conversion reproduces the pinned export and reports `native_attention_gate_semantics` with native loading blocked. MoE reports `native_moe_construction`. Neither packet has real-weight native or execution qualification. Individual-expert and packed MoE layouts remain unsupported.

## Source-index compatibility

All four inspected Gemma MLX 4B/27B 4/8-bit repositories retain an older published shard index. Some named shards are unavailable; available shards can also have changed assignments and additional affine parameter planes. Ordinary strict index loading initially rejected the real 4B 8-bit source.

Compatibility is declared only for these four catalog identities. It requires the exact raw index SHA-256, complete current flat shard filenames, full actual tensor-name inventory hash, and every original indexed tensor name. Duplicate names, invalid paths/JSON, inventory changes, unexpected files and source mutation fail. The original index stays in the hashed source manifest. Unknown or changed indexes with available shards retain ordinary assignment checks; other profiles and anonymous generic sources remain strict.

The shared selector is used by initial resolution, offline replay, source selection rechecks and checkpoint loading. A post-hash selection recheck closes mutation between initial selection and initial source hashing. The public catalog retains only hashes and filenames; raw model metadata remains private.

## Independent validation

Parallel workers owned LFM2.5 fixtures, Gemma wrapper/index handling, and later Qwen/MoE transforms. The coordinator owned registry, catalog, metadata, generic dispatch, packaging/version, real validation and documentation. A separate reviewer authored none of the production code, fixtures, benchmark harness or qualification proofs. The [technical review](MODEL_CONVERSION_MILESTONE_7_REVIEW.md) records its scoped disposition and independent audit.

The new frozen oracles use actual clean pinned Lalamo config/importer and full decoder export. They contain 36 LFM2.5 exports preserving the sixteen-layer topology at reduced dimensions, seven Gemma wrapper exports, and 36 later-Qwen/MoE exports. Public full-size config/header planning is recorded separately and does not qualify real payloads. Adversarial coverage includes bounds, alias ambiguity, tied heads, packed planes, expert ordering, missing/extra tensors, dtype/config contradictions, shard boundaries, changed source signatures/content, malformed indexes and offline replay.

The final integrated suite passed **4,000 tests / 104 skipped** in 63.51 seconds. The skipped tests require opt-in private language-metadata reference views. The five live comparisons below provide separate full metadata/tokenizer parity for their exact sources. Installed sdist-to-wheel validation passed outside the checkout, including local catalog/generic imports, byte-compilation provenance, benchmark parity and incomplete-folder errors. Ruff check/format, locked resolution, repository policy and seventeen policy unit tests passed. No hosted CI result is implied by these local checks.

## Five exact real-weight qualifications

Every selected BF16 output matches all reference tensor names, shapes, dtypes, payload hashes, logical specs, complete root config and normalized tokenizer. Whole-file serialization can differ between candidate and reference when logical metadata and every payload match.

| Catalog identity | Primary immutable revision | Source-manifest SHA-256 |
| --- | --- | --- |
| `LiquidAI/LFM2.5-350M` | `9e6c6ccf47cd318696e137d381a7ded8fe4df09f` | `2fc7d3dc9589c480f8645d2fcb9a4116c53a11ac0c3de7f3f967f99cc249f775` |
| `LiquidAI/LFM2.5-1.2B-Instruct-MLX-4bit` | `7ccafdb04c36936f4f1c4685198c6c9a40275932` | `cfbb1cb35cfc9485d678407e0e0ff61ebc90a6f3ae0fe559dc29f4c09c96507c` |
| `mlx-community/LFM2.5-1.2B-Thinking-8bit` | `35175c7b717f4ee0fef5690940b2c1681039fe8d` | `f586b6cc415cd7a4b875f8e22423446006085513129222f66679f35ca79811a7` |
| `google/gemma-3-4b-it` | `093f9f388b31de276ce2de164bdc2081324b9767` | `de0c4534765763cfa86360fea39b5dec247de5d38714221bb36840c971fcf663` |
| `mlx-community/gemma-3-4b-it-8bit` | `61cfff4de4bc5b975437be840d504f07f0424df2` | `3f249d20b11e74695d9e81442255fc378a9cd9ed0aa842f6c26314d180bced02` |

All five outputs passed a newly built Rust 1.98.1 locked release scoring worker with default/optimized features: construction/readiness followed by stdin EOF, **zero scoring calls**. The ledger now has **32 exact load-only records**, preserving the original 27 objects. Seventy-four implemented identities remain unqualified. Both exact Mirai classifier access probes remain blocked; other public classifier fixtures retain their separate prior proof scope.

The first qualification attempt preserved four passing proofs and one failed Gemma 8-bit source-index attempt. After the compatibility correction, all five were regenerated under the final production source. Initial attempt records remain separate from final proofs; their weights were retained for audit before cleanup. No timed observations were collected before that correction.

| Exact tuple | Candidate weights SHA-256 | Final qualification proof SHA-256 |
| --- | --- | --- |
| LFM2.5-350M | `57d9f8f983fa4641fdb84c2a2c968bf1fbfdc34276bc21e7b00d70a2b1a27324` | `ccd2b4c7981351c8af12d52daa675accb868c575fb96173a6dde4acbf080f8b2` |
| LFM2.5 Instruct MLX 4-bit | `e8142a3b4fe326827e7475a6e728a8f8ecfe9e1fc294b86ed1573aaad60dc189` | `cf3714d172ea6f9f8cd30440ff6188da386a9ac448068df575c830a54d1cef15` |
| LFM2.5 Thinking MLX 8-bit | `ad13907b214794ae5799980eab6e5324648de46b5da26053e4ec6199fd865435` | `2acb1319b5a841c8978a890c7c452c6c70a5f3815baecd4a84276c0089cb7c9e` |
| Gemma3 4B full-precision | `b3a96a3a602824871f4d9f3ab1c5d447f512eba4c10e42f9db3d96d79356ad1b` | `b838e1f0d1764670830a3d1d50a4aac1c91a425502cbb7d35a4c6e95980e9aa7` |
| Gemma3 4B MLX 8-bit | `1db3f9f41a179da21f4080da245c29d1a117d5e12c9db87cf62173a87af24b33` | `9e4de8c3de0fbddef76569f5d7ad394b694099cbfd4be39467558a623cdff67f` |

## Benchmark method correction

The original fifty-command run completed with all output parity comparisons passing. Its runner nevertheless failed all twenty-five fresh reference outputs at an extra whole-file byte-equality gate. An independent audit confirmed identical tensor payload hashes, logical specs, parsed complete safetensors headers including offsets, config bytes and tokenizer bytes. Only serialized header bytes differed. This extra gate was absent from the preregistered requirements, which bind candidate whole-file bytes and complete reference parity.

The original failed report, all fifty observations, its freeze and harness remain preserved. Frozen weight outputs were retained for audit before cleanup. A separately named corrected harness removes only that extra reference equality branch, retains every candidate/source/native/memory/parity gate, and records fresh reference byte hashes. Its five qualifications, freeze and fifty commands are separate records; no observations were removed or replaced. The same five models, repeat count and preregistered objectives remain unchanged.

| Preserved original binding | SHA-256 |
| --- | --- |
| Original failed fifty-command report | `18c1b81bc612a3d75364b85eb73f86360e96fa56b5195b5c96f9bf84a5a43399` |
| Independent serialization audit | `f524a021fa52bdc162dcfd9c97aeadf2113c7a2d6b23d8783e5adac17041ca90` |
| Original benchmark freeze | `1460907c6991314a0e1be2f78bcb3e5cace35ab64186fff96002da24c51e03c4` |
| Original benchmark harness | `294884e7b3c8d2e4a630a83c9ac94fbd4fcb2d1e2c57283bf19fc7e417911dd9` |

## Corrected Lalamo benchmark

The separate corrected run completed **fifty serial fresh complete commands**, five per backend per tuple. Source files were warmed before each command, outside timing; backend order alternated and its starting order rotated across tuples. Other task model work was paused during measurement. Every observation was saved before acceptance checks, without outlier removal or replacement.

| Exact BF16 output | Stark median, s | Lalamo median, s | Time result | Median sampled tree RSS, MiB, Stark / Lalamo | Lower RSS |
| --- | ---: | ---: | --- | ---: | ---: |
| LFM2.5-350M | 0.944 | 1.137 | 17.02% lower | 113.0 / 1,163.0 | 90.28% |
| LFM2.5 Instruct MLX 4-bit | 0.953 | 2.061 | 53.77% lower | 114.8 / 3,054.0 | 96.24% |
| LFM2.5 Thinking MLX 8-bit | 1.533 | 1.809 | 15.24% lower | 113.2 / 3,580.3 | 96.84% |
| Gemma3 4B full-precision | 6.909 | 2.993 | 130.85% higher | 465.0 / 9,437.7 | 95.07% |
| Gemma3 4B MLX 8-bit | 5.975 | 3.198 | 86.83% higher | 467.0 / 12,946.4 | 96.39% |

| Exact tuple | Stark range, s | Lalamo range, s |
| --- | ---: | ---: |
| LFM2.5-350M | 0.939–0.984 | 1.087–1.218 |
| LFM2.5 Instruct MLX 4-bit | 0.931–0.983 | 2.017–2.103 |
| LFM2.5 Thinking MLX 8-bit | 1.521–1.548 | 1.803–1.850 |
| Gemma3 4B full-precision | 6.813–7.010 | 2.970–3.122 |
| Gemma3 4B MLX 8-bit | 5.966–5.985 | 3.172–3.417 |

All fifty full parity, source/tool/build identity, exact qualification, workspace and sampled-memory checks passed. Each candidate stayed within an 8 MiB transformation workspace and below the preregistered 1 GiB sampled tree-RSS threshold. All five median memory hypotheses pass. The three LFM latency hypotheses pass; **both Gemma latency hypotheses fail**, so overall performance hypotheses remain false. Conversion parity and load-only acceptance do not close latency parity. No real-weight Qwen3.6/3.8/MoE benchmark or native qualification is claimed.

Candidate timing includes Python startup/imports, complete source verification, metadata, planning, export/output hashing, atomic publication, receipt and exact native qualification lookup. Reference timing uses actual pinned model import/save with weaker source/receipt checks. Setup, downloads, warming, artifact comparisons and native loading are excluded. RSS is sampled every 10 ms; it can miss peaks or double-count shared pages and is not an enforced memory ceiling.

The run occurred **2026-10-13 03:21:16.835–03:28:48.369 UTC** on arm64 macOS 26.7.1, eighteen logical CPUs and 128 GiB RAM. Five observations on one host establish the recorded medians/ranges, without statistical significance, cold-cache, family-wide, inference or quality claims. Every original and corrected observation remains retained separately. Model-dependent post-run audit precedes cleanup.

## Final source bindings

| Binding | SHA-256 |
| --- | --- |
| Final converter sources, including the 32-record ledger | `950d47adb66140a4edcb8f1b7cdab82653f99657d6ee9329eb1939ecb575600d` |
| Converter lock | `28d85c95bbd9175b78358789e24cf9a1a38417f45aa49b6bf483887e796a717f` |
| Named scoring worker | `61bb81a83fa368b7c8f647469ac468048cf6868d973ed74e7d9d62633bfba96d` |
| Native source manifest | `2b855098e046ebf4ebfaa7206403a07dd164b31218614882f8b6babfccd8846b` |
| Cargo lock | `49a27c8d8b2eebc3bdf47d6957f5b2c8dbbb822ddd66090589a33bcbf1f35ec1` |
| Rust toolchain receipt | `72da9f9e503464ba7393560f6f51a69fb06be72633d52ae7725fa0ba59dfda22` |
| Benchmark objectives | `8ca8ce78d4667c6f9a0ca489f441f0c4e5a4f4e27b8a9ca99bb320e0e442d758` |
| Corrected benchmark freeze | `5fc2c1e9111dc3b41214cc5f4d9c592b8715c4b5d2f32fe36977be4225e92142` |
| Corrected benchmark harness | `bbdd9366872db515e8ff9413206b05a671c847f6bcf338d4c9829f6bdc3466ea` |
| Corrected fifty-command benchmark report | `09fe0100aa96fc7a47dbb8e258d52987d4845cce17c454bbbcff6e0e0c9de994` |
| Independent final source/artifact/benchmark audit | `31ca49e0c17f63e0d02fa71aa9bd5251a92a96f06da90b9fdb6e6a3c0d68b3e5` |
| Reference runner | `e785c2272366f523f79237be847549f76ea07bdc44b20cf69b7d76b6e38ad010` |

## Completed model cleanup

After model-dependent audits, the independently reviewed cleanup unlinked **67 exact paths**: seven selected Hugging Face source-weight aliases, seven backing blobs and fifty-three generated weight files. It verified sixty unique regular files against retained hashes and found no shared/unselected cache references, hardlinks or active handles. All selected paths are absent. Regular removed bytes total **168,108,206,490**; observed free space increased **168,131,035,136 bytes**, about **156.58 GiB**. Storage readings can also reflect other host activity.

Metadata, source manifests, qualification attempts/proofs, both complete benchmark reports and all one hundred observations, scripts, hashes and native artifacts remain retained. Artifact-retention statements above describe their audit state before cleanup. Offline source replay requires restoring the removed pinned weights. The [review cleanup bindings](MODEL_CONVERSION_MILESTONE_7_REVIEW.md#model-cleanup) identify the exact inventory, verification, approval and completed receipt.

Full registry parity remains open. The [remaining eight-entry checklist](MODEL_CONVERSION_PLAN.md#remaining-eight-built-ins) starts with Gemma4 PLE, then Muse text export, Bonsai 1-bit, GPT-OSS MXFP4 and FishAudio TTS. Older validation, latency failures and qualification records retain their original source bindings. Upstream MIT copyright/license and Uzu provenance remain preserved; new adapted contracts come from the pinned LFM2, Gemma3, Qwen config/spec/loader sources and the inspected public source revisions above.
