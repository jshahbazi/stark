# Milestone 7 technical review

Review baseline: `88d5ea24cc8758000b99ee793389d2883da61d0c` on `codex/converter-parity`. The independent reviewer authored none of the production implementation, fixtures, benchmark harness or qualification proofs. The reference is Lalamo `831accb8252a6b6606c378048caeb44178f1a050`.

## Scope and disposition

The packet adds seventeen built-ins: seven LFM2.5 profiles, six Gemma3 wrapper text profiles, two later dense Qwen profiles and two full-precision Qwen MoE profiles. Implemented coverage is **102/110 built-ins plus four separate extensions**. Eight built-ins still need adapters. Thirty-two exact BF16 tuples have load-only ledger records; seventy-four implemented identities remain unqualified.

**Accepted for the documented conversion scope, five exact real-weight parity/load-only qualifications, bounded transformation workspace and recorded benchmark outcomes. No blocking production or evidence findings remain.** This does not close complete Lalamo replacement or latency parity. Both measured Gemma3 4B latency hypotheses fail. The new dense Qwen sources retain an attention-gate semantic block, and Qwen MoE retains a native construction block. Gemma wrappers export the text decoder; image execution remains unqualified.

## Independent source and artifact audit

Read-only code review covered catalog/profile dispatch, generation metadata, LFM2.5 reuse, Gemma wrapper prefixes/modal validation/tied-head equality, later Qwen/MoE ordering and bounded views, strict source consumption/index selection/replay, generic defaults and native status boundaries. The reviewer independently ran the focused LFM2.5/Gemma/Qwen/registry suite: **265 passed**. The coordinator owns the separate full-suite result below.

The reviewer independently rehashed thirty selected source files totaling **16,971,212,057 bytes**, the frozen candidate/reference artifacts, and converter, lock, harness, reference and native-build inputs. A separate standard-library safetensors parser checked every exported tensor's name, shape, dtype, payload hash, logical specification, offsets and complete payload coverage. Complete config and tokenizer parity also passed.

| Exact source | Exported tensors | Full parity | Load-only proof |
| --- | ---: | --- | --- |
| `LiquidAI/LFM2.5-350M` | 120 | Passed | Passed |
| `LiquidAI/LFM2.5-1.2B-Instruct-MLX-4bit` | 250 | Passed | Passed |
| `mlx-community/LFM2.5-1.2B-Thinking-8bit` | 250 | Passed | Passed |
| `google/gemma-3-4b-it` | 342 | Passed | Passed |
| `mlx-community/gemma-3-4b-it-8bit` | 616 | Passed | Passed |

The named Rust 1.98.1 locked release worker passed construction/readiness and stdin EOF for each artifact, with zero scoring calls. The review checked these source/artifact/build bindings and independently verified all original twenty-seven ledger objects were preserved. Native loading is separate from scoring, generation, quality and image execution.

## Index and benchmark findings

The real Gemma 8-bit source exposed a stale index whose named files existed but whose tensor assignments and inventory no longer matched those files. Compatibility is now limited to four registered Gemma identities and binds the exact original index hash, current flat filenames, complete tensor-name inventory and original-name subset. Ordinary index validation remains strict for other sources. Source selection is checked again after hashing, and the original index remains content-bound in replay. This compatibility does not waive malformed-index, duplicate-name, path or mutation checks.

The original fifty-command benchmark preserved a failed outcome: all twenty-five fresh reference outputs failed an extra whole-file byte-equality check, despite complete tensor/spec/config/tokenizer parity. The independent serialization audit verified identical parsed complete headers and payloads; only serialized header bytes varied. That extra reference gate was absent from the preregistered requirements.

A separately named corrected harness removes only that extra gate, records every fresh reference byte hash, and retains all candidate/source/native/parity/workspace/memory checks. Its five qualifications, freeze and fifty fresh observations remain separate from the original failed run. The reviewer recomputed all medians and resource outcomes from those observations and checked candidate receipts and frozen inputs. No observations were dropped or replaced.

All fifty corrected commands pass hard acceptance. LFM2.5 median command times are 15–54% lower than Lalamo; Gemma3 4B times are 87–131% higher. Sampled tree RSS is 90–97% lower across all five exact tuples. Candidate workspace stays within 8 MiB and observed sampled tree RSS below 1 GiB. The full [milestone evidence](MODEL_CONVERSION_MILESTONE_7.md#corrected-lalamo-benchmark) records ranges, timing scope and sampling limits. These one-host warm-file measurements do not establish statistical significance, cold-cache, inference or family-wide performance.

## Integrated verification and evidence

The coordinator's final model-free suite passed **4,000 tests / 104 skipped**. Skips require opt-in private metadata reference views; exact live metadata parity above remains separate evidence. Ruff check/format, locked resolution, repository policy, seventeen policy unit tests and installed sdist-to-wheel validation passed. No hosted CI outcome is implied.

| Final binding | SHA-256 |
| --- | --- |
| Converter sources including the 32-record ledger | `950d47adb66140a4edcb8f1b7cdab82653f99657d6ee9329eb1939ecb575600d` |
| Original failed benchmark report | `18c1b81bc612a3d75364b85eb73f86360e96fa56b5195b5c96f9bf84a5a43399` |
| Independent serialization audit | `f524a021fa52bdc162dcfd9c97aeadf2113c7a2d6b23d8783e5adac17041ca90` |
| Corrected benchmark freeze | `5fc2c1e9111dc3b41214cc5f4d9c592b8715c4b5d2f32fe36977be4225e92142` |
| Corrected fifty-command report | `09fe0100aa96fc7a47dbb8e258d52987d4845cce17c454bbbcff6e0e0c9de994` |
| Independent final source/artifact/benchmark audit | `31ca49e0c17f63e0d02fa71aa9bd5251a92a96f06da90b9fdb6e6a3c0d68b3e5` |

Model weights, tokenizer payloads, private native artifacts and raw model metadata remain ignored. Public documentation contains aggregate results, public identities/revisions and hashes. Model-dependent audits precede the user-directed weight cleanup; retained receipts preserve historical validation and do not replace available source weights.

## Model cleanup

The independent preflight checked the exact script, inventory and full hash-verification receipt; guarded legacy/global cache blob layouts; selected immutable revisions; current file identities; and preserved evidence/metadata hashes. It found zero unselected cache references, hardlinks, active handles or unbound outputs. Supplementary header-only checks verified complete contiguous payload coverage for all ten corrected frozen weight artifacts before deletion.

The user-directed cleanup unlinked **67 exact paths**: seven selected source aliases, seven canonical blobs and fifty-three generated weights. Regular removed bytes total **168,108,206,490**. Observed free space increased **168,131,035,136 bytes**, about **156.58 GiB**; storage readings can also reflect other host activity. Small evidence, metadata, receipts, all one hundred benchmark observations, scripts and native build artifacts remain retained. Source replay requires restoring the removed pinned weights.

Independent postflight confirms all sixty-seven approved paths are absent and the receipt removal set exactly matches the approval. All fifty retained evidence-file hashes are unchanged, and all twenty-three unique non-weight source metadata files still match their manifests. Benchmark, freeze and independent-audit bindings remain unchanged. Final documentation links/anchors and private-path checks pass. No remaining review gate or blocking finding remains.

| Cleanup binding | SHA-256 |
| --- | --- |
| Exact unlink script | `ca23a7c94d56d48c0259dcb09c371e1d3153ad6b1b2855dbf47ee792014c135d` |
| Sixty-seven-path inventory | `9f3793c54d9f3149ec083cbe246d5137081fbf3c3d1da2c2088fdae9a44fdbb2` |
| Full sixty-file hash verification | `60ba16e4235b18d95d252180ba5f313fe6ebee02033f05c48ccc60da0a3a5e5c` |
| Independent preflight approval | `187c4e0397aedfc32cf92d9d0707561d2b0e1ae70ccb82799b694492ca353327` |
| Completed cleanup receipt | `a259b9d5f944fc8f9905c77b9ba1e8f9afda92fb1d7631f912e490c84f119261` |
| Independent absence/evidence postflight | `439fdbe64a16a713992e32e172871d91c33bc2e3c22bb3737741c710056b1b8e` |
