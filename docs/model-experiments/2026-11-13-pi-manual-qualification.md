# 20261113-PI-01–08: guarded Pi manual qualification

Registered retrospectively on 2026-11-13 after execution, owner Codex. These
records preserve changed qualification recipes and prior failed/superseded
results. They do not imply a preregistered model benchmark. The governing
scope was [manual Pi integration](../AGENT_CLIENT_INTEGRATION_PLAN.md), with
exact CLI/tool/cancel/recovery evidence required before an artifact claim.

## Identity and decision

| Field | Value |
| --- | --- |
| Kind / evidence | Client compatibility, lifecycle and bounded task checks; short timing observations |
| Primary checkpoint | `mlx-community/Qwen3-4B-4bit`, revision `4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25` |
| Artifact weights / config / tokenizer | `7c2af9a45e5b0d196f06a5cbfd6fc5cc95adf29df79885d0540f2206fb3d40d8` / `5b505594bf00ef9682b373c6a292b88f94f1e1ceab449784c12accb835806c7f` / `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| Backend / installed build | Rust/Metal, Stark 0.20.5, source `9e33ab729581994269f56c64685e27a0bdd94c27`, worker `da8a9e5384300e488da486741f617981c36ff8e83c7d7e1f66925451ef835ee7` |
| Candidate | Guarded manual integration `d05490c6af5eccf7f8ca068f4a5840a8dd3fb76f`; installed Pi 1.5.2 |
| State / outcome | PI-01–07 failed or superseded as below; PI-08 completed, 21/21 live checks |
| Promotion | Guard adopted on feature branch; bounded installed-CLI/backend artifact profile qualified |
| Disposition | Pushed branch; no PR/merge/release or global profile installation |
| Related evidence | [Canonical run record](../archive/runs/2026-11-13-pi-guard-live.md), earlier [foundation](../archive/runs/2026-11-12-pi-manual-foundation.md) and [recheck](../archive/runs/2026-11-12-pi-manual-recheck.md) |

## Problem, controls and fixed scope

Raw Pi accepted missing usage or EOF without `[DONE]` after a finish reason.
The manual extension requires both before success/tool execution and uses a
custom API marker to prevent fallback when absent. Model templates, native
precision, kernels, cache implementation and artifact files were unchanged.

Primary weights are MLX affine 4-bit/group-64, with the unchanged packaged
runtime precision policy; no new numerical equivalence result is claimed.
Activation/accumulation/KV formats were not separately instrumented. Both source
artifacts are text-only Qwen3; images are N/A. MTP is unsupported/off in the
observed capability profile. The alternate Qwen3-0.6B BF16 source revision
`c1899de289a04d12100db370d81485cdf75e47ca`, weights
`87c3344a80b7376392cc5c44d5369680b68936e6930bb8ac476ca65a136f64c7`,
was used only for routing/recovery checks.

Host: Node.js 26.13.0, macOS 26.7.2, arm64 Mac17,7, 128 GiB RAM. No app rebuild
or engine compile was done. Context/output: 40,960/1,024 for the ordinary Pi
profile; eight-token truncation and 16-token direct controls. Native controls
were greedy with default seed, MTP/thinking off. Client retries and automatic
compaction were disabled. Manual compaction used reserve 2,048/recent 128.

Cases required explicit expected outcomes; failures remained failures. Native
work was serialized by this harness. Another observed Edge0 job finished
naturally before qualification; it was not interrupted. No host-wide quiet,
peak-memory, paging or full-window performance qualification is claimed.
Private temporary profiles, isolated backend state and synthetic workspaces
were removed after every recipe. No real credential or conversation was logged.

## Attempt history

The private receipts below contain only aggregates. Historical source hashes
not present in a receipt are **not recorded**; final source is sealed separately.
Preparation receipt `basic.json` failed before model load because an attachment
directory was missing; its hash is
`eb4737b5a1eb997a58286e89bd8b471b01a15aa755ecf834a028558b3537f270`.

| Attempt / receipt suffix | Outcome | Validity and disposition | Receipt SHA-256 |
| --- | --- | --- | --- |
| PI-01 / `basic-2` | 5 passed, literal-word follow-up failed | Nonempty answer; initial empty-answer description corrected. Task/formatting miss retained | `8b4d210fa11f71e3c43801a6eaf360176101e2c2a89c6f566a0b821640302dbc` |
| PI-02 / `full-1` | 14 passed, 6 failed | Word miss, ineligible compaction and stopped-API polling/cleanup cascades; no full qualification | `3d1424a414bfd73a72468d794e2fd305711460c3ba4900c71ca242a04b4ac1dd` |
| PI-03 / `full-2` | 19 passed, compaction failed | No eligible old turns; superseded input recipe | `0115ecb830e8b0baa6c12f7e64664e3e407a5e333740c87826b1f1e4fb37a56a` |
| PI-04 / `full-3` | 20 passed | Superseded: admission/text-based cancellation, weak command predicates and answer supplied in compaction instructions | `a36180a273199aa8182d806405ef8e93457b0d8498e82e9d24ae433282d51dc3` |
| PI-05 / `final` | 21 passed | Added matched controls; same weak predicates, superseded | `0a2faf7919ffac58f6f6ad3cb5a4eb72bdf97a9bf8208fe0810ff4d0397bf089` |
| PI-06 / `final-verified` | 20 passed, decode gate timed out | Wrong progress field; host exposes phase/sample counters. Superseded | `ba9254b00f9b3f6bee0c95ae625e431bfd05c5de6b1b6921080ab07ed71e6eab` |
| PI-07 / `qualified` | 20 passed, compaction ineligible | Stronger phase/command/error controls; insufficient old turns after tools were disabled for cancellation. Superseded | `a593f072aa68f4e7f86550fb3f198aadd53da0c0399cdb0ce0046643489e09bb` |
| PI-08 / `complete` | 21 passed | Final source-bound controls; independent review passed | `a2a945d09107100aa58b345108b612413a5845239cbf47dc0846b0a99f4bfe40` |

All receipt filenames start `pi-live-2026-11-13-` and end `.json`. PI-08's
harness SHA-256 is `852da1c26c699195aacaee1155d955d66137f1cca93e3b0c69549520fc7b3e36`.
Its source/test manifest is `e7a61f9f3fa3ad08c8458d0c95a0d966d3206d3802eb8ec99855246db21b45f4`.
The canonical record links the published aggregate JSON and hashing recipe.

## Correctness, timing and limits

PI-08 verifies exact requested test/diff commands and failure/recovery order in
memory, an unchanged test file, historical marker retention without supplying
the answer to compaction, native prefill/decode progress before cancellation,
expected context/model errors, restart/disconnect/interleaving, and exact greedy
reused/fresh output/finish/tool structure with 3,072 reused tokens. No comparison
to upstream numerical logits or broad coding score was run.

First visible client text was 336 ms cold and 110 ms on follow-up. ABBA short
raw/guarded controls used 3,527 input/two output tokens, with 3,072 reused tokens
in warmed arms. Two observations per arm and request/native telemetry are in
the canonical aggregate; they establish no production speedup. Peak memory,
full-window timing and image-resident comparisons were not measured.

The result covers installed Pi's CLI RPC and the isolated packaged backend with
manual ephemeral auth. The Swift setup/credential UI, terminal keybindings,
other artifacts/versions, managed grants and full UI-consumer backpressure
remain unqualified. A new runtime/artifact/client version requires a fresh run;
no broader promotion follows from this profile.
