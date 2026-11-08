# Native MaxxBench pilot — 2026-11-08

**Scope:** Single-task coding-agent feasibility and harness qualification for seven default artifact profiles. Five attempts were graded; two legacy attempts require reruns. No 89-task score or model ranking is reported.

## Method and execution identity

Each model attempt used `tomlkit-toml-table-converters` from MaxxBench revision `1a0df5e9c5d4bccbae131b01830d478088fea947`. The task has 76 feature checks and 964 regression checks. Stock oracle execution passed all of them. Full success requires upstream reward 1.

The agent used Pier 0.3.1, mini-swe-agent 2.4.6 and its stock task loop, prompts, bash tool and submission marker. Stark supplied one loaded native model through its local API. Temperature was 0 with seed 0, the output cap was 4,096 tokens, MTP and optional template thinking were disabled, and the step limit was 250. A model template may still require reasoning. The task allowed three hours for its agent and 30 minutes for verification.

| Cohort | Caller revision | Context tokens | Host client / API deadline |
| --- | --- | ---: | --- |
| Advanced feasibility | `f6eacfcedde1fa2a5a7cf7eb492c3d30f1f42893` | 131,072 | 600 / 1,800 seconds |
| Beginner pilot | `72276952f9ff6c751c926f6e17b3165889156098` | 40,960 | 1,810 / 1,800 seconds |

The legacy client deadline was shorter than the backend allowance. GLM and Flint transport failures remain unscored and require reruns with the corrected client deadline. Earlier 32,768-token probes and the first invalid Flint adapter run remain separate historical records.

Host: Apple M5 Max, 128 GiB unified memory, macOS 26.7.2; Xcode 26.4.1, macOS SDK 26.4 and Rust 1.98.1. The runner used Python 3.12.13 and the bridge used the locked Python 3.13.13 converter environment. Native build provenance is `b9a91a94ea07313cc0558d556f26a300c6603ff1`; the campaign receipts bind its executable hashes separately from the caller revisions above.

These local artifacts and disabled optional thinking differ from the provider models and reasoning settings in the [published MaxxBench results](https://maxxbench.unlap.ai/research/why-maxxbench/). Independent task images were force-built without Docker build cache; their image IDs are recorded per attempt. Byte equivalence between those builds was not established.

## Exact artifacts and storage precision

| Artifact | Pinned source | Prepared weight representation |
| --- | --- | --- |
| Qwen3.8 27B oQ4e | `Jundot/Qwen3.8-27B-oQ4e-mtp` @ `7dd276b77fcb27c982ebd8d8a434539345713fea` | Mixed affine 4/5/6/8-bit source; prepared streaming artifact; MTP unused |
| Qwen3.8 Flash Next Lily Q4-XL | `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl` @ `0f94842e3682b49acace4daeb4cf2e65a6f94f83` | Q4 GDN; Q8 attention, shared, head and embedding weights |
| Qwen3.8 Flash Next mixed | `ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit` @ `788c07a8725642bccf13a0a3b8a21954ab84b4f2` | MLX-Serve mixed 4/8-bit |
| GLM-5.3 Flash | `unsloth/GLM-5.3-Flash-GGUF` @ `a38483c8cd5df544f53d70fb281afe97369d5ab6` | Mixed GGUF `UD-Q2_K_XL`; this label does not mean uniform 2-bit tensors |
| Flint | `RecursiveSI/flint-int4` @ `3b93f9053b530488405ae9fca51d4900d6ce21f0` | Affine INT4 with BF16 scale planes; 8-bit embeddings |
| Qwen3 0.6B, BF16 recipe | `Qwen/Qwen3-0.6B` @ `c1899de289a04d12100db370d81485cdf75e47ca` | 113 BF16 and 113 F32 tensors; mixed BF16/F32 artifact, 1,192,276,800 weight bytes |
| Qwen3 4B, 4-bit recipe | `mlx-community/Qwen3-4B-4bit` @ `4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25` | MLX 4-bit group-64 packed weights; 290 BF16, 145 F32 and 145 U8 tensors; 2,263,417,416 weight bytes |

GLM metadata is pinned separately to `zai-org/GLM-5.3-Flash` @ `04c4e9e95c5da8862dced7e5056455116f83a7e0`. The 4B U8 buffers carry packed quantized values; their storage dtype does not mean an 8-bit model. Pilot receipts do not independently attest every artifact's activation, accumulation, recurrent-state or cache precision. Weight headers alone cannot establish those runtime policies or numerical equivalence.

Complete prepared-artifact SHA-256 fingerprints:

```text
Qwen3.8 27B       e437d98a9ac3b53f8e0e393652eee8b6bf6702175a56e36a871ef42eb1b1368f
Flash Next Lily  e8a92e5a992edfa5265df7fa5787bb13b4aaeef3218f5aeec25993df6d94995e
Flash Next mixed 14c71412308b5668ff4dfc9399858a123f66849db4e4e3a386dbd8762e05b964
GLM Flash        342a475b9544448aac3206385810963fdb1133272b2d5aae57f0bc5c89f64e3f
Flint            4c84481ae7657c35954de9d4e4e28ae0670febe162f83c30d550d3264b5e4764
Qwen3 0.6B       0a6f832045730183269c8d043404cbcd075dc8b3826389584ecec629b0b1c49e
Qwen3 4B         3814816ac6a0818b6dca4694d7bbe4c17614583d53b352ab0b3c3e00cb5abeea
```

## Observed task outcomes

Feature and regression counts concern collected patches. An empty patch grades the unchanged base state; a nonempty partial patch is graded even without a submission marker. No completed attempt below produced that marker.

| Artifact / cohort | Calls and terminal observation | Collected patch | Feature checks | Regression checks | Reward |
| --- | --- | ---: | ---: | ---: | --- |
| Qwen3.8 27B / 128K | 190 calls; context budget exhausted | 0 bytes | 0/76 | 964/964 | 0 |
| Flash Next Lily / 128K | 200 calls; context budget exhausted | 46,952 bytes | 66/76 | 964/964 | 0 |
| Flash Next mixed / 128K | 25 calls; stock `RepeatedFormatError` after 3 invalid tool calls | 1,002 bytes | 0/76 | 964/964 | 0 |
| GLM Flash / 128K | 3 calls; legacy transport/adapter failure | Ungraded | — | — | Unscored |
| Flint / 128K | 5 calls; legacy transport/adapter failure | Ungraded | — | — | Unscored |
| Qwen3 0.6B / 40K | 3 calls; stock `RepeatedFormatError`; no HTTP/API errors | 0 bytes | 0/76 | 964/964 | 0 |
| Qwen3 4B / 40K | 33 calls; context budget exhausted; no infrastructure error | 0 bytes | 0/76 | 964/964 | 0 |

Lily's patch failed 10 feature checks, so its reward remains zero despite passing all regressions. The empty Dense patch and the nonempty Lily/mixed patches represent different work states. These observations provide no basis for a model ranking. The 0.6B outcome applies to this coding-agent configuration; ordinary chat recommendations need separate evaluation.

Reported API usage on successful responses: Dense recorded 14,539,658 input and 9,207 output tokens; Lily 15,197,415 and 64,862; mixed 308,287 and 2,840. The 0.6B run recorded 4,251 and 143 tokens in 0.808 seconds of agent execution. The 4B run recorded 806,929 and 1,849 tokens in 72.726 seconds of agent execution, 107.366 seconds total. Failed API calls can omit usage. These are single unscreened task observations, with different termination paths; they do not qualify sustained throughput.

## Harness and operational qualification

Stock reference solutions passed three distinct targeted tasks:

| Task | Feature checks | Regression checks | Reward |
| --- | ---: | ---: | ---: |
| `tomlkit-toml-table-converters` | 76/76 | 964/964 | 1 |
| `adaptix-name-mapping-aliases` | 49/49 | 2,738/2,738 | 1 |
| `mashumaro-flattened-dataclass-fields` | 70/70 | 30,021/30,021 | 1 |

The latter two exercise the corrected `BASE_IMAGE` handoff, covering a historical one-key Compose override and the ordinary argument-based verifier. Seventeen of the 19 affected verifier paths still need oracle qualification. Passing these three tasks does not qualify all 89 environments.

A bounded Adaptix cleanup replay again passed. Its scoped cleanup removed one verified benchmark image and 13 new eligible cache records, preserved all 61 pre-existing volumes, and recorded deletion provenance. Host free space later measured approximately 25.5 GiB; that point-in-time value does not bound peak storage for another task. Source `72276952` passed 88 model-free pytest checks and 20 adapter checks, plus repository policy and Ruff.

Flint readiness reported no generation-prefix reuse capability. Its source has F32 attention/KV state but no exact prefix capture/restore implementation, so requests reprocess their prompt. This is a concrete implementation and qualification gap, not evidence that Flint's architecture cannot support snapshots or that its intrinsic model quality is poor. That gap and the multi-minute pilot calls remain scheduling concerns for a full campaign.

Before a full campaign: rerun GLM and Flint with matched deadlines, complete oracle qualification, and retain exact artifact/build/runtime identities for every profile. Full-task results and practical resource budgets must precede coding-agent recommendations. The [runner guide](MAXXBENCH.md) defines budget outcomes, cleanup boundaries and resume rules.

## Source-bound identities and disclosure

```text
native worker SHA-256  a589fb72b94ffd8381642ab898f348d53bc3d78d5e59c36ddf294e023ebb2d79
template helper       1765a2c28309db99b8095da015035b89acc2cf4759752fba931079121421eb97
upstream manifest     2c3f84fd5752dea080d6342f124775f4ee8689407e83b90402e0f3e1022d3f3b
legacy runner         9adc473c9b8579735d32f3859a1fee0ecbd41fc074250474834e3b0444d62178
legacy adapter        3c776b5f18ceb711126422d36e8169ff915778ef92f4ac782b0ea6ee3449c99a
72276952 runner       0797a521a83ac670a8bf1f7224585e11e0824b4a2cfaa5dd529b697ffd10b573
72276952 adapter      a4519c4773766d7f848ff39ed3969e34a035c327d8c163582b700fbb359db10f
dependency lock       bb91ad35bb41daca4f409d839a1ee67c947d9b16d84be25a740998182c6fd54e
```

Only aggregates, source pins and hashes are disclosed here. Task instructions, model conversations, token vectors, generated patches, verifier logs, credentials and private runtime paths remain excluded. Native and task artifacts stay ignored. No numerical-equivalence, whole-model quality, production-readiness or comparative-speed claim follows from this pilot.
