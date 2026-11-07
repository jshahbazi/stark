# GLM Flash, Flint and recent-model tool assessment

Date: 2026-11-07, America/New_York. Runtime implementation source:
`542d8e9f05db5e6a34036f9e05b615ad1100cd94`, PR #95. This is a new,
artifact-bound follow-up to the
[larger Qwen qualification](2026-11-07-recommended-model-tool-qualification.md).
Previous source-bound GLM EXL3 and numerical records retain their own outcomes.

## Environment and inputs

Apple M5 Max, 128 GiB unified memory, macOS 26.7.2 arm64. Runs use locked
uv/Python 3.13.13 dependencies and the PR's successful Apple silicon standalone
build, Actions run `42383934745`. The generation worker SHA-256 is
`07c3de5e19ada5ee739bd22b62affd88f029169ba896463247e6a649a18b7292`;
the Rust 1.98.1 native template helper SHA-256 is
`14d81380b4b0e3904ff5a0cb1048d7671ba2ba8818bf413365d47c8fcb66327a`.
Native qualification workloads are serialized. The installed app and its
separate idle worker are unchanged.

The production native preparation helper downloads original quantized payloads
and bounded metadata without executing repository code:

| Artifact | Pinned weights | Pinned metadata | Stored weight bytes |
| --- | --- | --- | --- |
| Flint INT4 | `RecursiveSI/flint-int4` at `3b93f9053b530488405ae9fca51d4900d6ce21f0` | Same repository/revision | 22,568,912,600 |
| GLM-5.3 Flash UD-Q2_K_XL | `unsloth/GLM-5.3-Flash-GGUF` at `a38483c8cd5df544f53d70fb281afe97369d5ab6` | `zai-org/GLM-5.3-Flash` at `04c4e9e95c5da8862dced7e5056455116f83a7e0` | 108,720,071,552 |

Each native run seals actual model payloads and runtime metadata, the worker,
template helper and validator before and after execution. The HTTP driver also
seals its source and converter toolchain. Persisted reports contain aggregate
checks, counts, identities and hashes. Messages, responses, generated calls,
arguments, credentials and token vectors remain in memory/private pipes.

## Qualification contracts

The tracked `scripts/validate_tool_calling.py` checks call selection/arguments,
returned-result use and exact ordinary chat. Runs use greedy sampling, an 8,192
token context, a 2,048-token output ceiling and a 600-second worker timeout.
MTP is unavailable on both tested paths; no draft requests are made.

The private HTTP driver uses the production `Desktop.run`/inference lifecycle
and LocalAPI with a disposable registry, ephemeral loopback port and in-memory
credential. It checks required, named and automatic calls in buffered and SSE
modes; none-mode responses and equality; returned-result use; ordinary chat;
integer arguments; repeated requests; cancellation and recovery; and actual MTP
settings. Cancellation requires server-side `request_cancelled`, idle queue/job
state and a successful following request.

Its cache policy must match the native worker's advertised capability. Flint
uses the fresh policy: repeated completed replies must match the exact target,
reuse counts must be absent or zero, and recovery must return the same completed
answer. Flint's full/sliding F32 KV state and logical position have no worker
snapshot/restore implementation, so prefix reuse stays disabled. These checks
do not establish prefix restoration. GLM GGUF's reuse qualification uses its
separate 1,024-token snapshot grid; it must cross the advertised boundary and
observe actual reuse. Repeated and recovery replies must complete and match
their exact target.

The first private Flint API report used equality-only repeat checks. A review
found that matching truncated replies could satisfy those checks. That report
and source capsule are preserved. The corrected driver requires completed,
nonempty exact replies and uses the declared output budget for repeat/recovery
requests. Earlier Qwen reports retain their original driver and scope.

These are bounded behavioral checks. They do not establish numerical parity,
throughput, larger-context behavior or broad coding/agent task competence. GLM
GGUF results do not qualify the optional EXL3/FP8 build or full GLM-5.3.

## Flint results

The exact pinned INT4 artifact passed **3/3 native smoke checks** and the
corrected **17/17 HTTP checks** at 8,192 context tokens. The HTTP output ceiling
was 512 new tokens, including repeat/recovery requests. The worker advertised
`prefix_cache.supported: false`, no chunk size and no MTP. Repeated requests
reported absent reuse counters. Buffered/streamed tool choices, typed arguments,
returned-result use, exact ordinary replies, completed repeated replies,
cancellation and completed recovery all passed. Both reports confirmed stable
runtime inputs.

The original and effective template hash is
`7940f4ffa3275c36caa4a582f845aaad79da11018b3961e8fbcad6ae29733466`;
the selected parser is `glm53-xml-v1`. The prepared configuration hash is
`06271655d67c1ce5cfc08ccc771faaddcc1a1c9719b3b2c499ddae1c1aa8cef0`.

| Flint artifact or private aggregate record | SHA-256 |
| --- | --- |
| Runtime artifact manifest | `cc4aa0fba97634425aa296de8805236d698235d93a4e7ddd7e4857caf28df97d` |
| Native smoke JSON | `e9410f44fd7c5d863ef7c3d6a76352164dcc771ec2baf9b7301c07d34b2f6c5f` |
| First equality-only API JSON, preserved | `95bc3373d927aaf059154ae468a35508067eef0dd0b85139c019b4f2281ddf4b` |
| Corrected API JSON | `ba4b921bcb156eeaae880fd01addf59db25525747105a58995b4da07496b6547` |
| Corrected cache-policy API driver source | `7386b751b76eeb72aef94c7b6a48743545dbd8ee9f4f469c33587516546234d4` |

Thirteen model-free private-driver/diagnostic checks passed after the repeat
validation correction. Exact source capsules and earlier reports remain
private and unchanged. No engine change was needed for this qualification.

## GLM Flash GGUF results

The pinned UD-Q2_K_XL artifact passed **3/3 native smoke checks** at 8,192
context tokens with a 2,048-token output ceiling. Call selection/arguments,
returned-result use and the exact ordinary reply passed; runtime inputs were
unchanged. The original/effective template hash is
`34d5ee66b12fa6446cdae131c352b8f68cd85369e0e6fda115583805fada3891`,
selecting `glm53-xml-v1`. The prepared configuration hash is
`d1e229307a382fc1deae8e4a48b6ffc2f47fc47769ab8789c023bdbd806072fd`.

The HTTP run passed **17/17 checks** at 8,192 context tokens and a 2,048-token
output ceiling, using the corrected driver listed above. The worker advertised
prefix caching with a 1,024-token chunk and no MTP. A 1,301-token probe observed
reuse counts of zero followed by 1,024, with completed exact replies matching.
Server-confirmed streamed cancellation released the job; the recovery request
reused zero tokens and returned the same completed exact answer. All tool-choice,
history/result, ordinary-chat, typed-argument and lifecycle checks passed.
The API report confirmed unchanged runtime inputs and host toolchain.

The source config declares 1,048,576 positions. The 8K request ceiling used here
does not reserve or qualify that full capacity. Before native launch, the host
resource probe reported 98,813,018,112 available bytes against 108,720,071,552
stored weight bytes. The native backend maps weights and selects range/bank
views; these tests make no fully resident fit or throughput claim. No
experimental `STARK_GLM*` selectors were inherited or enabled.

| GLM artifact or private aggregate record | SHA-256 |
| --- | --- |
| Runtime artifact manifest | `79999706f90f6e437fd973b0c88743321c1f8ebda816bc80f9aadd09a26e384e` |
| Native smoke JSON | `7532fdac106e8d1c0cb1b289b1bd6dbfc6b30f991bb1cb1b9330f4e1e27c55b0` |
| HTTP API JSON | `a6d45fee20264447a3e483605e316600a08eb876b56d8357a58750ab65dd3f66` |

## Recent-release intake

The following pinned config reads and source checks were completed on the same
date. No weights were downloaded or loaded for these six repositories. Their
architecture names establish an onboarding gap; they do not establish tool
training, parser compatibility or runtime quality.

| Repository and pinned revision | Config `model_type` | Source disposition |
| --- | --- | --- |
| [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170), `1bae0e1088993b9b82de08f59de5be5a34e42170` | `deepseek_v41` | Backend work required |
| [Kimi-K3.1-Pro](https://huggingface.co/moonshotai/Kimi-K3.1-Pro/tree/d9e99b0792a5e93b78ea9bea62a68167e36ddd9e), `d9e99b0792a5e93b78ea9bea62a68167e36ddd9e` | `kimi_k31` | Backend work required |
| [Beam-1.0-501B-A23B](https://huggingface.co/reflection/Beam-1.0-501B-A23B/tree/12950ae640b48cbbe9b5bb7735bf197507d6383f), `12950ae640b48cbbe9b5bb7735bf197507d6383f` | `beam` | Backend work required |
| [Nemotron-3.5-Bolt-74B-A9B-24GB](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB/tree/e0590ed2f3a564fd99725d9d891cb09deb7cc201), `e0590ed2f3a564fd99725d9d891cb09deb7cc201` | `nemotron_h_omni` | Backend work required |
| [Gemma 4 31B-it](https://huggingface.co/google/gemma-4-31B-it/tree/842da3794eaa0b77d5f08bae87a17459d91ff475), `842da3794eaa0b77d5f08bae87a17459d91ff475` | `gemma4` | Backend work required |
| [Full GLM-5.3](https://huggingface.co/zai-org/GLM-5.3/tree/ba32f8c19c8738e2c57c71798c601d477a7bf844), `ba32f8c19c8738e2c57c71798c601d477a7bf844` | `glm_moe_dsa` | Backend work required; distinct from Flash |

The generic adapter inventory in
`tools/converter/stark_convert/model_support.py` has no adapter for these model
types. Native hybrid dispatch in
`crates/stark-engine/src/engine/hybrid/mod.rs` recognizes `glm5_next`,
`qwen4_exp` and `flint`; the GLM loader validates Flash's explicit text geometry
and GGUF architecture. This does not admit the full `glm_moe_dsa` architecture.

NVIDIA's [October 15 announcement](https://blogs.nvidia.com/blog/nemotron-3-5-super-vl/)
describes Bolt 74B as a local 24GB variant. Google's
[release page](https://ai.google.dev/gemma/docs/releases) documents Gemma 4,
including its 31B variant. They are relevant larger local-model intake
candidates, with backend and artifact qualification still required on Stark.
Their published memory claims are not Stark measurements.

Unsloth's [Qwen3.8 Next guide](https://unsloth.ai/docs/models/qwen3.8-next)
describes V3.1 GGUF quants. No exact GGUF repository/revision or artifact was
qualified here. Stark's separate `gguf_import.py` experiment accepts Qwen3
Q8_0 only and validates Qwen3 tensor inventory. The existing Flash Next MLX
implementation and Lily's passing workflows do not qualify a GGUF format path.

For each candidate, first implement its explicit architecture/format contract
and qualify numerical execution and state lifecycle. Then review the exact
template and trained tool protocol, add bounded parsing/history tests and run
artifact-bound native and client workflows. The
[model onboarding requirements](../../MODEL_ONBOARDING.md) apply before adding
it to recommendations. This intake does not change Guided Setup admission or
recommendation ranking.
