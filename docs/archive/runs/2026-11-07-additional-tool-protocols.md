# Additional trained tool protocols

Date: 2026-11-07, America/New_York. Branch:
`codex/tool-calling-model-onboarding`. Base before these changes:
`dd14153e583776aec9314aa66a1b05cecf4034be`. The implementation is identified by
the source hashes below. Earlier [coverage](2026-11-07-tool-calling-coverage.md)
and [review](2026-11-07-tool-calling-review.md) records retain their original
source-bound results.

## Implemented scope

Added reviewed protocols and rendering/history adapters for Mistral v0.3,
Llama 3.1/3.2, Granite 3.0/3.1/3.3, LFM2/LFM2.5, SmolLM2/SmolLM3,
Phi-4-mini, FunctionGemma, and the rnj-1 Hermes template. Each registration
is tied to source bytes or an equivalent bounded Jinja program. The unchanged
fixtures, immutable source revisions, normalized/program hashes and licenses are
in the [fixture manifest](../../../tools/converter/tests/fixtures/tool_templates/README.md).
Registration establishes protocol compatibility; checkpoint behavior remains
unverified in import/library metadata.

The reviewed Mistral runtime asset splits one JSON-output expression to avoid
markup escaping of its surrounding quoted key. Its SHA-256 is
`dfbff0166eb003d00c7a23594654f9783f70efab223f91e2bddc0a8e0fa059ed`.
The host also escapes string-valued definition fields/results and maps public
call IDs to render-only nine-character IDs. The original imported template is
preserved, and the validator records both original and effective source hashes.

Llama's reviewed native history permits one call per assistant turn. History
that a template cannot represent is rejected before queue admission.
FunctionGemma receives a restricted concrete schema/value subset and a
tokenizer-derived native response-stop ID for tool requests; all four client
stop strings remain available. Ordinary chat does not tokenize that marker.
Its specialized tool-routing profile is excluded from general-assistant Guided
Setup recommendations even when the native memory-planning gates pass.
LFM's bounded Pythonic decoder parses data without evaluation or execution and
preserves documented prose following the completed call frame.

Exact native SmolLM3 rendering exposed missing string `.replace()` and date
formatting capabilities. The native helper now bounds replacement expansion
before allocation and admits the three local-calendar date formats used by the
reviewed SmolLM3, Llama and Granite programs. Process allocation, output, fuel,
CPU and cancellation limits remain in force. No model engine, precision policy
or generation-prefix cache algorithm changed. Tool requests retain the existing
single-worker serialization and exact-prefix reuse rules; these protocol checks
do not qualify prefix restoration for another artifact.

## Automated checks

- Full offline converter suite: **6,423 passed, 90 skipped**, 249.64 seconds.
- Final recommendation, Mistral/Llama and other-family integration checks:
  **147 passed**, including subsequently added prequeue rejection and specialized
  profile recommendation regressions. The final recommendation guard is covered
  by this focused run after the full suite.
- Rust template helper: **2 unit tests and 17 protocol tests passed** under
  Rust 1.98.1. Python tests additionally exercise native Unicode/empty-pattern
  replacement, bounded expansion and local-calendar rendering.
- Repository policy, converter Ruff lint/format, pinned nightly Rust format and
  whitespace checks passed. Source-bound upstream template/license fixtures
  retain their original CRLF/trailing spaces through scoped Git attributes.
- A wheel was built and imported directly from its archive. It includes the
  Mistral template and Apache license; the runtime resource reader resolves the
  expected hash-checked template from that wheel.

Synthetic tests cover native definition/history rendering, scalar and nested
values, public/native ID boundaries, ordered parallel results, buffered/streamed
API paths, split markers, malformed/truncated payloads, duplicate keys,
unknown/disallowed tools, schema limits and required/named tool choices. They
do not execute external tools or establish model task competence.

## Native model checks

Machine: Apple M5 Max, 128 GiB unified memory, macOS 26.7.2 arm64. The converter
uses locked uv/Python 3.13.13 dependencies. Tests use the installed production
`stark-generate` executable without replacing the app. All requests are serial,
greedy, with context 4096, maximum 512 new tokens, MTP disabled and timeout 120
seconds. These are behavioral smoke checks, with no performance or numerical
equivalence claim. Private artifacts contain aggregate booleans and hashes;
prompts, outputs, argument values and token vectors are not retained.

| Exact artifact | Call and arguments | Returned-result use | Exact ordinary-chat check | Overall |
| --- | --- | --- | --- | --- |
| `LiquidAI/LFM2-350M` @ `f37d3f5c8c5484bc01dad379a595cf4c68c4e70e`, converted BF16 | Pass | Pass | Fail | Fail |
| `mlx-community/functiongemma-270m-it-bf16` @ `bb327a9ad61044e1496a2bee2365a6b6a6684c72`, converted BF16 | Not run | Not run | Not run | Native loading blocked |

LFM2 run 4 used the rebuilt helper and reproduces run 3's partial result. The
ordinary-answer assertion remains exact; it was not relaxed. The run confirms
artifact bytes/file membership unchanged after execution. The three-check
workflow has **not** passed overall. Its aggregate artifact digest is
`c1c1193cb07bcfcace83271d89e4f446896d06abc951e32abe7ff2ff6619d1b9`;
the model Safetensors file SHA-256 is
`d4e548ab46e52fc5cae93c11e293b02dc71a7166edc8084c18524ba66b97be1c`.

FunctionGemma passes schema and native-template protocol checks but the installed
worker returns `ready:false` / `model_load_failed` before generation. That
category hides the inner engine error, so its cause is unknown. No native
FunctionGemma tool-execution qualification is claimed. Its aggregate artifact
digest is `a0c336e1506d85bc7ad847780e351c79ea435d048d211e8c31485cc3ae49fd6b`;
the model Safetensors file SHA-256 is
`163c27d4ba738c522e267b25d097344fa68bf788613e6cd154f68ae4ff195001`.

| Private aggregate record or executable | SHA-256 |
| --- | --- |
| LFM2 run 3 JSON, preceding helper | `6d2e076268f8d7f43d2219785302c88ccaed663e94c9d04bae54f5391e4fbf02` |
| LFM2 run 4 JSON, rebuilt helper | `dd081273d0e4007a8e0c0b2ac6196ff7d2017a8d909110a0c3871dd9d31914ef` |
| FunctionGemma run 1 JSON, preceding helper | `b8cef8007e93f34af67d4be74c5fb7a849d5d8818e8c6c33d48fd2e38890eac7` |
| Native generation worker, both artifacts | `6b00c8969f459f69ee843c697d0838796e3414c8b53b897c50c760f2d49c9b15` |
| Preceding template helper, FunctionGemma run 1 and LFM2 run 3 | `34a8eecd24a09b9d2870427f8a91e8dbacdbc52ccd45fcf13168642bc4bb92fa` |
| Rebuilt template helper, LFM2 run 4 | `14d81380b4b0e3904ff5a0cb1048d7671ba2ba8818bf413365d47c8fcb66327a` |

No weights were fetched for the larger newly registered families. Their native
model-generated workflows and real-client cancellation/agent acceptance remain
pending. LFM2.5-350M's source template is reviewed, but its conversion route is
still planned. Unreviewed programs and checkpoints lacking tool-training
evidence receive no new claim from these tests.

## Implementation identity

| Source | SHA-256 |
| --- | --- |
| Native template `engine.rs` | `3f26d926cfa71c3e2447cbe37380b5f817a9c639e56cb6e28e4b390d306b1341` |
| `api_tools.py` | `5b8f893c53d5961dd5de0db29931a32fb45cbc791e8a76a2f4b2400003f95b16` |
| `tool_protocols.py` | `c0a1d82cb5c0360753ed7a4adeb4ece655878d6e183279934fe89a11791e0aa6` |
| `tool_pythonic.py` | `07af8135b99364bd068e36f24e23c463310191ddc80b31e99af380fd6ef6c27d` |
| `tool_functiongemma.py` | `167cd26b28362f1a093b4f2082783d8cec28897086bedca50cfa39929d306e44` |
| `desktop.py` | `76cd53dac8ee52d88a1ff14ee5a22a9512d21bd5f90d2e16596174e23ed45e7d` |
| `guided_models.py` | `f2d24bd3367c3a31357a6a2413550375d773935884602161f5d2d95de3ae0bb9` |
| `validate_tool_calling.py` | `50e71839d21541df6b4f01a01b6d4c961415d9e8fa31a5b64cb616b85b9d01a6` |
