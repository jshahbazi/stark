# Tool-template compatibility validation — 2026-11-01

Date uses America/New_York. Implementation:
`46ac275be87e4aecbb9a95c1cba93eec77984e41`, based on
`8630ee4230bddd70517ef32a300370dbb9d2ee84`, branch
`codex/tool-template-compatibility`. This is source and native-worker evidence;
the installed Stark app was not rebuilt or replaced for this change.

## Behavior and boundary

The source-hash fast path remains. A bounded, versioned Jinja AST fingerprint
also admits syntax-equivalent versions of the four existing reviewed programs.
Matching uses the renderer's shared immutable sandbox, loop-control extension,
and whitespace settings. It never compiles or executes the candidate.

Every node class, ordered field name, typed scalar and ordered list contributes
to the fingerprint. Source locations are omitted, and the fixed rendering
environment is shared rather than serialized. Adjacent literal `TemplateData`
nodes within an `Output` are joined with their exact contents.
This accepts comments, equivalent expression spacing/quoting/escapes,
parentheses and non-emitting formatting. Literal text, emitted whitespace,
instructions, markers, branches, loops, calls and lookup behavior remain bound.
Unknown programs fail closed. Source UTF-8 bytes, traversal work/depth and the
eight-entry nonpersistent template cache are bounded.

Generated-call, schema, history, stream and lifecycle validators are unchanged.
No new parser grammar, credential route, dependency or execution facility was
introduced. Separate vision/artifact integrity and evidence gates remain exact.
Protocol recognition does not establish model task competence.

## Environment and checks

- uv 0.12.24; locked Python 3.13.13 and Jinja 3.1.6.
- Repository policy and its 41 tests passed under locked Python 3.12.13.
- Full offline converter suite: **5,601 passed, 89 skipped**, 189.66 seconds.
- Final focused admission/parser/chat suite: **358 passed**, 3.15 seconds.
- Independent final admission/parser review: **306 passed**, 3.52 seconds;
  additional adversarial syntax, malformed-input and resource-bound probes
  passed. No remaining review finding.
- Ruff lint/format, locked dependency check, whitespace check and pinned-nightly
  Rust formatting passed. Rust 1.98.1 is installed; no Rust source changed and
  no new Rust compilation or performance result is claimed.
- The existing converter wheel build completed offline. Fresh imports of chat,
  admission and the new module came from that wheel, which included the exact
  committed module bytes and admitted a comment-only template variant. Wheel
  SHA-256: `13912dddceee8ab80762c6c5f311747ead906ebccdb5dee8d72f4726638afd65`.
  This packaging smoke did not build or install a macOS application.

The 53 new tests cover harmless real-fixture edits and byte-identical rendering,
meaningful mutations, scalar types/order, attribute versus item lookup, invalid
Unicode/syntax, byte/depth/work limits, cache bounds, and no execution during
matching. A real Desktop controller with a synthetic worker exercises production
admission, capability reporting, tool-result history and HTTP streaming with one
tool delta per call. Unknown names, duplicate arguments, truncated/length-ended
calls and fenced examples remain rejected after compatible-template admission.

## Reviewed programs and metadata-bound rendering

Original template bytes were read locally and verified against the previously
registered source hashes. No additional upstream template text was copied into
the repository. The existing Apache-2.0 Qwen3-4B fixture remains unchanged.

| Reviewed source | Original UTF-8 SHA-256 | `stark-jinja-program-v1` SHA-256 | Parser |
| --- | --- | --- | --- |
| Qwen/Qwen3-0.6B, revision `c1899de289a04d12100db370d81485cdf75e47ca` | `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8` | `2537497c84cb9ae2a885de78e7531cd44c62f38dadfcfa7565793ce9e38481ac` | `qwen3-hermes-v1` |
| mlx-community/Qwen3-4B-4bit, revision `4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25` | `87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5` | `813d6e292f35cf32a98357c5382ec57ae39ddb658e5b16559c957b119a2859b5` | `qwen3-hermes-v1` |
| fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl, revision `0f94842e3682b49acace4daeb4cf2e65a6f94f83` | `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041` | `ece3bd43a4194cae4a2d3b6079214373b84f9083ff6258940d0b03634e93f08e` | `qwen38-xml-v1` |
| alibiserikbay/JevK5, revision `c4f7fdb3aeab5582336406e78d3bef11bf98833d` | `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715` | `a0feb6a6a23bced6ef842bcb46dad27dc43738bf9acd5751606f24aa343610d8` | `qwen35-xml-v1` |

For each source, five synthetic variants added comments/source lines, changed
expression spacing/parentheses, changed equivalent quoting, or used CRLF.
All **20 variants** retained their parser. Rendering ordinary chat,
tool-enabled/thinking chat and a completed call/result/follow-up history produced
**60 byte-identical comparisons**. All **8 meaningful mutations** (appended
literal whitespace or changed call markers) remained unsupported. These checks
read metadata and render in memory; they do not qualify new weights or load
mutated vision artifacts.

## Native model smoke

The parent agent ran a separate, short native worker using the installed
generator binary and the existing converted **Qwen/Qwen3-0.6B BF16** artifact at
revision `c1899de289a04d12100db370d81485cdf75e47ca`. A comment-only template
variant existed only in process memory and was automatically admitted. No model
files or installed application contents were modified.

| Artifact | SHA-256 |
| --- | --- |
| BF16 `model.safetensors` | `87c3344a80b7376392cc5c44d5369680b68936e6930bb8ac476ca65a136f64c7` |
| Model config | `dfa5763dc220e871b86a6119536f5c1c2862a5137a473eaf9d812be3cc53924e` |
| Reused installed generator | `f39ff75827573aaae66c74ce896fbf5dcf0158dfa9e85bb48cac0c84bdec1a0d` |
| Updated `api_tools.py` | `b5359dba441a3e099af66069033ac63434e19718242a98606de1c808bc083a76` |
| Updated `chat.py` | `38ee2d8110f12111baa76ed2315591a3d96069a5692907be810d57ec3574e166` |
| New `template_compat.py` | `7d51ad9776c4a9e6d30f0358c7defde033d5e38b610ada3b0f88c316811dedbc` |

Greedy buffered and streamed generation produced the same single valid known
function call and arguments, with **162 input tokens, 18 output tokens and 18
stream text chunks**, ending normally with `eos`. Strict parsing returned
`tool_calls` for both. The worker was reaped afterward. The run requested a
2,048-token context budget and at most 128 new tokens; it did not measure model
capacity or performance. No tool was executed by Stark.

This is a narrow rendering/admission/generation regression check. A new full
OpenWork or Goose workflow, app packaging/signing/upgrade check, and broad agent
competence evaluation were not run. Earlier desktop acceptance remains bound to
its [separate build and workflow record](2026-11-01-openwork-lily-acceptance.md).
The installed app retains that earlier build until explicitly rebuilt/installed.

## Artifact handling

The private driver under ignored `runtime/` printed only hashes, aggregate
counts, booleans and sanitized failure categories. Generated content and token
vectors stayed in memory/private pipes; no raw request/response dumps or real
conversation content were retained. No production credential or Keychain item
was read, copied, rotated or written. The unrelated untracked beginner-model
research document was preserved.
