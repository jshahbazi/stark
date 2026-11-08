# Lily thinking-prefix parser boundary — 2026-11-08

## Finding and scope

The optional benchmark thinking route exposed a host parsing bug.
`Desktop.infer` determined whether the trusted input had opened a reasoning
block with an exact suffix check. The pinned Lily template renders one line
feed after its opening marker. That made the host report a closed input state
while the generated text was continuing an open block.

With that state mismatch, a synthetic reasoning continuation followed by a
complete, valid Qwen XML call was rejected as `invalid_tool_call`. Passing the
correct initial-state flag to the existing parser accepted the same synthetic
call. The parser already supports this continuation; the caller misidentified
the rendered boundary.

This finding qualifies a harness defect. The earlier thinking-enabled pilot
on caller `f034fdab5ae6c337c5e422a4d0107ab5418f553c` remains an unqualified
source-bound attempt. Its three invalid-call errors cannot establish model
formatting quality. No model responses were inspected, so this record does not
claim that any particular rejected response would pass after the fix.

## Bound inputs

- Artifact source: `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl`, revision
  `0f94842e3682b49acace4daeb4cf2e65a6f94f83`.
- Artifact fingerprint:
  `e8a92e5a992edfa5265df7fa5787bb13b4aaeef3218f5aeec25993df6d94995e`.
- Reviewed template SHA-256:
  `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`.
- Exact template-helper SHA-256:
  `1765a2c28309db99b8095da015035b89acc2cf4759752fba931079121421eb97`.
- Pre-fix Desktop source SHA-256:
  `13cf1a1d28c7c3ab1a0122cbaf2c2f055c4af698cd508c88ff50a7ae5f17fd64`.
- Unchanged tool-parser source SHA-256:
  `5b8f893c53d5961dd5de0db29931a32fb45cbc791e8a76a2f4b2400003f95b16`.

The exact installed template and qualified helper were exercised with synthetic
messages and a synthetic typed function definition. Rendering and parser calls
stayed in memory. The observation contained boundary booleans, hashes, fixed
error codes and whitespace codepoints only; no rendered prompt, model response,
token vector, credential or exception text was recorded.

## Model-free reproduction

| Optional thinking | Trusted block open | Old host flag | Whitespace after opener | Synthetic continuation |
| --- | --- | --- | --- | --- |
| Disabled | false | false | None | Unexpected closing marker rejected |
| Enabled | true | false | U+000A | Valid call rejected with old flag; accepted with correct flag |

An explicitly opened and closed synthetic reasoning block already parses with
the ordinary closed-input state. An unclosed block remains rejected.

## Change and checks

The change normalizes only the suffix-detection view with `rstrip()` before
checking the trusted reasoning opener. The original rendered string still goes
to tokenization unchanged. No generated text is repaired or stripped, and no
native math, cache, precision policy or generation option changes.

Regression coverage exercises buffered and streamed XML calls with LF and
CRLF/tab/space suffixes. It checks correct and missing reasoning closure,
non-thinking inputs, unknown function names and a deliberately false worker
initial-state claim. Every tokenization request must receive the exact original
synthetic prompt bytes.

Validation using Python 3.13.13 and the locked converter environment:

- 32 new boundary cases passed.
- 12 existing rendered-prefix binding cases passed.
- 284 tool-parser cases passed.
- Ruff, staged repository policy and diff checks passed.

The tests used synthetic workers and the exact qualified template helper on
`PATH`, with the modified checkout's converter source on `PYTHONPATH`. There
was no model inference, GPU workload or native build. A fresh thinking-enabled
model run is still required before attributing task outcomes to that route.
