# Tool-calling pre-PR verification

Date: 2026-11-07, America/New_York. Reviewed implementation:
`c64fb79117a1dacdd5b62791d20dbebc42d2faa0` on
`codex/tool-calling-model-onboarding`, including main through
`8198f74872331765b8bcd9276c7171c09fe86890`.

This records a new verification run after the final Guided Setup guard and the
main-branch sync. The earlier [protocol and native-model evidence](2026-11-07-additional-tool-protocols.md)
retains its own source identities and outcomes.

## Review

Three independent read-only source reviews covered bounded JSON, Pythonic and
FunctionGemma parsing; rendering/history adapters; host streaming and native stop
handling; the native replacement/date additions; import capability assessment;
Guided Setup eligibility; and documentation/provenance claims. Additional direct
inspection checked the published SmolLM2 instruction shape, native template
requirements, queue admission and the Installed-row layout constraints.

No additional material correctness defect was identified in this bounded
review. Independent reviewers primarily inspected source; the executed checks
below provide the separate automated evidence.

FunctionGemma's parser is known to the protocol registry but excluded by
`GUIDED_TOOL_PARSERS`. Its `gemma3_text` architecture is present in
`DENSE_ARCHITECTURES`, so the parser-specific exclusion is necessary and covered
by a regression test. Import status continues to report protocol compatibility
with `model_behavior: unverified`.

## Executed checks

Environment: locked uv/Python 3.13.13 converter; locked uv/Python 3.12.13 policy
scripts; Rust 1.98.1; formatting with `nightly-2026-10-01`.

| Check | Result |
| --- | --- |
| Full offline converter suite | **6,426 passed, 90 skipped**, 250.23 seconds |
| Production Swift app-controller harness | **133 checks, zero failures, zero skips** |
| Native template helper tests | **2 unit and 17 protocol tests passed** |
| Native template release build | Passed |
| Native template Clippy, all targets, warnings denied | Passed |
| Repository policy and its self-tests | Passed; **41 self-tests** |
| Converter Ruff lint and format | Passed; 159 files formatted correctly |
| Pinned nightly Rust formatting | Passed |
| Converter dependency lock check | Passed; 29 packages resolved |
| Branch diff whitespace check | Passed with source-bound upstream fixture attributes |

Commands follow [CI local checks](../../CI.md#local-checks). The controller
harness compiles actual controller source with synthetic model identities and
isolated test state. It does not establish visual acceptance of the installed
app. The template and converter suites use synthetic/public fixtures.

## Native qualification boundaries

The preceding source-bound model records remain the behavioral evidence:
LFM2-350M passed the narrow call/argument and returned-result checks and failed
the exact ordinary-chat check; the tested FunctionGemma BF16 artifact failed
native loading before generation. Larger newly registered checkpoints still
require their own model-generated workflows and real-client acceptance.
Protocol support, conversion planning and synthetic test success do not
establish those results.
