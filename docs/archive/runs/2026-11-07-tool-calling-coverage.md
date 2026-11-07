# Tool-calling coverage and onboarding checks

Date: 2026-11-07, America/New_York. Branch:
`codex/tool-calling-model-onboarding`, based on
`53803d233b0bcb9b7eae95476fb3dbc4b2fb389d`. This is implementation,
native-template and synthetic-worker evidence. The installed app was not replaced.

## Changes and identities

- Added bounded GLM argument-key/value parsing with explicit scalar decoding,
  schema checks, complete-call requirements and the existing buffered/streamed
  API validation. The exact pinned GLM and Flint templates share this grammar.
- Admitted the reviewed Qwen2.5 Hermes template through the existing JSON parser.
  Definitions, null-content assistant calls, typed arguments, parallel history
  and reordered tool results were rendered with the native template helper.
- Added actual-codec tool assessments to import completion and library metadata,
  and fixed labels in Installed rows. Saved capability flags do not grant parser
  support or change loaded-model capabilities. Behavioral evidence stays separate.
- Added the required tool assessment to model onboarding, a current family
  coverage matrix and a bounded model-workflow validator. The validator rejects
  abnormal termination, checks returned-result use, hashes model/worker bytes
  before and after execution, and keeps output private and content-free.

Unchanged upstream fixture sources, immutable revisions, hashes and licenses are
listed in [fixture provenance](../../../tools/converter/tests/fixtures/tool_template_additions.provenance.md).
No weights, conversion rules, native state, prefix-reuse rules, credentials,
dependencies or client repositories changed.

| Implementation file | SHA-256 |
| --- | --- |
| `tools/converter/stark_convert/api_tools.py` | `7d476f037068390dae054d03e80e39439f1ea9a5e89160f20a8f14f76daefefb` |
| `tools/converter/stark_convert/model_inventory.py` | `f58c559f36f93bd262dccff2349ff8e5d637d7079f3f109145ca48d00aa7e373` |
| `scripts/validate_tool_calling.py` | `cfc8bfd97b6f857be263ee5af809d76b67d166b8fe0354f994f1da9a0696b933` |

## Validation

Environment: macOS 26.7.2 (25G332), Apple silicon; uv 0.12.24, locked Python
3.13.13 for converter checks, Python 3.12.13 for repository policy, Rust 1.98.1.
The native `stark-template` release helper was built with `--locked` in this
worktree's separate target directory. Native generation was not started.

- Full offline converter suite after the GLM/Qwen2.5 implementation: **6,218
  passed, 90 skipped**, 243.54 seconds. Subsequent additions were checked below.
- Final template/admission/parser/inventory/validator suite, including Flint:
  **421 passed**. This covers preserved scalar values and multiline Unicode,
  history order, malformed/quoted/truncated calls, unknown/duplicate arguments,
  schema ambiguity, output limits and every split of streamed tool markers.
- Desktop tool/API suite: **100 passed**, 190 deselected. The shared fixture
  includes the GLM grammar for tool choices, errors and streamed/buffered replies.
- Exact-codec import and already-installed completion checks: **4 passed**.
- Swift controller: **133 checks, zero failures**. Tool labels remain distinct
  from loaded-model runtime capabilities; malformed/legacy metadata is unchecked.
- Window harness: **7 passed**, including real SwiftUI compilation.
- Repository policy, Ruff lint/format and whitespace checks passed.

Read-only review identified whitespace normalization in the GLM function-name
path and metadata-only post-run artifact checks. The implementation now requires
an exact bounded function name and rehashes model and worker bytes. Regression
tests reject padded names, cancelled final answers, wrong calls/results and
same-size payload edits with restored modification times.

## Limits

No new Qwen2.5, GLM or Flint model-generated task result is claimed. Another
installed generation worker was resident during this work; this change did not
stop or replace it. The available small Qwen3 artifact already has separately
recorded historical evidence. No additional model weights were downloaded.
Newly admitted programs remain behaviorally unverified until the new validator
and appropriate client checks run on their actual artifacts/builds.

The current remaining-family dispositions are in [tool coverage](../../TOOL_CALLING.md).
Llama, Mistral, Granite and LFM2 native tool formats still need separate reviewed
integration. Other checkpoints need an explicit template/training assessment;
classifier conversion does not provide function calling. No universal model
competence or performance improvement is inferred from protocol fixtures.

Only public template metadata, fixed synthetic fixtures, counts and hashes were
used in the committed evidence. Generated responses, token vectors and private
runtime data were not recorded.
