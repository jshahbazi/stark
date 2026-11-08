# Tool-calling follow-up review

Date: 2026-11-07, America/New_York. Reviewed implementation:
`2de2628ec3a768d90a42b119059dde2a54681be4` on
`codex/tool-calling-model-onboarding`. This record supplements the
[original validation](2026-11-07-tool-calling-coverage.md); its source-bound
results remain unchanged.

## Findings and corrections

The review found two implementation gaps:

1. The pinned GLM generation prompt ends with an opening thinking tag. Native
   generation returns only the generated suffix, so a normal completion closes
   that input block before emitting a call. The original synthetic history test
   included both tags and missed this boundary; the parser rejected the actual
   suffix as an unmatched closing tag. The host now derives the initial state
   from the actual rendered prompt, overrides any worker-supplied flag, and
   passes that fact into prefix validation. A call before the block closes
   remains invalid. Flint shares the argument grammar but does not open this
   block; its unmatched closing tags remain invalid. Generated text and the
   streaming representation are preserved.
2. Native hybrid stopping behavior reads optional `generation_config.json` and
   `tokenizer_config.json`. Their bytes were absent from the validator's model
   digest. The file set now includes those files, the shard index and processing
   sidecars when present. It checks the optional file set again after execution,
   and binds the selected native template-helper executable before and after the
   run. It also rejects `expert_streaming` configurations before launching a
   worker, even when the external banks do not use the two known root filenames.

The validator now refuses an existing output path before hashing or loading a
model. Its ordinary-answer checks require closure of a prompt-opened thinking
block, so text produced entirely within an unfinished block cannot pass the
synthetic final-answer check.

## Validation scope

- Full offline converter suite: **6,258 passed, 90 skipped**, 247.36 seconds.
- Tool-support and qualification tests: **57 passed**. New cases include actual
  rendered generation prefixes, calls inside unfinished thinking, unclosed final
  answers, stop-token and shard-index metadata changes, helper replacement, new
  optional files and external expert-streaming configurations.
- Host prefix-handoff tests: **12 passed** across GLM, Flint and Qwen2.5,
  buffered/streamed replies, closed/unclosed input thinking, and worker attempts
  to spoof the initial state. Rejected cases never release executable calls.
- Three synthetic worker processes exercise the qualification command through
  the real native template helper and production `Worker` private pipes. These
  are protocol/process tests; their synthetic weight bytes are never loaded by
  a native inference engine and do not constitute model-behavior evidence.
- Independent read-only reviews covered parser admission, runtime file reads,
  import labels and qualification false-pass paths. Repository policy,
  Ruff lint/format and whitespace checks passed.

The environment remains the locked converter runtime and separately built Rust
1.98.1 template helper recorded in the original run. No native model generation,
app replacement, new weight download or credential operation occurred. GLM,
Flint and Qwen2.5 still require model-bound workflow qualification.

| Corrected implementation | SHA-256 |
| --- | --- |
| `tools/converter/stark_convert/api_tools.py` | `2858f292c7bd7159c29910c378ac2ec5c058402b07eaab64f50ba9a72644d854` |
| `tools/converter/stark_convert/desktop.py` | `dafa19ba5f2b9bd351432766c1f3f869cf062f626739ec89173003872674e0d5` |
| `scripts/validate_tool_calling.py` | `af2cd9298cdab586bc19b188e6312fa6c0470cae81f0fb06831921b7f7040a9f` |
