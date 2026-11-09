# Qwen 27B pull request review

Reviewed 2026-11-09. Fix commits: `81d4aa47` and `7a8cca8b`. Integration
baseline: main `5a615cd2` (including merged Lily PR #112).

## Independent review

A separate reviewer inspected the complete branch diff, including raw-buffer
sharing and copy-on-write isolation, strict artifact/head admission, canonical
text graph derivation, text/image dispatch, memory admission, state/session
identity, cache keys and suffix positions, cancellation and the Lily interaction.
No blocking runtime finding remained.

The coordinator independently checked the qualifying receipt hashes, recomputed
all 24 measured rows, confirmed identical outputs, checked actual MTP proposals
and accepts, and verified all 14 nonperiodic near-capacity lifecycle results.
The frozen benchmark worker remains
`34d6d9851c6b6bdf4a69b321eb8df29ba4d3b51a0daa5410a6791e52e4ef8fd4`.
Its timing and numerical evidence stay bound to QDIP-07, not relabeled as a new
performance run.

## Integration findings and checks

- Main integration conflicted only in the benchmark and experiment indexes.
  Both Lily and Qwen records were retained. A stale current-result paragraph
  still described the superseded 94.44% Qwen regression as unfinished; it now
  links the bounded QDIP-07 result and preserves the failed predecessor.
- The Qwen engine, parameter, weight-matrix and native-worker source files are
  byte-identical to `7a8cca8b` after integrating main. Upstream runtime changes
  concern the separate retained hybrid backend and its protocol/memory checks.
- Fresh post-integration release worker tests: **135 passed, 2 ignored**.
- Fresh focused Python tests: **677 passed**, covering Qwen dense, affine,
  partitioned exports, vision, MTP and hybrid-image memory. The native template
  helper was rebuilt from this checkout before the Python run.
- Package-scoped formatting, repository policy and whitespace checks passed.
- Hosted repository checks subsequently found a two-character Python line-length
  violation in the converter's MTP tensor filter. Ruff formatting corrected that
  line; converter lint and all 188 converter formatting checks then passed
  locally. This follow-up changes no Rust inference code or model payload and
  does not relabel the frozen benchmark evidence.

No GPU benchmark was repeated because integration did not change the qualified
Qwen execution path. Startup remains separate, the speed study is short, and
other artifacts do not inherit qualification. See the
[QDIP-07 record](2026-11-09-qwen27-shared-weight-fast-text.md) for identities,
controls, failures and performance limits. Hosted CI is a separate gate; this
record does not claim hosted checks, merge, release or installed-app validation.
