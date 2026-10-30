# Qwen3.8 Uzu execution paused checkpoint, 2026-10-30

The user requested preserving this work on `codex/qwen38-uzu-execution` and
returning to the optimized existing executor. This commit preserves an
experimental implementation. It is not an accepted production replacement.
No pull request or merge is requested.

## Performance decision

The latest 4,096-input-token, 256-output-token sampled runs used the same
converted artifact, temperature 1, top-k 20, top-p 0.95, seed 42 and no
speculation. Warm decode throughput was 77.8234 tok/s for the existing executor
and 44.6945 tok/s for the retained standard Uzu path. The latter is about
42.6% slower. These are sequential runs, with output hashes repeated exactly
within each engine; the two engines do not have identical numerical or RNG
policies. Small component gains have not closed this performance gap.

The converted weight file SHA256 is
`c493d4a3f876f6495f42127744b6415fe4cd3287f017fbc4b12ecd6d1245e657`.
The measured existing-worker SHA256 is
`3fca1cb56dc0ab2d8ab9b870a2c5e0ae9a92289900e88cf70768f237f3445544`.
The measured Uzu-worker SHA256 is
`1d5ae11058ca0e1280105b03dc071366d668626a367285fa8fdd2a5bedc99255`.
Private aggregate reports are `old-production-top20-4k-current-v1.json` and
`uzu-lexical-router-top20-4k-current-v2.json` under the worktree's ignored
`runtime/` directory. The subsequent 64K-capability worker has a different
identity; these throughput results must not be attributed to that binary.

## Preserved implementation and evidence

The branch contains the complete standard Engine/LanguageModel decoder,
canonical packed WeightMatrix loading, all Qwen block families, request-owned
state and commands, strict admission, normal GPU sampling and removal of the
production Lily executor. It retains the parallel lexical router and the
independently bounded F32 split-K partial correction. Existing precision
records and failed experiments remain separate.

A private teacher-forced control used one prefill followed by 128 individual
M=1 forwards at each of two prefixes, repeated twice per engine. All readout
hashes repeated exactly within each engine. Across 16 reported checkpoints,
top-1 agreed at 15, maximum KL was 0.0836745 nats, maximum total variation was
0.198468 and minimum top-20 overlap was 12. NLL covers eight checkpoints per
prefix, not every continuation step. This is bounded numerical evidence,
not a broad model-quality qualification. The synthetic trace bridge passed
both CPU and Metal manual-M1 controls.

The Qwen-only 65,536-token admission source passed 48 Qwen tests including
selector/cache boundaries, 52 focused Python cases, 229 affected Python cases,
and 53 generator tests (two ignored). Rust formatting, repository policy and
affected Python lint checks passed. The actual-model full-64K allocation and
65,280-input-plus-256-output benchmark remain unrun. Ordinary models retain
their existing context limits.

Private rowwise, wider-load, ILP, fused-projection, alternative affine-layout
and packed GDN experiments are not adopted in production source. Affine-layout
controls preserved coefficient bits and F32/BF16 outputs, but the measured
expert gains were small relative to order effects and self-control variation.
The packed GDN exploratory component was approximately 5% slower. Promising
Q4 rowwise component results have no completed sampled-generation trial.

## Remaining work if this branch is resumed

Refresh the full validation suite and final independent conformance review
against the resumed source revision. Qualify actual long-context operation,
preserve precision evidence, and require meaningful sampled-generation gains
before considering a production migration. The 77–80 tok/s target is unmet.

At the user's request, rebuildable Cargo targets, temporary executables and
Metal traces are removed to reclaim disk space. Compact private source
snapshots, patches, aggregate results and their identities remain available.
Historical binary hashes continue to identify the original runs even when
their rebuildable executable files have been removed.

## Model-copy cleanup after preserving the branch

The optimized existing worker was subsequently validated directly against the
retained original Lily checkpoint, revision
`0f94842e3682b49acace4daeb4cf2e65a6f94f83`. Three sampled 4K/256 replays
completed with the same output hash as the converted-format fast-path run:
`621ca90417af420aaf9defdc957f7cc1f57f318c08d908be764834abb382b0d0`.
Warm decode throughput was 77.1778 tok/s. Worker identity and sampling settings
were unchanged; this check establishes that retaining the converted weight
file is unnecessary for the current fast-path workload.

At the user's request, the redundant canonical Uzu artifact and the older
PipeNetwork Qwen checkpoint with different quantization were removed. Their
configuration, source receipts, indexes and small metadata were saved under
the ignored `runtime/native-hybrid/removed-model-metadata-20261030-v1/` in the
primary checkout. The original Lily checkpoint, its quantization policy and
the GLM checkpoint remain. Resume this experimental Uzu branch by reconstructing
the canonical artifact from the retained pinned Lily source and verifying
the recorded model hash before reusing earlier qualification results.
