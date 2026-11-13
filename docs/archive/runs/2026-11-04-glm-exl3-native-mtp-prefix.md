# GLM EXL3 native MTP and head-prefix mechanism validation

Date: 2026-11-04. Branch: `codex/glm-prefill-throughput`.
This is a separate mechanism-validation record for new native tooling. It does
not change the earlier serial F32 qualification or qualify throughput.

## Source and workload

The native source bundle is
`43379ce0d2acc566dade421a368111edf22950c446297115eeadc8ce6ab10765`.
The new native module patch is
`fa0e0b49669c6c5be294fae805b7e7dfce84abbd5eb3634be1c1d891422c8179`;
it follows the unchanged original prefix patch. Host/plugin, MLX/MLX-C,
dependency artifacts and the verified model payload remain individually pinned
in the private stage receipts. Upstream copyright and licenses are retained.

The model is
`OpensourceWTF/GLM-5.3-Flash-streaming-repack-exl3-3.05bpw` at
`50237d5c26ceb98bfd85f779990b0f5e6d7af144`. The verified download receipt is
`3f4f04462a6e155f4c1aa36238818777cc5e0914dab8acf69bb7b0a5c84ebdbe`.
No model files were modified.

The original 30-file public-source corpus is reconstructed in RAM from frozen
`d0c54deb` sources. The 16,384-token U32LE input SHA-256 is
`06da8fa52d1c92a827ae8022d756fbf26e3de95e706803769b7952db898b57c9`.
State validation uses A=16,449 tokens and B=16,513 tokens with the same first
16,384 tokens and eight committed output choices. Only aggregate comparisons,
counts and hashes are retained; prompts, responses, token vectors, readout/state
payloads and decision probabilities are not persisted.

## Native policy

The candidate connects the pinned plugin's chosen-token and committed-round
interface: BF16 MTP head, depth three, gate 0.5, exact acceptance, fused
verification and eager prompt-tail completion. Its policy identifiers are
`native-target-draw-match-v1` and `glm53-exl3-native-round-v1`.
They do not assert equivalence to the existing Rust sampler or to serial
execution with different numerical shapes. Prior diagnostics found greedy
serial/MTP divergence on this workload. Those results remain recorded in the
[comparison](../../benchmarks/2026-11-04-glm-mlx-stream-comparison.md).

The new loader bills the head even when parked. Mode-off parks it and restores
the original prompt/step route. Prefix capture includes recurrent/attention
state and the canonical head boundary with one pending head input. Snapshot
restore is tied to the loaded owner and state epoch. Errors and cancellation
invalidate it. Complete request cleanup drains already-submitted work before
reporting success and never samples another round during cleanup.

## Actual-model result

The v3 controller and v1e C checker passed with automatic admission at
119 prompt / 146 decode expert rows, context 32,768, 112 GiB ceiling and 2 GiB
margin, plus separately billed snapshot reserve. This is numerical-only work;
external CPU quietness was not a performance gate. The live memory, zero-paging,
single-model ownership, privacy, artifact and process cleanup guards remained
active. The child exited successfully, was reaped and its process group was
proved absent. All post-run identity guards matched.

| Check | Result |
| --- | --- |
| Native greedy 16K / 128-output reference | Exact same-policy fingerprint `882d4378e2910a4d`; 45 rounds, 99 verified drafts, 83 accepted |
| Restored B versus fresh B readout | All 154,880 F32 values finite; zero differing words, maximum absolute/RMS error zero |
| Restored B versus fresh B logical state | Before and after state hashes identical, including head state |
| Restored B versus fresh B output | All eight committed choices identical; hash `56942044c3a95d12`; three rounds, six verified, five accepted |
| Head parked, original serial step/readout | Original full-readout fingerprints `3379492f9916220b` / `1038fa289a111105` |
| Cancellation, mode change, invalid input | Stale snapshots rejected |
| Different loaded model owner | Stale snapshot rejected after a predeclared two-second post-destruction wait |

The before/after B state hashes are respectively
`fcf79b48ac8c8f7001dfc4c1ead556bb213e3d056642e337a6eb10142834e647`
and `1f06218a0fb84558b37bbd5ef7a1c1bfa5c56c640440b51354c8ae0e85ea3248`.

An earlier checker incorrectly required cancellation drain to return cancelled;
the owned drain deliberately returns success after quiescence. Its failed
receipt remains separate. After correcting that expectation, immediate model
replacement failed admission despite unchanged valid options. The v3 run adds
exactly one two-second wait before exactly one reload; it changes no context,
budget, rows or inference policy and makes no retries. The delayed reload passed.
Immediate reload and the proposed stale-admission explanation remain unqualified.

## Receipt bindings and limits

| Artifact | SHA-256 |
| --- | --- |
| Prefix native stage v8b receipt | `54e273200145cae7404048cd7aa3e221cb4713f87844cc6283d5daa69c11d51a` |
| C checker v1e CPU receipt | `3ce9ecf104be2b493a46891d8eb90aa81e934cbd04fe9e97be8a47e4bb39477f` |
| C checker v1e executable | `da69bc4046db1604f3e20ac8e2087167264403f855eb968250148f884015cb4f` |
| Controller v3 | `5a4535cb073bcba95e6bf3e2e5ebd8fc9774d5fb71f13ddcc26c7b0af7fa2913` |
| Completed actual-model result | `feec93cc6a7144f58d7646832e9b090686aac7aaec9da909e54163eb7263da64` |

This record establishes the tested native greedy mechanism and boundary reuse.
It does not qualify the new Rust worker, native mode-off sampler API, stochastic
requests, arbitrary contexts, immediate model replacement, broad model quality
or production throughput. Ordinary native capability bits remain disabled while
separately source-bound private artifacts run those checks. Any later source or
tooling change requires its own receipt and result binding.

## Private v9 native API and worker follow-up

A separately built v9 bundle adds default-disabled private capability metadata;
the inference patch is unchanged. Its native source SHA-256 is
`34da056a26a8b2ad64737f3dbf048021bfad5cf69a2c26df9e2e57d46761bd7d`.
The private prefix stage receipt is
`7743f9254b4b42dc70f53b1503faac0f43d42c7b239ac910747a0ef992b72ada`.
The v4 controller uses an independently reviewed C checker and archived sources,
and its complete actual-model result is
`e29fc39ef28c0244ea55fd3952f575956a60b667fd62a90f5cc10260881e3261`.
Automatic admission selected 115/141 rows. All source/model/artifact postguards,
the live memory/paging guards and strict process cleanup passed.

In addition to repeating the earlier state, fingerprint and invalidation checks,
this run exercised the actual native mode-off sampler API. All 128 choices
matched legacy serial execution in RAM, with fingerprint `35bc67a84cf320ba`,
128 cap-zero rounds, zero verified/accepted drafts, and both original full F32
readout fingerprints. The native seeded mode-on path matched the separate
temperature-1/top-p-0.95/seed-1 1,024-output reference exactly:
`89c3e863f47bad76`, 367 rounds, 889 verified drafts and 657 accepted drafts.
This is same-policy agreement; the two mode fingerprints do not assert
serial/MTP numerical equivalence.

The private Rust v9c worker SHA-256 is
`c5b81c959569b269ce787fa84bf866301314c13f5f081c1a1d6416ba448a1f54`.
Its CPU/source receipt is
`54d141d6d5a79028aba1d107600b3f3630e39e8abff6082c5e6aae67be5eb4f2`.
The scoring build used Rust 1.98.1. The final source passed 115 generator tests
(two ignored), nine filtered native engine tests, 75 generator tests without
default features, Clippy with warnings denied and the prescribed nightly
formatting check. Native stage CPU tests and C ABI checks also passed.

The actual worker matrix passed all 41 cases. Its result SHA-256 is
`fcb66ff6e0a7ec1ffdafb4e0e8b01b73ac4252ffb3a01a7089b67cd119358a2e`.
It matched all three native output/counter references, checked fresh versus
reused B in both request modes, required exactly 16,384 reused input tokens,
and covered budgets one through eight, request-capacity boundaries, explicit
stops, cancellation, invalid-request recovery and cache reset. The worker was
reaped, its process group proved absent and every postguard matched. An earlier
matrix used an invalid cache-scope format and was rejected before its first
scoped backend request; that failed tooling/result remains separately bound.

An independent review caught a loader-selection issue before publication: API
availability alone could preload a head even when capability bits were off.
The generic loader now requires a separate validated-capability configuration;
the API configuration continues to control FFI declarations. Rebuilding the
enabled private worker produced the identical executable hash above. The
ordinary reference worker also builds with API FFI present and native loader
availability absent. Its source/compile receipt is
`2a73b63fdb1762f228d12e34bcec535bca9d3a6de3dec0ac0c97c56039b2eaa8`;
that particular reference executable has not run a model.

The 41-case matrix is a bounded worker contract check. It does not inject every
native reset/end or callback failure, force an actual model EOS at a selected
position, qualify immediate reload, or establish model quality or throughput.
Native production capabilities remain disabled pending the separately recorded
quality and performance work.

## Objective quality follow-up

The corrected, separately versioned smoke run completed all 24 arms and passed
22. It retained all six original input/task/expected-answer hashes, both
sampling policies, both native modes and the original 1,024-output limit. The
verified GLM template opens a reasoning channel; this harness graded only its
strictly delimited final channel with the unchanged exact JSON/task grader.
Missing, repeated, malformed or truncated framing failed closed.

Both modes exhausted the output budget on the same greedy interval-merging
case. Every other paired task passed in both modes. No task was replaced,
selectively retried or given a larger budget. This bounded outcome does not
demonstrate an MTP-specific task regression, and the predeclared all-pass quality
gate remains unmet. Native production capability bits remain disabled. The
worker was reaped and all identity and cleanup guards passed.

The corrected result SHA-256 is
`b336fd1e1b97dcda1d9527a1cbf93ce3aac93e4c127245da93fdfcebe4942cd6`.
The earlier harness incorrectly graded the complete reasoning-plus-final
response as JSON. Its interrupted result is preserved separately and supplies
no model-quality verdict; discarded responses cannot be retroactively graded.
