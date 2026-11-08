# Beginner recommendation evidence

Beginner checks resource fit before ranking a bounded set of choices. Its
current ranking uses resource/capacity heuristics and explicit task metadata.
It has no measured coding, writing or file-task score. The native tool workflow
checks in this repository establish specific protocol/lifecycle behavior;
they do not establish general coding-agent competence.

## Current native and recurrent coverage

`guided_memory.py` inventories request state without loading weights. Fixed
native profiles in `guided_native_profiles.json` bind the weights repository,
metadata repository, immutable revisions, configuration and prepared-config
identities, template/parser, selected-file download manifest, stored weight
bytes and precision policy. A changed pinned source needs a refreshed profile;
it cannot inherit the old one.

| Route | Planner and limitations |
| --- | --- |
| Qwen3.5-family DeltaNet text exports | Full-attention KV, F32 recurrence and convolution, prefix snapshots, declared MTP attention and verifier state, scratch allowance. Conversion still checks the exact source layout and BF16/quantized precision policy. |
| Flint INT4 | Full and sliding F32 KV rows, host descriptors/growth and native workspace. Requests remain fresh because the existing Flint backend has no prefix restoration. |
| GLM Flash UD-Q2_K_XL | MLA/index caches, F32 KDA/control state, host fallbacks, external score planes and two graph scratch owners. Full weights remain charged to physical and GPU estimates. A per-encoder bank cap is not a whole-model residency bound. |
| Qwen Flash Next original mixed export | Native F32 graph/scalar cache, GDN/convolution/PLE and host workspace; full-weight GPU estimate. This source does not inherit Lily's paged GPU deduction or numerical evidence. |
| Qwen Flash Next Lily Q4-XL and MLX-Serve mixed | Q8 KV plus scales, sparse index buffers, F32 GDN, convolution/PLE history, MTP state, live/replaced prefix snapshots and prefill/decode scratch. The entire paged n-gram payload is charged to system memory; only its staging enters the GPU plan. |

All native plans include a 3 GiB loading/workspace allowance and round system
and GPU estimates up to GiB. These are conservative estimates, not observed
peak RSS or a guarantee that the loader will admit a machine under memory
pressure. The guide retains the 2 GiB physical reserve and 85% Metal budget
check, and rereads available memory after releasing its previous worker.

For the 128 GiB / 107.52 GB Metal-budget envelope, the current 32K-context
estimates are 29 GiB for Flint, and 111 GiB system / 82 GiB GPU for the paged
Qwen sources. GLM and the original mixed export estimate 111 GiB GPU and fail
this guide budget. This does not remove their Advanced paths or imply they
cannot execute using mapped ranges on that Mac.

Native preparation stores the original payload once, publishes by rename and
uses a 4 GiB setup reserve. An exact installed copy does not need download
space. Reuse also checks the complete prepared configuration, source metadata,
template/parser and quantization; a similarly named artifact is insufficient.
Cancelled downloads receive credit only for physically allocated payload in
their exact pinned, owned and currently unlocked staging directory. Sparse
logical sizes, foreign source markers and symlinks supply no credit. The guide
reports remaining setup space and rereads this credit before installation.
Text recipes do not enable image input or add a backend. Existing native prefix
reuse and its artifact-specific qualification remain unchanged.

## Proposed task evaluations

These evaluations are a plan. No benchmark runner, score import or benchmark
ranking is implemented by the native-coverage change.

| Beginner task | Recommended evidence | What it would establish |
| --- | --- | --- |
| Coding | [LiveCodeBench](https://livecodebench.github.io/) on one fixed recent window; later [SWE-bench Verified](https://www.swebench.com/verified) under one pinned agent harness | Code-problem pass rates first; repository issue resolution in the deeper track. They measure different tasks and should remain separate. |
| Tool use across coding and file tasks | [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) plus Stark/OpenWork workflow checks | Tool selection, arguments and multi-turn behavior under named tools and harnesses. A successful tool call is not a completed coding/file task. |
| Writing, explaining text and everyday assistance | [IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval) plus a small human-reviewed task suite | Explicit verifiable constraints; separate human rubrics for correctness, usefulness and writing quality. IFEval alone does not measure prose quality or factuality. |
| Files | A local OpenWork task suite with known input files and expected outcomes | Grounded reading, correct edits, path/source attribution and completion using the real file/tool workflow. No generic language leaderboard establishes this integration. |
| Unsure | Coverage of the general suite above | A general starting choice with stated evidence gaps. There is no separate universal score. |

Start with a bounded LiveCodeBench window, instruction-following cases and the
existing tool workflow suite. SWE-bench requires a much more expensive agent,
repository and container harness; it belongs in a separate evaluation effort.

## Score admission policy

A usable score must identify the exact model and revision, quantization and
artifact hash, Stark worker build, tokenizer/template, context, sampling/MTP
settings, benchmark split or dated window, evaluator/agent harness revision,
completed case count and run date. Changing conversion or backend precision
does not preserve an artifact's score automatically.

Only compare the same task, data window and harness. Repository-task scores
from different agent setups are not interchangeable; the
[SWE-bench FAQ](https://www.swebench.com/SWE-bench/faq/) documents harness-dependent
leaderboard comparisons. Keep component scores, coverage and uncertainty
visible in evidence records rather than averaging unrelated percentages.

Published upstream scores may be retained as attributed reference evidence.
They are not measurements of Stark's quantized exports. Missing results remain
unknown and must not silently become a zero, a claimed win or an invented
quality rating. Once comparable task results exist, they can rank choices that
pass resource admission; resource estimates must not be relabeled as task
quality.
