# Beginner recommendation evidence

Beginner checks resource fit before ranking a bounded set of choices. Its
fallback uses resource/capacity heuristics and explicit task metadata except for
“I’m not sure,” whose fallback prioritizes recorded starters and workflow checks
without a parameter-count objective. Coding and “I’m not sure” with Balanced
prioritize published task benchmark evidence among admitted models. Other tasks
use bounded comparisons of scored options. All scores come from an
offline snapshot of published upstream benchmarks; they do not measure Stark's quantized exports.
Native tool workflow checks establish specific protocol/lifecycle behavior;
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
| Qwen Flash Next Lily Q4-XL and MLX-Serve mixed | Q8 KV plus scales, sparse index buffers, F32 GDN, convolution/PLE history, MTP state, live/replaced prefix snapshots and prefill/decode scratch. Pinned preparations can use resident planning before download; installed copies must also pass exact artifact checks. Other sources retain the full-payload estimate. |

Unqualified native plans include a 3 GiB loading/workspace allowance and round system
and GPU estimates up to GiB. These are conservative estimates, not observed
peak RSS or a guarantee that the loader will admit a machine under memory
pressure. The guide retains the 2 GiB physical reserve and 85% Metal budget
check. Recommendations and startup use the same `execution_allowed` predicate:
current available memory must cover the effective system plan. Startup rereads
memory and revalidates the actual artifact after releasing its previous worker.
An installed copy waives download space, never current execution headroom.

For the 128 GiB / 107.52 GB Metal-budget envelope, the current 32K-context
estimates are 29 GiB for Flint, 87 GiB system for qualified Lily and 88 GiB for
qualified MLX-Serve text (81/82 GiB GPU respectively; 111 GiB system fallback). GLM and the original mixed export estimate 111 GiB GPU and fail
this guide budget. This does not remove their Advanced paths or imply they
cannot execute using mapped ranges on that Mac.

`guided_memory_profiles.json` is shared with the native request gate. It binds
prepared config, source receipt and index hashes and the complete payload
filename/size inventory, including Serve's external n-gram file. Payload bytes
are not freshly rehashed. Fresh recommendations obtain the same resident plan
from a SHA-256 binding to the complete pinned native preparation profile,
including source revisions, geometry, precision metadata, templates and file
inventory. They do not require an installed copy. A changed or unknown profile
retains conservative planning. Disk, current RAM and GPU checks still apply.
An existing copy is independently revalidated; if it fails exact artifact
checks, the report and startup both use its conservative plan. A source-only
estimate never authorizes an unverified installed copy. At <=32K context and
MTP depth <=2 the request gate
charges the complete native inventory once; that inventory already covers
overlapping old/new state and scratch. Both planning and request admission
retain 8 GiB host reserve plus 1 GiB paged working set. The system plan also
charges the complete nonpaged payload. Larger contexts, deeper MTP, images and
unknown/changed artifacts retain the existing conservative request gate.
See the source-bound [memory observations](archive/runs/2026-11-08-qwen-guided-memory.md)
for actual measurements and limitations. This does not change precision,
sampling, prefix cache keys or restoration rules.
The [fresh-install regression record](archive/runs/2026-11-08-fresh-guided-memory.md)
describes model-free coverage across hardware/resource configurations.

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

## Published task references

Beginner separates **Explain text** (summaries, notes and explanations) from
**Work with files** (reading or editing a folder through OpenWork). The files
choice requests the `files` task cohort, which uses BFCL only as a tool-use
reference; it does not claim file-task performance.

`recommendation_benchmarks.json` contains aggregate scores, source URLs,
immutable source revisions where available, content hashes, settings and
explicit coverage gaps. There is no runtime leaderboard fetch or automatic
name matching. `recommendation_benchmarks.py` validates this optional snapshot;
an invalid snapshot falls back to resource ranking without disabling setup.

| Beginner task | Current reference | Limits |
| --- | --- | --- |
| Coding | LiveCodeBench v6, the same comparison table in the official Qwen3.8-Flash-Next and Qwen3.8-27B cards | Developer-reported reference scores for two parent models. Window, pass@k, sampling, execution policy and aggregation are undisclosed. Never combine this table with another publisher or leaderboard window. It does not measure repository issue resolution. |
| Files | [BFCL v4](https://gorilla.cs.berkeley.edu/leaderboard.html), official 2026-04-12 overall CSV | Tool-use proxy under BFCL's harness. FC, Prompt and FC-thinking remain separate cohorts. This does not establish file-task completion or OpenWork integration. |
| Text, everyday and unsure | [IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval), published HF Open LLM Leaderboard BF16 chat results for seven Qwen2.5-Instruct models | Narrow instruction-following signal. The snapshot's generic score column does not identify the strict prompt/instruction aggregation, so its metric stays labeled as published. It does not measure prose quality, factuality or overall usefulness. |

The registry captured on 2026-11-08 contains 33 BFCL FC results, 11 Prompt
results, one FC-thinking result, seven IFEval results and two developer-reported
LiveCodeBench results. Inclusion in the score registry does not add engine or
catalog support. Coverage is incomplete, particularly for newer native models.
IFBench and Toolathlon are separate benchmarks and are not relabeled as IFEval
or BFCL.

### Comparison policy

1. Apply the existing RAM, GPU, disk, chat-template and route checks first.
2. Build the deterministic fallback order from comfortable memory headroom,
   task tags and the selected capacity/memory preference.
3. For Coding with Balanced, select one coding cohort with at least one
   eligible covered parent. Prefer greater distinct-parent coverage across all
   admitted models, then benchmark-publisher evidence, then cohort ID. Put
   covered models first, in descending score order. Uncovered models follow in
   fallback order; a missing score cannot retain first place above recorded
   evidence. A valid zero score is evidence, not a missing score. This policy
   prioritizes evidence availability; it does not prove that an unscored model
   has lower coding quality. Soft memory-headroom tiers and task tags cannot
   override coding scores after hard admission. Equal scores use fallback order.
4. For “I’m not sure” with Balanced, select one instruction-following (IFEval)
   cohort with at least one eligible covered parent, using the same coverage,
   publisher and cohort-ID rules as Coding. Covered candidates precede all
   unscored candidates, even across soft memory-headroom tiers. Within that
   cohort, prefer sources with recorded bounded native chat/tool workflow
   checks, then descending score. Unscored fallbacks prefer the exact recorded
   starters in declared order (Qwen3 4B, then Qwen3 0.6B), then other sources
   with recorded workflow checks, then unchecked sources. Within those latter
   groups, prefer comfortable headroom, lower runtime memory and fewer weight
   bytes. Installed status and source identity break remaining ties. Parameter
   count never ranks this path. If references are unavailable, including a
   missing or invalid snapshot, the report explicitly labels the fallback and
   unknown answer quality. Bounded workflow checks do not guarantee reliability;
   IFEval does not establish writing quality, factual accuracy or local artifact
   performance. A single covered parent establishes evidence availability,
   without proving superiority to an unknown model.
5. For other tasks with Balanced, select one task cohort with at least two
   distinct covered parents in the same memory-headroom tier. Prefer greater
   comparable-parent coverage across those tiers, then publisher authority,
   then cohort ID. Reorder only scored slots within each tier by descending
   score, retaining unknown positions and equal-score fallback order.
   Repeated quantizations count as one parent for cohort selection in every
   task. Never merge FC/Prompt, versions, publishers or metrics. Results from
   nonselected cohorts remain reference information. Lighter model and More
   model capacity retain their explicit resource objectives.
6. Select up to three choices. Beginner shows a short recommendation reason,
   estimated memory, a compact benchmark score, a brief published-model caveat
   and a link to the evaluated model's Hugging Face page. It retains actionable memory-pressure notices.
   Full revisions, settings, qualification text and caveats remain in the
   structured backend report and evidence records. They are not concatenated
   into Beginner's details. Existing copies do not show a new setup-space need.

If no eligible coding reference exists, the report explicitly identifies the
resource-based fallback and unknown coding quality. A lone covered parent can
be prioritized over unknowns; it is not presented as a pairwise benchmark win.

Repository identity admits a model-level public reference even when the
evaluated weights revision is unknown or differs from the candidate. This is
reported as `model_repository_reference_only`. Evaluated revisions are retained
when published; matching a repository never establishes artifact equivalence.

Quantized exports can reference a parent's results only through a reviewed
repository/revision binding whose pinned source card identifies that parent.
The four initial bindings cover the Jundot Qwen3.8-27B export and three native
Qwen3.8-Flash-Next exports. Changed revisions need review. Finetunes, Coder vs
general Instruct variants and similarly named checkpoints do not inherit scores.
The report retains both candidate identity and evaluated-parent identity.

## Score admission policy

A Stark artifact measurement must identify the exact model and revision, quantization and
artifact hash, Stark worker build, tokenizer/template, context, sampling/MTP
settings, benchmark split or dated window, evaluator/agent harness revision,
completed case count and run date. Changing conversion or backend precision
does not preserve an artifact's score automatically.

Only compare the same task, data window and harness. Repository-task scores
from different agent setups are not interchangeable; the
[SWE-bench FAQ](https://www.swebench.com/SWE-bench/faq/) documents harness-dependent
leaderboard comparisons. Keep component scores, coverage and uncertainty
visible in evidence records rather than averaging unrelated percentages.

Published upstream scores rank only under the reference comparison policy above.
They are not measurements of Stark's quantized exports. Missing results remain
unknown and must not silently become a zero, a claimed win or an invented
quality rating. Resource estimates must not be relabeled as task quality.

## Later evaluation work

MaxxBench is deferred until the user runs it against the models. This change
adds no MaxxBench scores, runner or fabricated coverage. Admit future results
with the exact artifact and run conditions above and define a separate cohort.

Local OpenWork file tasks with known outcomes and human-reviewed writing and
correctness checks remain needed. A fixed-window local LiveCodeBench run would
replace developer references with artifact-bound coding evidence.
[SWE-bench Verified](https://www.swebench.com/verified) requires its own pinned
agent/container harness; scores from unrelated agent setups remain separate.
