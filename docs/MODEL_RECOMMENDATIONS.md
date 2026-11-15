# Beginner recommendation evidence

Beginner asks for **General Work** or **Coding**, followed by **Speed** or
**Quality** with Quality selected by default. General Work continues into the
OpenWork guide; Coding continues into OpenCode. Recommendations first check
resource fit, then use the offline coding-index policy below. The index does
not measure Stark exports, tool workflows or general-work quality.

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
check. Recommendations use `recommendation_allowed`: installed physical RAM,
GPU budget and setup storage determine which models the Mac can support.
Current available memory does not filter or reorder offers, including when
the available-memory probe is missing. The card can advise freeing memory
before starting. Actual preparation retains a separate 2 GiB current-memory
working allowance for new installs. Startup uses `execution_allowed`, rereads
available memory and revalidates the actual artifact after releasing its
previous worker; available memory must then cover the effective startup plan.
An installed copy waives download space, never current execution headroom.
The [installed-RAM policy check](archive/runs/2026-11-13-guided-capacity-recommendations.md)
records stable native offers across different available-memory observations.
The [corner-case review](archive/runs/2026-11-13-guided-recommendation-review.md)
records the effective startup reminder, shared alternative policy and
discovery ordering/protocol regressions.

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
retains conservative planning. Disk, installed RAM and GPU capacity checks still apply.
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

## Frozen coding indexes

The Catalog and guided Quality/Speed flows use the 28 connected model-level
coding indexes in `llmbench_coding.json`, derived from the frozen November 14,
2026 input snapshot in
[`docs/benchmarks/data/catalog-coding-ranking-2026-11-14.json`](benchmarks/data/catalog-coding-ranking-2026-11-14.json).
These are individual model indexes, not family averages. The catalog offers
**Top rated (coding)**. General Work guided setup uses the coding index as a
proxy and does not claim to rank general-work quality.
The Top Models tab remains its separate family-level calculation and is
unchanged.

This is an offline reference. Hardware checks and supported-model discovery
remain fresh; the recommendation path makes no website leaderboard requests.
Unknown models remain unscored and retain the documented fallback order.
Indexes are relative coding references, not percentages, Stark artifact
measurements or proof of general-work ability. Source, revision, attribution,
license and calculation details are retained in the aggregate record and the
[dated method report](benchmarks/2026-11-14-catalog-coding-ranking.md).

### Selection policy

1. Exclude candidates without a complete supported conversion/native route, usable chat template, tool parser, 32K context and adequate installed RAM, GPU budget and disk. Current available memory affects startup readiness, never recommendation eligibility or rank. Index rank never grants compatibility.
2. Use the fixed coding-index snapshot; it has no RAM buckets. Missing model identities remain unscored.
3. Apply only explicit model identity bindings for the pinned snapshot. Quantization names, family averages or fuzzy matching do not create a model score. A changed source identity requires another review.
4. **Quality** selects descending coding index after admission for either task. For General Work this is an explicitly disclosed coding proxy. Alternatives must be within **10 index points** of the best admitted model and represent distinct models. Equal indexes retain deterministic fallback order. Unknown models never fill alternatives behind indexed choices. With no indexed models, offer one explicitly unknown benchmark/resource fallback. The 10-point distance is an editorial threshold; the relative index is not a percentage or a measure of artifact or general-work quality.
5. **Speed** admits only models with a coding index of at least **35 points** and exact export weights no larger than **25% of physical RAM**. Installed RAM, GPU and disk checks still apply. Within that pool, select descending coding index; ties prefer a smaller runtime plan, then smaller weight payload, then deterministic fallback order. Alternatives use the same **10-point distance** and distinct-model rule as Quality, measured against the best candidate inside Speed's eligible pool. Unknown models, candidates below the floor and exports over the budget are excluded from Speed alternatives. These are editorial thresholds, not measured response speed or quantized task quality.
6. Offer one primary recommendation with up to two alternatives behind a disclosure; fewer are shown when the policy admits fewer. Quality cards identify the coding index and its reference model. Historical benchmark results remain reference information when the coding index determines ordering.

The index applies only to its recorded model identities. A connected reference does not qualify a differently quantized Stark artifact, export, backend or tool workflow. Speed's budget uses actual export weight bytes. It does not measure response speed or establish that an export preserves coding quality. Snapshot inclusion adds no backend, conversion route or native qualification.

If the index snapshot is unavailable or covers none of the eligible models, **Quality** retains its resource fallback with quality marked unknown. **Speed** reports no qualifying recommendation, with a direct **Choose Quality** action and Advanced browsing; it never waives the score floor or weight budget. Historical `recommendation_leaderboard.json` and `recommendation_benchmarks.json` remain bundled for older behavior and historical evidence; the new Catalog and guided Quality/Speed paths use the coding index. New guides send `general`/`coding` and `quality`/`speed`.

### Guided discovery coverage

Guided discovery checks a bounded metadata window, not every repository. It
prioritizes the selected task's cached sources before inspecting that window;
popularity alone must not spend the candidate budget before scored models are
checked. Registered seeds join the same priority ordering, and overlapping
Hub queries contribute each repository once. Pinned seeds retain their source
revision and are not screened using metadata from a different head revision.
Only downloadable rows cross the subprocess boundary. Its row limit and
timeout derive from the supported family count, page size, three lookup
workers and per-family deadline; unavailable pages cannot exhaust the result
budget. The serialized response retains its 2 MiB bound.
Search priority does not grant a parent score or waive compatibility,
template, parser, context or resource checks. In particular, an unreviewed
quantized export may be discoverable without an admissible guide score.

Backend support also does not establish a guided memory plan. Gemma 4 currently
remains outside guided recommendations because its route-specific memory
layout is not covered by the guide planner. Its Advanced conversion support
does not authorize adding it to the dense planning allowlist. GLM Flash and
Flint currently have no matching cached Coding parent score in the 128 GiB
snapshot, so neither can displace a scored Quality choice on that evidence.
The built-in Qwen3.6-27B catalogue record also has no adapter and retains
`conversion: planned` / `qualification: pending`; the catalogue excludes that
record before guided ranking. Search prioritization does not promote that
checkpoint or replace its missing onboarding evidence.
The [November 13 investigation](archive/runs/2026-11-13-guided-alternatives.md)
records the screenshot's score gap, live aggregate memory checks, partial
metadata scans and the separate qualification/planning blockers.

The [RAM-scaled Speed check](archive/runs/2026-11-11-speed-recommendation-policy.md) records the current primary choices, exact source identities, hardware assumptions and model-free regression results.

### Catalogue ordering

**Model Catalog** uses the frozen coding-index snapshot to order a family's rows. This is a
browsing aid, separate from the Beginner selection policy above: it never
selects, installs or admits a model, and every row still passes the catalogue's
own compatibility checks. `catalog_ranking.py` attaches a reference score to a
row only through one of these relationships, which the row records:

| Relationship | Rule |
| --- | --- |
| `benchmarked_model` | The row's repository is the scored repository. |
| `reviewed_export` | A reviewed export-to-model binding matches the row's exact repository and revision. |
| `declared_quantization` | The row is a quantized format and its Hub tags declare exactly one `base_model:quantized:` parent, which has a coding index. |
| `named_quantization` | The row is a quantized format whose name is a scored model's name plus a quantization suffix that agrees with the checked format. The scored model must share the row's publisher, or be the only scored model with that name. |

The direct model row receives its model-level index. Declared/named
quantization rows and reviewed exports may inherit the referenced model's
index under the relationships above; those relationships rest on source
metadata or review and do not show that the export was benchmarked. Other
fine-tunes, merges and re-uploads remain unscored. Unindexed rows sort after
indexed rows. Model Catalog exposes **Top rated (coding)** only. The Top Models
tab separately displays family-level aggregates and is unchanged.

The November 14 coding-index snapshot is frozen. Its inputs, source revisions,
hashes, attribution, licenses and calculation are documented in the linked
benchmark report and aggregate JSON. Refreshes require a separately reviewed
calculation and explicit score-identity updates; do not infer scores for new
models or families.

`tools/converter/scripts/bundle_llmbench_coding.py` packages a reviewed aggregate
with explicit repository identities from a saved llmbench index. It preserves
the aggregate's scores and source hashes and records the identity index's date
and hash separately. Run it with `--aggregate`, `--index` and `--output`; it does
not fetch data or recompute scores. The shipped identity mapping uses the index
generated at `2026-11-15T02:21:26Z`; score inputs remain frozen at
`2026-11-14T18:14:50Z`.

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
