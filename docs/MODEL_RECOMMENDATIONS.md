# Beginner recommendation evidence

Beginner asks for **General Work** or **Coding**, followed by **Speed** or
**Quality** with Quality selected by default. General Work continues into the
OpenWork guide; Coding continues into OpenCode. Recommendations first check
resource fit, then use the offline evidence policy below. Parent-model scores
do not measure Stark exports or tool workflows.

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
pressure. The guide retains the 2 GiB physical reserve, but the current or
recommended GPU working-set value does not filter or rank recommendations.
Recommendations use `recommendation_allowed`: installed physical RAM and native
device capability determine fit. Current free disk and available memory do not
filter or reorder offers, including when those probes are missing. The card
shows a specific setup-space warning and up to two disk-ready smaller options
when the primary choice cannot be installed immediately. Actual preparation
still checks exact free space and retains a separate 2 GiB current-memory
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

## Offline llmbench references

`llmbench_catalog_ranking.json` bundles llmbench's published model-level coding
and non-coding summary scores from the November 17, 2026
[summary-score snapshot](https://llmbench.site/summary/). General Work uses the
non-coding score; Coding uses the separate coding score.

The recommendation path does not request llmbench at runtime. Hardware checks
and supported-model discovery remain fresh; Hugging Face metadata lookups are
separate. Clicking a source link intentionally opens the browser.

### Selection policy

1. Exclude candidates without a complete supported conversion/native route, usable chat template, tool parser, 32K context and adequate installed unified memory. GPU availability and unified-memory architecture are device capabilities; current GPU working-set usage and the recommended working-set budget do not gate or rank recommendations. Current free disk and available memory affect install/start readiness, never recommendation eligibility or rank. Leaderboard rank never grants compatibility.
2. Match exact parent repositories from the pinned source catalogue. Reviewed export-to-parent bindings also require the exact export repository and revision. No fuzzy names, quantization labels or mutable commands establish identity. Ambiguous parents receive no score. A changed export revision requires another review.
3. **Quality** selects descending llmbench summary score for the selected task after admission. Alternatives must be within **10 points** of the best admitted model and each offer must represent a distinct parent. Unknown scores never fill alternatives behind a scored choice. With no comparable llmbench evidence, retain the direct benchmark/resource fallback.
4. **General Work** uses llmbench's **non-coding** score. **Coding** uses its separate coding score; neither is substituted for the other.
5. **Speed** requires the selected task's parent-model llmbench score to be at least **35** and exact export weights no larger than **25% of physical RAM**. It selects descending score, then smaller runtime and payload for ties. The same 10-point and distinct-parent rules apply.
6. Offer one primary recommendation with up to two alternatives behind a disclosure. Every inherited score is labeled as parent-model reference evidence; its quantization is not claimed equivalent in quality.

Stark's 32K model-specific plan, actual export weights and state remain
authoritative. The score never authorizes a backend, conversion route or native
qualification. No generic bit-width quality penalty is invented.

If the snapshot is unavailable or malformed, Coding Quality retains its direct
benchmark/resource fallback and Speed reports no qualifying recommendation.
`recommendation_benchmarks.json` retains the separate IFEval, BFCL and
LiveCodeBench records. Former task/preference API values retain their legacy
behavior for older clients.

### Guided discovery coverage

Guided discovery checks a bounded metadata window, not every repository. It
prioritizes sources with the selected task's llmbench summary score before
inspecting that window;
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
does not authorize adding it to the dense planning allowlist. Flint has no
matching scored parent for the task, so it cannot displace a scored Quality choice
on llmbench evidence.
The built-in Qwen3.6-27B catalogue record also has no adapter and retains
`conversion: planned` / `qualification: pending`; the catalogue excludes that
record before guided ranking. Search prioritization does not promote that
checkpoint or replace its missing onboarding evidence.
The [November 13 investigation](archive/runs/2026-11-13-guided-alternatives.md)
records the screenshot's score gap, live aggregate memory checks, partial
metadata scans and the separate qualification/planning blockers.

The [RAM-scaled Speed check](archive/runs/2026-11-11-speed-recommendation-policy.md) records the current primary choices, exact source identities, hardware assumptions and model-free regression results.

### Catalogue ordering

**Model Catalog** uses the same frozen llmbench coding score from the official
summary snapshot. The beginner flow also carries the matching non-coding score.
This is a browsing aid, separate from the Beginner selection policy above: it
never selects, installs or admits a model, and every row still passes the
catalogue's own compatibility checks. `catalog_ranking.py` attaches a score
to a row only through one of these relationships, which the row records:

| Relationship | Rule |
| --- | --- |
| `benchmarked_model` | The row's repository is the scored repository. |
| `reviewed_export` | A reviewed export-to-parent binding matches the row's exact repository and revision. |
| `declared_quantization` | The row is a quantized format and its Hub tags declare exactly one `base_model:quantized:` parent, which is scored. |
| `named_quantization` | The row is a quantized format whose name is a scored model's name plus a quantization suffix that agrees with the checked format. The scored model must share the row's publisher, or be the only scored model with that name. |

Full-precision rows other than the scored repository itself never inherit a
score: a fine-tune, merge or re-upload has its own weights. The last two
relationships rest on what the export's publisher states, not on a review, so
they order rows and nothing more. The app labels every inherited score as the
parent model's and says that the quantized version was not benchmarked. The
displayed value is llmbench's coding summary score. Missing scores remain
unknown and sort after scored rows. A parent with more than one snapshot entry
is ambiguous and has no score. This frozen llmbench evidence is
also used by Beginner Coding choices.

### Manual refresh

There is no runtime refresh. A maintainer reviews a new llmbench capture,
retains its source hashes and method, regenerates the approved package subset,
and records a separate dated validation. Changing the score snapshot does not qualify a
model or supersede native validation.

The llmbench compilation is CC BY-SA 4.0 and incorporates whatmodel.sh /
Merlinsbeard Pty Ltd data. The package retains upstream attribution and
license notices; those data terms do not relicense Stark application code,
model weights or trademarks. Sources do not endorse Stark.

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
