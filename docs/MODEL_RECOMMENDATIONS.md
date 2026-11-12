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

## Offline leaderboard references

`recommendation_leaderboard.json` bundles a dated [whatmodel.sh](https://whatmodel.sh/) snapshot with separate [General Work](https://whatmodel.sh/?gpu=apple&max=0) and [Coding](https://whatmodel.sh/?gpu=apple&max=0&task=coding) rankings. General Work uses the requested all-task configuration. Both lists select Apple Silicon and local models only. Full lists cover 8, 16, 24, 32, 48, 64, 96, 128, 192, 256 and 512 GB RAM. Exact query URLs, response SHA-256 values, update times, source revision and model-catalogue hash remain in the snapshot. The [November capture](archive/runs/2026-11-11-whatmodel-recommendations.md) records the top three models per RAM/task list, including models Stark cannot run.

For benchmark information behind these rankings, start with whatmodel’s [About this repo](https://github.com/jakemanger/whatmodel#about-this-repo) section, which links to its scoring methodology and benchmark sources. The bundled snapshot retains the source revision used for Stark’s recommendations.

The recommendation path never requests whatmodel.sh. Hardware checks and supported-model discovery remain fresh; existing Hugging Face metadata lookups are separate. Clicking a source link intentionally opens the browser.

### Selection policy

1. Exclude candidates without a complete supported conversion/native route, usable chat template, tool parser, 32K context and adequate current RAM, GPU budget and disk. Leaderboard rank never grants compatibility.
2. Select the largest cached RAM bucket no greater than this Mac's physical RAM in GiB. Below the smallest bucket, cached coverage is unavailable; above the largest, use the largest bucket with actual resource checks.
3. Match exact parent repositories from the pinned source catalogue. Reviewed export-to-parent bindings also require the exact export repository and revision. No fuzzy names, quantization labels or mutable commands establish identity. Ambiguous parents have no automatic score. A changed export revision requires another review.
4. **Quality** puts covered candidates first by descending task score. Equal scores retain deterministic fallback order. Missing scores stay unknown and follow scored candidates. Signed relative scores are not percentages or native artifact accuracy measurements. Parent scores do not establish export equivalence.
5. **Speed** admits only candidates with a cached parent-model task score of at least **40 for General Work** or **35 for Coding**, and exact export weights no larger than **25% of physical RAM**. All normal runtime, GPU, free-memory and disk checks still apply. Within that pool, select descending task score; ties prefer a smaller runtime plan, then smaller weight payload, then deterministic fallback order. The increasing payload allowance lets larger Macs select stronger models without always choosing the smallest eligible checkpoint. Unknown scores, candidates below the floor and exports over the budget are excluded from all Speed alternatives. These are editorial product thresholds on separate relative score scales; review them when refreshing the leaderboard. They do not certify quantized task quality or measured response speed. Source speeds retain their provenance but do not rank these candidates.
6. Offer one primary recommendation with up to two alternatives behind a disclosure. Historical benchmark results remain reference information when the cached leaderboard determines ordering.

The site's memory/variant estimates use an 8K conversation allowance. Stark's 32K model-specific plan remains authoritative, including actual export weights and state. The captured site's ranking applies the same parent score across quantized variants; it does not benchmark each quantization separately. Speed's budget uses actual export weight bytes, so a quantized export may fit while its BF16 counterpart does not. Its parent score does not establish that the export preserves quality. The guide labels this limitation, and structured evidence explicitly marks quantization quality as unevaluated by this ranking. No generic bit-width quality penalty is invented. Site quantizations remain provenance and are not automatically installed. Snapshot inclusion adds no backend, conversion route or native qualification.

If the snapshot is unavailable, malformed or covers none of the eligible models, **Quality** retains its offline benchmark/resource fallback. **Speed** reports no qualifying recommendation, with a direct **Choose Quality** action and Advanced browsing; it never waives the score floor or weight budget. `recommendation_benchmarks.json` retains its source-bound IFEval, BFCL and LiveCodeBench records separately. General Work's Quality fallback uses instruction-following references; Coding uses the published coding cohort. Former task/preference API values retain their legacy behavior for older clients. New guides send `general`/`coding` and `quality`/`speed`.

The [RAM-scaled Speed check](archive/runs/2026-11-11-speed-recommendation-policy.md) records the current primary choices, exact source identities, hardware assumptions and model-free regression results.

### Manual refresh

There is no scheduled refresh. A maintainer reviews and commits a snapshot about monthly. The utility accesses the network only with explicit `--fetch`. Start from a reviewed upstream source checkout and record its commit. From the Stark repository root:

```sh
uv run --project tools/converter --locked --python 3.13.13 python tools/converter/scripts/refresh_recommendation_leaderboard.py --fetch --raw-dir /tmp/whatmodel-new-capture --source-dir /path/to/whatmodel --source-revision REVIEWED_COMMIT --captured-date YYYY-MM-DD --report docs/archive/runs/YYYY-MM-DD-whatmodel-recommendations.md
```

The utility requests both configurations for each RAM bucket, with `format=json&n=1000` to avoid default truncation. Omit `--fetch` to rebuild from saved responses and their hash manifest. Review changed identities, scores, coverage and notices; retain earlier dated reports. Run recommendation, native setup and packaging checks before pushing. Leaderboard changes do not qualify a model or supersede earlier native validation.

The numerical compilation and derived scores retain whatmodel.sh / Merlinsbeard Pty Ltd's CC BY-SA 4.0 terms. Changes here select fields and RAM/task lists and map explicit parent repositories. Source-specific attributions and full notices ship in `stark_convert/_third_party/whatmodel/`. Those data terms do not relicense Stark application code, model weights or trademarks. Sources do not endorse Stark.

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
