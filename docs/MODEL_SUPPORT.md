# Model-family support and installation preflight

Stark uses one family registry for conversion and native-preparation coverage, intended application
modes and early installation checks. The converter CLI and **Model Catalog**
tab in the menu bar app's **Manage Models** window consume that registry. A family
entry describes implemented paths and constraints; a source preflight checks selected checkpoint
metadata before weights are downloaded.

The [native hybrid path](NATIVE_HYBRID_MODELS.md) loads prepared
GLM-5.3-Flash UD-Q2_K_XL, Qwen3.8-Flash-Next mixed MLX and Flint checkpoints.
Its helper pins native-source downloads for local import. **Model Catalog →
GLM-5.3 Flash** offers the pinned `UD-Q2_K_XL` checkpoint through that same
preparation path. GLM is a separate native family in the inventory; arbitrary
GLM repositories remain outside generic converter preflight. The registry also
supports two `qwen4_exp` text import paths. Lily-format sources can be
converted to a typed `Qwen38LanguageModelConfig` artifact with canonical MLX
storage and unchanged mixed quantization. This conversion omits vision and MTP
tensors. Compatible external-table MLX-Serve sources are prepared for the
native loader with their original weight shards, index, and n-gram table.
With **Enable images** selected, Add Source prepares compatible retained
Flash Next affine checkpoints through the shared native image backend. Admission
uses the declared architecture, actual tensor packing, complete BF16 tower and
supported processor/template programs. It does not require a repository entry
in the catalog. Immutable source provenance and content hashes remain bound to
each installed artifact; automatic local image execution checks must pass before
registration. See [retained image preparation](NATIVE_HYBRID_MODELS.md#retained-lily-image-artifacts).
Sources without a processor sidecar use the validated standard family recipe;
generated processor provenance is recorded separately. Explicit unsupported
processing programs, other weight formats and quantized towers remain outside
this contract. Family recognition alone does not establish execution or image
quality for an arbitrary export.

The ordinary/catalog GLM GGUF path keeps MTP unavailable. The optional GLM EXL3/FP8
backend has an unchanged `headless-reference` default and a separately staged
`--profile bf16-native-experimental` profile that includes the BF16 draft head,
native MTP, and prefix support. Only that profile defaults to three drafts; requests
can explicitly disable MTP. Its evidence remains experimental: the worker
matrix passed 41/41, while the quality smoke passed 22/24 and missed its
all-pass gate. The 922.7/20.3 tok/s observation failed host screening and had
unverified clocks, so it does not qualify production throughput. See the
[backend reference](validation/GLM_EXL3_BACKEND.md) and the
[Uzu-format study](benchmarks/QWEN38_UZU_FORMAT_2026_10_29.md) records storage,
readout and sampled-generation controls for the pinned converted artifact.
The native loader validates tensors; generic Decision use still requires a
direct-answer template and checkpoint checks.

This native GLM path covers Flash's `glm5_next` architecture. Full GLM-5.3 uses
`glm_moe_dsa` and needs separate backend onboarding. A
[dated recent-model intake](archive/runs/2026-11-07-glm-flint-tool-qualification.md#recent-release-intake)
also records the unsupported architecture contracts for DeepSeek V4.1, Kimi
K3.1, Beam, Nemotron 3.5 Bolt and Gemma 4. Metadata discovery does not provide
native execution or tool qualification for those models.

Dense `qwen3_5` text conversion also admits the registered
`Jundot/Qwen3.8-27B-oQ4e-mtp` extension. Its mixed affine 4/5/6/8-bit matrices
are preserved without requantization. Uniform five/six-bit matrices retain
their packed bytes when they use group size 64, BF16 and output-input layout.
Eligible mixed-width fused projections use `MLXRowPartitionedSpec` to retain
contiguous output-row segments at their original widths; runtime planning keeps
the original fused output dimension and W8 row arithmetic. Other layouts,
including the input-output embedding, are widened losslessly to eight-bit
storage.

A complete, declared one-layer quantized MTP head becomes a typed native graph.
Dense Qwen MTP supports up to three drafts, recommends three and remains
disabled by default. The optional three/four-bit coarse draft readout uses a
64-candidate shortlist rescored by the original head; omitting it keeps the full
draft head. The target readout remains complete in every mode. These additions
do not transfer qualification from Qwen3.8-Flash-Next. See the
[original dense 27B record](benchmarks/2026-11-06-qwen38-27b-oq4e-mtp.md) and
[packed-weight follow-up](benchmarks/2026-11-06-qwen38-27b-packed-followup.md).

Converter inspection and catalog admission alone do not establish that a
complete model has loaded or executed. Exact
load qualifications, reference comparisons and local image-check receipts retain
their existing artifact/build/runtime identities. The loaded worker remains the
authority for live API and image capabilities.

The family registry lists uniform MLX affine 4-bit and 8-bit sources, imported
without requantization, for Qwen3, Qwen2, Llama, Mistral, Gemma 3 text, Gemma 4
text, OLMo 3, SmolLM3, Qwen3.5, Qwen3.5 MoE, Qwen3.8 Flash Next, LFM2 and
LLaMBA. Each family's tensor planner still applies its own layout limits.
Per-module overrides and five/six-bit widths remain limited to the families
that document them.
Qwen2 attention biases stay separate full-precision vectors beside the packed
projection. Granite, Gemma 2 and Phi-3 reject quantized sources. The Qwen2 and
Mistral MLX paths are not numerically qualified
([20261112-DENSE-MLX-01](archive/runs/2026-11-12-dense-mlx-qwen2-mistral.md)).
Their F16-parameter sources converted to the default BF16 miss the stock-MLX
logit gates, and a named chat-template list blocks conversion
([20261112-DENSE-MLX-02/03](archive/runs/2026-11-12-dense-mlx-qwen2-mistral-qualification.md)).
Converted with `--dtype float16`, the `mlx-community/Qwen2.5-0.5B-Instruct-4bit`
export and a template-substituted diagnostic copy of the Mistral 7B v0.3 4-bit
export pass the logit, greedy and scorer gates; the published Mistral revision
still fails conversion
([20261112-DENSE-MLX-04/05](archive/runs/2026-11-12-dense-f16-native-qualification.md)).

Native F16 is a separate precision profile for the plain full-attention dense
graph of Llama, Qwen2 and Mistral artifacts. The engine selects it only when
the converted decoder stores F16 tensors and no BF16 tensor. Weights,
embeddings, norms, biases and MLX affine scales and biases stay F16, and packed
affine codes are unchanged. Activations, the KV cache and the readout are F16.
Matrix products, affine dequantization, normalization, RoPE and attention
accumulate in F32, as on the BF16 path. Query/key norms, sliding windows,
soft caps, sinks, gates, per-layer embeddings, image embeddings, MTP heads and
speculators have no F16 path: such an F16 artifact fails to load with an
unsupported-precision error. BF16 artifacts keep their existing policy, and BF16
qualification does not transfer to an F16 artifact or the reverse.

Qwen3.5 MoE image conversion has two schema-3 profiles: affine-quantized source
weights use `qwen35-moe-affine-merged-v4`, while complete BF16 source weights
use `qwen35-moe-bf16-merged-v5`. Both produce BF16 artifacts; the BF16-source
route is image-only. Catalog discovery requires the source's
supported processor and reviewed template before offering that mode; BF16 MoE
text/Decision conversion remains unsupported. This is structural implementation
coverage. Full official 35B execution and reference parity are unverified; each
artifact still needs native admission and local image checks. The
[vision family contract](VISION_FAMILY_CONTRACT.md) records the numerical policy,
source layout, limits and disabled prefix-reuse disposition.

New model and inference paths follow the [model onboarding workflow](MODEL_ONBOARDING.md),
including generation prefix-reuse implementation where exact restoration is
possible and a recorded blocker where it is unavailable. The family registry
does not grant this runtime capability or transfer qualification between
checkpoints and quantizations.
[OLMo 3 and Gemma 4](OLMO3_GEMMA4.md) use the ordinary native Metal text path.
OLMo 3 adds whole-projection Q/K normalization and post-norm residual branches.
Gemma 4 text paths include dense PLE E2B/E4B and bounded no-PLE 31B layouts.
Pinned schema-4 BF16 image profiles cover E2B, E4B, 12B unified, 26B-A4B and
31B. Audio and video execution remain unsupported. E2B has a canonical HF CPU
image-feature comparison within the 2% gate on three fixtures plus bounded
worker and chat-client lifecycle checks; E4B has separate source and execution
validation. The 12B embedder and 26B MoE paths have component checks. The pinned
31B tower/projector passes all three unchanged canonical feature gates, with
relative RMS at most 0.00110689. Full 12B/26B/31B checkpoint generation is
unqualified. Installed desktop UI execution and image performance are
unmeasured. Exact checkpoint evidence remains tied to its reference profile and
engine build. See the
[original E2B validation record](archive/runs/2026-10-28-olmo3-gemma4-native.md)
and [31B vision component record](archive/runs/2026-11-07-gemma4-larger-vision-qualified-component.md).

## Browsing and removing models

The app's **Manage Models** window has **Installed**, **Model Catalog** and
**Add from Source** tabs. Buttons in the menu and Playground open it to select and
run models; Command-I selects Add from Source in the same window. The install form
keeps its source and options when switching tabs. Active progress and cancellation
remain accessible, and completion stays in management. Search filters the
current view immediately and does not trigger another Hugging Face request.
The menu and Playground show the running private version's saved name,
quantization and stored weight-file size. Missing metadata is omitted.
Search text is transient; sort choices persist independently for **Installed**,
the family overview and family detail. The saved-model view sorts by name A–Z or
Z–A, or total disk bytes from its visible versions ascending or descending. The
family overview sorts by name A–Z or Z–A; family detail sorts by name in either
direction or model size ascending or descending. Unknown numeric values sort last
in either direction. Choose a family to browse compatible models, sizes and
quantizations and download a choice directly.

Family detail opens on **Top rated (coding)**, with **Top rated (general)**
beside it. Both use the bundled whatmodel.sh reference scores described in
[Beginner recommendation evidence](MODEL_RECOMMENDATIONS.md#catalogue-ordering).
Rows that can be downloaded and whose weights fit this Mac's memory budget come
first, then the higher score, then the smaller download; unscored rows follow
scored ones, then models likely too large for this Mac, then unavailable rows.
A group is placed by its best row. A scored row shows its score as a link to the
source page, and its tooltip names the model the score belongs to. A row whose
weights exceed 85% of physical memory less 2 GiB is marked as likely too large;
that is a coarse check on weights alone, not a runtime plan. The catalogue also
inspects sources with a reference score before the rest of the first 300
listing positions, preferring models from the RAM list this Mac can use, so the
first pages hold the best-rated candidates. A saved sort choice is kept.

**Hide Unsupported** is checked by default and the choice persists. It leaves
out every row that cannot be downloaded: a weight format or quantization option
Stark does not import, a tensor layout the family's importer rejects, a source
without a usable Safetensors file or shard index, and a source whose tensor
headers could not be verified. Quantization tabs and counts follow the listed
rows. The list states how many rows are hidden, and when every row found so far
is hidden it offers to show them. Unchecking it lists those rows as
**Unavailable** with their reason.

A GPU-server quantization export (AWQ, GPTQ, bitsandbytes and similar) that
declares the model it was made from is folded under that model when the model
is a downloadable row among those loaded. A folded export is never a row of its
own and is not counted as hidden. With **Hide Unsupported** unchecked, the
original's group names its folded exports, each linked to its repository, up to
six with a count of the rest. Searching for a folded export's name, repository
or format shows the original with the same note, so a search for an AWQ build
lands on the model Stark can import. An export whose original is not loaded, or
is itself unavailable, stays an **Unavailable** row and links to the declared
original on Hugging Face. Folding follows the rows loaded so far, so a row can
fold when a later page brings in its original.
Conversion-only families are omitted. Cards show whether each model supports
text or text and images. Downloads automatically include image support when
available for that artifact. This requires metadata confirming a supported
Qwen3.5 vision wrapper, BF16 vision tower and reviewed processor; decoder weights
may use a supported affine profile. Image support still needs the normal
installation and local runtime checks.
The download and preparation progress remain in the browser. Completion adds
the model to the library without selecting it for use.

Family cards group model sizes and quantizations into compact variant rows.
Each family opens on **All models**, followed by tabs for the quantizations and
precisions present in its current catalogue. Tabs come from the full current
family list; search and sort narrow/reorder the selected tab without refetching
metadata. If a selected format disappears after refresh, the view returns to
**All models**. Switching families also resets that filter. The row's source ID,
download status and image capability remain unchanged by filtering.
Same-publisher size and quantization variants share a card. Base, Instruct,
Thinking, context-length variants and fine-tunes remain distinct. A cross-publisher
group is allowed only for a registered MLX 4-bit/8-bit export whose tokenizer
configuration points to a registered same-family base and whose basename matches
after removing the quantization export suffix. An optional `model_group` descriptor
provides its 32-character `id`, display `name` and `publisher`; it is presentation
metadata and does not change the immutable source entry ID. Family-size sorting
uses the minimum or maximum known model size among the visible versions. Counts
are never inferred from model names or quantized packed tensors.

The download catalogue combines registered reference sources with bounded live
Hugging Face searches. Architecture tags and model-name searches are interleaved
so a large tag cannot delay every other export type. Parsed configuration
identifies the family. Each page inspects up to 30 distinct candidates and
retains unsupported family variants with an **Unavailable** reason.
**Load more models** follows the remaining pages automatically, preserving
results already loaded. Failed pages can be retried at the same offset.

The Hub listing itself carries each repository's head revision, task tags and
file names. A source whose listing already shows another task, no model
configuration or no weight file uses its position without any request of its
own, so one page can pass up to 500 such positions and reach further down the
family. Metadata files are bounded by the sizes in the pinned file inventory
instead of a separate size request, and the Hub's parsed architecture rejects
an unsupported head revision before its configuration is fetched.

Rows appear as each source is checked; the page's sorted report follows. The
app keeps verdicts in `catalog-cache` inside its data folder, keyed by
repository and bound to the checked revision and the installed converter
sources. An unchanged revision is answered from that cache without contacting
the repository: for 30 days, or one hour when the outcome depended on access
(gated or missing files, configuration borrowed from another repository). A
registered source has no listed revision, so its head is confirmed with one
request and then trusted for ten minutes. Verdicts that could not be verified
are never stored. The cache holds public repository metadata only, and
installing repeats the pinned preflight.

Before enabling Download for a generic Safetensors source, the catalogue checks
the pinned tensor headers with the same tensor planner used by conversion.
Names, shapes, dtypes, affine packing and shard-index membership must match the
importer. For a checkpoint with a shard index, one Hub API call returns every
shard's parsed header; it is used only when the Hub labels it with the pinned
revision and each header agrees with the pinned file sizes. A single-file
checkpoint, a moved or older revision, or any reply that fails those checks is
read from the pinned files with exact HTTP byte ranges instead, and every
failure reason comes from those bytes. Header requests never consume weight
payloads and reject servers that ignore the requested range. A header exceeding the
catalogue's 4 MiB inspection limit remains unavailable with an explanation.
Duplicate tied-tensor payload equality is checked during full conversion.
Native loading and model execution remain separate checks.

Each family planner admits an explicit set of `config.json` keys. Before a
planner validates a configuration, one shared pass
(`stark_convert/config_normalization.py`) rewrites a small set of reviewed
forms that do not change the model. The planner applies it on the line that
reads its configuration, so catalogue admission and the source-backed
conversion that installs the model always judge the same normalized
configuration. The source `config.json` is never modified. The pass:

- drops publisher annotations and sampling defaults when their value has the
  expected shape: `unsloth_fixed`, `unsloth_version`, `transformers.js_config`,
  `model_name`, `generation_config`, `do_sample`, `top_p`, `top_k`,
  `temperature`, `is_llama_config` and `init_method`. A malformed value is
  refused rather than dropped;
- drops `rope_interleaved` only when it is `false` and `rope_scaling` only
  when it is `null`;
- renames `dtype` to `torch_dtype` when `torch_dtype` is absent;
- sets a nonzero `attention_dropout` within (0, 1] to 0.0. Dropout is inactive
  at inference. This is a separate judgement call that
  `ZERO_INACTIVE_ATTENTION_DROPOUT` turns off on its own;
- for planners that read a single flat `rope_theta` and reject a nested
  declaration (Qwen2, Mistral, Gemma 2, OLMo 3, Phi-3, Granite and SmolLM3),
  flattens a plain `rope_parameters` object holding only `rope_theta` and an
  absent or default `rope_type`. Qwen3, Llama and LFM2 already validate that
  plain form themselves and see it unchanged;
- for Gemma 2 and Gemma 3 text, drops `sliding_window_size` when it is an
  integer equal to the required `sliding_window` the planner uses;
- for LFM2, drops `tie_word_embeddings: true` unless `tie_embedding` is
  `false` (the planner reads `tie_embedding`, default tied), and sets an absent
  `block_ff_dim` from a positive integer `intermediate_size` (the Transformers
  `Lfm2Config` stores one as the other).

Gemma 4 and Qwen3.5 also apply the generic rules to their nested
`text_config`. Everything else still fails closed in the planner: unknown keys,
two differing declared RoPE bases, a plain `rope_parameters` beside a non-null
`rope_scaling`, scaled, YaRN, linear, partial-rotary or per-attention-type RoPE
forms, `rope_interleaved: true`, both `dtype` and `torch_dtype` declared, and
dropout outside its valid range. The Gemma 3 image wrapper, the external
Qwen3.8 profiles, LLaMBA and ModernBERT keep their configurations unchanged.
Image-mode contract validators that read the configuration separately are not
affected by this pass.

An **Unavailable** row states one reason, chosen in this order:

- **GPU-server quantization.** A `quantization_config` whose `quant_method` is
  anything other than MLX (for example AWQ, GPTQ, bitsandbytes, FP8,
  compressed-tensors, ModelOpt or AutoRound), with no MLX `quantization` block,
  is named as that format. The row's quantization label is the method name
  only, never a guessed bit width, and the reason says Stark imports the
  model's original weights or its MLX exports. This takes precedence over
  every other check, and no tensor header is requested for such a row. The
  rule applies only where the import-format check already rejects the source,
  so it never changes which rows can be downloaded. An MLX export, which
  writes a `quantization` block and usually a matching `quantization_config`,
  is not affected. When the Hub tags of such a row carry exactly one
  `base_model:quantized:<repository>` entry naming another valid repository,
  the row also carries that repository as its declared original. The tag comes
  from the export's own model card and is not verified; it is used only to
  place the row in the browser. The named repository is never requested on
  the export's behalf, and the pointer never selects or alters a download.
- **Configuration or format checks.** Preflight, quantization parsing and
  container checks report their own message, such as an unsupported MLX
  option, untied embeddings or a missing shard index.
- **Importer rejection.** When the tensor planner or header inventory rejects
  the checkpoint, the row shows the importer's own message, for example
  "RoPE scaling is outside qualified dense coverage." These are fixed
  messages written in the importer that name config fields and rules, never
  file paths or tensor contents. A message is cut to the 240-byte reason
  limit; one with control characters, and any unexpected exception type,
  falls back to the general statement that tensor names, shapes or packing do
  not match the family's importer.
- **Access approval.** When the Hub marks a repository as gated and its tensor
  headers cannot be read, the row asks the user to sign in to Hugging Face,
  accept the model's terms and refresh. Like an unverified header read, this
  verdict is never cached, so a refresh after approval checks the source
  again. A gated source whose configuration cannot be read is still left out
  of the list.
- **Header limits.** A header beyond the 4 MiB inspection limit keeps its own
  reason, and any other header read failure on a source that is not gated
  still asks for a refresh.

GLM and Flint searches also display other family exports with their import
limitations. Their supported native installers currently bind exact artifacts.
GLM's downloadable `UD-Q2_K_XL` GGUF entry keeps its source revisions, four
shards and 108,720,071,552 stored weight bytes owned by the preparation helper.
Flint keeps its pinned MLX affine INT4 checkpoint. These entries use bundled
metadata, while discovery of other variants uses the Hub. Download preserves
quantized bytes and validates the prepared identity before library registration.
Native downloads show a progress bar for completed model files against the pinned
weight total, with received network bytes separately. Partial allocations never
advance the bar; final weight validation still gates completion.
GLM image input and Decision mode are unavailable.
Known Gemma 4 image choices use the immutable revisions bound by their image
import profiles. Later repository revisions require their own image admission.
Lily Qwen3.8 uses a mixed `4-bit / 8-bit` label, including
its fixed Q8 control projections. The Qwen3.8 catalog also discovers compatible
MLX-Serve external-table exports by config and file inventory, including
importance-calibrated exports using the supported affine layout. Their download
size includes the external n-gram table. The catalog checks the pinned shard
headers, index, and n-gram table layout before enabling Download. Incompatible
external-table candidates display an explanation in their catalog row.
Embedding/reranking tasks, known internal test models and adapter-only
repositories are excluded. Family variants with unsupported decoder geometry,
quantization or containers retain their rejection reason. For example, Gemma 4
quantized no-PLE global K=V exports explain their BF16 import requirement, and
per-module affine formats explain the importer's quantization limit.
Qwen image profiles can retain supported packed decoder matrices
alongside a complete BF16 vision tower. The
implemented Qwen3.8 text path includes its MoE graph; Qwen3.5 BF16 MoE image
conversion is a separate image-only path.
An uncatalogued compatible fine-tune can appear through its architecture tag or
model name. Gated, inaccessible and malformed repositories may be omitted.
An untagged repository whose name has no family search term can still require
direct import.

Discovery reads bounded metadata, tensor headers and repository file information.
Parameter counts and
download bytes appear only when usable metadata supplies them. Floating sources
are shown as BF16, the native output format; packed tensor counts are not shown
as logical parameter counts. Missing metadata can leave a size unknown or omit
a candidate. Offline and failed discovery have fixed messages and can be retried.
Guided setup continues past pages with no downloadable entries within a shared
10-second budget and a maximum of ten pages per family. It reports when discovery
remains incomplete. Family budgets are checked between network operations.
The desktop host enforces cancellation and the overall child-process deadline.

**Installed** groups saved versions by their bound,
unambiguous source repository; local or incomplete metadata falls back to a
standalone entry with its stored name. Compact rows place the saved name,
quantization, stored weight-file size, repo link, Run/Stop and removal controls on
one line. Run resolves the current saved private ID and loads it in the current
app mode while leaving management open. Stop identifies the active model and
moves when another model starts. It can cancel loading or an active request;
row controls reject stale IDs and unrelated busy operations. The active version
is pinned in a separate section above other models. Search filters both sections;
the chosen sort applies to remaining groups after excluding the active version.
Closing the
management window cancels catalogue
browsing without stopping the loaded model.
Multiple versions retain a shared model heading. Row help retains the family,
image support and full source revision when known; removal confirmations also
include image support and the short source version. Matrix representation is
read from bounded native Safetensors header metadata; normal floating-point
convolution/control exceptions do not change the quantized matrix label. Receipt
identity is used only when its source-manifest schema/digest, current
configuration hash and declared output size match. No tensor payloads are read
for this display. Family labels must also agree with the bound generic profile
or source catalog adapter. The actually loaded
model is marked as running. Browsing does not select or load models or grant
runtime capabilities. The full registry and its operation constraints remain
available through the CLI; the app does not label families as decision models.

Each saved version has a removal action with a confirmation. An exclusively
owned managed folder offers **Move to Trash**. Imported, shared or otherwise
unowned paths offer **Remove from Stark**, leaving files untouched. The backend
rechecks the offered action and directory identity before moving files. A failed
registry update restores the library and attempts to restore the moved folder.
Removing the loaded model stops its worker; removing another model preserves
the active worker. Local qualification receipts retain their separate identity.

## Modes and support states

The registry uses these mode identifiers:

| Mode | Intended use |
| --- | --- |
| `text` | Text chat through a converted language artifact and the generation worker. |
| `decision` | Generic answer-choice scoring through a text artifact, compatible template/tokenizer and scoring worker. |
| `vision-language` | Image chat through supported full vision artifacts and model-specific host checks. Current paths include the Qwen3.5 schemas and Gemma 4 E2B BF16 schema 4. |

Decision scoring uses temperature 1 and produces a comparative distribution over
the supplied choices. A compatible family does not establish calibration or
inherit a model's published decision qualification. Template generation-prefix,
reasoning-state and unique single-token answer-letter checks still apply.
Vision-language artifacts remain Chat-only; a raw source may separately support
conversion to text and vision artifacts.

Matrix modes report `conditional` or `unsupported`; a source report can also use
`inconclusive`, including for individual modes when the source family could not
be identified:

| State | Meaning | Installation behavior |
| --- | --- | --- |
| `conditional` | An implemented path exists for the inspected metadata or family constraints. Full tensor, native and mode-specific checks remain. | Continue automatically to the selected conversion path. |
| `unsupported` | The selected path has a known implementation or metadata incompatibility. | Stop before weight download and explain the reason and other modes to check. |
| `inconclusive` | Required metadata could not be inspected sufficiently. | Stop before weight download and report that compatibility could not be verified. |

Absence from the checkpoint catalog does not make a compatible fine-tune
unsupported. A source whose parsed architecture has no registered implementation
is unsupported for the selected path. Missing access, unavailable metadata and
network failures do not prove an architecture incompatibility. Recognizing a
family alone does not validate its required fields or operations. The early
check detects selected metadata-proven exclusions; complete per-adapter
validation still follows.

The matrix also records conversion containers, source formats and output dtypes.
Source precision and native output precision are distinct: supported text
conversion can cast floating source weights to BF16. F16 output loads natively
only for full-attention Llama, Qwen2 and Mistral text and Decision use (the
native F16 profile above); other F16 output and all F32 output remain
conversion-only. Packed formats have their
own family/layout constraints. Classifier conversion coverage does not imply
that the desktop Chat or Decision workflow implements classification.

Conversion, template rendering and function tool parsing are separate support
claims. A converter adapter recognizes architecture/configuration and maps the
checkpoint's tensors into an artifact; it does not certify that the resulting
chat template can be loaded and rendered by the app. A renderable chat template
also does not imply tool calling: Stark enables function tools only when the
loaded template matches a reviewed tool wire format with an implemented parser.
The current parser registrations and their template identities are listed in
the [Function tools reference](LOCAL_API.md#function-tools). An unsupported
architecture stops raw-source installation before weight download with the
message: “This model architecture has no implemented converter.” This marks a
conversion boundary; adding a parser alone would not make that architecture
runnable. Successful conversion alone never enables function tools.

The [tool-calling coverage matrix](TOOL_CALLING.md) separates supported protocols
from model-bound evidence. Import completion and Installed rows assess the exact
artifact codec and show whether tools are available, unavailable or unchecked.
Adding a model for agent use requires the [tool acceptance checks](MODEL_ONBOARDING.md#assess-and-qualify-function-tools).

## Installation checks before weights

For a raw local or Hugging Face checkpoint, the app runs a bounded, cancellable
metadata preflight for the selected source and mode before starting a weight
download. A recognized conditional path continues without a confirmation dialog.
An unsupported or inconclusive result stays in the install sheet with a specific
explanation and other conditional modes to check. Text-only support is not silently substituted for
an image request.

The preflight pins the resolved primary revision. Actual conversion uses that
revision, and the source resolver independently repeats its metadata checks
before requesting weights. External catalog metadata is independently resolved
and rechecked during conversion; its revisions and bytes are not locked to the
earlier preflight report. The conversion manifest records the actual consumed
sources. Source, mode and operation identity fence late
results so that a check for a previous selection cannot authorize the current
download. Metadata inspection does not execute repository model code.

A browser download also pins the catalogue card's immutable primary revision.
The backend caches the offered repository, revision and artifact modes under
the catalogue request and opaque entry ID. The app submits that ID and selected
mode; it cannot supply a replacement source through this command. Installation
repeats generic preflight at the cached revision and rejects a changed revision before
weight download. A **Downloaded** marker requires a saved model with the same
bound repository, revision and artifact mode.

The fixed GLM entry selects only its exact native preparation choice. It bypasses
generic conversion and uses the helper's pinned weight and metadata revisions;
its prepared identity must match the catalog before publication. Stop cancels
the isolated preparation process. Failed or cancelled installations do not
register a model. Private, source-bound download staging is retained for retry,
and registry failures return the completed folder to that staging. A per-source
lock prevents concurrent installers from writing the same download.

A successful metadata check remains provisional. Full conversion validates the
actual tensor inventory, shapes, dtypes and payloads. Native admission validates
the emitted artifact. Runtime checks then establish the capability appropriate
to the selected mode. Memory availability and complete model execution are not
proven by a family row or metadata preflight.

Already converted artifacts follow their existing import/native-admission path;
their stored artifact type determines available modes. The registry does not
enable image input on an old text-only artifact, grant decisions to a vision
artifact, or replace the exact image/reference qualification boundary. See the
[vision family contract](VISION_FAMILY_CONTRACT.md).

## CLI reports

List the shared matrix without a checkpoint or weight download:

```sh
uv run --project tools/converter --locked stark-convert --list-families
```

Browse a family's representative download choices without fetching weights:

```sh
uv run --project tools/converter --locked stark-convert --family-models qwen3_5
```

This source-free command returns a schema-1 report with `family_id`, `status`
(`ready`, `offline` or `error`), a fixed `message`, `items` and `has_more`. Each
item binds an opaque `id` to `repo`, immutable `revision`, `name`, optional
`size_label`, `parameter_count` and `weight_bytes`, `quantization`, offered
`artifact_modes` and `default_artifact_mode`. It may also include an optional
`model_group` object with exactly `id` (lowercase 32-character hex), `name` and
`publisher` (the Hugging Face owner). This object is for presentation only; item
IDs, repository, revision, size and quantization continue to identify each
downloadable source row. A ready report may have no choices.
It does not create converted output or grant runtime capabilities. `--offline`
returns an offline discovery report without contacting Hugging Face.

Inspect a checkpoint's intended text use:

```sh
uv run --project tools/converter --locked stark-convert owner/model \
  --preflight --support-mode text
```

Inspect image conversion explicitly:

```sh
uv run --project tools/converter --locked stark-convert owner/model \
  --preflight --support-mode vision-language
```

`--support-mode` selects the intended use and metadata path for inspection.
Decision preflight checks future use of a text artifact; it does not create a
separate decision artifact. Normal conversion still selects its output through
`--artifact-mode`; supplying `--support-mode` also rechecks that intended native
use before weights. Without that optional mode, the CLI retains its separate
conversion-only paths. Explicit classifier conversion and F16/F32 export remain
possible even though they do not establish support for the desktop's application
modes. With `--support-mode text` or `decision`, F16 output passes preflight only
for Llama, Qwen2 and Mistral metadata without an active sliding window.

The report is JSON on standard output. `conditional`, `unsupported` and
`inconclusive` are successful report outcomes with exit status 0; callers inspect
`status`. A preflight report does not start conversion. Conversion itself still
rejects an unsupported path before requesting weight payloads. Normal argument
and command failures retain their error result.

The matrix has `schema_version: 1`, the three `modes`, a `families` array and a
separate `qualification` description. Each family records `family_id`, `name`,
`architecture_aliases`, `config_classes`, `model_kind`, conversion constraints
and per-mode `status`, `message` and `caveats`.

The qualification description reports `status: "not_assessed"`,
`scope: "checkpoint_and_build"` and `calibration: "not_assessed"`. This keeps the
matrix separate from recorded checkpoint evidence and decision calibration.

The per-source report has these fields:

| Field | Purpose |
| --- | --- |
| `schema_version`, `status` | Report version and selected-mode structural verdict. |
| `family_id`, `family_name`, `architecture`, `model_kind`, `mode` | Recognized identity and the intended operation being checked. |
| `message`, `reasons`, `warnings` | Fixed compatibility explanations and machine-readable diagnostics. |
| `modes`, `alternatives` | Per-mode states and other conditional paths to check for this source. Shared exclusions affect all modes; image-only exclusions can retain text/decision alternatives. |
| `pending_checks`, `qualification` | Checks/evidence still required; structural coverage cannot promote runtime qualification. |
| `source` | Logical `repo`, pinned `revision`, a filename-to-hash `metadata_sha256` map and `metadata_sources` records containing repo/revision/filename/hash/byte count. These records contain no local paths. |

Consumers must recognize the report schema and use the selected-mode verdict.
They must not interpret another mode's eligibility, a family adapter, or the
process exit status as permission to advertise runtime capabilities. Reports
contain metadata identities and compatibility diagnostics, not model outputs,
prompts or credentials.

Reason codes distinguish implementation limits from incomplete inspection. For
example, `unsupported_architecture`, `unsupported_embedding_layout` and
`unsupported_vision_geometry` describe known exclusions;
`metadata_access_failed`, `metadata_size_limit` and
`metadata_revision_unavailable` describe an inconclusive inspection.
`converted_artifact_requires_native_admission` directs a converted folder to its
import/native checks. Image-only metadata failures can have `image_`/`vision_`
reasons while leaving other paths conditional. Consumers use `status` and the
per-mode map rather than deriving a verdict from a reason's name.
