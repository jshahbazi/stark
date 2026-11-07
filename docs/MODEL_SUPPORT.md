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
includes a `qwen4_exp` converter for Lily-format Qwen3.8 text models. It writes
a self-contained typed `Qwen38LanguageModelConfig` artifact with canonical
MLX storage and unchanged mixed quantization, sharing the existing specialized
Qwen kernels. Vision and MTP tensors are omitted from this typed text-converter
path. Retained Lily checkpoints have a separate pinned native image path; that
image support does not come from the `qwen4_exp` text artifact. The
ordinary/catalog GLM GGUF path keeps MTP unavailable. The optional GLM EXL3/FP8
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
validation. The 12B embedder and 26B MoE paths have component checks, while the
31B feature gate currently fails. Full 12B/26B/31B checkpoint generation is
unqualified. Installed desktop UI execution and image performance are
unmeasured. Exact checkpoint evidence remains tied to its reference profile and
engine build. See the
[original E2B validation record](archive/runs/2026-10-28-olmo3-gemma4-native.md).

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
Conversion-only families are omitted. Cards offer text or image artifacts when
both are available. Image output is selected by default only when metadata
confirms a supported Qwen3.5 vision wrapper, BF16 vision tower and reviewed
processor; decoder weights may use a supported affine profile. An image choice
still needs the normal installation and local runtime checks.
The download and preparation progress remain in the browser. Completion adds
the model to the library without selecting it for use.

Family cards group model sizes and quantizations into compact variant rows.
Each family opens on **All models**, followed by tabs for the quantizations and
precisions present in its current catalogue. Tabs come from the full current
family list; search and sort narrow/reorder the selected tab without refetching
metadata. If a selected format disappears after refresh, the view returns to
**All models**. Switching families also resets that filter. The row's source ID,
download status and text/image choice remain unchanged by filtering.
Same-publisher size and quantization variants share a card. Base, Instruct,
Thinking, context-length variants and fine-tunes remain distinct. A cross-publisher
group is allowed only for a registered MLX 4-bit/8-bit export whose tokenizer
configuration points to a registered same-family base and whose basename matches
after removing the quantization export suffix. An optional `model_group` descriptor
provides its 32-character `id`, display `name` and `publisher`; it is presentation
metadata and does not change the immutable source entry ID. Family-size sorting
uses the minimum or maximum known model size among the visible versions. Counts
are never inferred from model names or quantized packed tensors.

The download catalogue combines registered reference sources with a bounded live
Hugging Face search. Parsed configuration identifies the family; repository
names alone do not establish compatibility. Each page inspects up to 30 distinct
candidates and returns compatible choices. **Load more models** follows the remaining
pages automatically, preserving results already loaded. Failed pages can be retried
at the same offset. Generic discovery offers supported floating Safetensors and implemented MLX 4-bit/8-bit
paths. GLM instead has one fixed `UD-Q2_K_XL` GGUF entry, with its source
revisions, four shards and 108,720,071,552 stored weight bytes owned by the native
preparation helper. Browsing that entry uses bundled metadata and makes no Hub
requests. Download uses the pinned weight and tokenizer repositories, preserves
quantized bytes and validates the completed prepared identity before library
registration. Download progress remains indeterminate until the stored byte
total is verified. GLM image input and Decision mode are unavailable.
Lily Qwen3.8 uses a mixed `4-bit / 8-bit` label, including
its fixed Q8 control projections. Embedding/reranking tasks, known internal test
models, quantized vision towers, MoE architectures outside the implemented
Qwen3.5 and Qwen3.8 paths, adapter-only and unsupported packed formats are
excluded. Qwen image profiles can retain supported packed decoder matrices
alongside a complete BF16 vision tower. The
implemented Qwen3.8 text path includes its MoE graph; Qwen3.5 BF16 MoE image
conversion is a separate image-only path.
An uncatalogued compatible fine-tune can appear. This is
a metadata-filtered list of compatible candidates; gated, inaccessible or unsupported
repositories may be omitted.

Discovery reads bounded metadata and repository file information, without
requesting weight payloads or executing repository code. Parameter counts and
download bytes appear only when usable metadata supplies them. Floating sources
are shown as BF16, the native output format; packed tensor counts are not shown
as logical parameter counts. Missing metadata can leave a size unknown or omit
a candidate. Offline and failed discovery have fixed messages and can be retried.

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
conversion can cast floating source weights to BF16. F16/F32 output remains
conversion-only where the native engine requires BF16. Packed formats have their
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
modes.

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
