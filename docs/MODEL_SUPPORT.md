# Model-family support and installation preflight

Stark uses one family registry for conversion coverage, intended application
modes and early installation checks. The converter CLI and **Model Catalog**
tab in the menu bar app's **Manage Models** window consume that registry. A family
entry describes implemented paths and constraints; a source preflight checks selected checkpoint
metadata before weights are downloaded.

The original-format [native hybrid path](NATIVE_HYBRID_MODELS.md) separately
loads GLM-5.3-Flash UD-Q2_K_XL and Qwen3.8-Flash-Next mixed MLX 4/8-bit weights.
Prepare those pinned downloads with its helper, then import the prepared local
folder. Their original sources are not offered by the ordinary converter's
family catalogue or source preflight. The native loader validates their tensors;
generic Decision use still requires a direct-answer template and checkpoint
checks.

Neither result establishes that a complete model has loaded or executed. Exact
load qualifications, reference comparisons and local image-check receipts retain
their existing artifact/build/runtime identities. The loaded worker remains the
authority for live API and image capabilities.

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
both are available, with image output selected by default only when the full
Qwen3.5 BF16 model's vision configuration and processing metadata pass inspection.
An image choice still needs the normal installation and local runtime checks.
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
names alone do not establish compatibility. At most 45 candidates are inspected
and 30 choices are returned, preserving representative sizes and formats when
available. Only supported floating Safetensors and implemented MLX 4-bit/8-bit
paths are offered. Embedding/reranking tasks, known internal test models,
quantized vision, MoE, adapter-only and unsupported packed formats are excluded.
An uncatalogued compatible fine-tune can appear. This is
a representative list, not an exhaustive inventory of Hugging Face.

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
| `vision-language` | Image chat through the supported full vision artifact and local image checks. |

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
repeats preflight at the cached revision and rejects a changed revision before
weight download. A **Downloaded** marker requires a saved model with the same
bound repository, revision and artifact mode.

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
