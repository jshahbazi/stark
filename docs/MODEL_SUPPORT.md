# Model-family support and installation preflight

Stark uses one family registry for conversion coverage, intended application
modes and early installation checks. The converter CLI and the menu bar app's
**Supported model families…** view consume that registry. A family entry describes
implemented paths and constraints; a source preflight checks selected checkpoint
metadata before weights are downloaded.

Neither result establishes that a complete model has loaded or executed. Exact
load qualifications, reference comparisons and local image-check receipts retain
their existing artifact/build/runtime identities. The loaded worker remains the
authority for live API and image capabilities.

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
