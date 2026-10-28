# Human-facing model browser, 2026-10-27

This work starts from `0ffc5bb0ea9334b49947b5708a9b631a06b10052` on
`codex/qwen35-vision-artifact-contract`. The requested view shows available
families and downloaded models, including quantizations and other saved
versions. The family catalogue no longer presents capability columns,
checkmarks or decision-model claims.

## Delivered behavior

**Models…** opens two tabs. **On this Mac** presents naturally ordered model
cards with saved versions grouped under their source model. Variant rows retain
stored names and show matrix precision/quantization, disk size and source
version when known. **Running** uses the actual loaded model ID. **Model
families** presents 12 app-family names in a two-column card grid. It has no
technical explanation or mode matrix. The full structural registry remains in
the CLI and developer documentation.

The local library event adds an optional schema-1 `display` descriptor; model
registry and artifact files are unchanged. Grouping requires an unambiguous
source identity, internally consistent source-manifest schema/digest, matching
current configuration hash and matching declared output size. Different source
revisions and quantizations from one repository group together. Distinct
fine-tunes stay separate. Incomplete, inconsistent or malformed descriptors
retain a standalone entry with its exact stored name.

Quantization labels come from native matrix specs and their actual tensor
representation. Floating convolution/control exceptions do not turn a BF16 or
quantized matrix label into a misleading different format. Reads are bounded to
1 MiB for weight headers and 2 MiB for each config/receipt. No tensor payload is
read to populate the browser. No parameter count is inferred from filenames or
repository names. Display metadata does not grant runtime capabilities or
checkpoint qualification. Installation compatibility checks remain active.
Family labels also require agreement with the bound generic profile or source
catalog adapter; an inconsistent family label is omitted.

## Completed checks

| Evidence | Result |
| --- | --- |
| Full locked converter suite, Hub offline | 4,833 passed; 88 skipped. |
| Focused inventory tests | 30 passed, including manifest/schema consistency, conflicting sources, config/output binding, precision variants, safe fallbacks and a sparse 1 GiB file whose payload is never read. |
| Exact-source Swift controller suite | 55 passed; includes grouped versions, natural ordering, duplicate names, unknown descriptors, empty inventory and running/selected identity separation. |
| Swift compilation | All app sources typechecked; final application compiled. |
| Builder suite | 23 passed; 13 subtests passed, including the new packaged inventory module. |
| Ruff, scoped formatting, repository policy and diff checks | Passed. |
| Independent review | No remaining functional blocker; source ambiguity and manifest-consistency findings corrected before the final full run. |
| Actual installed UI | Both tabs inspected through real macOS controls and screenshots; family names only, grouped model cards and badges visible without clipping. |
| User data | Model registry byte-identical; selected model and mode preserved. No model was downloaded or removed by this work. |

Actual existing models are **JevK5**, whose receipt identifies Qwen3.5/BF16 and
8,413,483,320 weight-file bytes, and **qwen3-dense**, whose bound source is
Qwen3-0.6B/BF16 with 1,192,276,800 weight-file bytes. The installed view shows
8.41 GB and 1.19 GB, with source versions `c4f7fdb` and `c1899de`. Multiple
downloaded quantizations/version rows are covered by fixtures; no additional
real versions were downloaded to manufacture that evidence.

The signed local build was installed into `~/Applications/Stark.app`, with a
verified private backup and exact candidate/tree/signature checks. The final
installed app is running. It is Developer ID signed with hardened runtime and a
secure timestamp, and is not notarized. Native worker code and the exact image
qualification catalog are unchanged; numerical parity was not repeated for
this browser change. Private models, bundles, receipts and raw logs remain
under ignored `runtime/human-models-validation-2026-10-27/`. Earlier candidates
and source-bound runs remain separate from the final identities below.

## Final identities

| Item | SHA-256 |
| --- | --- |
| Installed application tree | `1d327d5a7b2fcbdb30615946737b4f80e38330d218b0308aa788f5429d716700` |
| Signed Swift executable | `70abdc0bd2a5f48e36af45f2da2528f34d6ef411a1ab3fa4190c8928eb8220f8` |
| Application manifest | `1ab2073487ae974941d134016db037c030af9e0da82063a2d2a1227bf74ed89d` |
| Candidate signing identity report | `0a44d9de953feae21efc5707fd6793e8a95726b90591213daf05346035c14ab7` |
| Installation verification report | `409c59cde8599853651f9a7f7a2e79257c272cae875cfe7f836e4c3924e7de6d` |
| Installed-state and GUI aggregate | `a2d877c19bc04b57bb09a3d9fa97f2f45e235962e5ce218a90d1b135f8ea3d5d` |
| Full converter-suite log | `44954107618bb667ae43e08f6f0d7853b5505a63fef623ebeb93bf815b0525ba` |
| Controller log | `8bd12045ad72d40c8825d90fc007be99a420d6a1d357437691a54fc3aed9e129` |
| Builder log | `5c5fbb223ceef355f5e3c407ddc56dd010af2b921bc6245e5ad070587b1c99f9` |
| Focused inventory log | `a20368d3dd921d9ef1e9595decde4c3a53485604704f83246b078137418cdca0` |
