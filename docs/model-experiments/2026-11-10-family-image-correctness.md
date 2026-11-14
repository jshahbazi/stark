# Image loading and request admission correctness

Attempt ID: `20261110-IMAGE-01`. Work starts from `746e482f` (merged #121).
Status: implemented on branch; component controls passed, artifact qualification pending. No release or installation.

The candidate computes each component digest once per load, retains its selected
file descriptor and checks inode, length and nanosecond write/change times
before and after materialization. The same selected object must remain unchanged;
ctime must reject same-size edits even when mtime is restored. No persistent
digest cache is introduced. Tests must cover unchanged descriptors and mutation.

PNG decoding uses the already locked png 0.18.1 codec directly, ignoring ICC and
text chunks without decompressing them while retaining chunk CRC and frame
validation. Encoded RGB, alpha and EXIF policies remain. MPO uses only the primary
JPEG; ordinary concatenated JPEGs remain rejected. Require exact pixel equality
against metadata-free controls and unchanged malformed/animated/depth limits.

Image request admission now combines decoder allocation with conservative tower
submission storage before decoder state allocation. Count patches/constants,
BF16/F32 intermediate buffers, score matrices, projected/readback copies and
allocation margin. Qwen and Gemma 4 retain an entire tower submission; Gemma 3
waits after each block. Tests must reject checked arithmetic overflow and account
for eager score matrices. These are conservative allocation inventories, not
measured peak RSS or a new hardware capability claim.

Qwen image decoder graph optimization and the existing 4,096 encoder-patch limit
remain open. Raising geometry limits requires new workspace, preprocessing and
full-model numerical/resource qualification. No inference speed claim is made.
All actual artifact qualification and matched image/text performance arms remain
pending for these changes; model-free checks cannot qualify the full matrix in
`IMAGE_DEFAULT_PARITY_PLAN.md`. Weight/activation/cache precision is unchanged.

The first native component run passed the attention/sampler controls, then image
checks found that png 0.18.1 skips CRCs for ignored ancillary chunks. The new
malformed-metadata control failed (44 image checks passed, one failed, six
ignored). Header preflight now independently checks every chunk CRC using the
already locked crc32fast 1.5.3 dependency. The failed attempt is retained; the
corrected run is pending. The ordinary PNG pixel path and metadata-free output
controls did pass in that first run.

## Final component outcome

The sealed CPU/Metal executable at runtime source `b4120152` passes 46 selected
image controls (six actual-artifact controls ignored), plus the descriptor
mutation test and tower accounting/overflow test. PNG palette/transparency,
alpha discard/composite, primary MPO, ignored compressed ancillary payloads,
independent CRC checks, malformed formats and existing bounds are covered.
Weight/source precision is unchanged. Initial CRC failure and the final lint
iteration repair are preserved in the
[aggregate record](../archive/runs/2026-11-10-model-family-triage.md).
No actual converted image artifact load, full image generation, image-loaded
text/MTP parity, warm startup timing or peak RSS was measured for this change.
The resident-state and workspace acceptance matrix remains artifact-specific.

## Workspace estimate continuation

The later independent review found the original depth-multiplied scratch bound
could reject ordinary Gemma 4 E2B images unnecessarily. Attempt
`20261110-FAMILY-RECHECK-01` replaces that bound with per-profile allocator-rounded
stage inventories and a separate persistent-constant budget. The original
`b4120152` result above remains source-bound; revised controls and numerical
limits are recorded in the [continuation](2026-11-10-family-double-check.md) and
[workspace inventory](../archive/runs/2026-11-10-vision-workspace-inventory.md).
