# 20261109-GLMVIS-05: complete GLM native image-worker integration

Registered 2026-11-09 before integration changes. Source baseline `3d8d70e0`
(GLM components plus current main, including retained-text-MTP policy fixes).
Predecessor: [GLMVIS-04](2026-11-09-glm-actual-vision-components.md).

## Artifact and intended change

Use the existing complete pinned native GGUF artifact at
`stark-tool-calling/runtime/recommended-tools/glm53-q2-production`, with its four
Unsloth shards at `a38483c8cd5df544f53d70fb281afe97369d5ab6` and metadata from
`zai-org/GLM-5.3-Flash@04c4e9e95c5da8862dced7e5056455116f83a7e0`.
Create a separate ignored image artifact by sharing immutable decoder bytes;
leave the original folder untouched. Add the already verified 1,164,010,176-byte
sidecar under `vision/`, never as a fifth decoder shard.

Implement a content-bound image manifest, preparation/upgrade, loaded-model
identity and request-owned GLM image dispatch through the existing worker.
Preserve original text weights, arithmetic, prefix state and sampling. Text
requests retain supported MTP; image requests have zero drafts/proposals/reuse
and invalidate text state. The EXL3 qualified native-stage/default-three policy
must survive image loading when its own image ABI is implemented; do not
silently fall back to a headless stage.

## Fixed gates

1. Fail closed on changed source/profile/config/tokenizer/template/processor,
   shard/sidecar hashes, file identity, tensor inventory and native bounds.
   Bind actual mapped decoder/sidecar descriptors to the verified files.
2. Run an actual complete native GGUF image request through prepare/generate/
   discard and following text recovery before enabling catalog/host defaults.
   Record source/build/artifact hashes and counters, never request content.
3. Exercise errors/cancellation/restart and single-use model-bound image state;
   preserve fencing on uncertain device completion. Real image requests must
   clear previous text prefixes and publish no image-derived prefix.
4. Qualify ordinary text on the default image-loaded worker against fresh
   controls, including the 1,024-token native GLM reuse grid, scope/context and
   supported state-option switches. Compare matching greedy/seeded algorithms.
5. Compare prior qualified text, candidate text and candidate default-image
   loading at identical target weights, arithmetic, context/output budget,
   sampling and supported MTP policy. Separate cold/follow-up timing, first
   token/prefill, useful decode, end-to-end, loading and memory. Predeclare
   performance bounds from repeated baseline variability before timing.
6. Charge resident tower/head, projection scratch, target/predictor state and
   snapshot replacement in load/request planning. Preserve the 8 GiB disk
   reserve, serialize model/GPU work, fetch no further full checkpoints and
   retire only owned reproducible build targets after sealing binaries/evidence.
7. Ordinary GGUF source MTP execution and EXL3 image ABI are concrete separate
   missing implementations. Neither is N/A, and neither inherits another
   backend's qualification. All user gates remain open until actually checked.

## Status

Integration in progress. No complete GLM image request, image-loaded text cache,
matched text speed or image-resident text MTP result exists for this attempt yet.

The first converter increment adds pinned four-shard plus sidecar preparation
and bounded schema-7 identity inspection, with exact source/profile/metadata and
payload contracts. Decoder/config bytes remain unchanged; the sidecar is linked
under `vision/` and the new descriptor is published atomically. Inspection does
not rehash 109 GB during model-list refresh; native admission must rehash payloads
and bind the actual mapped descriptors before execution. The real complete
source folder passed the bounded source/config check without weight I/O.

Twenty-eight focused Python source/manifest tests pass, including immutable
decoder preservation, sidecar sharing outside the split set, payload changes,
metadata/symlink rejection and strict scalar/path contracts. Formatting/lint,
whitespace and repository-policy checks pass. These are preparation/inspection
controls; actual runtime integration remains in progress and catalog defaults
have not been enabled by this increment.

## Native integration source, before runtime qualification

The next increment connects schema-7 GLM artifact admission and its separately
mapped vision tower to the native model and generation worker. It preserves
F32 vision patches, binds request preparation to one loaded owner, expands the
fixed still-image token span and sends tower features through the previously
checked causal embedding bridge. It invalidates native snapshot ownership at
image entry, clears captures, resets on cancellation, and fences the model after
an uncertain device error. Ordinary text continues through the existing graph.
Request planning conservatively charges the resident sidecar and 512 MiB vision
scratch even when the current request is text.

Admission now pins the complete finished source config
`d1e229307a382fc1deae8e4a48b6ffc2f47fc47769ab8789c023bdbd806072fd`,
the original tokenizer
`19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d`
and source template
`34d5ee66b12fa6446cdae131c352b8f68cd85369e0e6fda115583805fada3891`.
Both converter inspection and native admission reject a coherently rehashed
alternate text graph or tokenizer. The native loader parses only its exact
hash-bound config/tokenizer buffers. Its EOS policy uses the admitted finished
text config (preparation already folds generation EOS into it) and admitted
token-codec EOS; it does not reopen auxiliary generation/tokenizer metadata.
Actual mapped decoder descriptors must match the four payload descriptors that
were hashed. The sidecar retains its independent mapped-descriptor check.

An independent source review found the earlier separately reopened metadata
could escape that binding during parent-directory substitution. The loader now
owns the admitted buffers, with a directory-replacement control. That review also
found that a geometry-only config check did not establish the complete pinned
decoder graph; the fixed finished-config identity closes that gap. The reviewer
rechecked both fixes and found no further source/state blocker. This review did
not run a model or qualify memory/performance.

Host functionality recognition now understands the GLM schema/source/native
14-by-2 geometry, the pinned template and the separate sidecar. It remains
distinct from an execution receipt. Preparation can fetch only the pinned
sidecar into `vision/` or use the already verified local sidecar; it cannot
silently add a fifth decoder shard. Source planning reports execution as
unchecked, rather than inferring a runtime result from metadata.

The current focused Python pass is **186 passed**, including 33 GLM
source/preparation controls and the shared preparation, handshake and image
checks. Lint, changed-file Rust formatting, whitespace and repository policy
checks pass. The complete pinned installed source still passes bounded
source/config/tokenizer inspection without decoder payload I/O. A full
converter pass with a preserved older helper produced 6,826 passed, 96 skipped
and one large-request boundary failure; that case passes using the newer
preserved helper, and a final full pass is pending. No native compilation or
full GGUF image execution result is claimed for this integration increment.

The EXL3 image ABI remains unimplemented and fails explicitly. The image loader
retains native decode-stage selection rather than hardcoding it off; the EXL3
head/default-three policy must still be exercised after its image bridge exists.
Ordinary GGUF predictor execution also remains missing. Catalog defaults,
full image generation, loaded-image text cache/performance and both backend MTP
gates remain open. Native build/model work is deferred while shared disk space
is below the required 8 GiB reserve; no additional model payload was requested.
