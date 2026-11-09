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
