# Model-family review disposition, 2026-11-10

This work starts at `746e482f`, the merge of #121, on
`codex/model-family-correctness`. The supplied report was a static review.
The table distinguishes implemented repairs from existing behavior and work
that still needs implementation or artifact qualification. No new performance
figures are inferred from dispatch counts or old runs. Source work does not
establish release or installed-app behavior.

| Supplied finding | Current disposition |
| --- | --- |
| Gemma 2 attention logit soft cap ignored | Confirmed; cap reaches CPU/Metal online attention before masking/softmax. Component numerical controls pass; actual Gemma 2 checkpoint numerical/performance qualification remains open. |
| Tiny finite temperature causes worker reload | Confirmed; reject an overflowing F32 reciprocal before native generation. Recoverable request and subsequent-request controls are required. |
| Dense MTP changes top-k plus top-p meaning | Confirmed; dense Qwen and Gemma assistant MTP use full-vocabulary nucleus mass before top-k intersection. Hybrid policy is separate. RNG/rounding equivalence across MTP modes is not claimed. |
| Unquantized MLX Qwen3.5 doubles norm offset | Confirmed; use saved convolution layout as the raw/baked norm marker, reject ambiguous/mixed layouts. Synthetic effective-gain controls pass; full-checkpoint qualification is pending. |
| Granite thinking does nothing | API field selection was repaired in #121. Source-bound native-rendering controls pass on this base; no duplicate API patch. |
| Ordinary dense NaN becomes token zero | Confirmed; invalid readouts now produce an error sentinel, with host validation and pending-work cleanup/fencing. Final integrated native controls are recorded separately. |
| Five advertised models cannot convert | Three inactive config-annotation cases repaired: DeepSeek, Nanbeige, Reka. RNJ and Polaris are marked planned, with false README support claims removed; their YaRN attention/extrapolation semantics and RNJ layer topology need implementation and numerical qualification. |
| Qwen3.5 2B/4B override digests fail, 2B Q4 uses wrong profile | Confirmed; exact digest profiles resolve and 2B Q4 uses the 2B defaults. Source-bound synthetic metadata controls pass. |
| Text-only Qwen3.5/Gemma lack image-wrapper snapshot path | Confirmed dispatch gap; ordinary snapshot-capable/non-rewindable text now uses existing capture/restore, capability reporting and matching memory admission/planning. Production artifacts still require their own qualification. |
| LFM2/LLaMBA/Flint lack reuse and blocker docs | Concrete state inventories and missing capture/restore/admission work are documented in GENERATION.md. These paths remain fresh per request. |
| Flint scalar CPU attention | Confirmed in hybrid/flint.rs; no GPU attention implementation or speed claim added. Porting requires exact position/window traversal, F32 reference comparisons and complete artifact qualification. |
| Qwen image turns use reference decoder | Confirmed; shared fast text graph does not replace the image decoder graph. New image kernels must preserve spatial positions, image spans, reference BF16 boundaries and prefix invalidation. Open; old text timing does not measure an image turn. |
| PNG profiles / MPO files rejected | Fixed at decoder scope: skip unused PNG ICC/text without inflation, independently check CRCs, and decode only the primary MPO JPEG. Pixel/alpha/EXIF/limit controls pass. |
| Qwen images above about one megapixel rejected | Existing 4,096-patch admission remains. Raising it requires new tower workspace/geometry admission and full-model resource/numerical qualification. No source-resize recipe or context contract was silently changed. |
| Image loads hash weights twice | Removed the second scan; selected descriptors are checked for inode/size/nanosecond mtime/ctime changes through loading. A persistent warm-start digest cache remains unimplemented on converted image routes, so the first scan still occurs on each load. No startup latency claim is made. |
| Vision memory omitted after decoder state allocation | Combined decoder and conservative tower submission workspace is admitted before allocating decoder state. It includes patch uploads, BF16/F32 buffers, score matrices/readbacks and slack. This is conservative accounting, not measured peak RSS. |
| Checkpoint defaults stored but unused | Greedy remains the intentional host default selected in the preceding task. Stored metadata is preserved; callers can choose stochastic sampling. No evidence here establishes better quality from changing that product default. |
| Reasoning never separated | Leading explicit/rendered `<think>` reasoning is separated by #121, including API streaming. Other trained reasoning protocols need source-bound parsers and controls; this does not claim universal format support. |
| Routed MoE dispatch overhead | Confirmed per-row/per-expert gather/up/gate/gather/down/accumulate loop in encodable_block/mlp/routed.rs. The batched native hybrid Qwen path is a different implementation. Grouped route/expert kernels are an open optimization; the report’s whole-request impact was not measured here. |
| Gemma 4 MoE lacks Qwen 256-row prefill policy | Confirmed: decoder ordinary_prefill_execution_cap selects only MixtureOfExpertsConfig/Qwen. A Gemma policy requires exact physical chunk/precision and image-bidirectionality controls plus its full checkpoint; deferred without changing the shared 1,024-token snapshot grid. |
| Sliding-window prefill masks unused keys | Optimization remains open. Skipping keys must preserve source traversal and reductions for full/ring/shared KV and image spans; no 2–4 percent claim is adopted. |
| Dense MTP snapshots discarded immediately | Fresh requests without a cache scope now skip copying. Scoped requests retain exact snapshots. Admission remains conservative; claimed 150 MiB or runtime savings were not measured. |
| Flint output and Gemma 3 ring docs stale | Removed nonexistent fixed 4,096 output-cap claim; Gemma 3 docs now describe implemented snapshots and pending artifact qualification. |

The complete converter suite passed **6,936 tests**, with **94 skipped**, in
319.73 seconds using locked uv Python 3.13.13 and the unchanged native template
helper from the merged #121 worktree. The earlier 939-test family slice overlaps
this suite. Ruff on converter source/tests/exporter and repository policy pass.
Final native validation and source/executable identities follow below. Previous
failed attempts remain in their records. No full production model conversion
or new speed benchmark ran.

Attempt records: `20261110-CONVERT-01`, `20261110-DENSE-01`,
`20261110-PREFIX-01`, and `20261110-IMAGE-01` in the model experiment index.

## Final source and validation

Runtime source: `b41201523b547ea646f1378104319e7f501b3ac7`, preceded by
converter commit `2aa63d34`. Subsequent validation-document changes do not alter
the tested source. Host: Mac17,7, 128 GiB RAM, macOS 26.7.2 (25G332).

| Check | Observed result |
| --- | --- |
| Complete converter suite, locked uv Python 3.13.13 | 6,936 passed, 94 skipped; 319.73 s |
| Pinned nightly CPU/Metal engine compilation | Passed |
| Selected CPU/Metal attention, sampling, image, lifecycle, snapshot and memory controls | 63 passed; six image artifact tests ignored |
| Explicit existing dense MTP synthetic artifact controls | Two passed; no-scope capture omits snapshot, capture-on and continuation remain exact |
| Rust 1.98.1 release generator tests | 144 passed, four ignored |
| Explicit snapshot-only worker test | Passed on two synthetic artifacts, full-attention and sliding-ring; greedy and seeds 0/7/42, expected 1,024 reused rows, fresh-output comparison, divergence, scope/context changes, scope-less/shorter prompts and clear |
| Release scorer tests | 34 passed in default configuration and 34 passed with optimized feature; builds were separate from generator |
| Real converted dense worker smoke | Qwen3-4B Q4 rejects tiny temperature with invalid_sampling_request, then emits four greedy tokens successfully in the same worker |
| Generation process boundary harness | Six cases passed |
| Release generator Clippy, all targets, warnings denied | Passed |
| Converter Ruff, pinned Rust formatting, repository policy and diff checks | Passed |

The ordinary/capped attention control fixes tolerances at F32 2e-5 and BF16
0.008 against an independent F64 oracle. No tolerance was relaxed. Synthetic
state tests do not establish production checkpoint fidelity. The full native
Metal suite, full CPU/grammar suite, full Gemma 2 reference replay, complete
image-family matrix, output quality, peak RSS and performance remain unrun.

The generator build uses Rust 1.98.1 and locked dependencies. The engine test
uses nightly-2026-10-01 with `cpu,metal`, debug information disabled for dev/test
and incremental compilation disabled to bound disk use. The distinct new
worktree owns its own target directory. No model or installed application was
modified. Runtime artifacts and content-bearing validation scripts stay ignored.

| Bound artifact | SHA-256 |
| --- | --- |
| Release stark-generate | `f8aaab73a77622a64011784c1b2db8f562c88bab824d7e5651dce258510d57ce` |
| CPU/Metal engine test executable | `6456fe05084621e490aff01d42d8a26e3707ae3bf7768dd3cf940c457d0c68f9` |
| Cargo.lock | `6b59e304e35126a5e7b987a5d4a7ea8b571ec02f6b231e1344c8791b3173ab74` |
| Converter uv.lock | `df7711726b1ed7e235ded21b9e6e1cbe130924be4d2c580d5d6bd9005248f8d5` |

## Real dense smoke boundary

The local converted Qwen3-4B Q4 artifact records source revision
`4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25`. Its config, weight, tokenizer and
conversion-receipt SHA-256 values are respectively
`5b505594bf00ef9682b373c6a292b88f94f1e1ceab449784c12accb835806c7f`,
`7c2af9a45e5b0d196f06a5cbfd6fc5cc95adf29df79885d0540f2206fb3d40d8`,
`aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`, and
`0dcbe259f66839c53d7ea5c1e9dc2870b20f0f3f463891c30e13589f2200bd3c`.
The matched worker process used context 512, output budget four, greedy, seed
42 and MTP off. An earlier tiny-temperature request returned a recoverable
error without native dispatch. The next valid request succeeded; four generated
IDs hash to `1735f922da391a9e850e42fa617a2b77406211dbe9d06b348dd12a1ad817529d`.
Only counts and hashes are published. This is a bounded process-recovery smoke,
not a quality, complete prefix, Gemma 2 or image-loaded qualification.

The first smoke harness expected the wrong error string and stopped before its
valid follow-up; the production worker correctly returned
`invalid_sampling_request`. The harness was corrected and rerun on the same
artifact. No product code changed for that setup error.

## Attempts that did not pass initially

The first engine build missed two newly extended attention call sites/imports;
these were corrected. The first ignored-metadata image control caught the png
codec skipping ancillary CRCs; independent header CRC checks repaired it. The
first Clippy run rejected constant-sized chunks_exact_mut; the conversion loop
now uses as_chunks_mut with identical pixel arithmetic. All final checks above
use the corrected source.

Initial snapshot fixture loads failed before generation. The old synthetic
no-MTP config still referenced a file containing 20 unused MTP tensors and five
MTP metadata leaves; the first copy also enabled an unqualified reference flag.
The corrected owner-private fixtures retain the original target tensors exactly,
remove the undeclared MTP head/metadata, use the existing default graph and add
only a 128-token attention window for the ring control. No installed or original
fixture was changed. Both corrected fixtures then passed; those failures were
invalid setup, not output parity failures.
