# Gemma image-default continuation

Registered 2026-11-09 before new implementation or execution. Baseline is
`900752b4`, combining Gemma `7468dbc4` with reviewed Qwen PR #114 at `c18f1d23`.
The preceding Gemma cache evidence remains attached to its original binaries;
new validation is recorded separately below. Previous results do not qualify
the new Gemma3 path or a Gemma4 assistant.

## 20261109-G3IP-01: Gemma3 4B text prefix restoration

State: implemented and actual 4B cache lifecycle controls passed. Gemma3 now
uses model-bound physical-ring capture/restore, scoped worker retention,
snapshot memory admission, suffix-only prefill and the actual-prefill-count
guard. Image arithmetic is unchanged. Larger/quantized artifacts remain open.

Qualification starts with the retained original-BF16 4B image artifact, whose
selected files and binary will be sealed before execution. Context 2,048,
nonperiodic input 1,792 tokens, eight greedy/seeded outputs, chunk boundary 1,024.
Require exact fresh/cache hashes, actual suffix counts, scope/context changes,
real image zero reuse, cancellation/error/reset and same-worker recovery. The
cached request must remain within context capacity, making duplicate replay
observable. No MTP capability is inferred for Gemma3.

The bounded speed control uses the same source graph and payload with tower
omitted/resident, and the retained previous text worker as a separate baseline.
ABBA loads, one warmup plus three measurements per load, context 2,048, same
short input, 64 greedy outputs, seed 42, MTP0, no prefix scope. Loading is excluded.
Record first token, decode and whole request; predeclare loss bound as greater
of 5% and twice baseline relative range. Stop on paging growth, insufficient
headroom or host interference. Larger/quantized artifacts remain unqualified.

## 20261109-G4AP-01: execute the official E2B assistant

State: exact E2B companion intake and native-layout payload conversion complete;
native component compilation and execution controls are being recorded below.
Serving MTP remains disabled.
Follow the pinned [assistant stages](../GEMMA4_ASSISTANT_PLAN.md), beginning with
typed companion conversion and target/KV handoff, then actual four-layer draft
execution and exact target verification. Preserve target embedding scaling,
normalization and ring/shared-KV ownership; do not enable unsupported partial
ring speculation. E4B requires its own complete intake and execution results.

No MTP capability is advertised before actual proposals and complete session,
cache, memory and image-transition controls pass. Serial proposal execution,
if used for a correctness component, does not qualify useful MTP performance.

The new converter binds exact pinned E2B/E4B source configuration hashes and
complete 50-tensor inventories, preserves source BF16 matrix/scalar payloads, validates
the entire I64 ordering as a vocabulary permutation, and binds the companion
to the selected target config/weight identity. Configuration reads are limited
to 1 MiB; the permutation plane is 2 MiB. It is a separate companion plan API,
not a catalog entry, chat model or enabled runtime head.

The initial schema-1 raw layout at `76298dc5` was a planning/component increment
and was never executed as a native head. Schema 2 maps to existing native linear
and normalization blocks: four up/gate fusions preserve source rows, 21 norm
vectors widen exactly from BF16 to F32, and every matrix has an explicit full
precision output/input layout. Output has 46 tensors and 11,264 more payload
bytes than the 50-tensor source. No numerical result transfers from schema 1.

Eleven CPU converter checks pass under locked Python 3.13.13. Sparse fixtures
use independently fetched pinned source header inventories; they cover both
small-profile payload sizes and target binding, unsupported source, missing,
extra, wrong-dtype/wrong-shape tensors and duplicate/negative/out-of-range
ordering entries. Ruff passes. These are component controls, not a payload
conversion or proposal result.

Pinned Transformers 5.15.1 assistant source SHA-256 is
`a77e67673ce7d2248294e361c3c074c29d8d6a092655d7b27b95fc9777a2d0d1`;
candidate generator SHA-256 is
`411ea9b7c8062344ba6e2ad936c54927c19062fc198ee40f4b6ca50958672c1e`.
All four assistant layers borrow target KV and use a constant position within
a round; the assistant has no independent KV cache. Existing shared-layer
attention expects an uncommitted suffix. Passing the fully committed target
state directly would include an extra row; dedicated external-KV query
semantics must precede native proposals. Target activation, query/readout,
verification/rollback, cache/lifecycle and useful-speed gates remain open.

Stage B now has a source-written dedicated committed-KV attention component,
compiled with dedicated CPU/Metal controls passing. Its lifetime-bound target borrow resolves
shared owners, carries the committed target length as the unprocessed draw's
absolute query position, and excludes
shifted rotary state. The single-query path borrows full caches and linearizes
at most 1,024 local rows into separately accounted K/V scratch; it reads exactly
the committed row count without adding a phantom suffix. Ordinary attention
dispatch and partial-ring speculation guards are unchanged. Production model
identity, plain-text state and exact target geometry must still be checked by
the future assistant round before this low-level component is used. Source
assistant numerical comparison and useful-speed qualification remain open.

Stage C also has source-written native four-layer execution, pre/post
projections, BF16 residual/scalar boundaries, own readout and centroid masking.
The first component computes the full device readout and then applies the
source top-32 cluster mask; efficient selected-row readout remains open. Target
frame prefill skips trailing shared-KV query layers for earlier prompt rows and
captures only the final normalized activation. Its greedy draft component keeps
the frame immutable and chains projected feedback at a fixed position; it
rejects foreign loaded-model frames. Actual payload execution and independent numerical comparison are recorded
separately below. Target verification, accepted
state commit/rollback, head-aware prefix reuse, worker capability/default
loading and useful MTP speed remain unimplemented; no serving MTP is enabled.

Source masking review found an additional wrapped/batched handoff requirement.
Transformers 5.15.1 `masking_utils.py` (SHA-256
`e8c497af6979274fc6ae78980ad9893e7850bdb750e46d459f09178123992196`)
uses inclusive distance `<=512` before reversing the assistant's KV mask.
It can therefore consume 513 supplied local rows. The assistant-only query
mask admits that count; ordinary target ring capacity remains 512. A source
handoff after a wide prefill/verification batch needs to retain the additional
visible row before target acceptance, separately from the ordinary ring. The
current short/final-row component receives at most 512 rows. It cannot qualify
that larger source handoff until the extra-row interface and oracle controls
are implemented. No wrapped/full generation MTP claim follows from three
functional short-context proposals.

## Resources and promotion

Source/CPU-light work precedes the GLM → Qwen → Gemma resource handoff. Builds,
Metal, model loads and downloads are serialized. Use a distinct ignored target;
retain 8 GiB disk reserve and preserve all existing weights. No full model
download is authorized by this attempt. Sealed disposable outputs may be
retired only after preserving receipts. Runtime data stays private; publish
aggregate hashes/counts/timings. No PR or release promotion is claimed.

### Resource hold before native validation

After the Qwen resource phase, free space was observed at about 6.1 GiB,
below the 8 GiB reserve. The new Gemma worktree had no build target or assistant
payload download; only source, metadata and aggregate driver preparation were
present. Native compilation, Metal/model checks and the bounded companion
download remain deferred until reserve is restored. `vm.swapusage` then
reported 11 GiB allocated and about 9.5 GiB used; without a before measurement
this does not establish the cause of the filesystem change. Existing model
weights and prior build/evidence artifacts were not deleted.

### Additional predeclared assistant reference gate

Before the first actual E2B reference run, the one-step component gate uses
Transformers 5.15.1 eager BF16 CPU, the exact native committed target KV,
normalized final target activation, and fixed next-target position. Tensors pass
only through ephemeral subprocess stdin; the receipt retains aggregates only.
Compare feedback, centroid scores and masked logits: maximum absolute error
no greater than 0.5 and relative L2 error no greater than 0.02, with identical
greedy first proposal. Independently read only the root target embedding row
from the selected target safetensors and require exact scaled BF16 equality.
This bounded one-step oracle does not qualify target acceptance or rollback.

The first Gemma3 lifecycle receipt passed 33 controls before a driver expectation
failed at 1,025 tokens: policy requires at least two suffix tokens after a retained
boundary. The corrected control explicitly requires zero reuse there, adds
1,026 tokens, and passed all 40 cases. The failed receipt is retained unchanged.

### Native build and Gemma3 execution receipts

Rust 1.98.1 release worker SHA-256
`00c2c6bed00b0d7bf848978611bf4e3c828305805675ef33cd3f15a904ab7ebe`;
matching template helper `8d79b1c2fa652749b981c4336ff27193ff615d7f96589ca35ad06578d3ddf587`.
CPU/Metal committed-KV tests: two passed. Existing prefix regression filter:
32 passed, eight ignored. The first scratch-lifetime fixture failed on CPU;
the fixture and component retain scratch through the owning submission, and
the corrected CPU/Metal controls pass. Original failed logs remain private.

Exact original-BF16 Gemma3 4B weights
`50b031efe933a6023b4188a718b1f4e2d654d6eac5b5c9336205de59a3426805`,
config `69bd51786943ca8320926a2359947e618fc11a8f8cf09f1640329f48b75bcc98`.
Corrected lifecycle receipt `f72df229e0ba55a5424eb9b0864174948ed0db244035648f81a4c5490c34e418`:
40 cases passed, including actual image zero reuse, post-image text restore,
nonperiodic/changed-tail greedy and seeded equality, scope/context changes,
cancel/error/reset recovery and 892/1024/1025/1026/1792 boundary cases.

The text/image ABBA arms had six measured samples each, matching 64-token
output hashes, zero paging growth and eligible host guards. Median decode:
text 68.10 tokens/s, image resident 67.04 tokens/s (1.55% lower); first token
26.29/21.82 ms; whole request 966.94/977.29 ms. These paired arms meet the
predeclared bound. The same receipt includes a separate previous-worker
baseline with four swap-ins in one measurement; its baseline gate is ineligible
and is repeated separately. Loading was excluded from all timing.

Actual E2B companion source SHA-256
`93682eb1c97639d18f007704dc880bd74cbe530adaf7b1bb561213863fdad2a6`,
157,565,344 bytes, pinned source revision as in the assistant plan. Native
schema-2 export SHA-256 `8b81d6337f1d3e8a562c13d57af43ce0cbd30698683a56eccadaf39f7549bb9f`,
157,581,400 bytes, 46 tensors, bounded 8 MiB conversion peak. No result
qualifies E4B, wide wrapped source handoff, verification/rollback or serving MTP.

### Actual E2B functional execution and failed numerical gate

The exact source-bound E2B target (config
`9acc540130bd576271650c3805467141136570ffc4111c186e317442cc3dc762`,
weights `5c26289c1536f6afda47e62616c02ae63df489dc5c37d6034624413b2fd75993`)
loaded through the normal image loader with its tower resident. The actual
schema-2 head emitted three proposals; their aggregate hash is
`9a4e16ae6e74a18e22a3c450d749536ae237a405aa4fe9db971bc8e0afc2bf40`.
The ordinary target draw equals the frame root, target position remains seven,
local/global owners are 13/14, and drained draft cancellation followed by retry
reproduces the same proposals. Target embedding lookup independently matches
the selected weight row and BF16 scaling exactly.

The pinned Transformers eager-BF16 one-step oracle has the same greedy first
proposal. Feedback maximum absolute error 0.4375 / relative L2 0.01609 and
centroid error 0.375 / 0.00908 pass the declared component bounds. Masked logits
fail: maximum absolute error 47.28125 and relative L2 0.02197. This is a failed
numerical gate; cluster-support discontinuity and score/probability rounding
need diagnosis. The tolerances are unchanged. Functional proposals do not
qualify source numerical parity, target verification/rollback, MTP serving,
head-aware prefix reuse or useful performance.

The first actual run found a missing tagged attention configuration type;
the second found input constant handles retained across Metal end-encoding.
Both are fixed, and their failed logs are retained. The third run reaches the
independent numerical gate above. Final dedicated CPU/Metal committed-KV
controls pass two tests, and prefix regression controls pass 32 with eight
ignored. Independent read-only reviews of Gemma3 tuple/graph dispatch and the
committed-KV/cancellation component found no additional source blocker; those
reviews do not replace the failed numerical gate.

The separate prior-text timing repeat is also ineligible: its first measured
request incurred four swap-ins and later requests exceeded the declared host
CPU guard. No old-text speed qualification follows from either observed trace.
Both receipts remain unchanged. Current candidate text/image paired speed and
cache correctness are qualified only for this exact original-BF16 Gemma3 4B
artifact. Larger profiles remain open. All heavy processes were closed and
the slot handed directly to GLM after this bounded phase.

Final native component test binary SHA-256
`071cbc221f3f7209ecbbcdec04cbacb99d24c4866e897ac8865ac33114a73855`,
source/receipt seal `970a151a4a4cdaf0ad154e47c57da681de2d6d1863320f0eeb815b58111e7d3d`,
actual E2B aggregate receipt
`2c2ce563229ceaf49de9b395d0ef977168b08c07b42aba732fc80ec8a7fd2b47`.
All tensors sent to the reference subprocess remained in ephemeral stdin;
receipts contain counts, hashes and numerical aggregates only.

## 20261109-G4AP-02: source BF16 committed-query arithmetic

Registered before the follow-up edit or execution. The first source oracle
failed without changing its declared limits. Source review shows that the
assistant eager reference materializes BF16 attention scores and probabilities,
while the initial native head used fused F32 online softmax. Implement that
source arithmetic only for the dedicated immutable committed-KV query, reusing
the bounded temporal ring copy and existing materialized kernels. Preserve
ordinary attention dispatch, source position and partial-ring guards. Rebuild
and repeat the same exact E2B artifact/input/oracle gate when the serialized
resource queue grants the slot after GLM and Qwen. No result or tolerance
transfers from the failed first arithmetic policy.

### Target verification component to add before the next build

Add a consuming, serial greedy verification control: accept only already
verified target input rows, compare each draft to the actual next target draw,
and return a new frame plus accepted outputs. Cancellation or failure consumes
and discards the private frame; no partially advanced state escapes. Test
outputs against ordinary target generation, including deliberate wrong drafts
and cancellation followed by fresh prefill. This establishes functional
verification without entering unsupported partial ring acceptance. It is not
a batched verification implementation or useful-performance result; serving
integration and transactional batched ring acceptance remain open.

The serial component returns committed tokens and the next unprocessed draw
in separate typed fields. Its controls require `new_position = old_position +
1 + accepted_drafts` and `committed_tokens.len() = accepted_drafts + 1`, so a
future caller cannot mistake the carry draw for already committed KV. The
independent source review confirmed the underlying boundary arithmetic before
this result type made the distinction explicit. These follow-up controls remain
unbuilt until the next resource slot.

The follow-up oracle adds aggregate diagnostics for centroid-boundary ties,
selected-token support differences and numerical error on common selected
rows. It retains the original full masked-logit acceptance limits unchanged.
No centroid scores, token IDs or tensors are logged; only support/tie counts
and numerical error aggregates leave the ephemeral reference process.

Before the same build, add explicit source-bound E2B companion admission:
exact file size/hash and selected-descriptor construction, tied to the loaded
target config/weights and opaque model-instance identity. Reject unbound or
foreign heads at the language-model draft entry. This component loader does
not enable serving/default loading or infer support for E4B/larger profiles.

### Crash recovery checkpoint

The 2026-11-09 machine restart interrupted the unbuilt follow-up. Recovery found
the saved `dccb8ed0` head and all nine changed paths intact. The recovered unified
patch exactly matches the pre-restart unbuilt patch (SHA-256
`0fc9202a4d36370073dd2142919fa8a859a3c268c098a6a98eb50a85157867ea`).
Separate private recovery receipts preserve the full source files, patch and
artifact hashes. No native worker or KV state survived the restart, and no
new numerical result is inferred from the retained prior binary.

Fresh full-file reads confirm the unchanged exact target config/weights,
157,565,344-byte source companion, 157,581,400-byte native companion and pinned
Transformers 5.15.1 assistant/candidate/masking sources. The support-diagnostic
oracle revision is sealed separately. Package formatting and repository policy
checks pass. Added serial controls cover one accepted proposal followed by a
rejection and zero proposals. These controls and G4AP-02 remain unbuilt pending
the explicit serialized resource handoff. The crash cause is undetermined.

### Post-restart native build and held execution

A fresh distinct ignored target compiled with pinned nightly-2026-10-01,
locked dependencies, two build jobs, release debug zero and incremental off.
The first build found two test-only `tempfile` calls without a library dev
dependency. Reusing the existing private test-directory helper fixed that
blocker without changing dependencies. The second build passed. A subsequent
test rebuild was interrupted to apply formatting; the final fourth build
passes. Original failed/interrupted logs remain private.

Final test executable SHA-256
`4d43d120a350dab9167ca27242865f494e5403068e1c7480110235fb43ede993`;
1,185-file source graph digest
`ae5abe4196b0d8512841035a29bb823ab23013405f6970e83b6f797f3089b3cb`;
source-seal file SHA-256
`a0c5ab9bf1dcbcad3eb5ed2e70762c424f974092b329142f539d3b1313908527`;
build/receipt seal SHA-256
`88ef83ce24e62eb17dd437504ada488ff1d7d05f5c43023d0bc585306667d8ed`.
The final executable passes both exact committed-query CPU controls, including
BF16 phantom-suffix exclusion and ring scratch ownership. Formatting,
repository policy and whitespace checks pass. An independent read-only review
of patch `fb9cc8fb` found no additional defect in serial commit/carry arithmetic,
cancellation draining or the committed eager attention geometry/lifetimes.
That source review supplies no numerical or performance qualification.

The earlier broader prefix filter passed 22 tests with six ignored. Skipping
test names ending in `_metal` did not exclude generic tests that invoke
Metal internally. This was a mixed CPU/Metal diagnostic run, with no model
load; its log is retained separately. Its earlier executable hash was not
retained before the final test rebuild. It is not labeled CPU-only or used to
close the new source-bound Metal/oracle gates.

The assistant component now rejects draft positions above 512 until the source
513-row handoff is implemented and qualified. Ordinary target context support
is unchanged. Added model-bound controls admit a draft at position 512, reject
position 513 and continue ordinary serial target state to 514; these controls,
all serial acceptance/cancellation cases and the unchanged numerical oracle
remain unexecuted in this follow-up.

The next exact local command is `zsh runtime/gemma-recovery-next-command.sh`,
whose SHA-256 is
`dad3231a47870d65f89ee30c8397e8240f43e06094ae84445245ed307dd99e04`.
It selects exactly two dedicated Metal controls and then the same pinned E2B
artifact/input/oracle with a new, create-only aggregate receipt. The script
references the sealed final test executable and artifact/reference paths;
`zsh -n` passes. Before running it, the coordinator must clear the externally
loaded user-owned Stark worker and explicitly grant the serialized resource
slot. Recovery did not kill or unload that worker. No Gemma model was loaded
after restart. The masked-logit limits remain maximum absolute error 0.5 and
relative L2 error 0.02. The prior failed oracle, old-text benchmark ineligibility,
E4B/wide-handoff gaps and disabled serving MTP remain unchanged.
