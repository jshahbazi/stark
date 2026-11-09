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

### Exact E2B follow-up execution after the resource hold cleared

After the user unloaded the installed Stark model, the coordinator confirmed
no worker/compiler processes, 87% free memory and zero swap before granting
the exclusive first execution slot. The saved script ran unchanged on source
`9053f805` and the sealed final executable. Both dedicated committed-query
Metal controls pass. The exact E2B component/oracle test passes in 40.57 seconds,
with 10,417,152,160-byte peak footprint and zero swaps.

The unchanged pinned one-step eager-BF16 oracle now matches feedback, centroid
scores and masked logits exactly: maximum absolute and relative L2 errors are
zero for all three. Root embedding and first greedy proposal are identical;
selected-token support difference is zero across 4,096 selected tokens, with
no cutoff tie ambiguity. This closes the declared short-input component
numerical gate for the new assistant-only BF16 score/probability policy.
The original failed G4AP-01 receipt and its original arithmetic remain separate.

Three actual proposals execute with the image tower resident. Their aggregate
hash is unchanged. **Zero of those three actual proposals were accepted** by
the target on this input. Serial ordinary-output equality, supplied all-accept
and partial-accept controls, deliberate rejection, zero drafts and drained
cancellation/fresh recovery all pass. These supplied acceptance controls do
not establish useful real-model acceptance or speed. Position-512 drafting,
position-513 rejection and ordinary target continuation to 514 pass. The
boundary control is functional; numerical parity at that boundary remains open.
Strict wrong-size/hash/symlink/target/instance admission controls pass.

Aggregate receipt SHA-256
`770a6bc0ba68b628a445d27956c027dfdcc0d21f910e95cbf647eff3de6b61a1`;
execution/receipt seal SHA-256
`c2a53fd8b6cc97747f8ec4cefd2326521fff9340ed2680c69cfd526d334c2ae4`.
All owned model/reference processes closed before the explicit Qwen handoff.
Serving MTP remains disabled. Normal-worker integration, head-aware prefix
reuse, multi-round/lifecycle controls, wide source handoff, useful real
acceptance and matched text-only/image-loaded MTP performance remain open.

## 20261109-G4AP-03: source handoff and normal-worker qualification

Registered before new execution. Preserve the passed G4AP-02 source/binary and
exact short-input oracle. Add a separate immutable local-KV handoff captured
before ordinary target acceptance, retaining the source-visible inclusive
513 rows while the ordinary target ring remains 512. Qualify wrapped/near-cap
handoffs and chained proposals against the same pinned oracle before removing
the short-context draft guard.

For a useful target verification attempt, submit a flat full-accept batch of
the root plus actual drafts. Capture physical target snapshots before that
batch. On rejection, restore the complete snapshot and repair only the verified
prefix through the already-qualified serial route; never bypass ordinary
partial-ring guards. Require fresh ordinary target equivalence for real,
all/zero/partial/rejected proposals, cancellation and context boundaries.

Then integrate a separately disabled-by-default worker feature for exact E2B
companion admission, greedy text speculation, complete head/mode-bound prefix
reuse and image exclusion. Qualification must use the normal worker protocol,
real images and actual proposals/acceptance. MTP0 keeps the existing ordinary
text path. Do not advertise serving support or qualify speed from components.
Before changing the production default, require complete lifecycle correctness
and matched text-only/image-loaded MTP0/MTP-on performance with frozen
artifacts, builds and quiet-host guards. Failed speed or acceptance attempts
remain recorded and cannot be relabeled as acceleration.

### G4AP-03 first long-input gate: failed, preserved

The fresh distinct engine target first failed compilation because the new
handoff module lacked the `Buffer` trait import. After that import, the second
locked nightly release build passes. Test executable SHA-256
`e225f41a1cd02dd1812c41dc72413b65e35c58d38daea28e5a4090963f11af81`;
source graph digest
`8aa840984a76fd25d9a564d25dd9025a573e394e04cdc6894ac819571d78d397`;
source-seal SHA-256
`674be333ec8a9b4d1fc6970b23f715bdf85d1867f52164c6d3247cf4757e30b2`.
All six dedicated CPU/Metal controls pass, including exact preaccept temporal
row retention at short, 512/513, wrapped and near-cap synthetic positions.
The independent immutable source review found no confirmed physical rollback
or scratch-lifetime defect; execution still has to establish those gates.

The separate three-step oracle is SHA-256
`be71c7475d16adaba90c8a0842ebad2ed0653944b2ff53df71a6cf3bd63fe3dd`;
its imported pinned Transformers Gemma4 text implementation is
`4f874549f79deda4bb9ce7fb7d7b8e7349122be1711f733f7a4157e46ed10cc3`.
It chains reference feedback independently and keeps the original maximum
absolute 0.5 / relative L2 0.02 limits. The original short oracle is unchanged.

The first 512-token case fails before target batch/cache controls execute.
Step 1 feedback 0.375 / 0.00821 and centroids 0.25 / 0.00654 pass, while masked
logits fail at 55.125 / 0.01727 with 256 selected-token support differences.
Step 2 passes all planes. Step 3 feedback fails maximum absolute error at
0.625 / 0.01417. Root embeddings and all three greedy proposals match; that
does not establish numerical parity. Cases 513/1024/1026/1792 and all actual
transactional batch, rollback and cache controls remain unexecuted.

The reused unchanged Qwen resource watchdog reports no foreign worker, no
paging growth and no resource/deadline stop; maximum observed test RSS is
10,741,563,392 bytes. Aggregate oracle receipt SHA-256
`145a3048826a2752e1e98fc99cd02db2561e5871541ce32b72b4f0d0624a9183`;
process receipt `b012759ba2133ab40156dad04d32678096cab5f9c92887b66a2ede150f019f61`;
failed-gate seal
`9f3cd5141d088789c9086cf0d8145fdcb29e1e7cce3aae0e2435c1b637129b13`.
All owned model/reference processes closed before the direct GLM handoff.

Next diagnose the first divergent intermediate and centroid cutoff support
before changing head arithmetic. Do not loosen tolerances, remove the public
512-position guard or enable worker serving. The exact G4AP-02 short result
remains qualified only for its original source, binary, input and one-step scope.
Normal-worker integration must retain published Qwen commits `48c5aa52` and
`e4144be5`, including the ordinary headed MTP0 dispatch/cache/memory fix.

The following integration merge retains that complete published Qwen lineage.
Its only code conflict resolves the shared snapshot-cache rename while keeping
Gemma3/Gemma4 capture, restore and suffix paths. Both benchmark index entries
remain. Formatting and whitespace checks pass; a native build and checks on
this merged source are still required in the next exclusive build slot. Prior
Gemma/Qwen binaries and artifact results retain their original source scope.

### G4AP-03 diagnostic control: first divergence

Before changing arithmetic, add test-only staged capture for the same failing
512-token three-step input. Compare pre-projection, per-layer normalized input,
raw/normalized/rotated queries, BF16 attention scores/probabilities/weighted
values, attention output, residual/MLP boundaries, final normalization and full
readout against pinned independent reference stages. Capture tensors only in
ephemeral subprocess stdin; retain first-divergence names, counts and error
aggregates. Require instrumented and ordinary native outputs to match exactly.
Report common-selected readout error, centroid support/tie counts and chained
feedback error separately. This diagnostic supplies no tolerance change,
production arithmetic change, serving capability or performance claim.

The staged diagnostic source passes package formatting, Python syntax and
whitespace checks; compilation/execution are queued after GLM and Qwen.
Immutable native diagnostic patch SHA-256
`4767179faa7ce66b4cef27e0e04f0287cb0c4e3d086f5320e7d38e89d4673625`;
diagnostic oracle SHA-256
`7a9b831e64cf8ca4c51b778e2f779694518e8c303d1075837e6978b88d308d7c`.
The latter additionally compares native stable centroid ties against Torch
top-k on the same native scores using support-difference counts only. The
preceding diagnostic tool revision is retained separately. No new head
arithmetic or runtime result is implied by this source checkpoint.

### First-divergence execution: MLP boundary isolated

The merged diagnostic source `d2dc5ce5` compiles in a new distinct target.
Executable SHA-256
`b636b695c59181c45398abf51ec1d99eeccb975221316a493973c8b1fe2713a9`;
source graph digest
`c59e840336ea38f456385f6b522d9f9ccf7bcd25513b60b47bd0372e61325f46`;
source-seal SHA-256
`7cddc98448075ae33a54fd9dbf497890873259cfc462b81992e7a259881f90e5`.
The guarded single 512-token diagnostic completes, and instrumented final
feedback, centroids and full readout exactly equal ordinary native encoding
on all three chained steps. This validates the instrumentation, not parity.

On step 1, the first independent-reference difference is **layer 1 MLP output**:
one of 256 values differs, maximum absolute 0.015625 / relative L2 0.000374.
The operator-isolated comparison fed native inputs identifies the same boundary.
Every preceding pre-projection, normalization, query projection/rotation,
attention score/probability/value and layer-0 MLP boundary is exact. Additional
isolated errors later are small: one layer-2 MLP value at 0.00000191, one global
query projection value at 0.0000305 and one feedback readout value at 0.0000305.
The initial MLP difference propagates through independent feedback, producing
step maximum feedback errors 0.375, 0.25 and 0.625. Later-step pre-projection
differences already receive different feedback; they are not the first cause.

Step 1 swaps one selected centroid cluster: symmetric support differences are
two clusters / 256 vocabulary positions. Native stable selection and Torch
top-k on the same native centroid scores agree. The native cutoff is unique;
the independent reference cutoff has two tied clusters. On the 3,968 common
selected tokens, maximum logit error 0.5 / relative L2 0.00371 passes, while
full masked-logit maximum 55.125 fails through changed support. This separates
the support discontinuity from the smaller underlying arithmetic error.

Diagnostic aggregate SHA-256
`3cc8f86b810c1236edc2e2e67e69b29479e6646853bf8e4bec198caad3a78704`;
process receipt `1eccc0a804f21d40a96966d80240b5055ac384b0c8f3e1c438cc2d45ca7b853d`;
execution seal
`bfb1b82679ea03541646092ff286cbe314746294ca42897d004cbc3522eb529f`.
The continuous guard records no foreign worker, no paging growth and peak
observed RSS 10,892,689,408 bytes. All owned processes close before the GLM
handoff. No head arithmetic, tolerance, public guard or serving setting changed.
Next split the MLP into fused up/gate projection, GELU, BF16 product and down
projection to identify the exact primitive before changing its policy.

The next staged diagnostic exposes fused up/gate projection, standalone legacy
GELU, actual BF16 product and down projection, with independent and native-input
operator comparisons. It also measures the existing pinned-source
`VisionGeluBF16` kernel and its separate BF16 product/down path as diagnostic
candidates; those candidates do not feed normal native output or feedback.
All additions are test-only, and production arithmetic remains unchanged.
The previous `b636b695` executable is preserved as an independently hashed APFS
clone before reusing this same checkout's distinct diagnostic target for the
next build. Every new candidate still needs its own source/binary/tool seals.

The finer test-only source `4e316404` compiles in 1 minute 52 seconds with the
same pinned locked nightly release settings. New executable SHA-256
`ec3a8e6bde5dfd6789dd1c8f3fc7025e67b9a9c7502065973b6f0e760515efb2`;
source graph digest
`36af1ba38163ba7429a1a32cc7af5366c1359a8c01c501fade6c32262c88a40d`;
source/binary seal
`04d564a5dfdb1225e7ff70bb47082d4d3d439e16e8f08ea8691b814f7568263b`.
Its separately versioned MLP oracle is
`25c1b5a3b1d1a08784e0e3ed88d001db288dfb93f777adf4c1c242ad6f194cad`,
including reference stable-sort versus Torch tie-support counts. Execution
remains held while the external resource-budget native-test reservation owns
the GPU window. No head arithmetic has changed.

### Finer primitive execution: source GELU candidate is exact

The single guarded 512-token three-step MLP diagnostic completes on source
`4e316404`, with ordinary/instrumented outputs and duplicated MLP outputs
identical. The finer first difference is legacy GELU in layer 0: 18 of 2,048
activation values differ, maximum absolute 0.0000153. Actual BF16 products
match the product of native activation/up values. Sparse fused-projection and
down-projection differences also exist and remain separately visible.

The existing pinned-source `VisionGeluBF16` candidate matches source activation,
BF16 product and down output exactly when given the same native inputs, across
all four layers and all three steps. This is causal primitive evidence for an
assistant-only activation change; it is not a rerun of the complete corrected
head or a closed numerical gate. Neither native nor reference stable-sort
selection differs from Torch top-k on its own centroid scores in these cases.
The old masked-logit/feedback failures remain unchanged.

The continuous guard reports no foreign worker or paging growth and observed
peak RSS 10,920,427,520 bytes. All owned processes close before the explicit
GLM tiny-control handoff. Aggregate/tool identities are recorded separately
below, and the old `ec3a8e6b` binary is retained as an immutable APFS clone:
aggregate `f8ec889d92658ef93b84d58514e911d6354ffedd909da24e5039378f8e222794`;
guard `d1c9695fe4fb387c37e5a8598d3cd17890db4fd7c19f907fe40588818598b902`;
execution seal
`c06c67fc82ce5bd232001f225498001cb88b9b3814dc4dc6c13400fa8d1ce4f4`.

## 20261109-G4AP-04: assistant-only source GELU

Registered before implementation/execution. Replace only the assistant's GELU
with the already source-matched `VisionGeluBF16` kernel and an explicit separate
BF16 product before its existing down projection. Keep all ordinary target
activation/attention paths and assistant legacy linear arithmetic unchanged.
Do not remove the public position guard or enable worker serving before the
same unchanged long/chained oracle and transactional/cache controls pass.
Bind the unchanged exact E2B artifacts to a new source/binary/policy seal.

The assistant now uses `gemma4-assistant-eager-bf16-source-gelu-v1`:
source GELU and a separately rounded BF16 product, with existing legacy
up/down linears. Independent review of immutable patch
`813a0b7e72e64c36fa4f21522b973a6b8e2a9de389937043cfd29e743ff9caf1`
finds no confirmed blocker; the extra 4 KiB live activation per one-row MLP
must be included in future serving admission. The locked j2 release test
build passes in 1 minute 40 seconds. Source graph
`af1d7656037926210b3d55cb3416ace7616d4fd287f1c6525b9c4b34a0eacc8b`,
executable `60de2cf152d89a31060cdf26d83ec5b9960decef7cb7e2ea62934364472dbed1`,
and source/binary seal
`64022cd3609f8009253c3813bc4029893085c91233a157dc966e626db6143db9`
identify this candidate separately from the previous diagnostic binaries.

The unchanged long oracle still fails its first 512-token case: first-step
masked logits have maximum absolute error 55.125 with one exchanged centroid
cluster (256-token support difference), and third-step feedback has maximum
absolute error 0.625. All three greedy proposals and root embeddings match.
The remaining long contexts and **all transactional batch/rollback/cache
controls are not executed**, because the numerical gate aborts first.
No tolerance, ordinary-target arithmetic, public position guard or serving
setting changes. The failed aggregate receipt is
`507d55792f6291ccffb28b0b35165f35580a3d6cb42af28fbd8b1f60f03c1df8`;
continuous-guard receipt
`8fa59af7425c79bfa82af6ce881cbdfdd22f37c4f1f670dc6fc14b70d462c6ef`.

One bounded follow-up uses the existing finer trace oracle unchanged. Its
instrumented and ordinary outputs remain identical, and the corrected
layer-0 activation/product/output are now exact. The first common-input
operator difference is layer 1 fused MLP up projection: one of 4,096 BF16
values differs by 0.000030517578125, which propagates through the product
and down projection. This moves the next causal investigation to legacy
linear accumulation; it does not close the full numerical gate. Diagnostic
receipt `39177cfb4fe99019d6cd060ea83b53606355eef8af142b06d935d64aa07128ea`,
guard `750cbe55b963e71fe3b10396674197b865a0b0742401a226c93e07688f0d1995`,
and execution seal
`10758dec148ca73fdc5bc7ac4e1d787019af5888adb0da9e4e2eb4e431ecd104`
preserve both outcomes. Both runs have zero paging growth and no foreign
worker; peak observed RSS is 10,744,102,912 and 10,913,726,464 bytes.
All owned workers, reference children and compilers close before the explicit
slot release. The next source-only step isolates the remaining linear
rounding using the same native inputs before any further math change.

### Next test-only linear isolation

Before the next build, register two additional planes in the existing finer
MLP trace: `ReferenceBF16Matmul` fused-up projection on the identical native
MLP input, and reference down projection on the identical native BF16 product.
These use the existing PyTorch-provenance eight four-lane F32 accumulator
schedule and the same admitted full-BF16 weights. Both are test-only and do
not feed ordinary head output, feedback or the target. A separate private
oracle extends the previous finer oracle with these two aggregate comparisons;
the long oracle and all acceptance limits remain unchanged.

The test-only projection build passes in 1 minute 41 seconds, and its single
512-token three-step trace completes. Both reference projections are exact on
the same native operands in every layer and step: 49,152 up-projection and
3,072 down-projection BF16 output values have zero difference from the pinned
source operators. Instrumented outputs remain identical to ordinary outputs.
The diagnostic does not feed the reference candidates into the head, so the
failed full numerical gate and unexecuted batch/cache controls remain open.
The companion loader has no shared-weight cache: diagnostic fields add
12,582,912 bytes of duplicate MLP weight storage in this test build. These
fields are excluded from production builds.

The standard immutable unified review artifact is
`e6b4f63c6fd6b8ee0eb9bc9a5f5169725b92766fe4bdda792822e68f9b7c09ba`;
earlier rendered review text is preserved separately. Executable
`bc7bcfb14ca8d9b1995c7efcc51aa5a740570caefeef74b829dab41c66923980`,
source graph `3c5ce9dd73e2dbb7ed49a953b7e1560fa476dc79319ad672d8e0f14bcfa79f81`,
source/binary seal
`431013303fb29ea1083d0b0da064ae46c4e769a4815b4ea36508f688f5593d4d`,
and separately versioned oracle
`0bd8ab8fd3ca851fde0f7b8615fc3573f6a7bca64ea8142828a57a86f745c6f2`
bind this experiment. Aggregate receipt
`29e187a0c188148c850b1bb2188fdd010c03a1118be6738ce9685d25c6503884`
and guard receipt
`6557478e33b4e5bc1d07460307d09c269fd382b826306470cd62f5f25cb8432b`
record zero paging growth, no foreign worker and observed peak RSS
10,898,489,344 bytes. All owned processes close before the slot release.
