# Image-support work: findings, failed attempts and remaining qualification

## Scope and evidence

This report reviews the **Image Support for Stark Models** conversation and the
implementation, numerical, lifecycle and review records it produced. All six
turns exposed by the desktop conversation-history reader were reviewed; it
reported no older page. That history is useful for intent and chronology, but
does not expose every internal command or replace a source-bound receipt.
The dated records linked below are the evidence for numerical claims.

The documentation snapshot starts at main commit
`7dc5183499eb9248ab6805474291980dbf0a57db`. Model results remain bound to their
original builds, not automatically to that newer commit. No model was rerun and
no new timing measurement was made while writing this report.

The detailed ledgers register **39 Qwen entries** and **26 Gemma entries**:

- [Qwen attempts and checkpoint identities](2026-11-08-qwen-images.md)
- [Gemma attempts and checkpoint identities](2026-11-08-gemma-images.md)
- [How to record the next model change](README.md) and [attempt template](TEMPLATE.md)

An entry can describe an implementation change, a negative control, an invalid
setup, a diagnostic or a replay. These counts are not counts of independent
performance experiments. The ledger-registration date is separate from the
historical execution dates. Missing measurements remain missing.

## What was requested and delivered

The initial question was which models still lacked images. Dense Qwen3.5 BF16
already had a working path; Gemma 4 E2B BF16 also had earlier branch evidence.
The new targets were quantized Qwen dense checkpoints, Qwen3.5 MoE, Qwen3.8 dense
27B and retained Flash-Next/Lily, Gemma 3 multimodal, and other Gemma 4 sizes.
The work was split by family and compared with pinned upstream implementations
and prior Stark paths. Pause/resume preserved completed references and used new
output directories rather than replacing earlier evidence.

The resulting source supports explicit one-PNG/JPEG image artifacts through the
converter or retained preparation path, native admission and execution, worker,
terminal chat, desktop host and local API. Each artifact needs its own local
checks. Implementing those routes did not qualify every checkpoint size. All
image requests clear/bypass prefix reuse and disable image MTP; Gemma text ring
restoration remains a separate documented reuse blocker.

The integration also carried the prerequisite OLMo 3/Gemma 4 native foundation.
That historical text work is described in [OLMo 3/Gemma 4 support](../OLMO3_GEMMA4.md)
and its linked numerical/performance studies. OLMo 3 image support was not a
result of this conversation. GLM-5.3-Flash and Flint image execution were not
implemented here. Their existing text paths must not inherit Qwen/Gemma image
qualification by analogy.

## Findings by model and artifact

| Model / actual path | What worked | What remains unverified or unsupported |
| --- | --- | --- |
| Dense Qwen3.5 BF16, pinned 0.8B reference | Existing source-bound numerical and image execution baseline; 744,960 final logits exact in the recorded replay. Eight answer comparisons agreed with the source, but only six were correct in that small fixture set. | Agreement is not universal answer correctness. Other compatible sizes/fine-tunes require their own artifact checks. |
| Qwen3.5 0.8B quantized decoder + BF16 tower | Both fixed packed-reference cases matched all 248,320 logits per case exactly after the arithmetic repairs. | This does not qualify arbitrary quantizations or quantized vision towers. |
| Dense `Jundot/Qwen3.8-27B-oQ4e-mtp` | Complete packed-decoder image conversion, extracted BF16 tower checks and real image/cancellation/recovery worker lifecycle passed. | Full large-decoder source-logit parity and image throughput were not established. This artifact is distinct from both BF16 Qwen3.5 and Flash-Next. |
| Qwen3.5 35B-A3B, affine MoE image path v4 | Complete BF16 tower plus affine expert/decoder inventory; pinned artifact tower and actual worker lifecycle passed. | Does not transfer decoder parity or memory/throughput guarantees to other MoE sources. |
| Qwen3.5 BF16-source MoE image path v5 | Separate conversion/admission and controlled router, selected-expert/full-bank and scratch-request tests passed. | Full official 35B loading, source parity and image worker execution remain unverified. |
| Retained `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl` | Schema-6 source/artifact binding, all 27 tower layers exact on both source fixtures, independent stage controls and actual worker lifecycle passed. | Full retained-decoder numerical parity and image throughput remain unverified. |
| Retained `ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit` | Explicit image preparation/manifest path implemented. | Its distinct large artifact has no image execution qualification in this conversation. |
| Typed Uzu Lily text conversion / original PipeNetwork text route | Text behavior remains separate. Typed text conversion intentionally omits the tower; the PipeNetwork text loader has no new image-fusion bridge. | These are not image-enabled merely because a related retained format contains vision weights. |
| `google/gemma-3-4b-it` BF16 | Exact preprocessing patches; three canonical feature/logit gates and three eight-step greedy trajectories passed, followed by 14 worker/API lifecycle gates. | Full logits are within the fixed error limits, not bit-exact. A small synthetic fixture set does not establish general image-answer quality. |
| Gemma 3 12B / 27B BF16 | Original-wrapper structural conversion/admission implemented. | Actual full-model numerical replay and worker execution remain unverified. |
| Gemma 4 E2B / E4B BF16 | Separate source/execution evidence; final worker replay passed all three retained canonical cases for each. Strict decoder admission preserves their recorded arithmetic. | Final retained greedy replay is not full logit parity for all newly converted canonical artifacts. Image throughput and installed desktop UI execution were not measured. |
| Gemma 4 12B BF16 | Encoder-free patch/embedder route; extracted source-bound three-image component comparison passed. | Full 12B decoder generation remains unverified. |
| Gemma 4 26B-A4B BF16 | Distinct no-PLE/global-K=V decoder, standardized tower route and bounded MoE controls implemented. | Actual released expert/decoder weights and full 26B generation remain unqualified. Synthetic MoE controls are not a checkpoint replay. |
| Gemma 4 31B BF16 | Extracted tower passes all three original feature gates after source-compatible scalar repair; largest relative RMS is `0.0011068880`. | This is component similarity, not a bit-exact full tower or full 31B generation proof. A broad scalar control retains one known source difference in 106,525 cases. |

Exact repositories, immutable revisions, artifact hashes, numerical policies and
per-attempt evidence are in the two family ledgers. The
[integration record](../archive/runs/2026-11-07-image-family-integration.md)
and [pre-PR review](../archive/runs/2026-11-07-image-family-pr-review.md)
bind the aggregate qualification and its limits.

## What the failed attempts taught us

### Matching architecture and nominal precision was insufficient

The affine Qwen encoder needed the source's BF16-effective coefficients and F32
contraction order. Merely retaining weights and calling the same mathematical
operation did not produce matching logits. The final small packed source
comparisons are exact without materializing a dequantized weight bank. Larger
tower/lifecycle results remain a different evidence class.

Retained Lily needed successive repairs to contractions, nonlinearities,
normalization, positional/rotary handling and source-ordered streaming
attention. Several intermediate candidates still failed a tower gate. The final
two source fixtures are exact at zero, one and all 27 blocks. The legacy 54
normalization controls test the legacy policy; they do not measure the final
Welford production normalization. The production attention avoids quadratic
N-by-N scratch, but that storage result does not establish faster requests.

### Solid-color images hid real defects

Gemma 3's nonuniform image exposed a processor mismatch: fixed Pillow 22-bit
resize coefficients differed from the pinned CPU processor in 799 patch
elements. After fixing resize, the patch convolution still used the wrong
reduction order. Matching the source's four streams and bias/cast boundary
reduced pattern feature relative RMS from `0.0338022` to `0.0000713012`.
Both failed replays remain preserved. This is why a pattern case and independent
stage captures are needed alongside red/blue controls.

### A more accurate scalar formula could still disagree with the pinned source

Gemma 4's larger tower required source-compatible logical-width normalization
before zero padding,
QK contraction, fused denominator behavior, exponential materialization and
rescaling. Exact captured attention did not by itself make all original full
feature cases pass. The nearest-F32 scalar candidate passed two cases but the
pattern still failed at relative RMS `0.0264942683`.

The source-compatible 128-entry quadratic policy subsequently passed all 8,220
native source scalar controls on CPU/Metal and all three unchanged feature
gates. The earlier 8,215 nearest-policy controls remain separate; their 42
one-ULP source-libm differences were not silently reclassified. The broader
106,525-input check still has one source counterexample, so no universal
source-bit-exact scalar claim is justified.

Source-informed synthetic Gemma no-PLE trained-gain controls passed while a
sharper six-layer unit-Q/K
stress remained a precision blocker. Passing a representative component did
not resolve that negative control or qualify full 12B/26B/31B generation.

### Artifact labels were weaker than the decoder contract

Independent review found a Gemma 4 named image profile could accept changed
decoder semantics, including layer count or global RoPE. Artifact hashes were
distinct; this was admission/provenance correctness rather than evidence
inheritance. The fix hashes and parses the same pinned source bytes and compares
the full typed decoder against its canonical profile. Actual retained E2B and
E4B configurations are controls. E2B's historical paired-false embedding policy
is accepted separately from the current paired-true policy; mixed pairs fail.

## Integration, invalid setups and delivery log

| ID | Attempt / outcome | Evidence and disposition |
| --- | --- | --- |
| IMG-20261108-XI-001 | Template helper missing or resolved through the wrong directory: **invalid setup**. | Preserve the failed aggregate receipt; check the actual helper and live Python module paths before any numerical/regression claim. [Review record](../archive/runs/2026-11-07-image-family-pr-review.md#validation). |
| IMG-20261108-XI-002 | After merging main, an older sealed template helper produced 26 converter failures: **mismatched tool setup**. | Rebuilt the current Rust 1.98.1 helper rather than treating template failures as image arithmetic defects; final suite passed. Helper SHA256 `b02fbabb7ffa56ecdea3ccfb73a3088d00a11e1441326501580d83d19491ea27`. |
| IMG-20261108-XI-003 | Final formatting check found seven unformatted converter files: **failed check, then repaired**. | Ruff formatting/import ordering committed as `a612db68`; full live converter run followed. [Review](../archive/runs/2026-11-07-image-family-pr-review.md). |
| IMG-20261108-XI-004 | Main synchronization had host imports/controller-test conflicts: **resolved and checked**. | `b5ef8164` retained both protocol-stop and image routing and both capability checks. Controller passed 134 checks; numerical core identities were checked separately. |
| IMG-20261108-XI-005 | Documentation review found stale support/failure wording, a retained-Lily setup link, UI label and benchmark-copy claim: **corrected**. | README/native guide distinguish actual formats and qualification. Website v62 preserved the existing hosted layout; four HTML pages/72 local references and desktop/mobile views were checked. Historical GitHub targets absent locally were not independently verified online. |
| IMG-20261108-XI-006 | Final integrated regressions: **passed within their suites**. | 6,604 converter tests passed, 96 skipped; 931 local CPU/Metal tests passed, 130 ignored and six benchmarks filtered; 134 controller checks passed. These counts do not execute every supported full checkpoint. |
| IMG-20261108-XI-007 | Independent review and hosted checks: **passed, then merged**. | PR [#97](https://github.com/jshahbazi/stark/pull/97) head `dcb932283548569073d17fd45af3af025779d36d` had successful [CI](https://github.com/jshahbazi/stark/actions/runs/42436850726) and [Security](https://github.com/jshahbazi/stark/actions/runs/42436850685), plus the other checks, before merge `b9a91a94ea07313cc0558d556f26a300c6603ff1`. Hosted engine coverage includes Metal compilation and CPU/grammar execution; local Metal execution is separately recorded. |
| IMG-20261108-XI-008 | Signed patch release: **published**. | The [signed workflow](https://github.com/jshahbazi/stark/actions/runs/42438276063) succeeded, and [v0.18.3](https://github.com/jshahbazi/stark/releases/tag/v0.18.3) was published at `2026-11-08T02:53:29Z`. Its tag points to the PR #97 merge commit. This report verified GitHub workflow/release/tag state, not a new receiving-Mac installation or rerun of every model. |

The final native test executable is bound to `f8293f87`, SHA256
`f7b3372d90de9e5a3fa5b6097f22720492df965ed450f35adc9031c59bacc93b`.
The final release-worker replay executable is
`a49cb0620f5634772712d9ce44c6fbb2c61f845593d2cb05d9b0cb7fc5296988`.
The converter run began at `a612db68`; documentation commit `94697f10` landed
during execution without changing converter sources. Earlier 6,307/928/122
test counts describe earlier suites/builds and must not be substituted for the
final integrated results. Full hashes and receipt links are in the review record.

Lily's large worker retry is recorded in the Qwen ledger as **resource-deferred**
before a later pass. Another worker reduced available memory below the retained
admission bound; only task-owned processes were stopped. The later successful
replay sampled roughly 61.3 GB owned RSS. This is neither an out-of-memory
numerical failure nor a throughput result. The E2B/E4B final replays also had
competing work present and make no quiet-host speed claim.

## Performance conclusions and next work

This conversation qualified image correctness and lifecycle behavior for the
named artifacts. It did **not** establish an image end-to-end throughput gain,
an image-MTP speedup, general image-answer quality or installed-UI execution.
Streaming scratch bounds and exact arithmetic controls are useful implementation
results, but cannot substitute for a screened paired performance study.

The next model work should be registered before running it:

1. Qualify actual complete BF16-source Qwen MoE 35B, Gemma 3 12B/27B and Gemma 4
   12B/26B/31B artifacts, with resource admission and the original fixed gates.
   Actual 26B released expert/decoder weights need their own replay.
2. Keep retained and newly converted Gemma artifacts separate. Add full source
   logit/trajectory evidence for new canonical arithmetic rather than borrowing
   the retained E2B/E4B greedy replay.
3. Investigate the no-PLE unit-Q/K stress and the remaining broad scalar
   counterexample as separate attempts. Do not weaken the original feature gates.
4. For image speed work, fix checkpoint/quantization, image sizes, token lengths,
   sampler/MTP/reuse policy and host screens; run baseline/candidate paired
   end-to-end measurements after correctness passes. Record invalid or excluded
   arms and distributions, not just the best rate.
5. Qualify additional image-answer tasks and installed desktop interactions
   separately from tensor parity and worker/API cleanup.

The new ledger rule in `AGENTS.md`, onboarding link, template and benchmark-index
cross-link make future attempt recording part of the repository workflow. The
ledger is maintained Markdown; no automatic collection, exhaustive historical
backfill or new model execution is claimed by this documentation change.
