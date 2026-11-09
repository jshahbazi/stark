# 20261108-LVMTP-01: text MTP with retained vision loaded

Registered before implementation and model execution, 2026-11-08. Owner: Codex.
Kind: retained model capability, memory admission, lifecycle and text performance.
State: completed for the named artifact and controls. Disposition: branch `codex/lily-vision-text-mtp`, based on
`64b454a7` from the still-open prefix-reuse PR #110. No installed/released change.

Problem: retained Lily image loading passes `with_mtp=false` and masks worker MTP
capabilities, removing the normal text optimization for the entire instance.
The vendor loader supports both resident vision and MTP weights. Image execution
already resets the model and explicitly turns MTP off before image prefill.

Change planned: load supported heads with retained images and report the same
text MTP capabilities as text loading. Preserve zero image drafts, image cache
invalidation, reset and MTP selection in the prefix key. Update the exact Lily
startup memory recipe for resident predictor weights and any additional state.
No kernel, weight precision, sampling or image arithmetic changes are planned.

Primary artifact: existing retained schema-6 Lily Q4-XL, repository
`fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl`, revision
`0f94842e3682b49acace4daeb4cf2e65a6f94f83`. Text and vision loads use the same
55 weight shard inodes (106,905,155,192 bytes), config and tokenizer. Do not copy,
download or reprepare weights. Bind worker/manifest/config hashes in private
aggregate receipts. Other retained sources require separate qualification.

Fixed acceptance: text MTP is advertised and actually proposes drafts; matched
greedy text outputs agree between text-only and image-loaded instances; image
requests execute with zero drafts; following text can use MTP again. Prefix
snapshots remain keyed by MTP selection and image operations invalidate them.
Model-free tests check protocol dispatch and memory inventory. Test a synthetic
image transaction with text before/after, plus a short paired text benchmark
using two drafts, a warmup and repeated measurements on each loading mode.
Keep prompts, responses and vectors in memory; write only aggregates and hashes.

GPU work is sequential on the dedicated M5 Max, 128 GiB RAM. Model-loading/hash
validation is excluded from inference timing. A short unscreened benchmark is a
regression sanity check, not general throughput or quality qualification. Stop
on load, numerical, lifecycle or memory failures and preserve failed receipts.

## Results

Implementation and the bounded artifact gates passed. The paired observations
and source identities are recorded below; the installed app was not changed. The existing prefix
implementation captures DeltaNet/convolution/predictor state at a 4,096-token
boundary; restoration matches loaded instance, context, scope and MTP selection.
Image operations always discard it. This change must not weaken those guards.

Initial protocol regression test expected the generic option-validation error
for an unknown image MTP field. The parser correctly rejects it earlier as
`invalid_request`; the test expectation was corrected before model execution.

The startup recipe adds 1,842,067,616 resident head bytes. MTP adds 23,089,152
state bytes (new total 417,705,984, within the existing 512 MiB reserve),
346,788,072 speculative scratch bytes and 83,886,080 extra prefill bytes at
2,048 rows. Decoder reserve increases from 256 to 768 MiB and prefill reserve
adds 128 MiB. Native request admission already includes loaded heads even when
disabled. These are typed allocation estimates, not measured process RSS.

Review also found the hybrid key advertised `mtp-selection` while snapshots
match an enabled/disabled boolean. It now reports `mtp-enabled`, matching the
existing restoration semantics and the validator's boolean-key handling.

## Completed validation

Inference source: `c93f76b4` (initial change `6a63323c`) on base `64b454a7`.
Stable Rust 1.98.1 release build, separate checkout target, default Metal
features, locked dependencies. Host: Apple M5 Max, 128 GiB RAM, macOS 26.7.2.
Original mixed Q4-XL target/head storage and F32 recurrent state, BF16 vision
tower, target arithmetic, sampling and upstream/vendor source remain unchanged.

- 91 model-free worker tests passed, including a text-default-MTP2 → image0 →
  text2 protocol regression and rejection of explicit MTP fields on images.
- Image/host and memory tests: 46 passed before the final allocation-margin
  refinement; all 14 memory tests passed again after that refinement.
- Rust formatting, converter Ruff and repository policy checks passed.
- Loaded image worker advertised both image input and MTP, kept default drafts
  at zero, and actually proposed drafts on all measured two-draft text runs.
- Greedy text outputs, seeded stochastic output and the long-prompt reference
  hash matched the previous text-only worker.
- At 4,104 input tokens/context 8,192, MTP2 repeats reused 4,096 tokens.
  MTP2→0 and 0→2 missed the cache; repeated MTP0 and MTP2 reused it.
  MTP2→1 retained the same enabled snapshot and matched the reference output.
- A completed synthetic PNG image request reported zero drafts, zero proposed
  tokens and zero reused tokens. Following text used two drafts with actual
  proposals, initially reused zero, then reused 4,096 on repetition. Both
  post-image outputs matched the fresh reference.

The reference and fixed workers used the same retained artifact folder. No
additional download or checkpoint conversion ran. Actual image cancellation or
device-fault injection was not part of this bounded run; existing shared
lifecycle tests cover invalidation, while the real transaction covered successful
image-to-text recovery. The separate retained mixed-4/8 artifact has no new
qualification here. Converted vision formats that omit MTP weights are outside
this loading-policy change.

## Paired short text observations

Three runs per arm, first excluded as warmup; same 54-token prompt, greedy
sampling, context 512, two MTP drafts, 64 requested outputs (31 sampled).
Two measured observations per arm. Model loading/hash validation excluded.
GPU work ran sequentially; background load/thermal state and peak RSS were not
screened. This is a short regression sanity check, not broad throughput proof.

| Arm | Warm median useful decode tok/s | Warm first token seconds |
| --- | --- | --- |
| Previous text-only worker | 104.7092 | 0.129836 |
| Fixed vision-loaded worker | 105.4884 | 0.131749 |

All short outputs were identical. The observed decode difference is under 1%.
The original MTP-off image-loaded observation around 84 tok/s is a separate
recipe; the paired MTP2 comparison above directly tests restored text behavior.

| Identity | SHA-256 |
| --- | --- |
| baseline worker | `b3a4f851515a44d42975e120599d084a647fd36a8aa7010b915b4829bf9bc2fa` |
| fixed worker | `297d2803fc752a6da59d3b837f640a355701cf43f7dc1350e9d224bca6e63699` |
| retained image manifest | `cd1e467d66062bdeeea841123ccdef48a84a3a60f468f3a88004e129c9768092` |
| private aggregate receipt | `b551bbb762c7df582693e0da055bbd7f8a15d33033ff6bf25e42b5df4f674557` |
| Cargo.lock | `85c73ea6bdb1ece383aca9502820095be4dba8818d6d7e348caca938507e18d6` |
| converter lock | `df7711726b1ed7e235ded21b9e6e1cbe130924be4d2c580d5d6bd9005248f8d5` |

Private receipt: `runtime/lily-vision-text-mtp-2026-11-08-v1.json`. No prompts,
responses, token vectors or probabilities are published. Added/qualified on the
branch for this artifact and these controls; not installed or released.

## Final PR review

Independent review found no blockers in capability exposure, zero-draft image
execution, reset/prefix recovery or memory accounting. Current worker binaries,
manifest and private receipt hashes match the recorded evidence. The release
worker unit suite passed 135 tests with two existing ignored tests; all 46
image-host/memory Python checks passed again. Repository policy, Rust formatting,
converter Ruff and patch whitespace checks passed. No inference code changed
after the recorded paired benchmark. The PR is stacked on #110.
