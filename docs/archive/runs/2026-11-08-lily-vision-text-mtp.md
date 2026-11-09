# 20261108-LVMTP-01: text MTP with retained vision loaded

Registered before implementation and model execution, 2026-11-08. Owner: Codex.
Kind: retained model capability, memory admission, lifecycle and text performance.
State: planned. Disposition: branch `codex/lily-vision-text-mtp`, based on
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

Pending. Exact artifact precision, build identity, memory accounting, tests and
measured medians/ranges will be appended after execution. The existing prefix
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
