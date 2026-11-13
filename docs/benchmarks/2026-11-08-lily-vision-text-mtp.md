# Lily text MTP with vision loaded: quick paired check

Attempt `20261108-LVMTP-01`, 2026-11-08. The
[source-bound record](../archive/runs/2026-11-08-lily-vision-text-mtp.md) binds the
artifact, executable hashes, memory recipe, lifecycle checks and source commits.
This is a short regression sanity check and does not supersede earlier long
Lily throughput studies.

Same retained Lily Q4-XL checkpoint and 55 shard inodes; old worker loaded
text-only, fixed worker loaded both the vision tower and declared MTP head.
Both used two drafts, greedy sampling, the same 54-token text prompt, context
512 and 64 requested output tokens. Each run sampled 31 tokens. Three runs per
arm, first excluded as warmup, two measured observations. GPU work was sequential
on the M5 Max with 128 GiB RAM. Loading/hash validation is excluded. No thermal,
background-load or paging screen and no peak RSS measurement was performed.

| Loading / worker | Warm median useful decode | Warm median first token |
| --- | --- | --- |
| Previous text-only, MTP2 | 104.71 tok/s | 129.84 ms |
| Fixed vision-loaded, text MTP2 | 105.49 tok/s | 131.75 ms |

The observed decode difference is under 1%. All short outputs matched, and every
measured request actually proposed MTP drafts. A separate seeded stochastic
output and the 4,104-token reference output also matched across loading modes.
These controls establish the named execution behavior, not general answer
quality or a statistically significant speedup.

On the fixed image-loaded worker, repeated text MTP reused a 4,096-token prefix;
switching MTP off/on missed, while repeated off/on modes reused their snapshots.
Changing nonzero draft depth preserved the enabled snapshot and matching output.
A real synthetic image transaction reported zero drafts, zero proposals and
zero reuse. Following text used MTP again, matched the fresh reference and
subsequently reused the prefix. Image-request MTP remains disabled.

Only this exact Lily artifact was run. Other retained sources need their own
qualification; converted vision artifacts that omit predictor weights require
separate implementation. The installed app and releases were not changed.

Private aggregate receipt:
`runtime/lily-vision-text-mtp-2026-11-08-v1.json`, SHA-256
`b551bbb762c7df582693e0da055bbd7f8a15d33033ff6bf25e42b5df4f674557`.
The [experiment ledger](../model-experiments/README.md) links this attempt.
