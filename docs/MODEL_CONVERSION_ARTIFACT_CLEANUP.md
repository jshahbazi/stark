# Completed conversion model cleanup

On 2026-10-12, after milestone 5 was committed as `ceb6d37d`, the user directed removal of models after completed work to free local disk space. This supersedes continued weight retention for completed conversion packets. The [conversion plan](MODEL_CONVERSION_PLAN.md#model-artifact-cleanup) now requires cleanup after validation, benchmarks and review.

## Removed files and storage

An independently reviewed inventory and unlink-only script removed **389 generated model weight files** from completed milestone 1–5 conversion directories, plus **two selected LLaMBA source weight blobs and their four Hugging Face cache aliases**. All 395 selected paths were absent after cleanup. No recursive directory deletion was used.

| Observation | Bytes | GiB |
| --- | ---: | ---: |
| Removed logical file sizes, including hardlink aliases | 379,228,400,001 | 353.18 |
| Removed unique-inode file sizes | 364,731,775,233 | 339.68 |
| Available space before cleanup | 155,748,933,632 | 145.05 |
| Available space after cleanup | 520,491,778,048 | 484.75 |
| Observed available-space increase | 364,742,844,416 | 339.69 |

The available-space readings cover a shared APFS volume; other activity can affect their difference. One historical generated-output pair shared an inode and is counted once in the unique-inode total. Payloads were not rehashed for deletion; the original source-bound hashes and independent validation records remain unchanged.

The script checked every file's identity, size, timestamp, mode and initial link count before deletion, confined generated outputs to exact model weight filenames in completed milestone directories, and selected only the two known source blobs. Independent review and a fresh execution-time check found no open candidate files, no unselected aliases to either source blob, and no active serving model paths in the deletion set. Serving models, externally supplied source models, unrelated cached weights and the unresolved prototype benchmark/smoke roots were excluded.

## Evidence lifecycle

Small source manifests, configs, tokenizers, receipts, qualification proofs, complete benchmark observations, audit records, scripts and pinned revision/content hashes remain private and retained. Original validation and review records describe the artifacts available at that time. Their retention statements precede this later user-directed cleanup.

The four frozen milestone 5 candidate/reference weight files and its two downloaded input weights have now been deleted. Existing source manifests cannot replay missing files. A future check must reacquire the recorded immutable inputs, regenerate any required outputs and verify their recorded hashes. Cleanup does not grant new conversion, loading, execution or performance qualification.

The cleanup completed at **2026-10-12 14:47:01.750 UTC**. The converter source hash remains `0612178540e4c5e3af5aeef556225787ca512549a10971a0108141229a3a06f5`; its lock remains `5401164a38a3ffd8e6bf4f8252d6d583e798347e9ecf9931b725378e317526f2`. No production code, native ledger, original benchmark report or qualification proof changed.

| New private record | SHA-256 |
| --- | --- |
| Final cleanup inventory | `71d1167fe9c62f8be3aa08a5fdb17b9d7ae095e6102e598cf14d927f0bfba7b9` |
| One-time cleanup script | `c75c61b13a339a52a8dbe2d9d2c550d760908c25386ce651ec73ef3e0d72ea49` |
| Completed cleanup audit | `b485adef7493d966ec458f1ead71ddbab87ef239db1b43c586c9f98f344e990e` |
