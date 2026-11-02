# GLM-5.3 MTP Same-View FFN Reuse Diagnostic v6

Date: 2026-11-02. Status: **test-only diagnostic**. This run repeated the FFN computation immediately with the same selected view objects, compared all output F32 bits, and returned the first result. It measures a repeat-call path; it does not validate a cache, admission policy, or end-to-end speedup.

## Result

The guarded run recorded 23/24 continuation agreement and passed eight target-logit alignment rows, eight cache-alignment rows, and the unchanged owner-release and memory checks. Across 33 same-view pairs, all 135,168 compared F32 elements were bitwise equal; there were zero mismatches. The aggregate still records 59 prompt rows, three cases, 24 continuation comparisons, and zero target rollback calls. These are bounded teacher-forced measurements without speculative decoding.

The 24 measured warm heads report first-call FFN compute-and-wait at 17.967179 ms wall / 1.434076 ms GPU per head. The second call, with the same selected view objects, reports 1.668899 ms wall / 1.281580 ms GPU. Submit-to-return is 17.837319 / 1.520826 ms for first/repeat; wait-return is 17.833116 / 1.513644 ms; commit-call is 0.003285 / 0.006453 ms. The exact per-head values and call counts are in [the aggregate JSON](reuse-v6.json).

These phases measure two serial FFN computations and their fences. The full-head inclusive phase includes both computations and bitwise comparison: 22.389179 ms wall / 3.936774 ms GPU. The observed first/repeat timing difference does not isolate view-object reuse from warmed command, pipeline, scratch, or other reusable state. A runtime cache needs separate admission and performance measurements; subtracting these phases cannot establish its net gain.

The whole head's reported warm decode phase includes that diagnostic work and likewise cannot be treated as the normal optimized head. The measured prefill cache-only head is a separate path and is not a substitute for decode-head cost. No throughput, production wait-policy, production memory admission, or runtime-cache claim follows from this run.

## Memory and identity guards

The run recorded 33 selected mapping calls and release-owner checks, zero selected live bytes at return, 73,269,248 selected registered bytes at peak, and 609 static cache entries. Sampled Metal allocation peaked at 106,012,295,168 bytes against a 109,676,289,225-byte cap. The prospective production overlap projection retains a 371,913,609-byte margin under its stated assumptions; this is a projection, not production admission evidence. V6 memory gauges match v5.

The run's before/after source, metadata, weight identity, and binary guards passed. The corpus and target continuation hashes match v5. The four GGUF file identities were checked through the pinned weight identity receipt; raw prompts, responses, token IDs, vectors, logits, and process output remain private.

## Source and validation provenance

The source freeze is based on `379e5c79d1cfae249e947740732b783b1a6fec82`, SHA-256 `f66ba80e2610419091614b3afd4bf119f76ea58856f0690f1da35d8423652096`. The engine test binary is `caed04ebe1316a63bb545ad722621656bc1c50e32a0c1f74ab66512164dac970`; the driver is `4579ea4984b793b08866198eef8174ae18e14b946f7d532b0dee313d8716154d`. The probe result and guard receipt SHA-256 values are `bf866b6f3ed1dc2b77537fe736189780db3ee0da8fb085d1d38c8cc3576e8bb1` and `2bfaad89b17859a15e82fd439f32ce2ff2e1ead7b6f65b4ade0e8815be6497e3`.

The release no-run build receipt is `a1dca32599e52ed1a57d3638b19d9fb735741fe251b5db226fec7b99e33699b6`; the parser receipt is `3fc92e0586f54ab960c29d446549c2a0e106587633401cdb3230bf08de9669d3` (33 model-free cases passed); and the two model-free native unit receipts are `0522036ff3a0803cfbb5f0c52383e3181dda22a3de1cdd3913e083d15eaf6576` (2 passed). These are recorded prior validations, not checks rerun while writing this report.

The [incremental v6 source patch](reuse-v6.patch) has SHA-256 `427df440f927096fb429fd2eb65fa0fc6ee9696d9730d31a915a089934e53ceb`. It changes five source files. Its baseline is the exact nine-file v5 source snapshot identified by v5 freeze SHA-256 `8b17a190cc950278d62bc0a224132777652086d56fd95391e0f8e75600c876c9`; forward reconstruction reached all nine v6 freeze hashes, and reverse application restored all nine v5 hashes.

Patch application needs the zero-context flag. Applying the public [v5 patch](wait-v5.patch) from clean base `379e5c79d1cfae249e947740732b783b1a6fec82` with `git apply --unidiff-zero` reproduced all nine v5 freeze hashes; reverse application restored the clean base. The private v6 increment then applied normally to that frozen v5 tree and reproduced all nine v6 freeze hashes. The published incremental patch reverses to the v5 hashes and reapplies to the v6 hashes. The v5 patch bytes and hash are unchanged.

At this checkpoint, V7 is a source-only investigation of a fixed 192-expert prefix, represented by 576 per-plane views within the existing 8 GiB reserve. It has not been run; no speedup result is implied.
