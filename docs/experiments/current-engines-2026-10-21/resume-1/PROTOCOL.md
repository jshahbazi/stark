# Current-engine comparison resume protocol

This is a new source-bound resume campaign for 2026-10-21. It preserves the original campaign gates and keeps the earlier interrupted attempt as a separate failed diagnostic record. No earlier timing cells may be pooled with this campaign.

## Frozen provenance

The Stark source is main commit `97f15f38201cc0b465a81c45e26a4b5b135d903c`. The corresponding verified GitHub Actions build is `39707877019`, artifact `12396251357`, archive SHA-256 `d17cb1b4ab77230abc344f1e4d415294147dc752e26bd16edc17505995a32559`. The downloaded Stark executable SHA-256 is `1cd922689dca98d41efc0727dab32fe472e6e7d199546130c1e555c8a151e517`. All bundled file hashes and the manifest hash matched the verification receipt. Local rebuilding is blocked by unavailable `rustls-webpki 0.103.16`; the verified exact-main bundle is therefore the source-bound executable for this resume.

The peer pins for this resume are MLX-LM commit `72be8c955d9a0d3099bd01865af288e0e947c3fa`, TensorFold 1.1.1 with drafting on and off, llama.cpp b11747, and vLLM Metal 0.32.0 with its locked dependencies. The vLLM lock SHA-256 is `7e2318038023d5df589e83fbe1a349b5d50c0e32a4c6774b3efb1dc9d1e00cc2`; the current tools lock SHA-256 is `f89cb16e0be95e81c690d5587a3d550b98c6440132e45ea0af1ed63560b79e1d`. These are explicit comparison pins, not latest-release claims.

Use the Qwen3.5-2B affine W4/G64 source revision and unchanged Stark repack from the original protocol. Borrowed model files remain read-only. New Python environments and caches are task-owned under the resume runtime.

## Admission and timing gates

1. Complete setup and verify every retained executable, model, archive, and dependency-lock hash before admission. Run the unchanged stock driver's `--probe` once over all six configurations. Preserve every admission failure; repair only setup or invocation paths in a separately recorded attempt.
2. Use the unchanged `benchmark_clean_qwen.py` for acquisition: four independent loads per configuration, three measured repeats of each of six performance fixtures, one measured repeat of each of four sanity fixtures, and one full warmup of all ten fixtures per load. Use one loaded task model and one in-flight request, context 8192, greedy decoding and seed zero where available.
3. Retain the driver's rotate/reverse ordering, realized order, and per-block ranges. Do not infer universal rankings or confidence intervals from this bounded design.
4. Measure actual output tokens divided by complete client request seconds, including prefill, generation, transport/readout, and cleanup. Loading, warmup, and quiet admission are excluded. Fixed work requires explicit output limits and limit termination; sanity results remain outside speed aggregates.
5. Apply the original quiet-window guards unchanged: ten newly observed contiguous quiet seconds before loading and after full warmup; 120-second admission timeout; per-process and total external CPU at most 80% and 150%; known model CPU at most 10%; device GPU at most 5%; nominal thermal state. During measurement reject external process CPU above 100%, total above 200%, known model CPU above 10%, non-nominal thermal state, monitor failures or gaps above two seconds, and pageout/swapout increases. Preserve diagnostics and do not stop unrelated processes or disable security services.
6. Complete setup, verification, and investigations before acquisition. Keep the idle-sleep assertion and wall-clock checks. Device GPU and basename CPU attribution do not prove exclusive hardware isolation.

## Acceptance and reporting

The campaign qualifies for a fresh cross-engine table only when every planned configuration and all four loads complete with fixed work, correct input counts, passing host screens, verified provenance, successful cleanup, `timing_accepted: true`, zero sleep events, and less than two seconds of wall/monotonic discrepancy. Preserve failed and excluded blocks. Do not replace loads, pool favorable cells, or relax guards. At most one subsequent complete campaign may be started if this campaign is invalidated by interference.

If the campaign is not accepted, retain block-level values as diagnostics and publish no replacement all-engine ranking or speedup. Report within-load medians, geometric means across accepted load medians, per-load values, rates and ranges, output hashes, fixed-work checks, and sanity outcomes. Do not persist prompts, responses, token vectors, logits, probabilities, credentials, or exception dumps; publish scalar/hash aggregates and path-redacted provenance only.

This protocol was frozen before fresh admission and timing, following the user's explicit request to resume the full run.
