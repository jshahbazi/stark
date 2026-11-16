# Lily verifier projection reuse: final native validation

Date: 2026-11-16. Attempt: `20261116-LILY-REUSE-01`.
Base: `60bfa00a8f1759b7af73e224ff8786181a18d371`.

The [attempt record](../../model-experiments/2026-11-16-lily-projection-reuse.md)
defines the hypothesis, fixed correctness gates, artifact and adoption decision.
The implementation makes packed-weight reuse the normal short-verifier path
for group-64 Q4/BF16 with even output width, and Q8/BF16 or Q8/F32. It retains
independent accumulation and ordinary reduction order. Two drafts remain the
enabled recommendation. Other formats retain their previous implementation.

## Identities and evidence

- Final generator SHA-256:
  `3a6609d2ef297dd84309af56288f4a83cbcd107fbbe6e1c196c7d20aca6efb4d`.
- Complete four-file code/test patch SHA-256:
  `d9b37b4e33b19692061e567a568e58c669f4de2c70a72dac0fded4072282d5a8`.
  [Source archive](../experiments/2026-11-16-lily-projection-adoption/README.md)
  includes that patch and byte-identical as-run harness variants.
- [Public aggregate evidence](data/2026-11-16-lily-projection-reuse.json), SHA-256
  `f264cc0e9bc850b5801690294e19ecf39fbbe20cba658df7a124b1e40026ee6a`.
  It binds five archived sources, nine private receipts, source/lock/build
  hashes, safe per-request outcomes and the prior-artifact preservation check.
- Model: `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl`, revision
  `0f94842e3682b49acace4daeb4cf2e65a6f94f83`. Existing payload manifest
  `97c11dd6f7ee5a69e23759b9774acb21a786c48552ca2ede9dac8b6be4b5bc98`;
  unchanged mixed precision and image/draft-head loading.
- Apple M5 Max, 128 GiB, macOS 26.7.2 / `25G332`; Xcode 26.4.1,
  Metal compiler `32023.883`. Workers use Rust 1.98.1 with locked dependencies.

## Local checks

| Check | Result |
| --- | --- |
| Default projection dispatch versus ordinary per-row GEMV | 240 cases, 10,436,400 bytes, zero unequal; offset views, canaries, zero/subnormal/cancellation patterns, actual projection dimensions, all row counts and fallback groups/layouts |
| Existing decode-row exactness | Pass |
| Existing fused GDN projection and every rollback prefix | Pass |
| Lily core suite | 97 passed, 55 ignored |
| Generation worker suite | 145 passed, five ignored |
| Physical CPU/Metal engine suite | 1,076 passed, 163 ignored; nine integration and three documentation tests passed |
| Locked release generator build | Pass |
| Repository policy, changed Rust formatting, whitespace | Pass |
| Independent source review | No arithmetic, indexing, buffer or state-restoration blocker |

The source archive reproduces the relevant commands. The complete engine suite
used `nightly-2026-10-01` as required by its custom test framework. Diagnostic
timing tests and model-dependent ignored tests were not broadly enabled.
Scoring does not reach this verifier dispatch; a scoring model replay is N/A.

## Final native controls

| Phase / private receipt | Checks | Bounded scope and result |
| --- | ---: | --- |
| `native-lifecycle-v2.json` | 18 | Greedy MTP0/2 cold/reuse/changed-tail equality, malformed/real image invalidation, zero image MTP/reuse, text recovery, staging cleanup, reset/restart and identity guards |
| `arms-v3.json` | 25 | Final/default, sealed prototype and baseline match output/text hashes and MTP work at off/one/two/three/automatic-two depths; seeded prefix reuse and actual streamed cancellation with zero-reuse exact recovery |
| `book-v3.json` | 5 | Native input exactly 88,398 tokens; fixed 64-token warmup and complete 1,339-token EOS response match prior output/text hashes and MTP work |
| `prompts-v3.json` | 8 | Original coding/planning inputs match; 64-token warmups and full 1,109/1,958-token EOS responses match prior output/text hashes and MTP work |

The seeded cohort uses 4,104 input tokens, context 8,192, 64 output tokens,
seed 42, temperature 1, top-k 20 and top-p 0.95. Its cancellation control uses
streaming and a 1,024-token ceiling. Original natural prompts retain context
8,192 and output ceilings 2,048/4,096. The book retains context 98,304 and
ceiling 4,096. Those replays use two drafts and prefix caching off; the explicit
cache controls use a single in-memory predecessor and verify reported reuse.
Every model worker has the image tower and supported draft head resident.

## Preserved setup failures and limits

The original host-wrapper lifecycle attempt started zero native workers. A
subsequent independent host-admission probe observed 88,728,893,312 bytes
required versus 86,126,608,384 available. Its failed receipts and separate
probe are preserved, without inventing a native failure or exact probe time.
Native continuation keeps loader/request guards; host admission is separately
resource-deferred. No host guard or other app/worker was changed.

The first seeded-arm harness omitted streaming, so no progress callback could
send its cancellation. It stopped after the final-build depth/reuse checks;
the terminal category was not captured. The corrected continuation is separate
and observes one progress event followed by actual cancellation and recovery.
The attempt record also retains the initial shell PATH setup failure.

These are finite functional and trajectory controls, not a new controlled
speed comparison, universal numerical proof or full image/answer quality
qualification. Historical paired performance remains **+5.40% decode and
2.39% less request time** for the whole-book case; earlier short-input gains
and the failed 10% gate remain unchanged. Incidental times in the functional
receipts must not be treated as a new A/B result. Peak memory and exclusive
GPU use were not measured. All 232 prior data/archive artifacts are unchanged.

Disposition: adopted on the implementation branch. No PR, merge, release,
installation or installed-app claim.
