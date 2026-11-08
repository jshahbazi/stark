# Flint prefix core qualification

Date: 2026-11-08. **Result: the bounded native replay passed.** It checks the
new complete-KV snapshot path on the pinned Flint INT4 artifact. The
[implementation record](2026-11-08-flint-prefix-reuse.md) remains unchanged as
historical evidence of the design and its then-pending validation.

This run contains 15 successful native generations, each limited to eight
output tokens, and one successful cancellation. It establishes the reported
prefix behavior and short-output agreement for this artifact and build. It
does not establish whole-model numerical equivalence, coding-agent quality,
filled-maximum-context execution or qualified throughput.

## Source, artifact and host

The implementation is commit `6cbf595227746a4bb1350835b9104fc21e03df61`.
The integration worker and template helper were built with Rust 1.98.1 release
at `aae3d17761552e360d8f1497ab853fc45047ba18`. Before the artifact run, the
integration passed the three Flint prefix CPU tests and 124 generator tests;
two generator tests were ignored. The CPU prefix run was made at `65b076ce`,
whose relevant Rust sources are byte-identical to the integration build.

Host: Apple M5 Max, 128 GiB unified memory, macOS 26.7.2 arm64. All native
workers were serialized under the shared hardware lock. The replay used the
locked converter Python environment. Root permitted light model-free checks
during the replay; timings were not screened for background host activity.
The worker PIDs were reaped with exit code zero, and the
shared lock was available after the run.

Artifact: `RecursiveSI/flint-int4` at
`3b93f9053b530488405ae9fca51d4900d6ce21f0`. Quantized weights, native matrix
execution and sampling were unchanged by the prefix implementation. The
attention KV and copied snapshot payloads remain F32.

```text
candidate worker   c5514016280ed85c69dac972373a688db85d10ab5810d134a5b3fe2ed8449415
fresh reference    a589fb72b94ffd8381642ab898f348d53bc3d78d5e59c36ddf294e023ebb2d79
template helper    fdbf8a879cd15f6bea2c84e03d6afb6f16e6b49c828fce8a656d3b403a8cc588
weights manifest   b1bd62dedb254b9f41dcedf422ccc2ac4a75a575ad8b131e44dbe928cf38128e
prepared config    06271655d67c1ce5cfc08ccc771faaddcc1a1c9719b3b2c499ddae1c1aa8cef0
tokenizer          a263cce0b7e2b03f7ee07ce809783ef4b47f160c3f268a4ea1524f638da45b68
chat template      7940f4ffa3275c36caa4a582f845aaad79da11018b3961e8fbcad6ae29733466
source receipt     74353a73c45b4816cc628e33fba346d0173962a2ba7886dc6da7c3856992e6cc
public validator   66c6370b4942434f4f55b7e9f66ec8d712082379f11078a5b4f264ef4c347fe7
private driver v2  b2c5788a24e1adec07943430afa4f12e4ae64da31d34df9758ec3090e4157309
aggregate report   446e25d0178115a08d4407cd576d455e9dd201f1864b86f5068f5639b7c52a43
```

The weights manifest hashes sorted shard names and each shard's SHA-256 using
the public validator's algorithm. It is distinct from the earlier complete
prepared-artifact fingerprint
`4c84481ae7657c35954de9d4e4e28ae0670febe162f83c30d550d3264b5e4764`.
The run sealed the actual model file identities before execution, hashed the
weights, and checked the file set/stat identities and sealed configuration,
tokenizer, worker and helper hashes again after execution. No changes were
observed.

## Request and comparison scope

Every generation requested the artifact's full configured 262,144-token
context. Context includes prompt plus generated tokens. Actual synthetic
histories contained 1,152 tokens and a growing 1,183-token follow-up, rendered
with the artifact's native template and one synthetic function tool. The
session used seed 113, one instruction paragraph and one-sentence follow-ups.
Both histories share the full 1,024-token retained prefix and cross the
513-row sliding-window eviction boundary.

Greedy requests used seed zero. The stochastic case used seed 29, temperature
0.7, top-k 20 and top-p 0.95. MTP was explicitly zero. Successful requests
ended at the eight-token output budget with `finish_reason=length`; those are
bounded next-token comparisons, not completed-answer quality checks.

The cached candidate was compared with the same candidate without prefix
reuse and the sealed pre-change reference worker without prefix reuse. For
each of the first greedy prompt, growing greedy prompt and seeded stochastic
prompt, generated-token, decoded-text and finish-reason hashes matched both
fresh references. The other repeat/recovery checks matched the corresponding
greedy outcome. No equality of error replies was counted as output agreement.

| Native check | Expected reused input | Observed |
| --- | ---: | ---: |
| First request | 0 | 0 |
| Repeated first request | 1,024 | 1,024 |
| Growing follow-up | 1,024 | 1,024 |
| Seeded stochastic follow-up | 1,024 | 1,024 |
| Different opaque host scope | 0 | 0 |
| Following request in that scope | 1,024 | 1,024 |
| First request after explicit reset | 0 | 0 |
| Cancellation | Prior progress reports 1,024 | `generation_cancelled` |
| Recovery after cancellation | 0 | 0 |
| Restarted worker, same host scope | 0 | 0 |

Reuse counts in progress frames agreed with terminal replies. Cache-disabled
workers omit the optional reuse count; the harness interprets that absence as
zero only when prefix reuse is disabled.

The direct worker commands kept the public validator's existing CLI shape,
allowing the old reference binary to run unchanged. These small requests do
not approach its default request-byte ceiling. This record verifies full
context admission and prefix state at the actual short histories; separate
native/API transport checks cover expanded byte budgets.

## Observed timings and memory

These are single, unscreened observations from the clean v2 pass. There was no
alternating repeated timing experiment, clock/thermal screening or isolated
snapshot-copy measurement. The native prefill stage includes the applicable
restore/capture work and remaining prompt execution.

| Request | Input tokens | Reused tokens | Native prefill seconds |
| --- | ---: | ---: | ---: |
| Cached candidate, first | 1,152 | 0 | 97.48 |
| Cached candidate, repeat | 1,152 | 1,024 | 13.95 |
| Cached candidate, growing | 1,183 | 1,024 | 17.45 |
| Cache-disabled candidate, first | 1,152 | 0 | 96.75 |
| Cache-disabled candidate, growing | 1,183 | 0 | 98.18 |
| Sealed fresh reference, first | 1,152 | 0 | 96.15 |
| Sealed fresh reference, growing | 1,183 | 0 | 98.74 |

At the observed 1,024-token capture boundary, the configured row geometry
implies a 132,263,936-byte snapshot payload, about 126.14 MiB. That excludes
deque/vector metadata, allocator overhead and matrix work. The
[implementation record](2026-11-08-flint-prefix-reuse.md#memory-accounting)
derives the separate maximum-context payload bound of about 42.10 GiB for
live state plus the old and pending snapshots. Neither number is measured
peak RSS.

## Harness correction and limits

The first private driver incorrectly required a reuse counter in progress
frames from cache-disabled workers. Its first fresh output matched the cached
cold output, but the optional-field check stopped the run. That attempt is a
harness failure and remains unscored. Its source and report are preserved;
the report SHA-256 is
`baa0b3adbeb9fc38ae6bd455f042b981456558ef1cd0759ebd2bd7fd665604bd`.

The corrected driver passed five offline optional-counter regressions and
reran the entire bounded core sequence into a new output directory. The clean
v2 result above does not merge checks from the first attempt.

Artifact-level shorter/diverging prompts, context changes, unscoped requests,
image-operation invalidation, idle expiry and failed output delivery were not
replayed in this core run. The shared worker and CPU state tests cover their
common contracts, but this record does not extend the artifact-run scope to
those cases. A filled 262,144-token history, its observed peak memory and
snapshot-copy latency remain unmeasured. Other Flint artifacts and precision
paths require their own qualification.

Prompts, responses, token vectors and snapshot payloads remained in memory or
private pipes. Persisted records contain aggregate counts, timings, fixed
statuses, source identities and hashes only.
