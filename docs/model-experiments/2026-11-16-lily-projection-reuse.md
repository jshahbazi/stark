# 20261116-LILY-REUSE-01: adopt verifier projection reuse

Registered and executed 2026-11-16. Owner: Codex. State: completed.
Kind: kernel scheduling. Disposition: adopted on this branch for the admitted
group-64 formats; not installed or released.

## Hypothesis and fixed acceptance

Sharing packed weights across independent verifier rows reduces memory traffic
without changing the ordinary per-row affine expression or reduction order.
The user approved adoption after the
[whole-book comparison](../benchmarks/2026-11-15-lily-long-context-infinitebench.md)
observed a median paired 5.40% decode-rate improvement (8/8 pairs) and 2.39%
request-time reduction (7/8). This is an engineering decision based on that
modest benefit; it does not change the earlier short-input study's failed 10%
performance gate or create a new performance claim.

Port the sealed v2 prototype into normal group-64 dispatch for two through four
BF16 input rows: Q4/BF16 with even output width, and Q8/BF16 or Q8/F32. Remove
the experimental environment selector. Other group sizes, single rows,
odd-width Q4 and Q4/F32 keep the existing implementation. Two MTP drafts remain
the enabled recommendation; omitted MTP remains off.

Acceptance before adoption:

- Exact component bytes against ordinary per-row GEMV, with offset input,
  weights and output views and output canaries. Exercise all row counts,
  supported output policies, fallback layouts and group-size boundaries.
- Relevant physical Metal and worker tests plus a locked Rust 1.98.1 release
  generator build. No experimental attribution patch or projection-reuse
  selector in the final source.
- Final-build, image-resident model controls for matched seeded text/MTP work,
  exact prefix restoration, image invalidation and text recovery. Replay the
  benchmark-derived whole-book case to verify final dispatch preserves output
  and MTP work. A mismatch blocks adoption.
- Preserve all prior source patches, aggregate measurements and decisions.
  Push the implementation branch; no PR or installed-build change.

## Bound identities and scope

Base source: `60bfa00a`. Prototype patch SHA-256:
`9f406ddd07961d1d529622d13699cb119aacdbf9dbeba33fef0d6e4e5e4b0768`.
Its worker SHA-256:
`096cceb6b370fea71b49523c928c68c1ac172ca77b63a968acf7ceb0a26dbe2e`.
The [verifier study](../benchmarks/2026-11-15-lily-small-row-verifier.md)
binds its earlier component and long-answer evidence; those runs do not become
final-build results.

Exact model: `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl`, revision
`0f94842e3682b49acace4daeb4cf2e65a6f94f83`, retained schema-6 image-loaded
Lily backend. Existing payload manifest:
`97c11dd6f7ee5a69e23759b9774acb21a786c48552ca2ede9dac8b6be4b5bc98`.
Precision stays mixed Q4/Q8 weights, BF16 activations, F32 accumulation,
recurrent state and readouts, and Q8 KV. No conversion or weight changes.

The unchanged baseline worker SHA-256 is
`61b9c8cc724407573169eb2abad7f908e4943eead7ac3122eeec3ef841a48640`,
source `cc8e47a824f802e24d442183ebbdd29fd25ecaaa`. Final model checks revalidate
the existing 63-file size/mtime/inode receipt and nine ancillary file hashes;
they do not repeat the entire 106.9 GB payload rehash. The linked historical
reports bind the config, tokenizer, template and payload manifests.

Host: Apple M5 Max, 137,438,953,472 bytes RAM (128 GiB), macOS 26.7.2 /
`25G332`. Other work remains running. No quiet admission, cooling, retries for
timing, or host changes are used. Functional runs load workers sequentially,
with a 900-second per-request timeout and native resource admission intact.
Peak RSS, GPU clocks/power and exclusive execution are not measured here.

The implementation changes only stateless projection scheduling. Attention,
indexer, convolution, recurrent and MTP snapshots, restore fences, reuse keys
and invalidation rules remain unchanged. Existing exact prefix restoration is
retained and was rechecked with the new dispatch. Images keep MTP off and
invalidate text prefixes. The tower and supported text draft head stay loaded.

## Results and disposition

Component controls pass 240 cases / 10,436,400 compared bytes with no unequal
bytes. The existing decode-row and fused GDN projection/rollback controls also
pass. Core tests pass 97 (55 ignored). Worker tests pass 145 (five ignored).
The complete physical CPU/Metal engine suite passes 1,076 unit tests
(163 ignored), nine integration tests and three documentation tests.
All 56 final native controls pass. The
[source-bound run record](../archive/runs/2026-11-16-lily-projection-reuse.md)
binds the aggregate receipts, archived harness variants and complete code patch.

Final locked release generator SHA-256:
`3a6609d2ef297dd84309af56288f4a83cbcd107fbbe6e1c196c7d20aca6efb4d`.
Source patch SHA-256, relative to the base, for the two production kernel files:
`d6b69592956668b6630109e7b2a7b1acf807f4a2b8736cf6169d48fa428684d9`.
The shader bodies match the sealed prototype apart from their comment;
host changes remove the selector, narrow group size and document dispatch.
The root and vendor Cargo locks retain their prior hashes. Builds use Rust
1.98.1; the engine suite uses pinned nightly 2026-10-01. Local Xcode is
26.4.1 (`17E202`), Metal compiler `32023.883`.

An independent source review found no arithmetic, indexing or buffer blocker.
`Tensor` admits contiguous row-major storage and honors binding offsets.
Normal prefill/decode and scoring do not select `BatchMode::VerifyHost` and
therefore do not reach this changed route. No scoring replay is required for
this verifier-only change. A stale diagnostic label is updated from row-axis
to generic rows: its unfused reference can now share weights before the
separate rounding pass. Historical timing records retain their original labels.

The first formatter/build invocations could not find cargo/rustup in the
shell PATH and exited before compilation or model execution. Explicit tool
paths fixed setup; no package or dependency changes were made.

## Model validation and setup history

The first lifecycle wrapper stopped before starting a worker. An independent
read-only host-admission probe after that constructor failure reported
88,728,893,312 bytes needed versus 86,126,608,384 available. That probe is a
separate observation without an exact timestamp, not a reconstructed native
error. The wrapper run is resource-deferred; its original failed receipts are
preserved. No host memory guard was changed or disabled.

The native continuation uses the same direct worker protocol as the benchmark
studies, retaining native loader/request admission. Eighteen checks pass:
greedy MTP0/2 cold, 4,096-token reuse and changed-tail exactness; malformed-image
invalidation; actual synthetic image execution with zero MTP/reuse; image-to-text
exact recovery and reuse; staging cleanup; reset, restart and identity guards.
This qualifies the final native path, separately from host admission.

The first seeded-arm harness omitted `stream: true` from its cancellation
control. Native progress is stream-only, so its callback could never send a
cancellation. The phase stopped at the cancellation assertion after five
successful final-build draft-depth requests and exact seeded reuse. The
terminal reply category was not recorded; this is a harness setup failure,
not evidence of a native cancellation failure. Its source and receipt remain
unchanged. A separate continuation enables streaming and a bounded 1,024-token
budget; it observes actual `generation_cancelled`, zero-reuse cold recovery
and identical recovered output/work.

Cross-build comparisons pass all five choices (off, one, two, three and
automatic two drafts) on final/default, sealed experimental and baseline
workers, with matching output/text hashes and MTP counters. These use 4,104
input tokens, context 8,192, 64-token output budgets and the study's stochastic
temperature 1 / top-k 20 / top-p 0.95 / seed 42 policy. The final worker also
passes seeded 4,096-token prefix reuse and actual cancellation recovery.

The final default replays both original natural requests, with a 64-token
warmup followed by natural EOS: coding emits 1,109 tokens, planning 1,958.
The whole-book case re-fetches the pinned sources in memory, matches all
88,398 native input tokens, then matches its 64-token warmup and 1,339-token
complete summary. Input, output/text hashes and MTP work match the historical
records exactly. All replay requests use two drafts and the resident image
tower/head, matching the original contexts and budgets.

This is final-build trajectory and lifecycle evidence, not a new paired speed
comparison, full logit-distribution proof or quality evaluation. The performance
case for adoption remains the source-bound 5.40% long-context decode gain and
2.39% request-time benefit, alongside the smaller original-prompt gains. The
earlier 10% screening failure stays recorded. Other artifacts and hardware
retain their own qualification requirements.

All 232 pre-existing benchmark data and archived experiment files match the
base byte-for-byte. New private receipts stay under ignored
`runtime/lily-projection-adoption/`; public artifacts contain aggregates,
source code and hashes only. The experimental selector is absent from the
adopted implementation. Two drafts remain the enabled recommendation. No build
has been installed or released, and no PR has been opened.

## PR preparation follow-up

The later user-authorized
[independent review and clean PR preparation](../archive/runs/2026-11-16-lily-projection-pr-review.md)
uses a new branch from current main, excluding inherited unrelated changes.
It preserves this attempt and all original source/build-bound data. Its fresh
build/component controls are separate from these model-validation runs.
