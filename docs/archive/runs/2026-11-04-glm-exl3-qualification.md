# GLM EXL3/FP8 artifact qualification

Date: 2026-11-04

Source parent: `d274f4fb56e2801545c981b7be266306bed2ef53` plus the v7b source
bundle below. The [initial implementation record](2026-11-03-glm-exl3-backend.md)
and its earlier stages remain preserved.

Status: the pinned pack is complete. V7b passed full-readout, complete-state and
ordinary-worker prefix-reuse checks. The screened serial benchmark measured
723.751 median prefill tok/s and 17.070 median decode tok/s on its 16K workload.
The historical upstream recipe remains an unmatched comparison.

## Source, pack and sealed executables

The native composition is mlx-serve
`dc92a3e88ad193a82c70f24ef96918b8dc1d4af6`, mlx-stream
`7d29eaa7217ea8d0e1b50000eb23ad37b06f5d80`, MLX
`64ea011cb65f14d9ce2737e60db9a4ae91ed7441`, MLX-C
`56b2d39fc831f2c0eb5bb94d82ef7191f7b31fa6` with its pinned gather-QMM patch,
and sushi `0bc61aa690cadfed63637997f86b5de641d6645c`. The
[bridge provenance](../../../vendor/glm-exl3-bridge/UPSTREAM.md) and lock retain
licenses, compiler/artifact pins and patch provenance.

The selected pack is
[`OpensourceWTF/GLM-5.3-Flash-streaming-repack-exl3-3.05bpw`](https://huggingface.co/OpensourceWTF/GLM-5.3-Flash-streaming-repack-exl3-3.05bpw/tree/50237d5c26ceb98bfd85f779990b0f5e6d7af144),
revision `50237d5c26ceb98bfd85f779990b0f5e6d7af144`. All 24 flat files,
130,767,909,118 bytes, were hash-verified. The completed private download receipt
SHA-256 is
`3f4f04462a6e155f4c1aa36238818777cc5e0914dab8acf69bb7b0a5c84ebdbe`.
The pack was left unchanged. Each run guards the verified files' identity,
size, modification time and inode before and after execution.

The precision policy retains EXL3 experts and FP8/BF16 residents, F32 KDA
checkpoint state, BF16 DSA open tails, and complete F32 host vocabulary
readouts. State and C timing probes use serial host greedy sampling with MTP
disabled. The worker matrix also covers stochastic sampling. Qualification
applies to this pack, source tuple and precision policy.

The Rust 1.98.1 release build receipt SHA-256 is
`ae043f0ccc97b3737ad10d01bb78f146f75b47b58184abbebb146152487d796a`,
binding source bundle
`7e0a79c87c42f16f8f5772e7d8fe5becbba7ee08c9d144eaea72626cd0d4643e`.
Both workers launch without `DYLD_LIBRARY_PATH` and use the sealed MLX stage.
The ordinary prefix worker was built without the diagnostic validation API.

| Executable | SHA-256 |
| --- | --- |
| serial worker | `72ab861ecdfb82ee220a848644683ebb7c6a6bcdde79922c77530a2d22f5f797` |
| ordinary prefix worker | `e1ddb4b05067967815914c2c9497a20452c6ffa29bdecebbb80817efe4ca9d71` |
| state-validation example | `83660f43d12c928335a8d3c6b0bb8dc78b5bf5989c8f5978cbd63ad4e02d8c7c` |

The serial and prefix stage receipt hashes are respectively
`2756c597bd440156c5539d6f322a654d51dd320d999f6a0ce27260def7427ed2`
and `5d0dbf265e3a7bb8691289a1a945a336d1cf453d717751f9c3214cd902f72392`.
Both bind native source
`fa5617ae3f8c5de87550d840603524fb9b992ca3de33e2df59b92c099a48ff71`.

## Admission and residency corrections

V7b applies the native host's admission census before constructing the phase
bills. It counts and evicts the pack's file cache, charges foreign file cache,
and retains the non-file memory floor. Expert-row counts come from the completed
admission plan. The ceiling remains 112 GiB with a 2 GiB margin and an additional
prefix snapshot/restore reserve where applicable.

After loading and draining, the bridge sets this process's MLX wired capacity
to the minimum of the admitted model's bill, its remaining budget, and Metal's
recommended working-set maximum. Teardown and load-error cleanup release model
buffers and the free-buffer cache before restoring the prior capacity.

A minimal locked MLX getter observes the previous capacity before setters can
mutate it. Its adapter returns a sanitized status. The getter patch SHA-256 is
`72563c4aa3d81125df33a6d9502b8fbce1c4100cc9ffcda6f3eec05f97395ecd`;
the complete approved MLX source diff hash is
`458411eb8ddd7a7a4ebaeeed6324ccffd5b3c68a86095074f49d58cf14305cfe`.
MLX-C, JACCL and the existing NAX metallib remain bound to their original
artifact hashes. These changes affect process-local allocator policy.

## Correctness and implementation checks

The v7b state check used a 16,384-token prefix, first and changed-suffix prompts
of 16,449 and 16,513 tokens, and four greedy continuation steps. Fresh and
restored execution matched all five full 154,880-value F32 readout rows and all
five complete logical KDA/DSA state hashes. Maximum absolute readout error was
zero. Stale snapshot, invalid-token and cancelled-callback rejection and
snapshot invalidation all passed.

| Aggregate | SHA-256 |
| --- | --- |
| private state receipt | `304a3f75d7492467ac98b4113afc8126763210ab776befc1cddcd393332310c5` |
| restored readout bundle | `5b11ef57be4467e1828501badf98723f49b945a42387fb3c25a18fa7442b5dc5` |
| restored state bundle | `761bb608d9ef4af672511ed4ea97d587888bb831fe08edbbd1d1b4aa9c75ac03` |

The two bundle hashes equal the preserved v5 results. The state run occurred on
a busy host and establishes restoration equality only; it carries no throughput
qualification.

Both native stages passed the C ABI smoke checks. Five Rust native
admission/ABI/selection tests, 112 worker tests (two ignored), and six
content-free process checks passed. Worker and state-example Clippy checks
passed with warnings denied. The focused Python checks passed 46 tests and
seven subtests.

Independent review found no actionable issues and verified all 757 sealed
source hashes, all 24 owned bridge files and both native stage receipts against
the current checkout and lock. Repository policy and formatting checks passed.

The ordinary-worker matrix passed all 32 cases. Every case matched both fresh
execution in the prefix-capable worker and the serial worker, and all 32 final
reuse counts matched the documented rule. Progress consistency also passed for
the 31 requests before restart; restart checks the final reuse count. Follow-ups reused
16,384 input tokens. The checks cover greedy and stochastic sampling, changed
prefixes, shrink/regrow, boundary edges, context/scope changes and restart.
Failure replies, unscoped requests, explicit reset, image operations and
cancellation cleared reuse; tokenization and decoding helpers preserved it.
The synthetic templated sessions contain 18,185, 18,808, 19,200 and 20,423 input
tokens within a 32,768-token context, with eight generated tokens per case and
MTP off. This exactness run establishes no worker timing claim.

The private worker receipt SHA-256 is
`873522a363b6a45afcce5e095f68c60f8e9515d7f0e8fb7d9765057e97a62952`.

## Screened serial throughput

The final repeat used the v7b serial stage, 16,384 raw-source input tokens,
128 greedy decode steps, MTP off, one excluded warmup and two measured reset
runs. The local arm64 host has 128 GiB of physical memory, 18 CPU cores and
macOS 26.7.1. The admitted expert rows were 142 prompt / 169 decode; installed
process-local wired capacity was 81,086,260,508 bytes, with a previous setting
of zero. Prefix state reuse was off for this uncached throughput measurement.

| Run | Prefill seconds | Prefill tok/s | Decode seconds | Decode tok/s |
| --- | --- | --- | --- | --- |
| 1 | 22.540766 | 726.861 | 7.520782 | 17.020 |
| 2 | 22.735315 | 720.641 | 7.476512 | 17.120 |
| Median | | 723.751 | | 17.070 |

The measured-window screen passed: zero pageouts and swapouts, zero external
model CPU, no compiler/download candidates, no truncated process audit, no
sampling gaps or failures. Maximum external CPU was 118.668 percent and maximum
thermal reading was 1, within the existing screen limits. Screening starts at
the validated warmup-complete marker and ends at the final measured-run marker,
including the between-run reset. Startup, warmup and teardown observations are
retained separately. Native phase timers are authoritative; marker receipt and
monitor samples have boundary overlap.

Both measured prefill readout hashes were `3379492f9916220b`; both final readout
hashes were `1038fa289a111105`. They match the preserved diagnostic runs. The
benchmark input U32LE SHA-256 is
`06da8fa52d1c92a827ae8022d756fbf26e3de95e706803769b7952db898b57c9`.

The private screened receipt SHA-256 is
`0f37986e8f163cd97e6aac0eaab270bc0c743c679d30587f9edb2346b8690a09`;
its harness SHA-256 is
`d25ba0bfaecdde648033896459383380d6bc8f6589aff38798eab52f35023a56`.
The audited detector conservatively classifies known compiler/download command
forms and retains only executable basename, PID, category and argument hash.
Seven pure checks and independent review covered its reported classification
and audit-merging gaps. This syntactic screen does not prove an absence of all
possible network activity. Prior harness versions and receipts are preserved.

## Preserved diagnostic timings

Both attempts below used 16,384 raw-source input tokens, one excluded warmup and
two measured reset runs. Native timers include the final full-readout copy;
serial decode includes host argmax and readout copies. Prompts and token vectors
were carried in memory through anonymous pipes. Private receipts retain source,
artifact, fixture and executable hashes.

| Attempt | Prefill tok/s, runs 1 / 2 | Decode tok/s, runs 1 / 2 | Admitted prompt / decode expert rows | Qualification |
| --- | --- | --- | --- | --- |
| v5, one decode step | 838.579 / 821.499 | 2.481 / 1.837 | 178 / 205 | Rejected: external CPU and pageouts; decode is a single-step probe |
| v7b, 128 decode steps | 822.955 / 800.673 | 17.533 / 17.213 | 142 / 170 | Rejected: download-activity detector |

The v5 receipt hash is
`32d167bbce3a6a80d533cf0c5288bc19b83c692559bad8ac43ea16b3f5158edb`.
The v7b receipt hash is
`dd61c67562d06e6d97a0ff4d36687bb08cae5b7030d510c196e73ef56cf79a28`.
V7b recorded 81,374,211,580 bytes of installed wired capacity, with a previous
setting of zero. Its measured window had no paging, swapouts, compiler activity,
sampling gaps or sampling failures. The old detector retained only a boolean;
the triggering process cannot be established from that record. The attempt
remains unqualified. An updated detector retains a bounded audit of executable
basename, PID, category and argument hash, with raw arguments held only in RAM.
The screened repeat above uses a new receipt and the audited detector; the
earlier attempt retains its original rejection.

The prefill full-readout FNV-1a64 hash was `3379492f9916220b` in all four runs;
the v7b 128-step final hash was `1038fa289a111105` in both runs. The diagnostic
rates do not isolate the effects of residency changes, host baseline or expert
rows.

## Upstream comparison limits

The [upstream benchmark record](https://github.com/davidtai/mlx-stream/blob/7d29eaa7217ea8d0e1b50000eb23ad37b06f5d80/docs/benchmarks/glm53-flash/README.md)
assigns 1,093.5 prefill tok/s and 33.15 decode tok/s to historical g25-dsa
plugin `88d2aad` and host `a8f6c6b8`. Its cell used 16,381 templated prompt
tokens, 1,024 generated tokens, temperature 1.0, top-p 0.95 and seed 1. The MTP
lane was enabled with depth 3 and gate 0.5; prompt/decode expert rows were
190/217. The historical executable and prompt fixture are unavailable publicly.

This record's C benchmark uses 16,384 raw-source tokens, 128 serial greedy
decode steps and MTP off, with its own admitted rows and current public source
tuple. A controlled MTP-on/off comparison at equal rows and workload is needed
to measure each contribution to the decode gap. The current results do not
establish equivalent upstream performance.

See the [backend reference](../../validation/GLM_EXL3_BACKEND.md) for the current
build and reuse contract. Model files, binaries, prompts, responses, token
vectors, probabilities and state payloads remain private and ignored.
