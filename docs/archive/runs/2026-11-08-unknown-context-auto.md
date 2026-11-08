# Automatic output for undeclared contexts

Date: 2026-11-08

## Source and scope

Baseline: `2acae95082369d1ec521d26b89b1c75ff1aab45b`.

Runtime-source manifest SHA-256: `5041d316d40847aa4e6a1bdeed943dbfa0a307e008e18b513c6cb0080d60e6ea` (850 files). The manifest sorts repository-relative files under `crates/`, `apps/menu-bar/Sources/` and `tools/converter/stark_convert/`, plus the root Cargo manifest/lock and converter project manifest/lock. Each entry is `path + NUL + binary SHA-256(file) + LF`; the aggregate is SHA-256 over those entries. Ignored models, binaries and this record are excluded.

This change replaces the manual-output requirement for text models without a declared context capacity. After tokenization, the host asks the native worker for a finite execution allowance. Model discovery continues to report an undeclared capacity.

The planner estimates loaded attention cache geometry and precision, full/ring/shared state, recurrent buffers, scratch allocator size classes, activation/readout workspaces and CPU token history. Dense MTP includes both target and predictor state, verification and handoff buffers, and overlapping prefix snapshots. It checks device buffer limits and the current attention kernels' 32-bit element-index range. Planning allocates no inference state and changes no inference precision or numerical operations.

Planning uses three quarters of a quarter of available RAM after a 256 MiB reserve. A fresh unknown-context allocation rechecks the quarter-RAM envelope after dropping a nonreusable retained state. Exact state reuse bypasses a duplicate allocation check. Existing context, model-instance, MTP and scope reuse keys remain in force. Stable admitted capacities can retain their prefix; a changed capacity cannot reuse a mismatched state.

## Native HTTP and CLI smoke

All existing local trained artifacts inspected for this run declared finite sequence limits. A new tiny synthetic ShortConv fixture was generated privately to exercise genuine undeclared capacity: one recurrent layer, four-dimensional embeddings/vocabulary, six tensors and 288 tensor payload bytes. It contains no external trained weights. Generated fixtures and helpers remain ignored under `models/` and `runtime/`.

On the physical Apple silicon Mac with Rust 1.98.1 and Native Metal:

- The worker reported `max_context_length=None` and advertised memory planning.
- The HTTP request supplied only messages, with no context or output settings. Model discovery reported `context_length: null` and `context_capacity_known: false`.
- The rendered input contained six tokens. The planner selected a finite execution allowance of 16,777,222 tokens with 16,777,216 available output slots, below the protocol ceiling.
- Both HTTP requests returned 200, terminated on EOS with zero completion tokens, and preserved the same worker object and PID.
- A separate direct native request confirmed EOS ID zero and `finish_reason: eos`.
- CLI chat without `--context-tokens` or `--max-new-tokens` exited successfully after one input and EOS. Its output was captured and discarded.

The large allowance was selected, not filled. These checks establish automatic admission, state creation, EOS handling and repeated-request recovery for this synthetic artifact. They do not qualify long-context model quality, memory consumption throughout a filled allowance, or the performance of a trained model.

Artifact SHA-256 values:

| Artifact | SHA-256 |
| --- | --- |
| Release generator | `4979e17cc6345c59a3c763ca03a2b431fc1efe6505697eeb43e1a42faa292835` |
| Synthetic model config | `ded4a9a5f5e353ebebc630b929b0fe38553f5db3e9afdb881b6f22fc9ee06283` |
| Synthetic safetensors | `f2b231154dd49edfcbcb90905a288008ba52cc748c0d8d8f9f31c61283f2c5a4` |
| Synthetic tokenizer | `47d98e224aed3adc2272e02d1575feb1502b44dc8a7ced5ed82e926db2eb3ade` |
| Private fixture generator | `ef2d215a10e3f1e63ea582ea1bf639671ce615e7171e4e285bf09ad2eec4bd1d` |

No prompt, completion or token vector was recorded.

## Regression checks

- Full converter suite: 6,483 passed, 90 skipped. Additional malformed-allowance and legacy-recovery cases passed in a subsequent focused run: 78 passed.
- CPU/grammar engine suite with the pinned nightly: 456 passed, 21 ignored; C++ string tests, profile tests and doctests also passed. Five focused planner tests cover geometry, precision, shared/ring buffers, allocation classes, dense MTP, device limits, index bounds and checked overflow.
- Generation worker: 123 passed, two ignored. The tests include bounded unknown-capacity planning, preservation/clearing of prefix state, invalid requests, no-fit recovery and the RAM-budget fraction.
- Shared resource-budget tests: five passed on macOS, including kernel-counter revision compatibility.
- Swift controller: 133 checks passed. Window suite: seven tests passed.
- Clippy with warnings denied, Rust/Python formatting, Python lint and repository policy checks passed.

The native smoke exposed a memory-probe compatibility defect: this macOS kernel returns a valid older statistics prefix, while the locked libc definition includes additional trailing counters. The probe now accepts the initialized prefix it actually reads, bounds the returned count by the destination size, and retains the zero-initialized tail. It no longer mistakes this supported revision for unavailable memory.

## Limits

Admission is an estimate and current headroom is not a memory reservation. Other applications can consume memory after planning. Existing byte budgets, cancellation, deadlines, sandbox execution controls and recoverable rejection remain active. Image requests still require their separately qualified finite-context path. This run does not qualify a real unknown-context trained artifact or an unknown-context Dense MTP artifact.
