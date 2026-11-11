# 20261108-VTP-01: text prefix reuse on vision artifacts

## Identity and acceptance

Registered on 2026-11-08 during implementation, before model-bound execution.
Owner: Codex. Kind: prefix reuse and lifecycle correctness. State: completed for the named correctness and timing workloads.
Disposition: adopted on `codex/vision-text-prefix-reuse` within the qualification scope below.
No installed-build or release claim is made.

The catalog's image-capable download previously selected a worker route that
always recreated decoder state, including for ordinary text. The hypothesis is
that text-only requests can use the admitted decoder's existing exact prefix
restoration without executing the vision tower or retaining image state.

Baseline: `540a21a9` plus catalog-only commits through `f77100f5` (no Rust or
converter changes). Candidate: the branch patch recorded with the results below.
No model weights, kernels, accumulation policy or sampling algorithm are changed.

Fixed correctness gate: successful cached greedy and seeded stochastic results
must exactly match the candidate with caching disabled and the baseline worker;
every advertised reused-token count must match the existing 1,024-token boundary
policy. Scope/context changes, reset, errors, cancellation, image operations and
restart must clear reuse. One retained state, publication after successful reply,
same loaded model identity and no persistence remain required.

## Planned artifact and state inventory

Primary artifact: existing official `Qwen/Qwen3.5-0.8B` schema-2 vision artifact,
revision `2fc06364715b967f1860aea9cf38778875588b17`, BF16 weights and activations,
F32 recurrent state, existing reference accumulation and KV policy unchanged.
Hashes, worker binaries, lockfiles, hardware and OS will be bound by the private
validator receipt. Other artifacts do not inherit its numerical qualification.

The decoder owns attention KV and positions, DeltaNet recurrence and convolution
history. Reuse delegates to its existing non-final prefill boundary snapshot and
restoration, with unchanged chunk grouping and sampler reseeding. Image-prefilled
states also carry image-position/multimodal markers and are excluded. Image
transactions continue to invalidate the worker entry before preparation or use.
The wrapper must enforce model identity, failure fencing and image-state rejection
without exposing a mutable underlying language model.

Gemma 3/4 and other admitted decoder profiles require a separate state-layout
assessment. MTP requires actual retained and qualified draft heads; the image
artifact does not gain them through prefix caching. Concrete dispositions will
be recorded below.

## Workload and limits

Use the model's own template with synthetic agent sessions. All prompts,
completions and token vectors remain in memory; write aggregates and hashes only.
Workers run sequentially, with a private per-worker image staging directory.
Use the validator's greedy and seeded exactness controls, chunk-edge and prefix
divergence cases, and a completed synthetic image transaction where supported.
Do not download another model or alter another task's processes.

Performance is not measured until a separate controlled timing run is recorded.
Functional equality and reused counts do not establish a throughput gain.
Stop a model run on failed startup, timeout, correctness mismatch or resource
rejection; record the failure before changing the recipe or implementation.

## Attempt history

| Trial | Observation | Validity and disposition |
| --- | --- | --- |
| Initial baseline build invocation | `cargo` was absent from the shell PATH; retry uses the installed absolute cargo path | Setup only; no inference ran |
| Initial engine check | Rewind error import used the wrong module path; corrected to the state module | Compile setup failure, before model execution |
| Source review | Capture position zero and rearming a larger boundary required explicit handling; restored-state proof now survives dropping the old snapshot | Fixed before model execution; numerical gate still required |
| First synthetic unit run | Hybrid recurrent fixture passed; two full-attention tests selected a mixed scalar/spatial RoPE fixture that deliberately disables multimodal admission | Invalid fixture for the intended spatial wrapper test; corrected to the admitted single-spatial-layer fixture, with inference source unchanged |
| VTP-01 official artifact exactness | 32/32 successful cases matched both fresh candidate and baseline; all documented reused counts and six invalidation/preservation checks passed. One follow-up correctly reused zero because the template changed its saved prefix | Overall strict reuse-coverage gate failed; numerical correctness passed within this workload. Receipt retained unchanged |

VTP-01 used seed 11, two tools, two instruction paragraphs, first-token target
2,300, follow-up sentences 4/4/4, context 8,192, eight output tokens and MTP zero.
Its actual turns were 2,111 / 2,228 / 2,334 / 2,409 tokens. The third follow-up's
common prefix was only 737 tokens, below the retained 2,048-token boundary.
The run completed in 185.8 seconds. Its valid miss does not justify relaxing
the existing reuse-coverage gate.

## Follow-up 20261108-VTP-02: stable-prefix workload

Registered before the second model run on 2026-11-08. Same artifact, baseline,
candidate inference source and fixed exactness/lifecycle criteria as VTP-01.
Only the synthetic workload changes: seed 11, four tools, two instruction
paragraphs, first-token target 1,800, follow-up sentences 4/4/4, context 8,192,
eight output tokens, MTP zero. Tokenizer-only preflight predicts turn lengths
1,616 / 1,718 / 1,817 / 1,914 and a shared 1,024-token boundary for all follow-ups.
This satisfies the existing coverage gate without changing its criteria.
The first receipt remains evidence of a valid cache miss. VTP-02 completed in
140.4 seconds: all 32 successful cases matched candidate fresh and baseline
outputs exactly; every reused count matched the boundary rule, all six
invalidation/preservation checks passed, and all three follow-ups reused 1,024
tokens. The overall exactness gate passed. This is numerical and lifecycle
evidence for the named artifact and workload, separate from performance.

## Results and promotion

Implementation and the named artifact correctness checks pass. A separate paired
timing run is recorded below. Peak RSS, quality comparisons, additional artifact
qualification and installation remain outside this result.

The [ledger](../../model-experiments/README.md) indexes this attempt.

## Follow-up 20261108-VTP-03: paired timing

Registered before timing execution. Same admitted artifact, inference source and
8,192-token context as VTP-02. Use the validator timing session (seed 12), four
tools, two instruction paragraphs, first-token target 1,800, follow-up sentences
4/4/4, eight output tokens and MTP absent. Run two paired blocks with two rounds
each, reversing baseline/candidate order in the second block. GPU work is
serialized; no owned compilation or unit tests overlap. Warm up each loaded
worker once before measuring; the first measured turn is cache-cold, not a
cold model load.

Fixed functional timing gate: identical outputs in every arm, zero baseline
reuse, zero candidate first-turn reuse and positive reuse on all follow-ups.
Report cache-cold cost and warm follow-up medians/ranges separately. Timing is a
bounded synthetic observation on this machine; no broad throughput or quality
claim is inferred. Peak RSS and thermal eligibility are not measured.

## Environment and payload accounting

Apple M5 Max, hardware model Mac17,7, 128 GiB unified RAM, macOS 26.7.2.
Release workers use Rust 1.98.1, default Metal features, locked dependencies and
separate per-checkout build targets. The native template helper is built from
the matching baseline source.

For this admitted artifact, 18 DeltaNet layers each snapshot
`6,144 × (4−1) × 4 + 16 × 128 × 128 × 4` bytes, totaling
20,201,472 bytes (19.265625 MiB). Existing active recurrence uses the same
payload separately. Replacement drops the old snapshot before allocating a new
one. Six full-attention layers retain KV without copying its payload.

At context 8,192, attention KV address capacity including 1,024 suffix rows is
`6 × 2 × (8,192+1,024) × 2 × 256 × 2 = 113,246,208` bytes (108 MiB).
On sparse Metal buffers, committed pages follow the highest prepared row and
device page rounding; rewinding does not unmap them. Before page rounding,
aggregate KV payload is `12,288 × highest_prepared_rows` bytes. These are
allocation payload/capacity calculations, excluding model/vision weights, scratch,
allocator overhead and process RSS. Peak memory was not measured.

## Bound source and verification

Inference source: `0000f093` on base `540a21a9`; `41190f9c` corrects only the
synthetic test fixture. Baseline: `f77100f5` in the separate catalog worktree,
whose difference from its main base contains no inference changes. Rust and
Python dependencies stayed locked.

| Identity | SHA-256 |
| --- | --- |
| baseline worker | `83b3430918e3ea03e5b01aeb9dee542b0a227ad4ddd5346ca90cb222b57045d7` |
| candidate worker | `a16af2a91d0000210d6fc3deefbe432a79c866282d780527efda98049c2821b6` |
| harness | `58ad8a0435183fc12d481afda3cdb4142cca529efd8a17cd54914ef166e7a82b` |
| model weights | `7c5960e7225b3b36844fa397a85c5e20d281d732e382475062f552c0737c79d4` |
| model config | `3f0cecb7cb3633bc271eabf69b06ac9f74b377b9ec17b441f5ceac331a71f42a` |
| tokenizer | `87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4` |
| chat_template.jinja | `273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80` |
| conversion.json | `e35a5934e467b00920b344c56a114af5af3fc7cfd67d4fed1596e4a30d8bf3ea` |
| preprocessor_config.json | `27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516` |
| Cargo.lock | `85c73ea6bdb1ece383aca9502820095be4dba8818d6d7e348caca938507e18d6` |
| converter lock | `df7711726b1ed7e235ded21b9e6e1cbe130924be4d2c580d5d6bd9005248f8d5` |
| native template helper | `be202a70a27eaf6b8157f031618287a427125a8d06be17ab33354e684661c86a` |
| private official-v1-exactness.json receipt | `e96bda134c74168733cd6c0537cccc5abae4d519192b68c7593a78164820a740` |
| private official-v2-exactness.json receipt | `cc4009ab5626db24260112dca43ad8993928e8f9b0c38ca75070d7fcc6a5be4e` |

Checks actually run:

- Stable release worker build and engine/worker check passed.
- Six new native numerical/guard tests passed: three CPU and three Metal,
  including a mixed DeltaNet/attention spatial-capable fixture.
- Default-feature worker unit suite: 134 passed, two pre-existing ignored tests.
  Model-free no-default-feature suite: 90 passed.
- Host startup, chat image transactions and validator tests: 136 passed, nine
  subtests. Targeted API/image lifecycle checks: 17 passed.
- Repository policy, Rust formatting, converter Ruff lint and formatting passed.
- Independent reviews found and resolved zero/unchanged/later capture boundary
  handling and memory-admission scope; final review found no actionable issue.

Untested artifacts do not inherit this numerical result. Gemma 3/4 have
overwriting sliding KV rings without payload snapshot/restore; the corresponding
text decoder paths also lack exact reuse. Their capability stays false. Image
conversion omits MTP predictor weights; raw BF16 text conversion also omits them,
whereas some quantized text routes retain them. Enabling vision MTP requires
separate artifact construction and numerical qualification. No image-state reuse
or general model/quality equivalence is claimed.

VTP-03 completed with the fixed functional timing gate passing. The
[paired timing record](../../benchmarks/2026-11-08-qwen35-vision-text-prefix.md)
contains the four-observation medians/ranges, worker and receipt hashes, and
host-screening limits. Implementation is adopted on the branch within this
artifact qualification scope; other artifacts remain unqualified by these runs.

## Follow-up 20261108-VTP-04: restore failure fencing

Registered during the final review before PR creation. Worker execution already
exits after a native restoration failure; direct public vision API callers also
need the loaded instance fenced after a restoration operation fails. Guard
rejections must remain recoverable. This follow-up separates these error
dispositions without changing the successful restore arithmetic or public method
signature. Preserve the prior worker, receipts and timing source identities.

Validation planned: focused error-disposition and CPU/Metal state tests, stable
release build and the existing 32-case official artifact exactness workload with
the VTP-02 recipe. Existing timings remain bound to their original executable.

VTP-04 completed on inference source `eb5ea9b9` after syncing main through
`43cd4323` (the main update only changed app UI). Seven focused tests passed:
three CPU numerical/guard tests, three Metal equivalents, and one CPU error-
disposition/sticky-fence policy test. Actual GPU device faults were not injected.
The stable engine/worker check and release build passed. The 32-case VTP-02
artifact workload was rerun against the final worker: every case was successful
and exactly matched both fresh candidate and baseline output, every reuse count
matched the rule, all six invalidation/preservation controls passed, and all
three follow-ups reused 1,024 tokens. Overall gate: passed.

Final worker SHA-256: `b3a4f851515a44d42975e120599d084a647fd36a8aa7010b915b4829bf9bc2fa`.
Private VTP-04 receipt SHA-256: `8d1bddb2d42e338b2d4d6bec461f7c00a5fd0e4131ee0cecae3c2f8e761562d3`.
The earlier qualified worker is preserved at ignored
`runtime/vision-text-prefix-reuse-2026-11-08/stark-generate-qualified-0000f093`.
Earlier VTP-03 timings retain their original executable identity and were not
rerun for this error-path-only follow-up.

## Retrospective 20261108-VTP-05: quick Lily text sanity check

Registered during final PR review after the user requested a quick existing-
script run. Existing Lily Q4-XL installed artifact; original worker and the
VTP-01 candidate were run sequentially with `benchmark_hybrid.py`, greedy
sampling, context 512, 64 requested output tokens and three runs per build.
Each completed at 31 sampled tokens; the first run was excluded as warmup.
Warm median useful decode rates were 83.6896 and 84.3240 tok/s, respectively;
first-token medians were 0.129586 and 0.129160 seconds. All output hashes match
across workers and repeats. This short unscreened check found no apparent
regression; it does not establish a statistically significant speedup or
qualify vision support on Lily. No fresh full-weight fingerprint was made.

| Private quick receipt | SHA-256 |
| --- | --- |
| lily-quick-text-2026-11-08-baseline.json | `6e17c3ff102ed1b7d87c70080c878f15dc15632cf835dc585d403830baa9e95a` |
| lily-quick-text-2026-11-08-candidate.json | `533cc6f9f6dab4bfd08d05ba8e49b77177a166cad2c5f6d4d218b45d90251562` |
