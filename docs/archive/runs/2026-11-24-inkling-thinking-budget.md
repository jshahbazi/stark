# 20261124-INKBUDGET-01 · Inkling 2 Mini thinking budget

## Identity and state

| Field | Value |
| --- | --- |
| Attempt IDs and dates | `20261124-INKBUDGET-01` and `20261124-INKBUDGET-02`, registered and executed 2026-11-24 by Codex |
| Kind and state | Sampling, correctness and lifecycle. Both worker runs completed. |
| Evidence scope | Model-free framing, protocol and UI checks plus synthetic full-model text, tool and image requests on one installed checkpoint. |
| Promotion and disposition | Opt-in. Adopted on `codex/inkling-thinking-budget` in `0abb9cf20997`, defaults off. |

This opt-in sampling experiment addresses replies that use the entire output
allowance on reasoning. The
[Playground diagnostic](https://github.com/jshahbazi/stark/blob/9faa7e9b/docs/archive/runs/2026-11-24-inkling-playground-generation.md) confirmed
the empty-answer outcome at a hard total cap on the installed model. The
[inference.rs experiment](https://github.com/christopherthompson81/inference.rs/pull/1069)
supplied the closing-token bias idea. Stark counts across the whole reply to
address the reported [reopening behavior](https://github.com/ggml-org/llama.cpp/issues/32836).
See the [model experiment ledger](../../model-experiments/README.md).

## Hypothesis and fixed checks

- Use the installed 8-bit dense vision-language artifact with its published
  DSpark head. Verify the budget can recover a visible answer on the synthetic
  review whose unmodified 512-token request stops before answering.
- Compare identical budgeted requests with DSpark disabled and enabled, with
  both greedy and fixed-seed sampling. Check output and finish agreement,
  cumulative reasoning counts, and actual native drafting proposals.
- Exercise a retained prefill prefix against a fresh request, then an image
  request with the budget and drafting disabled, and a following text request
  with drafting and prefix restoration resumed.
- Check ordinary answer and tool framing, a simple correctness control, and
  unchanged output when the option is omitted. Keep raw token IDs, reasoning
  and answers in memory. Record aggregate results only.

No throughput or default-promotion gate is planned for this correctness probe.

## Artifact, source and environment

Both arms used the same image-loaded Inkling 2 Mini checkpoint on an Apple M5
Max with 128 GiB of RAM and macOS 26.7.2. The prepared source revision was
`6e7c8248f75265bdd4fbab5273304243b5d8c2cb`, with native vision profile
`inkling2-mini-native-v1`. The worker reported target SHA-256
`c9e14848ad017467dc0deb91c6aa67022e181dbcdfe02a1b04024434bf9acddd`.
The artifact retains published NVFP4/MXFP8 experts and the published DSpark
head (SHA-256 `7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2`).
Dense projections, embedding and head use Stark's group-64 affine 8-bit pack.
The patch changes token selection and leaves model computation and weights intact.

| Native worker | Source | Executable SHA-256 |
| --- | --- | --- |
| Baseline | Installed signed app build `20261124153109038856` | `ce5a983645f5d87ee1127b2c6ce4864919c37dc8895984e03dc4db76b94dadfa` |
| `20261124-INKBUDGET-01` | Development worktree on `a2af4698`, before the repeated-opener guard | `fbc8afa691001beddc74bd24fdcf6d4c839699bd0493dfabfb1406e3bb8754ac` |
| `20261124-INKBUDGET-02` | `0abb9cf20997`, including the repeated-opener guard | `692fcaadb1f45c67a68c689352054a5e86b685f492427b087560294a8803c6a4` |

Candidate workers were built with Rust 1.98.1, Cargo `--locked --release` and
`MACOSX_DEPLOYMENT_TARGET=26.4`. Cargo lock SHA-256 was
`07b6d42497825a651719685a28a107bfc0a32119b8e8b8adeab2d6c01a96695b`.
The Python probe used the locked converter environment, native JSONL worker,
and private temporary image staging. The existing app process was left alone.
Peak request memory and throughput were not measured.

## Workload and controls

- Fixed 8,192-token context. The code-review follow-up used 780 rendered input
  tokens, 512 output slots and a requested reasoning budget of 128. Matched
  arms used greedy with zero or three drafts, and temperature 0.7, top-p 1.0
  and seed 73 with zero or two drafts. Parity means matching generated IDs,
  decoded output and finish reason, compared in memory.
- The arithmetic, tool and image controls used explicit output limits of 128,
  256 and 128 and thinking budgets of 16, 32 and 24, respectively. The text
  prefix control used 3,316 input tokens, an output limit of 160 and a thinking
  budget of 32. It exercised a retained prefix created with the budget off.
- The final worker repeated the checks with the same settings and added an
  output limit of 1,024 on the same review, with a thinking budget of 128 and
  three drafts. Workers ran sequentially. DSpark proposal counts vary with
  the native scheduler. Equality was checked on the output rather than the
  scheduling counters.

## Attempt history

| Trial | Observation and disposition |
| --- | --- |
| Probe setup | Two harness launches stopped before model load because the standard environments lacked Pillow. Replaced the synthetic image generator with a Python standard-library PNG and reran. |
| `20261124-INKBUDGET-01` | The stock and omitted-option review hit 512 and produced no visible text. Both sampling modes passed DSpark parity and produced visible text with the budget. The cache, tool, arithmetic and image controls passed. Aggregate receipt SHA-256 `6c4b325db71c00c00e198644a580169d423c10294cb1d9c7719a5c4a8a7df92d`. |
| Repeated-opener review | A duplicate thinking opener inside a thinking block consumed the output allowance but bypassed the requested reasoning count. Added a counter and regression test before the final native rerun. |
| `20261124-INKBUDGET-02` | All fixed checks passed. The larger-output review finished at EOS. Aggregate receipt SHA-256 `5b0125f6710d15629c70e461a5ca2a795b9659d14b660eae88211bac04bbb94a`. |

## Correctness and lifecycle

The prefill cache stores only model state from the prompt. The budget and
reasoning counter start after restoration on each reply, so the setting does
not change the prefix key. Tests exercised ordinary closing, forced closing,
multiple thinking blocks, a duplicate opener, JSON with a quoted delimiter,
and the short-output reserve. The worker, converter and app controller tests
ran on the final code.

| Final worker case | Observed result |
| --- | --- |
| Stock and option omitted, 512 output slots | Both consumed 510 reasoning tokens and stopped at the output limit with no visible answer. Generated IDs, text and finish matched across builds. |
| Budget 128, greedy, 512 output slots | Identical IDs, text and finish with zero and three drafts. Both used 128 reasoning tokens and produced visible text before the output limit. DSpark proposed 456 tokens in the enabled arm. |
| Budget 128, seeded sampling, 512 output slots | Identical IDs, text and finish with zero and two drafts. Both used 128 reasoning tokens and produced visible text before the output limit. DSpark proposed 393 tokens in the enabled arm. |
| Budget 128, greedy with three drafts, 1,024 output slots | Produced visible text and reached EOS after 737 sampled tokens, including 128 reasoning tokens. |
| Synthetic arithmetic and tool controls, three drafts | Arithmetic was correct and reached EOS within 16 reasoning tokens. The tool call parsed with the expected name and argument, used 31 reasoning tokens and reached EOS. |
| Prefix and image lifecycle | Budgeted text restored 2,048 prompt tokens from a reply with the budget off and matched fresh text exactly. The image response was correct with 24 reasoning tokens and drafting disabled. The next text request drafted and reused zero prompt tokens, then the following text request resumed 2,048-token reuse. All three text outputs matched the fresh result. |

Rust worker tests passed 173 with five existing artifact-dependent checks
ignored, and the build without default features passed 104. Python desktop
and image transport tests passed 440 with one skip. Swift controller and
window checks passed 172 and 17 respectively.

## Evidence, decision and limits

Signed development app `20261124175121486473` was staged locally. The embedded
manifest SHA-256 is
`112ac43d9721636f51f470db9371e5d9b1463fc3ce53280237e8a8d57df77aa7`.
The staged signature passed deep strict verification. The bundled Python host
and signed native worker passed an isolated Playground command smoke with the
installed model and a temporary registry. The host reported the budget off
before configuration, then returned the same correct arithmetic answer in
both DSpark modes with the budget set to 16. Aggregate receipt SHA-256
`b73cb16ba59a3b58910b2cf2816d80780fa69d88ec7510cf3b9fb6074ae047ff`.

These controls assess the one 8-bit artifact and the listed sampling settings.
The budget bounds reasoning and preserves output slots. The model still
chooses its next message and when to stop. Broader task quality, exhaustive
sampling parity, live cancellation with the new option and performance were
outside this run.
