# Website product story — 2026-11-06

## Scope and acceptance

The website update is intended to improve the product story while retaining
Stark's recognizable large cat hero and existing visual identity. It should:

- Explain the local Mac app and its guided setup near the top of the page.
- Surface local decision scoring near the top as a smaller feature, with JevK5
  as a concrete model example.
- Use genuine screenshots of the Stark Mac app, with setup, model catalog and
  Decision mode represented by separate assets.
- Keep performance and model claims within the recorded evidence. Do not imply
  a hosted Jev service comparison or claim that Stark is equivalent to, or
  competes with, a Jev service.
- Preserve the existing cat illustration and overall visual identity.

The user's product hypothesis is that local decision models could compete with
hosted services such as Jev. Current evidence does not test that comparison.
The site can present local decision scoring as a capability and JevK5 as a
supported model, without turning the hypothesis into a performance or quality
claim.

## Screenshot provenance

Three fresh native-app screenshots were captured on 2026-11-06 using
`/Applications/Stark.app`. The app reported
`CFBundleShortVersionString=0.0.0` and build
`20261105204635627083`, a development build. The capture showed actual app
windows in their blank/setup states. No prompt, response, score, or decision
result was present. The screenshot files retain the captured JPEG bytes, with matching `.jpg` filenames.

| Asset | Captured view | Format and dimensions | SHA-256 |
| --- | --- | --- | --- |
| `website/public/assets/stark-guided-setup.jpg` | Beginner guided setup | JPEG, 1400 × 1370 | `04fad40fc07db6da9e774f5d198aefc75c1c650fe8b9c69a1bf6fc3320a8fba0` |
| `website/public/assets/stark-model-catalog.jpg` | Model catalog | JPEG, 1280 × 1322 | `b198c5c8c8dd63e4aabbce2de47fd9d39f45dbb940481d0f13da66e7652fb74c` |
| `website/public/assets/stark-decision.jpg` | Decision mode | JPEG, 1000 × 1362 | `c045b10f1323ec11ca2018d5061fb543844da5479c06bfec26f00698aa6435b4` |

The main Stark app UI was returned to its original Beginner no-model state
after capture. The screenshots were inspected in this worktree on the capture
date; file metadata, image format, dimensions and hashes were read from the
original assets.

## Decision capability and JevK5 evidence

The appropriate near-top claim is that Stark can score answer choices locally
with a decision model. JevK5 is a BF16 Qwen3.5-4B checkpoint supported for
candidate probability scoring and the app's Decision workflow. Suggested
compact copy:

> Score choices locally with JevK5.

Suggested smaller proof point:

> 243 recorded scoring cases matched frozen scorer references.

The parity record is bound to Stark source commit
[`97f43755dd833b9c33f35db19bf859057ee0830c`](https://github.com/jshahbazi/stark/tree/97f43755dd833b9c33f35db19bf859057ee0830c),
Rust 1.98.1 and the optimized build. On 243 JevK5 BF16 evaluation cases, the
native scoring worker matched frozen Uzu references for prompt tokenization,
raw logits, log probabilities, calibrated candidate probabilities and typed
decisions. Separate calibration-boundary and full-vocabulary digest cases
also matched. The record measured no timing. It establishes bounded parity
with that reference implementation, not decision accuracy, calibration
quality, broad task competence, or comparison with a hosted Jev service.

The older JevBench comparison is not a Stark-versus-Jev service test. Its
231 public cases compare historical Uzu scoring and llama.cpp on JevK5; both
had 202/231 gold argmax accuracy and 229/231 cross-engine argmax agreement,
with probability differences. Equal accuracy did not establish probability
or calibration parity. Do not reuse those figures as current Stark results.

The current source registers tool-call parsing for reviewed Qwen3, Qwen3.8
and JevK5 templates; that template-protocol support does not prove reliable
tool selection, argument quality or task completion. For JevK5, a separate
OpenWork record verified discovery and capability reporting only and explicitly
did not establish task competence. The public release identified by tag
`v0.16.5` should be distinguished from the inspected development app: do not
imply every current-source capability or screenshot corresponds to that
released build.

## Follow-up performance evidence

### Controlled synthetic comparison

The strongest direct cache comparison is the pinned Qwen3-4B run on an Apple
M5 Max with 40 GPU cores and 128 GiB memory, macOS 26.7.1, Rust 1.98.1,
locked dependencies and release worker builds. It used the
`mlx-community/Qwen3-4B-4bit` checkpoint at revision
`4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25`.

The deterministic synthetic sessions rendered roughly 18,200-token first
turns with 40 instruction paragraphs, 24 tool definitions and tool-call/result
history. Four alternating-order rounds compared the baseline worker against
the candidate with prefix reuse. Follow-up medians without/with reuse were
7.567/0.848 seconds, 7.838/0.593 seconds and 8.569/1.350 seconds for prompts
of 18,864, 19,292 and 20,543 tokens. Outputs matched in the recorded cases.
These figures are the worker's `prefill_seconds` first-token stage, not full
client response times. A cold first turn remained about 7.2 seconds. The run
was synthetic, its candidate was not installed or signed, and the results
apply to this model/build and workload.

Suggested concise website copy:

> In a controlled Qwen3-4B test on an M5 Max, prefix reuse cut follow-up
> prefill from 7.6–8.6 seconds to 0.6–1.4 seconds on synthetic 19K-token
> agent prompts.

If space permits, state that these are four-round synthetic worker timings;
first turns still process the full prompt and results vary with the model and
how much of the conversation prefix stays unchanged.

### Live OpenWork illustration

One separate user-driven OpenWork session used Qwen3.8-Flash-Next Lily q4-xl
on an M5 Max with a signed development build run from a staging folder. It
recorded 18 replies. The first reply had a 19,064-token prompt and 11.30-second
time to first text. Replies 2–8, with roughly 19,600–20,500 input tokens,
took 1.87–3.61 seconds to first text. Across all replies, 430,080 of 541,004
prompt tokens were reused (79%).

This is illustrative evidence, not a benchmark: it is one session/task, has
no same-session no-reuse control, and was not controlled for other GPU load.
First-text times include tool-call decoding when a response made a call. MTP
selection was not recorded, and the development build was staged rather than
installed over the signed app. If used, label it as one OpenWork session and
do not attribute its observed latency difference solely to prefix reuse.

## Source records

- [JevK5 scoring parity validation](../../validation/SCORING_PARITY.md).
- [Synthetic Qwen3-4B prefix reuse](2026-11-01-generation-prefix-reuse.md).
- [Live Qwen3.8 OpenWork session](2026-11-02-openwork-qwen38-live-prefix-reuse.md).
- [Tool template compatibility and scope](2026-11-01-tool-template-compatibility.md).
- [JevK5 historical scoring comparison and limits](../../benchmarks/INFERENCE_COMPARISON_EVIDENCE.md#33-jevk5-scoring-original-precision-mismatch-and-bf16-controls).
- [Current local API tool contract](../../LOCAL_API.md#function-tools) and
  [prefix-reuse contract](../../LOCAL_API.md#prefix-reuse).
- [Release `v0.16.5`](https://github.com/jshahbazi/stark/tree/v0.16.5).

## Verification

Implemented on `codex/site-product-story`, starting from
`e77f2f00bea7c7984aa35b7cca48f4460b79ac75`, in an isolated worktree.

- Static build passed with Node 24.15.0; JavaScript syntax and
  `git diff --check` passed.
- The locked repository policy check passed with Python 3.12.13 through `uv`.
- Browser review at 1280, 768 and 390 CSS pixels found no horizontal page
  overflow. The large cat hero, screenshot gallery, memory guide, compact
  decision feature and follow-up section were visually inspected.
- All three screenshot assets loaded with the recorded native dimensions.
  The source JPEGs were preserved without cropping or generated alterations.
- The rendered page had no duplicate IDs, broken local fragment links,
  missing ARIA targets or missing image alternative text.
- Model tabs worked by click and arrow key. The developer deep link selected
  its panel. Benchmark workload and relative-view controls updated the table;
  MLX-LM remained the 1.000× baseline. The follow-up methodology disclosure
  opened and exposed the pinned evidence link. No browser errors were observed.
- The existing cat illustration and four benchmark JSON assets were unchanged.
  This website update did not run new inference or establish new model quality
  or speed results.

Claim review corrected the initial same-build wording: the published timing
numbers compare distinct baseline and candidate development workers. It also
kept the fixed 16 GiB requirement specific to the two built-in Qwen3 starters;
dynamic recommendations calculate model-specific requirements and may admit
other models below that memory size. The page's Qwen3.8-27B availability was
updated after verifying release `v0.16.5` at
`3462a5098397cbb11c7e9f3a7c9c1cec23df214b`.
