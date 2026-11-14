# Model launch fixes: independent double-check — 2026-11-09

## Review and follow-up corrections

Three independent reviews checked PR #121 at `1a0517d6`: GLM/Qwen native
ownership and admission, Lily MTP/prefill lifecycle, and Python reasoning/API
behavior. No further concrete native execution defect was found. Two Python
control defects were reproduced and corrected in `f52ceced`:

- Image generation still used an absolute transaction timer after ordinary
  text switched to activity-based timeouts. Real-pipe progress and delta
  controls failed at the timeout despite continuous valid activity. Validated
  image generation activity now renews the shared inactivity timer. Image
  preparation, blocked writes, discard, cancellation cleanup and the absolute
  image-verification deadline retain their bounds.
- The reasoning helper always passed `enable_thinking`. Cataloged Granite3.3
  declares `reasoning_config.field_name = "thinking"`; its retained template
  therefore ignored the API toggle. The helper now honors the two recognized
  boolean field names, with the existing fallback and exact GLM effort mapping.
  Native-template controls check Granite, Qwen and GLM.

The final combined converter slice passed **696 tests with 1 skip** in 164.30
seconds. It covers chat, image transactions, reasoning, Desktop/API/image
integration, request validation and image verification. Ruff check/format,
repository policy and diff checks also passed. An earlier integrated slice
had 170 passes and two failures because the new in-memory cancellation test
fixture lacked a real pipe writer; the fixture was corrected without weakening
runtime I/O checks. No new native runtime source or shader changes were made.

The original PR head `1a0517d6` completed
[CI](https://github.com/jshahbazi/stark/actions/runs/42752529794),
[Security](https://github.com/jshahbazi/stark/actions/runs/42752529857) and
[CodeQL](https://github.com/jshahbazi/stark/actions/runs/42752529468) successfully.
The engine CI job compiled Metal tests and executed the CPU/grammar suite.
These CI results belong to that original head; subsequent commits require their
own CI run. The earlier incomplete local full-engine run remains incomplete.

## Native Lily review and image-resident controls

Attempt [20261109-LAUNCH-RECHECK-01](../../model-experiments/2026-11-09-model-launch-double-check.md)
checks the ordinary-decode MTP repair with the retained image tower resident.
The previous [launch checks](2026-11-09-model-launch-correctness.md) loaded
text-only workers. No native runtime source or worker binary changed during
this follow-up.

Independent source review found no additional concrete defect in the Lily
catch-up, controlled prefill or request reservation. The new catch-up views the
completed trunk hidden and actual input ID, keeps its remaining intermediates
in prefill scratch, and does not overwrite the target logits. Its dependency
barriers are full Metal memory barriers. Captured speculative verification
keeps its separate catch-up. Request reservation uses the same complete
input-plus-output span as admission, after restoring a retained prefix; image
prefill retains that reserved capacity. Cancellation is checked between
completed prefill passes and clears pending request state before reuse.

### Setup and source boundary

The installed ddalcu folder lacks `stark-image-artifact.json`. An initial
`--vision-staging-root` launch therefore failed before model load in 0.047
seconds. This was an invalid image setup and produced no inference result.
The existing text-only receipts were preserved.

For the corrected setup, an owner-private APFS clone retained the exact
installed payload bytes. The existing `prepare_image_artifact` routine fully
hashed all 102 payload files (107,303,406,192 bytes) and added schema-6 image
metadata. Config identity remained unchanged. The original installed folder
was not modified and is not qualified as an image artifact by this run.

| Identity | Value |
| --- | --- |
| Model source | `ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit` |
| Model revision | `788c07a8725642bccf13a0a3b8a21954ab84b4f2` |
| Native worker source | `1a0517d6`; unchanged release binary from the original fix checks |
| Worker SHA-256 | `388df7f9d09bf5bbd719cb5998f018bfb5df480f07d2b4a196e781cfcbf56d13` |
| Config SHA-256 | `84a69b2bcb20a57b033c3827bc98fa21a5db4d18b55e0640dd3687c99a1b765f` |
| Image manifest SHA-256 | `2ea0257f3369fbc38bf1646d2c9c64f5504a2eca774fec68257d0a784f68dd7d` |
| Verified payload aggregate SHA-256 | `3c9b158adf55514da375b0a88f382ba8f4d80284e99c0f5a25f810b749da41ee` |

Both candidate workers used owner-private staging directories, required image
input and MTP in readiness, and advertised enabled 4,096-token prefix reuse.
Only one worker ran at a time, and both were closed after the checks. Host:
Mac17,7, 128 GiB unified memory, macOS 26.7.2. Python used the locked converter
environment with Python 3.13.13. Weights, BF16 activation policy, F32 recurrence
and reductions, and Q8Affine64 K/V policy were unchanged.

### Correctness and lifecycle results

All text requests used temperature 0.7, top-k 20, top-p 0.95, seed 42 and three
effective MTP drafts. Short history requests had context 512, 17 input tokens
and an output budget of 16. Prefix requests had context 8,192, 4,113 input
tokens and the same output budget. The unrelated request had 65 input tokens.
Inputs remained private and matched the earlier controls.

| Control | Result |
| --- | --- |
| Fresh tower-resident short request | 16 output tokens, length stop; exact earlier text-only output hash |
| Short request after unrelated history | Same output count, finish reason and hash as fresh |
| Prefix prime / repeat / unrelated / prime | Reuse counts `0 / 4096 / 0 / 0`; all output hashes match the corresponding earlier text-only controls |
| Actual synthetic image request | One output token, length stop; zero effective drafts and zero reused rows |
| Text after image | Three effective drafts; zero reuse; exact pre-image output hash |
| Repeat after image | Three effective drafts; 4,096 reused rows; same exact output hash |

The short-request output-ID hash was
`3ee4b7f1d37bda57fec8f9683888dd25cab616951ac7a2d459eb4eec70e9a2b5`.
The main prefix requests, including both post-image controls, emitted 11 tokens,
stopped at EOS and had output-ID hash
`7a4a102067397b64c34a67fe26dff9ff630eb79d0b0a842d22230717f6879462`.
This is exact bounded lifecycle evidence, not model-quality or full-window
qualification. It covers this ddalcu artifact and the unchanged named worker;
other Lily-format artifacts retain their own qualification.

### Timing observations and limits

The native follow-up took 41.376 seconds including its two model startups.
The first short request took 0.290 seconds and its interleaved counterpart
0.230 seconds. Main cold prefix requests took 2.061–2.319 seconds; eligible
repeats took 0.235 and 0.259 seconds. The image transaction took 0.592 seconds.
These are individual observations with no controlled performance comparison.

Across this run, host VM counters increased by 596 pageouts, 536 swapouts,
4,801,884 compressions and 3,754,883 decompressions; swapins increased by zero.
Consequently this run establishes no screened speed, throughput or
image-loading performance parity. Peak worker memory was not sampled.

### Private evidence seals

| Receipt or script | SHA-256 |
| --- | --- |
| Initial invalid setup | `7df7a09891ad16dc1b75c45845425c10b355f09257f8722f897941d75019d1a4` |
| Completed native qualification | `6b9643c0a5fa4f40c906e68eb73274ff42de44244fa8b31ad66c3a79e546d072` |
| Private validation script | `82b0c00f9418fc441936736a0bf5826df2609995560f503c1b21ca0373838b30` |
| Original text-only history receipt | `13c2f73338480d80d20754e3b0df18e52057101fed00401f64c5cfe61b3416ad` |
| Original text-only prefix receipt | `9cc92d561189c0b9a1e91d950f91d8b05078913ed21e8af01bb0a8e77bc7f6c0` |

Receipts, script, staged model and inputs remain ignored under `runtime/`.
Only aggregate counts, timings and hashes are published here.
