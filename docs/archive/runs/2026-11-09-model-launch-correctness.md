# Model launch correctness fixes — 2026-11-09

## Source and scope

Baseline: `1ac5b9dc`. Native, worker and Python fixes: `1a9c36b5` on
`codex/model-launch-correctness`. Commit `e8ea4288` separately repairs three
existing build/check issues: a stale memory-estimator test call, an unused
private attention wrapper and a divisibility lint. No model weights, precision
policies, shaders or sampling defaults were changed.
The menu bar wording is recorded separately in `cdaca736`.

The pasted review mixed correctness defects, intentional defaults and possible
performance experiments. This run fixes the confirmed defects and preserves
artifact-specific qualification boundaries. The remaining performance topics
are itemized in the [triage](2026-11-09-model-launch-performance-triage.md).

## Changes

- GLM lookahead does not encode an attention-cache expansion into a graph that
  an EOS/stop can discard. Successful draining preserves the completed prefill
  snapshot; failed completion still fences the model and discards it.
- Lily updates the MTP head for ordinary decode, including the held final
  prompt token on the official affine route. Speculative verification keeps
  its existing separate catch-up.
- Lily reserves the admitted input-plus-output span before prefill. Prompt
  and decode no longer independently double the same cache capacity.
- Lily cancellation is checked between existing prefill passes. Completed
  work is fenced before resetting a cancelled request; exact pass partitioning
  is preserved.
- GLM/Lily prefill emits count/time-only progress at safe control boundaries.
  Terminal chat consumes streaming frames internally. Worker/API inactivity
  timeouts renew on valid progress; a stalled worker still times out.
- Assistant history separates reasoning from visible content. Buffered and
  streaming API replies expose `reasoning_content` and `content` consistently,
  including split boundary tags, images and tool workflows. Native tool-output
  validation remains authoritative.
- API `thinking` controls boolean templates. The pinned GLM template supports
  Low/High/Max effort, so false maps to Low and true to Max, with an explicit
  `reasoning_effort` override. This does not claim that GLM can disable reasoning.
  Advanced Options labels the GLM toggle as maximum reasoning effort and
  explains the Low/Max mapping beside it.
- AppleDouble `._*` metadata companions are excluded from weight shards.
- F32 Qwen scalar attention buffers grow independently of KV caches. Normal
  split attention retains at most 256 scalar rows per head. Forced scalar and
  oversized-split fallback retain matching allocation admission.

## Validation

Host: Mac17,7, 128 GiB unified memory, macOS 26.7.2 (25G332). Worker/scorer builds
use Rust 1.98.1; engine tests use `nightly-2026-10-01`; Python checks use locked
uv environments with Python 3.13.13, and policy checks Python 3.12.13. GPU tests
ran serially. These are correctness checks, not screened performance runs.

| Check | Result |
| --- | --- |
| Final converter suite | 6,898 passed, 96 skipped |
| Generator release unit suite | 143 passed, 3 ignored |
| Native hybrid suite, CPU and Metal enabled | 240 passed, 60 ignored |
| CPU/Metal text-prefix regressions | 8 passed |
| Source-bound GLM/Qwen checks below | 7 passed |
| Installed mixed-affine seeded worker checks | History independence and 4,096-token MTP prefix reuse passed |
| Default and optimized scorer builds, Clippy and tests | Both passed; each ran 34 production tests and 35 probe tests |
| Generation worker process boundary | 6 passed |
| Formatting, Ruff, repository policy, diff checks | Passed |
| Menu bar window harness | 10 passed |

The full native engine suite was started and stopped during unrelated common
matmul CPU-reference comparisons. It did not complete and is not reported as
passing. The changed graph, scalar/split/fallback, prefix and model-state paths
were checked separately above. No Metal arithmetic source changed.

The initial integrated Python run had 6,876 passing tests, 96 skips and three
failures. Two old tests assumed every API delta was a string. One fake image
worker streamed text different from its terminal reply. Fixtures now follow
the real protocol; an explicit mismatch rejection test preserves the stronger
runtime check. An initial AppleDouble fixture lacked the required config file;
that setup was corrected before the passing native hybrid run.

### Source-bound native checks

GLM uses the private complete synthetic
`glm53-native-corrected-iq2xs-iq3xxs-pool4` fixture. Fresh-versus-restored full
readouts matched bit for bit under default settings and, separately, the
retained `with_glm53_optimized.sh` recipe. The new early-stop regression captures
at 1,024 tokens and drains at the 2,048-token attention growth boundary. It
checks snapshot survival, fresh/resumed equality and failure fencing. These
three checks qualify the mechanism on this fixture; they do not qualify the
108.7 GB production GLM artifact or promote the bounded experimental recipe.

Lily uses the installed ddalcu mixed-affine checkpoint at metadata/weight
revision `788c07a8725642bccf13a0a3b8a21954ab84b4f2`. Two model checks passed:
first-draft readout hashes are independent of an intervening request, and a
cancelled multi-pass prefill resets cleanly and replays identically.

A separate release-worker check ran a seeded stochastic request with three MTP
drafts in a fresh process and after an unrelated request. Both terminal replies
confirmed three effective drafts, emitted 16 output tokens, ended at the length
limit and had output-ID hash
`3ee4b7f1d37bda57fec8f9683888dd25cab616951ac7a2d459eb4eec70e9a2b5`.
Sampling was temperature 0.7, top-k 20, top-p 0.95 and seed 42, with context 512.
This is one history-independence case, not model-quality or broad equivalence
evidence. Weight payloads were not rehashed in this run.

A second worker check exercises MTP prefix reuse on that same artifact. With
three effective drafts, context 8,192 and the same stochastic sampling/seed,
the first request reused zero tokens, its repeat reused 4,096, and a repeat
after an unrelated request reused zero. All three main requests emitted 11
tokens and had output-ID hash
`7a4a102067397b64c34a67fe26dff9ff630eb79d0b0a842d22230717f6879462`.
This checks the MTP-enabled recurrent/head snapshot after the decode catch-up
change. It complements the separate draft-history and cancellation checks.

F32 Qwen uses the complete synthetic `qwen-fixture-f32-v1` fixture. Its context
is extended from 128 to 512 solely for a 257-prefix scalar/split transition
check. Every full readout matched the unchanged staged path within the
predeclared absolute tolerance `2e-5`. The same check passed separately with
forced scalar attention. The ordinary hybrid suite also checks independent
workspace growth and scalar row-stride parity. This is component evidence;
the PipeNetwork production checkpoint was not loaded. See
[20261109-QMEM-01](../../model-experiments/2026-11-09-qwen-attention-memory.md).

### Hashes

| Artifact | SHA-256 |
| --- | --- |
| Release generation worker | `388df7f9d09bf5bbd719cb5998f018bfb5df480f07d2b4a196e781cfcbf56d13` |
| Native engine test executable | `fe4ada582760304f894f807366c38cd02be0634a51c4b9069d2fa032c8cde6ca` |
| Installed mixed-affine config | `84a69b2bcb20a57b033c3827bc98fa21a5db4d18b55e0640dd3687c99a1b765f` |
| GLM fixture config | `0c3873ef79ae2f67a06b87b89cd1b9ad899c80078c56bc41da474ba6c9c01822` |
| GLM fixture GGUF | `cb4ab0a503c246cbea222d09c7589cbdb6194ece4850b506826d954ac63df39b` |
| GLM fixture input IDs | `b3b45e80e20031590dcbc70eb6e02838e0b182a370cd01df0f74859b2f9ab40e` |
| Qwen fixture config | `d165d68b570f33abefbaece4ac3d8aff82cfd71f2b3e64bc51181c3794424ba6` |
| Qwen fixture weights | `abf134190c66ee7f30533dd64422663aca2caa54e872ff49972777fe6383dcbd` |
| Qwen fixture input IDs | `bc65c88734aaee1d057174d72ff2c1fc9d7f975e776708889aa58271f3453821` |

Only aggregates and hashes are published. Private scripts, model payloads,
input/output IDs and runtime logs remain ignored. No installed app was replaced
and no PR was opened.
