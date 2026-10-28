# Qwen3.5 qualified conversion and serving replay

Fresh conversion, the ordinary catalog-gated Local API, and the terminal
`--image` path pass on the exact runtime in the
[execution qualification](2026-10-26-qwen35-image-execution.md). This follow-up
records activation and final regression checks without changing that evidence
document or its catalog hash.

## Identity

- Qualification evidence SHA-256:
  `933380b5f7fd2d3db3c2198a4f52a81dd4da08431f71dec0a25ee07ed4521f3b`.
- Generation worker SHA-256:
  `5fc92516c9f803df90316599a3930c1e2104cf320ae23361ff807e48bde08ec6`.
- BF16 weights SHA-256:
  `7c5960e7225b3b36844fa397a85c5e20d281d732e382475062f552c0737c79d4`.
- Configuration SHA-256:
  `3f0cecb7cb3633bc271eabf69b06ac9f74b377b9ec17b441f5ceac331a71f42a`.
- Metal, Apple M5 Max, Apple GPU family 1010, 40 cores, MXU enabled,
  macOS 26.7.1. All five artifact hashes match the execution record.

Qualification used native source manifest
`a96c7fa72958a68244bd517b3682ea286c713da2e8a443557b0a54f50bc96bee`.
Removing a trailing blank line from a Metal include produced final source
manifest `43e29407e9f64c27ac6373087b9c9a966ef0c66857835dec6558f3a27aa7c4ca`.
A Rust 1.98.1 release rebuild produced the byte-identical qualified worker.
Native arithmetic and the executable remain unchanged.

## Fresh conversion

The CLI converted the local, byte-verified official checkpoint offline using
`--artifact-mode vision-language --dtype bfloat16 --require-native-support`.
The source revision was pinned to
`2fc06364715b967f1860aea9cf38778875588b17`. It published a new private output
directory and source manifest atomically. The receipt reports
`native_loading: qualified_tuple` and the exact passed worker record. Every
artifact file matches the qualified identities. Conversion itself performs no
inference; the following checks load the newly published output.

## Ordinary API and CLI

The HTTP replay used the active qualification catalog with no test override.
The freshly converted model passed buffered and SSE requests with matching
content and usage, malformed-image recovery, a 4,197,244-byte body, client
disconnect, and a successful following request. Image-inclusive prompt counts
were 89 for the small fixture and 1,049 for the 1-megapixel fixture. No staged
files remained. Observed disconnect cleanup was 1.695 seconds; peak child RSS
was 2,076,000,256 bytes and API RSS was 129,318,912 bytes. These are local
observations, not hard memory ceilings or performance guarantees.

The real `scripts/stark chat --image` command loaded that fresh conversion,
exited successfully, and recognized a public synthetic circle. The temporary
image was mode 0600 inside a mode-0700 directory and was removed. Terminal
output stayed in memory; only predicates, counts, and hashes were recorded.

## Final lifecycle review and regressions

The final review added a stream fence for epoch, worker, and model identity.
The lifecycle lock orders callback delivery with Stop/replacement. A stale
callback triggers cancellation and terminal drain; cleanup of an old failed
request cannot close a replacement worker. Model-free tests exercise image
and text streams, Stop, worker replacement, model-ID changes, and valid retry.
The ordinary HTTP replay passed again after this change.

The connection-capacity test now reads the server's early 429 response before
uploading a body. Its previous client-side body write raced accept-time socket
rejection. Production connection limits and rejection behavior are unchanged.
The corrected wire check passed 20 consecutive runs. Tests that require an
empty image qualification catalog now construct that scenario explicitly.

Final results:

- Converter/Python: **4,338 passed, 88 skipped** with the active catalog.
- Engine CPU/Metal: **560 passed, 29 ignored**, plus seven grammar integration
  checks and three doctests. The source goldens cover convolution/history,
  RMS/SiLU boundaries, a 65-row chunk with nonzero state, and FP64-prefix edges.
- Actual-model guarded API cancellation/recovery test passed.
- Generation worker: **40 passed**; process checks: **six passed**.
- Packaging/launcher: **55 passed**. Required notice hashes and copy registration
  independently verified.
- Existing text regression: two generation sequences and four full-vocabulary
  scoring digests equal baseline `9eadf771`; cache hits remain
  `false, true, false, false`.
- Rust formatting, Python Ruff, diff checks, and repository policy pass.

The complete decoder logit gate and independent layer audit remain bit-exact
as recorded in the qualification. Swift attachment input and a signed packaged
installation remain outside this developer-path qualification. Raw fixtures,
responses, native arrays, models, and runtime reports remain ignored.
