# Decider HTTP latency diagnosis

Date: 2026-11-16. Attempt: `20261116-DFHTTP-01`; see the
[experiment ledger](../model-experiments/2026-11-16-decider-http-latency.md).

Stark's native Decision worker is now exposed through authenticated
`POST /v1/decisions` and the `/v1/systemone` alias. The measured latency is
dominated by the native full-sequence forward. The new transport adds roughly
13–17 ms beyond the worker round trip in these observations, including request
encoding. No native speedup or general performance qualification is claimed.

## Run identity and method

| Binding | Identity |
| --- | --- |
| Main profile source | `00bc52a401020ba822d78b062a9bc181e40c1f72`, clean at launch |
| Installed app source | `dac9676bdda71cb952b7a995e1a6af4a98021dbb`, clean at build |
| Checkpoint | `perplexity-ai/pplx-decider-flash-v1.3` at `d06fb3f2d97e0a52325a0d463c450cf0338f44fe` |
| Model configuration SHA-256 | `d859a249a3089252f164c755d2a584e41c1041ab91dc32751a742ddad29e8337` |
| Model weights SHA-256 | `c3ab2961a63c3342ff3390b41036c6029d62095414db4cffbd59a8944c754c97` |
| Tokenizer SHA-256 | `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523` |
| Template SHA-256 | `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715` |
| Main profile native worker SHA-256 | `a47ac69571360f153d05ffb534453b9dba715a3a046c3bf6efc1b1a0a19e40df` |
| Main profile template helper SHA-256 | `9b3b4569c1e439902adedeff2d513dc0df9a41f5c0a1f9043fafbd6d8dfb0d7e` |
| Installed signed native worker SHA-256 | `649a302713ea062de8065502b2237692c65d5fe3b4ea2abe2f5d451c941d6e9e` |
| Installed signed template helper SHA-256 | `8d94221b0c43e5732d98a1c2cd30d3652d8b352451c29cb72dd1e55f4d3dc6fa` |
| Installed app manifest SHA-256 | `dc8e217c04359f1daa07e8d7c819a10b34c46c8c7a89a8a7b2f21657f6ea88c3` |

Host: Apple M5 Max, 128 GiB physical RAM, macOS 26.7.2. The inference worker
was reused from the installed release, then copied and re-signed for the new
local app; no inference Rust/kernel changes were made. The template helper was
built separately with Rust 1.98.1, release and locked dependencies. The existing
BF16 weight/activation and F32 reduction/recurrent-state policies are unchanged.

The profiler uses a temporary private registry and an ephemeral loopback HTTP
listener with an in-memory random bearer key. It exercises the actual desktop
callback and worker. No installed credential is read. Request/response content,
tokens, logits and probabilities are not written to artifacts. Public results
below are aggregates; private receipts remain under ignored
`runtime/decider-http-profile-20261116/` in the implementation worktree.

The default image tower remains loaded for all text requests. Each request has
one question and performs a fresh full-sequence pass, without prefix/result
reuse or MTP. The synthetic state has 0, 64 or 256 unique record words; the
rendered input counts below include the template, question and option text.
Each length excludes one warmup and has three measured serial requests. Load
time, 1.293 s, is excluded. No compiler or test suite ran during the profile.

Before loading and before each sample, the profiler rejects competing model
processes and samples per-process accumulated Metal GPU time over one second.
Known desktop UI clients are allowed. This is a bounded host screen; it does
not prove complete GPU exclusivity or measure CPU contention, peak memory,
thermal throttling or paging. Those remain unmeasured.

## Text observations

| Expanded input tokens | HTTP median (range), s | Native round-trip median, s | Native forward median, s | Encoding median, ms | Native state creation median, ms |
| --- | --- | --- | --- | --- | --- |
| 80 | 1.106 (1.105–1.109) | 1.092 | 1.082 | 13.44 | 9.00 |
| 400 | 4.939 (4.861–4.990) | 4.925 | 4.845 | 12.96 | 9.45 |
| 1,360 | 19.299 (18.019–19.476) | 19.283 | 19.267 | 14.73 | 11.83 |

Phase medians are independent and need not sum. The native `prefill_seconds`
boundary includes host command encoding, the complete decoder forward, GPU
decision-head projection and command completion. `readout_seconds` measures
CPU conversion/checking of the resulting 255 logits; its observed medians were
2–4 microseconds. This does not isolate individual Metal kernels or establish
their relative contributions.

The native forward accounts for roughly 98–99.8% of HTTP wall time in these
median comparisons. The next optimization target is that native forward path.
State allocation and HTTP framing do not account for the observed seconds.
Every question still requires a separate pass over its complete state. Sending
four questions in one HTTP body retains four forward passes.

These synthetic requests do not reproduce the other application's workload.
Its historical 2.912 s median and 4.339 s p95 remain separate caller-side
observations. This profile establishes strong input-length dependence, not a
universal average or a comparison against another engine.

## Image and installed-package controls

The main profile's single generated PNG request returned HTTP 200, used 152
expanded positions and took 2.232 s. This is transport/processing evidence;
there was no new reference comparison or image-quality gate.

The signed development app was installed at `~/Applications/Stark.app` and
launched. Its listener was observed on port 8000. Its installed
`api_decisions.py`, `desktop.py`, `desktop_decider.py`, `local_api.py` and
`chat.py` hashes match the tested source. The bundle identifies itself as a
local development build, version 0.0.0; no release was published.

A separate isolated run used the installed bundled Python with `-I -B`, its
installed frontend, signed worker and template helper. It passed two text HTTP
calls (one warmup, one measured) and one PNG call. The measured text call took
1.107 s at 80 positions; the PNG call took 2.172 s at 152 positions. These
single observations are functional controls and do not extend the main timing
sample. Deep strict signature verification passed after this run, and the
reopened app's normal listener remained on port 8000. The normal listener was
not loaded with a model or sent inference traffic during this control.

| Private receipt | SHA-256 |
| --- | --- |
| `short.json` — earlier separate 80-token diagnostic | `4edb330160936b350097ec1d6c1ddfbf113b7a9c87a68b2afb34f82258b36e35` |
| `profile-corrected.json` — main three-length profile and PNG control | `852f17f4f3dfc8b4a6300d02164e21d15db185925bd69a1d525efc6e73580b08` |
| `installed.json` — initial package control, followed by bundle cache-write failure | `e3e5f91de2f97fc9148b010a3b2f54a888210030b0b1d48e82a2a89efd009518` |
| `installed-sealed.json` — restored package, `-I -B`, signature preserved | `11de33acc01f003058358038995897d18702b2d0bc62281ca55b4d5869892c6e` |

## Failures, checks and disposition

- The first broad controller test run lacked the native template helper on its
  PATH; import setup failed. The corrected controller run passed 368 tests with
  one skip. The complete converter suite then passed 7,909 tests with 95 skips.
  A later test-only synchronization improvement passed all 36 Decider tests,
  including cancellation both before and after native dispatch.
- Two profile attempts were stopped by the foreign-GPU screen without producing
  timing receipts. A longer CPU-only sample identified `loginwindow` UI work as
  an excluded client. The diagnostic UI allowlist was corrected and tested;
  arbitrary Python GPU activity remains rejected.
- An admitted profile failed without preserving its partial timing rows. The
  profiler's image control supplied a bare data URL where the endpoint requires
  a URL object. The exact failure HTTP status was not retained in that run.
  Safe HTTP error-code reporting and the image control were corrected. No
  measurements from the failed run are promoted.
- The initial app build pruned unused `libpython`, then signing required it.
  Incorporated the existing fix from original commit `a93bbdfc` as `dac9676b`.
  The signed build then passed. The signing suite passed 24 tests and the app
  builder suite passed 48 tests.
- The first installed control omitted `-B`; Python created bytecode caches in
  the sealed app and strict signature verification failed afterward. The exact
  staged signed bundle was reinstalled and the control repeated with `-I -B`.
  Final deep strict signature verification passed. No application source or
  model registry was patched in place.
- Profiler unit tests: five passed. Repository policy, Ruff, diff checks, the
  actual HTTP controls and final signed-bundle verification passed.

Disposition: HTTP transport adopted on the branch and installed locally.
Performance diagnosis is complete within this artifact and workload. Native
kernel optimization, joint-question execution, full 4,096-position timing,
other artifacts/quantizations, task quality and broad numerical requalification
remain outside this result. Existing numerical qualification keeps its original
source and artifact identities.
