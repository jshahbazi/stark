# Native chat-template rendering

Date: 2026-11-03. Apple M5 Max, 128 GiB RAM, macOS 26.7.1 (25G241), arm64.
This is the current component latency study. It supersedes the Python process
timings for the current implementation; the [initial Python isolation baseline](CHAT_TEMPLATE_ISOLATION_2026_11_03.md)
remains unchanged as source-bound historical evidence.

## Change and measured result

A fresh Rust `stark-template` process, using locked MiniJinja 2.24.0, now compiles
and renders each template. The host no longer starts a Python interpreter for
template evaluation and has no Python rendering fallback. The fresh process,
private JSON pipes, cancellation/reaping and bounded resource policy remain.
The Python application host and converter continue to run normally.

The complete final campaign contains 20 measurements per method in each case.
Every measured render matched the previous isolated Python evaluator byte for
byte, including both decision-boundary variants.

| Component | Python median (ms) | Rust median (ms) | Median time reduction |
| --- | --- | --- | --- |
| Short: 2 messages / 128 content bytes | 47.780 | 8.964 | 81.2% |
| History: 32 messages / 65,536 content bytes | 59.107 | 11.530 | 80.5% |
| Two sequential decision-boundary renders | 99.558 | 18.248 | 81.7% |
| Template admission / compilation check | 49.500 | 10.793 | 78.2% |

The [machine-readable aggregates](NATIVE_CHAT_TEMPLATE_2026_11_03.json) retain
sample counts, minimum/maximum timings and exact implementation/input hashes.
These timings cover host serialization, fresh child startup, compilation,
rendering, private-pipe transport and cleanup. Decision rendering remains two
sequential operations; no parallel evaluator or long-lived rendering worker was
introduced. Native prefill, token generation and scoring are outside these
measurements.

## Source, build and method

The Python baseline is commit `50fc76ce4f259b46f275be02388fcfc469db73ca`, whose
runtime SHA-256 is
`1ca2a313bfd7b48059aeab52d5f3077053cb7ea663b15db91b1e48813f6d9353`.
It runs in its own locked uv environment with CPython 3.13.13, Jinja2 3.1.6 and
psutil 7.2.2. Its original `-I -B -m` launch is preserved. Baseline dependencies
and project modules were compiled to bytecode before warmup so import caching
matches an installed distribution; `-B` permits reading those caches.

The native release uses Rust 1.98.1, locked dependencies, the repository release
profile and `MACOSX_DEPLOYMENT_TARGET=26.4`. Its SHA-256 is:
`eda0036eeb558500e3fe0c72c542f681b2ebd1732a140631dd3e790335cccdb9`.

The native source, Cargo lock, host wrapper and measurement driver are sealed
by the SHA-256 manifest in the aggregate JSON. The candidate builds on the same
baseline commit; no inference kernel, model, quantization or precision path
changed.

The public [Qwen3-4B template fixture](../../tools/converter/tests/fixtures/qwen3_4b_chat_template.jinja)
and its [provenance](../../tools/converter/tests/fixtures/qwen3_4b_chat_template.provenance.md)
are shared by both methods. Its SHA-256 is
`87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5`.
Short/history inputs alternate user and assistant roles, with respectively
64 or 2,048 ASCII characters per message, no tools, thinking disabled and an
assistant generation prompt. Decision input uses the existing application
system instruction, 64 synthetic text characters, a synthetic question and
two synthetic answer labels; both prompt-boundary variants are rendered.

Each case performs three warmup pairs, then ten complete ABBA blocks
(Python, Rust, Rust, Python), timing complete calls with `time.perf_counter()`.
All measured samples are included. Construction is timed separately from render
calls. Only aggregate timings, counts and hashes are retained; no request or
rendered content is printed or saved.

Earlier implementation trials and an initial run with uncompiled baseline
dependencies were exploratory. The table uses the final helper and the complete
campaign after matching baseline bytecode-cache preparation. The machine was
not reserved, and power, background activity and thermal state were not
controlled. These are local component measurements; no general latency floor,
end-to-end first-token improvement or human-perception claim follows.

## Security and compatibility

Every operation still has a fresh process and bounded input/output, CPU and
wall deadlines, VM fuel, nesting limits and sampled RSS monitoring. The native
allocator bounds cumulative Rust allocation requests across input parsing,
compilation and all intermediate values. This accounting differs from the
previous Python expansion estimate; see [current limits](../GENERATION.md#chat-template-evaluation).

MiniJinja supports a qualified subset of the previous template language.
Unsupported method variants, non-ASCII classification methods and out-of-range
JSON integers fail closed. Tuple-valued expressions are rejected; a narrow
constant-string `in` / `not in` operand is admitted because it yields the same
membership result. Includes, imports, reflective attributes, loaders and
file/network globals are unavailable. The native helper withholds output until
rendering succeeds and emits fixed errors without diagnostics.

Twenty-four parity controls cover six inherited model templates with thinking
and generation-prompt toggles; separate controls cover tools, JSON spacing,
key order, Unicode escaping, character indices, boolean/None formatting and
loop controls. The installed Qwen3.8 Lily template was admitted and both
generation-prompt variants matched the isolated Python baseline (47 and 88
UTF-8 output bytes). Its source SHA-256 is
`c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`.
This metadata-only control loaded no model weights.

A Qwen3-0.6B converted-model smoke request rendered through the Rust helper and
completed generation with a 512-token context, 32-token output limit and greedy
sampling. It produced nonempty output, 120 UTF-8 bytes, with length termination.
No prompt, response or token vector was retained. The existing generator hash
was `078b654bd3691b55012db695ef52b0cd4a7723140bc0c25a7af0dcb317cb2847`.

## Validation and replay

Validation includes native resource/protocol regressions, the full offline
converter suite, packaging/install/launcher/signing tests, installed-wheel
rendering outside the checkout, repository policy checks, formatting and lint,
and the locked dependency advisory/license gate. CI is configured to exercise
the native helper on Linux; this local run used macOS. A full signed application upgrade,
all third-party templates, inference quality and numerical parity are not
established by these checks.

| Check | Result |
| --- | --- |
| Rust 1.98.1 release build, native protocol/resource tests, Clippy at `-D warnings` | Passed; 16 tests |
| Full offline converter suite on the final helper | 5,923 passed, 89 skipped |
| Packaging, installer, launcher, signing, DMG, updater and qualification unit suites | 209 passed |
| Repository policy unit suite | 41 passed |
| Editable/sdist/wheel/provenance checks and installed native rendering outside the checkout | Passed |
| Actual distribution helper verification and dependency notice harvesting | Passed; arm64, system-only linkage, macOS 26.4 target, 141 registry dependency notices |
| Locked dependency advisory/license/bans/source gate | Passed; existing maintenance warnings retained |
| Ruff, Rust formatting, shell syntax, lock resolution, source hashes and documentation links | Passed |

The converter skips remain 88 opt-in external-reference cases without supplied
private artifacts and one inapplicable Hermes assertion. An existing loopback
API grant test exposed a timing race: its synthetic non-revoked client waited
for cancellation for the same three seconds allowed by the caller's join. The
harness now waits directly on each client's intended event; production API code
is unchanged. The final full-suite replay passed after that harness correction.

To replay, prepare a separate checkout at the Python baseline commit and run
`uv sync --project BASELINE/tools/converter --locked --python 3.13.13`. Compile
its dependencies and project modules with `python -m compileall` using that
environment. Then run from the native checkout:

```sh
MACOSX_DEPLOYMENT_TARGET=26.4 cargo +1.98.1 build --release --locked -p stark-template
uv run --project tools/converter --locked --python 3.13.13 python \
  tools/converter/scripts/benchmark_template_runtime.py \
  --baseline-project BASELINE/tools/converter
```

The driver fails with a fixed diagnostic on output mismatch and prints only
aggregate measurements and implementation hashes. A future change to the
evaluator lifecycle or compatibility policy should record a new study with
matching source, template, environment and resource controls.
