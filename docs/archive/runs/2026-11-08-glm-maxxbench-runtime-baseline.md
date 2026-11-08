# GLM native API runtime diagnosis — 2026-11-08

## Scope and source

This synthetic diagnostic separates the earlier MaxxBench client-deadline
mismatch from GLM's native prefill cost. It is not a MaxxBench score, a broad
quality result or a qualified throughput comparison. No task was executed.

The caller was `1678e1a7`; its Rust sources are byte-identical to the native
build revision `b9a91a94ea07313cc0558d556f26a300c6603ff1`. The model was the
existing GLM-5.3 Flash `UD-Q2_K_XL` prepared artifact: weights from
`unsloth/GLM-5.3-Flash-GGUF` at
`a38483c8cd5df544f53d70fb281afe97369d5ab6`, metadata from
`zai-org/GLM-5.3-Flash` at
`04c4e9e95c5da8862dced7e5056455116f83a7e0`. Its previously sealed artifact
fingerprint is `342a475b9544448aac3206385810963fdb1133272b2d5aae57f0bc5c89f64e3f`.
This run checked metadata and file identities before and after; weight payloads
were not rehashed for this diagnostic.

The host was Apple M5 Max with 128 GiB unified memory, macOS 26.7.2. The native
worker used Rust 1.98.1. The caller used the locked Python 3.13.13 converter
environment and the model's native template helper. Inputs, responses and token
vectors stayed in memory; only aggregate counts, booleans and hashes were saved.

| Executable | SHA-256 |
| --- | --- |
| Baseline generator | `a589fb72b94ffd8381642ab898f348d53bc3d78d5e59c36ddf294e023ebb2d79` |
| Baseline template helper | `1765a2c28309db99b8095da015035b89acc2cf4759752fba931079121421eb97` |

## Corrected deadline and prefix reuse

The API and worker deadline was 1,800 seconds; the HTTP client deadline was
1,810 seconds. The earlier 600-second client was not used. Every request used
the artifact's configured maximum context, **1,048,576 tokens**, greedy sampling,
seed zero, eight requested output tokens, MTP off and optional thinking off.
The actual synthetic inputs were modest; this does not validate a million-token
prefill or prove the memory feasibility of filling that window.

All four HTTP requests succeeded:

| Case | Input | Output | Reused input | Native prefill s | Native decode s | HTTP wall s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cold | 1,180 | 8 | 0 | 238.429 | 2.819 | 241.270 |
| Growing follow-up | 1,203 | 8 | 1,024 | 52.052 | 2.111 | 54.204 |
| Repeated follow-up | 1,203 | 8 | 1,024 | 41.564 | 1.197 | 42.795 |
| Explicit reset, fresh replay | 1,203 | 8 | 0 | 242.459 | 1.249 | 243.740 |

The repeated and reset-fresh API choices were exactly equal. This demonstrates
correct reuse for this bounded greedy session through the authenticated API's
principal scope. It does not qualify other sampling, MTP or long-context paths.
There was no transport timeout in this run. The old pilot remains an unscored
attempt; these synthetic requests do not replace it with a model score.

The shared execution lock serialized the run. Admission required 30 GiB free
disk and a 12 GiB live reserve. File/source identities matched afterward, and
the owned worker was reaped without cleanup failure. The diagnostic did not
screen clocks or establish a general performance floor.

Aggregate receipt SHA-256:
`4e6ddd83192009f28026c12a12c70d86b4d852948d0070b81b8dd310a275de35`.

## Actual execution path

A separate fresh-worker replay enabled only the existing
`STARK_NATIVE_PROFILE_OUTPUT` aggregate counter mode. It used the same baseline
bytes, configured context and modest synthetic request. That selector enables
fixed numeric native counters; it does not enable the BF16-specific stage hook,
MMA, encoder banks or another precision policy. Its timings are kept separate
from the profiling-disabled baseline above.

The profiled call succeeded with 1,180 input and eight output tokens, no reused
tokens, 216.970 seconds prefill and 2.095 seconds decode. Positive Metal
execute/GPU/static-cache/concurrent-expert counters attest actual Metal work.
The prefill recorded 50,740 command execute/wait completions, approximately 43
per input token. Inclusive native wait time was 204.867 seconds; reported GPU
time was 69.258 seconds and encoding time 7.926 seconds. These inclusive categories
must not be added together or interpreted as utilization percentages.

Mapping recorded 1,190,034 calls and 3.27 TB of cumulative mapped-range accounting.
That is not physical I/O, resident memory or allocated payload size. Scalar
prefill's synchronous command boundaries remain a performance limitation; this
run does not establish CPU fallback or an optimization result.

The profiler replay used an explicitly different runtime-only disk admission:
24 GiB minimum with the same 12 GiB live reserve, no Docker builds and the same
modest actual request. That policy does not change the 30 GiB admission for
future Docker benchmark tasks. Its worker was reaped and cleanup passed.

| Private aggregate | SHA-256 |
| --- | --- |
| Profile replay receipt | `cecf53bc186ac4ce70e0562c9e810678a5720f0b85c3cce420c1304e4b32e012` |
| Fixed native profile counters | `c9cb178f877d891c33e6b76a507b31714da177f3c9348e026dff4e5fac6095d2` |

## Transport correction and remaining qualification

The baseline also exposed byte guards that would prevent a full context from
reaching native inference: generator pipe input was fixed at 1 MiB, and the
template path had fixed input, output, allocation and execution budgets. The
implementation accompanying this record adds trusted per-process overrides and
explicit Python client budgets. Production defaults and the template sandbox
remain unchanged; request data cannot select a larger policy. Nothing truncates
a prompt, token sequence or response to fit the transport.

Model-free checks cover explicit large request admission, readiness agreement,
template process isolation, preserved default rejection and policy injection
rejection. Non-Metal generator tests exercise the changed framing logic; the
Metal backend call sites and new release executables still require the combined
integration build and artifact qualification. The baseline results above are
bound to the old executables and are not reissued for the new transport code.

On the accompanying implementation tree, 82 non-Metal generator tests, two
template unit tests and 19 template protocol tests passed under Rust 1.98.1.
The focused Python template/generation-process suite passed 121 tests, including
Unicode/control-byte round trips, a cancelled expanded render, and an abrupt
owner-exit case that first checks the helper is still running. Those lifecycle
checks use tiny synthetic requests, not maximum-size allocations. Template
Clippy passed with warnings denied; generator Clippy for the non-Metal binary
configuration encounters existing unused production-path definitions, so its
Metal check remains part of the combined integration validation. Ruff,
formatting and repository policy checks passed. The model-free aggregate
receipt is `f6ac8b6d87bfd5d07750995d1c83d0bb7910bdec2cb3776cfb48ce60faf029f2`.
