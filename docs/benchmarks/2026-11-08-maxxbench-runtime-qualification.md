# MaxxBench runtime and API qualification — 2026-11-08

**Result:** the bounded Flint prefix replay and two native API transport checks
passed on the new integration build. These checks qualify their stated runtime
paths. A fresh 89-task oracle cohort and 623 model/task attempts remain pending;
there is no new MaxxBench score or model ranking.

The [earlier coding-agent pilot](2026-11-08-maxxbench-native-pilot.md) and
[GLM baseline diagnosis](../archive/runs/2026-11-08-glm-maxxbench-runtime-baseline.md)
retain their original sources, executables, deadlines and observations. The
[Flint core qualification](../archive/runs/2026-11-08-flint-prefix-core-qualification.md)
separately records 15 successful short native generations, one cancellation,
prefix-reuse counts and agreement with fresh candidate/reference executions.
Its actual histories contained 1,152–1,183 tokens at a configured 262,144-token
context. Those results do not establish a filled maximum window or throughput.

The API checks used caller `3d2d5d76fc32a01ed90f7057d2f0a46e27b9c66e` and
Rust 1.98.1 release binaries built at
`aae3d17761552e360d8f1497ab853fc45047ba18`. The intervening caller change was a
test-fixture correction; native sources and executable bytes were unchanged.
Host: Apple M5 Max, 128 GiB unified memory, macOS 26.7.2. The controller used the
preserved, locked Python 3.12.13/Pier 0.3.1/mini-swe-agent 2.4.6 environment;
the bridge used the locked Python 3.13.13 converter environment. Each model
held the shared hardware lock through cleanup. Both bridges exited cleanly,
their owned workers were reaped, and the lock was available afterward. Both
cleanup receipts record `failure: null` and `cleanup_failed: false`; the final
release state has no pending cleanup (`cleanup_pending: false`).

| API check | Configured context | Actual HTTP body | Actual prompt / output tokens | Result |
| --- | ---: | ---: | ---: | --- |
| Production-default HTTP admission control, no loaded model | — | Header announced 2,307,535 bytes; zero body bytes sent | No completion | HTTP 413 `body_too_large` |
| Qwen3 0.6B, isolated benchmark policy | 40,960 | 2,307,535 bytes | 18,078 / 16 | HTTP 200 |
| GLM-5.3 Flash, isolated benchmark policy | 1,048,576 | 145 bytes | 17 / 16 | HTTP 200 |

The Qwen request contained a valid synthetic call/result history with a
2,306,876-byte tool observation, mostly spaces bracketed by non-whitespace.
The template retained that span. The request exceeded the old 1 MiB HTTP/text
and generator-input guards, 2 MiB template-request guard, and 256 KiB
tool-result guard while fitting the configured token context. It supplied the
pinned bash schema and `tool_choice: none`; no tool was executed. The default
HTTP control rejected its declared length before reading a body or calling
completion. The GLM request used ordinary raw chat with no tools. It verifies
real startup/readiness and a short completion through the new native/helper
budget interfaces at the full configured million-token context.

Both diagnostics used temperature 0, seed 0, MTP off, optional thinking off and
**16 requested output tokens**. The production matrix still requests 4,096;
the native 4,096-token ceiling was unchanged. Context, step limit 400, pinned
bash schema, serializer and policy-envelope inputs were otherwise unchanged.
The 400-step policy bounds framing; each diagnostic made one API request.
Envelope inputs were 16 calls, 6,785 messages, 7,382,759 request-framing bytes
and 6,266 response-framing bytes, with `json-utf8-compact-v1` serialization.

The policy sidecar retains every numerical budget and its tokenizer SHA,
normalizer, byte-bound version, template/parser/argument-shape proof, schema
SHA, context/output/step inputs and codec metadata byte count. Its reconstructed
digests matched actual readiness. It was sealed later at `96bbaeaa` after
verifying all nine recorded executed source-file hashes, executable hashes and
artifact identities remained unchanged; the executed caller stays `3d2d5d76`.

| Policy | Qwen3 0.6B | GLM-5.3 Flash |
| --- | --- | --- |
| Diagnostic, 16 output tokens | `4321eb28198037dc61e225535b22c7fd0cd8889b95c98d5841f4de200671bcc8` | `923d2fb3a249aaecf39c2a423e4e3c9a1c777a9fa7fc5569d36142de687f8174` |
| Production matrix, 4,096 output tokens; not executed here | `9179334905e3cf625ab37f2bb0325641c50ffbd0bfe88bf6110c9a9a1c91f5d2` | `a13d9169d29f337225e3a3449c7d7b0bc469b5e81064500887d3c7287de21a90` |

The declared context maxima were not lowered. Request histories and observations
were retained intact; generation used the stated output budget. The actual
18,078-token and 17-token inputs do not fill
their 40,960-token and 1,048,576-token windows. These are transport/admission
checks, not long-context quality, numerical-equivalence, memory-feasibility or
performance results. Resource-guard failures remain infrastructure outcomes;
they cannot become model-quality scores. The [runner guide](MAXXBENCH.md)
defines the scoped policy and unchanged production defaults.

Validation coverage is recorded without combining unlike runs: the integration
passed three Flint CPU prefix tests and 124 generator tests, with two ignored.
The converter run reported 1,268 passed, one skipped and one failing media
fixture. That fixture still patched obsolete module globals; the test-only
correction migrated it to immutable instance limits while retaining rejection
before body reads and reservation release checks. All nine media cases then
passed; the complete converter suite was not rerun after that correction.
Separately, the API change passed 514 focused API/parser/bridge/runner tests
and 24 adapter tests. The initial focused selection omitted the media module;
the follow-up media/LocalAPI/limit selection passed 56 tests.

Only aggregates and hashes were retained. Inputs, model replies, token vectors
and credentials stayed in memory or private pipes. Existing artifact receipts
matched all four Qwen and 12 GLM file identities before and after execution;
these diagnostics did not rehash weight payloads. Exact weight source and
representation remain those in the linked pilot and GLM baseline records.

```text
worker                  c5514016280ed85c69dac972373a688db85d10ab5810d134a5b3fe2ed8449415
template helper         fdbf8a879cd15f6bea2c84e03d6afb6f16e6b49c828fce8a656d3b403a8cc588
production matrix       76fa913490615f0bb437f585aea653ecf35c9e94dcf3a21bc52f0caf806e0ca0
Qwen artifact           0a6f832045730183269c8d043404cbcd075dc8b3826389584ecec629b0b1c49e
GLM artifact            342a475b9544448aac3206385810963fdb1133272b2d5aae57f0bc5c89f64e3f
runner                  afdb65f875ab493a4e2aa0295cb3b84b10a0b5839689a9364f7ccaec0c04444b
adapter                 4b818813ecccdb1ea71d32078778182285609e4ad091c59cd121e1c4d47e9b9f
bridge                  460ac4f0160bd446d3ea344cbedda35390b2f2b768ed51890105e53b7ca287b6
byte policy source      841df0f5018af9546f0a715b66058e657d7b5593b9e05ce0a0d2c763354a1578
diagnostic observer     53fa69ad50d4c66990322ee2dda9efe7cd1d4056045f713e79e0524774cfaee8
policy-input sidecar    2980c3071b6eb60e4ad9f996f6440073ab47494e59f28def0659c07cf492cb9e
Qwen result             1453b27df5eeb0a70c07721345f60c3c4eb95f6d75b1cb7440b482a30d3af6f5
Qwen bridge cleanup     277edaed2db6bc69931070d2862960023f1164f67b1746de087693e305552d1d
GLM result              d10a5fbf61e8b435b24c4308b33be79c4c11a54a59aa47c6d6039e246d9dfd45
GLM bridge cleanup      bfaf0eed12f855a57dfe6517ae1dbb33411bd9b7c76083927f0d92afc354be8a
```
