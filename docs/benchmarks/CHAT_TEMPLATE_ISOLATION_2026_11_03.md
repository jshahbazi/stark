# Chat-template isolation latency

Date: 2026-11-03. Initial component baseline; no earlier template-isolation
measurement is superseded.

## Why this latency exists

Model-provided Jinja templates are executable, untrusted inputs. The evaluator
starts a fresh restricted Python process for each compilation check and render.
The parent can enforce deadlines, monitor resources, terminate cancelled or
runaway work and reap the child. Each operation receives fresh evaluator state
and resource budgets. The template handle retains source and recompiles it for
every render, so process startup, imports, compilation and private-pipe JSON
transport contribute to request latency.

Normal chat renders once per turn before tokenization and inference. Decision
scoring renders twice to verify the assistant answer boundary. Template loading
also performs an initial compilation check. Native token generation does not
call the evaluator.

Any future reduction in this overhead must preserve isolation of model-provided
programs, independent request budgets, cancellation and child cleanup, and the
rule that conversation content is never logged or persisted by the evaluator.
Measure rendering and end-to-end latency separately when comparing a replacement.

## Source and environment

- Implementation: `dacaebca29ce6926c87389a0adadda9b6da0bfc2` ([PR #70](https://github.com/jshahbazi/stark/pull/70)).
- Previous behavior: `chat_template_environment().from_string(source)` compiled
  once in the host, followed by repeated `template.render(**arguments)` calls.
  This reproduces the renderer used at base
  `4545a409d32703bbf69ccc9be85d2d2f43aa5c4d`.
- Isolated behavior: construct `IsolatedTemplate(source)`, then repeatedly call
  `render(**arguments)`, including a fresh child and compilation in every call.
- Host: Apple M5 Max (`Mac17,7`), 128 GiB RAM, macOS 26.7.1 (25G241), arm64.
- Runtime: uv 0.12.24, CPython 3.13.13, Jinja2 3.1.6, psutil 7.2.2, using the
  locked `tools/converter` environment.
- Template: checked-in [Qwen3-4B fixture](../../tools/converter/tests/fixtures/qwen3_4b_chat_template.jinja)
  with its [provenance](../../tools/converter/tests/fixtures/qwen3_4b_chat_template.provenance.md).

| Input or implementation | SHA-256 |
| --- | --- |
| `qwen3_4b_chat_template.jinja` | `87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5` |
| `template_runtime.py` | `1ca2a313bfd7b48059aeab52d5f3077053cb7ea663b15db91b1e48813f6d9353` |
| `template_compat.py` | `7d51ad9776c4a9e6d30f0358c7defde033d5e38b610ada3b0f88c316811dedbc` |
| `tools/converter/uv.lock` | `df7711726b1ed7e235ded21b9e6e1cbe130924be4d2c580d5d6bd9005248f8d5` |

## Method and retained aggregates

Both renderers used the same template and synthetic messages alternating
`user` and `assistant` roles. Each short-case message contained 64 ASCII `x`
characters; each history-case message contained 2,048. Arguments were
`tools=None`, `add_generation_prompt=True`, `enable_thinking=False`,
`bos_token=""` and `eos_token=""`. Rendered outputs matched between the two
implementations for both cases; no conversation content was printed or saved.

Construction was timed five times per implementation. Then each conversation
case ran three warmup pairs and 20 measured pairs, always the previous renderer
followed by the isolated renderer, using `time.perf_counter()` around each full
call. Imports in the parent and template construction were outside render timing;
child startup, imports, compilation, transport and cleanup were inside isolated
render timing. All measured render samples were included.

| Case | Messages / content bytes | Previous median (ms) | Isolated median (ms) | Previous min–max (ms) | Isolated min–max (ms) |
| --- | --- | --- | --- | --- | --- |
| Short | 2 / 128 | 0.142 | 48.653 | 0.127–0.244 | 45.694–69.286 |
| History | 32 / 65,536 | 0.357 | 61.709 | 0.309–0.429 | 49.854–68.353 |

Median construction time was 5.575 ms for the previous renderer and 50.028 ms
for the isolated handle, including its initial child compilation check. The
tables retain aggregates; individual sample timings were not retained.

## Interpretation and limits

The observed additional render latency was approximately 49–61 ms. Decision
scoring's two renders suggest approximately 100–125 ms of added latency for
similar inputs; that is an extrapolation, and decision requests were not timed.
These component measurements do not establish end-to-end first-token latency,
generation throughput, model-load timing or whether a person notices a delay.
This startup cost can be a larger share of latency for short requests.

This was one local campaign with fixed implementation order, warm filesystem
caches and no background-activity, power or thermal controls. It loaded no model
and ran no inference. Other templates, tool schemas, conversation lengths,
packaged Python environments and machines can produce different timings.
