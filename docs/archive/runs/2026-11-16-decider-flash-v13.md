# Decider Flash v1.3 native text/image qualification

Date: 2026-11-16. Attempt: `20261116-DF13-03`. Implementation:
`61a2a27d1cd0359b085b3e7d00b9396f870b7ae7` on `codex/decider-flash-v13`.
Predecessors and failed attempts are retained in the
[model experiment ledger](../../model-experiments/2026-11-16-decider-flash-v13.md).

The native BF16 Rust/Metal worker passed the fixed numerical and lifecycle gates
against Perplexity's unmodified FP32 MPS reference. This qualifies the named
artifact and bounded corpus. Matched tower-residency text timing was
inconclusive under the predeclared variability rule. No release, installation,
desktop-catalog integration or broad performance promotion is claimed.

## Source and artifact bindings

| Binding | Identity |
| --- | --- |
| Source | `perplexity-ai/pplx-decider-flash-v1.3` at `d06fb3f2d97e0a52325a0d463c450cf0338f44fe` |
| Native artifact | Private `runtime/decider-flash-v13/native-v2`; default image tower retained |
| Native configuration SHA-256 | `d859a249a3089252f164c755d2a584e41c1041ab91dc32751a742ddad29e8337` |
| Native weights SHA-256 | `c3ab2961a63c3342ff3390b41036c6029d62095414db4cffbd59a8944c754c97` |
| Native worker SHA-256 | `4e23cb1cb0e611fd0d2d2ea0eaff27e6ccf0ca722de85c39e065c2b61b8f3aeb` |
| Reference source aggregate SHA-256 | `6a1555c5998ffcaba54a299599b45591002a7941b5d383bcef5bf35bf4d49481` |
| Reference runner SHA-256 | `2e94e02c84feea14186e57ee70a734f35f7f3072c63d4191795eb1f85c03f54c` |
| Qualification script SHA-256 | `5f0d8298786b76d6ca34b08fc40a6e8bd8fd702244adead04043d6222e3fd074` |
| Fixture SHA-256 | `3c302223fcd3491aaa47389dc91743a11627f0b8f61b555ffce6877e59969d4c` |
| Aggregate receipt SHA-256 | `8653ade4d7717c0acb0d77ca806c5263606b2cf0579add482048ff30146588c6` |

The aggregate receipt is private
`runtime/decider-flash-v13/qualification-03.json`. Inputs, images, token vectors
and responses are not published. Reference/native response traffic remained in
memory and private pipes; no raw response files were written.

Host: Apple M5 Max, 128 GiB RAM, macOS 26.7.2 (`25G332`). Native build:
Rust 1.98.1, release, locked, `stark-score --bin stark-decide` without the
scorer's optional optimized features. Native matrices/activations are BF16;
F32 reductions, controls, convolution history and recurrent state follow the
Qwen reference operator policy documented in the ledger.

The locked reference uses torch 2.14.0, torchvision 0.29.0, transformers 5.17.0,
safetensors 0.8.0, pillow 12.3.0 and huggingface-hub 1.31.0. It loads the verified
upstream source without patches, with `device="mps"`, `fast=False`,
`locate=False`, `mode="separate"`, 4,096-token admission and eight CPU threads.
Upstream selects FP32 on MPS. This does not qualify its CUDA BF16/compiled path.

## Numerical and lifecycle results

The private synthetic corpus has 12 requests / 15 decisions: ten text decisions
and five image decisions. It covers binary, custom choices, ordinal scoring,
Unicode/structured state, escaped control markers, 255 options and longer text.
Image controls cover PNG/JPEG, portrait geometry, a source larger than the saved
pixel maximum, and transparent PNG. These are simple colour controls, not a
general vision-quality benchmark. Inputs span 79–2,106 positions; the largest
image request has 1,616 positions after native preprocessing and expansion.

| Gate | Result |
| --- | --- |
| Top-choice agreement | 15/15, including 5/5 image decisions |
| Maximum absolute valid-option probability error | 0.00595570394, below the fixed 0.02 limit |
| Mean absolute probability error, weighted over valid options | 0.0000553653163 |
| Maximum image probability error | 0.000152042326 |
| Input-token usage and sequence counts | Exact for all 12 requests |
| Repeat after intervening image requests | Exact native response |
| Invalid-schema/reset recovery | Passed, same worker |
| Invalid encoded image | Rejected; Python client terminated its worker |
| Cancellation | Owned worker killed with a long request pending; no reply yet |
| Restart after failure/cancellation | Exact fresh native response |
| Cross-request state/results | None retained; readiness advertises no prefix/result cache or MTP |

The earlier independent formatter differential also matched all 6,460 tokens
across the 15 sequences exactly. It used the official image grid for placeholder
expansion; this full-model run additionally verifies native preprocessing through
exact usage and matched image decisions.

Every complete decision is serialized. Each forward owns fresh attention KV,
F32 convolution history and recurrent state. A causal prefix is invalid because
earlier full-attention representations depend on later question tokens. No
autoregressive generation or compatible predictor tensors exist in this export;
MTP and generation-prefix controls are explicitly N/A for this route.

## Matched text with the image tower resident

The same 79-position text question, decoder, head and arithmetic were run in
four sequential worker loads: tower off, on, on, off. Each load excluded one
warmup and measured three worker round trips, yielding six observations per arm.
All valid-option readout hashes were identical across modes. No compilation, downloads,
other qualification processes or concurrent reference/native inference ran
during these measurements.

| Metric | Tower off | Tower on |
| --- | --- | --- |
| Median worker latency | 1.372959 s | 1.374153 s |
| Observed range | 1.253354–1.464096 s | 1.293100–1.451129 s |

Median ratio: 1.000870 (approximately +0.087%). The baseline relative range was
15.35%, exceeding the predeclared 10% variability ceiling. The timing gate is
**inconclusive**, despite similar medians. No speedup or established performance
parity is claimed, and the noisy run was not rerun until it passed.

Existing swap use stayed 3,299.06 MiB. Pageouts remained 146,097 and swapouts
220,868 from pre-run through completion. No new paging was observed. Peak RSS
and device working-set high water were not measured.

The current reference-operator implementation is slow on longer requests:
observed functional-run latency was 36.45 seconds for 2,106 text positions and
35.17 seconds for the 1,616-position large-image request. These are single
functional observations. Throughput and comparative performance remain
unqualified. Optimization is a separate follow-up.

## Checks, failures and disposition

The initial complete converter suite passed 7,844 tests (95 skipped), and the
CPU/Metal engine suite passed 1,081 tests (163 ignored) plus integration groups.
Subsequent fixes were checked with 170 focused converter/Qwen tests, the new
CPU-only release admission regression, real-artifact rejection/forward controls,
and this complete model replay. Repository policy, its 11 regression tests,
formatting, lint and qualification-comparison controls passed.

Two native integration defects were corrected before this result: a live token
constant handle at command submission, and missing spatial M-RoPE metadata in
the dedicated writer. Their failed runs and identities remain in the ledger;
no qualification transfers from those artifacts. The old scalar-RoPE artifact
is now rejected before weight allocation. No input/exception instrumentation
remains in the worker.

Disposition: adopted on the development branch, with bounded numerical and
lifecycle qualification. Performance remains inconclusive; broader image tasks,
the complete 4,096-position ceiling, joint-question mode, Locate, quantizations,
the 27B checkpoint, desktop integration, installed builds and releases remain
outside this result. Keep the [native usage and limits](../../DECIDER.md) explicit.
