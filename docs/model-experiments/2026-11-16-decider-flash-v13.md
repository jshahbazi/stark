# Decider Flash v1.3 native image-capable scoring

Attempt ID: `20261116-DF13-01`

## Scope and fixed acceptance

Implement Perplexity `pplx-decider-flash-v1.3` BF16 classification/scoring with
images retained by default. Baseline source is Stark `5212b7c2`; branch
`codex/decider-flash-v13`. No PR, release or app installation is requested.

Hypothesis: the existing dense Qwen3.5 vision/DeltaNet engine can execute the
modified backbone with sequence-wide noncausal full-attention layers and a
separate marker readout. Linear attention remains causal. Match the upstream
default `separate` question mode before considering joint execution.

Before model comparisons, pin the upstream revision and source/file hashes.
Keep all model files, fixtures, raw outputs and reference artifacts ignored.
Publish aggregate metrics and hashes only. Never publish request content,
tokens, probabilities or exception dumps.

Fixed initial correctness gate: finite outputs, 100% top-choice agreement on
a bounded corpus containing binary, choice, ordinal and real image requests;
maximum absolute probability error <= 0.02 versus the matching BF16 upstream
reference. Report numerical error separately from agreement and task accuracy.
If this gate fails, retain the failed result and diagnose before promotion.
Model-free controls must prove future-token visibility in full attention,
preserved causal recurrence, marker/head mapping, option normalization,
malformed-input rejection and same-worker recovery/reset.

## Precision, state and image policy

Initial target: unquantized BF16 weights/activations and the existing native
Qwen3.5 recurrent/accumulation policy, recorded precisely after source review.
Retain the vision tower and processor. Establish matched text execution with
image weights loaded; compare explicit diagnostic tower-off loading only at
matching arithmetic. Initial performance gate: repeated alternating paired
text latency within 10%, provided baseline dispersion is below that bound;
otherwise label the timing inconclusive. Performance is not yet measured.

This is a non-generative checkpoint with a 255-row decision head rather than a
vocabulary head. Inventory actual source tensors before deciding MTP status.
Autoregressive generation/tool calling are outside this model's contract.
Assess cache correctness independently: full-attention prefix representations
depend on later question tokens, so ordinary generation prefix snapshots are
invalid. Retain at most the immediately preceding identical complete text
request if implemented; image/failure/reset must clear retained state/results.

## Workload and resource bounds

Use synthetic private fixtures and generated local image controls, then record
their hashes. Reference and native inference are serialized. Begin with short
requests and bounded images; no full-context claim. Account for source plus
converted weights and independent build output within current free disk.
Initial machine reports 128 GiB physical RAM and approximately 59 GiB disk free.
Stop model acquisition/conversion if it would consume the reserved 15 GiB free
disk margin. Do not alter another task's model workers.

## Attempt history

| Trial | Observation | Disposition |
| --- | --- | --- |
| Source assessment | Upstream exports Qwen3.5 backbone plus separate decision readout; current generic Decision path is causal next-token scoring. | Native Decider route required. |

## Identity and results

Upstream artifact, dependency/tool versions, executable/configuration hashes,
actual trials, measured memory, numerical results, lifecycle controls and
delivery status will be appended as work completes. At registration the state
was planned, with no native or reference model run. Locate span extraction and optional isolated
multi-question execution require separate implementation/qualification if not
included in this attempt.

### Implementation checks before model execution

Native implementation uses `DeciderModelConfig`, full-sequence noncausal full
attention, unchanged causal DeltaNet, marker readout and image-enabled default
loading. It admits one PNG/JPEG and at most 4,096 expanded positions. Locate is
explicitly deferred following the user's question about whether it is needed;
it is unnecessary for this classification/scoring scope.

- Repository policy, formatting, whitespace and focused Python lint: passed.
- Converter suite: 7,844 passed, 95 skipped, 425.22 seconds.
- Complete native CPU/Metal engine suite: 1,081 passed, 163 ignored; all four
  subsequent integration/doc-test groups passed (7, 2, 1 and 2 tests).
- Dedicated worker tests: 2 passed. Qualification-comparison controls: 3 passed.
- Actual saved tokenizer/template preflight: 255 code mappings validated;
  15 planned sequences, 79–2,106 unexpanded positions. Full-model comparison
  has not run at this checkpoint.
- Reference environment imports and MPS triangular-solve prerequisite passed.
  The unmodified reference chooses FP32 on MPS/CPU; the initial full-model
  comparison will use MPS FP32 versus native Metal BF16. This refines the
  initial matching-BF16 intent above: no CUDA reference is available locally.
  The predeclared 0.02 probability error gate is unchanged.

The first complete engine run had 1,079 passes and one failure: an existing
internal spatial-position test asserted the old 4,096-patch ceiling. The shared
tower now permits Decider's 6,400 patches. The corrected tests cover exact
6,400 admission and larger-grid rejection, plus a separate real-image check
that existing public vision wrappers still reject above 4,096. The final
complete suite above passed; the initial failure is not discarded.

Initial worker compilation found an ambiguous `File::by_ref` call; it was
disambiguated to `Read::by_ref` and the worker built/tested successfully.
The shell initially lacked Cargo on PATH; subsequent commands used the
installed `/Users/johnshahbazian/.cargo/bin/cargo` with the pinned toolchain.

Acquisition first stalled using the default Xet transport. An owned downloader
was stopped after inactive connections and unchanged partial files were
confirmed; the same pinned files resumed through official Hugging Face HTTP
transport. No registry, credentials or access policy changed. The upstream
release manifest's NOTICE size/hash differs from the actual file at the pinned
revision; actual NOTICE matches that revision's Hugging Face Git blob. Pinned
file identities are the acquisition authority; the original manifest and
NOTICE are preserved.

| Identity | Value |
| --- | --- |
| Checkpoint revision | `d06fb3f2d97e0a52325a0d463c450cf0338f44fe` |
| Host | Apple M5 Max, 128 GiB RAM, macOS 26.7.2 (`25G332`) |
| Native toolchain | Rust 1.98.1 (`48a229cea`), nightly-2026-10-01 tests/formatting; uv 0.12.24 |
| Native executable SHA-256 before model run | `e44600418d9489925e499b63a23aa12932c0de2bb0bebb3a3e42192c6ddcb8a3` |
| Template executable SHA-256 | `99a09d9c92d0a78214bce3053a4f013e94b8a1526dc4ff7195061a0d540d40a9` |
| Cargo lock SHA-256 | `7e6dfa207bd932d707c78223e70ef058f2b65d547f8ca6064b1b41117496066c` |
| Converter lock SHA-256 | `df7711726b1ed7e235ded21b9e6e1cbe130924be4d2c580d5d6bd9005248f8d5` |
| Reference lock SHA-256 | `531ebccfcda64b6d548cf4448212c194108a1a897994828e8f76cc478e072712` |
| Reference dependencies | torch 2.14.0, torchvision 0.29.0, transformers 5.17.0, safetensors 0.8.0, pillow 12.3.0, huggingface-hub 1.31.0 |

The private qualification fixture contains 12 requests / 15 decisions, including
binary, choice, ordinal, Unicode/structured state, escaped protocol tokens,
255 options, a longer text and five image controls (PNG, JPEG, portrait, large
image requiring max-pixel resizing, transparent PNG). Raw inputs and responses
remain private. This is a bounded engineering comparison, not a quality
benchmark. Neither full-context accuracy nor installed-app behavior is claimed.

### Exact upstream encoding differential

Before loading weights, the official unmodified processor/Encoder and Stark's
formatter were run in separate locked Python 3.12/3.13 child processes. Across
all 12 fixture cases / 15 sequences (five image sequences), 6,460 tokens matched
exactly, with zero marker, segment, question-key or option-count mismatches.
For this formatting comparison only, the native image placeholder was expanded
using the official grid; native image preprocessing itself still requires the
full-model check. No model or GPU was loaded for this differential.

- Matching sequence SHA-256:
  `7ebe2adda92daacfc494daddb64bc386b5f595b396200bc32b916332e511c7cd`.
- Fixture SHA-256:
  `3c302223fcd3491aaa47389dc91743a11627f0b8f61b555ffce6877e59969d4c`.
- Upstream source aggregate SHA-256:
  `6a1555c5998ffcaba54a299599b45591002a7941b5d383bcef5bf35bf4d49481`.
- Reference runner SHA-256:
  `2e94e02c84feea14186e57ee70a734f35f7f3072c63d4191795eb1f85c03f54c`.

State at this checkpoint: implemented and model-free/formatter checks passed;
full-model numerical qualification and matched residency timing pending.

Staging the new reference lock exposed the repository's scan-coverage gate.
The lock was added to the existing OSV scanner and report-gate inventories;
repository policy and all 11 policy regression tests then passed. This records
coverage configuration, not a completed OSV vulnerability scan. Original
upstream licence/notice bytes (including CRLF) are preserved using the existing
file-specific Git attributes convention.

### Candidate seal before numerical execution

Implementation commit: `5c582fa0f6ff8ac13ad27a5a7b910c75063e43dd`, pushed to
`codex/decider-flash-v13`. Rebuilding from that commit produced the same native
executable SHA-256 recorded above. No PR was opened. The initial GitHub Build
run was still in progress when checked; no CI pass is inferred from local tests.

| Native precision/state component | Policy |
| --- | --- |
| Embeddings, dense projections, vision and decision matrices | BF16, no weight quantization |
| Decoder residuals, projection outputs and readout output | BF16 |
| Full-attention QK/AV dot accumulation | F32, with reference BF16 probability/output materialization |
| Attention KV | BF16; fresh allocation and logical length per request |
| Zero-centered RMSNorm | F32 normalization/gain through final BF16 cast; gated DeltaNet normalization follows its existing reference cast boundary |
| DeltaNet convolution coefficients/control vectors | F32 conversion/loading policy |
| DeltaNet convolution history and recurrent accumulator | F32; fresh zeroed state per request |
| Complete-result or generation-prefix reuse | Disabled; no retained cross-request tensors/results |
| MTP | Non-generative checkpoint; absent predictor tensors and no supported speculative path |

To preserve the 15 GiB conversion disk margin, this task's completed debug/test
build cache was cleaned with Cargo (14.3 GiB). Release executables, test logs,
source models, fixtures and source-bound receipts were retained. No other task's
worktree or cache was changed.

### DF13-01 full-model attempt and DF13-02 correction

Acquisition verified all 29 pinned files. The upstream release manifest also
has a stale README checksum; both README and NOTICE match the pinned repository
identities, and all source bytes are preserved. The BF16 preparation completed
with 604 operations / 724 input tensors, with output SHA-256
`c3ab2961a63c3342ff3390b41036c6029d62095414db4cffbd59a8944c754c97` and configuration
SHA-256 `44dec9e9ca8211aafd998aa011185540fa3aaef12571e390ce331a45d4702f9c`.

`20261116-DF13-01` completed all 12 reference requests on unmodified MPS FP32,
then failed on the first native forward before producing decisions. The
input-token constant handle remained alive through command submission, violating
Metal's constant allocator lifetime assertion. This is a native integration
failure, not a numerical result. Its private progress receipt is
`runtime/decider-flash-v13/qualification-01.progress.json`. A temporary diagnostic
collected only the static Rust source location, never the panic message,
backtrace, inputs or outputs; that diagnostic was removed before rebuilding.

Follow-up `20261116-DF13-02` explicitly drops the host token constant handle
after decoder encoding and before submission; the submitted command retains its
constant page. A rebuilt single-request native control passed (79 input tokens).
Corrected executable SHA-256:
`9cfda7079e80ed4ceb43726b0eefa1dcc86965bee556386b6965704be3d927d7`.
The same corpus, references and fixed numerical gates will be replayed in a new
receipt. A real artifact forward is the regression control for this lifetime
failure; the prior component tests did not exercise it.

At pre-run screening and after the corrective probe, system pageouts remained
146,097 and swapouts 220,868; existing swap use stayed 3,299.06 MiB. Existing
swap is historical host state, not new paging caused by the experiment.
