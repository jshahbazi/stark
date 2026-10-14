# Qwen3 GGUF compressed import experiment

Date: 2026-10-13. Implementation: `2642d705f63e8080b434d7ec835749d5c71ad77c`.

The official Qwen3-0.6B Q8_0 GGUF can be imported into Stark's existing native
compressed format and executed with the current Metal kernels. Every integer
code and emitted scale was independently verified. The import has a measurable
FP16-to-BF16 coefficient rounding cost and remains an experimental route.

On this machine and workload, the import reduced peak process RSS by 40.9%
against native dense BF16 and increased completed decode throughput by 44.7%.
Native MLX 8-bit had similar decode throughput, a smaller weight file, and faster
prefill. These results support a limited interoperability feature. They do not
establish an overall performance advantage over Stark's existing compressed
route or qualify general generation quality.

## Scope and source identity

| Input | Immutable Hugging Face revision | Representation |
| --- | --- | --- |
| `Qwen/Qwen3-0.6B-GGUF` | `23749fefcc72300e3a2ad315e1317431b06b590a` | Official `Qwen3-0.6B-Q8_0.gguf` |
| `Qwen/Qwen3-0.6B` | `c1899de289a04d12100db370d81485cdf75e47ca` | Original checkpoint, converted by existing native converter |
| `Qwen/Qwen3-0.6B-MLX-8bit` | `e53d3ae02ebc79eac17f1fa9e23a2b65ce227893` | Existing affine MLX 8-bit checkpoint, repacked by existing native converter |

The importer accepts a single little-endian GGUF v3 file with Q8_0 matrices and
F16/F32 tensors for this Qwen3 route. It requires the original registered HF
config and tokenizer sidecars. It checks architecture dimensions, attention,
RoPE, normalization, tokenizer vocabulary IDs, BPE merges, and stop metadata.
Other GGUF quantization types and architectures are explicitly rejected.

All three runtime models use Stark's `config.json`, `tokenizer.json`, and
`model.safetensors`. The experiment measures the imported weight representation
and its kernels. Direct GGUF loading is outside this implementation.

The independently implemented GGUF/Q8_0 contracts reference
[llama.cpp revision `1b70350c92563449576bcbbf552994b1d1ed847d`](https://github.com/ggml-org/llama.cpp/tree/1b70350c92563449576bcbbf552994b1d1ed847d),
which is MIT licensed. Existing engine and Lalamo provenance records are
preserved. Model source licenses remain with their pinned source repositories;
model artifacts are private and ignored.

## Compression and numerical adaptation

The importer transforms each signed Q8_0 code into Stark's unsigned symmetric
8-bit encoding with `xor 0x80`. It retains group size 32, fuses the existing QKV
and MLP projections, and arranges coefficients in the existing native matrix
layouts. It computes no new quantization. An independent readback checks every
emitted code and coefficient against the source, including coefficient padding.

Stark selects BF16 execution. Q8_0's F16 scales are rounded to BF16 before native
execution. The weight error below measures this coefficient adaptation against
the original Q8_0 represented weights; the source's existing quantization error
is separate.

| Check | Result |
| --- | ---: |
| Integer codes checked | 595,984,384 |
| Code mismatches | 0 |
| BF16 scale coefficients checked | 18,624,512 |
| Emitted scale bit mismatches against expected conversion | 0 |
| Relative RMS weight error from scale adaptation | 0.16955% |
| Maximum relative scale error | 0.38911% |
| Maximum absolute weight error from scale adaptation | 0.00387573 |
| Native weight file | 633,562,928 bytes |
| Peak explicitly tracked importer buffers | 6,920,704 bytes, within an 8 MiB budget |
| Import time including verification and hashes | 2.724 seconds |

The buffer figure excludes Python, NumPy, tokenizer, and metadata allocations.
The import time is a single warm-cache observation. Initial conversion and final
coefficient-readback validation are separate private receipts; the final receipt
is the one bound to the published evidence. Neither the catalog nor stored native
qualification tuples were promoted.

## Runtime method

- Host: Apple M5 Max, 18 CPU cores, 128 GiB unified memory, Darwin 25.6.0 arm64.
- Build: Rust 1.98.1, locked Cargo dependencies, release with default and
  `optimized` features. Worker SHA-256:
  `0cb4e1fcc02286af21213cf9c9b4871c758143839d570069d70baa56f4f5fc80`.
- Six deterministic authored cases, with input lengths 23, 23, 24, 26, 128, and
  512 tokens; context capacity 1,024; fixed-count greedy generation of 32 tokens.
- Two fresh-process rounds per model. Label order was shuffled with seed 1729:
  `mlx8, dense, gguf_q8`, then `dense, gguf_q8, mlx8`.
- One warmup and three measured repeats for each case in each round: 36 measured
  scoring calls and 36 generation calls per model. Models executed serially.
- Completed decode throughput counts the 31 returned tokens after the first,
  divided by total measured decode plus stream-drain time across all calls.
  Native timings exclude JSON and pipe serialization.
- RSS uses native `getrusage` and resident snapshots; tracked Metal allocations
  are reported separately. Unified memory shares pages, so these figures must
  not be added together. Filesystem cache, power state, and thermal state were
  uncontrolled.

The independent source reference used installed llama.cpp 0.5.0, build 11146,
commit `7fe450e19`, running the original GGUF through a loopback-only server.
Tokenizer IDs and exact input token arrays matched all six cases. Its timing is
used only for reference correctness. Reference binary and library hashes are
recorded in the evidence; library hashes were collected after execution.

Prompts, generated text, token vectors, logits, and HTTP responses remained in
private process pipes and memory. The published evidence contains aggregate
measurements and hashes. Models, conversion receipts, and native artifacts stay
ignored.

## Memory and performance results

MiB means 1,048,576 bytes. Peak figures are maxima across the two model rounds.
The prefill column is the range of per-round medians for the 512-token case.

| Native representation | Weight file (MiB) | Peak RSS (MiB) | Peak tracked Metal (MiB) | Completed decode (tokens/s) | 512-token prefill (tokens/s) | Load time (seconds) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense BF16 | 1,137.0 | 1,313.3 | 1,307.0 | 340.6 | 32,640–34,119 | 0.246–0.249 |
| Existing MLX 8-bit | 586.5 | 774.1 | 744.5 | 502.1 | 33,478–34,747 | 0.193–0.199 |
| Imported GGUF Q8_0 | 604.2 | 776.5 | 774.2 | 493.0 | 27,715–28,964 | 0.195–0.234 |

GGUF import decode throughput was 1.8% below MLX 8-bit in this run. The prefill
gap was larger. The representations use different quantizers and coefficient
layouts, so this comparison includes those differences. Repeated rounds on this
single machine provide limited evidence about stable rankings or other models.

## Correctness results and boundaries

| Comparison | First-token argmax agreement | Mean top-32 overlap | Exact 32-token generation hash agreement |
| --- | ---: | ---: | ---: |
| Imported GGUF Q8_0 versus native dense BF16 | 6/6 | 97.40% | 5/6 |
| Existing MLX 8-bit versus native dense BF16 | 6/6 | 97.92% | 5/6 |
| Imported GGUF Q8_0 versus llama.cpp original GGUF | 6/6 | 97.40% | 5/6 |

Against llama.cpp, imported Q8_0 had mean reference-top-32 log-probability RMSE
0.09475, with a per-case maximum RMSE of 0.21187. This aggregate measures the
source runtime's selected top 32 entries using each runtime's own normalization.
Full-vocabulary cosine similarity against native dense BF16 averaged 0.99922;
mean dense-to-import KL divergence was 0.001184. Native dense and MLX also had
generation differences against the source Q8_0 runtime.

For each native model, all 36 measured score hashes and all 36 generation hashes
matched their same-model reference. Generation's first token agreed with scoring
on all 36 calls. Those are repeatability and internal consistency checks.

The integer-code and emitted-coefficient checks establish the import's layout
contract. Runtime results establish scoped loading, scoring, and fixed-count
generation on these six cases. The divergent sequence and measured coefficient
rounding prevent a bit-exact source-runtime claim. These cases do not establish
semantic task accuracy, general generation quality, larger-model behavior, or
support for K-quants, IQ-quants, MoE, or arbitrary GGUF files.

## Reproduction and validation

Use the repository-pinned Rust 1.98.1 toolchain and existing locked uv project.
Download the source table's immutable revisions with Hugging Face tooling, then
convert the dense and MLX baselines using `stark-convert` as documented in the
[converter README](../tools/converter/README.md). For a new output directory:

```sh
uv run --project tools/converter --locked python -m stark_convert.gguf_import \
  --source /absolute/path/Qwen3-0.6B-Q8_0.gguf \
  --hf-source /absolute/path/original-qwen3-checkpoint \
  --source-revision 23749fefcc72300e3a2ad315e1317431b06b590a \
  --hf-revision c1899de289a04d12100db370d81485cdf75e47ca \
  --output models/gguf-experiment/qwen3-gguf-q8-verified

cargo build --release --locked -p stark-score --features optimized \
  --bin stark-gguf-experiment

uv run --project tools/converter --locked python scripts/gguf_experiment.py \
  --model dense=models/gguf-experiment/qwen3-dense \
  --model mlx8=models/gguf-experiment/qwen3-mlx8 \
  --model gguf_q8=models/gguf-experiment/qwen3-gguf-q8-verified \
  --output runtime/gguf-experiment/new-native-runtime-aggregate.json \
  --rounds 2 --warmups 1 --repeats 3 \
  --llama-server /absolute/path/llama-server \
  --gguf /absolute/path/Qwen3-0.6B-Q8_0.gguf
```

The importer and harness reject existing output destinations. The Rust worker's
full-value IPC is intended exclusively for the private harness; do not redirect
its output to a transcript.

Validation completed with 3,665 converter tests passed and 87 skipped, seven
experiment math/protocol tests passed, and 39 Rust scoring/protocol tests passed.
Python lint and formatting, the experiment Rust formatting check, and repository
policy checks passed. The optimized worker built and executed successfully.
The installed `rust-objcopy` emitted a nonfatal missing `libLLVM.dylib` warning.
Workspace-wide stable Rust formatting failed on existing vendor/workspace
formatting differences; those files were left untouched.

The [aggregate evidence](experiments/qwen3-gguf-2026-10-13/aggregate.json) records
the source revisions, model/build/tool hashes, distributions, repeatability,
reference comparisons, and qualification disposition. Earlier validation records
remain unchanged.

A subsequent [prefill optimization experiment](QWEN3_GGUF_PREFILL_OPTIMIZATION.md)
records new measurements and kernel changes against this preserved baseline.
