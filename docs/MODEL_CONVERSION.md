# Model conversion prototype validation

Validation date: 2026-10-11. This evidence applies to the implementation and dependency hashes below. Native builds used Rust 1.98.1 and the optimized `stark-score` worker. Models and detailed benchmark artifacts remain private in ignored storage; this document publishes aggregates and hashes only.

Historical record: this document describes the prototype committed at `77e9a425`, including its Lalamo fallback and original timing/lock hashes. The current replacement policy removes that runtime fallback and expands independent conversion. See the [replacement plan](MODEL_CONVERSION_PLAN.md) and [first milestone evidence](MODEL_CONVERSION_MILESTONE_1.md) for current implementation and qualification. The results below remain unchanged and apply to their recorded source state.

## Implementation

The [conversion tool](../tools/converter/README.md) streams verified dense Qwen3 full-precision checkpoints into Stark's existing model format. Other registered models, quantized weights, unsupported metadata profiles, and dense dtype changes use the separately locked Lalamo backend. The fallback preserves upstream origin behavior and resolves cross-repository tokenizer and generation files from their own immutable revisions.

The streaming backend uses an 8 MiB copy workspace, writes fused projections consecutively, copies full-precision embeddings directly, and widens normalization scales to F32. Full checkpoint headers are validated before writing. Unexpected tensors and malformed inputs in the supported layout fail; they do not select fallback. Input file signatures are checked again after writing, including an omitted but verified identical tied output head. Config and weights share one conversion plan. Completed output is published atomically without replacing an existing directory.

## Benchmark

The source was public `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca`, converted to BF16. The local source manifest covers 1,517,510,512 bytes. Both outputs contain 226 tensors and a 1,192,276,800-byte safetensors file.

The host was an Apple silicon Mac, arm64, 18 physical/logical CPUs, 128 GiB RAM, Darwin 25.6.0, with managed Python 3.13.13. Each backend ran three times in a fresh process. The order alternated, source files were warmed before each run, and dependency setup and downloads were excluded.

| Metric | Streaming | Lalamo |
| --- | ---: | ---: |
| Median complete CLI time | 0.976 s | 1.783 s |
| Complete CLI time range | 0.955–0.996 s | 1.770–1.893 s |
| Median reported conversion phase | 0.477 s | 1.291 s |
| Median sampled process-tree RSS | 206.2 MiB | 1,981.8 MiB |

The final run measured **1.83×** complete CLI speedup, with **89.6% lower sampled RSS**. The reported conversion-phase ratio was 2.70×. Complete CLI time includes startup, conversion, output hashing, and receipt creation. The reported phase begins after common converter imports and includes isolated Lalamo startup for that backend; it is not a pure tensor-operation comparison. Earlier source-bound runs remain separate in private storage. These results bind the exact implementation and project configuration being committed, including interruption handling.

RSS was sampled every 10 ms and summed over launcher and descendants. Shared pages can be counted more than once and brief peaks can be missed. These are descriptive results for one small checkpoint on one host with warm filesystem caches. They do not establish the same ratios for larger models, cold disks, other architectures, or inference.

Every repetition passed exact comparison of all tensor names, shapes, dtypes, payload hashes, weight specifications, complete config JSON, and normalized tokenizer JSON. Lalamo's whole-file safetensors hashes varied with header serialization; all tensor payload hashes and logical metadata matched in every run.

The optimized native worker loaded both final BF16 outputs successfully and exited on stdin EOF. Each emitted readiness, exited with status zero, and recorded zero native scoring calls. This validates model construction and consumption of every tensor. No prompts, token scoring, generation, or private evaluation were run.

## Fallback and automated validation

The automated converter suite passed **122 tests**, including the opt-in metadata reference checks against pinned Lalamo. Tests cover all eight supported metadata profiles, deterministic dense weights, tied/untied embeddings, attention biases and sliding windows, cross-shard fusion, BF16 normalization widening, dtype routing, tiled transposition, malformed headers, source mutation, backend selection, safe publication, quiet diagnostics, independent parity checking, and revision-aware external metadata routing. Ruff lint and formatting passed. A locked optimized scoring build also passed; its existing debug-strip helper emitted a libLLVM warning without preventing the build or native loading. Both new lockfiles are included in the repository's existing OSV scan and gate inventories; repository policy checks passed. The existing pinned OSV Scanner 2.6.0 checked both new lockfiles with its cached PyPI database and reported 93 package rows with zero advisory findings; the report gate passed. Its optional VCS commit metadata is accepted only as a validated 40-character SHA, with 17 advisory-policy tests passing. The existing workflow passed actionlint.

Automatic fallback also converted public `Qwen/Qwen3-0.6B-MLX-4bit`, revision `173234aa840d113125e9f2271100ddbaf16c9620`. The result selected `backend=lalamo` and `fallback=true`, produced 452 tensors, and used tokenizer, tokenizer configuration, and generation metadata from the original Qwen3-0.6B repository. This exercises the cross-repository path that a simple local-origin replacement would lose. Its output weight file was 316,959,744 bytes. The optimized native worker loaded this output and exited successfully with zero native scoring calls. This single fallback smoke is not a quantized conversion performance benchmark.

Broader conversion coverage remains bounded by the pinned Lalamo registry, model access, and optional upstream dependencies. Current Stark runtime capabilities are a separate requirement: dense loading selects BF16 and MoE construction is unsupported. The prototype does not qualify every Lalamo model for native scoring.

## Source-bound hashes

| Evidence | SHA256 or revision |
| --- | --- |
| Lalamo source revision | `831accb8252a6b6606c378048caeb44178f1a050` |
| Stark Uzu base revision | `8d01581474760bd1fc7f4a547ff1d02f7dada437` |
| Converter sources and project configuration | `c4cb1990198c7947721f06c7dd0452d9a0c34c80d52a5c9125b8d824c1932ccd` |
| Streaming environment lockfile | `5eb7b6a53fb30826e196c2d7dae2f15abae52069871d1a3cb45373b0c5c7b2c9` |
| Lalamo environment lockfile | `c52ef4d012dc48f44831d4e496a0418ca673dc0b4edb9469ea23a4425a4e7f67` |
| Input source manifest | `81f1c7475176e5c4da140003dc9a418c7437bc69d6a7303c620ca5aab4dbb635` |
| Full benchmark aggregate report | `d9b80e48e496cbb3f4eda5607dc55aec53f400580c91bf447e77099f92849691` |
| Shared output tensor manifest | `9ac5f59ef0e581fe5d4ed79ff2ae29487a5e344d39877d944bb06068fc8af875` |
| Shared canonical config JSON | `8df51ff885bdefad6429b01d7b7563c225cd65696e5afbb7235d8a59818fdf2a` |
| Shared canonical tokenizer JSON | `31a65acea9fb889730123fce1471c7bf1d27b08459de049f3ee61246ac8fb61d` |
| Streaming safetensors file | `87c3344a80b7376392cc5c44d5369680b68936e6930bb8ac476ca65a136f64c7` |
| Quantized fallback source-revision manifest | `7d11b7fbacb6537b1913370d0fa7f55456d971a161c11cd660de70d4feb1350d` |
| Quantized fallback safetensors file | `eedff76a3d7466f2031dd4bae2a13138f025cb501c0316243ff31581d858805a` |

The benchmark rejects input or converter-source changes during measurement. New changes to converter source or dependencies require a separate validation run and new evidence hashes. The adapted upstream sources and preserved MIT notice are documented in the converter README and `LICENSE.lalamo`.
