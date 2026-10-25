# Independent conversion milestone 1

Date: 2026-10-11. Independent review accepts the scoped conversion/load implementation. Raw conversion latency remains an open optimization goal; this milestone does not replace the entire Lalamo registry.

## Delivered scope

The repository now owns independent Qwen3, Qwen2/Qwen2.5-Coder, selected Llama 3.x, and dense Mistral conversion. The Lalamo runner, dependency project/lock, automatic fallback, explicit backend, and fallback tests were removed. Runtime conversion has no Lalamo, JAX, PyTorch, or Transformers dependency. Lalamo was used externally as a pinned reference during qualification.

The catalog accounts for all 110 pinned built-ins: 107 language models, two classifiers, and one TTS model; all origins are Hugging Face, with 109 safetensors containers and one PyTorch container. Nineteen built-in profiles are implemented, plus three separately counted Stark extensions. Four exact BF16 source/output tuples have stored load-only qualifications. Other profile sizes/configs remain unqualified; 91 built-in entries still need independent adapters.

Contracts separate source validation, conversion coverage, native loading, and execution. Source resolution pins all primary/external repositories and hashes every selected consumed file. Private 0600 manifests support offline replay. Complete layout checks, selected-file binding, full final content verification, recursive tooling hashes, owner-only staging, sanitized errors, and atomic no-replace publication prevent incomplete output publication.

After resolver/replay content verification, planning repeats selection and signature checks without repeating a payload hash. Final publication still rehashes all inputs. Sequential output hashing covers the exact prefix, padded header, and payload while writing; seeking/transpose uses a completed-file hash. Successful publication alone reports complete source validation. Mixed-dtype inputs within one fused matrix/bias group are explicitly unsupported because reference promotion can alter signaling NaN payloads.

## Qualified tuples

Every tuple below passed independent exact tensor names/shapes/dtypes/payloads, tensor spec metadata, complete root/nested config, generation/token-codec metadata, and normalized tokenizer comparison against the pinned reference. Stark readiness followed by EOF loaded every artifact successfully with zero native scoring calls.

| Public checkpoint | Immutable source revision | Tensors |
| --- | --- | ---: |
| Qwen/Qwen3-0.6B | `c1899de289a04d12100db370d81485cdf75e47ca` | 226 |
| Qwen/Qwen2.5-Coder-0.5B-Instruct | `ea3f2471cf1b1f0db85067f1ef93848e38e88c25` | 170 |
| unsloth/Llama-3.2-1B-Instruct | `5a8abab4a5d6f164389b1079fb721cfab8d7126c` | 98 |
| mistralai/Mistral-7B-Instruct-v0.3 | `c170c708c41dac9275d15a8fff4eca08d52bab71` | 195 |

Llama qualification uses the explicit ungated Unsloth extension. No access or qualification claim is made for the official Meta profiles. The Mistral extension uses the pinned HFMistralConfig with an explicit independently constructed standard LanguageModelSpec; it is not counted as a Lalamo built-in registration.

Six sharded synthetic structural variants also passed complete output comparison: Qwen2 early-layer sliding windows; unscaled Llama 3; Llama 3.1 factor-8 scaling; tied Llama 3.2 factor-32 scaling; and full/sliding Mistral. Frozen reference config fixtures and IEEE BF16/F16/F32 edge-case hashes cover rounding, overflow, subnormals, signed zero, infinities, and NaN payloads. Uniform-dtype dense casts are implemented for new adapters; Qwen3 remains restricted to matching source/target weight dtype.

Stored [native qualifications](../../../tools/converter/stark_convert/native_qualifications.json) match catalog identity plus source-manifest, weight-file, config, and tokenizer SHA-256. `--require-native-support` requires an exact stored match before publication. Each receipt names the tested worker/source/dependency/toolchain hashes, Rust 1.98.1, release profile, and default plus optimized features. This supports the named build and artifact only. It is not a guarantee for arbitrary current/future binaries. F16/F32 stay runtime-blocked; execution remains unverified.

## Inference preservation

This patch changes no Rust, Metal, Cargo dependency, scoring path, runtime model format, or inference behavior. Existing converted Jev/JevK5 models retain their existing execution path. No Jev/JevK5 conversion, inference-latency, or judge-quality experiment was performed. New fine-tunes require a compatible supported base profile; quantized and hybrid conversion still need later adapters.

## Benchmark protocol and results

Before timing, the protocol fixed three repeats, warm source cache, alternating backend order, exact parity, an 8 MiB transformation-workspace limit, and observed streaming tree RSS below 1 GiB. Lower complete CLI time and at least 50% lower median RSS were performance hypotheses. Every hard acceptance check and memory hypothesis passed. The raw latency hypothesis failed for every checkpoint.

Runs used fresh direct Python CLI processes serially, without concurrent model execution. Setup/downloads, cache warming, and independent parity checks were outside timed intervals. Both produce the same runtime artifact. Stark time includes offline full source verification, complete layout/metadata checks, output hashing, final source verification, and qualification receipts. The external reference performs weaker source-verification/receipt work, so the time ratio measures these complete commands rather than isolated equivalent transforms. RSS samples occur every 10 ms and can miss short peaks or count shared pages multiple times.

| Checkpoint | Stark median (range), seconds | External reference median (range), seconds | Median tree RSS, Stark / reference MiB | RSS reduction |
| --- | ---: | ---: | ---: | ---: |
| Qwen/Qwen3-0.6B | 1.895 (1.878–1.904) | 1.389 (1.245–1.602) | 187.4 / 1887.7 | 90.07% |
| Qwen/Qwen2.5-Coder-0.5B-Instruct | 1.250 (1.235–1.270) | 1.193 (1.186–1.223) | 123.2 / 1555.1 | 92.08% |
| unsloth/Llama-3.2-1B-Instruct | 2.997 (2.973–3.003) | 2.006 (1.662–2.242) | 263.9 / 3393.5 | 92.22% |
| mistralai/Mistral-7B-Instruct-v0.3 | 15.435 (15.368–15.666) | 6.568 (6.524–7.351) | 97.0 / 14423.7 | 99.33% |

The first complete benchmark preceded the repeated-hash/output-hash optimization and is retained separately. Its source hash was `6088a3ed789c9bbd655d1bebede1576813f478b133837400f28ac557d2fd1b4b`. The optimization kept output hashes and exact parity unchanged; it reduced measured Stark median time by the percentages below. These sequential phases were not randomized together, so improvements are descriptive.

| Checkpoint | Initial Stark median, seconds | Final Stark median, seconds | Reduction |
| --- | ---: | ---: | ---: |
| Qwen/Qwen3-0.6B | 2.437 | 1.895 | 22.2% |
| Qwen/Qwen2.5-Coder-0.5B-Instruct | 1.671 | 1.250 | 25.2% |
| unsloth/Llama-3.2-1B-Instruct | 3.702 | 2.997 | 19.0% |
| mistralai/Mistral-7B-Instruct-v0.3 | 24.080 | 15.435 | 35.9% |

Host: Darwin 25.6.0, arm64, 18 physical/logical CPUs, 128 GiB RAM; managed Python 3.13.13. Converter lock resolves NumPy 2.5.3, ml-dtypes 0.6.0, tokenizers 0.23.3, huggingface-hub 2.2.0, and psutil 7.2.2. Reference uses JAX CPU. The prototype’s historical 1.83× ratio is preserved in [its original record](MODEL_CONVERSION.md) and does not describe this implementation.

The four-model comparison above is bound to converter source `889ab888e44acff52257dfff85219514ec2ee2b8fb81d971fe33b688bcd4d785`. After that measurement, only the benchmark harness changed to drain large receipts concurrently and reap subprocess groups/readers on interruption. CLI, adapter, metadata, and writer conversion behavior did not change. The final source hash is `f794a77cee3c8339d7806bd070a0a18389e5e3b973a4b42f7a37225d75bcaa96`.

A separate final-source official Qwen3 harness run compared frozen independent artifacts in three fresh processes: median complete CLI 1.980 s, range 1.932–2.018 s, median sampled tree RSS 206.8 MiB. All tensor/spec/config/tokenizer checks passed. Its aggregate SHA-256 is `7543f2afe85239a53955732d5df0b44d6ef9ba12619dbea0b0506053e0c569e8`. This run includes the uv wrapper and supplies no live reference timing or speed ratio. An earlier final-source harness smoke was run alongside metadata checks and its timing was excluded; its artifact/parity result remains retained separately.

## Source-bound evidence

| Input/evidence | SHA-256 |
| --- | --- |
| final converter_sources_sha256 | `f794a77cee3c8339d7806bd070a0a18389e5e3b973a4b42f7a37225d75bcaa96` |
| four-model measured converter_sources_sha256 | `889ab888e44acff52257dfff85219514ec2ee2b8fb81d971fe33b688bcd4d785` |
| converter_lock_sha256 | `2030e19c3997c05b92df598f4d0d05eb4a88d409189d5f87ff03c96710f6be32` |
| catalog | `dd2d86200556c8f020334fdd42625577fc87b4d0acf3d22280b0b70460ddf670` |
| native qualification catalog | `393c39e7b68962b459d805579a65623da61e0928b4f0cda6c0fc8de39063f1bd` |
| four-model benchmark aggregate (889 source) | `dd6eff4887c55e3dcb7e84a3adfc1b8ed559832558cf8cdeea7538e02d155071` |
| final-source isolated Qwen3 benchmark aggregate | `7543f2afe85239a53955732d5df0b44d6ef9ba12619dbea0b0506053e0c569e8` |
| initial benchmark aggregate | `7fd5f019aa982a64e024fdc55239a8402aaa8bc79e50f477eeb89371d40571a3` |
| structural variant aggregate | `4a5cdd1fe8b3dcf5e3d6f7b406b8244b3d2d7c787bdd261a5a6eaac31f4f4592` |
| external benchmark coordinator | `33227c622be3e5cade24ce7c4841c45f7c073a2a31da6ff730c210e2720fc31c` |
| external reference runner | `c76aee04ea05ba3357a66c01be83b96ddaa4bcd149eaf88380bb346801aaefe1` |
| converter OSV aggregate | `7eb73e29ba2f796f8326b40d14f37e52c2b67d4dfbf8f6a9b69fcdb61ec27faf` |
| worker_sha256 | `61bb81a83fa368b7c8f647469ac468048cf6868d973ed74e7d9d62633bfba96d` |
| native_source_manifest_sha256 | `0cc926bcf73db3921eab03f127185626b7f4311b73c62dad79b407c642d0c9af` |
| cargo_lock_sha256 | `65444f3e0e0707c43521539e43c95365cfc6a51ec6f9e90180bc43bc8fb4603e` |
| rust_toolchain_sha256 | `72da9f9e503464ba7393560f6f51a69fb06be72633d52ae7725fa0ba59dfda22` |

Permitted private aggregate evidence and operational artifacts are under ignored `runtime/converter-milestone-1`; source manifests contain private local cache paths. Public evidence contains only identities, revisions, aggregate outcomes, and hashes. No prompts, responses, token vectors, decision probabilities, credentials, exception dumps, or raw native execution captures were retained.

## Validation and disposition

- Final converter suite: 418 passed, 18 unsupplied private-reference profiles skipped. This includes 414 model-free tests and four independently supplied real-reference metadata checks.
- Ruff lint/format and converter lock checks passed. The existing quality workflow now runs offline model-free converter tests, lint, format, and lock checks; inherited release/signing/cloud infrastructure remains disabled.
- Repository policy checks, 17 advisory-report gate tests, and actionlint with the existing disabled-job exception passed.
- Cached offline OSV scan of the changed converter lock found no known advisories in its database snapshot; this is not a fresh online advisory-database claim.
- Rust 1.98.1 locked release scoring build succeeded; four BF16 readiness/EOF checks passed using the named worker. The build reused current artifacts and emitted the existing debug-info stripping warning; executable identity is recorded above.
- Independent [technical review](MODEL_CONVERSION_MILESTONE_1_REVIEW.md) resolves source-selection/root-binding, recursive exporter/native-template/vendor provenance, staging cleanup, exact comparator validation, qualification mutation, mixed-dtype promotion, hash-path findings, and large-receipt benchmark backpressure/reader cleanup.

G0 inventory/contracts and G1 independent integration are accepted for this scoped milestone. Selected G2 conversion, artifact/load qualification, transformation-memory, and observed RSS checks pass. The raw conversion latency objective remains open. General execution qualification, broader registry replacement, compressed formats, additional dense/hybrid/state-space/classifier/TTS adapters, strict process-memory enforcement, and public distribution remain future work. Further latency work must preserve full source integrity and exact output parity rather than remove checks to improve a ratio.

## Reference dependency scope

The pinned reference base uses JAX/Equinox, cattrs, tokenizers/Jinja, Hugging Face access, and general utilities. The 109 safetensors language/classifier rows use that base import path. The FishAudio TTS PyTorch container needs lazy Torch loading and may require FishAudio’s torch/transformers/tiktoken/blobfile extras; those rows remain unsupported here. Audio playback, server, CUDA/TPU, and Darwin Cartesia/MLX extras are distinct reference runtime/development dependencies and do not become Stark conversion dependencies. No built-in NeMo origin is registered at this pin; archive-origin support and its lazy Torch path remain later scope. Access and unimplemented-row optional execution dependencies are unassessed until their corresponding adapter work.

Pinned reference: `831accb8252a6b6606c378048caeb44178f1a050`; preserved [MIT license](../../../tools/converter/LICENSE.lalamo), Copyright (c) 2025 Mirai Tech Inc. Uzu attribution remains `8d01581474760bd1fc7f4a547ff1d02f7dada437`. The [replacement plan](MODEL_CONVERSION_PLAN.md) supersedes the historical fallback policy reviewed at `e51ed763`.
