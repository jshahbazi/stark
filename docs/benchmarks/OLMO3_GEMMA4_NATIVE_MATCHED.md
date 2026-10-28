# Matched native OLMo 3 and Gemma 4 text benchmark

Date: 2026-10-28. Implementation and benchmark harness revision: `0f3e855f69b130a3de922fc15fdea760b92d9c92`.

Status: serialized diagnostic measurements complete. This study compares Stark's direct Metal generation path with pinned stock MLX-LM on the same two affine-Q4 sources. Both exports pass bounded worker generation, scoring, replay and recovery checks, but neither passes every numerical comparison. These timings do not qualify Q4 accuracy. Published engine comparisons and source-licensing research remain in [Mac engine evidence](OLMO3_GEMMA4_MAC_ENGINES.md).

## Host, software and precision

| Component | Identity |
| --- | --- |
| Host | Apple M5 Max, 128 GiB unified memory, macOS 26.7.1 |
| Stark | Rust 1.98.1, release build, generation feature set; no scoring-only pruning or KV-only feature unification |
| Native harness | [dense_profile.rs](../../crates/stark-engine/examples/dense_profile.rs) |
| MLX harness | [dense_profile_mlx.py](../../scripts/dense_profile_mlx.py) |
| MLX-LM | `1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7`, MIT, Apple copyright |
| Reference environment | Python 3.13.13; MLX and MLX-Metal 0.32.3; Transformers 5.7.0; frozen private `uv.lock` |
| Weight format | Affine Q4, group size 64, no per-module override declaration; unchanged source snapshots for both engines |
| Native precision | BF16 activation path with the native kernels' declared accumulation and normalization policies; this is mixed-precision inference |

The snapshots are MLX-community quantizations of the two model families. Their packed Q4 source weights are shared across engines; native conversion changes tensor organization and artifact hashes. Stock MLX-LM retains the source's floating-parameter dtypes and its own operation boundaries. Native BF16 boundaries can produce different logits or greedy trajectories. Source-weight identity alone does not establish numerical parity. Preserve existing full-precision reference results and qualify this Q4 path separately against its matching reference.

## Source and fixture identities

| Identity | OLMo 3 7B Instruct | Gemma 4 E2B Instruct |
| --- | --- | --- |
| Source repository | `mlx-community/Olmo-3-7B-Instruct-4bit` | `mlx-community/gemma-4-e2b-it-4bit` |
| Source revision | `d732c91ae02e90cd5d86e810fbeb9741794b4dd9` | `238767527555cb75a05732a84dff5d6ba0dd6809` |
| Source `model.safetensors` bytes | 4,105,983,200 | 3,550,670,554 |
| Source weights SHA256 | `a97f02f033af038a222c4ae3cd22a3df580160d559dc985c6a6a2d512b5210cd` | `038e39a37a7667373d2c3991375446b10c96ae1d717a68674870343db376b76e` |
| Source config SHA256 | `92408f393c6c724d087c8a1ef913a507d1c4aff689e2019fc7577fcacecfdd64` | `6397cb6eca41b911d1dcab74e17941351057bd759284052a2331918ff6f9246c` |
| Private fixture SHA256 | `5cf5dfbeb5902ccc5bea12803ca625d2990d760d43eb82b9b9c4b7335fb72c52` | `676b328ec630e9a7bc65d12f90db06e25d35eb4278a33faffde9d09d31f07b65` |
| Input SHA256 | `383fbe095bec25537a3692c9800bfa1dfdde7ce854caa2c5ed05877be5f534bf` | `63eb18db375879d80163bc4592e18777f68214840d010aae2125d754e398f37b` |
| Native weights SHA256 | `80741d14a2d64a740c65c285e2dfca7783f59ce63fa1a2465c12ea31306c3db8` | `0118a58bcb1ec4a0bc97db53172581eef9e1762238e4b733972a29ec3ba2551b` |
| Native config SHA256 | `48432cc2edbd4b2c163c8bfe99b3c3db056e850530ea756cba242779e9a7323c` | `a1205ad1906124f45cbd084f95357f46202e6d9c851c2cc0639293e887c456e1` |
| Conversion receipt SHA256 | `511307b7431788c477a3b53afcadf9ca5b2b21caeb7376d0b885ea02a3bff00b` | `aa3dc2d2254762773f9903da3528ac03361ebccc179ff83aee9afebc79304e73` |

Input hashes use the ordered token IDs encoded as little-endian unsigned 32-bit integers. Fixture preparation repeats a private validation input and truncates it to exactly 512 tokens. The input bytes and generated content remain private. Each engine receives the identical input for its model.

The native harness checks the conversion receipt against converted weights/config hashes and the source snapshot's revision, component hashes and sizes. MLX independently rehashes the source files and checks its installed source revision before loading. The frozen reference lock SHA256 is `caa51ccf8c64acf6a850765121808930be0caf88075ef9a6095925953b62063b`.

## Work and timing protocol

1. Load one model per engine process. Record loading separately and exclude it from request timing. Reserve the GPU for one benchmark process at a time, with no competing model runs, builds, large file hashing or other heavy I/O during measured runs.
2. Run one warmup followed by three measured requests. Every request starts from fresh model/cache state, uses a 512-token input, returns 128 greedy tokens, ignores EOS, performs 128 decode forwards, and drains its final queued lookahead before the total timer stops. Final context length must equal 640 tokens.
3. Native timing starts before `create_empty_state`, reserves a 1,024-token context, and follows `LanguageModel::stream`. Dropping the stream drains the final forward. The harness rejects speculative models and verifies its prefill/decode/returned-token counters.
4. MLX timing starts before prompt/cache creation. Stock model-specific caches use their native allocation policy. The loop first evaluates a 511-token cache-only prefix, then processes the last input token and schedules greedy lookahead with asynchronous evaluation. It synchronizes the stream at the end and checks cache offsets. It does not compute unused log-probability vectors.
5. Record total request time, setup time, first-return time, remaining-return time and final-drain time separately. First-return timing includes lookahead scheduling in both implementations. Report the median of the three measured total times and `640 / median_total_seconds` as combined work tokens/s.

The primary rate includes input processing, generated tokens, request-state setup and the final lookahead drain. It is not a standalone decode-throughput measurement, so it cannot be compared directly with a published `tg128` decode rate. The matched counts do not force the engines to execute identical internal graphs: model-specific prefix pruning, cache allocation and kernel fusion remain part of their implementations. Native memory and MLX peak-allocation counters are recorded with their original meanings; they do not share an allocation-accounting definition.

Within-backend output-hash repeatability is a required stability check. Report cross-engine output-hash agreement separately. Agreement on this input does not establish general answer quality; divergence requires numerical analysis under the recorded precision policies before an exact-output claim.

## Measurements

| Model | Stark median request seconds | MLX median request seconds | Stark combined work tokens/s | MLX combined work tokens/s | Stark/MLX rate ratio | Cross-engine output hashes |
| --- | --- | --- | --- | --- | --- | --- |
| OLMo 3 7B Q4 | 1.532450 | 1.317913 | 417.63 | 485.62 | 0.8600 | Different |
| Gemma 4 E2B Q4 | 0.629455 | 0.614142 | 1,016.75 | 1,042.10 | 0.9757 | Different |

Stark's combined rate was 14.0% below MLX for OLMo and 2.4% below for Gemma on this workload. Three repetitions do not resolve small performance differences across devices or operating conditions. All four processes produced identical output hashes across their own warmup and measured requests. Cross-engine 128-token output hashes differ for both models; the forced-length timing workload does not establish exact-output parity.

The [aggregate dataset](OLMO3_GEMMA4_MATCHED_2026_10_28.json) contains every measured request, load timing, memory counter, component identity and output hash. Its SHA-256 is `9ca1b226a257fd2fbb398f2304396e89c5cd6858fa39137882319da312cea92c`. The native executable SHA-256 is `1ba42a051905b89f1058b450c4208a9b8fbd7e00436fe8f34a761b2bd196da24`; both MLX snapshots report BF16 floating parameters. Each inner timing receipt's `passed` means work-count and within-engine repeatability checks passed. The dataset separately records the failed numerical qualifications.

| Aggregate receipt | SHA-256 |
| --- | --- |
| OLMo Stark | `0e49081298da925e9e4b8eb3ba9dc2b82dadc4e5858974213b0a9ae27ff2b72e` |
| OLMo MLX | `d2f773c0c89619a6ab7a7d2d9e1f8af8ef691e3f46884ef35215f2565db65106` |
| Gemma Stark | `ca1af9fdfd3dcbf3529327c8a6c69d1e15416497333a82ce5b069bdd686e96c2` |
| Gemma MLX | `2bbaef41f3a96833b045a8abc6f71110388db3b7b5e2b533f5ecf38651176070` |

Earlier exploratory timings were excluded because they overlapped CPU checks or possible large-file hashing. The reported four processes ran sequentially after those activities finished. No production runtime code changed between them.

OLMo Q4 passed four of five numerical cases; the single-token case exceeded both the raw RMSE bound and its combined centered-error bound. Gemma Q4 passed three of five numerical cases against its separate MLX references under the unchanged Gemma BF16 gates. All ten next-token greedy comparisons and both models' two short worker generation cases matched their references. The longer benchmark trajectories diverged. Numerical-policy investigation is required before promoting either Q4 export to full qualification; see the [validation record](../archive/runs/2026-10-28-olmo3-gemma4-native.md).

## Published engines and techniques already used

The strongest controlled published Gemma 4 E2B comparison found used an M5 Pro 48 GB, nominal Q4 weights, five repetitions and 128 generated tokens. BaseRT v0.1.6 reported 149.8 decode tokens/s, MLX-LM 0.31.3 reported 146.9, and llama.cpp b9960 reported 103.3. BaseRT's pp512 result was 13,444 tokens/s against MLX's 9,203. These are publisher measurements on another chip and timing protocol. [Paper and tables](https://arxiv.org/html/2607.19438v1).

No comparable published multi-engine result for this exact OLMo 3 7B source was found. A first-person MLX report gives 41.2 decode tokens/s at 4-bit on an M4 Max 128 GB but omits prompt/output lengths and source revision. It is context, not a ranking baseline. [Report](https://www.linkedin.com/posts/michaelbaeriswyl_machinelearning-mlx-apple-activity-7412507946422558720-M1dT).

BaseRT's public repository at `a869c1cd86b6b34f8fd208d2c4d4ac3b2188aa08` licenses its converter, format, headers and bindings under Apache-2.0; its runtime and Metal kernels ship as proprietary binaries. The inspected v0.1.6 tree also lacks engine source. Its papers can inform independent implementation, but its kernels were unavailable for copying. [License boundary](https://github.com/basecompute/baseRT/blob/a869c1cd86b6b34f8fd208d2c4d4ac3b2188aa08/docs/reference/engine-releases.md).

Stark applies the useful architectural mechanisms: native packed-weight matvec execution, separate prefill/decode paths, asynchronous lookahead, model-specific ring/shared KV caches, OLMo full-width Q/K normalization and post-norm residuals, and Gemma proportional RoPE/PLE/shared-KV handling. Gemma's new image tower reuses the existing noncausal single-pass attention path with online F32 softmax, avoiding a quadratic attention-score allocation. The native text backend also contains online-softmax attention in [attention_two_pass.metal](../../crates/stark-engine/src/backends/metal/kernel/attention/attention_two_pass.metal). No BaseRT runtime or kernel source was copied.

The next optimization should follow the accepted matched measurements and kernel profiling. M5 tensor-assisted prefill, larger fused decode regions and Gemma assistant-model speculation require separate correctness and performance qualification. Neither this two-engine study nor the published evidence establishes a globally fastest Mac engine.

## Reproduction

Run these from the repository root with local source paths set to the exact snapshot revisions above. GPU commands must run sequentially in a quiet window. The private fixtures already bind the inputs and sources; do not regenerate them between engines.

```sh
cargo +1.98.1 build --release --locked -p stark-engine --example dense_profile
./target/release/examples/dense_profile runtime/olmo3-gemma4/olmo3-q4 runtime/olmo3-gemma4/olmo3-q4-benchmark-fixture.json
uv run --project runtime/olmo3-gemma4/reference-env --locked --python 3.13.13 python scripts/dense_profile_mlx.py run --source "$OLMO3_Q4_SOURCE" --fixture runtime/olmo3-gemma4/olmo3-q4-benchmark-fixture.json --output runtime/olmo3-gemma4/olmo3-q4-mlx-profile.json
./target/release/examples/dense_profile runtime/olmo3-gemma4/gemma4-e2b-q4 runtime/olmo3-gemma4/gemma4-q4-benchmark-fixture.json
uv run --project runtime/olmo3-gemma4/reference-env --locked --python 3.13.13 python scripts/dense_profile_mlx.py run --source "$GEMMA4_Q4_SOURCE" --fixture runtime/olmo3-gemma4/gemma4-q4-benchmark-fixture.json --output runtime/olmo3-gemma4/gemma4-q4-mlx-profile.json
```

The MLX output paths must not already exist. Both harnesses suppress model-framework diagnostics and exception content, and emit counts, timings, identities and hashes only. Keep model files, private inputs, raw runtime artifacts and generated distributions ignored.
