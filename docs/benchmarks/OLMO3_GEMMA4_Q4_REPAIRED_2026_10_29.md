# Repaired OLMo 3 and Gemma 4 Q4 benchmark

Date: 2026-10-29. Native implementation: `598b3f6eaddf2b0b87a08032b92d2e26e4288358`. Numerically repaired C2 baseline: `2e96ec8672398fae1887f6160cea5f598151a010`.

On the matched 512-input/128-return workload, Stark's median paired combined-work rate is **1.55% above MLX for OLMo** and **2.15% below MLX for Gemma**. Relative to C2, median paired request latency falls **19.46%** and **3.07%**, respectively. Both families retain matching 128-token stream hashes across all accepted native and MLX runs.

The [public aggregate](olmo3-gemma4-q4-repaired-2026-10-29.json) contains all load/request measurements, guards, identities and qualification hashes. Its SHA256 is `082113e5ed0f2608f938eb0999d7ff427ac49c9b8ee172bf4b1436b938d305d1`. The [2026-10-28 diagnostic comparison](OLMO3_GEMMA4_NATIVE_MATCHED.md) remains unchanged with its original numerical failures and timing results.

## Accepted measurements

| Model | C2 native request seconds | Repaired native request seconds | MLX request seconds | Paired repaired/MLX rate | Paired latency reduction from C2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| OLMo 3 7B Instruct Q4 | 1.567938 | 1.259285 | 1.277698 | 1.0155 | 19.46% |
| Gemma 4 E2B Q4 | 0.646835 | 0.627277 | 0.614090 | 0.9785 | 3.07% |

Request times are medians of the four load-level medians. Paired rates and latency reductions are computed within each matching block and then summarized by their median; they need not equal a ratio of the displayed overall medians. All four paired blocks are accepted for both families, with no best-run selection.

Combined work is **640 / complete-request seconds**, including input processing, generation, request setup and final lookahead drain. The median load-level rates are 508.26 native versus 500.93 MLX combined-work tokens/s for OLMo, and 1,020.28 versus 1,042.19 for Gemma. These are complete-request rates, not standalone decode throughput. The paired candidate/C2 rate ratios are 1.2416 and 1.0316.

## Host and method

The host is Apple M5 Max with 40 GPU cores and 128 GiB unified memory, running macOS 26.7.1 build 25G241. Native release builds use Rust 1.98.1 and the generation feature set. Stock MLX uses version 0.32.3, MLX-LM `1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7` and Transformers 5.7.0. The frozen reference lock SHA256 is `caa51ccf8c64acf6a850765121808930be0caf88075ef9a6095925953b62063b`.

Each family has four independent loads of each arm: C2 native, repaired native and stock MLX. Each load runs one warmup and three measured requests. Every request starts with fresh state, processes 512 input tokens, returns 128 greedy tokens, performs 128 decode forwards and drains the final queued lookahead; loading and warmup are excluded. Native context capacity is 1,024 tokens.

The four process orders are baseline/candidate/MLX, candidate/MLX/baseline, MLX/baseline/candidate and baseline/MLX/candidate. GPU jobs are serialized. Host guards limit external CPU activity to 200% and background model activity to 10%, reject sleep/clock discrepancies of two seconds or more, and require frozen source, executable, fixture and artifact identities. All guards and identity checks pass. Across both families this gives 24 process loads and 72 measured requests, plus 24 warmups.

The native [profile harness](../../crates/stark-engine/examples/dense_profile.rs) and [MLX harness](../../scripts/dense_profile_mlx.py) use the same request boundary and source-bound private fixtures. Inputs and generated content remain private; published records contain aggregate measurements and hashes.

## Production change and precision

Contiguous M=1 learned Q/K normalization now reuses the existing parallel source RMS kernel. OLMo normalizes each 4,096-element full-width Q or K projection as one row. Gemma batches eight query-head rows or one key-head row, each 256 or 512 elements. The host changes row geometry and buffer subranges while retaining the four-contiguous-value F32 source reduction, padded SIMD reduction, configured epsilon, precise reciprocal square root and BF16 normalization/gain materialization.

The path requires BF16, the explicit source reduction profile, `OnlyNormalization`, learned F32 scales, zero scale offset, and no bias or mean subtraction. Gemma's branch requires eight query heads and at most one KV head. Weightless V, multirow input, canonical profiles and other geometries retain their existing routing and validation. Stored Gemma gains use offset zero. The production build contains no experimental routing switch or timestamp instrumentation.

Weights remain affine W4/group-64; activations and KV remain BF16. Source BF16 affine sums, QMM partials, rotary, embedding and materialized-attention boundaries are retained, with F32 dot/statistics accumulation where specified. This remains native Rust/Metal execution. The MLX adaptation source pin is `64ea011cb65f14d9ce2737e60db9a4ae91ed7441`; licensing and adaptation scope are in [upstream provenance](../../provenance/UPSTREAM.md).

Private frequency-cache and direct-scatter-removal trials did not justify promotion. Pointer/dequantization alternatives also remained private after their numerical/edge or performance controls. Their evidence is preserved in the [precision investigation](../archive/runs/2026-10-28-olmo3-gemma4-q4-precision.md).

## Qualification

| Check | Final result |
| --- | --- |
| Unchanged Q4 numerical gates | 28/28 pass; all greedy choices agree |
| Full-vocabulary equality | OLMo 13/14 cases; Gemma 14/14 |
| Gemma cached trajectory | 4,902/4,902 component hashes equal across 129 forwards; public/manual/teacher-forced 128-token hashes agree |
| Physical native suite | 715 passed, 54 ignored, zero failures |
| CPU/grammar suite | 474 unit tests, 7 integration tests and 3 doctests passed; 21 ignored, zero failures |
| Canonical BF16 text | 5/5 cases per family; all ten native full-vocabulary hashes unchanged |
| Canonical HF BF16 images | 3/3 cases; complete metrics bytes unchanged |
| Workers and image client | Four generation and four scoring cases; replay/restart/cache-failure recovery; three image-worker cases and two cold/cached client checks pass |

The non-exact OLMo case has 4,101 tokens and passes its unchanged numerical bounds. Image qualification uses the separate official Gemma BF16 artifact; these timings cover Q4 text. Results apply to the pinned checkpoints, compiled binaries and this M5 Max workload. They establish neither a global Mac-engine ranking nor answer quality.

## Source and artifact identities

| Field | OLMo 3 7B Instruct | Gemma 4 E2B |
| --- | --- | --- |
| Source repository | `mlx-community/Olmo-3-7B-Instruct-4bit` | `mlx-community/gemma-4-e2b-it-4bit` |
| Source revision | `d732c91ae02e90cd5d86e810fbeb9741794b4dd9` | `238767527555cb75a05732a84dff5d6ba0dd6809` |
| Source weights digest | `004d3f52c66fe5485b42fb7ccc6b4632586c137617f138109d5edcad16193510` | `7522cb8396b675c572b2d30f9131984f917752513a902d1827a84128b021f1fd` |
| Native weights SHA256 | `86eca5f98623ae556c4af1e681529579c10fac9bc710801bf6d182679a265400` | `6b340c2c5de885e6a29fd72671cd89ab878d7d582863a6ef1775ee4da81038ea` |
| Native config SHA256 | `6c55239ab67ac5c3bb7bc6605d507123657c4187a1f69d105c1e7d2726af2dc1` | `db1850dce7370c6cb797f703e48609e4b112eebb8d18ff86ce0f1bb6628875e0` |
| Conversion receipt SHA256 | `c67580f7a4fc15c213b228ba30f8a8b4cde32e2f3882876593e4cdf65fc4e3c1` | `b248e76393ba22adba4877b300661a43f8560f7c63e5c5b8518aed3899289b62` |
| Private fixture SHA256 | `5cf5dfbeb5902ccc5bea12803ca625d2990d760d43eb82b9b9c4b7335fb72c52` | `676b328ec630e9a7bc65d12f90db06e25d35eb4278a33faffde9d09d31f07b65` |
| 128-token output SHA256 | `a71d52d60eff63c4060c047d2d0b756655facee0cc6892ec3cfaa6d3a9462561` | `03f17cff595a19aba064fb7ddf00547b52fd0cdf3c908b77db96f2727aba52a1` |
| Campaign receipt SHA256 | `99dfb0f37d615e085ae0db2271ad0453120148da9bc8a2937b9e11d72e44bdad` | `ab1d0ceb5997ab0479da20763d5bd62ac0d184c3625c032d65f30c0afb189919` |

| Build/harness field | SHA256 |
| --- | --- |
| Repaired profile executable | `ca1ad5a405afb30c88a6a29b8cef34d213c11a039ae8c08267f880f969b35e51` |
| C2 profile executable | `90daa697db02c3ec7077e2e00d6361538ff4df6dd34d85c52a07e5b9e1ae05ca` |
| Native profile harness | `2db59539c08bc5c6d9a21cbcae6abdc0862efb04e4fc02f53c4c42718ba61c1b` |
| MLX profile harness | `6b3b779314de7b84dc8e41fa6e615a3186b6bfb8bcfa3b9e97ee04ab5bf35a3a` |
| Parity executable | `be192964f8765cdca31711397140c760651b569789f2b2929c9d71e6fd6a9c2c` |
| Unit executable | `503475f5e9698dffdec7c86afcf8ce012246a643d63eff3dc39ad461d3727b67` |
| Numerical/canonical qualification receipt | `b4dcaa80a2d5cfb65a69296a5a4baaf90961a233751f30a481cb0ee17ee8a435` |
| Native/CPU/worker qualification receipt | `6ce31d1067c585577d9c711618a2ef2c00479743cf4cae3d0fd88aaa00508750` |

The aggregate additionally binds source-config/input hashes, every measured request and all build/campaign receipts. The [support guide](../OLMO3_GEMMA4.md) records conversion and runtime scope; external published measurements and their source-licensing limits remain in [Mac engine evidence](OLMO3_GEMMA4_MAC_ENGINES.md).
