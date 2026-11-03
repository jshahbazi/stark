# OLMo 3 and Gemma 4 source fusion benchmark

Date: 2026-11-02. Native implementation: `5aeaab96cae9354ef95eda1cd070fb2c51da11fa`. Baseline: the preceding C3 engine at `598b3f6eaddf2b0b87a08032b92d2e26e4288358`.

Gemma's median paired complete-request work rate improves **2.57%** over C3, bringing it to **MLX parity** on the matched 512-input/128-return workload. Its measured paired rate is 0.44% above MLX, a small margin that does not establish a general engine lead. OLMo shows no material change versus C3 (paired rate -0.05%) and measures 2.31% above MLX on this workload.

This comparison supersedes the [2026-10-29 C3 measurements](OLMO3_GEMMA4_Q4_REPAIRED_2026_10_29.md) for these pinned Q4 checkpoints. Earlier source-bound validation and measurements remain preserved. The initial November campaigns were rejected by the background-model CPU guard; their [validation record and receipt hashes](../archive/runs/2026-11-02-olmo3-gemma4-source-fusions.md) remain separate. The reruns below followed the user closing the Stark desktop app and retain every paired block.

The [public aggregate](olmo3-gemma4-source-fusions-2026-11-02.json) contains all request timings, load timings, guard results, identities, diagnostic aggregates and qualification hashes. Its SHA-256 is `f2b5c592095bb97ca26f215c5a69d15aad4992cc3a9f8d481cd63de65caf85d6`.

## Accepted measurements

| Model | Previous C3 seconds | Source fusion seconds | MLX seconds | Paired gain over C3 | Paired native/MLX rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| OLMo 3 7B Instruct Q4 | 1.253494 | 1.251954 | 1.276458 | -0.05% | 1.0231 |
| Gemma 4 E2B Q4 | 0.624734 | 0.609411 | 0.611821 | +2.57% | 1.0044 |

Times are medians of four load-level medians. Each rate comparison is computed within its matching block, then summarized by the median of the four ratios. It need not equal a ratio of the displayed overall times; this accounts for OLMo's slightly lower displayed median time alongside its slightly negative paired gain. Gemma improves in all four native/C3 pairs, with rate gains from 2.31% to 2.90%. OLMo's results vary around parity.

Median paired request latency falls 2.51% for Gemma and rises 0.05% for OLMo. Combined-work rates are 1,050.19 native versus 1,046.06 MLX work tokens/s for Gemma and 511.21 versus 501.39 for OLMo. Work rate is **640 / complete-request seconds**, including input processing, generation, request setup and the final queued lookahead drain. These rates do not measure standalone decode throughput.

## Host, protocol and controls

The host is Apple M5 Max with 40 GPU cores and 128 GiB unified memory, running macOS 26.7.1 build 25G241. Frozen native release executables use Rust 1.98.1 and generation features. The unchanged reference environment uses MLX 0.32.3, MLX-LM `1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7` and Transformers 5.7.0. Its verified lock SHA-256 is `caa51ccf8c64acf6a850765121808930be0caf88075ef9a6095925953b62063b`.

Each family uses four independent loads per arm: previous C3, source fusions and stock MLX. Each load runs one warmup and three measured requests. Every request starts with fresh state, processes 512 input tokens, returns 128 greedy tokens, performs 128 decode forwards and drains the queued lookahead. Native context capacity is 1,024 tokens. Request timing excludes model loading and warmup; load timings are reported separately.

The four process orders remain baseline/candidate/MLX, candidate/MLX/baseline, MLX/baseline/candidate and baseline/MLX/candidate. GPU jobs are serialized with the OS file lock. The existing guards retain the 200% per-background-process CPU limit, 10% detected background-model-process CPU limit, sleep/clock checks and frozen executable, fixture, artifact and reference identities. All guards and identities pass. Across both families there are 24 process loads, 24 warmups and 72 measured requests. All 128-token stream hashes agree across every engine and load within each family.

Opt-in diagnostics reuse `hybrid_host_metrics.py` from Qwen/GLM revision `6f953e8ccc9934d89889a812482faa256fc5199f`. Identical polling applies to every arm; it does not change acceptance gates. Covered windows report nominal thermal state, zero pageouts/swapouts and zero detected background-model GPU counter deltas. Some other GPU counters advanced. IORegistry counter units are unverified, clients can appear/disappear between samples, and boundary intervals are excluded. These diagnostics cover whole engine-load/request windows and establish neither exclusive GPU ownership nor fixed clocks. GPU frequency was not measured.

## Implementation and correctness

Both families reuse each thread's four BF16-rounded RMS inputs through the unchanged source reduction and normalization. Gemma additionally fuses its post-attention residual addition with pre-FFN RMS normalization and its final PLE residual addition with layer scaling, removing two dispatches per layer. These host selections preserve the original BF16 sum boundaries and require the explicit source profile. Weights remain affine W4/group-64, activations and KV remain BF16, and source F32 accumulation boundaries are retained.

The source RMS row limit remains 4,096. Gemma residual fusions additionally require reference PLE; canonical BF16 and legacy PLE profiles retain their separate operations. Arithmetic licensing and the independently applied Qwen scheduling idea are recorded in [upstream provenance](../../provenance/UPSTREAM.md).

No engine source changed between the [fresh end-to-end qualification](../archive/runs/2026-11-02-olmo3-gemma4-source-fusions.md) and these timing reruns. Source-file and executable hashes were verified unchanged. The qualification includes 28/28 Q4 cases and greedy matches, 10 canonical BF16 text cases, three canonical BF16 image cases, text/image worker recovery, cold/cached image-client checks, 717 native tests and 486 CPU/grammar checks. Gemma remains exact in 14/14 source cases and OLMo in 13/14; its accepted 4,101-token discrepancy and numerical gates are unchanged.

These timings cover the named Q4 text checkpoints. Gemma image validation uses a separate BF16 artifact, with no image performance claim. This workload does not establish a global Mac-engine ranking or answer quality.

## Source and evidence identities

| Field | OLMo 3 | Gemma 4 E2B |
| --- | --- | --- |
| Source repository | `mlx-community/Olmo-3-7B-Instruct-4bit` | `mlx-community/gemma-4-e2b-it-4bit` |
| Source revision | `d732c91ae02e90cd5d86e810fbeb9741794b4dd9` | `238767527555cb75a05732a84dff5d6ba0dd6809` |
| Accepted campaign receipt SHA-256 | `11e7953b8597c285b4af898d2e94f43f31f3b479e52035907d9286ddb7fc9f04` | `58f33d0c32f1a2fbb1e37cc113f8e77878b40a0c0c8bcd5bda1147bb8a49fc0e` |
| Diagnostic receipt SHA-256 | `5e9a6cdb5ac7e173c02c4a98b0781f7540257bbf1a35bc0fac1a8a01156c31ff` | `dccb23509e99ed117ba79d0751f6580827e09dbfd0407c4c0c93682ff7599fa5` |

| Common field | SHA-256 |
| --- | --- |
| Source fusion profile executable | `48a4e660035152657e7a19a71696486c68da7f2852d917aa072bdc729079a7c7` |
| Previous C3 profile executable | `ca1ad5a405afb30c88a6a29b8cef34d213c11a039ae8c08267f880f969b35e51` |
| Complete engine source patch | `9718c93efb5c78a89abd5488031672db24324b00b177b940259b0f167c105c54` |
| Fresh qualification receipt | `28daaa71d6f80447c552cb8530f366fb229210dae7756e1e65ca1e2274bd28c9` |
| Rerun wrapper | `b792c2f2b485f95823ebc5db125dfdb1bb1cd1af89a967f3a6f3504984288eac` |
| GPU/thermal collector | `39ea2ebf7a0f27b9b92f7b807e53cd013ba51b20a32563f542075960589ec14f` |

Model-weight, configuration, conversion, fixture and output identities are retained in the aggregate. Raw inputs, generated content, model artifacts and executables remain private and ignored.
