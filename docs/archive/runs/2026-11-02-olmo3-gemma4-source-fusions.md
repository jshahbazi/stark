# OLMo 3 and Gemma 4 source fusion validation, 2026-11-02

The source RMS reuse and Gemma PLE residual fusions on `codex/native-olmo3-gemma4` pass fresh end-to-end qualification. This change is based on `3394cbe72fed7450c531d33cc333771ad1104cba`. The implementation and frozen builds were prepared in the [2026-10-29 checkpoint](2026-10-29-gemma4-source-residual-fusions.md); their source and executable hashes were verified unchanged before and after the new checks.

Performance follow-up: after the user closed the Stark desktop app, the [separate source fusion comparison](../../benchmarks/OLMO3_GEMMA4_SOURCE_FUSIONS_2026_11_02.md) passed all four paired blocks for both models with these unchanged executables. It records Gemma's 2.57% gain over C3 and MLX parity, and OLMo's unchanged C3 performance. The timing section below preserves the earlier rejected campaigns and their original qualification status.

## Changes

Source RMS normalization on both families retains each thread's four BF16-rounded values through the existing reduction, then reuses them during normalization. This removes a second buffer read without changing the reduction tree or materialization boundaries. The enforced source row limit remains 4,096 elements.

Gemma additionally combines its post-attention residual addition with pre-FFN RMS normalization, and its final PLE residual addition with layer scaling. Each sum still rounds to BF16 at the original boundary. These two fusions remove two dispatches per layer on the qualified 35-layer E2B source profile. They require BF16, reference PLE and explicit source RMS metadata; canonical BF16 and legacy PLE profiles retain their separate operations.

The Qwen scheduling idea is thread-local reuse of already-rounded values. No hybrid kernels or Lily arithmetic were copied. Arithmetic attribution remains in [upstream provenance](../../../provenance/UPSTREAM.md).

## Fresh qualification

The same pinned Q4 sources, canonical BF16 text artifacts, separate BF16 vision artifact, fixtures and numerical gates are used. Reference captures and conversions were not regenerated. GPU checks are serialized with the private OS file lock.

| Check | Result |
| --- | --- |
| OLMo Q4 | 14/14 pass, 14 greedy matches |
| Gemma Q4 | 14/14 pass, 14 greedy matches |
| Native Q4 full-vocabulary output hashes | All 28 unchanged from the prepared candidate's prior qualification |
| Canonical BF16 text | 5/5 per family; all 10 native output hashes unchanged |
| Canonical Gemma BF16 images | 3/3; complete aggregate metrics bytes unchanged |
| Text generation/scoring workers | Both families: 2 generation and 2 scoring cases; restart, replay and recovery pass |
| Gemma image worker | 3/3; restart/replay, malformed-image recovery, counterfactual and staging cleanup pass |
| Gemma image client | 2/2 cold/cached checks; staging cleanup passes |
| Native CPU/Metal suite | 717 passed, zero failed, 54 ignored |
| CPU/grammar suites | 476 unit, 7 integration and 3 doctests passed; zero failed, 21 ignored |

Gemma retains exact agreement with all 14 Q4 source references. OLMo retains 13/14 exact cases and the previously accepted 4,101-token discrepancy under the unchanged gates. The earlier [precision record](2026-10-28-olmo3-gemma4-q4-precision.md) preserves that limitation. These results qualify the named checkpoints and fixtures, including Gemma's separate BF16 image artifact; they do not qualify every structurally supported variant or image performance.

Independent static review found no actionable correctness issues. Kernel controls include BF16 rounding-sensitive inputs, fused/separate comparisons, tails, broadcasting, in-place operation and buffer sentinels. Exhaustive nonfinite and subnormal equivalence is not established.

## Source and evidence identities

Private evidence is under the ignored `runtime/olmo3-gemma4/q4-repair/easy-improvements-20261102/` directory. Only aggregates and hashes are published. The earlier build record retains all five frozen executable hashes and original toolchain identities: locked Rust 1.98.1 examples/workers, optimized scoring and pinned nightly native tests.

| Evidence | SHA-256 |
| --- | --- |
| Complete engine source patch, including added files | `9718c93efb5c78a89abd5488031672db24324b00b177b940259b0f167c105c54` |
| New qualification wrapper | `845d70319b6a3595da4cdc5609f4cfa5e2dcbffeb27bf5b35143dfd11f52cfb1` |
| `qualification/qualification-receipt.json` | `28daaa71d6f80447c552cb8530f366fb229210dae7756e1e65ca1e2274bd28c9` |
| `qualification/native-full.log` | `cb0c81a2a7f34adf3d6d50881ec94f633ad916a20d7cf3d0864221016e8a3569` |
| `qualification/cpu-grammar.log` | `78589cbd2b7f83a79ccfc44b227c572ad32d77ccdb807db451668ca4be329ec4` |
| `qualification/olmo3-workers.log` | `b76fb35b44a5f045c544c8cd06aef259c6507ea8b9f746f7dcbce1498b08643c` |
| `qualification/gemma4-workers.log` | `704ebceb4ccd7863e7176d9b9c5ff31dfa9235d1078decda40fd10e6c8b3b85d` |
| `qualification/image-workers.log` | `f6bbdbff3cd935935ab010ef5c0705637880f826b2c2e627495a501b1ddc1945` |
| `qualification/image-client.log` | `55f2fe0a0a6e7411600f6bcc32f4807276736792225903ac1676fe590205a8e6` |
| Canonical image aggregate metrics | `64cc154be30bb3984f41f26b932c421d237f996de67645596c109840093e00a1` |

## Timing qualification

The fresh campaigns for both families use the existing balanced baseline/candidate/MLX order, four independent loads per arm and one warmup plus three measured requests per load. Each request includes 512 input tokens and 128 greedy decode forwards/returned tokens. Across 24 process loads and 72 measured requests, all fixture, executable and artifact identity checks pass, and all completion hashes agree within each family. The unchanged host guard rejects both campaigns because background processes classified as models exceed its 10% CPU limit. These runs support no qualified speed claim. No thresholds were relaxed and no best-run subset was promoted.

Opt-in aggregate diagnostics use the Qwen branch's `hybrid_host_metrics.py` from revision `6f953e8ccc9934d89889a812482faa256fc5199f`. The private wrapper applies identical polling to all arms, records whole-load/request windows, and leaves acceptance gates unchanged. GPU-client counters have unverified units; appearing/disappearing clients and sample boundaries limit coverage. Thermal state was nominal throughout the covered Gemma windows. These observations do not prove exclusive GPU ownership or stable clock frequency.

At this checkpoint, the [2026-10-29 repaired comparison](../../benchmarks/OLMO3_GEMMA4_Q4_REPAIRED_2026_10_29.md) remained the last qualified speed result for its frozen C3 build. These rejected campaigns established no qualified throughput result or closure of the remaining Gemma/MLX gap.

| Private timing evidence | SHA-256 |
| --- | --- |
| Campaign wrapper | `880ada6bcebb51089797776a1da6ea585608799bc182d8cc78d6e9eccf4dcef0` |
| GPU/thermal diagnostic collector | `39ea2ebf7a0f27b9b92f7b807e53cd013ba51b20a32563f542075960589ec14f` |
| Rejected Gemma comparison receipt | `e7a325e7ec91215b25e1ebda0f5261d256fa95e276172f070c18424dc5635878` |
| Gemma diagnostic receipt | `4a6b75cb0db4f39fc0b07c43d8a6ffc1eaeff6cf7200baa76b85b0e4e89d8e32` |
| Rejected OLMo comparison receipt | `0e2de4440a529a3fd7bf8b50919b6d0f78be9d32b7c1927c0ba99f1efffd4c75` |
| OLMo diagnostic receipt | `84925025ef066bc19b9438173e362fba6d3a222df264e12c7cb14a86a937f6ba` |
