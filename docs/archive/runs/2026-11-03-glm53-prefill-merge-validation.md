# GLM-5.3 Bounded Metal Prefill Validation

**Date:** 2026-11-03

**Validation source:** `cbb66f481c75e16ed8c076290392bcad35df29be`

**Tree:** `be90407c3ffc3c065b2d8f808136397b451f7edd`

This record qualifies an opt-in GLM-5.3 execution profile and one bounded same-host prefill comparison. The public aggregate with full hashes and observed configuration flags is [here](../../benchmarks/data/glm53-prefill-merge-validation-2026-11-03.json).

## What changed

The opt-in wrapper selects the complete GLM profile and eight-token prefill chunks. Engine defaults remain off. Resident-bank admission accounts for the complete inventory, static views, and context-dependent state under 95% of Metal's recommended working-set size. The per-graph scratch cap is 512 MiB; greedy overlap allows two graphs under a 1 GiB combined cap. A context is limited to 4096 tokens. Unsupported complete-profile inventories or memory requirements fail closed. Pipeline capability failures after MMA admission surface as errors; this report does not claim that such failures fall back to non-MMA. Activations and accumulation remain F32; the verified weight artifact is UD-Q2_K_XL.

These are admission and dispatch boundaries. They do not establish that arbitrary long prompts fit a host's available memory. On the recorded 128 GiB host, the wrapper and artifact planning estimate was about 109 GB; this is not a guarantee of available memory on another host or with other applications running. The recipe is opt-in, and defaults are unchanged.

The native scratch profile for the short MMA-enabled smoke reported `Counter.max_bytes` of 337,641,472 bytes (322 MiB). This is an allocator-counter maximum, not total model memory or process peak RSS; that profiled smoke is not timing evidence.

## Source and artifacts

All engine, worker, scorer, benchmark, harness, parity, smoke, and performance receipts reported below bind to the source commit and tree above. Their source receipts report a clean tracked tree before and after the runs. The generator worker SHA-256 is `29163c3a12d88cc78003b252903c906c7408ee8b1817bc74d0735d6b5c5e769b`, the scorer worker SHA-256 is `b9293a544d492b4a79398b6e5870b762c71776cd5bac0f967df3dc0858d92ec5`, and the engine test binary SHA-256 is `f5fe83f16aeddc7a1510facbd58734b86901f0d3e0ebf1b0642ae5e2ae761df2`.

The benchmark protocol used the verified GLM-5.3-Flash UD-Q2_K_XL artifact on a Mac17,7 (`arm64`, 128 GiB physical memory, macOS 26.7.1). The pinned `native-source.json` provenance records weights from `unsloth/GLM-5.3-Flash-GGUF` revision `a38483c8cd5df544f53d70fb281afe97369d5ab6` and model metadata from `zai-org/GLM-5.3-Flash` revision `04c4e9e95c5da8862dced7e5056455116f83a7e0`. The source receipt SHA-256 is `7b8a61782577efbc30f6791f26e91d539a0934f14671f82d9548a313ec67903d`. The full shard identities and benchmark helper, wrapper, configuration, and harness hashes are in the aggregate JSON.

The incoming main revision `4545a409d32703bbf69ccc9be85d2d2f43aa5c4d` adds menu-bar README and status-color UI changes relative to the validation head's main ancestor. It changes no GLM engine, worker, benchmark, or harness source. The evidence in this record remains bound to `cbb66f48…`; exact checks for the final integrated head are pending and are not represented as passed here.

## Local and hosted checks

- Generator release build and Clippy with warnings denied passed; 112 tests passed and 2 were ignored.
- Scorer release build and Clippy passed; test targets passed 1, 25, and 25 tests.
- Engine build and tests passed; 761 tests passed and 78 were ignored. The focused Metal runtime suite passed 25 actual cases with source, binary, and fixtures unchanged.
- Formatting and policy checks passed on the validation source.
- Earlier supporting checks remain separately source-bound: the converter suite passed 5,839 tests on `c72f5a3417990ddb4425beee86f436d919ea63f4` (tree `518b41d4875776b86a8eaf32ecba54c605456257`); selected Python checks passed 100 tests on `850832f29dffba6fa7c7039c41ae9f8f3279bfe8` (tree `8bdb0fab9f475f60267cd693191d41b12013a420`). These are not cbb receipts.
- Hosted CI `41581861625`, Build `41581770120`, and Security `41581863934` completed successfully for the cbb validation head. Exact final-head hosted checks after the later UI integration remain pending. The Security result is a hosted check, not a code review.

## Scorer agreement

The candidate scorer compared 16 fixed single-token candidates across 12 contexts (192 points) against the sealed v46 reference worker (`52150ee9fabd300fc5b6bc70d13f7680b31158f3344055aac58c24ba6b8c4ea7`). The maximum absolute log-probability delta was `1.3879155950746735e-5` against a `0.001` threshold; RMS delta was `4.188137035436993e-6`. The 12 requests covered prefix lengths 96, 128, 256, and 512.

The post-request software probe observed positive Q5 and Q6 dispatch counts in each context: 9,724 Q5 and 10,268 Q6 calls total. These counters do not measure GPU utilization or answer quality. The corpus SHA-256 `0be89f00dd2df144bc175cfbb659450516ff0af7613042ba9b6f15de1e41362b` identifies the input bytes used with source revision cbb; it is not a hash claim about a later checkout.

## Generation smoke

The default profile, opt-in MMA-disabled recipe, and opt-in MMA-enabled recipe each passed three exact greedy replays (one warmup) at 14 input tokens, context 512, and 32 requested output tokens. Every replay produced token-ID hash `5377eeb8513c16603090026f7a432327b456842a96d54565044e24bdb5986e1c`. These short runs establish replay consistency only; they do not establish long-context stability or performance.

## Screened performance comparison

The comparison measured the entire prefill route, including chunk scheduling, with the same generator binary and model artifact. It used greedy generation at 96 input tokens, 8 output tokens, and context 104. The control ran first with `STARK_GLM53_PREFILL_MMA=0`, then the candidate with that flag set to `1`; the remaining observed configuration values, worker SHA-256, configuration SHA-256, weight receipt, and source pins matched. This was one sequential sweep, not a randomized or repeated independent study.

Each arm used one warmup and four accepted measured requests. Four additional attempts per arm were excluded by the host screen. Every accepted measured request passed the screen with zero pageouts and zero swapouts; warmups were not used for acceptance. All 18 completed outputs, including excluded attempts, had token-ID hash `8669bd40bd02bcc35680c255fc660be01a22a86ff01201463acfa7c457d64cad`.

| Metric | MMA disabled | MMA enabled | Change |
| --- | ---: | ---: | ---: |
| Median warm wall time | 2.970159 s | 2.564564 s | 13.655688% lower |
| Median warm prefill rate | 34.895805 tokens/s | 40.872903 tokens/s | 17.128413% higher |
| Median warm decode rate | 34.487968 tokens/s | 34.382681 tokens/s | 0.305284% lower |

This is a bounded observed comparison for the recorded host, artifact, context, and recipe. Four accepted measurements per arm support the reported comparison but do not establish a statistical or general throughput floor. The 96-token input is short; the result does not establish behavior for long prompts. No answer-quality or MLX comparison is claimed.

## Historical boundary

The November 2 v48 report remains historical and unchanged. It is bound to source HEAD `379e5c79d1cfae249e947740732b783b1a6fec82`, tracked diff and archived patch SHA-256 `3e8a7f2ed4c63c159657209f732398bcce9706180c0b5736a09f5e6633c15c0c`, and hybrid source bundle SHA-256 `0ba85f6a4aa181a39651c2fb83491bdef4d1f2c399a2cf818f1e7df5313eccb2`. Its v48 generator and scorer hashes are `41fd421767f14cf9fa5a6dd441eec73b5a2c0e4ec1eeba10e11951b6cc11cedf` and `49045d592af366a9af405cc3eddead7eb8596a107136954bcc1e5a55148e61fd`. Those measurements are not substituted for this cbb-bound comparison. See the [November 2 report](../../benchmarks/GLM53_BATCHED_PREFILL_2026_11_02.md) and its [historical aggregate](../../benchmarks/data/glm53-batched-prefill-2026-11-02.json).
