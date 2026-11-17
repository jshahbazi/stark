# Flash Next grid on fresh 8K, 12K and 32K coding prompts

**20261117-FLASHSPEED-07 · November 17, 2026 · Cold-prefill control**

The matched [Pi coding-agent study](2026-11-17-flashnext-coding-agent-grid.md) found a warm-prefix advantage for 2,048-token chunks. Its matched fresh prompts topped out at 2,205 tokens. This test measures the cold contribution of the two physical grids with MTP on at greater input lengths.

## Prespecified comparison

Use three coding instructions paired with real source excerpts from pinned Rust generation, Python Desktop and Lily prefill files. Render each with the checkpoint's normal chat template and tune the source cutoff to reach exactly 8,192, 12,288 and 32,768 tokens. The fixture builder checks every source and tokenizer hash. All launches see the same nine requests. Use 40,960 context tokens, 96 maximum output tokens, two exact drafts, temperature 0.6, top-k 20, top-p 0.95 and seed 42. Keep the Lily Q4-XL artifact, loaded vision tower and worker binary fixed.

Run 4,096 → 2,048 → 2,048 → 4,096, with one worker resident. Warm the worker and reserve the largest request shape with one excluded 32K request per launch. Send every measured request without a cache scope and require zero reused tokens. Send the same rotating sequence in the first pair and reverse the sequence in the second pair. Match the three prompts within each length and adjacent pair, giving six paired observations at each input length. Preserve first-sample, decode, wall and host paging observations separately.

Use native time to first token as the primary cold-prefill measure. Compare decode and total wall only on matching input, output, finish, sample counts and MTP work. The prior promotion rule permits at most a 5% cold-prefill regression under matched work. Read each length and both orders against that limit. Treat material paging or an order-dependent marginal result as a limit on the inference. A cold pass addresses the fresh-prompt criterion. The prior Pi study accepted two of eight full pairs against a requirement of six, and the artifact-scoped cancellation and image lifecycle checks also belong to any eventual default change.

## Result

**Keep 4,096 as the default.** The 2,048 grid missed the 5% cold limit at every length in both launch orders. The previous Pi result still supports an opt-in 2,048 grid for repeated coding-agent prefixes.

| Fresh prompt | 4,096 median first token | 2,048 median first token | Paired change with 2,048 |
| ---: | ---: | ---: | ---: |
| 8,192 tokens | 4.38 s | 4.89 s | +12.0% |
| 12,288 tokens | 6.40 s | 7.36 s | +14.1% |
| 32,768 tokens | 17.36 s | 20.00 s | +15.9% |

The change column is the median of six within-pair ratios at each length. The 4,096 grid was faster on all 18 first-token and full-request comparisons. First-token penalties by length were +11.6%, +13.5% and +8.9% when 4,096 ran first, and +12.5%, +14.8% and +16.1% when 2,048 ran first. Seventeen of 18 first-token differences exceeded 5%. Median paired full-request penalties were 9.0%, 11.9% and 15.2%. Median decode changes ranged from −1.8% to −0.3%, with mixed directions within each length.

All 36 measured requests reported zero reused tokens, emitted 96 tokens and performed positive two-draft work. Each prompt's input and emitted-token digests, text digest, finish reason, sample count and exact verifier counters agreed across both grids and both launch orders. Four 32K warmups were recorded separately. The checkpoint's artifact identity check passed after the run.

The worker was the sealed Rust/Metal release from the [earlier experiments](2026-11-17-flashnext-speed-experiments.md), SHA-256 `bdf851bedf895ab09859cbdc40cab583a9976ddaf06f01d98ad3872dcd8f7dbe`, with the pinned Lily Q4-XL revision `0f94842e3682b49acace4daeb4cf2e65a6f94f83`. The M5 Max host ran on AC with 128 GiB and macOS 26.7.2. Measured request windows included 1,653 system-wide pageouts with 4,096 and 587 with 2,048, with zero swapouts in either arm. System paging provides context and leaves dynamic GPU clocks unmeasured. The three fixtures at each length share their two launch pairs. These observations characterize the complete physical-grid change, including its grouped-MoE tile eligibility, at aligned prompt lengths.

After the run, the installed Lily worker reached Ready. Authenticated health, discovery and a local MTP chat completion each returned HTTP 200. [Runner](../archive/experiments/2026-11-17-flashnext-speed/fresh_coding.py) · [Aggregate results and fixture identities](../benchmarks/data/flashnext-fresh-coding-grid-2026-11-17.json)
