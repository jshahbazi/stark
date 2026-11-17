# Flash Next grid on fresh 8K, 12K and 32K coding prompts

**20261117-FLASHSPEED-07 · November 17, 2026 · Cold-prefill control**

The matched [Pi coding-agent study](2026-11-17-flashnext-coding-agent-grid.md) found a warm-prefix advantage for 2,048-token chunks. Its matched fresh prompts topped out at 2,205 tokens. This test measures the cold contribution of the two physical grids with MTP on at greater input lengths.

## Prespecified comparison

Use three coding instructions paired with real source excerpts from pinned Rust generation, Python Desktop and Lily prefill files. Render each with the checkpoint's normal chat template and tune the source cutoff to reach exactly 8,192, 12,288 and 32,768 tokens. The fixture builder checks every source and tokenizer hash. All releases see the same nine requests. Use 40,960 context tokens, 96 maximum output tokens, two exact drafts, temperature 0.6, top-k 20, top-p 0.95 and seed 42. Keep the Lily Q4-XL artifact, loaded vision tower and worker binary fixed.

Run 4,096 → 2,048 → 2,048 → 4,096, with one worker resident. Warm the worker and reserve the largest request shape with one excluded 32K request per launch. Send every measured request without a cache scope and require zero reused tokens. Send the same rotating sequence in the first pair and reverse the sequence in the second pair. Match the three prompts within each length and adjacent pair, giving six paired observations at each input length. Preserve first-sample, decode, wall and host paging observations separately.

Use native time to first token as the primary cold-prefill measure. Compare decode and total wall only on matching input, output, finish, sample counts and MTP work. The prior promotion rule permits at most a 5% cold-prefill regression under matched work. Read each length and both orders against that limit. Treat material paging or an order-dependent marginal result as a limit on the inference. A cold pass addresses the fresh-prompt criterion. The prior Pi study accepted two of eight full pairs against a requirement of six, and the artifact-scoped cancellation and image lifecycle checks also belong to any eventual default change.
