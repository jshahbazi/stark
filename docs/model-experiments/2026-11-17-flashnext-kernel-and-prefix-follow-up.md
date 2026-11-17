# Flash Next kernel and prefix follow-up

**November 17, 2026 · Lily Q4-XL on an M5 Max with 128 GiB**

| Attempt | Change | Evidence and decision criteria |
| --- | --- | --- |
| 20261117-FLASHSPEED-08 | Keep the 4,096 physical grid and place recurrent snapshots at 2,048-token boundaries. | Assess the opportunities in the existing Pi trace and size the required state. An implementation needs exact capture neutrality and a fresh-versus-resumed comparison with matched MTP trajectories, context, sampling, and admitted cache scope. |
| 20261117-FLASHSPEED-09 | Fuse grouped Q4 expert gate/up and SwiGLU on the existing 4,096 physical grid. | For further qualification, require identical output and MTP work on every pair, median native first-token improvement of at least 5% in both launch orders, and no length with a median first-token regression over 5%. |
| 20261117-FLASHSPEED-10 | Share packed Q8 HC down weights across the verifier rows, as implemented by `STARK_LILY_VERIFY_HC_DOWN_REUSE`. | For further qualification, require identical output and MTP work on every pair and at least 3% median decode improvement in both launch orders. First-token time is a host-variation control because the selector only affects verification. |

## Controls

Screen each kernel separately with one model resident and the sealed Lily Q4-XL worker from [attempts 02–05](2026-11-17-flashnext-speed-experiments.md). Model `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl` is pinned at `0f94842e3682b49acace4daeb4cf2e65a6f94f83`, with retained vision and MTP loaded and artifact digest `18c7bcc1e97d1fefdd361472c2daa59d81d89fcd6f3a61983d995a032b44bfb4`. Worker SHA-256 is `bdf851bedf895ab09859cbdc40cab583a9976ddaf06f01d98ad3872dcd8f7dbe`, built from `3e956d960c631beacf573053a6a2fa0ba0c99b3e`. The only runtime difference is the relevant kernel selector.

Use one previously pinned repository-source coding fixture at each of 8,192, 12,288 and 32,768 tokens. Compare launches in off/on/on/off order, with the same fixture order inside each pair and the reverse fixture order in the second pair. Exclude an 8,192-token warmup on each launch. Use no cache scope, 40,960 context, seed 42, temperature 0.6, top-k 20, top-p 0.95 and two exact-rejection MTP drafts. The MoE cap is 96 samples and the HC cap is 192. EOS is valid if a pair matches. Record matched native first-token, decode, and verifier duration, complete wall time and system paging. Treat altered trajectories as correctness evidence; exclude those pairs from matched-work timing. Stop an arm on native failure, worker timeout, artifact change, or geometry or MTP admission mismatch. Restore the installed service after the run.

This is a directional screen with six matched observations per selector and three distinct fixtures. A passing screen earns follow-up qualification. Default promotion would also require broader numerical and lifecycle controls with the exact checkpoint, including cancellation and the image transition.

## Snapshot feasibility and trace evidence

The [matched Pi trace](2026-11-17-flashnext-coding-agent-grid.md) contains 76 matched native requests. Each of its 38 incremental cache hits on the 2,048 grid was a reuse at position 2,048, gained over zero reuse with 4,096. That midpoint became eligible once per session in the eight candidate sessions. The other two hits reused 4,096 tokens in both arms. Across this trace, recovering that one midpoint could avoid at most 77,824 of the 185,516 tokens the 4,096 arm processed fresh (41.95%). Those are prefill token counts. The measured Pi latency gains also include the changed physical partition, so the trace cannot determine the midpoint feature's latency.

One snapshot is 115,662,848 bytes (110.30 MiB), including 36 F32 GDN states, their BF16 convolution windows, PLE's convolution window, and the draft head's hidden row. The n-gram history and positional metadata are small separate fields. Attention and indexer caches remain resident. The existing estimate accounts for two snapshots during replacement. Retaining every nonfinal 2,048 boundary at a 32,768-token input would consume about 1.62 GiB of snapshot payload. A single selected snapshot keeps payload independent of prompt length. [Snapshot implementation](../../vendor/lily-core/src/qwen4exp/model.rs) · [Memory accounting](../../vendor/lily-core/src/qwen4exp/request_memory.rs)

| Input tokens | Latest 2,048 boundary | One checkpoint payload | Payload written if capturing every interior midpoint |
| ---: | ---: | ---: | ---: |
| 8,192 | 6,144 | 110.30 MiB | 220.61 MiB |
| 12,288 | 10,240 | 110.30 MiB | 330.91 MiB |
| 32,768 | 30,720 | 110.30 MiB | 882.44 MiB |

These are logical payloads. Restore also reads the snapshot and writes live state. The current snapshot API runs after a complete physical pass and clones its terminal state. An in-pass checkpoint would capture the F32 GDN accumulator at one 64-row scan boundary, reconstruct the GDN and PLE convolution windows and n-gram history, and save the matching trunk hidden row for MTP. The draft cache must restore visibility using its one-token offset. Reuse from the midpoint needs a bridge to the next absolute 4,096 boundary. The bridge can change grouped MoE tile selection, attention geometry, and draft head batching relative to a fresh pass. It needs exact checks for the recurrent and attention state, logits, and the sampled MTP trajectory. A capture-only comparison at unchanged geometry can first establish that adding the checkpoint leaves the main result exact.

Attempt 08 is a trace and source feasibility check. The trace supports one conditionally captured boundary for the immediately preceding scoped request. A grid of retained snapshots has no demonstrated benefit on these sessions. No midpoint kernel or resume path participated in this run, so midpoint overhead and live agent impact remain unmeasured. A live comparison needs scoped fresh requests that actually write a checkpoint, followed by matched cached and unscoped continuations. Unscoped fresh controls by themselves would skip the candidate write.

## Grouped MoE cold prefill

Attempt 09 completed 12 measured requests and four excluded warmups. The six matched pairs had identical emitted token and text hashes, sampled lengths, finishes and MTP counters. The paired median first-token change was **+2.87%** with fusion, with three wins. The baseline-first order measured a median **+8.94%** and zero wins. The candidate-first order measured **−7.43%** and three wins. Across the two orders, the median changes were −1.49% at 8K, +3.98% at 12K, and +2.38% at 32K. The candidate misses the first-token gate. Measured windows saw 156 system pageouts for baseline and 557 for fusion, with zero swapouts in either arm. The unaffected decode path also varied. The screen supports bounded trajectory agreement and leaves the timing advantage unresolved.

The 4,096 grid already qualifies for the existing 64×4 grouped tile. This experiment changes how grouped gate/up outputs are consumed and keeps that tile policy. An ordinary-prefill stage profile would show where further work pays off. It should measure GDN, sparse attention, expert routing and projection stages, and draft catch-up before another fusion or a tile policy change. Existing native prefill GPU spans combine these stages.

## Q8 HC down during verification

Attempt 10 completed 12 measured requests and four excluded warmups. The six matched pairs had identical emitted token and text hashes, sampled lengths, finishes and MTP counters. The selector runs a short-batch kernel that loads and unpacks the packed coefficients, scale and bias once per work item and uses separate accumulators for the token rows. The earlier three-row component test measured 1.115× speedup in isolation. [Implementation](../../vendor/lily-core/src/kernels/metal/hc.metal) · [Component checks](2026-11-17-flashnext-speed-experiments.md)

| Metric | Median paired change | Baseline-first median | Candidate-first median | Candidate wins |
| --- | ---: | ---: | ---: | ---: |
| Native decode | −1.99% | −4.85% | −0.99% | 4 of 6 |
| Native verification | −1.89% | −5.43% | −1.07% | 4 of 6 |
| Native first token | −1.65% | −4.13% | +1.59% | 4 of 6 |
| Full request wall time | −0.84% | −4.31% | +5.00% | 3 of 6 |

The HC down path misses the 3% decode gate in the candidate-first order. A change also appeared in first-token latency, which runs before verification. Measured windows saw 185 system pageouts for baseline and 323 for HC reuse with zero swapouts. This is exact trajectory evidence and a small, host-sensitive timing signal. Leave the selector default-off. The one-state GDN replay design remains a candidate for reducing a larger source of verifier writes. It would save one 108 MiB logical intermediate state in a three-row round, with a replay on partial acceptance and an exactness requirement across stream truncation.

## Disposition and reproduction

Keep 4,096 as the physical default. Both kernel variants remain experimental. The measured requests used one fixture at each length and shared their worker launch with two other requests. Paging counts are system-wide. No live agent or image request ran on these kernel arms. The installed Lily worker returned to Ready after the tests, and the authenticated local completion returned the expected reply with MTP enabled.

The recipe was registered at `6d7ba067`. Private receipt digests are `a480ee574d343d39b74dc71fec198fd17cb89c6ba3d5f2cf7e0fba77f970c257` for MoE and `72a2b515c0b565976dcdb27b631ef502c03f9a415d03158e154a1ad84e2632d6` for HC. Bulk request envelopes and payloads are excluded from the published aggregate. [Runner](../archive/experiments/2026-11-17-flashnext-speed/kernel_screen.py) · [Receipt verifier](../archive/experiments/2026-11-17-flashnext-speed/kernel_screen_summary.py) · [Aggregate results](../benchmarks/data/flashnext-kernel-screen-2026-11-17.json)
