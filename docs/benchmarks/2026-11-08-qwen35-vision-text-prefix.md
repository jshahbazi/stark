# 20261108-VTP-03: Qwen vision text prefix timing

Current scoped observation, 2026-11-08. This does not supersede results for other
models or workloads. The [source-bound validation record](../archive/runs/2026-11-08-vision-text-prefix-reuse.md)
binds the implementation, artifact hashes, precision, failed coverage recipe,
passing exactness workload, snapshot accounting and lifecycle tests. The
[model experiment ledger](../model-experiments/README.md) indexes VTP-01 through VTP-03.

## Source, artifact and outcome

Candidate inference source `0000f093`, base `540a21a9`; test-only fixture follow-up
`41190f9c` does not change inference. Baseline `f77100f5` contains only catalog
changes relative to that main base. Official `Qwen/Qwen3.5-0.8B`, revision
`2fc06364715b967f1860aea9cf38778875588b17`, existing schema-2 BF16 vision artifact.
Weights, activations, accumulation, recurrent F32 state, KV precision, decoder
kernels and sampling policy are unchanged. Text requests bypass the vision tower.

Host: Apple M5 Max, Mac17,7, 128 GiB RAM, macOS 26.7.2. Workers built separately
with Rust 1.98.1, locked dependencies and default Metal features. Registration,
fixed criteria and recipe were recorded before this run in the source record.

Outcome: functional timing gate passed, with identical output hashes in all arms
and expected reused counts. This is an observed first-token latency benefit in
a bounded synthetic session. Adoption on the branch is supported by the separate
32/32 model exactness checks, six CPU/Metal snapshot tests and lifecycle checks.
No installation, release, general quality, decode-throughput or other-artifact
qualification follows from these timings.

## Workload and measurement

The validator uses deterministic synthetic agent history rendered with the
artifact template, four tools and two instruction paragraphs. First-token target
1,800; actual timing turn lengths below. Seed 12 for the session; greedy sampling,
eight output tokens, context 8,192, MTP absent. Follow-up additions use 4/4/4
sentences. Two paired blocks, two rounds per arm per block; order baseline then
candidate in block one, reversed in block two. Four observations per arm/turn.

One worker is loaded at a time. Each has one untimed unscoped warmup. Owned
compilation and unit tests did not overlap timing. The first measured turn is
cache-cold on a warmed loaded model, and excludes model loading. The measurement
is worker wall time to first sampled token, including state allocation/setup
and possible native decode lookahead; it is not isolated GPU prefill time.
Background GPU/CPU load, thermal eligibility and paging were not screened.
Peak RSS was not sampled. These limits prevent a broad performance guarantee.

## Results

Seconds; median followed by minimum–maximum. Every follow-up reused 1,024 input
tokens. Baseline and candidate first turns reused zero.

| Turn / input tokens | Baseline | Cached | Median ratio | Reused tokens |
| --- | --- | --- | --- | --- |
| 1 / 1593 | 2.101 (2.069–2.129) | 2.115 (2.056–2.248) | 0.99x | 0 |
| 2 / 1709 | 2.254 (2.205–2.378) | 0.927 (0.917–0.940) | 2.43x | 1024 |
| 3 / 1813 | 2.456 (2.377–2.500) | 1.058 (1.053–1.060) | 2.32x | 1024 |
| 4 / 1891 | 2.539 (2.473–2.569) | 1.199 (1.158–1.212) | 2.12x | 1024 |

The first-turn median differs by about 0.7%; this sample does not establish a
meaningful cold-turn regression. Follow-up baseline/cached latency ratios are
2.12–2.43.
The extra snapshot payload is 20,201,472 bytes (19.265625 MiB), separately from
existing live recurrence and retained KV capacity. No additional KV prefix is
copied. See the source record for allocation formulas and sparse-page limits.

## Private evidence identity

The private aggregate receipt is `runtime/vision-text-prefix-reuse-2026-11-08/official-v3-timing.json`.
Only aggregate timings/counts and hashes are published here.

| Identity | SHA-256 |
| --- | --- |
| receipt | `449fc2693da72d10548cefab556cca8fcc12fd9be4eb8bff5bf6dc10d87a7e27` |
| baseline worker | `83b3430918e3ea03e5b01aeb9dee542b0a227ad4ddd5346ca90cb222b57045d7` |
| candidate worker | `a16af2a91d0000210d6fc3deefbe432a79c866282d780527efda98049c2821b6` |
| harness | `58ad8a0435183fc12d481afda3cdb4142cca529efd8a17cd54914ef166e7a82b` |
| weights | `7c5960e7225b3b36844fa397a85c5e20d281d732e382475062f552c0737c79d4` |
