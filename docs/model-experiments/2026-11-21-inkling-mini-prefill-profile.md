# 20261121-INKPROFILE-01 · Inkling 2 Mini 4K prefill stage profile

## Identity and state

| Field | Value |
| --- | --- |
| Attempt and execution | `20261121-INKPROFILE-01`, 2026-11-21, Codex |
| Result | Completed. All 200,058 continuation logits matched the uninstrumented grouped path bit for bit. |
| Scope | Connected native Metal prefill attribution on the original checkpoint with vision loaded |
| Disposition | Opt-in diagnostic |
| Predecessor | [20261121-INKPREFILL-02](2026-11-21-inkling-mini-full-target.md#20261121-inkprefill-02--64-row-shared-projection-and-grouped-experts) |

## Artifact and controls

| Field | Value |
| --- | --- |
| Original checkpoint | Revision `6e7c8248f75265bdd4fbab5273304243b5d8c2cb`, target shard SHA-256 `65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1` |
| Build | Stark `4962169e888f1b7e8558a42826f7331976bb0ab8` plus profiling and trace worktree changes. Release Rust 1.98.1, `--locked --no-default-features --features inkling-slice`, binary SHA-256 `1b5fa50ae39aec03a32d8ae7daa191050547aba58b2149dc401a49886290c2ce`. |
| Lockfile | SHA-256 `07b6d42497825a651719685a28a107bfc0a32119b8e8b8adeab2d6c01a96695b` |
| Host | Apple M5 Max, 128 GiB, macOS 26.7.2. `pmset -g therm` reported no thermal or performance warnings before or after the run. Peak memory and paging were unmeasured. |
| Workload | 4,096 input tokens from the pinned greedy fixture, context 4,100. Private token bundle SHA-256 `74d6457708d31fe3ba6aeeab41b836d6ac7cd825cb3837fc3ddacc64e0c19719`. |
| Execution | Same loaded target and BF16 cache in both arms, 64-row grouped tiles, prefix reuse and DSpark off. One fresh-cache uninstrumented replay, reset, and one fresh-cache profiled replay. A singleton continuation checked output bits outside the timed span. |

The profile places calibrated Metal counter samples at the encoder boundaries inside a tile. It keeps one submission and completion per tile. The command validates samples and checks that the sum of measured intervals fits within the command GPU time, allowing 1% or 1 µs for clock and counter granularity. The underlying prefill schedule, kernel selection and precision policies follow the grouped path.

```sh
cargo +1.98.1 run --release --locked -p stark-engine --no-default-features \
    --features inkling-slice --example inkling_slice -- \
    --profile-prefill "$ORIGINAL_SHARD" "$FULL_ORACLE_4096" 4096
```

`FULL_ORACLE_4096` points to the private full-target fixture with `row-0.safetensors` and `tokens.safetensors`. The command emits only aggregate timings, counts and a continuation parity flag. [Fixture provenance](../archive/experiments/2026-11-21-inkling-oracle/README.md#full-target-extension)

## Results

The profiled replay used 64 GPU completions and 5,747,712 dispatches. Its summed command wall time was 34.658 s and its summed command GPU time was 32.799 s. The first and last tiles used identical row widths.

| GPU phase | Full 4K time (s) | Share of command GPU time | First tile (ms) | Last tile (ms) |
| --- | ---: | ---: | ---: | ---: |
| Attention dense projections | 3.873 | 11.81% | 60.096 | 60.026 |
| Relative/cache attention | 4.395 | 13.40% | 39.955 | 80.464 |
| Routed experts | 13.456 | 41.03% | 202.152 | 212.968 |
| Other encoded work | 10.904 | 33.25% | 167.409 | 167.907 |
| Command GPU gap | 0.170 | 0.52% | 2.649 | 2.634 |

Dense covers Q, K, V, R, attention gate, sink, optional dynamic K/V projections and attention output. Relative/cache attention covers K/V convolutions, normalization and writeback, relative-table construction and the attention reduction. Experts includes grouped routing, expert products and scratch transfers, and serial expert fallback for layers that use it. Other encoded work includes embedding, controller and residual operations, remaining convolutions and normalization. The measured replay ends before readout and sampling.

The relative/cache band doubled from the first tile to the last while dense time stayed near 60 ms. Halving both bands would yield a theoretical 1.14× command GPU speedup under the measured attribution. The largest individual measured band was the expert path. The profile measures time by stage. Establishing whether a particular stage is limited by arithmetic, bandwidth or dispatch cost would require a targeted control. [Performance playbook](../PERFORMANCE_PLAYBOOK.md#find-out-what-binds-the-stage)

## Evidence and limits

Aggregate receipt SHA-256 `45b01de4c87cd466ea6b759d4cf0b08e5f60daf5617824cae6c57e220065dc5d`. This executable captured the uninstrumented continuation for parity. The CLI subsequently added uninstrumented ingest duration to the JSON. The initial release check overlapped the separate trace implementation and found its declared module file missing. The completed source passed the locked release check and the artifact ran on-device.

This is one instrumented grouped-native diagnostic with a continuation exactness control. Cross-runtime numerical parity, cache snapshot equality, production worker throughput, installed build, memory, image requests and active MTP are outside this measurement. The prior worker timings and acceptance floors are tracked in the worker benchmark. [Worker benchmark](../benchmarks/2026-11-21-inkling-mini-native-worker.md)

## Profile after the matched-width repair

The final example also profiled the routed 512-row diagnostic from the matched prefill attempt. It used the same 4,096 input IDs and context 4,100 as the B64 profiler. The replay retained the source-shaped BF16 dense products, local relative attention and FP32 router. The eight global visits and the internal 64-row expert product tiles retained their native kernels. This command compares continuation logits between profiled and uninstrumented executions of the selected 512-row path. [Matched numerical trace at context 4,352](2026-11-21-inkling-mini-matched-prefill.md)

| GPU phase | B64 production graph (s) | 512-row diagnostic graph (s) |
| --- | ---: | ---: |
| Attention dense projections | 3.873 | 3.166 |
| Relative/cache attention | 4.395 | 3.117 |
| Routed experts | 13.456 | 12.698 |
| Other encoded work | 10.904 | 10.242 |
| Whole command GPU time | 32.799 | 29.370 |

The 512-row path reproduced all 200,058 continuation logits when profiled. It used eight completions and 8,143,320 dispatches. Uninstrumented ingest took 30.870 s. Instrumented commands took 30.999 s wall, and Metal tracked 318,226,432 bytes of additional workspace, or 303.5 MiB. The existing worker reserves 128 MiB for prefill workspace. These arms are separate instrumented observations of different geometries and arithmetic. The timing difference does not establish a paired worker speedup. The 512-row implementation stays in the diagnostics.

The final binary was `2794f11765305a6607534e4ce3d4f136cb1dbba16823239af770681a7930968a`. The aggregate is `e7aac894ea9039d0aa2ed3f02f59bf6b5ddd24b096b3b86d6960ffc660081da9`. The host reported no thermal or performance warnings before or after the run. [4K worker rerun without profiling](../benchmarks/2026-11-21-inkling-mini-native-worker.md#4k-worker-rerun-after-the-matched-trace)
