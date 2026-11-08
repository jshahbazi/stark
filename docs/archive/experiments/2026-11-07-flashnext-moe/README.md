# Flash-Next MoE projection source patches

Each candidate patch applies independently to engine baseline
`72b5aaef96c939809bc906d10d45a0eef70386c0`. They retain the original upstream
copyright and licenses. The pinned Lily source revision and local provenance
are in [UPSTREAM.md](../../../../vendor/lily-core/UPSTREAM.md).

| Patch | Implementation | SHA-256 |
| --- | --- | --- |
| [Candidate 1](candidate1-fixed-projections.patch) | Fixed Q4 expert gate/up and Q8 shared-down dimensions; original dispatch order | `ae704e6611811868054bbcad01427af49723bf6fff9a7199e053abd2a101dee8` |
| [Candidate 2](candidate2-shared-first.patch) | Fixed Q4 expert gate/up; ordinary shared down queued first in the existing dependency level | `34640e336d7ad228fc0368d5790f1df726af7ccb0ba3651ef39d22394b601c19` |
| [Candidate 2, final controls](candidate2-final-controls.patch) | Same production sources as candidate 2; test-only addition of repeated expert routes | `a897e69e365721bae1bc56e9975b53d3bfb29d968ca125439b5cac215f27c723` |

Both candidates are withdrawn from the current engine and retained as patches.
The original two patches bind the measured component and worker receipts.
The final patch changes only the ignored numerical test, expanding its
12 permutation cases to 16 cases with explicit repeated routes. The release
generator rebuild retains the same executable hash. All patches include
their source-bound synthetic controls. Candidate 1's
separate shared-down specialization is retained only as an experiment.
Apply one patch only.

## Diagnostic follow-up sources

The [sustained diagnosis](../../../benchmarks/2026-11-07-flashnext-sustained-diagnosis.md)
records the later experiments. The new aggregate receipt records SHA-256 for
every file below. The native-counter patch applies to either engine baseline
`72b5aaef96c939809bc906d10d45a0eef70386c0` or candidate-2 base
`6698d2c1ba89a85110b976202e6f677a4ac1e9ef`. The gate-only worker patch applies
independently to `72b5aaef`; idle and crossover patches apply independently to
`6698d2c1`. Do not stack patches or apply them on top of restored current source.

| Patch | Purpose |
| --- | --- |
| [Native GPU spans](native-gpu-busy.patch) | Profile-gated reads of existing GPU-busy counters; used in baseline, gate-only and final telemetry builds with the corresponding original/candidate core sources |
| [Gate-only telemetry worker](diagnostic-gate-worker.patch) | Complete native-counter plus fixed gate/up patch against `72b5aaef`, with original shared-down order and its numerical control source |
| [Idle prototype](idle-probe.patch) | Native counters plus private empty GPU queue signal operation |
| [Crossover prototype](crossover-worker.patch) | Native counters, private idle operation and serialized three-way projection selector; clears retained prefix state on each selector change |
| [macmon intervals](macmon-intervals.patch) | Source-read interval brackets, against macmon `5df67a54e082e0b30a0aa2b48b7c7f78a36f1795`; includes an empty Cargo workspace boundary for isolated builds |

The macmon patch retains its [MIT license](LICENSE.macmon). It is temporary
benchmark tooling, with no new runtime dependency or globally installed tool.
The private worker operations are absent from the current production worker.
The [frozen v3 cadence harness](telemetry-v3/benchmark_qwen38_cadence.py) and
[process helper](telemetry-v3/qwen38_process_metrics.py) bind the 64-request
telemetry grid. Their unchanged dependencies are bound by hashes in the receipt.
The current scripts bind v4 idle and v1 crossover measurements. The initial
16-request clock-only probe has a recorded v2 harness hash without a retained
byte-exact source snapshot; this limitation is preserved in the report.

The [report](../../../benchmarks/2026-11-07-flashnext-moe-projections.md) and
[aggregate receipts](../../../benchmarks/data/flashnext-moe-projections-2026-11-07.json)
record the executable identities, numerical checks, component measurements
and incomplete/ineligible matched attempts. Whole-model performance remains
unqualified. A patch applying cleanly is not performance qualification.
