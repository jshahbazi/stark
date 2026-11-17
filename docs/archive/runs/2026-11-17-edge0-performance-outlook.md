# Edge0 performance outlook

2026-11-17. Summary of attempts 21 to 23 on the pinned Edge0-145B Q3_K
artifact, where the time goes now, and a guess at what further work could
reach on the same Apple M5 Max. The ceilings below are estimates, not
measurements, and qualify nothing. Measured values come from the attempt
records and benchmark reports linked here.

## What attempts 21 to 23 changed

Every schedule below is opt-in, defaults off and matched the grouped schedule
bit for bit under its verification selector.

| Schedule | Selectors | Prefill tok/s | Decode tok/s | Record |
| --- | --- | ---: | ---: | --- |
| Grouped, the default | none | 14.70 | 11.98 | |
| Direct routes | `STARK_EDGE0_DIRECT_ROUTES=1` | 19.43 | 13.97 | [Attempt 21](../../model-experiments/2026-11-17-edge0-direct-routes.md) |
| Token-batched route and trunk rows | `token-routes` | 30.11 | 14.84 | [Attempt 22](../../model-experiments/2026-11-17-edge0-token-rows.md) |
| Token-batched mixes and mixer outputs too | `tokens` | 39.30 | 14.90 | Attempt 22 |
| Same, 16-token prefill batches | `tokens`, `STARK_EDGE0_PREFILL_TOKENS=16` | 48.26 | 14.79 | [Attempt 23](../../model-experiments/2026-11-17-edge0-prefill-batches.md) |
| Same, 32-token prefill batches | `tokens`, `STARK_EDGE0_PREFILL_TOKENS=32` | 54.18 | 14.78 | Attempt 23 |

The first four rows are one set of sessions of the attempt 22 binary. The last
two are from the attempt 23 binary, whose 8-token arm ran at 41.49 prefill
tok/s. All use the 128-token input with a 16-token output limit and MTP off.
On a 1,024-token input through the local loading route, request time fell
from 55.0 seconds on the direct schedule to 20.1 seconds at 32-token batches.

## Where the time goes now

Stage-timed command GPU time per token, `tokens` schedule.

| Stage | Prefill at 32-token batches, ms | Decode, ms |
| --- | ---: | ---: |
| Selected experts | 5.4 | 13.4 |
| GDN and attention input projections | 1.5 | 8.1 |
| Hyper-connection mixes | 2.1 | 7.6 |
| GDN and attention state, with output projection in decode | 2.5 | 6.7 |
| Batched mixer output projection | 0.5 | |
| U projection | 1.8 | 4.3 |
| D projection | 0.8 | 3.9 |
| Vocabulary readout | | 2.4 |
| Router and shared expert | 0.3 | 2.4 |
| Other stages and gaps | 1.2 | 1.3 |
| Command GPU total | 16.2 | 50.2 |

A decoded token also waits on 49 command buffers. Its blocked wait exceeds
GPU execution by about 13.5 ms, so one token takes about 67.7 ms of wall time.

## What bounds further gains

- **Reduction contract.** Every output row is one 256-thread group with a
  256-column stride and an ordered eight-group F32 sum. All three attempts
  kept it, which is why their results are bit-identical. It fixes the kernel
  shape: on the order of a billion GPU threads per decoded token, each doing a
  few multiply-adds.
- **Weight reads per decoded token.** The artifact's metadata lists 6.6
  billion active parameters, about 7 GB of stored weights that a decoded token
  must read once. The large one-token Q8_0 kernels read 250 to 280 GB/s of
  weight bytes. If that rate is a hardware limit, decode needs about 25 ms of
  GPU time per token however the kernels are written. Whether it is the
  hardware's limit or the kernel's is not established.
- **Router round trips.** Routing runs on the host in F64, so each layer
  commits and waits. Moving it to the device changes routing authority and
  the selected-expert paging design.
- **Causal state.** The GDN and attention recurrences run one token at a time
  in prefill as in decode.
- **Routing diversity.** Each selected expert receives 1.75, 2.18 and 2.83
  rows at 8, 16 and 32 tokens, so expert sharing grows slowly with the batch.

## Estimated ceilings

| | Measured now | Bit-exact ceiling | Everything built out |
| --- | ---: | ---: | ---: |
| Prefill tok/s | 54 | about 70 to 80 | about 100 to 150 |
| Decode tok/s | 14.8 | about 16 | about 30 to 35 raw; about 40 to 60 with MTP drafts |

- The bit-exact prefill figure assumes wider token kernels, batches above 32
  tokens and fused per-token dispatches in the mixes.
- The larger prefill figure assumes row reductions shaped like a matrix
  multiply. The state recurrence and expert sharing then set the floor.
- Raw decode assumes kernels at the observed weight-read rate and routing on
  the device. Bit-exact decode has only small steps left.
- MTP drafts are the only lever below the per-token weight-read floor: a
  batch of drafted tokens shares its weight reads. Draft acceptance on this
  artifact was not measured.

The least certain inputs are the weight-read rate and draft acceptance. If
the read rate has headroom, raw decode could pass 40 tok/s. If acceptance is
poor, MTP adds little.

## Remaining levers

Bit-exact, verifiable with the existing selectors:

1. Token kernels of 16 rows per dispatch for trunk, mix, D and U rows.
   Estimated from the kernel probe at about 10 percent more prefill at
   32-token batches.
2. Prefill batches above 32 tokens. More expert sharing, at the cost of
   cancellation latency and scratch per graph segment.
3. Fusing the per-token normalization, activation and combination dispatches
   of the hyper-connection mixes.
4. One encoder scope per layer for all selected experts. Attempt 21 measured
   2.6 percent more decode and did not retain it because its paging effect is
   untested.

Needing tolerance-based qualification, because summation association changes:

5. Applying U once per projection group instead of once per route.
6. Row reductions that are not bound to one 256-thread group per output row.

Changing routing authority or the request path:

7. Routing on the device, removing 48 of the 49 round trips per decoded token.
8. MTP drafts verified as token batches.
9. Prefix reuse, [Phase 6 of the plan](../../EDGE0_PLAN.md#phase-6-prefix-reuse).
   It removes prefill for a repeated prefix and does not change tok/s.

## Before adoption

Every selector stays off by default on both loading routes. Turning any of
them on for the local route still needs the attempt 20 context and lifecycle
controls with the selectors on, the paging screens of attempts 12 to 14,
inputs above 1,024 tokens, cancellation latency at the selected batch, peak
memory, and strict timing qualification. MTP-enabled requests were not run
with the new schedules.
