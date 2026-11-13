# 20261113-EDGE0-18: short observed generation speed

| Field | Value |
| --- | --- |
| Date / owner / scope | 2026-11-13 / Codex / local non-commercial evaluation only |
| State / result | Completed observational speed run; MTP off 6.35 decode tok/s, one draft 2.77 tok/s |
| Source | `ca0b2e5b47d5857be9b733cc4c1d9d01fbc8a9a7`; unchanged inference binary |
| Artifact | `Edge0/Edge0-145B-q3k@d7de835cfce7d9bb22a3d5ebb7d1e8565d4ecf74` |
| Plan / predecessor | [Phase 5](../EDGE0_PLAN.md#phase-5-mtp); [attempt 17](2026-11-13-edge0-arm-attribution.md) |
| Evidence | [Observed speed record](../benchmarks/2026-11-13-edge0-observed-speed.md) |
| Disposition | Practical speed observation; strict performance qualification remains pending, MTP default off |

## Question and authorized scope

After strict qualification repeatedly rejected timing samples, the owner asked
for a working generation run and an observed token rate. This separate recipe
runs immediately, without quiet admission or a claim of host isolation. Earlier
registered qualification screens and rejected results retain their original
status. This observation does not convert a failed qualification gate to a pass.

## Implementation and workload

Reuse the existing native Metal worker, original weight formats, F32 arithmetic,
state and cache, selected-record bindings, context 512 and unavailable prefix
reuse. No inference implementation, weight, default or installation changes.
The source manifest, worker, corpus, receipt and helper identities are sealed
before the run and checked again after worker cleanup. Prior payload hashes keep
their recorded immutability assumption; no new large-payload rehash is claimed.

Use the retained 128-token input, greedy seed 42 and a maximum of 16 outputs.
Warm MTP off and one draft once each; measure off/on/on/off once, two requests
per route. All six requests emit/sample the same 16 tokens with a length stop
and match the predecessor output hash. Warmups are excluded from the medians.
Only aggregate rates, counters and hashes are retained. Paging endpoints are
recorded without subtraction; no full host monitor or waiting loop runs.

## Results and limits

| Median observed rate | MTP off | One draft |
| --- | ---: | ---: |
| Decode tok/s, first token excluded | 6.35 | 2.77 |
| Output tok/s including prompt processing and request overhead | 1.20 | 0.98 |

MTP was slower in these four measured requests. Leave the existing off default in place.
The worker exits normally and is reaped. Script syntax and an independent
calculation/source/cleanup review pass; no new numerical or lifecycle suite is
claimed. Two measured requests per route establish a short local observation,
without a general speed guarantee or a completed Phase 5 performance gate.

This attempt has no native failure or deferred group. Host activity remains
uncontrolled, and broader context, prefix reuse, images, product admission and
commercial use remain outside its scope.
