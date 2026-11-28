# Shared generation slots

Stark Desktop requests four text generation slots for one loaded model. Each
slot admits one active request. The backend advertises the slots it can
initialize within the current budget. The Desktop queues other requests in
arrival order. The 600-second queue deadline is separate from the active
request's 300-second inactivity watchdog. Check `worker_slots` and
`running_requests` in `/v1/health` for a live load. Set `STARK_GENERATION_SLOTS=1` to request the serial path.
[API limits and queueing](LOCAL_API.md#limits-queueing-and-cancellation)

New text generation support requires four slots by default on a qualified
configuration. The [onboarding gate](MODEL_ONBOARDING.md#qualify-four-shared-generation-slots)
defines promotion evidence and memory fallbacks. The [worker protocol](WORKER_PROTOCOLS.md)
describes the pipe and readiness contract.

## Ownership and scheduling

The Hy4 OpenCode workload showed why admission and GPU throughput need separate
measurements. Two slots moved a second long request out of the host queue
sooner, but their prefills competed for GPU time. Four slots admitted the whole
agent group and preserved the solo path. GPU phase contention still increased
per-agent latency. [Hy4 fan-out measurements](model-experiments/speed-hy4-stq.md#opencode-agent-fan-out)

Reuse the Desktop queue, inherited request/reply/cancellation pipes, and worker
admission handshake. Give each slot its own mutable request state, sampler,
stream and cancellation handling. Share the already loaded read-only weights.
The backend owns residency, synchronization and memory admission:

| Backend | State and memory policy | Qualification |
| --- | --- | --- |
| Hy4 Sushi STQ | Shared mapped weights and phase-local scratch behind a FIFO GPU phase gate. Separate KV, IndexPool, draft state and Engram readers. Up to four fitted slots with a common API ceiling and a separately fitted primary. | Pinned 4bpw pack, 16K and 27K OpenCode groups, solo control, distinct outputs, flexible context and image/prefix lifecycle. [HY4STQ-SERVE-02](model-experiments/2026-11-27-hy4-stq-four-slots.md) |
| Native language and vision-language decoders | Shared loaded decoder. Each request reserves its own KV, recurrent state and workspace against live headroom. Preserve model context policy. | Qwen3 4B MLX4 kept its 40,960-token ceiling and matched solo outputs under overlap. Qwen3.5 0.8B BF16 Vision matched concurrent text and a queued image with MTP off. [GEN-SERVE-01](model-experiments/2026-11-27-shared-generation-slots.md) |
| Flint and Ripple | Share the mapped weights, reserve their residency once, and fit independent request graphs and caches. | CPU isolation tests and a Metal build passed. Real Metal checkpoint qualification remains open. [GEN-SERVE-01](model-experiments/2026-11-27-shared-generation-slots.md) |
| Lily | One committed Metal weight residency and one paged n-gram table. Each slot owns its queue, context, KV, recurrent state, scratch and draft controls. Retire auxiliary growth after the request. | Retained-vision Q4-XL kept its 262,144-token ceiling and matched four 16K and 27K responses to solo. Median solo decode held at 88.96 and 88.94 tok/s before and after. [GEN-SERVE-02](model-experiments/2026-11-27-lily-shared-generation-slots.md) |

Keep images and Playground on the primary. Images wait for active text
requests to drain. Only the primary can retain a prefix across requests.
Auxiliary activity breaks the consecutive-request reuse chain. Match context,
scope and draft policy when restoring the next eligible prefix. Decision
continues to serialize each complete evaluation. [Generation prefix
contract](GENERATION.md#prefix-reuse)

## Remaining backend work

The existing serial GLM native, GLM EXL3, generic Qwen hybrid, Inkling, and
Hy4 AngelSlim GGUF paths need their own implementations and artifact evidence.
GLM needs an independent recurrent/attention state per request and a safe
shared expert lifecycle. EXL3 also has a model-thread-affine bridge. Generic
Qwen hybrid needs a split between immutable layer parameters and mutable GDN,
attention and index state. Inkling needs separate target, draft and workspace
owners. AngelSlim GGUF needs separate streaming, NextN and sequence state.
Qualify each checkpoint, quantization, and image/draft configuration
separately. [Backend coverage and
blockers](model-experiments/2026-11-27-shared-generation-slots.md#coverage-and-remaining-backend-work)
