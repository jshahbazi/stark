# GEN-SERVE-02 · Lily shared generation slots

## Artifact and implementation

The qualified artifact is the retained-vision
`fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl` revision
`0f94842e3682b49acace4daeb4cf2e65a6f94f83`, family profile
`qwen38-lily-retained-bf16-vision-v1` (version 1, artifact schema 6).
The installed artifact identity is
`18c7bcc1e97d1fefdd361472c2daa59d81d89fcd6f3a61983d995a032b44bfb4`.
The prepared model contains 55 payload files totaling 106,905,155,192 bytes.
The workload ran on an M5 Max with 128 GiB via Stark's normal loopback API
using a private copy of the model registry. The vision tower and native MTP
head were loaded in every arm from the existing prepared model.

The Lily loader commits its resident GPU weights to one Metal residency set
that all text slots attach to their own queues. The paged n-gram table is
shared too. Each slot has its own KV and index cache, recurrent state,
prefill scratch, MTP state, and cancellation channel. Queue and tensor
mutability stay on the owner thread. The host uses the existing FIFO and
primary-only image scheduling. The primary uses the existing prefill,
decode, MTP and prefix paths. Only the primary can retain a prefix.
Auxiliary activity invalidates the primary's previous prefix eligibility. [Metal residency-set attachment](https://developer.apple.com/documentation/metal/mtl4commandqueue/addresidencyset%28_%3A%29)

The startup fitter admits auxiliary lanes if existing headroom can cover a
cold prefill chunk for each. Once admitted, slots retain the loaded context
ceiling and reserve request memory against live headroom minus peer
reservations before allocation. Auxiliaries release grown KV, prefill, and
MTP verification storage before serving another request. This restores
headroom after long concurrent turns. Idle workers keep their small initial
state and compiled pipelines.

## Earlier candidate and review

The first live candidate admitted four text slots at 262,144 tokens, matched
four serial replies with MTP both off and on, and ran the queued image on
the primary after all text finished. Its solo timing bracket measured a
median paired ratio of 1.011. Review found a type-safety problem with
publicly transferable Metal buffer handles and noted that idle slots
retained grown request allocations. The final candidate confines transfer
to the private read-only weight owner, detaches residency when a model is
dropped, and retires auxiliary growth on completion. Earlier candidate
results apply only to that executable.

## Final qualification

The verified Rust 1.98.1 release worker (`metal,hy4`) has SHA-256
`bc99ca1f7e0b77819c5e31022fd2bea5f747673f597aa93b64ea054a78527c50`.
The previous shared-slots branch worker, `2063df092`, has SHA-256
`75c93b3ada818eb8f2509548f7acdb712b069b5ac355550e61989f35f02bd057`.

- The paired solo bracket ran base, candidate, candidate, base in separate
  warmed loads. Each load ran the same four distinct 1,770–2,625-token
  inputs with 96 outputs each. Two inputs used adaptive MTP and two used
  MTP off. Output hashes, finish reasons and token counts matched. Median
  solo useful decode was **88.96 tok/s** on the previous worker and
  **88.94 tok/s** with all four slots enabled. The median of the eight
  paired ratios was **1.0001**. Readiness reported 262,144 tokens in all
  four loads.
- Four overlapping text requests matched their serial controls across slots
  with MTP enabled and disabled. An image queued behind those requests
  completed its prepare, generate and discard transaction on the primary.
  Its response and the following fresh text response matched serial.
- Each slot independently planned a 16,384- and a 27,648-token ceiling and
  generated matching token IDs under both ceilings.

The default four-lane launch then ran four distinct fresh prompts at each
size with the native draft head on. Every result matched its solo control
and generated 64 output tokens. Group time is loopback wall time from
dispatch through the last reply. Prefill and decode columns are per-request
medians from the worker.

| Input token count | Solo prefill | Solo decode | Four-lane prefill | Four-lane decode per agent | Four-request wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16,335–16,339 | 7.54 s | 91.14 tok/s | 27.32 s | 11.59 tok/s | 33.30 s |
| 27,600–27,609 | 13.40 s | 92.01 tok/s | 48.01 s | 13.73 tok/s | 53.23 s |

Sampled Metal allocation after both batches returned to 74.21 GiB.
Before each batch it was 74.31 GiB. The sampled peaks were 81.90 GiB at
16K and 82.54 GiB at 27K. A scoped solo follow-up reused 8,192 tokens.
Cancelling a separate auxiliary during prefill cleared the primary's reuse
eligibility. The primary then reproduced its fresh output and the
cancelled slot served another request successfully.

Local checks passed 190 release generator tests with six ignored, 615
Python chat, desktop, API, image and Decision tests with one skipped,
Clippy with warnings denied, formatting, repository policy and research
links.

| Private aggregate receipt | SHA-256 |
| --- | --- |
| Text, queued image, and explicit windows | `5b0a3a11fadb9ebc9179e6bddce6e78df5b966842694edf6a26efed46678b176` |
| Solo paired timing and output parity | `2724afbfa3d0a4746f5e660d707d8ee6ea471cfd9db3f7b13f601a9aad656363` |
| Fresh long requests, allocation and cancellation | `ba8ab6f3bb741a1fcd5fa981d1afafa7b4f6e21e96f964c342c363c2c2a0b2bc` |

## Disposition

Default in the shared generation slots branch for the resident Lily backend
when enough memory is available. Other imported Lily-family artifacts still
need independent qualification. The measured rates apply to these synthetic
loopback requests. More active agents share GPU bandwidth, so per-agent
decode falls under contention.
