# Sushi and Stark on Qwen3.8-Flash-Next: source and record comparison

Date: 2026-11-19. Attempt: `20261119-SUSHI-01`.

## Scope

This record answers one question: why does Sushi report a higher decode rate
for its `Qwen3.8-Flash-Next-Sushi-4bpw` pack than Stark records for
`fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl`, and how different are the two
engines overall.

It is a reading of both projects' sources and published records. **Nothing was
built, run or measured for this record.** Sushi was read at commit
`297d1e2cf360984340298cfcf86261abbc2eb223` (2026-11-19) from a private clone
and was neither built nor executed. Every Sushi number below is Sushi's own
published figure; every Stark number is from an existing Stark record. Both
sets were taken on an Apple M5 Max with 128 GB, with different prompts,
sampling, builds and dates, so no row of the tables is a matched comparison.
As [the native hybrid reference](../NATIVE_HYBRID_MODELS.md) states, published
Sushi or MLX throughput does not qualify Stark's throughput.

Values marked *derived* are arithmetic on recorded values and are not
measurements. The Stark profile that follows this reading is a separate
attempt, `20261119-SUSHI-02`.

## Finding

Sushi is not faster at running the model. It is faster at speculating.

On one-token-at-a-time decode, Stark's Lily path is ahead of Sushi's 4bpw
pack. Sushi's headline rate comes from multi-token prediction (MTP) rounds
that verify many draft tokens for little more than the cost of one, measured
greedy on predictable code. On low-predictability text the two engines record
about the same rate.

## Recorded numbers

| Quantity | Sushi 4bpw | Stark Lily Q4-XL |
| --- | --- | --- |
| Decode without speculation | 62.0–62.3 tok/s (`--no-mtp`, 512 greedy tokens on a code prompt) | 72.97 tok/s at 4,096 input; about 80 tok/s at 512 input (sampled) |
| One-row forward, *derived* | about 16 ms | 13.96 ms (1,020 steps in 14.24 s) |
| Forward by verify rows 1 to 8 | 17.52, 20.57, 22.43, 24.99, 27.07, 30.81, 34.15, 37.36 ms (decode meter on the STQ-2.75bpw pack at 8K context, not the 4bpw pack) | not recorded |
| Cost of each extra verified row | about 2.9 ms (Sushi's statement for STQ-2.75, widths 2 to 8) | about 9 to 9.6 ms, *derived*, and it includes the draft step, host readout, host sampling and rollback for that position |
| Draft depth | Chosen per round; cold-start cap of 6 drafts on this model, at most 8, and up to 8 in a prompt-lookup round | 1 to 3, fixed per request; 2 recommended; 0 by default |
| Round cost | 24.10–24.29 ms on the 4bpw pack (live, 512 greedy tokens on a code prompt) | 32.1–33.2 ms at two drafts, *derived* (388 rounds in 12.45–12.89 s at 4,096 input); about 29.4 ms, *derived*, on a short natural coding prompt |
| Tokens per round | about 3.5, *derived* from 146.2 tok/s and a 24.1 ms round taken on different builds | 2.63 at two drafts (632 accepted in 388 rounds); 2.81 on natural coding |
| Decode with speculation | 146.2 code and 79.5 prose at the 4k rung (greedy, `ignore_eos`) | 82.85 tok/s at 4,096 input; 95.69–98.10 natural coding and 79.32–82.61 planning (sampled, temperature 1, top-k 20, top-p 0.95) |
| Prefill | about 2,100–2,400 tok/s | about 1,800–2,500 tok/s, *derived* from prefix-reuse records |

Sources for the Sushi column: `README.md`, `docs/benchmarks.md`,
`docs/bench/v1.6.0/Qwen3.8-Flash-Next-Sushi-4bpw.md`, `docs/qwen4-perf.md`
and `docs/engine-mtp.md` at the commit above. Sources for the Stark column:
the [2026-10-31 MTP aggregate](../../website/public/benchmark-qwen38-mtp-lily-results.json),
the [2026-11-15 Lily baseline](2026-11-15-llama-v1-flashnext-lily-assessment.md),
the [small-row verifier study](2026-11-15-lily-small-row-verifier.md) and the
[midpoint prefix record](../model-experiments/2026-11-17-lily-midpoint-speed-and-release.md).

Two readings follow directly:

- Stark accepts drafts well. At two drafts it commits 2.63 of a possible 3.0
  tokens per round. Acceptance is not what limits it.
- A Stark two-draft round costs about 2.3 times a serial step and returns
  about 2.6 tokens, so speculation gains roughly 13 to 20 percent. A Sushi
  eight-row forward costs about 2.1 times its one-row forward and can return
  up to eight tokens.

## Where the difference comes from

### 1. Verify rows cost less in Sushi

Sushi's verify runs all rows through one forward. A source reading at the
commit above finds three things that keep an added row cheap, and none of
them relaxes exactness: Sushi's stated intent is that each verify row is
bit-identical to the serial step for that position, up to nine rows, and each
kernel is byte-tested against its one-row form.

- **Each weight word is read once for a group of rows** in the affine 8-bit
  trunk projections, the output head and the hyper-connection projections.
  Its row kernels keep one output's codes in registers and stream every row
  through them, reproducing the one-row lane order and BF16 rounding per row.
  Hyper-connection down and up projections take rows in groups of eight.
- **The recurrent GDN state is not captured per row.** A verify writes only
  the final state and keeps the pre-verify state with the gate inputs. On a
  partial accept it reruns the recurrence over the accepted rows from the
  pre-verify state, which reproduces the serial state. The earlier design
  stored a full F32 state per row, 113 MB per row across 36 layers.
- **Many small operations are folded into few GPU launches.** The router and
  shared-expert gate are one kernel, the shared expert rides the routed
  expert launches (an MoE layer went from seven launches to three), and the
  hyper-connection graph is compiled once per width. Sushi describes decode as
  bound by dispatch gaps, about 860 kernels per token.

The routed experts are not shared across rows on this model: each row and
expert slot runs its own launches. Sushi's stage split at 1, 4 and 8 rows
shows that cost directly, with the expert MLP growing 4.9, 10.3, 17.4 ms
while GDN, hyper-connections and attention stay nearly flat. Its record of
the path shows the eight-row forward falling from 44.45 ms to 37.36 ms across
four changes (prepared hyper-connection cache and dense rows, GDN replay, a
single-verify gate at eight rows, GDN prework across the threadgroup).

The M5 matrix units appear to play little or no part in a verify on the 4bpw
pack. They are used in Sushi's trellis-trunk row kernels for other packs, and
in prefill. This is inferred from the kernel gates and was not exercised.

Stark verifies through the vendored Lily batch path with
`BatchMode::VerifyHost` ([host_spec.rs](../../vendor/lily-core/src/qwen4exp/host_spec.rs)).
Sharing packed weights across verifier rows was first adopted on 2026-11-16
and improved paired useful decode by 1.84 to 5.40 percent
([adoption record](../model-experiments/2026-11-16-lily-projection-reuse.md)).
The Stark decode-share table from the verifier study shows where the row cost
sits: hyper-connections about 16 percent, expert gate/up about 16.5 percent,
expert down/combine about 11 percent, GDN about 19 percent in three stages,
attention about 9 percent, the vocabulary head about 5 percent.

### 2. Sushi drafts deeper and chooses depth per round

Sushi's policy picks a depth for each round. An expected-value planner
maximizes expected tokens per millisecond, using a per-position acceptance
average and a measured round-cost table keyed by verify width and KV-size
bucket. On this model the cold-start cap is six drafts and the table may plan
above it on widths it has measured, to at most eight. Rounds can extend in a
second chunk when the draft head's confidence clears a threshold, and at long
context the policy falls back to serial decode when rounds are losing. It
also has prompt lookup: a 4-gram index over the prompt and committed output
proposes up to eight drafts with no draft-head forward when the output is
copying earlier context (at least 8 matching tokens across a line break, or
32). A lookup miss is verified and rolled back like any other draft. Sushi
records 16 to 24 percent from lookup on copy and edit tasks.

Stark fixes the depth for the request. `MAX_DRAFTS` is 3
([model.rs](../../vendor/lily-core/src/qwen4exp/model.rs)), two is recommended
and zero is the default. Three drafts was 13.75 percent slower than off at
4,096 input with top-k 20 in the 2026-11-15 baseline, which is the expected
result when each added row costs most of a serial step.

### 3. Sushi does less host round-tripping per round

In the Stark Lily host path every draft step commits a command buffer, waits
for it and reads a complete F32 vocabulary row (248,320 values) back for host
sampling. The verifier then returns one complete F32 row per verified
position for host acceptance and residual sampling
([GENERATION.md](../GENERATION.md), `draft_host` and `verify_host_batched`).
This is the `exact-rejection-v1` contract: accept with `min(1, p/q)` and
sample the residual from `max(p - q, 0)`.

Sushi's draft chain stays on the GPU. Each draft step feeds the previous
step's token into the next without a host read. The draft head reads out
through a coarse copy of the output head (requantized to 3 bits), keeps the
top 32 and re-scores those 32 rows exactly. Complete logits are never read
back. A greedy round reads back the draft token ids and the verify argmaxes,
a few integers, and accepts by prefix compare on the host, which makes its
output byte-identical to serial decode. A sampled round on its MLX path uses
the same `min(1, p/q)` rule with a residual correction as Stark, but reads
back only the per-position probabilities and a pre-sampled correction token
per row. Sushi states one forward is more than 4,000 graph operations and is
GPU-bound, and it carries patches to MLX that reduce host overhead.

How much of Stark's round is host readout, host sampling and synchronous
waits, as opposed to GPU work, was not recorded anywhere this reading found.
That is the subject of `20261119-SUSHI-02`.

### 4. The benchmark conditions differ

Sushi's bench protocol is greedy, `ignore_eos`, 512 decoded tokens on one
fixed seeded synthetic TypeScript corpus, and 128 tokens on *Pride and
Prejudice* for the prose arm. The corpus is generated from a fixed seed as
repeated module shapes (an import, an interface, a service class with retry
and cache methods) over short fixed word lists. Greedy decoding on a regular
code corpus is
close to the best case for speculative acceptance and for prompt lookup. At
the 4k rung the same build records 146.2 tok/s on code and 79.5 tok/s on
prose, a ratio of 0.54.

Stark's records are mostly sampled at temperature 1 with top-k 20 and top-p
0.95 on natural or public prompts. Its natural coding figure with two drafts
is 95.69–98.10 tok/s. Stark has no greedy fixed-corpus measurement to set
beside Sushi's code arm, so the size of the gap under matched conditions is
unknown.

### What is not the reason

The quantization format does not explain the speed. Sushi's own plain affine
4/8 control pack decodes serially at 63.6–65.7 tok/s, about the same as its
trellis-coded packs. Sushi's 4bpw pack is EXL3 trellis-coded experts with an
affine 8-bit trunk and still runs on MLX; Stark's pack is affine Q4 for GDN
projections and experts with Q8 for attention, the shared expert, readout and
embedding.

## Derived estimate

If Stark's extra verified positions cost about 3 ms instead of about 9 ms, a
two-draft round would take about 20 ms instead of 32 to 33 ms. At the recorded
2.63 tokens per round that is roughly 130 tok/s. This is arithmetic on two
engines' separate records. It assumes the host work could be removed along
with the GPU row cost, and it is not a measurement or a target.

## Overall design differences

| Area | Sushi | Stark |
| --- | --- | --- |
| Language and stack | Zig; a detached fork of mlx-serve on MLX and mlx-c, with custom Metal kernels injected into MLX's lazy graph and patches to MLX host overhead. A second engine of its own serves its STQ packs; affine, MXFP4 and the 4bpw pack stay on MLX | Rust and Metal 4 with no MLX; the Flash-Next path is the vendored Lily core (`vendor/lily-core`, from Lily revision `e22d58fa7cc488493fb1b5d810a198bd29730aeb`) behind `Forward::Lily` |
| Quantization | Builds its own packs (EXL3 trellis and its own STQ format) and gates them on KL divergence | Runs the third-party Lily pack unchanged and qualifies each artifact separately |
| Speculation | On by default; adaptive depth to 8 (cold-start cap 6 on this model); prompt lookup; measured round-cost table; an optional separate drafter | Opt-in per request; fixed depth up to 3; exact rejection sampling on the host with complete F32 logits |
| Serving | HTTP server with multi-slot batched decode, an SSD-persisted prefix cache, expert streaming from SSD and context to 1M | Serialized worker with one loaded model and one in-memory prefix snapshot that is never persisted, by policy |
| Precision and state | Q8 KV option, F32 GDN state | BF16 activations, F32 reductions, logits and GDN state, Q8 KV |

Both engines keep speculation exact with respect to their serial output
distribution, both aim for verify rows that are bit-identical to the serial
step, both keep the recurrent state in F32, and both contain kernels for the
M5 matrix units. Sushi's source is MIT licensed, with Apache-2.0 notices for
some third-party portions.

## Follow-up

The Stark profile was run the same day as attempt
[`20261119-SUSHI-02`](../model-experiments/2026-11-19-lily-mtp-round-profile.md).
It measures an extra verified row at about 5.7 ms of GPU time and host work at
2.5 to 3.8 ms per round, so the derived 9 ms per extra position above is
mostly verify row cost plus one draft step, not host readout. The text above
is left as written.

## Limits

- No matched measurement exists. The Sushi verify-row meter was taken on the
  STQ-2.75bpw pack, not the 4bpw pack.
- The Stark per-row figure is derived from whole-round times and mixes GPU
  row cost with draft steps and host work.
- Sushi's mechanisms are described from its documentation and source at one
  commit. They were not exercised.
- Nothing here qualifies a Stark change. Any optimization prompted by this
  reading needs its own attempt, exactness controls and matched Stark
  before/after timing on the same artifact.
