# Qwen3.8-Flash-Next three-slot MTP runner-up

**Decision. Keep the shipped adaptive MTP schedule.** The runner-up reached 68.10 tok/s on prose and 96.04 tok/s on coding. Its gains over fixed-three MTP were within the paired uncertainty on each workload. The shipped adaptive policy reached 83.04 and 98.29 tok/s respectively.

## Held-out results

All 24 prompts produced 192 tokens before the length limit. All 176 measured requests matched their prompt's frozen MTP-off output hash, termination reason, and output length. Decode throughput counts the 191 sampled tokens after prefill per request. Whole-request throughput counts all 192 emitted tokens against send-to-reply wall time. The worker and nonstreaming protocol were identical across arms.

| Workload | Fixed three, tok/s | Runner-up, tok/s | Shipped adaptive, tok/s | Runner-up vs fixed three (paired 95% interval) | Runner-up vs shipped (paired 95% interval) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prose | 67.34 | 68.10 | 83.04 | +1.13% (−0.39% to +2.86%) | −17.99% (−20.37% to −15.48%) |
| Coding | 95.92 | 96.04 | 98.29 | +0.12% (−1.04% to +1.19%) | −2.29% (−4.86% to +0.39%) |

- The runner-up beat fixed-three on 6 of 12 prose prompts and 7 of 12 coding prompts across the two passes. Prose gains were +2.14% and +0.14% by pass. Coding moved +0.54% and −0.29%. The two identical width-two controls differed by −0.51% on prose and −0.46% on coding across four prompts per domain. Intervals use 20,000 paired bootstrap resamples of prompts, with both repetitions retained together.
- Whole-request tok/s in fixed-three, runner-up, and shipped order were 63.88, 64.53, and 77.69 on prose and 88.69, 88.80, and 90.62 on coding. Shipped adaptive MTP beat the runner-up on all 12 prose prompts.
- The shipped arm used `mtp:true` and the worker's default adaptive policy with prompt lookup, allowing up to three drafts. The desktop app's two-draft profile has a different limit. Prompt rendering disabled thinking. Inputs ranged from 60 to 79 tokens. The operating system reported a locked screen during timing, AC high-power mode, and no thermal or performance warning at the end.

### What the third slot bought

Counts use one pass over the 12 distinct prompts in each domain. Repetition reproduced the same branch counts. A first miss is the first wrong primary on a round reached by the candidate. A lost third choice is a displaced primary that would actually have matched after the first two matched.

| Workload | Eligible rounds | Rounds with sibling | First misses | Sibling offered at first miss | Sibling caught first miss | Lost correct third choices | Target rounds, fixed three → runner-up |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Prose | 979 | 381 | 763 | 220 | 72 (9.4% of first misses) | 46 | 1,002 → 983 |
| Coding | 700 | 123 | 251 | 63 | 33 (13.1% of first misses) | 33 | 710 → 703 |

On prose, the runner-up converted 72 of 220 offers at the first miss (32.7%). On coding it converted 33 of 63 (52.4%), but spending the slot on the runner-up displaced 33 correct third tokens. The exact totals of target rounds above include the resulting changes in subsequent round boundaries. Across both repetitions, prose verification time fell by 1.28 s while drafting and rollback together rose by 0.53 s. Coding verification fell by 0.33 s and drafting plus rollback rose by 0.28 s.

## Registered comparison

Attempt `20261125-LILYRUNNER-01` was registered on 2026-11-25 against Stark `e0e6916d8e966597ed904c36f01068ad70bac28d`. The question is whether spending a verifier slot on the MTP head's second choice improves greedy decode rate on held-out prose and coding.

- Use the image-loaded Lily Q4-XL checkpoint at `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl`, revision `0f94842e3682b49acace4daeb4cf2e65a6f94f83`. The worker's registered artifact digest is `18c7bcc1e97d1fefdd361472c2daa59d81d89fcd6f3a61983d995a032b44bfb4`. The host is an Apple M5 Max with 128 GiB, running macOS 26.7.2 (25G332) on AC power.
- Retain the 12 prose and 12 coding prompts used for the held-out Inkling test, re-rendered and tokenized for Flash Next. Use all 24 with their original domain assignments and a maximum of 192 greedy output tokens. Fix the runner-up threshold at a top-two logit gap of at most `ln(2)` before inspecting Flash Next outputs.
- Compare ordinary fixed-three MTP against the same schedule with a runner-up. Prompt lookup is off in both. Take the earliest eligible low-margin split. The target can accept the runner-up and emit its verified next token in the same round. Include an ordinary two-draft A/A pair and the shipped adaptive schedule with prompt lookup as separate controls. This is a short-prompt worker experiment.
- Freeze target-only output hashes and termination information before timing. Check every measured request against its own reference. Validate the complete target readout on both fork depths and the readout after committing either path, including early termination. A mismatch invalidates the affected timing.
- Alternate adjacent arms in one loaded worker and counterbalance by prompt and repetition. Report pooled post-prefill tokens per decode second, paired uncertainty clustered by prompt, sibling hits, and accepted primary tails forgone. Evaluate the two domains individually against the A/A spread and the shipped schedule.
- Do not select prompts, thresholds, or depths based on observed timing. Keep private inputs and native output data out of version control.

## Implementation and scope

The verifier computes the two paths in a packed short batch. The sibling uses its logical rotary position, attention prefix, GDN recurrence, dilated PLE convolution and hashed n-gram history. Committing a sibling promotes its captures and cache entries into the accepted prefix before MTP catch-up. The switch is request-scoped and greedy-only. The candidate uses the normal model's projection, accumulation, recurrent-state and cache precision. Source extraction is pinned to Lily revision `e22d58fa7cc488493fb1b5d810a198bd29730aeb`.

The pinned checkpoint uses mixed Q4/Q8 weights, BF16 activations, F32 logits and recurrent state, and the native Q8 KV cache. Full rounds still evaluate the three autoregressive MTP head steps. When a sibling qualifies, the host replaces its pre-encoded linear verifier graph with the selected packed graph. Other implementations could have different overhead.

### Correctness and provenance

- The forced-fork control compared all F32 logits after primary and sibling paths at depths one and two. It exercised three packet shapes, each early-delivery prefix, the post-rollback MTP head and the next target response. All 69 comparisons matched the serial target bit for bit. An independent Metal test passed 144 ranking and threshold cases. The worker passed 105 CPU protocol tests. Ordinary register recurrence matched 648 verification cases and 3,240 replay cases. Fused GDN projection matched 128 cases, 448 rollback prefixes, and 160 replay prefixes.
- At 4,102 to 4,161 input tokens, a scoped ordinary → runner-up → ordinary → new-scope runner-up sequence reused 0, 4,096, 4,096, and 0 input tokens respectively. All four outputs matched the cold target references. Fork eligibility ends when a packet exceeds the 2,051-token dense attention window on this artifact, so the two long requests with the candidate enabled ran the ordinary chain. Sparse-context forks and sampling remain unqualified.
- The image tower and MTP head loaded together. All timed requests were text-only. Peak memory was not measured. The implementation is an opt-in diagnostic on the experiment branch. The desktop continues to use its existing installed build and schedule.
- Rust 1.98.1 release worker SHA-256 `08b64b489f08297bce37555e01c13a29aff15682179203244fbf7bdfb66603f7`, lockfile `07b6d42497825a651719685a28a107bfc0a32119b8e8b8adeab2d6c01a96695b`, timed driver `6e19c5f203a5f3fda49b3697c524b66eccb175df9cb0cc7b6d106c4314f9982c`. The forced-readout executable used pinned nightly `nightly-2026-10-01`, SHA-256 `96c3f71c0ab8c6b73c4142f489f964dfa25b0be470f66166ea65e7467715ab4c`.
- Tokenizer SHA-256 `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`, template `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`. The 24 distinct inputs reproduced the original Inkling manifest under its own tokenizer and had zero exact matches to the four distinct development-trace inputs. The Flash Next references were collected before timing. Canonical reference-set SHA-256 `60c5f5bb78fd662eb096c6863e0445e75dde9194018f2ca7aa97cbe0a4536852`. The private timing receipt hashes to `525ccb334447d15305b980c34f6e4d8e03e88e18fcc56ee8fa85d0b007df2530` and the prefix receipt to `882973e6cf2de8250fc67e6fe91d2501040dd66fa0d1bfa6d65120e86a084098`.

[A/B driver](../../scripts/benchmark_flashnext_runner_up.py) · [Experiment ledger](README.md)
