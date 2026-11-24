# 20261121-INKLING-12 · Original Inkling 2 Mini DSpark

The original three-layer DSpark head passed the first-mask proposal gate on the M5 Max with the image tower resident. One proposal comes from the first mask of the full eight-row block. The target uses its ordinary sampler and random-number sequence for every emitted token. The serial verifier stays opt-in. [Port plan](../INKLING_2_MINI_PLAN.md#4-qualify-dspark-then-choose-a-default) · [Oracle procedure](../archive/experiments/2026-11-21-inkling-oracle/README.md#original-dspark-capture) · [Ledger](README.md)

## Identities and controls

| Artifact | Identity |
| --- | --- |
| Original target SHA-256 | `65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1` |
| Original draft SHA-256 | `7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2` |
| Source and runtime | Ollama `6949a42ad24450cf567d7ce1ff7d278ebb3eb5d8`, released MLX Metal v4 libraries |
| Source executable SHA-256 | `b1aaf5f6484447a480a0dbea6b2f4b501ce614a6e716b6219d3e5c431672a220` |
| Private source stage fixture SHA-256 | `ef99246907007d9e983c11f7e6429008781a7568822ec39035d8a455cfdca259` |
| Initial isolated stage probe SHA-256 | `fdaff8ae7941f7f1fc387b5324fbbe0c88289edde1daaab2e0eb9fbe1eeff09a` |
| Fixed isolated stage probe SHA-256 | `f9293c26c37413b63b46c253af3328eda8327934897da6cccfeed570a99f7a02` |
| Fixed aggregate receipt SHA-256 | `1d7eb3c6f4dc20052e2423f38e906b21fa62e6411b21f487cbdd3c02cce352e0` |
| Probe build | Rust 1.98.1 locked release, `--no-default-features --features inkling-slice --example inkling_slice`, original target plus draft on M5 Max |

The private capture uses a four-token prefix and four anchors. It feeds the same committed target taps to the pinned draft and native draft for component comparison. The connected control then compares 16 serial target steps with draft off and on, including exact target logits, emitted tokens, prefix snapshot restoration, rejection of another draft mode's snapshot, and recovery after disabling the draft. A proposal writes ephemeral scratch. Committed target taps update separate draft context and the latest Markov feature. Snapshots include that state when draft mode is enabled.

## Attempt history

| Arm | Observation | Disposition |
| --- | --- | --- |
| Pinned runtime with benchmark import | The target-only import had no draft descriptor. | Load the actual publisher draft shard and configuration through the pinned draft loader in an isolated capture process. |
| Initial native proposal and target verification | Exact target logits and tokens across all 16 steps. Draft snapshot replay was exact and a draft-off restore rejected the draft-on snapshot. Nine of 15 proposals agreed with the target. All four isolated pinned-runtime proposals disagreed. | Retain the serial verification and lifecycle control. Investigate proposal fidelity. |
| Pinned stage probe | The normalized Markov feature and sampled folded bias were exact. Worst relative L2 was 0.00953 for first-mask hidden and 0.00969 for latent. The sampled folded weights had relative L2 1.0. | Replace the strided Metal tensor views with contiguous FP32 products and an explicit load-time transpose. |
| Fixed fold and serial verification | All four source-ingress proposals matched. Sampled folded weights and bias and the normalized Markov feature were bit-exact. Hidden and latent relative L2 stayed at 0.00953 and 0.00969. The connected target logits and tokens matched in all 16 steps. Exact snapshot replay, mode mismatch rejection and serial recovery passed. Ten of 15 proposals agreed with the target. | Keep the fixed fold and bound the qualification to this short-prefix probe. |

## Timing and remaining gates

The initial short diagnostic took 0.439 seconds for 16 target steps with draft off and 0.506 seconds with the serial verifier. Proposal calls accounted for 67.1 ms. The fixed build took 0.467 and 0.635 seconds in those arms. Its 15 proposal calls accounted for 78.0 ms. These single-process observations are unscreened and include the same resident image tower. The current verifier performs an ordinary target forward per committed token and supplies a correctness control for subsequent batched verification.

At this short prefix, the fixed arm's 29.17 ms serial target step, 5.20 ms per proposal and 10 of 15 proposals accepted imply a two-row verifier budget of roughly 43 ms per round to break even, including target rollback and host work in that budget. Preserve the convolution slot the candidate overwrites and keep unverified draft context out of the committed snapshot. The experiment leaves sustained 4K timing, source parity across a broader set of prompts and contexts, peak memory and sampled rates for separate qualification.
