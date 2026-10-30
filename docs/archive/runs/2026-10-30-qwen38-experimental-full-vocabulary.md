# Experimental Uzu full-vocabulary observations, 2026-10-30

The 35-prefix Qwen3.8 and three-prefix ordinary-model controls completed
against experimental worker `e2b9180f…`. All readouts were finite. These
observations do not establish model accuracy or a production precision
acceptance threshold. The selective GEMV trial in this source was later
reversed after a separate same-loaded sampled-generation control failed
cross-arm output parity. This record does not qualify the restored source.

The [rejected tuning record](../../benchmarks/QWEN38_UZU_REJECTED_TUNING_2026_10_30.md)
preserves that sampled-generation failure and the source/binary identities.
Earlier failed precision and split-K records remain unchanged. Fresh seals,
matching retained-source numerical controls and a new 35-prefix study remain
pending for final Uzu qualification.

## Scope and method

The private `qualify_forward_v3.py` controller obtains one complete F32
vocabulary readout after each fixed teacher-forced prefix, at temperature
1 and context capacity 4,352. Reference and candidate load sequentially,
use identical tokenization for each model and retain each model's existing
quantization. There is no speculation. Every score request starts a fresh
inference decision. The vocabulary has 248,320 entries in both cases.

Qwen prefixes are 32 through 256 in increments of eight, then 512, 1,024,
2,048, 2,051, 2,052 and 4,096 tokens: 35 positions total. The ordinary-model
prefixes are 32, 128 and 512 tokens. Both studies use the same pinned source
corpus, with each model's tokenizer. Seven Qwen reference readout hashes
must match the previously sealed anchor map. Artifact file identities are
checked before and after the study; configuration, tokenizer, conversion
receipt, corpus, controller, helper and worker hashes are recorded.

Full vocabulary vectors, probabilities, token IDs and corpus content remain
in memory. Private artifacts contain aggregate errors, distribution
summaries and hashes. Scoring request/output limits are widened only in
private source copies so a complete vocabulary can be requested; this
overlay is absent from the production protocol. Profiling environments are
removed at worker launch. These runs supply no performance evidence.

## Observations

KL is reference-to-candidate divergence in nats, computed from normalized
full-vocabulary distributions. Mean NLL is the arithmetic mean at only the
selected next-token positions. It is not corpus perplexity.

| Aggregate | Qwen3.8, 35 positions | Ordinary model, 3 positions |
| --- | ---: | ---: |
| Finite readout pairs | 35/35 | 3/3 |
| Exact full-readout F32 bit matches | 0/35 | 0/3 |
| Top-1 agreements | 34/35 | 3/3 |
| Mean KL, nats | 0.0296089777971 | 0.0000776134313506 |
| Maximum KL, nats | 0.266117972269 | 0.000208324251282 |
| Maximum total variation distance | 0.287455208631 | 0.00457053004507 |
| Minimum top-20 overlap | 14/20 | 20/20 |
| Reference selected-position mean NLL, nats | 3.08491805136 | 1.63332702760 |
| Candidate selected-position mean NLL, nats | 3.08613990991 | 1.63470614719 |
| Candidate minus reference mean NLL, nats | +0.00122185855505 | +0.00137911959189 |

The precision differences include those described in the
[Uzu precision policy](../../QWEN38_UZU_EXECUTION.md#precision-and-sampling),
including BF16 HC normalization staging and different prefill arithmetic.
This experiment also contains the later-rejected selective GEMV changes.
These measurements do not isolate the contribution of any individual
operation. A small mean NLL difference does not bound the worst-prefix
distribution change, whose observed KL is recorded above.

There is no predeclared quality threshold for this bounded corpus study,
and no accuracy pass is claimed. These observations cannot establish broad
model quality, sampled-generation parity, long-context behavior beyond the
tested prefixes, request recovery, performance or normal desktop behavior.
The three ordinary-model positions provide limited shared-backend
regression evidence. They do not qualify every supported model or precision.

## Source and binary identities

The candidate source label is `exact-gemv-qualified-v3`; “qualified” is its
historical experiment label. The private scoring overlay changes only
`crates/stark-score/src/scoring.rs` between the recorded before/after source
manifests. The receipt's production and private manifest digests identify
this experiment and do not identify the later restored working tree.

| Identity | SHA-256 |
| --- | --- |
| Experimental score worker | `e2b9180f07813b9c750c9ad8bee669901aa9448cbfc2403586a91a4a9cd33d84` |
| Sealed reference score worker | `1c404543d1b2e5371d8b602edbeb9190b58f563022634be5a995c6f3db3d5ef6` |
| Candidate receipt production source-manifest digest | `db56912b76015d4080cb61504ec2d8a856da6ac52d807935aad88451e8cc65fb` |
| Candidate receipt private source-manifest digest | `b60ec0100a295ea53b8b6614740e522b9dfbd058bda5f50c7236798cbfd5f438` |
| Candidate private scoring overlay | `105135533820467d1565876057658cc605fd0dfeb7e72df8c6113ec48b926048` |
| Reference native source-manifest digest | `1161305cb9c6b52787658f31a65acbf3cd2d5121b60242d10d552ce13eb8b616` |
| Controller | `31480e0ae2826e25daf0e8741a6a15acaa56835e4e6a612b87c0680fe00b9b42` |
| Helper | `8253f1b6ee6987ef1bfc6766a2a21cf64ebd2a90fde0d78125319eb140106ffb` |
| Source corpus | `daecdf178c6429a721f9e1b5ae379c664adb8f8af55b381535355636ec469955` |
| Qwen configuration | `cd9fef82e5016e282dc1920600776a1fee5c3560b86bc7787b883b83c732410e` |
| Qwen conversion receipt | `fe495b0be28cb114d042e98f9e1451be6118ac6d28b14b7b73bfae688db2f302` |
| Qwen tokenizer | `87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4` |
| Ordinary-model configuration | `9e8e47e82f80cc37a0b318115df950f183a5e9823e57e36829647113201d357a` |
| Ordinary-model conversion receipt | `aa6d01e21056e1bb7e7cee6a6e8f49aa6d568221d4d4c7c568deb12ec64472a8` |
| Ordinary-model tokenizer | `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523` |

Private aggregate files are preserved under ignored `runtime/`:

| Record | SHA-256 |
| --- | --- |
| `full-vocab-control/exact-gemv-qualified-v3-forward-35-v1.json` | `3f06d5ed0d30bff75d1c3a0092e1e082b2b50910c804bc30128c43cbec08b317` |
| `full-vocab-control/exact-gemv-qualified-v3-ordinary-3-v1.json` | `de5cf1204fb191031bc184def28cf849746a533dbf3ec62074be19156861f7df` |
| `full-vocab-control/exact-gemv-qualified-v3-source-receipt.json` | `3445a41dc080c15b2f6c67868c5ea6c97aa575293d820b795021c7cc56133354` |
| `full-vocab-control/exact-gemv-qualified-v3-source-before.json` | `77a753a31bb4b39275220cb107f544f8647cc336eb08a64dc3878e8171da02aa` |
| `full-vocab-control/exact-gemv-qualified-v3-source-after.json` | `d0704f6692db64656eed9e9916eb84289b3ef13d4b274d965a02b7c36ac95e4f` |
| `full-vocab-control/reference-build-seal.json` | `de9d34d5338683c2fd07974b153e26c4dcdec36a76bb2ce34c6f753670a14cb1` |

Final architectural and numerical qualification must use new immutable
source/binary evidence for the retained implementation. This completed
experimental study leaves that requirement open.
