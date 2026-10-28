# Native hybrid full-checkpoint validation

Date: 2026-10-28. Branch: `codex/native-glm53-qwen38`.
Host: Apple M5 Max, 128 GiB, macOS 26.7.1.
The correction source is the commit introducing this record, following
implementation checkpoint `8a0dea5246a34723ba9db2578ef21432bf584d8d`.

This record supplements the
[initial implementation validation](2026-10-28-native-hybrid-implementation.md).
Production checkpoints are the pinned sources listed in
[native hybrid models](../../NATIVE_HYBRID_MODELS.md). Complete local weight
files were checked against the Hugging Face LFS SHA-256 values at those exact
revisions. Models and private captures remain ignored.

## Correction exposed by the production GLM run

The first GLM generation failed at the first routed MoE layer. The initial
loader assigned IQ2_XXS and IQ2_XS the wrong GGML numeric IDs. Its synthetic
fixtures used the same assignments, so matching decoded values did not prove
format identity. The real checkpoint also uses IQ3_XXS.

The corrected GGML identities are `16 = IQ2_XXS`, `17 = IQ2_XS`, and
`18 = IQ3_XXS`, with 66-, 74-, and 98-byte blocks respectively, each encoding
256 values. CPU and Metal implement all three formats. The reference generator
now obtains enum IDs and block sizes from the pinned C headers. GGUF admission
rejects the separate internal Safetensors integer tags, and packed Metal
dispatch rejects unsupported types before submitting a command.

The original packed-codec results remain recorded against their source
checkpoint; their enum-identity claim is superseded by this correction.

Independent header coverage verifies all 1,412 production tensors: each encoded
span ends at the next aligned tensor offset, and every shard ends exactly at
EOF. The private aggregate coverage receipt has SHA-256
`4538d930db9ca5cb88676b4de0ffc3eded123ed3372e69f81e8055dc037c150d`.

## Corrected reference and worker checks

The corrected GLM fixture uses IQ2_XS gate/up experts and IQ3_XXS down experts,
with C-derived type IDs and block strides. Both profiles use four mHC streams
and 32 sparse-indexer heads. Each profile compares 94 positions across three
sequences and reset replays, with no ambiguous sparse cutoff ties.

| GLM profile | CPU maximum error / relative L2 | Metal maximum error / relative L2 |
| --- | --- | --- |
| Pool 2, sparse budget 4 | `2.18153e-5` / `1.81105e-6` | `3.30210e-5` / `2.36188e-6` |
| Pool 4, sparse budget 8 | `4.99487e-5` / `3.38414e-6` | `3.38554e-5` / `2.87444e-6` |

The corrected GLM worker fixture passes EOS union, tokenization/decode, four
greedy comparisons, A/B/A request isolation, strictly consecutive score-cache
behavior, invalid-request recovery and clean shutdown. Its maximum scoring
logit error is `1.43051e-5`. Private fixture GGUF SHA-256 is
`234361135393c4d5de973fb87ed453e21bb815e8b4b21e99b76a49ea93a80c02`;
first-sequence reference SHA-256 is
`1dab482f8aba58eac1eac5461813776490c9d2bfde711f34efe11d12dbe14e3b`.

Qwen's grouped mixed/BF16 fixture and four-stream variant also pass the same
worker checks on the final binaries. Maximum scoring errors are `2.53320e-7`
and `1.78814e-7`, respectively, with four out of four greedy comparisons in each
fixture. Four-stream CPU and Metal complete-forward comparisons cover 13
positions, with maximum errors `3.57628e-7` and `2.38419e-7`, exact reset/failure
recovery, and 13/13 greedy agreement.

Final release worker SHA-256 identities:

- Generation: `10b0956567edfdc8a6074c6161a4745e3f280f330aabf9c0dac461ce45e36b82`.
- Optimized scoring: `f3dab9a94e8804ad72193df835250f1bff0e9dcf4a394f93aa71d7f53a56cf06`.

The GGML enum/size regression fixture has SHA-256
`6f4a7869dbc8fe10b8cc2ae063d8b30515c19a13d1ab6eee6d94c376124c8455`.
The corrected GLM reference generator has SHA-256
`4b46adf838f3361a9c7a750767d653146d0f8e09795c2eaea7297daa43779d87`.

## Execution scope

The production smoke exercises the real generation and scoring worker
protocols: source chat-template rendering, tokenization, eight greedy output
tokens, nonempty decode, exact request replay, invalid-request rejection with
subsequent recovery, and finite full-vocabulary readout with normalized
candidate scores. Neither reference runtimes nor another inference server
execute these requests.

These are short native text smoke tests and small-model numerical comparisons.
They do not establish model quality, decision calibration, BF16 runtime parity,
long-context memory limits, or installed-app qualification. RSS measurements
refer to the worker process sampled every 50 ms, not total system or GPU memory.
Sequential prefill and compressed uploads per matrix call remain performance
limitations.

## Final regression checks

| Check | Result |
| --- | --- |
| Full physical CPU/Metal engine suite | 603 passed, 33 ignored; 7 CXX tests and 3 documentation tests passed |
| Release generation worker | 43 passed |
| Release optimized scoring worker | 19 passed, repeated by the digest binary; experiment protocol test passed |
| Preparation helper tests | 7 passed |
| Release worker Clippy with warnings denied | Passed for each worker built separately |
| Pinned Rust formatting, reference-generator Ruff/format, repository policy, whitespace | Passed |

The initial checkpoint's complete Python suite (4,962 passed, 88 skipped), menu
bar controller/build checks and distribution checks remain applicable; this
correction changes packed decoding and the independent GLM reference generator.
The final private engine test log has SHA-256
`cb588967cc7e0596bd7f8f961a4d60c4c9b5cb58be10b201301bec152d6f9387`.

## Full GLM checkpoint

All four shards, totaling 108,720,071,552 bytes, match their pinned LFS hashes.
Weight-manifest SHA-256:
`ddd66fd67819dd0ccdb5627f7d9a701c18d9c52d0092b72a0f5a1f3a04e0acd4`.

The final workers pass generation and scoring with 14 input tokens and eight
greedy output tokens. Decode is nonempty, replay is exact, invalid requests are
rejected with subsequent recovery, and scoring produces finite readouts with
normalized candidate scores. The completion hash is
`e6711425acbbce6cf2fda8021560d166cac4afc9b5cea670fa673683595b2de5`.

The first request took 76.97 seconds and its repeat took 50.26 seconds. Reported
decode rates were 0.31 and 0.43 tokens/second. Scoring took 32.65 seconds.
Sampled peak generation RSS was 33,186,037,760 bytes; scoring RSS was
26,690,387,968 bytes. The engine regression suite ran concurrently, so these
timings are smoke observations rather than controlled throughput measurements.

The private aggregate GLM validation report has SHA-256
`1f5b5ee84f67eaced2ac846051b462a6c15943cc6906fd96d6da81c7278b4957`.
The privacy-preserving production protocol probe has SHA-256
`aa4ca97f0c4092bbec15cd5d7301a2623c33943d49129fdc6043cd1b7d9339d2`.
It records only aggregate results and hashes, without publishing prompts,
responses, token vectors or candidate scores.

## Full Qwen checkpoint

All 11 shards, totaling 106,195,031,237 bytes, match their pinned LFS hashes.
Weight-manifest SHA-256:
`0113db1a74552eeca5328c94a2028ea40a80ae6610b2046f525c09c655a0cc79`.

The final workers pass the same generation, replay, recovery and scoring checks
with 14 input tokens and eight greedy output tokens. The completion hash is
`8a17fc0dc61e8520bdbe3a735b000358a6476cbe9f0e3d86c54a51cf26b5d009`,
also matching the earlier worker build's completion.

The first request took 27.72 seconds and its repeat took 23.11 seconds. Reported
decode rates were 0.83 and 0.92 tokens/second. Scoring took 16.27 seconds.
Sampled peak generation RSS was 14,776,434,688 bytes; scoring RSS was
12,505,219,072 bytes. These short runs are not a sustained throughput benchmark.
The private aggregate Qwen validation report has SHA-256
`55c8662d6d1f7fcce4c46bd107814a8f8bbd0a6a707c9fac15242b95d757dc7d`.

## Prepared local artifacts

The validated original-format directories are `runtime/native-hybrid/glm` and
`runtime/native-hybrid/qwen38`, relative to this checkout. They remain ignored
and can be passed directly to the terminal client or imported through the local
folder flow. The two weight sets occupy approximately 200 GiB together.
