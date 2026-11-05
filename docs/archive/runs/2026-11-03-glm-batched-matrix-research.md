# GLM batched-matrix research record

Date: 2026-11-03
Scope: experimental dense-wide, grouped-expert and MLA prefill batching on the
native GLM-5.3-Flash UD-Q2_K_XL artifact.
Source checkout: `e97383623dc528e3bf3275ee52d75c889503a95e` plus the recorded
dirty working-tree patch.
Status: research only; no experimental selector is promoted.

## Findings

The v1 full-model readout receipt covers two fixed prefixes, 129 and 513 input
tokens, with five full-vocabulary rows per prefix: ten rows total over a
154,880-entry vocabulary. The rows were bit-exact against the T8 reference
(zero maximum absolute and relative L2 error). The candidate enabled
`STARK_GLM53_PREFILL_WIDE=64`, batched controls, register KDA, dense-wide,
grouped experts and batched MLA; the reference left those selectors unset.
This is a bounded readout result; it does not establish prefix reuse, lifecycle
behavior, or whole-request qualification.

The aggregate GPU stage profile's `native.glm.prefill.gpu.routed_experts`
counter measured a median 11.212288463 seconds (11.21 s) for the all-selectors
v1 run and 4.115337669 seconds (4.12 s) for the matched stage reference. The
v2 expert run measured a median 28.015389251 seconds (28.015 s). Each median
uses the two measured profile rows after the initial warm-up row. These are
routed-expert GPU stage times, not whole-request latency. The v1 result rejects
promotion on this stage-performance measure; the separate v2 run does not
establish an improvement over v1 or qualify grouped experts.

Model-bound prefix qualification and full-model v2 parity were not run.
No result here supports enabling any experimental
selector by default. The three experimental selectors remain opt-in and the
existing optimized recipe remains the reference.

## Run controls and boundaries

The private diagnostic receipts identify a serialized native-text workload
with greedy sampling, 512 requested input tokens, eight requested output
tokens, context 520, three replays including one warm-up, and an unscreened
host. Their selector snapshots include the experimental dense-wide,
expert-batch and MLA-batch controls along with the existing wide scheduling,
batched controls and register KDA controls. Selector presence records
configuration, not proof that every matrix was admitted. The phase-two
diagnostic throughput samples are a separate workload and must not be
conflated with the matched-readouts result above.

The matched-readouts receipt SHA-256 is
`8afec5bcc7b9607eb8a58057e9421becceb3990669146e15d85b0b9921d49540`.
Its scope explicitly excludes answer-quality and performance qualification.
The model artifact is privately bound by receipt
`7b8a61782577efbc30f6791f26e91d539a0934f14671f82d9548a313ec67903d`; the
configuration hash is
`d1e229307a382fc1deae8e4a48b6ffc2f47fc47769ab8789c023bdbd806072fd`, and the
corpus hash is
`6b3970cf0eed2c317fad045564839e19d052e526a961b209a188c48bd904e510`. Model
files, prompts, responses, token vectors and private evaluation artifacts
remain outside this record.

## Provenance hashes

Hashes below bind the artifacts in the ignored private runtime directory
`runtime/glm-prefill-throughput-2026-11-03/`; they are references for
reproduction, not public copies of those artifacts.

| Artifact | SHA-256 |
| --- | --- |
| v1 grouped-expert source hash manifest | `2a679c27de3110a6f01f1099d2b0a67e1954a07704ad316405dd26921018ccc2` |
| v1 all-selectors matched-readouts receipt | `8afec5bcc7b9607eb8a58057e9421becceb3990669146e15d85b0b9921d49540` |
| phase-two all-selectors diagnostic receipt | `df6737dfd94363e67f686a9d960f684e2aa6d5aa2c7afd1d6978f7fce3e168a7` |
| phase-two v2 expert diagnostic receipt | `66bea432b64d715bae9a6a401cbd9f5c9920afe9ddc56ace97f93f66e8ec77b5` |
| all-selectors GPU profile JSONL | `1ff14aa61e89da6146a38c07a61124026192d50dc25c3fc96e04169d633fb82e` |
| matched reference GPU stage profile JSONL | `f21f98ee3f731bb7fd7c324fb2cd48909473c40ca7e4f96f9170af775bb0ba87` |
| v2 expert GPU profile JSONL | `1c700b562fa1064f1a2c32dd4500febeee11dd22acfcde2b14583d69211d6db2` |
| MLA substage GPU profile JSONL | `db1461772feb3991cfb24617b4667a2f6c663678b49baa8eb6199a6949f1f267` |

The v1 source manifest records these source hashes:

| Source | SHA-256 |
| --- | --- |
| `decode_graph_experts.rs` | `5ab2839dc0fbd83d76801be6bc6f29fd4e17081a99a7492f2ebf300c1fa907bd` |
| `grouped_expert_prefill.metal` | `a3f6fe69bde7e8a10b676e412b86e4d76db2622a531631d977df9bdc2bdf79f7` |
| `grouped_expert_prefill_tests.rs` | `fb42e6b9843bf78d80b02e1d6f1b0a7b81c4857dd93c3f9276924e90f2a7dc6d` |

The phase-two diagnostic receipts bind their workers and benchmark harnesses
independently from the source checkout. GPU profile files use the
`native_hybrid_aggregate_profile` JSONL schema. Each contains three 512-input,
8-output profile rows; the first is treated as warm-up and the reported stage
medians use the other two rows. The measured counter is
`native.glm.prefill.gpu.routed_experts`. These profiles are diagnostics, not
production whole-request throughput runs. The all-selectors diagnostic receipt
reports exact repeated output hashes within its short generation workload; this
does not substitute for full-model v2 parity or prefix restoration
qualification. The matched-readouts receipt
binds the harness checkout to head `e97383623dc528e3bf3275ee52d75c889503a95e`,
tracked-diff SHA-256
`07405d0ca0eeba4a35f6dd8b3d5d96b600f528d2fb36e31e74f20946f2e38a56`, and
hybrid-source-bundle SHA-256
`0f8b43945f60d1e69e0ac166b7fc2344dd1c848728e805a123ed60cd01922829`.
The v1 grouped-expert source manifest identifies the three expert implementation
files by hash in the table above; the broader dirty patch is bound by the
checkout hashes here.

## Remaining work

The final experimental source passed the native Metal suite on this host:
788 tests passed and 86 were ignored, followed by seven integration tests and
three doctests. The benchmark/readout Python checks passed 57 tests; repository
policy and formatting checks passed. These model-free checks have a separate
private receipt, SHA-256
`4acb985f228b2ae49ffb2bb2278896d3e124de82add1c53ecc22895c49466c4a`,
binding the source bundle
`762af63659add1ca4a2d8483e049aacab73dc14a4da7e81d131dfd338a7757d9`
and native-suite log
`72dda612cbe39898967f8a50016877acbaa56860cbd0d93f5f81e34415e335b9`.

Before any promotion, run the complete artifact-bound phase-two comparison for
each selector independently and for the proposed combination, establish
full-model readout and generation parity for the exact source/build, qualify
failure/reset/cancellation and prefix capture/restore behavior, and measure
whole-request prefill on a screened quiet host. Preserve private prompt and
response data; publish only aggregate findings and hashes.

See [experimental GLM batched matrices](../../validation/GLM_BATCHED_MATRICES.md)
for implementation details, admission boundaries and qualification gates, and
[experimental GLM wide prefill](../../validation/GLM_WIDE_PREFILL.md) for
the wider scheduling and KDA controls.
