# Independent conversion milestone 2

Date: 2026-10-11. Independent review accepts the scoped G3 conversion, named-build load-only, and memory evidence. Broad latency qualification remains open.

## Delivered scope and coverage

Version 0.3.0 adds Gemma2/3 text, tied Phi-4-mini/Phi3 with partial LongRoPE, Granite, and SmolLM3 adapters. SmolLM2 and the remaining uncompressed Qwen/Llama catalog rows use their reviewed dense operators with explicit metadata profiles. Qwen3 now supports all nine uniform BF16/F16/F32 source/target combinations. Mixed source dtypes inside one fused group remain explicitly unsupported.

The reproducibly exported inventory remains 110 built-ins. Implemented coverage rises from 19 to **33 built-ins**, with **four separate extensions**: Unsloth Llama, Mistral v0.2/v0.3, and Google Gemma2. The pin contains a Gemma2 config/loader but no Gemma2 registry entry; its extension uses an independently constructed standard LanguageModelSpec. Thirty-seven profiles are implemented; 77 built-ins still need adapters. Implemented profiles without exact qualification remain unqualified, including Rnj, Nanbeige, Polaris, Reka, DeepSeek, larger Granite variants, and other untested sizes. No family-wide qualification is implied.

Gemma2 records tied embedding scale, four normalization points, alternating attention windows, GELU, and softcaps. Gemma3 text additionally records Q/K norms, periodic local/global attention, and explicitly supported unscaled/linear/YARN global RoPE. Phi reorders gate/up through contiguous row views without materializing a matrix. SmolLM3 preserves the reference RoPE mask. Granite scales residual projections and final normalization in source precision before output casts, including pinned JAX CPU reciprocal/underflow behavior. Frozen arithmetic hashes cover every BF16/F16 bit pattern and F32 edge/random inputs.

Metadata adds Granite's parser and `thinking` field with reasoning off by default, explicit single/double-newline reasoning boundaries, and static inline format defaults selected by reviewed role/hash pairs. Literal defaults are runtime format code; public catalog/evidence keeps their hashes. Smol full-attention annotations must agree with the explicit mask/layout. SmolLM2's browser-export dictionary is packaging metadata ignored by the pinned importer; tensor/config validation still determines the actual output layout. No checkpoint-provided code executes.

## Exact qualified artifacts

Every new BF16 tuple below passed exact tensor names/shapes/dtypes/payloads, matrix specs/logical metadata, complete root/nested config, generation/codec settings, and normalized tokenizer comparison against externally generated pinned-reference artifacts. The named Stark worker loaded each artifact with readiness followed by EOF and zero scoring calls.

| Checkpoint | Immutable source revision | Output tensors |
| --- | --- | ---: |
| google/functiongemma-270m-it | `39eccb091651513a5dfb56892d3714c1b5b8276c` | 182 |
| google/gemma-3-1b-it | `dcc83ea841ab6100d6b47a070329e1ba4cf78752` | 262 |
| google/gemma-2-2b-it (extension) | `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8` | 210 |
| HuggingFaceTB/SmolLM2-1.7B-Instruct | `31b70e2e869a7173562077fd711b654946d38674` | 146 |
| HuggingFaceTB/SmolLM3-3B | `a07cc9a04f16550a088caea529712d1d335b0ac1` | 218 |
| ibm-granite/granite-3.3-2b-instruct | `707f574c62054322f6b5b04b6d075f0a8f05e0f0` | 242 |
| microsoft/Phi-4-mini-instruct | `cfbefacb99257ffa30c83adab238a50856ac3083` | 194 |

The [native qualification catalog](../../../tools/converter/stark_convert/native_qualifications.json) now contains eleven exact tuples, including the four from milestone 1. `--require-native-support` matches registered identity, BF16 dtype, source-manifest, weights, config, and tokenizer hashes and returns the named-build receipt. It does not qualify arbitrary builds, other variants, or execution. F16/F32 remain conversion-only and runtime-blocked.

Twenty-six full synthetic artifact comparisons also passed: three Phi, eight Gemma2/3, six Granite/SmolLM3, and all nine Qwen3 source/target dtype combinations. All tensor/spec/config/tokenizer fields matched external pinned-reference output; transformation workspace stayed within 96 bytes. Seventeen complete frozen decoder-config variants and 126 scalar cases provide additional model-free regression evidence. No model inference was run.

## Latency work and measurement protocol

One delegated agent replaced serial allocated reads with reusable 256 KiB digest buffers and at most four workers for independent large shards. Initial resolution/replay and final publication still perform full SHA-256 verification. Every file signature is checked before and after the whole hash batch; root/index selection is rechecked after reads. All workers join on failure or interruption, and exceptions remain sanitized. The digest-buffer budget is at most 1 MiB on pinned CPython 3.13.13, separate from the 8 MiB transformation workspace and process RSS.

A second agent measured reusable writer buffers across eight alternating 256 MiB tensor runs. The approximately 0.95% median difference was within variation; the proposed writer optimization was discarded. Exploratory source-hash measurements during ongoing work suggested a larger sharded-model benefit and are not the official complete-command result.

Before official timing, the protocol fixed three repeats, warm source files, alternating backend order, fresh direct Python processes, exact artifact parity, transformation workspace at most 8 MiB, and observed candidate tree RSS below 1 GiB. Hypotheses were lower candidate complete time than the previous converter for Qwen3/Mistral, lower time than external Lalamo, and at least 50% lower median RSS than Lalamo. Nine checkpoints are measured serially with other model-dependent work paused. Setup, downloads, warming, and parity checks are outside timed intervals; receipts bind source/tooling identities and mutations invalidate evidence. Redundant timed artifacts are removed only after their hashes/parity/receipts are retained; one independent reference and qualified output per checkpoint remain private.

The baseline is the archived converter at `3b069795`, source hash `f794a77cee3c8339d7806bd070a0a18389e5e3b973a4b42f7a37225d75bcaa96`, using the same resolved converter dependency versions. The final candidate source is `0a4a17a1273b0387b0a77ee019473a366726dd286e7eacbd7841ea18b1c55e15`. Lalamo uses the external pinned CPU reference environment. Its source verification/receipts are weaker, so comparisons measure these complete commands and do not isolate equivalent tensor transforms. Ten-millisecond RSS sampling can miss peaks or count shared pages repeatedly. A strict total-process ceiling is not implemented.

Host remains Darwin 25.6.0 arm64, 18 CPUs, 128 GiB RAM; managed Python 3.13.13. Converter dependency versions are unchanged from milestone 1; the lock changes only the companion package version.

All nine checkpoints passed hard acceptance and the memory hypothesis. All 60 timed-output artifact comparisons passed, including 27 final-source candidate conversions. Source, reference, baseline, and candidate hashes remained unchanged. The latency hypothesis against Lalamo passed only for FunctionGemma; eight other tuples remain slower.

| Checkpoint | Stark median (range), seconds | Lalamo median (range), seconds | Median tree RSS, Stark / Lalamo MiB | RSS reduction |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 1.835 (1.822–1.883) | 1.234 (1.178–1.308) | 187.6 / 1888.0 | 90.06% |
| Mistral-7B-v0.3 | 9.790 (9.687–10.421) | 6.305 (4.432–6.519) | 81.5 / 14419.1 | 99.43% |
| SmolLM2-1.7B | 3.777 (3.772–3.914) | 1.622 (1.604–1.692) | 76.1 / 4002.9 | 98.10% |
| SmolLM3-3B | 6.032 (5.988–6.212) | 2.628 (2.477–2.993) | 271.6 / 6928.2 | 96.08% |
| FunctionGemma-270M | 1.047 (1.044–1.051) | 1.338 (1.324–1.364) | 464.2 / 1522.3 | 69.51% |
| Gemma2-2B (extension) | 5.707 (5.659–5.791) | 2.406 (2.374–2.585) | 268.3 / 6615.2 | 95.94% |
| Gemma3-1B | 2.576 (2.568–2.584) | 1.672 (1.670–1.787) | 463.9 / 3150.4 | 85.27% |
| Granite3.3-2B | 6.132 (6.039–6.191) | 2.079 (2.013–2.127) | 96.7 / 5413.0 | 98.21% |
| Phi-4-mini | 6.656 (6.537–6.849) | 3.375 (3.269–3.527) | 209.0 / 9104.4 | 97.70% |

The controlled old-converter comparison met both non-regression hypotheses. Qwen3 changed from median 1.851 s (1.841–2.407) to 1.835 s, approximately 0.9%, within observed variation. Mistral fell from 15.663 s (15.629–18.988) to 9.790 s, **37.5% lower observed median**. This is the meaningful latency improvement attributable to the revised sharded-source path; single-shard improvement is not established. FunctionGemma's 1.279× complete-command ratio applies only to this recorded tuple/protocol. No general Lalamo speedup is claimed. Source-bound results from milestone 1 remain unchanged.

## Source-bound evidence

| Input/evidence | SHA-256 |
| --- | --- |
| final converter sources | `0a4a17a1273b0387b0a77ee019473a366726dd286e7eacbd7841ea18b1c55e15` |
| final nine-checkpoint benchmark aggregate | `bebac6881dee0a75bf2f60ea359b955abcfe9651188a66a3cbf914c475a755e6` |
| converter lock | `c7a77fc0e6451524d8276c3aba584929ef0e0907a2f255ad1f160af69ac7e0d5` |
| reproducibly exported catalog | `fd6d280caa8c36f48bd055346011af806e4fa57f94eecd584aa7b539f3712e5e` |
| eleven native qualifications | `1e8aeadd0088204b60db03d79d0e5bd7f63491156007706790b1b182a23aa1a4` |
| seven initial qualification aggregates | `842229ff5c3898faf50bbf52ec845521ae39e6e1eb8db57e25346340678b8f82` |
| 26 complete synthetic comparisons | `e116e87fb51995097a4b200068095b2684f7ab4e57c6abb37e8abbf7e1f5a749` |
| Gemma frozen decoder configs | `ba72b6444b9935ea0048d8cc12a2f2d320b132a5d9339dcec527c55830eee6d0` |
| Phi frozen decoder configs | `fef74530d03b4cad55e6ce3ef8395120ac01bc3b24a7c2f0da22790fcc148d34` |
| Granite/Smol configs and arithmetic oracle | `dedbeb00105959f272dc1d931a56b7ef64cd89fce908e4a483308829f488cf6e` |
| external reference runner | `87c83da3fa6a57c4ad0dc42a967eee2288fd294352735b677045f68e62ca378d` |
| official benchmark coordinator | `f5073d1230ab2a2002b73dd775f9a963762e2a98b9c18dd23b70ebfd5ad458cf` |
| worker | `61bb81a83fa368b7c8f647469ac468048cf6868d973ed74e7d9d62633bfba96d` |
| native source manifest | `0cc926bcf73db3921eab03f127185626b7f4311b73c62dad79b407c642d0c9af` |
| Cargo lock | `65444f3e0e0707c43521539e43c95365cfc6a51ec6f9e90180bc43bc8fb4603e` |
| Rust toolchain | `72da9f9e503464ba7393560f6f51a69fb06be72633d52ae7725fa0ba59dfda22` |

Initial untimed real qualification receipts retain three implementation hashes: `957336cccb8bbb80dbde1ee02af6aab35f68267cf67bbbc3f3c36dcfb9361de8`, `10af38234b41fa0e0b70378eb7ef4c031447317d4956da64c4843362c6f0c122`, and `e89fa9c8ada55ea823d3802e7b39ce7bcfb71ac0dcb7f6744713adabe5585e45`. Their aggregate names each original source version. Synthetic validation binds the final source hash. Adding native qualification records changes converter provenance without changing the emitted artifact tuples; final timed runs must reproduce those exact qualified artifacts. Earlier tool-mutation aborts published no output. A synthetic Smol fixture initially supplied conflicting template sources; the corrected fixture passed, and the initial aggregate is retained separately. Earlier milestone evidence is unchanged.

## Validation, review, and remaining scope

- Final suite: **858 passed, 26 unsupplied metadata-reference profiles skipped**, comprising 847 model-free cases and eleven supplied independent real-reference metadata checks.
- Ruff lint/format, converter lock, repository policy, 17 advisory-report tests, and actionlint with the existing disabled-job exception passed. Existing CI remains model-free/offline; no hosted run is claimed here.
- Locked Rust 1.98.1 release/default-plus-optimized build passed using cached artifacts. The existing debug-info stripping warning persists; executable/native input hashes remain identical to milestone 1.
- Independent [technical review](MODEL_CONVERSION_MILESTONE_2_REVIEW.md) assesses exact operators, metadata, source integrity, buffer bounds, promotion status, and source-bound latency results. Parallel agents used separate ownership; model-dependent qualifications/timing ran serially.

No Rust, Metal, Cargo dependency, runtime format, scoring path, or inference behavior changes. Existing converted Jev/JevK5 artifacts retain their execution path. No Jev conversion, inference latency, judge-quality, or broader execution qualification was performed. Quantized conversion, multimodal/newer Gemma, untied Phi3, state-space/hybrid/MoE/classifier/TTS work, total-process memory enforcement, and public distribution remain later scope. Lalamo remains external reference tooling only; no runtime fallback is restored.

Operational inputs, models, and permitted aggregate evidence remain beneath ignored private storage. Public evidence includes identities/revisions, aggregate outcomes, and hashes. No prompts, responses, token vectors, decision probabilities, credentials, exception dumps, or native execution captures are retained.

Pinned Lalamo reference remains `831accb8252a6b6606c378048caeb44178f1a050`, MIT, Copyright (c) 2025 Mirai Tech Inc.; [license](../../../tools/converter/LICENSE.lalamo) preserved. Adapted sources include the attributed `huggingface/{gemma2,gemma3,phi3,granite,smollm3,qwen3}.py` config files, `loaders/huggingface.py`, codec/model specs, normalization/linear/embedding/RoPE contracts, and the frozen external CPU dtype/arithmetic oracle. Uzu base remains `8d01581474760bd1fc7f4a547ff1d02f7dada437`.
