# Generation context policy follow-up

| Field | Value |
| --- | --- |
| Attempt | `20261126-CTX-02`, 2026-11-26 |
| Scope | Worker and managed API compatibility, context admission, and Lily prefix restore |
| Implementation | `382eb15aac8da80805dff5008f717658f881de3c` |
| Baseline | PR #203 audit at `f6dc9f3f1`; historical receipts listed in the [inventory](../CONTEXT_SIZING_INVENTORY.md#installed-generation-receipts-from-the-audit) |
| Production worker | Rust 1.98.1, release, locked, `hy4` enabled, SHA-256 `89b4b635b4802d2bf4cb44657ec6dd4ab22df77168045f2b70f4521ad4adfc1b` |
| Budget-injected worker | Same source plus an isolated runtime available-memory override, SHA-256 `6407de14e303f1306f5ae7e2a8bddf9b1aa9f3e18b3971ca446fe5d0fcedde76` |
| Host and lock | Apple M5 Max, 128 GiB, macOS 26.7.2 (25G332). Cargo.lock SHA-256 `07b6d42497825a651719685a28a107bfc0a32119b8e8b8adeab2d6c01a96695b` |
| Disposition | Implementation pushed to the follow-up branch. Runtime results apply to the named worker and exact local receipts. |

## Production worker and managed API

Each row used one worker at a time. The raw worker completed a short greedy request. Finite rows rejected a request with `context_tokens` one past the selected window, then reproduced the first result. The prefix controls compared fresh and warm output digests at one boundary. The isolated Desktop host used the same candidate binary and queried authenticated `/v1/models` under an OpenCode client grant before and after valid requests and worker overflow. The host retained the worker throughout.

| Artifact | Candidate config SHA-256 | Selected / managed context | Sizing budget | Result |
| --- | --- | --- | ---: | --- |
| Local Qwen3 dense, registered `qwen3-dense` | `dfa5763dc220e871b86a6119536f5c1c2862a5137a473eaf9d812be3cc53924e` | 40,960 / 40,960 | 112,521,756,672 B | Checkpoint bound. Overflow and recovery passed. 1,024 reused. |
| Local JevK5 | `9e8e47e82f80cc37a0b318115df950f183a5e9823e57e36829647113201d357a` | 262,144 / 262,144 | 103,728,562,176 B | Checkpoint bound. Overflow and recovery passed. 1,024 reused. |
| Local Qwen3 4B affine4 | `5b505594bf00ef9682b373c6a292b88f94f1e1ceab449784c12accb835806c7f` | 40,960 / 40,960 | 108,865,044,480 B | Checkpoint bound. Overflow and recovery passed. 1,024 reused. |
| LLaMBA 1B affine4, `cartesia-ai/Llamba-1B-4bit-mlx@24fcab78470f75b4f9082fc4caa74baf1699ee7e` | `dbae5c4611302c0b318fd5aeda93beb61bb2eab1669c9aab3271f62ee4b07b42` | `u32::MAX` sentinel / `null` | per request | The worker advertised `undeclared`. An 8,192-token planning ceiling returned a finite request plan, and repeated text succeeded. Desktop published `context_capacity_known: false`. Prefix reuse and MTP were unavailable. |
| Retained-image Lily Q4-XL | `2a91155be52ebbb0f6dff20e61dcd8ae7484154a56b4afe067246c59da46f899` | 262,144 / 262,144 | 39,652,327,424 B | The managed listing remained stable through overflow, recovery and a synthetic image request. The separate raw worker verified the midpoint contract below. |
| ddalcu MLX-Serve, prepared retained-image clone | `84a69b2bcb20a57b033c3827bc98fa21a5db4d18b55e0640dd3687c99a1b765f` | 262,144 / 262,144 | 39,223,984,128 B | The managed listing remained stable through overflow, recovery and image generation. The raw worker reused 4,096 tokens in serial and two-draft text, matched the greedy outputs, then generated from an 83-token image-expanded input with drafting off. Text recovered with a cold prefix and the same output. |
| ddalcu MLX-Serve, original text installation | `84a69b2bcb20a57b033c3827bc98fa21a5db4d18b55e0640dd3687c99a1b765f` | 262,144 / 262,144 | 28,080,619,520 B | The raw worker reused 4,096 tokens in serial and two-draft text with matching outputs. Overflow, recovery and stable managed discovery passed. |
| Inkling 2 Mini affine8 with native image tower and DSpark | `a63529b36d0ce7fac8b13705c54463e4b1cbb34936d27eefa6a756ab2bd7d4fd` | 860,160 / 854,016 | 83,310,313,472 B raw, 82,682,871,808 B managed | Each load was memory-bound and held its own window through overflow and recovery. Greedy serial and DSpark output matched with 2,048 tokens reused in both modes. The raw worker generated from a 25-token image-expanded input with drafting off, cleared the warm prefix and reproduced the text output. Desktop also passed its image request. |
| Qwen3-0.6B converted from Q8_0 GGUF | `dfa5763dc220e871b86a6119536f5c1c2862a5137a473eaf9d812be3cc53924e` | 40,960 / 40,960 | 107,576,754,176 B | Overflow, recovery and stable managed discovery passed. The raw worker reused 1,024 tokens. |
| Qwen3.8-27B native vision | unavailable during final pass | — | — | Candidate run deferred. The original audit's vision result remains historical. |

All four finite converted-native rows independently reported a 4,193,279-token KV buffer/index bound. Their checkpoint bound selected the public window. Config hashes above were computed from the `config.json` read before each run. The earlier audit records a field named artifact/config SHA-256. Direct equivalence between those two hash fields is unverified. The LLaMBA conversion receipt SHA-256 is `5163fa9df9330e795b152c375edbadb25440374797727a1820c55959277c8d2e`.

Inkling's retained affine8 target, DSpark head and config matched the pinned [artifact record](2026-11-23-inkling-mini-affine8-default.md). Its M5 backend independently reported a 1,048,576-token bound. The Qwen3 import binds GGUF revision `23749fefcc72300e3a2ad315e1317431b06b590a`, HF metadata revision `c1899de289a04d12100db370d81485cdf75e47ca` and converted weight SHA-256 `d866fa25264aa5244de920525c747d6be8d9afbdb78624f2e09a762750178725`. Its conversion receipt SHA-256 is `0a6d1ec6576f12061b83fb997396ccb9863c75f5192d7ad8047157abd115f726`.

## Controlled memory

Only the instrumented binary read a private mutable available-memory value. The test changed it after startup, kept the same worker running, and restored the measured headroom for recovery. All three finite native artifacts returned `resource_budget_exceeded` at zero sampled headroom. Desktop mapped that to `request_too_large`. The next valid request succeeded, and both the loaded capacity and the listing stayed fixed.

| Artifact | Injected startup headroom | Budget after reserve | Raw fit | Advertised |
| --- | ---: | ---: | ---: | ---: |
| Qwen3 dense | 536,870,912 B | 268,435,456 B | 683 | 580 |
| Qwen3 dense | 8,589,934,592 B | 8,321,499,136 B | 40,960 | 40,960 |
| ddalcu MLX-Serve text | 4,294,967,296 B | 4,026,531,840 B | 106,496 | 90,112 |
| ddalcu MLX-Serve text | 8,589,934,592 B | 8,321,499,136 B | 262,144 | 262,144 |

Both low-budget rows reported `memory` as the limiting factor and both high-budget rows reported `checkpoint`. Each listing retained its startup window after the injected budget was released and during the zero-headroom rejection and recovery sequence.

The retained-image Lily worker also rejected zero sampled headroom with `resource_budget_exceeded`, recovered after headroom was restored, and kept advertising 262,144 tokens. An earlier attempt began with about 30.5 GB of available memory and failed before readiness. The retry ran after host headroom recovered.

LLaMBA affine4 also rejected zero sampled headroom, recovered on the next request and continued reporting unknown capacity in the managed listing. Inkling affine8 did the same while keeping its 862,208-token window. That separate load had its own measured startup headroom.

## Local tests and receipt identity

- 583 converter, Desktop, guided-model, catalog and prefix validator tests passed, with one skipped and 28 subtests passed. The generation worker suite passed 109 tests without Metal and 187 tests with Metal and Hy4 enabled, with six ignored in that build. The release binary passed eight direct process cases, including invalid finite hybrid geometry and the explicit EXL3-disabled refusal. The EXL3 unit cases cover the fixed cap and exact pack dispatch contract.
- Production managed and raw receipt SHA-256 `0ed2a353ce12f0bd820c647d1270c627a079f603fc31f5fa8e49431357cadf1f`. LLaMBA receipt `2fe83760f92b7641850873eb5aafa548c4b56f155340473c77a34e2f9da851d7`. Instrumented pressure receipt `a03fd828043706e41d08d40ea2080c0eb9a5b2e91a10ddcf59c3e9f31b0784de`. Controlled native budget receipt `1a9995b9ca11c80b7a8f4da68fab0b7e4d01774c4fa73e335c122597d740d903`.
- Raw ddalcu text receipt SHA-256 `2e32dbd0b393c542a8c67bd38a73c0a4d9657fed6b6eb5114fc8dad18ad91371`. Managed ddalcu text receipt `90bebb9cafeb4047fc341e71a83e1c9fdeccf56e0f4cf79fdcdd43cb6124580b`. Controlled hybrid budget receipt `5a1a1fca5bcde27ee7c050a908c646bed194c4350fd7bb25895608c2261758ed`.
- Instrumented retained-image Lily pressure receipt SHA-256 `42d314e14b75fddcd5846bc0b6900abc651532d0be9c09e9f4f370218156a305`. An earlier ddalcu text attempt stopped when the guard detected another inference worker. The uninstrumented and instrumented completions above used one model worker at a time.
- Inkling affine8 and the Qwen3 GGUF import share aggregate receipt SHA-256 `7b6572f27bcfb73f21f8670c5d00e4ea63ffdc10b892e41df0577430fb4f5f68`. Inkling affine8 pressure receipt `4c784079b4835ff23be5e52af6cee37a0aa419ec9912364ad81d6982b455f84b`. LLaMBA affine4 pressure receipt `f2a6e4ed3c0a3c73f8e562ca1e3ca35ac28e7da6f0205b5dc56a51d3f3096591`.

## Lily midpoint restore

The retained vision manifest SHA-256 was `18c7bcc1e97d1fefdd361472c2daa59d81d89fcd6f3a61983d995a032b44bfb4`. In an 8,192-token request context, six restore cases passed. At position 2,048, 16 suffix tokens produced zero reuse and 17 produced 2,048 reused tokens. At position 4,096, two suffix tokens produced 4,096 reused tokens. Serial and two-draft MTP outputs matched fresh output in all six cases. The image transition disabled drafting and cleared the warm text prefix. Disabling midpoint mode restored the 4,096-token grid and removed the midpoint-specific readiness fields. The guarded run completed with 33,427,390,464 bytes minimum available and 1 MiB maximum swap growth. [Validator](../../scripts/validate_generation_prefix_cache.py)

A prior pilot was interrupted by a host restart, with approximately 45 GiB of swap in the last sample. The restart cause remains undetermined. The guarded retry above provides the completed result.

Lily midpoint receipt SHA-256 `1691d05430afa242e9cb4d9ffc37015ab062efb2ceb7a5458e9a6c31f3dc3e4f`. Managed Lily receipt `155afa04c872767dd51684234a813de06898f4299574d8f83608d14c92ed54ca`. Managed ddalcu retained-image receipt `bbc304ac953fd7f9a76e11f3a970e444292b1abba5a9638c2588d5cb0049a1e5`. Raw ddalcu retained-image receipt `498af1f3005de53f4c7f578593c03b1b53b440cbe655cbf7a26495652ea35e8b`. The ddalcu image manifest SHA-256 was `2ea0257f3369fbc38bf1646d2c9c64f5504a2eca774fec68257d0a784f68dd7d`.

## Deferred hardware checks

- Inkling original and floating LLaMBA stayed incomplete. Initial Xet attempts produced no usable checkpoint. Direct Hub retries advanced the partial files, then were stopped with resumable stages retained. Inkling affine8 and LLaMBA affine4 each have their own completed results above.
- Native Qwen3.8-27B vision and Hy4 Q3_K require complete matching local receipts. The available historical records used earlier workers. GLM EXL3 requires its pinned pack and separately prepared bridge build. The EXL3-disabled process case passed on this candidate.
- Retained-image results cover the stated lifecycle and parity fixtures. Matched image-loaded performance and generation near the advertised ceilings remain outside this run. [Remaining artifact queue](../CONTEXT_SIZING_INVENTORY.md#hardware-queue-and-artifact-availability)
