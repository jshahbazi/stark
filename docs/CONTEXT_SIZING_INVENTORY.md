# Generation context artifact inventory

Snapshot reconciled on 2026-11-26 against the shipped conversion registry and the artifact audit of `f6dc9f3f1` (v0.24.2). Use the [follow-up plan](CONTEXT_SIZING_FOLLOW_UP.md#4-repair-the-inventory-and-queue-the-remaining-hardware-runs) to schedule candidate execution.

## Shipped conversion registry

The table follows [catalog.json](../tools/converter/stark_convert/catalog.json) order. Rows 1–110 are base entries and 111–120 are extensions. The shipped conversion status is implemented for 94 language models and two classifiers. It is planned for 23 language entries and one TTS entry. Planned rows describe registry entries. An implemented format or image profile still requires exact checkpoint admission and its own candidate-worker check.

The audit provides revision evidence for 104 rows and leaves 16 unresolved. The per-row source labels account for 62 local Hub snapshots, 34 catalog cache entries, seven profile pins and one catalog pin. The audit's aggregate 99/5 split does not match those per-row labels. A cached resolution establishes a historical source commit. The live downloadable item exposes its immutable `revision` before installation, including values resolved by the [Hub metadata check](../tools/converter/stark_convert/model_catalog.py). Inkling additionally exposes `download_revision` for the reachable byte-verified mirror of its qualified pin. Installation receipts identify the converted or prepared bytes exercised on hardware.

`—` in the pin column means the registry and checked catalog policy provide no explicit pin for that row. A source commit and prepared artifact identity are independent. The registry table describes text conversion, so its rows have no enabled tower. Draft state is resolved from the admitted converted artifact and worker readiness, including at the same source commit. The separate preparation and receipt rows below record observed components and artifact identities.

| # | Source repository | Shipped status / generation path | Source format | Shipped pin | Audited source revision | Revision provenance / disposition |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | `EssentialAI/rnj-1-instruct` | planned | safetensors | — | 5ef96d7a82b2bff8009192979432c9bc64e870cd | local Hub snapshot; no admitted catalog path |
| 2 | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | implemented / Native text | safetensors | — | 31b70e2e869a7173562077fd711b654946d38674 | local Hub snapshot; candidate replay pending |
| 3 | `HuggingFaceTB/SmolLM3-3B` | implemented / Native text | safetensors | — | a07cc9a04f16550a088caea529712d1d335b0ac1 | local Hub snapshot; candidate replay pending |
| 4 | `LiquidAI/LFM2-1.2B` | implemented / Native text | safetensors | — | 40f3da0d0164913923aee9462c23077868b816a3 | local Hub snapshot; candidate replay pending |
| 5 | `LiquidAI/LFM2-2.6B` | implemented / Native text | safetensors | — | 36ed799f7024ef169ba92ffe184821835e15cc66 | local Hub snapshot; candidate replay pending |
| 6 | `LiquidAI/LFM2-350M` | implemented / Native text | safetensors | — | f37d3f5c8c5484bc01dad379a595cf4c68c4e70e | local Hub snapshot; candidate replay pending |
| 7 | `LiquidAI/LFM2-700M` | implemented / Native text | safetensors | — | 86f49fc9a3800c3a325b7320bde179c318062583 | local Hub snapshot; candidate replay pending |
| 8 | `LiquidAI/LFM2.5-1.2B-Instruct` | planned | safetensors | — | unresolved | no exact audit evidence; no admitted catalog path |
| 9 | `LiquidAI/LFM2.5-1.2B-Instruct-MLX-4bit` | planned | safetensors | — | 7ccafdb04c36936f4f1c4685198c6c9a40275932 | local Hub snapshot; no admitted catalog path |
| 10 | `LiquidAI/LFM2.5-1.2B-Instruct-MLX-8bit` | planned | safetensors | — | unresolved | no exact audit evidence; no admitted catalog path |
| 11 | `LiquidAI/LFM2.5-1.2B-Thinking` | planned | safetensors | — | unresolved | no exact audit evidence; no admitted catalog path |
| 12 | `LiquidAI/LFM2.5-350M` | planned | safetensors | — | 9e6c6ccf47cd318696e137d381a7ded8fe4df09f | local Hub snapshot; no admitted catalog path |
| 13 | `Nanbeige/Nanbeige4.1-3B` | implemented / Native text | safetensors | — | 9c4555c37921c0982af8fffeee1e0a6e6b953c4f | local Hub snapshot; candidate replay pending |
| 14 | `POLARIS-Project/Polaris-4B-Preview` | planned | safetensors | — | dcab747284e6f4818d8a871315f63dc1223ee394 | local Hub snapshot; no admitted catalog path |
| 15 | `Qwen/Qwen2.5-Coder-0.5B-Instruct` | implemented / Native text | safetensors | — | ea3f2471cf1b1f0db85067f1ef93848e38e88c25 | local Hub snapshot; candidate replay pending |
| 16 | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | implemented / Native text | safetensors | — | 2e1fd397ee46e1388853d2af2c993145b0f1098a | local Hub snapshot; candidate replay pending |
| 17 | `Qwen/Qwen2.5-Coder-14B-Instruct` | implemented / Native text | safetensors | — | aedcc2d42b622764e023cf882b6652e646b95671 | catalog cache; candidate replay pending |
| 18 | `Qwen/Qwen2.5-Coder-32B-Instruct` | implemented / Native text | safetensors | — | 381fc969f78efac66bc87ff7ddeadb7e73c218a7 | catalog cache; candidate replay pending |
| 19 | `Qwen/Qwen2.5-Coder-3B-Instruct` | implemented / Native text | safetensors | — | 488639f1ff808d1d3d0ba301aef8c11461451ec5 | local Hub snapshot; candidate replay pending |
| 20 | `Qwen/Qwen2.5-Coder-7B-Instruct` | implemented / Native text | safetensors | — | c03e6d358207e414f1eca0bb1891e29f1db0e242 | catalog cache; candidate replay pending |
| 21 | `Qwen/Qwen3-0.6B` | implemented / Native text | safetensors | — | c1899de289a04d12100db370d81485cdf75e47ca | catalog cache; linked receipt below |
| 22 | `Qwen/Qwen3-0.6B-MLX-4bit` | implemented / Native text | MLX affine 4 | — | 173234aa840d113125e9f2271100ddbaf16c9620 | local Hub snapshot; candidate replay pending |
| 23 | `Qwen/Qwen3-0.6B-MLX-8bit` | implemented / Native text | MLX affine 8 | — | e53d3ae02ebc79eac17f1fa9e23a2b65ce227893 | local Hub snapshot; candidate replay pending |
| 24 | `Qwen/Qwen3-1.7B` | implemented / Native text | safetensors | — | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | catalog cache; candidate replay pending |
| 25 | `Qwen/Qwen3-1.7B-MLX-4bit` | implemented / Native text | MLX affine 4 | — | 21457c6f51ed54a7c16e988c0844db973815c137 | local Hub snapshot; candidate replay pending |
| 26 | `Qwen/Qwen3-1.7B-MLX-8bit` | implemented / Native text | MLX affine 8 | — | 95400cbada1bef81d2c4fb6f9d2b10aa27f1f1a0 | local Hub snapshot; candidate replay pending |
| 27 | `Qwen/Qwen3-14B` | implemented / Native text | safetensors | — | 40c069824f4251a91eefaf281ebe4c544efd3e18 | catalog cache; candidate replay pending |
| 28 | `Qwen/Qwen3-14B-MLX-4bit` | implemented / Native text | MLX affine 4 | — | ba63a5141812f9870287df53341123a71ba41433 | local Hub snapshot; candidate replay pending |
| 29 | `Qwen/Qwen3-14B-MLX-8bit` | implemented / Native text | MLX affine 8 | — | 7f11bdbf4e522b155526983c77d33771330d5b8f | local Hub snapshot; candidate replay pending |
| 30 | `Qwen/Qwen3-32B` | implemented / Native text | safetensors | — | 9216db5781bf21249d130ec9da846c4624c16137 | catalog cache; candidate replay pending |
| 31 | `Qwen/Qwen3-32B-MLX-4bit` | implemented / Native text | MLX affine 4 | — | ceb4c4aad0164033af0c8dd37a0641693d0d6321 | local Hub snapshot; candidate replay pending |
| 32 | `Qwen/Qwen3-32B-MLX-8bit` | implemented / Native text | MLX affine 8 | — | 5fd4a0905f0e34cc45682cf9e52769681727e38c | local Hub snapshot; candidate replay pending |
| 33 | `Qwen/Qwen3-4B` | implemented / Native text | safetensors | — | 1cfa9a7208912126459214e8b04321603b3df60c | catalog cache; candidate replay pending |
| 34 | `Qwen/Qwen3-4B-Instruct-2507` | implemented / Native text | safetensors | — | cdbee75f17c01a7cc42f958dc650907174af0554 | catalog cache; candidate replay pending |
| 35 | `Qwen/Qwen3-4B-MLX-4bit` | implemented / Native text | MLX affine 4 | — | 52a5ab34fa604bc8af6d3ce0cac0cab10b7eb495 | local Hub snapshot; candidate replay pending |
| 36 | `Qwen/Qwen3-4B-MLX-8bit` | implemented / Native text | MLX affine 8 | — | 315fb4813e1d13d630f4ee43ea06eb03c6150afb | local Hub snapshot; candidate replay pending |
| 37 | `Qwen/Qwen3-4B-Thinking-2507` | implemented / Native text | safetensors | — | 768f209d9ea81521153ed38c47d515654e938aea | local Hub snapshot; candidate replay pending |
| 38 | `Qwen/Qwen3-8B` | implemented / Native text | safetensors | — | b968826d9c46dd6066d109eabc6255188de91218 | catalog cache; candidate replay pending |
| 39 | `Qwen/Qwen3-8B-MLX-4bit` | implemented / Native text | MLX affine 4 | — | 383413e909f3bc5303ce195ebbdf0339c5a1a2a3 | local Hub snapshot; candidate replay pending |
| 40 | `Qwen/Qwen3-8B-MLX-8bit` | implemented / Native text | MLX affine 8 | — | 6a20f8c9329f958e0f2e8e7d92aa002022a9a70d | local Hub snapshot; candidate replay pending |
| 41 | `Qwen/Qwen3.5-0.8B` | implemented / Native text | safetensors | 2fc06364715b967f1860aea9cf38778875588b17 | 2fc06364715b967f1860aea9cf38778875588b17 | catalog cache; candidate replay pending |
| 42 | `Qwen/Qwen3.5-27B` | implemented / Native text | safetensors | — | fc05daec18b0a78c049392ed2e771dde82bdf654 | catalog cache; candidate replay pending |
| 43 | `Qwen/Qwen3.5-2B` | implemented / Native text | safetensors | — | 15852e8c16360a2fea060d615a32b45270f8a8fc | catalog cache; candidate replay pending |
| 44 | `Qwen/Qwen3.5-35B-A3B` | planned | safetensors | — | 59d61f3ce65a6d9863b86d2e96597125219dc754 | catalog cache; separate vision profile below |
| 45 | `Qwen/Qwen3.5-4B` | implemented / Native text | safetensors | — | 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a | catalog cache; candidate replay pending |
| 46 | `Qwen/Qwen3.5-9B` | implemented / Native text | safetensors | — | c202236235762e1c871ad0ccb60c8ee5ba337b9a | catalog cache; candidate replay pending |
| 47 | `Qwen/Qwen3.6-27B` | planned | safetensors | — | 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9 | local Hub snapshot; no admitted catalog path |
| 48 | `Qwen/Qwen3.6-35B-A3B` | planned | safetensors | — | 345d467b117d1fd5eda9b879a5988de9ae5f47a7 | catalog cache; no admitted catalog path |
| 49 | `Qwen/Qwen3.8-27B` | implemented / Native text | safetensors | 37561816e66b5b9153b1b64c10968c18c1e6f4a0 | 37561816e66b5b9153b1b64c10968c18c1e6f4a0 | catalog pin; linked receipt below |
| 50 | `RekaAI/reka-flash-3.1` | implemented / Native text | safetensors | — | dcf8d6ee3219ca177b519cab47777d42864b54bd | local Hub snapshot; candidate replay pending |
| 51 | `cartesia-ai/Llamba-1B` | implemented / Native text | safetensors | — | 212c94fc91c28f607542232da347693dde248063 | local Hub snapshot; undeclared capacity, candidate replay pending |
| 52 | `cartesia-ai/Llamba-1B-4bit-mlx` | implemented / Native text | MLX affine 4 | — | 24fcab78470f75b4f9082fc4caa74baf1699ee7e | local Hub snapshot; undeclared capacity, candidate replay pending |
| 53 | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | implemented / Native text | safetensors | — | ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562 | local Hub snapshot; candidate replay pending |
| 54 | `fishaudio/s1-mini` | planned / TTS | pth | — | unresolved | no exact audit evidence; outside chat generation |
| 55 | `google/functiongemma-270m-it` | implemented / Native text | safetensors | — | 39eccb091651513a5dfb56892d3714c1b5b8276c | local Hub snapshot; candidate replay pending |
| 56 | `google/gemma-3-1b-it` | implemented / Native text | safetensors | — | dcc83ea841ab6100d6b47a070329e1ba4cf78752 | catalog cache; candidate replay pending |
| 57 | `google/gemma-3-27b-it` | planned | safetensors | — | 005ad3404e59d6023443cb575daa05336842228a | catalog cache; separate vision profile below |
| 58 | `google/gemma-3-4b-it` | planned | safetensors | — | 093f9f388b31de276ce2de164bdc2081324b9767 | catalog cache; separate vision profile below |
| 59 | `google/gemma-4-E2B` | planned | safetensors | — | unresolved | no exact audit evidence; no admitted catalog path |
| 60 | `google/gemma-4-E2B-it` | implemented / Native text | safetensors | 3e22461f65e89153144f8adb70e3b8c2cc9845a7 | 3e22461f65e89153144f8adb70e3b8c2cc9845a7 | profile pin; candidate replay pending |
| 61 | `google/gemma-4-E4B` | planned | safetensors | — | unresolved | no exact audit evidence; no admitted catalog path |
| 62 | `google/gemma-4-E4B-it` | implemented / Native text | safetensors | 4ecc2ba63bcac23ba4a767fc87025b1450f84c9c | 4ecc2ba63bcac23ba4a767fc87025b1450f84c9c | profile pin; candidate replay pending |
| 63 | `ibm-granite/granite-3.1-2b-instruct` | implemented / Native text | safetensors | — | bbc2aed595bd38bd770263dc3ab831db9794441d | local Hub snapshot; candidate replay pending |
| 64 | `ibm-granite/granite-3.1-8b-instruct` | implemented / Native text | safetensors | — | 4009206d5fc95d2e65a7b7633e159d6e97e25d35 | local Hub snapshot; candidate replay pending |
| 65 | `ibm-granite/granite-3.3-2b-instruct` | implemented / Native text | safetensors | — | 707f574c62054322f6b5b04b6d075f0a8f05e0f0 | local Hub snapshot; candidate replay pending |
| 66 | `ibm-granite/granite-3.3-8b-instruct` | implemented / Native text | safetensors | — | 51dd4bc2ade4059a6bd87649d68aa11e4fb2529b | local Hub snapshot; candidate replay pending |
| 67 | `meta-llama/Llama-3.1-8B-Instruct` | implemented / Native text | safetensors | — | unresolved | no exact audit evidence; candidate replay pending |
| 68 | `meta-llama/Llama-3.2-1B-Instruct` | implemented / Native text | safetensors | — | 9213176726f574b556790deb65791e0c5aa438b6 | local Hub snapshot; candidate replay pending |
| 69 | `meta-llama/Llama-3.2-3B-Instruct` | implemented / Native text | safetensors | — | 0cb88a4f764b7a12671c53f0838cd831a0843b95 | local Hub snapshot; candidate replay pending |
| 70 | `meta-models/Muse-Glimmer-30B` | planned | safetensors | — | unresolved | no exact audit evidence; no admitted catalog path |
| 71 | `microsoft/Phi-4-mini-instruct` | implemented / Native text | safetensors | — | cfbefacb99257ffa30c83adab238a50856ac3083 | local Hub snapshot; candidate replay pending |
| 72 | `mistralai/Codestral-22B-v0.1` | implemented / Native text | safetensors | — | 28b1c1a51dabe9d86ca8c41420ada1984632498f | local Hub snapshot; candidate replay pending |
| 73 | `mistralai/Devstral-Small-2505` | implemented / Native text | safetensors | — | c2a9d81a2989af566682b4cecc828c84556076c5 | local Hub snapshot; candidate replay pending |
| 74 | `mlx-community/LFM2-1.2B-4bit` | implemented / Native text | MLX affine 4 | — | 3843e4ad0fcb8b7ed8a050908ac8f0bb5320d1bf | local Hub snapshot; candidate replay pending |
| 75 | `mlx-community/LFM2-1.2B-8bit` | implemented / Native text | MLX affine 8 | — | 9384737a4b9172faf97f4d65d4bf7799bee182fa | local Hub snapshot; candidate replay pending |
| 76 | `mlx-community/LFM2-2.6B-4bit` | implemented / Native text | MLX affine 4 | — | 493071ebca3592c63085e19612ded87192a2a0dd | local Hub snapshot; candidate replay pending |
| 77 | `mlx-community/LFM2-2.6B-8bit` | implemented / Native text | MLX affine 8 | — | bf50a6b1531ef001c22a2b1aefd503b753f8334f | local Hub snapshot; candidate replay pending |
| 78 | `mlx-community/LFM2-350M-4bit` | implemented / Native text | MLX affine 4 | — | 18dc72abf3b2337f9123cfd6eeeb58dfa7947066 | local Hub snapshot; candidate replay pending |
| 79 | `mlx-community/LFM2-350M-8bit` | implemented / Native text | MLX affine 8 | — | 3b5014a1966c54313045fd0a23a098a220083b1e | local Hub snapshot; candidate replay pending |
| 80 | `mlx-community/LFM2-700M-4bit` | implemented / Native text | MLX affine 4 | — | 6b32895938db95f2d40f2d0d70c10117fba5297a | local Hub snapshot; candidate replay pending |
| 81 | `mlx-community/LFM2-700M-8bit` | implemented / Native text | MLX affine 8 | — | cc0f7efcc1c15649dc385f686f6cf142215d1daf | local Hub snapshot; candidate replay pending |
| 82 | `mlx-community/LFM2.5-1.2B-Thinking-4bit` | planned | safetensors | — | unresolved | no exact audit evidence; no admitted catalog path |
| 83 | `mlx-community/LFM2.5-1.2B-Thinking-8bit` | planned | safetensors | — | 35175c7b717f4ee0fef5690940b2c1681039fe8d | local Hub snapshot; no admitted catalog path |
| 84 | `mlx-community/Llama-3.1-8B-Instruct-4bit` | implemented / Native text | MLX affine 4 | — | 90215b22ec18e72f623dde2ea7af4097025160e2 | local Hub snapshot; candidate replay pending |
| 85 | `mlx-community/Llama-3.2-1B-Instruct-4bit` | implemented / Native text | MLX affine 4 | — | 08231374eeacb049a0eade7922910865b8fce912 | local Hub snapshot; candidate replay pending |
| 86 | `mlx-community/Llama-3.2-1B-Instruct-8bit` | implemented / Native text | MLX affine 8 | — | d48cdf0a4ea22d893b7c63a99d6a693e24822795 | local Hub snapshot; candidate replay pending |
| 87 | `mlx-community/Llama-3.2-3B-Instruct-4bit` | implemented / Native text | MLX affine 4 | — | 7f0dc925e0d0afb0322d96f9255cfddf2ba5636e | local Hub snapshot; candidate replay pending |
| 88 | `mlx-community/Llama-3.2-3B-Instruct-8bit` | implemented / Native text | MLX affine 8 | — | ff054899609078569493def2823f9acd2780c0c9 | local Hub snapshot; candidate replay pending |
| 89 | `mlx-community/Qwen3.5-0.8B-MLX-4bit` | implemented / Native text | MLX affine 4 | — | 5d894f8cc4ef3e6c88537bf3746ed262f549da6a | catalog cache; candidate replay pending |
| 90 | `mlx-community/Qwen3.5-0.8B-MLX-8bit` | implemented / Native text | MLX affine 8 | — | 45917392c34ba9b095b192a5acfad126ab764ed9 | catalog cache; candidate replay pending |
| 91 | `mlx-community/Qwen3.5-27B-4bit` | implemented / Native text | MLX affine 4 | — | 45797d2985a12c55e6473686e9ea91b95e959553 | catalog cache; candidate replay pending |
| 92 | `mlx-community/Qwen3.5-27B-8bit` | implemented / Native text | MLX affine 8 | — | d49462a9487464de69e2240d2bd65e50bb7d1cbc | catalog cache; candidate replay pending |
| 93 | `mlx-community/Qwen3.5-2B-MLX-4bit` | implemented / Native text | MLX affine 4 | — | 93760be4f1f69842a46bc13dbdc0f19e291392a3 | catalog cache; candidate replay pending |
| 94 | `mlx-community/Qwen3.5-2B-MLX-8bit` | implemented / Native text | MLX affine 8 | — | e6ffd0033d03c9efa880984e611028e0da63905f | catalog cache; candidate replay pending |
| 95 | `mlx-community/Qwen3.5-4B-MLX-4bit` | implemented / Native text | MLX affine 4 | — | 32f3e8ecf65426fc3306969496342d504bfa13f3 | catalog cache; candidate replay pending |
| 96 | `mlx-community/Qwen3.5-4B-MLX-8bit` | implemented / Native text | MLX affine 8 | — | 5319bbbe4f1cbe6c0b3c80f4f7de4f0338c3906d | catalog cache; candidate replay pending |
| 97 | `mlx-community/Qwen3.5-9B-MLX-4bit` | implemented / Native text | MLX affine 4 | — | 938d8919941c6e7efd3c7150eff7fe9d12afa631 | catalog cache; candidate replay pending |
| 98 | `mlx-community/Qwen3.5-9B-MLX-8bit` | implemented / Native text | MLX affine 8 | — | 84f7c2deea248d8df56240f88102def51c7ed5d6 | catalog cache; candidate replay pending |
| 99 | `mlx-community/SmolLM3-3B-4bit` | implemented / Native text | MLX affine 4 | — | d3a7e0594d6642dbcfb7d149bed8b0bdf49f95ce | local Hub snapshot; candidate replay pending |
| 100 | `mlx-community/SmolLM3-3B-8bit` | implemented / Native text | MLX affine 8 | — | 316f091e34982bc6eaf7f6cc1db82bdb77ac2103 | local Hub snapshot; candidate replay pending |
| 101 | `mlx-community/gemma-3-1b-it-4bit` | implemented / Native text | MLX affine 4 | — | 2d44e83dc9e80843d22fb941d3d699a0b1351aa6 | local Hub snapshot; candidate replay pending |
| 102 | `mlx-community/gemma-3-1b-it-8bit` | implemented / Native text | MLX affine 8 | — | 7b963136f21d05ca8b367c93a9c47a7944ad281a | local Hub snapshot; candidate replay pending |
| 103 | `mlx-community/gemma-3-27b-it-4bit` | planned | safetensors | — | unresolved | no exact audit evidence; no admitted catalog path |
| 104 | `mlx-community/gemma-3-27b-it-8bit` | planned | safetensors | — | unresolved | no exact audit evidence; no admitted catalog path |
| 105 | `mlx-community/gemma-3-4b-it-4bit` | planned | safetensors | — | 93724907d4ed1745d2fe50baadf3b0b01a65abf2 | local Hub snapshot; no admitted catalog path |
| 106 | `mlx-community/gemma-3-4b-it-8bit` | planned | safetensors | — | unresolved | no exact audit evidence; no admitted catalog path |
| 107 | `openai/gpt-oss-20b` | planned | safetensors | — | unresolved | no exact audit evidence; no admitted catalog path |
| 108 | `prism-ml/Bonsai-4B-mlx-1bit` | planned | safetensors | — | unresolved | no exact audit evidence; no admitted catalog path |
| 109 | `trymirai/chat-moderation-router` | implemented / classifier | safetensors | — | unresolved | no exact audit evidence; scoring-only |
| 110 | `trymirai/health-related-router` | implemented / classifier | safetensors | — | unresolved | no exact audit evidence; scoring-only |
| 111 | `Jundot/Qwen3.8-27B-oQ4e-mtp` | implemented / Native text | MLX affine 4/5/6/8 | — | 7dd276b77fcb27c982ebd8d8a434539345713fea | catalog cache; candidate replay pending |
| 112 | `allenai/Olmo-3-7B-Instruct` | implemented / Native text | safetensors | 6e5971d9eba42665f5bd5a0fcf047f299ce1dccc | 6e5971d9eba42665f5bd5a0fcf047f299ce1dccc | profile pin; candidate replay pending |
| 113 | `empero-ai/Qwen3.8-4B-Distill` | implemented / Native text | safetensors | c83cb7aa2999d2f35c43e9ae0634a30eb8985a1e | c83cb7aa2999d2f35c43e9ae0634a30eb8985a1e | profile pin; candidate replay pending |
| 114 | `google/gemma-2-2b-it` | implemented / Native text | safetensors | — | 299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8 | catalog cache; candidate replay pending |
| 115 | `google/gemma-4-12B-it` | implemented / Native text | safetensors | 43c9d014b231474fa7212dbb48a762566c2b255d | 43c9d014b231474fa7212dbb48a762566c2b255d | profile pin; candidate replay pending |
| 116 | `google/gemma-4-26B-A4B-it` | implemented / Native text | safetensors | 4d7ae4984b7db7de8f8457170b3f1a419ee76d52 | 4d7ae4984b7db7de8f8457170b3f1a419ee76d52 | profile pin; candidate replay pending |
| 117 | `google/gemma-4-31B-it` | implemented / Native text | safetensors | 842da3794eaa0b77d5f08bae87a17459d91ff475 | 842da3794eaa0b77d5f08bae87a17459d91ff475 | profile pin; candidate replay pending |
| 118 | `mistralai/Mistral-7B-Instruct-v0.2` | implemented / Native text | safetensors | — | 63a8b081895390a26e140280378bc85ec8bce07a | catalog cache; candidate replay pending |
| 119 | `mistralai/Mistral-7B-Instruct-v0.3` | implemented / Native text | safetensors | — | c170c708c41dac9275d15a8fff4eca08d52bab71 | catalog cache; candidate replay pending |
| 120 | `unsloth/Llama-3.2-1B-Instruct` | implemented / Native text | safetensors | — | 5a8abab4a5d6f164389b1079fb721cfab8d7126c | local Hub snapshot; candidate replay pending |

## Installed generation receipts from the audit

The exact six receipts reached their advertised window, rejected one-token overflow with `context_overflow`, recovered, reused text prefixes and kept the same window in managed `/v1/models`. The two image-enabled receipts passed a synthetic image request and prefix invalidation. All three enabled text draft heads passed the audit's MTP parity and reuse controls. These are finite-window behavior and bounded lifecycle results.

Production worker SHA-256 `8432272c01443f17c69817ef4d9f2dd6eea56faee00759328e634dd6bf40b45a`. Separate budget-injected worker SHA-256 `46e853aec6669fcae4c98cddcfd317da7114a6bfbfc38782108ef55f889e15be`.

| Receipt / source | Receipt revision | Config SHA-256 | Loader / format | Tower / text draft | Advertised tokens |
| --- | --- | --- | --- | --- | ---: |
| Qwen3 dense, `Qwen/Qwen3-0.6B` (row 21) | `c1899de289a04d12100db370d81485cdf75e47ca` | `87c3344a80b7376392cc5c44d5369680b68936e6930bb8ac476ca65a136f64c7` | Native converted | none / none | 40,960 |
| Qwen3 4B, `mlx-community/Qwen3-4B-4bit` | `4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25` | `7c2af9a45e5b0d196f06a5cbfd6fc5cc95adf29df79885d0540f2206fb3d40d8` | Native converted affine | none / none | 40,960 |
| `alibiserikbay/JevK5` | `c4f7fdb3aeab5582336406e78d3bef11bf98833d` | `ea3fd57de724a9cc34cce999f42e78f30245e11bdf18938377bd367d4bbf0dc3` | Native converted | none / none | 262,144 |
| `Qwen/Qwen3.8-27B` (row 49) | `37561816e66b5b9153b1b64c10968c18c1e6f4a0` | `0bf350d1722bb1086073ef7964e3cb68f43a49c586d216fa7d953b7a968834f8` | Native converted BF16 vision | enabled / MTP | 262,144 |
| Lily retained, `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl` | `0f94842e3682b49acace4daeb4cf2e65a6f94f83` | `18c7bcc1e97d1fefdd361472c2daa59d81d89fcd6f3a61983d995a032b44bfb4` | Hybrid retained Lily affine | BF16 / MTP | 262,144 |
| ddalcu text, `ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit` | `788c07a8725642bccf13a0a3b8a21954ab84b4f2` | `84a69b2bcb20a57b033c3827bc98fa21a5db4d18b55e0640dd3687c99a1b765f` | Hybrid external affine | disabled / MTP | 262,144 |

## Native choices and distinct preparations

Fixed installer pins and formats come from [native_catalog.py](../tools/converter/stark_convert/native_catalog.py), [hybrid_prepare.py](../tools/converter/stark_convert/hybrid_prepare.py) and each linked preparation contract. The checkout and build in each linked run govern its evidence. “Pending” refers to candidate sizing or admission on that exact preparation.

| Choice / availability | Loader / format | Shipped pin or resolved source revision | Enabled tower / text draft | Artifact identity and disposition |
| --- | --- | --- | --- | --- |
| `glm53-q2`, fixed catalog | Hybrid GGUF UD-Q2_K_XL | Weights `a38483c8cd5df544f53d70fb281afe97369d5ab6`, metadata `04c4e9e95c5da8862dced7e5056455116f83a7e0` | none / none | Candidate pending, complete local payload unconfirmed. [Preparation](NATIVE_HYBRID_MODELS.md#checkpoints-and-preparation) |
| GLM EXL3, optional feature and API1 staged pack | Hybrid EXL3/FP8 bridge, backend-fixed cap | Pack `50237d5c26ceb98bfd85f779990b0f5e6d7af144` | none / headless default, BF16 draft in explicit experimental stage | Both stage profiles retain separate receipts. Candidate build and complete pinned stage unconfirmed. [Validation and staging](validation/GLM_EXL3_BACKEND.md) |
| `qwen38-mixed`, fixed catalog | Hybrid Pipe mixed affine 4/8-bit | `b2c422f3c643e36f04227a64d61796b44a4b1029` | none / none | Fresh per request, prefix reuse unsupported. Candidate pending. [Preparation](NATIVE_HYBRID_MODELS.md#checkpoints-and-preparation) |
| `qwen38-lily-q4-xl`, fixed catalog | Hybrid Lily affine, original format | `0f94842e3682b49acace4daeb4cf2e65a6f94f83` | BF16 in retained mode / MTP | Retained image receipt above. Fixed catalog also offers text mode, which needs a separate receipt. |
| Typed Uzu Lily, explicit conversion | Hybrid typed `qwen4_exp` | `0f94842e3682b49acace4daeb4cf2e65a6f94f83` for prior converted study | none / none | Separate conversion receipt, prior storage/readout/generation controls. Candidate pending. [Format study](benchmarks/QWEN38_UZU_FORMAT_2026_10_29.md) |
| `qwen38-mlx-serve-mixed`, fixed catalog text | Hybrid external affine | `788c07a8725642bccf13a0a3b8a21954ab84b4f2` | disabled / MTP | Installed receipt above. |
| `qwen38-mlx-serve-mixed`, fixed catalog retained-image preparation | Hybrid external affine, schema 6 | `788c07a8725642bccf13a0a3b8a21954ab84b4f2` | BF16 / MTP | Clone manifest SHA-256 `2ea0257f3369fbc38bf1646d2c9c64f5504a2eca774fec68257d0a784f68dd7d`, payload aggregate `3c9b158adf55514da375b0a88f382ba8f4d80284e99c0f5a25f810b749da41ee`. Bounded text MTP and prefix replay, one-token image and cold text recovery passed under paging. A launch against the original folder without the manifest was rejected before inference. Candidate load sizing and admission with tower and predictor resident remain pending. [Clone run](archive/runs/2026-11-09-model-launch-double-check.md) |
| Indexed retained affine, Add Source | Hybrid indexed affine with separate PLE/MTP | `nace-ai/Qwen3.8-Flash-Next-MLX-KQuant` at `05cb5c56c918b8e32ed440c71d17a8e8f72fff90` | BF16 / MTP | Bounded image and text-cache lifecycle passed. Candidate pending. [Run](archive/runs/2026-11-12-flashnext-general-images.md) |
| `flint-int4`, fixed catalog | Hybrid MLX affine INT4, 8-bit embeddings | `3b93f9053b530488405ae9fca51d4900d6ce21f0` | none / none | Prefix reuse unsupported. Candidate pending, complete local payload unconfirmed. [Preparation](FLINT.md) |
| `ripple-q8`, fixed catalog | Hybrid GGUF Q8_0 | `57066fdb2348a53f2261f096e0a793b9554b4106` | none / none | Prefix reuse unsupported. Candidate pending. [Preparation](RIPPLE.md) |
| `ripple-q4-k-m`, fixed catalog | Hybrid GGUF Q4_K_M | `df550a7178a87bbfb6ee2bdbd37603c553d536ee` | none / none | Prefix reuse unsupported. Candidate pending. [Preparation](RIPPLE.md) |
| `ripple-bf16`, terminal preparation | Hybrid Safetensors BF16 | `e27e171792f303d74c048184e8259ba663cfa0d5` (also GGUF metadata pin) | none / none | Prefix reuse unsupported. Candidate pending. [Preparation](RIPPLE.md) |
| `inkling2-mini`, fixed catalog | Inkling affine8 dense plus source experts | Qualified `6e7c8248f75265bdd4fbab5273304243b5d8c2cb`, download `1fb4134b9a74f9681054f4ef82e4bf1f458a2337` | native / DSpark on source-verifier path | Target `c9e14848ad017467dc0deb91c6aa67022e181dbcdfe02a1b04024434bf9acddd`. Prior flexible-window and resize/prefix pass. Candidate pending. [Run](model-experiments/2026-11-24-inkling-mini-request-context.md) |
| `inkling2-mini-original`, terminal preparation | Inkling original NVFP4/MXFP8 | Same qualified source and download mirror as affine8 | native / DSpark on source-verifier path | Original target `65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1`. Candidate pending as a distinct artifact. [Preparation](NATIVE_HYBRID_MODELS.md#checkpoints-and-preparation) |
| `hy4-q3-k`, fixed catalog, feature build | Hy4 streamed GGUF Q3_K | Weights/projector `c50d4dd30be42f2b91138e5731de0e2f32ad56b2`, metadata `eccd3123eec73f2040f4c9ed0de1883593470510` | BF16 projector / NextN | Pinned shard/projector hashes in `hy4_prepare.py`. Prior packaged worker advertised 452,608 and passed short text and image requests. Candidate explicit cap, automatic selection and readiness pending. [Packaged run](model-experiments/2026-11-26-hy4-desktop-catalog.md) |
| Experimental Qwen3 GGUF Q8_0, explicit import | Native converted GGUF codes | Two immutable input revisions required by recipe, checkpoint-dependent | none / none | Output identity established by import receipt. Candidate pending. [Import implementation](../tools/converter/stark_convert/gguf_import.py) |
| Decider Flash v1.3, fixed Decision | Native Decider BF16 | `d06fb3f2d97e0a52325a0d463c450cf0338f44fe` | supported / none | Scoring-only. [Profile](../tools/converter/stark_convert/decider_specs.py) |
| Decider v1.3, fixed Decision | Native Decider BF16 | `9c2ebb27470d3dec2ad7ab727727e40d142c7213` | supported / none | Scoring-only. [Profile](../tools/converter/stark_convert/decider_specs.py) |
| Elephant 1, fixed Decision and installed receipt | Native Decider BF16 slot-LM | `b3f8a93462fb1051a9de031e94a7ff380e254072` | retained / none | Scoring-only. [Evidence](MODEL_RESEARCH.md) |

## Converted vision profiles

These profiles use the [Native vision loader](../crates/stark-generate/src/native.rs). Enabled components are specific to the converted mode. Gemma 3 catalog discovery seeds official BF16 sources separately from legacy planned rows. The quantized Gemma 3 registry rows have no corresponding admitted image converter in this inventory.

| Source / profile | Revision evidence | Tower / text draft | Receipt or disposition |
| --- | --- | --- | --- |
| Official Gemma 3 4B BF16, schema 5 | Row 58, recorded source run | SigLIP / none | Weights SHA-256 `50b031efe933a6023b4188a718b1f4e2d654d6eac5b5c9336205de59a3426805`; native numerical replay and worker/HTTP passes. [Exact run](archive/runs/2026-11-07-gemma3-images.md) |
| Official Gemma 3 12B BF16, schema 5 | unresolved | SigLIP / none | Structural conversion and admission, candidate artifact unavailable in installed set. [Profile](GEMMA3_IMAGES.md) |
| Official Gemma 3 27B BF16, schema 5 | Row 57, historical snapshot | SigLIP / none | Structural conversion and admission, full checkpoint and sizing unverified. [Profile](GEMMA3_IMAGES.md) |
| Pinned Gemma 4 E2B / E4B / 12B / 26B-A4B / 31B BF16, schema 4 | Rows 60 / 62 / 115 / 116 / 117, profile pins | BF16 vision, unified embedder for 12B / none in base vision artifact | Small profiles have exact archive receipts and bounded runtime evidence. Larger profiles have distinct component/structural evidence. External E2B assistant staging has separate controls. [Identity and evidence by size](model-experiments/2026-11-08-gemma-images.md) |
| Official Qwen3.5 dense BF16 and eligible affine dense v3 | Rows 41–46 (dense entries) and 89–98 where source meets vision contract, historical revisions above | BF16 / absent in baseline 0.8B runs | 0.8B BF16 and affine each have their own converted hashes and recorded controls. Larger sizes require separate admission. [Artifact ledger](model-experiments/2026-11-08-qwen-images.md) |
| Official Qwen3.8-27B BF16 vision | Row 49, catalog pin and installed receipt | BF16 / MTP | Installed receipt above. |
| Jundot Qwen3.8-27B mixed affine v3 | Row 111, historical resolved revision | BF16 / omitted by recorded vision conversion | Qualified tower and bounded worker lifecycle, separate from the official 27B and Jundot text artifacts. [Artifact ledger](model-experiments/2026-11-08-qwen-images.md) |
| `mlx-community/Qwen3.5-35B-A3B-4bit`, affine MoE v4 | `1e20fd8d42056f870933bf98ca6211024744f7ec` from exact converted run | BF16 / no recorded enabled head | Affine weights `2f762b26…` and actual lifecycle; candidate sizing pending. [Artifact ledger](model-experiments/2026-11-08-qwen-images.md) |
| Official `Qwen/Qwen3.5-35B-A3B`, BF16 MoE v5 | Row 44, historical snapshot and metadata inspection | BF16 / no recorded enabled head | Implemented image conversion/operator profile. Complete official checkpoint conversion and execution pending. [Artifact ledger](model-experiments/2026-11-08-qwen-images.md) |

The [image inventory](MODEL_IMAGE_STATUS.md) and [image parity plan](IMAGE_DEFAULT_PARITY_PLAN.md) describe additional applicability and remaining decoder and cache gates. Exact source revisions, component choices and output manifests are required for any newly discovered compatible source.

## Hardware queue and artifact availability

The managed registry and known staging locations were checked for the first follow-up targets on 2026-11-26. Old metadata, partial stages and auxiliary weights do not provide a runnable artifact. Check the candidate receipt and hash before reusing any stage.

| Order | Target | Preparation and next gate |
| --- | --- | --- |
| First | LLaMBA floating and affine4 (rows 51–52) | Earlier conversion metadata survives without its converted weight payload. Prepare a matching source revision, then check the unknown-capacity readiness sentinel, the default managed model listing, and per-request admission. [Converter record](archive/conversion/MODEL_CONVERSION_MILESTONE_5.md) |
| First | GLM EXL3 | Complete pinned pack, configured optional worker build and prepared profile are unconfirmed locally. Test the explicit feature-disabled refusal and backend-fixed window with the enabled and staged worker when available. [Staging contract](validation/GLM_EXL3_BACKEND.md) |
| First | ddalcu retained image | The exact text payload is in the managed store. Recreate the retained manifest in a separate prepared clone, then run normal image-enabled loading and admission checks. The earlier clone is independently attested by manifest and payload hashes above. [Clone recipe and prior evidence](archive/runs/2026-11-09-model-launch-double-check.md) |
| Next | Inkling affine8 and original | The checked managed stage lacks complete target and draft files. Prepare or locate each pinned variant. Test both separately with native images and text DSpark on the supported path. [Pinned preparation](NATIVE_HYBRID_MODELS.md#checkpoints-and-preparation) |
| Next | Hy4 Q3_K | A prior artifact passed in its packaged build. A retained projector and prior range metadata remain locally, with neither complete trunk shard found at the checked staging paths. Reuse any verified matching download files during native preparation and repeat automatic/explicit-cap sizing per receipt. [Install and resume record](model-experiments/2026-11-26-hy4-desktop-catalog.md) |
| As available | Ordinary GLM GGUF, Pipe, Flint, Ripple, typed Lily, Qwen3 GGUF and converted text/vision rows | Use the preparation and loader entries above. Prioritize a distinct estimator or dispatch path. Preserve the per-artifact tower, draft, prefix and admission outcome. |
