# Tool-template fixtures

These files are bounded public chat-template sources used by synthetic protocol
tests. The byte hashes below identify source content; the versioned program
fingerprints identify normalized Jinja AST structure and are not byte hashes.
Normalization removes only the Transformers `generation` / `endgeneration`
tracking tags before hashing. No model weights are included. Fixture presence
is not model conversion, runtime, or tool-use qualification evidence.

`Raw SHA-256` hashes the checked-in `.jinja` bytes. `Normalized SHA-256` hashes
the UTF-8 bytes after `normalize_chat_template()` removes only the
Transformers-only generation tracking tags. `Program SHA-256` is the
versioned AST fingerprint returned by `template_program_sha256()` for that
normalized source. Parser names are the local reviewed protocol implementations.

| Fixture | Source repository and immutable revision | Extracted template source URL | Raw SHA-256 | Normalized SHA-256 | Program SHA-256 | Parser | License |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `functiongemma_mlx_chat_template.jinja` | `mlx-community/functiongemma-270m-it-bf16` @ `bb327a9ad61044e1496a2bee2365a6b6a6684c72` | [chat_template.jinja](https://huggingface.co/mlx-community/functiongemma-270m-it-bf16/blob/bb327a9ad61044e1496a2bee2365a6b6a6684c72/chat_template.jinja) | `db61fb01017bd82401d3ffca4f8e066cd56ff6d38d2a30c4058770c0bf7ab49b` | `db61fb01017bd82401d3ffca4f8e066cd56ff6d38d2a30c4058770c0bf7ab49b` | `16eb40376f3dd05b4fa6adfa6ec4603d8c03ace3d91a4a83d55a0a1cd07c9b6d` | `functiongemma-v1` | Gemma terms; see [`functiongemma.LICENSE.md`](functiongemma.LICENSE.md) and [model card](https://huggingface.co/mlx-community/functiongemma-270m-it-bf16/blob/bb327a9ad61044e1496a2bee2365a6b6a6684c72/README.md). |
| `functiongemma_unsloth_chat_template.jinja` | `unsloth/functiongemma-270m-it` @ `79eaaa84125b897f5a0ad6638362b864fec813ad` | [chat_template.jinja](https://huggingface.co/unsloth/functiongemma-270m-it/blob/79eaaa84125b897f5a0ad6638362b864fec813ad/chat_template.jinja) | `0dbaf70819c970c9bacd7125be8e9db46b7bc0e6ed3bc303d92bd58fb9dd604a` | `0dbaf70819c970c9bacd7125be8e9db46b7bc0e6ed3bc303d92bd58fb9dd604a` | `5bf9ad01e979cbe145818337ff144bc8cb280793c63b063aee03e404b1e0b360` | `functiongemma-v1` | Gemma terms; see [`functiongemma.LICENSE.md`](functiongemma.LICENSE.md) and [model card](https://huggingface.co/unsloth/functiongemma-270m-it/blob/79eaaa84125b897f5a0ad6638362b864fec813ad/README.md). |
| `granite30.jinja` | `ibm-granite/granite-3.0-2b-instruct` @ `5ad66c190631382717bd92d7b052adb1a7b669e7` | [chat_template.jinja](https://huggingface.co/ibm-granite/granite-3.0-2b-instruct/blob/5ad66c190631382717bd92d7b052adb1a7b669e7/chat_template.jinja) | `09d68eeb5c0eaa551adea6cc088a05c7298966f395f203126e76b74f5a77a8ff` | `09d68eeb5c0eaa551adea6cc088a05c7298966f395f203126e76b74f5a77a8ff` | `bf894d352517d84d8ba8052ef76c597bad4d04e208d2329aa44718e3e34bb7cc` | `granite30-json-v1` | Apache-2.0; see [`apache2.LICENSE.txt`](apache2.LICENSE.txt) and [model card](https://huggingface.co/ibm-granite/granite-3.0-2b-instruct/blob/5ad66c190631382717bd92d7b052adb1a7b669e7/README.md). |
| `granite31.jinja` | `ibm-granite/granite-3.1-2b-instruct` @ `bbc2aed595bd38bd770263dc3ab831db9794441d` | [tokenizer_config.json](https://huggingface.co/ibm-granite/granite-3.1-2b-instruct/blob/bbc2aed595bd38bd770263dc3ab831db9794441d/tokenizer_config.json) | `08962c2f15d56767854b46dfc4070b37f4c443551833bba65b417191735f3187` | `08962c2f15d56767854b46dfc4070b37f4c443551833bba65b417191735f3187` | `fe4cefe90a431b88138e882192891494712dc65c9bd2d50264e984c957ee10a8` | `granite33-json-v1` | Apache-2.0; see [`apache2.LICENSE.txt`](apache2.LICENSE.txt) and [model card](https://huggingface.co/ibm-granite/granite-3.1-2b-instruct/blob/bbc2aed595bd38bd770263dc3ab831db9794441d/README.md). |
| `granite33.jinja` | `ibm-granite/granite-3.3-2b-instruct` @ `707f574c62054322f6b5b04b6d075f0a8f05e0f0` | [chat_template.jinja](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct/blob/707f574c62054322f6b5b04b6d075f0a8f05e0f0/chat_template.jinja) | `0bea7711316005a1c37c9941a7ee2a26311d223c979abee416aab0a402fc5b56` | `0bea7711316005a1c37c9941a7ee2a26311d223c979abee416aab0a402fc5b56` | `b55e692ca03fc5a1791041c5b804b5d46689b76942f9c7d1d42b79a0de349545` | `granite33-json-v1` | Apache-2.0; see [`apache2.LICENSE.txt`](apache2.LICENSE.txt) and [model card](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct/blob/707f574c62054322f6b5b04b6d075f0a8f05e0f0/README.md). |
| `lfm25_1_2b.jinja` | `LiquidAI/LFM2.5-1.2B-Instruct` @ `b9d816ab510f9a28dca37cb182e9da8b0226e13c` | [chat_template.jinja](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct/blob/b9d816ab510f9a28dca37cb182e9da8b0226e13c/chat_template.jinja) | `ba551d58630afa3190b1be3602e28301f3d2e9bbac978dfc49d6d825171648b6` | `2e7245c2c41abbc5cb19e67f18ee7d7999bd88f6f74eca7eef80362b0d124e24` | `e3b6d05880efbac801343bb51b37397624dba782d869b6a6a13a43e9d9a64501` | `lfm2-pythonic-v1` | LFM Open License v1.0; see [`lfm-open-license-v1.0.txt`](lfm-open-license-v1.0.txt) and [model card](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct/blob/b9d816ab510f9a28dca37cb182e9da8b0226e13c/README.md). |
| `lfm25_350m.jinja` | `LiquidAI/LFM2.5-350M` @ `9e6c6ccf47cd318696e137d381a7ded8fe4df09f` | [chat_template.jinja](https://huggingface.co/LiquidAI/LFM2.5-350M/blob/9e6c6ccf47cd318696e137d381a7ded8fe4df09f/chat_template.jinja) | `ba551d58630afa3190b1be3602e28301f3d2e9bbac978dfc49d6d825171648b6` | `2e7245c2c41abbc5cb19e67f18ee7d7999bd88f6f74eca7eef80362b0d124e24` | `e3b6d05880efbac801343bb51b37397624dba782d869b6a6a13a43e9d9a64501` | `lfm2-pythonic-v1` | LFM Open License v1.0; see [`lfm25-open-license-v1.0.txt`](lfm25-open-license-v1.0.txt), the exact source license file (SHA-256 `4d28ca14dedc0b3d0fcc2b3339f0e79931faa33874f3d24f522183a8fc70068c`), and [model card](https://huggingface.co/LiquidAI/LFM2.5-350M/blob/9e6c6ccf47cd318696e137d381a7ded8fe4df09f/README.md). |
| `lfm2_1_2b.jinja` | `LiquidAI/LFM2-1.2B` @ `40f3da0d0164913923aee9462c23077868b816a3` | [chat_template.jinja](https://huggingface.co/LiquidAI/LFM2-1.2B/blob/40f3da0d0164913923aee9462c23077868b816a3/chat_template.jinja) | `5504d163d879bfb6fd4135b7fa3d56768b2e69dd718338f7b146750289f46ab0` | `5504d163d879bfb6fd4135b7fa3d56768b2e69dd718338f7b146750289f46ab0` | `55ff13408d61d3cde699acb45fddd155abbccc68b943fdc1bcc89561d39b0b28` | `lfm2-pythonic-v1` | LFM Open License v1.0; see [`lfm-open-license-v1.0.txt`](lfm-open-license-v1.0.txt) and [model card](https://huggingface.co/LiquidAI/LFM2-1.2B/blob/40f3da0d0164913923aee9462c23077868b816a3/README.md). |
| `llama31.jinja` | `mlx-community/Llama-3.1-8B-Instruct-4bit` @ `90215b22ec18e72f623dde2ea7af4097025160e2` | [tokenizer_config.json](https://huggingface.co/mlx-community/Llama-3.1-8B-Instruct-4bit/blob/90215b22ec18e72f623dde2ea7af4097025160e2/tokenizer_config.json) | `e10ca381b1ccc5cf9db52e371f3b6651576caee0a630b452e2816b2d404d4b65` | `e10ca381b1ccc5cf9db52e371f3b6651576caee0a630b452e2816b2d404d4b65` | `c9a73db2f091123af3ea6862594741bf44c5fc77241603bdd11fb9ba01059734` | `llama-json-v1` | Llama 3.1 Community License; full official terms [`llama31.LICENSE.txt`](llama31.LICENSE.txt), from [Meta llama-models](https://github.com/meta-llama/llama-models/blob/1f0feb795a4130697ced243fb53051670d591653/models/llama3_1/LICENSE). Mirror model card: [README](https://huggingface.co/mlx-community/Llama-3.1-8B-Instruct-4bit/blob/90215b22ec18e72f623dde2ea7af4097025160e2/README.md). |
| `llama32.jinja` | `unsloth/Llama-3.2-1B-Instruct` @ `5a8abab4a5d6f164389b1079fb721cfab8d7126c` | [tokenizer_config.json](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct/blob/5a8abab4a5d6f164389b1079fb721cfab8d7126c/tokenizer_config.json) | `5816fce10444e03c2e9ee1ef8a4a1ea61ae7e69e438613f3b17b69d0426223a4` | `5816fce10444e03c2e9ee1ef8a4a1ea61ae7e69e438613f3b17b69d0426223a4` | `ac8a185b982fa2d2457459bbe86bd64680248fe536dae5531cee537c36f9ab6b` | `llama-json-v1` | Llama 3.2 Community License; full official terms [`llama32.LICENSE.txt`](llama32.LICENSE.txt), from [Meta llama-models](https://github.com/meta-llama/llama-models/blob/8d29d93fa5700a60532e0061a02ffa89d0acd3fc/models/llama3_2/LICENSE). Mirror model card: [README](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct/blob/5a8abab4a5d6f164389b1079fb721cfab8d7126c/README.md). |
| `mistral_v03.jinja` | `mistralai/Mistral-7B-Instruct-v0.3` @ `c170c708c41dac9275d15a8fff4eca08d52bab71` | [tokenizer_config.json](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/blob/c170c708c41dac9275d15a8fff4eca08d52bab71/tokenizer_config.json) | `e16746b40344d6c5b5265988e0328a0bf7277be86f1c335156eae07e29c82826` | `e16746b40344d6c5b5265988e0328a0bf7277be86f1c335156eae07e29c82826` | `152f9ceacd72be7c708cb2ec9e317f243d8272a9f404a1452797ae2492a184a2` | `mistral-json-v1` | Apache-2.0; see [`apache2.LICENSE.txt`](apache2.LICENSE.txt) and [model card](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/blob/c170c708c41dac9275d15a8fff4eca08d52bab71/README.md). |
| `phi4mini.jinja` | `microsoft/Phi-4-mini-instruct` @ `cfbefacb99257ffa30c83adab238a50856ac3083` | [tokenizer_config.json](https://huggingface.co/microsoft/Phi-4-mini-instruct/blob/cfbefacb99257ffa30c83adab238a50856ac3083/tokenizer_config.json) | `febf589225c9728ab791f52e8897d7607a823d45368f0a4c92fa68997b40cce9` | `febf589225c9728ab791f52e8897d7607a823d45368f0a4c92fa68997b40cce9` | `1cf2a9d562d08814b91466fa42c56c8a656d13c1088d29c1add5d1ac21b84009` | `phi4-json-v1` | MIT; full license [`phi4mini.LICENSE`](phi4mini.LICENSE), from the [pinned model repository](https://huggingface.co/microsoft/Phi-4-mini-instruct/blob/cfbefacb99257ffa30c83adab238a50856ac3083/LICENSE). |
| `rnj1.jinja` | `EssentialAI/rnj-1-instruct` @ `5ef96d7a82b2bff8009192979432c9bc64e870cd` | [chat_template.jinja](https://huggingface.co/EssentialAI/rnj-1-instruct/blob/5ef96d7a82b2bff8009192979432c9bc64e870cd/chat_template.jinja) | `88ad4f413fb9967035d2d6bfc0570266adcf0bfc9d5501278ed3da00bf1b6397` | `2dbcdd619c0c5830b42a1a1d08de864a3e9df6a2d7bffa0e981b7dcf93892cd9` | `0934a9a5ef6b9e105bbe456ab6273d386fb10a74cb397d191c2a8f7779f01c43` | `qwen3-hermes-v1` | Apache-2.0; see [`apache2.LICENSE.txt`](apache2.LICENSE.txt) and [model card](https://huggingface.co/EssentialAI/rnj-1-instruct/blob/5ef96d7a82b2bff8009192979432c9bc64e870cd/README.md). |
| `smollm2.jinja` | `HuggingFaceTB/SmolLM2-1.7B-Instruct` @ `31b70e2e869a7173562077fd711b654946d38674` | [tokenizer_config.json](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/blob/31b70e2e869a7173562077fd711b654946d38674/tokenizer_config.json) | `872be49dbb638044ad01b60388f48d469ff2980e5f0dccdc22ec907db54d0788` | `872be49dbb638044ad01b60388f48d469ff2980e5f0dccdc22ec907db54d0788` | `ff802fefdd364f94905651d6f8abf16d9001cb8b4d8b1fb9b72f1cab2645fecf` | `smollm2-json-v1` | Apache-2.0; see [`apache2.LICENSE.txt`](apache2.LICENSE.txt) and [model card](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/blob/31b70e2e869a7173562077fd711b654946d38674/README.md). |
| `smollm3.jinja` | `HuggingFaceTB/SmolLM3-3B` @ `a07cc9a04f16550a088caea529712d1d335b0ac1` | [chat_template.jinja](https://huggingface.co/HuggingFaceTB/SmolLM3-3B/blob/a07cc9a04f16550a088caea529712d1d335b0ac1/chat_template.jinja) | `3e0e275a527f249670056e0078e474eadbdd007c134b4fdfae5a7b3089702814` | `5389af391984d661fc19bb01456dd698df94babd9e054e6656f1953c586167d0` | `96229b4a4c5f52570ad056a7d53857a52675848e1d4ac983d6922678fbd4c5ed` | `smollm3-hermes-v1` | Apache-2.0; see [`apache2.LICENSE.txt`](apache2.LICENSE.txt) and [model card](https://huggingface.co/HuggingFaceTB/SmolLM3-3B/blob/a07cc9a04f16550a088caea529712d1d335b0ac1/README.md). |

## Mistral runtime source

`mistral_v03.jinja` is the unmodified upstream template fixture. Runtime tool
rendering uses the reviewed patched asset
[`mistral_v03_tools.jinja`](../../../stark_convert/chat_templates/mistral_v03_tools.jinja),
whose only template change splits the JSON value output from the surrounding
string concatenation. This preserves JSON serializer escaping for values that
contain template or HTML-like markup. It does not add tool-return framing. Its
SHA-256 is
`dfbff0166eb003d00c7a23594654f9783f70efab223f91e2bddc0a8e0fa059ed`; its
normalized template-program fingerprint is
`f836889ad320ef027e124af2b8f924abd3fc5942d48f37540753d6854c7f098b`.
The local patch has separate provenance; it is not represented as upstream
source.

## License files

- [`apache2.LICENSE.txt`](apache2.LICENSE.txt) is the complete [Apache
  License 2.0](https://www.apache.org/licenses/LICENSE-2.0.txt) text, used for
  the Apache-licensed Granite, Mistral, RNJ, and SmolLM sources (SHA-256
  `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`).
- [`llama31.LICENSE.txt`](llama31.LICENSE.txt) and
  [`llama32.LICENSE.txt`](llama32.LICENSE.txt) are complete Meta Llama 3.1 and
  3.2 Community License texts respectively (SHA-256
  `bf2eac60b81e5c5f779fc3b4849d3bfacfd63f97f3eb68b44e9c6868c5801712` and
  `8cc15535a8a34b41888f644b339a1a9eb428af793a4f5e24df58a3e5b1487d74`). The
  templates are sourced from public mirrors; this fixture provenance does not
  assert those mirrors' model weights are Meta checkpoint bytes.
- [`lfm-open-license-v1.0.txt`](lfm-open-license-v1.0.txt) is the exact pinned
  license file for LFM2-1.2B and LFM2.5-1.2B-Instruct (SHA-256
  `61d7e939a05911c765b7e98ffaa1ab5ca6c0174a65350766c25cb10197d19fc8`).
  [`lfm25-open-license-v1.0.txt`](lfm25-open-license-v1.0.txt) preserves the
  distinct exact license-file bytes from LFM2.5-350M (SHA-256
  `4d28ca14dedc0b3d0fcc2b3339f0e79931faa33874f3d24f522183a8fc70068c`). Both
  files identify LFM Open License v1.0 and differ in newline encoding only.
- [`phi4mini.LICENSE`](phi4mini.LICENSE) is the complete MIT license supplied
  with the pinned Microsoft model repository (SHA-256
  `fa8235e5b48faca34e3ca98cf4f694ef08bd216d28b58071a1f85b1d50cb814d`).
- FunctionGemma's source cards declare `license: gemma`; its separate existing
  [`functiongemma.LICENSE.md`](functiongemma.LICENSE.md) records that fact and
  the applicable upstream terms pointer without substituting a different
  license.
