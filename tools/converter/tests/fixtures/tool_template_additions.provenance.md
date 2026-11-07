# Additional tool-template fixtures

These public upstream programs are retained unchanged for admission, native
rendering and synthetic protocol tests. They do not establish model task quality.

| Fixture | Source and immutable revision | UTF-8 SHA-256 | License |
| --- | --- | --- | --- |
| `qwen25_chat_template.jinja` | `chat_template` string from [Qwen/Qwen2.5-0.5B-Instruct tokenizer_config.json](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/tokenizer_config.json) | `cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f` | Apache-2.0; preserved in `qwen25_chat_template.LICENSE` |
| `glm53_chat_template.jinja` | [zai-org/GLM-5.3-Flash chat_template.jinja](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/04c4e9e95c5da8862dced7e5056455116f83a7e0/chat_template.jinja) | `34d5ee66b12fa6446cdae131c352b8f68cd85369e0e6fda115583805fada3891` | MIT, copyright 2026 Z.AI Co., Ltd; preserved in `glm53_chat_template.LICENSE` |
| `flint_chat_template.jinja` | [RecursiveSI/flint-int4 chat_template.jinja](https://huggingface.co/RecursiveSI/flint-int4/blob/3b93f9053b530488405ae9fca51d4900d6ce21f0/chat_template.jinja) | `7940f4ffa3275c36caa4a582f845aaad79da11018b3961e8fbcad6ae29733466` | Apache-2.0; preserved in `flint_chat_template.LICENSE` |

The version-1 Jinja program fingerprints are respectively
`8f92b77faedd343fd1d45a534e1fc93a77efb76adddd98a6819e6cd17a0b9bfa` and
`55dff1823a10de87fa2f4ca4c0986498cd577ed4d495642f6c371e0abbcc243a`.
The GLM metadata revision is already pinned by Stark's native preparation route;
the weight repository has a separate identity. Qwen2.5-Coder-3B-Instruct revision
`488639f1ff808d1d3d0ba301aef8c11461451ec5` has the same Qwen2.5 program
fingerprint, checked independently against its public tokenizer metadata.

Flint's program fingerprint is
`c61d6063b6aeb383d9a6e80cb62f48164363c6fabb4369d85005c0ae9b9237b8`.
Its argument framing and scalar encodings reuse `glm53-xml-v1`; its distinct
history/result rendering is checked against the native template helper.
