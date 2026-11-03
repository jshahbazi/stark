# Qwen3-4B template fixture provenance

`qwen3_4b_chat_template.jinja` is the exact `chat_template` string in the official
Hugging Face `tokenizer_config.json` for `mlx-community/Qwen3-4B-4bit`, pinned to
revision `4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25`.

- Source file: [`tokenizer_config.json`](https://huggingface.co/mlx-community/Qwen3-4B-4bit/blob/4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25/tokenizer_config.json)
- Repository card and license: [`README.md`](https://huggingface.co/mlx-community/Qwen3-4B-4bit/blob/4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25/README.md), Apache-2.0
- UTF-8 byte length: 4,116
- SHA-256: `87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5`

This fixture exists to bind parser admission and template-rendering regression
checks to upstream bytes. It does not assert model competence or generation
quality. The source template remains under its upstream Apache-2.0 license.

The `stark-jinja-program-v1` fingerprint of these bytes is
`813d6e292f35cf32a98357c5382ec57ae39ddb658e5b16559c957b119a2859b5`.
It uses the locked Jinja parser and the renderer's shared environment settings.
Source locations and comments are excluded; adjacent literal `TemplateData`
within an `Output` is joined byte-for-byte. Every remaining node class, field,
typed scalar and ordered list is retained. This enables equivalent formatting
without broadening the generated-call grammar. The compatibility tests keep
the original fixture unchanged and compare synthetic rendering across variants.
