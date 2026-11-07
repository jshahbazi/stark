# FunctionGemma chat-template fixtures

These unchanged public mirror templates support synthetic rendering and wire
format tests. They do not establish model behavior or task quality.

| Fixture | Immutable source | SHA-256 | Jinja program fingerprint |
| --- | --- | --- | --- |
| `functiongemma_mlx_chat_template.jinja` | [`mlx-community/functiongemma-270m-it-bf16/chat_template.jinja`, revision `bb327a9ad61044e1496a2bee2365a6b6a6684c72`](https://huggingface.co/mlx-community/functiongemma-270m-it-bf16/blob/bb327a9ad61044e1496a2bee2365a6b6a6684c72/chat_template.jinja) | `db61fb01017bd82401d3ffca4f8e066cd56ff6d38d2a30c4058770c0bf7ab49b` | `16eb40376f3dd05b4fa6adfa6ec4603d8c03ace3d91a4a83d55a0a1cd07c9b6d` |
| `functiongemma_unsloth_chat_template.jinja` | [`unsloth/functiongemma-270m-it/chat_template.jinja`, revision `79eaaa84125b897f5a0ad6638362b864fec813ad`](https://huggingface.co/unsloth/functiongemma-270m-it/blob/79eaaa84125b897f5a0ad6638362b864fec813ad/chat_template.jinja) | `0dbaf70819c970c9bacd7125be8e9db46b7bc0e6ed3bc303d92bd58fb9dd604a` | `5bf9ad01e979cbe145818337ff144bc8cb280793c63b063aee03e404b1e0b360` |

Both model cards identify `google/functiongemma-270m-it` as their base model and
declare `license: gemma`. These mirror files are the pinned fixture sources;
the original Google tokenizer metadata is gated and was not accessed. The
FunctionGemma syntax and stop sequence were cross-checked against Google's
public [formatting guide](https://ai.google.dev/gemma/docs/functiongemma/formatting-and-best-practices)
and [complete call sequence](https://ai.google.dev/gemma/docs/functiongemma/full-function-calling-sequence-with-functiongemma).

The templates emit `<start_function_response>` immediately after a final
assistant call block in history; Google's guide specifies that marker as an
additional generation stop sequence. Their fallback scalar rendering turns
Python `None` into the literal `None`; tests cover that source-proven spelling.
The decoder also accepts JSON `null`, though these templates do not emit it for
a Python `None` value.

The source was checked with the locked converter environment using
`uv --project tools/converter run --locked --python 3.13.13`; fingerprints use
Stark's `stark-jinja-program-v1` implementation. No weights were downloaded or
executed.
