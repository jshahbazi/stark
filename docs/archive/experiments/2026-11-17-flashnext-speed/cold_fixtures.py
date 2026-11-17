"""Build exact-length public-source prompts without loading model weights."""

from hashlib import sha256
from pathlib import Path
import struct
import subprocess
from unittest.mock import patch


REVISION = "2cc7345369d3ae34d7f293b2a6ac20db87bc21cb"
LENGTHS = (8192, 12288, 32768)
ASSETS = {
    "config.json": "2a91155be52ebbb0f6dff20e61dcd8ae7484154a56b4afe067246c59da46f899",
    "chat_template.jinja": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
    "tokenizer.json": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
}
TASKS = (
    (
        "worker-cancellation",
        "crates/stark-generate/src/generation.rs",
        "c4c72f832e8726be35aae77fa7422ecdb174b8540d335a7b5207d90ceea6100c",
        "rust",
        "Write a Rust test module for a generation request state machine. Show three tests covering "
        "cancellation before sampling, cancellation mid-stream, and recovery on the next request. "
        "Include test setup and concrete assertions. Return code only.",
    ),
    (
        "desktop-streaming",
        "tools/converter/stark_convert/desktop.py",
        "b9d3a3ae8c5a9200df8c19a930a72b8bbb41f953b3a26ee8d2559b8c7085a94b",
        "python",
        "Write a Python helper that consumes a stream of progress, text delta, and terminal reply "
        "events. Reject malformed order, a second terminal reply, and a truncated stream. "
        "Include three pytest cases. Return code only.",
    ),
    (
        "prefill-boundary",
        "vendor/lily-core/src/qwen4exp/model.rs",
        "551b35f67a59130fb7df09c152d065fd8f1f7d6e58ff6a1350a4acf43bab3387",
        "rust",
        "Write a Rust helper that computes a safe reusable prefill snapshot boundary for a growing "
        "prompt. Leave at least two tokens for the next pass. Include tests for an exact chunk "
        "boundary, a one-token suffix, and a mismatched prefix. Return code only.",
    ),
)


def digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def fit(render, source, length):
    # BPE can merge differently at a cutoff. Search rendered token counts,
    # then check nearby character boundaries if bisection skips a length.
    low, high = 0, len(source)
    while low <= high:
        cut = (low + high) // 2
        rendered, ids = render(source[:cut])
        if len(ids) == length:
            return cut, rendered, ids
        if len(ids) < length:
            low = cut + 1
        else:
            high = cut - 1
    for cut in range(max(0, low - 32), min(len(source), low + 32) + 1):
        rendered, ids = render(source[:cut])
        if len(ids) == length:
            return cut, rendered, ids
    raise ValueError("cold_fixture_exact_length_unavailable")


def make_prompts(repo: Path, model: Path) -> tuple[dict[tuple[str, int], list[int]], dict]:
    from stark_convert import chat, reasoning, template_runtime
    from tokenizers import Tokenizer

    if any(digest((model / name).read_bytes()) != expected for name, expected in ASSETS.items()):
        raise ValueError("cold_fixture_asset_identity_mismatch")
    helper = template_runtime._template_executable()
    tokenizer = Tokenizer.from_file(str(model / "tokenizer.json"))
    matrix = {}
    metadata = {
        "source_revision": REVISION,
        "asset_sha256": dict(ASSETS),
        "template_helper_sha256": digest(helper.read_bytes()),
        "thinking": False,
        "prompts": [],
    }
    with patch.object(template_runtime, "_template_executable", return_value=helper):
        codec, template = chat.load_template(model, expected_hashes={
            "config_sha256": ASSETS["config.json"], "template_sha256": ASSETS["chat_template.jinja"],
        })
        system = codec.get("default_system_prompt")
        messages = [{"role": "system", "content": system}] if system else []
        template_options = {
            **reasoning.reasoning_options(codec, thinking=False),
            "tools": None,
            "add_generation_prompt": True,
            "bos_token": codec.get("bos_token") or "",
            "eos_token": codec.get("eos_token") or "",
        }
        for task, path, expected, language, instruction in TASKS:
            source_bytes = subprocess.run(
                ["git", "show", f"{REVISION}:{path}"], cwd=repo, check=True, capture_output=True,
            ).stdout
            if digest(source_bytes) != expected:
                raise ValueError("cold_fixture_source_identity_mismatch")
            source = source_bytes.decode("utf-8")

            def render(excerpt):
                content = (
                    f"Use the following checked-in source excerpt from {path} as context.\n\n"
                    f"```{language}\n{excerpt}\n```\n\n{instruction}"
                )
                rendered = chat.render_template(
                    template, messages=[*messages, {"role": "user", "content": content}], **template_options,
                )
                return rendered, tokenizer.encode(rendered, add_special_tokens=False).ids

            for length in LENGTHS:
                cut, rendered, ids = fit(render, source, length)
                matrix[task, length] = ids
                metadata["prompts"].append({
                    "task": task, "source_path": path, "source_sha256": expected,
                    "instruction_sha256": digest(instruction.encode()),
                    "source_excerpt_bytes": len(source[:cut].encode()), "input_tokens": length,
                    "rendered_sha256": digest(rendered.encode()),
                    "input_token_ids_sha256": digest(struct.pack(f"<{length}I", *ids)),
                })
    return matrix, metadata
