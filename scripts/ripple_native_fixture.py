"""Build private Ripple fixtures with the publisher's pinned PyTorch model.

Run in the locked ``tools`` uv environment. Only this validation helper imports
the independently downloaded reference; production loading never executes it.
Stored BF16 weights are promoted to F32 before reference execution, matching
the native baseline's F32 activations, accumulation, residuals and KV state.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import struct
import subprocess
import sys
import types
import urllib.request
from pathlib import Path

SOURCE_REPO = "ricursiveAI/Ripple-65B-Preview"
SOURCE_REVISION = "e27e171792f303d74c048184e8259ba663cfa0d5"
SOURCE_HASHES = {
    "configuration_ricursive.py": "002cb3915be9685eb962d617623bec14d45b5d6b5d46e689fac98a3b38cb6cf6",
    "modeling_ricursive.py": "f83865ef643e273a6c9aecf1575e85c95ad01fd995b1e43c3c7a892765006bb3",
}
MAX_SOURCE_BYTES = 1024 * 1024


def save_q8_fixture(model, path: Path) -> None:
    """Quantize tiny source matrices and reload their exact decoded values."""
    import numpy as np
    import torch

    aliases = {
        "input_layernorm": "attn_norm",
        "post_attention_layernorm": "post_attention_norm",
        "pre_feedforward_layernorm": "ffn_norm",
        "post_feedforward_layernorm": "post_ffw_norm",
        "self_attn.q_proj": "attn_q",
        "self_attn.k_proj": "attn_k",
        "self_attn.v_proj": "attn_v",
        "self_attn.o_proj": "attn_output",
        "self_attn.g_proj": "attn_gate",
        "self_attn.q_norm": "attn_q_norm",
        "self_attn.k_norm": "attn_k_norm",
        "self_attn.value_res_lambda": "attn_vres_lambda",
        "mlp.gate_proj": "ffn_gate",
        "mlp.up_proj": "ffn_up",
        "mlp.down_proj": "ffn_down",
    }
    tensors, restored = [], {}
    for name, parameter in model.state_dict().items():
        value = parameter.detach().numpy().astype("<f4")
        if name.startswith("model.layers."):
            _, _, index, suffix = name.split(".", 3)
            suffix = suffix.removesuffix(".weight")
            gguf_name = f"blk.{index}.{aliases[suffix]}.weight"
        else:
            gguf_name = {
                "model.embed_tokens.weight": "token_embd.weight",
                "model.norm.weight": "output_norm.weight",
                "lm_head.weight": "output.weight",
            }[name]
        if value.ndim == 2:
            if value.shape[-1] % 32:
                raise ValueError("tiny Q8 matrix has incompatible row width")
            blocks = value.reshape(-1, 32)
            scales = (np.max(np.abs(blocks), axis=1) / 127).astype("<f2")
            scales[scales == 0] = 1
            codes = (
                np.rint(blocks / scales.astype(np.float32)[:, None])
                .clip(-127, 127)
                .astype("i1")
            )
            packed = np.empty(
                len(blocks), dtype=np.dtype([("scale", "<f2"), ("codes", "i1", (32,))])
            )
            packed["scale"], packed["codes"] = scales, codes
            payload, kind = packed.tobytes(), 8
            value = (
                codes.astype(np.float32) * scales.astype(np.float32)[:, None]
            ).reshape(value.shape)
        else:
            payload, kind = value.tobytes(), 0
        restored[name] = torch.from_numpy(value.copy())
        tensors.append((gguf_name, value.shape, kind, payload))

    def string(value: str) -> bytes:
        data = value.encode()
        return struct.pack("<Q", len(data)) + data

    header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 1))
    header.extend(
        string("general.architecture") + struct.pack("<I", 8) + string("ricursive")
    )
    payloads, offset = bytearray(), 0
    for name, shape, kind, payload in tensors:
        offset += -offset % 32
        payloads.extend(b"\0" * (offset - len(payloads)))
        header.extend(string(name) + struct.pack("<I", len(shape)))
        for width in reversed(shape):
            header.extend(struct.pack("<Q", width))
        header.extend(struct.pack("<IQ", kind, offset))
        payloads.extend(payload)
        offset += len(payload)
    header.extend(b"\0" * (-len(header) % 32))
    path.write_bytes(header + payloads)
    model.load_state_dict(restored, strict=True)


def source_module(cache: Path):
    """Load exact hash-pinned source without any semantic transformations."""
    package_name = "stark_ripple_pinned_reference"
    package = types.ModuleType(package_name)
    package.__path__ = [str(cache)]
    sys.modules[package_name] = package
    for name, expected in SOURCE_HASHES.items():
        path = cache / name
        if path.exists():
            if (
                path.is_symlink()
                or not path.is_file()
                or path.stat().st_size > MAX_SOURCE_BYTES
            ):
                raise ValueError("invalid cached reference source")
            data = path.read_bytes()
        else:
            url = (
                f"https://huggingface.co/{SOURCE_REPO}/resolve/{SOURCE_REVISION}/{name}"
            )
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read(MAX_SOURCE_BYTES + 1)
        if len(data) > MAX_SOURCE_BYTES or hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("reference source hash mismatch")
        if not path.exists():
            cache.mkdir(parents=True, exist_ok=True, mode=0o700)
            path.write_bytes(data)
        module_name = f"{package_name}.{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ValueError("reference source import failed")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module


def require_private_destination(output: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", str(output)], cwd=root, check=False
    )
    if ignored.returncode != 0 or output.exists():
        raise ValueError("fixture must be a new Git-ignored destination")


def fixture(output: Path, cache: Path, bf16: bool, q8: bool = False) -> dict:
    import torch
    import transformers
    from safetensors.torch import save_file
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    require_private_destination(output)
    transformers.utils.logging.set_verbosity_error()
    module = source_module(cache)
    torch.set_num_threads(1)
    torch.manual_seed(1729)
    config = module.RicursiveConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=5,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        sliding_window=4,
        use_gated_attention=True,
        gate_full_precision=True,
        use_valres=True,
        ffn_clamp_threshold=0.35,
        embed_scale=3.25,
        tie_word_embeddings=False,
        bos_token_id=62,
        eos_token_id=[63],
        attention_bias=False,
        attention_dropout=0.0,
    )
    model = module.RicursiveForCausalLM(config).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "value_res_lambda" in name:
                index = int(name.split(".")[2])
                parameter.fill_(-1.4 + index * 0.8)
            elif "norm" in name:
                parameter.copy_(
                    torch.linspace(0.65, 1.35, parameter.numel()).reshape(
                        parameter.shape
                    )
                )
            else:
                parameter.copy_(
                    torch.randn_like(parameter) * (0.7 / parameter.shape[-1] ** 0.5)
                )
    if bf16:
        model = model.to(torch.bfloat16)
    output.mkdir(parents=True, mode=0o700)
    vocabulary = [f"word{index}" for index in range(config.vocab_size)]
    vocabulary[:8] = ["<unk>", "A", "short", "check", "of", "local", "inference", "."]
    vocabulary[62:] = ["<bos>", "<eos>"]
    tokenizer = Tokenizer(
        WordLevel(
            dict(zip(vocabulary, range(config.vocab_size), strict=True)),
            unk_token="<unk>",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(output / "tokenizer.json"))
    serialized = config.to_dict()
    serialized["torch_dtype"] = "bfloat16" if bf16 else "float32"
    (output / "config.json").write_text(json.dumps(serialized, indent=2) + "\n")
    weights = {
        name: value.detach().contiguous() for name, value in model.state_dict().items()
    }
    weight_file = output / ("model.gguf" if q8 else "model.safetensors")
    if q8:
        save_q8_fixture(model, weight_file)
    else:
        save_file(weights, weight_file)
    del weights
    model = model.float()

    clipping = {
        "gate_above_threshold": 0,
        "gate_below_negative_threshold": 0,
        "up_outside_threshold": 0,
    }

    def gate_hook(_module, _inputs, values):
        clipping["gate_above_threshold"] += int(
            (values > config.ffn_clamp_threshold).sum()
        )
        clipping["gate_below_negative_threshold"] += int(
            (values < -config.ffn_clamp_threshold).sum()
        )

    def up_hook(_module, _inputs, values):
        clipping["up_outside_threshold"] += int(
            (values.abs() > config.ffn_clamp_threshold).sum()
        )

    handles = []
    for layer in model.model.layers:
        handles.append(layer.mlp.gate_proj.register_forward_hook(gate_hook))
        handles.append(layer.mlp.up_proj.register_forward_hook(up_hook))
    tokens = [1, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 12, 3, 2, 1, 4]
    cache_state = None
    rows = []
    with torch.no_grad():
        for token in tokens:
            result = model(
                torch.tensor([[token]]), past_key_values=cache_state, use_cache=True
            )
            cache_state = result.past_key_values
            rows.append(result.logits[0, -1].detach().cpu())
    for handle in handles:
        handle.remove()
    values = torch.stack(rows)
    with torch.no_grad():
        full = model(torch.tensor([tokens]), use_cache=False).logits[0]
    full_error = float((values - full).abs().max())
    if (
        not torch.isfinite(values).all()
        or full_error > 2e-5
        or not all(clipping.values())
    ):
        raise ValueError("reference fixture coverage or consistency failed")
    reference = values.numpy().astype("<f4").tobytes()
    (output / "tokens.json").write_text(json.dumps(tokens) + "\n")
    (output / "reference.f32").write_bytes(reference)
    receipt = {
        "schema_version": 1,
        "source_repository": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "source_sha256": SOURCE_HASHES,
        "source_transformations": [],
        "reference_runtime": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "source_storage_dtype": "GGUF Q8_0/F32"
        if q8
        else "bfloat16"
        if bf16
        else "float32",
        "execution_dtype": "float32",
        "kv_dtype": "float32",
        "positions": len(tokens),
        "vocabulary": config.vocab_size,
        "layers": config.num_hidden_layers,
        "sliding_window": config.sliding_window,
        "full_vs_cached_max_abs_error": full_error,
        "clipping_coverage": clipping,
        "weight_sha256": hashlib.sha256(weight_file.read_bytes()).hexdigest(),
        "token_sha256": hashlib.sha256(
            (output / "tokens.json").read_bytes()
        ).hexdigest(),
        "reference_sha256": hashlib.sha256(reference).hexdigest(),
    }
    (output / "fixture.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reference-source", type=Path, default=Path("runtime/ripple-reference-source")
    )
    storage = parser.add_mutually_exclusive_group()
    storage.add_argument("--bf16", action="store_true")
    storage.add_argument("--q8", action="store_true")
    args = parser.parse_args()
    try:
        receipt = fixture(
            args.output.resolve(), args.reference_source.resolve(), args.bf16, args.q8
        )
    except Exception:  # noqa: BLE001 - reference exceptions must never expose private values.
        print(
            json.dumps(
                {"status": "failed", "reason": "ripple_reference_fixture_failed"}
            )
        )
        return 1
    print(json.dumps({"status": "created", **receipt}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
