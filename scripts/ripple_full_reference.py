"""Evaluate a packed Ripple checkpoint with streamed publisher layers.

Run in the locked ``tools`` uv environment. The independent PyTorch oracle
dequantizes one layer at a time on CPU and executes the exact hash-pinned
publisher decoder. It never expands the full model or invokes llama.cpp.
Reference readouts and token fixtures stay in a new Git-ignored directory;
stdout contains only aggregate measurements and hashes.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

from ripple_native_fixture import (
    SOURCE_HASHES,
    SOURCE_REPO,
    SOURCE_REVISION,
    require_private_destination,
    source_module,
)


def _hash_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(16 * 1024 * 1024):
            value.update(data)
    return value.hexdigest()


def _ordered_shards(sources: list):
    """Admit complete split GGUF inventories with metadata on shard zero."""
    split = any(
        any(
            key in source.metadata
            for key in ("split.no", "split.count", "split.tensors.count")
        )
        for source in sources
    )
    if not split:
        if len(sources) != 1:
            raise ValueError("multiple GGUF files require split metadata")
        ordered = sources
    else:
        by_index = {}
        declared_tensors = None
        for source in sources:
            index = source.metadata.get("split.no")
            count = source.metadata.get("split.count")
            tensors = source.metadata.get("split.tensors.count")
            if (
                type(index) is not int
                or type(count) is not int
                or count != len(sources)
                or not 0 <= index < count
                or index in by_index
                or type(tensors) is not int
                or not 0 < tensors <= 2_000_000
                or (declared_tensors is not None and tensors != declared_tensors)
            ):
                raise ValueError("incomplete or inconsistent GGUF split metadata")
            declared_tensors = tensors
            by_index[index] = source
        if (
            set(by_index) != set(range(len(sources)))
            or sum(len(source.tensors) for source in sources) != declared_tensors
        ):
            raise ValueError("incomplete GGUF split tensor inventory")
        ordered = [by_index[index] for index in range(len(sources))]
    if ordered[0].metadata.get("general.architecture") != "ricursive" or any(
        source.metadata.get("general.architecture", "ricursive") != "ricursive"
        for source in ordered[1:]
    ):
        raise ValueError("GGUF architecture mismatch")
    return ordered


def dequantize_k_blocks(data: bytes, kind: str):
    """Independent NumPy GGML K-block decoder, without native engine calls.

    Layout/equations: llama.cpp 1b70350c92563449576bcbbf552994b1d1ed847d,
    ggml/src/ggml-quants.c dequantize_row_q4_K/dequantize_row_q6_K.
    Returns F32 in GGML element order; packed source bytes remain untouched.
    """
    import numpy as np

    size = {"Q4_K": 144, "Q6_K": 210}.get(kind)
    if size is None or not data or len(data) % size:
        raise ValueError("invalid K-block payload")
    raw = np.frombuffer(data, dtype=np.uint8).reshape(-1, size)
    result = np.empty((len(raw), 256), dtype=np.float32)

    def half(offset):
        return raw[:, offset : offset + 2].copy().view("<f2").reshape(-1).astype(np.float32)

    if kind == "Q4_K":
        d, dmin = half(0), half(2)
        packed_scales = raw[:, 4:16]
        codes = raw[:, 16:]
        for group in range(8):
            if group < 4:
                scale = packed_scales[:, group] & 63
                minimum = packed_scales[:, group + 4] & 63
            else:
                scale = (packed_scales[:, group + 4] & 15) | ((packed_scales[:, group - 4] >> 6) << 4)
                minimum = (packed_scales[:, group + 4] >> 4) | ((packed_scales[:, group] >> 6) << 4)
            nibble = (codes[:, (group // 2) * 32 : (group // 2 + 1) * 32] >> (4 * (group % 2))) & 15
            result[:, group * 32 : (group + 1) * 32] = (
                (d * scale)[:, None] * nibble - (dmin * minimum)[:, None]
            )
    else:
        d = half(208)
        low, high = raw[:, :128], raw[:, 128:192]
        scales = raw[:, 192:208].copy().view(np.int8)
        lanes = np.arange(32) // 16
        for half_index in range(2):
            for group in range(4):
                packed_low = low[:, half_index * 64 + (group % 2) * 32 : half_index * 64 + (group % 2 + 1) * 32]
                packed_high = high[:, half_index * 32 : (half_index + 1) * 32]
                codes = (((packed_low >> (4 * (group // 2))) & 15)
                         | (((packed_high >> (2 * group)) & 3) << 4)).astype(np.int16) - 32
                scale = scales[:, half_index * 8 + group * 2 + lanes].astype(np.float32)
                begin = half_index * 128 + group * 32
                result[:, begin : begin + 32] = (d[:, None] * scale) * codes
    return result


class Q8Weights:
    """Read packed GGUF tensors independently of the native Rust decoder."""

    def __init__(self, directory: Path):
        # Reuse only the bounded GGUF container parser. Numeric decoding below
        # is NumPy, independently of the production Rust/Metal implementation.
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/converter"))
        from stark_convert.gguf_import import read_gguf

        self.files = sorted(directory.glob("*.gguf"))
        if not 1 <= len(self.files) <= 8:
            raise ValueError("invalid GGUF shard inventory")
        sources = []
        for path in self.files:
            if path.is_symlink():
                raise ValueError("GGUF source must be a regular file")
            sources.append(read_gguf(path, reference_k_quants=True))
        sources = _ordered_shards(sources)
        self.files = [source.path for source in sources]
        self.tensors = {}
        for source in sources:
            for name, tensor in source.tensors.items():
                if name in self.tensors:
                    raise ValueError("duplicate GGUF tensor")
                self.tensors[name] = (source.path, tensor)

    def _read(
        self, name: str, first_row: int | None = None, row_count: int | None = None
    ):
        import numpy as np
        import torch

        path, info = self.tensors[name]
        shape = info.shape
        if len(shape) > 2 or info.kind not in {"F32", "F16", "Q8_0", "Q4_K", "Q6_K"}:
            raise ValueError("unsupported full-reference tensor")
        offset, size = info.offset, info.nbytes
        if first_row is not None:
            if (
                len(shape) != 2
                or row_count is None
                or not 0 <= first_row < shape[0]
                or first_row + row_count > shape[0]
            ):
                raise ValueError("invalid embedding row selection")
            row_bytes = info.nbytes // shape[0]
            offset += first_row * row_bytes
            size = row_count * row_bytes
            shape = (row_count, shape[1])
        with path.open("rb") as stream:
            stream.seek(offset)
            data = stream.read(size)
        if len(data) != size:
            raise ValueError("truncated reference tensor")
        if info.kind == "Q8_0":
            if math.prod(shape) % 32:
                raise ValueError("invalid Q8 tensor geometry")
            blocks = np.frombuffer(
                data, dtype=np.dtype([("scale", "<f2"), ("codes", "i1", (32,))])
            )
            values = blocks["codes"].astype(np.float32)
            values *= blocks["scale"].astype(np.float32)[:, None]
            values = values.reshape(shape)
        elif info.kind in {"Q4_K", "Q6_K"}:
            values = dequantize_k_blocks(data, info.kind).reshape(shape)
        else:
            values = (
                np.frombuffer(data, dtype="<f4" if info.kind == "F32" else "<f2")
                .astype(np.float32)
                .reshape(shape)
            )
        if not np.isfinite(values).all():
            raise ValueError("nonfinite reference weights")
        return torch.from_numpy(values)

    def tensor(self, name: str):
        return self._read(name)

    def embedding(self, tokens: list[int]):
        import torch

        return torch.cat(
            [self._read("token_embd.weight", token, 1) for token in tokens], dim=0
        )

    def layer(self, index: int) -> dict:
        names = {
            "input_layernorm.weight": "attn_norm.weight",
            "post_attention_layernorm.weight": "post_attention_norm.weight",
            "pre_feedforward_layernorm.weight": "ffn_norm.weight",
            "post_feedforward_layernorm.weight": "post_ffw_norm.weight",
            "self_attn.q_proj.weight": "attn_q.weight",
            "self_attn.k_proj.weight": "attn_k.weight",
            "self_attn.v_proj.weight": "attn_v.weight",
            "self_attn.o_proj.weight": "attn_output.weight",
            "self_attn.g_proj.weight": "attn_gate.weight",
            "self_attn.q_norm.weight": "attn_q_norm.weight",
            "self_attn.k_norm.weight": "attn_k_norm.weight",
            "self_attn.value_res_lambda": "attn_vres_lambda.weight",
            "mlp.gate_proj.weight": "ffn_gate.weight",
            "mlp.up_proj.weight": "ffn_up.weight",
            "mlp.down_proj.weight": "ffn_down.weight",
        }
        return {hf: self.tensor(f"blk.{index}.{gguf}") for hf, gguf in names.items()}


def reference(
    directory: Path, output: Path, cache: Path, positions: int, threads: int
) -> dict:
    import numpy as np
    import torch
    import transformers
    from tokenizers import Tokenizer

    require_private_destination(output)
    transformers.utils.logging.set_verbosity_error()
    torch.set_num_threads(threads)
    module = source_module(cache)
    raw_config = (directory / "config.json").read_bytes()
    serialized = json.loads(raw_config)
    if serialized.get("model_type") != "ricursive":
        raise ValueError("reference architecture mismatch")
    config = module.RicursiveConfig(**serialized)
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    base = tokenizer.encode(
        "A short check of local inference.", add_special_tokens=False
    ).ids
    if not base:
        raise ValueError("reference tokenization failed")
    tokens = (base * ((positions + len(base) - 1) // len(base)))[:positions]
    weights = Q8Weights(directory)
    started = time.monotonic()
    with torch.no_grad():
        hidden = weights.embedding(tokens).unsqueeze(0)
        if config.embed_scale is not None:
            hidden = hidden * config.embed_scale
        seed = hidden
        pos = torch.arange(positions)
        for index in range(config.num_hidden_layers):
            with torch.device("meta"):
                layer = module.RicursiveDecoderLayer(config, index)
            state = weights.layer(index)
            layer.load_state_dict(state, strict=True, assign=True)
            layer.eval()
            hidden = layer(
                hidden, position_ids=pos[None], cache_position=pos, value_residual=seed
            )[0]
            del state, layer
            gc.collect()
        with torch.device("meta"):
            norm = module.RicursiveRMSNorm(config.hidden_size, config.rms_norm_eps)
            head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        norm.load_state_dict(
            {"weight": weights.tensor("output_norm.weight")}, assign=True
        )
        head.load_state_dict({"weight": weights.tensor("output.weight")}, assign=True)
        values = head(norm(hidden))[0].detach().cpu().numpy().astype("<f4")
    elapsed = time.monotonic() - started
    if not np.isfinite(values).all():
        raise ValueError("nonfinite full reference")
    output.mkdir(parents=True, mode=0o700)
    (output / "config.json").write_bytes(raw_config)
    # Hardlinks retain immutable source bytes without a second model-size copy.
    for path in weights.files:
        os.link(path, output / path.name)
    (output / "tokens.json").write_text(json.dumps(tokens) + "\n")
    readout = values.tobytes()
    (output / "reference.f32").write_bytes(readout)
    receipt = {
        "schema_version": 1,
        "source_repository": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "source_sha256": SOURCE_HASHES,
        "source_transformations": [],
        "harness_sha256": _hash_file(Path(__file__)),
        "harness_dependencies_sha256": {
            "ripple_native_fixture.py": _hash_file(
                Path(__file__).with_name("ripple_native_fixture.py")
            ),
            "tools/converter/stark_convert/gguf_import.py": _hash_file(
                Path(__file__).resolve().parents[1]
                / "tools/converter/stark_convert/gguf_import.py"
            ),
        },
        "reference_runtime": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "weight_format": "GGUF packed weights, independently dequantized to F32",
        "tensor_storage_kinds": sorted({info.kind for _, info in weights.tensors.values()}),
        "execution_device": "CPU",
        "execution_dtype": "float32",
        "positions": positions,
        "layers": config.num_hidden_layers,
        "vocabulary": config.vocab_size,
        "cpu_threads": threads,
        "reference_seconds": elapsed,
        "config_sha256": hashlib.sha256(raw_config).hexdigest(),
        "weight_files": [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": _hash_file(path),
            }
            for path in weights.files
        ],
        "reference_sha256": hashlib.sha256(readout).hexdigest(),
        "token_sha256": hashlib.sha256(
            (output / "tokens.json").read_bytes()
        ).hexdigest(),
    }
    (output / "fixture.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reference-source", type=Path, default=Path("runtime/ripple-reference-source")
    )
    parser.add_argument("--positions", type=int, default=8)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.positions <= 32 or not 1 <= args.threads <= 16:
        parser.error("reference bounds exceeded")
    try:
        receipt = reference(
            args.model_path.resolve(),
            args.output.resolve(),
            args.reference_source.resolve(),
            args.positions,
            args.threads,
        )
    except Exception:  # noqa: BLE001 - reference exceptions must never expose private values.
        print(
            json.dumps({"status": "failed", "reason": "ripple_full_reference_failed"})
        )
        return 1
    print(json.dumps({"status": "created", **receipt}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
