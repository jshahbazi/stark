"""Private exact-word controls for Gemma Q4 prefill operand/partial precision.

Run with the locked tools environment. Only the first Q/K/V projections are
reconstructed in RAM. Private output vectors and aggregate receipts are written;
no checkpoint is materialized and no GPU is used.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from gemma4_hf_reference import digest


def run(args) -> dict:
    import numpy as np
    import torch
    from safetensors import safe_open

    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if (
        output.exists()
        or subprocess.run(
            ["git", "check-ignore", "--quiet", "--", str(output)], cwd=root, check=False
        ).returncode
    ):
        raise ValueError("output_must_be_new_and_ignored")
    config = json.loads((args.source / "config.json").read_text())
    text = config.get("text_config", config)
    quant = config["quantization"]
    if quant.get("bits") != 4 or quant.get("mode", "affine") != "affine":
        raise ValueError("control_requires_w4_affine_source")
    width = text["hidden_size"]
    values = np.fromfile(args.fixture / "actual_qkv_input.f32", dtype="<f4")
    if values.size % width or not 2 <= values.size // width <= 256:
        raise ValueError("invalid_projection_input")
    x = torch.from_numpy(values.reshape(-1, width))
    torch.set_num_threads(8)
    planes = []
    with safe_open(
        str(args.source / "model.safetensors"), framework="pt", device="cpu"
    ) as weights:
        for branch in ("q", "k", "v"):
            stem = f"language_model.model.layers.0.self_attn.{branch}_proj"
            packed = weights.get_tensor(stem + ".weight").to(torch.int64)
            scales = (
                weights.get_tensor(stem + ".scales")
                .float()
                .repeat_interleave(quant["group_size"], dim=1)
            )
            biases = (
                weights.get_tensor(stem + ".biases")
                .float()
                .repeat_interleave(quant["group_size"], dim=1)
            )
            codes = (
                ((packed[..., None] >> torch.arange(0, 32, 4, dtype=torch.int64)) & 15)
                .reshape(scales.shape)
                .float()
            )
            planes.append((codes, scales, biases))
    exact = torch.cat(
        [codes * scales + biases for codes, scales, biases in planes], dim=0
    )
    double_bf16 = torch.cat(
        [
            ((codes * scales).to(torch.bfloat16).float() + biases)
            .to(torch.bfloat16)
            .float()
            for codes, scales, biases in planes
        ],
        dim=0,
    )
    double_f16 = torch.cat(
        [
            ((codes * scales).to(torch.float16).float() + biases)
            .to(torch.float16)
            .float()
            for codes, scales, biases in planes
        ],
        dim=0,
    )
    policies = [
        ("f32_dequant_f32_accum", exact, None),
        ("f16_dequant_f32_accum", exact.to(torch.float16).float(), None),
        ("bf16_dequant_f32_accum", exact.to(torch.bfloat16).float(), None),
        ("bf16_double_round_f32_accum", double_bf16, None),
        ("f16_double_round_f32_accum", double_f16, None),
        (
            "bf16_dequant_bf16_k128_partials",
            exact.to(torch.bfloat16).float(),
            torch.bfloat16,
        ),
        (
            "f16_dequant_bf16_k128_partials",
            exact.to(torch.float16).float(),
            torch.bfloat16,
        ),
    ]
    references = {
        "stock_mlx": np.fromfile(args.fixture / "actual_qkv.f32", dtype="<f4"),
        "native": np.fromfile(args.native / "native-qkv.f32", dtype="<f4"),
    }
    output.mkdir(parents=True, mode=0o700)
    rows = []
    source_partials = []
    offset = 0
    # MLX v0.32.3 quantized.cpp qmm_splitk targets 512 32x32 tiles,
    # retaining BF16 partition outputs before its separate axis-zero sum.
    # Preserve these candidate partials privately so the reduction itself can
    # be checked on the physical source device without reloading the model.
    for branch, (codes, scales, biases) in zip(("q", "k", "v"), planes):
        columns = codes.shape[0]
        splits = min(
            max(512 // (((x.shape[0] + 31) // 32) * ((columns + 31) // 32)), 1),
            width // quant["group_size"],
        )
        while splits > 1 and width % (splits * quant["group_size"]):
            splits -= 1
        chunk = width // splits
        operand = (codes * scales + biases).to(torch.bfloat16).float()
        partials = torch.stack(
            [
                torch.nn.functional.linear(
                    x[:, i : i + chunk], operand[:, i : i + chunk]
                )
                .to(torch.bfloat16)
                .float()
                for i in range(0, width, chunk)
            ]
        ).numpy()
        partial_path = output / f"source-{branch}-partials.f32"
        partial_path.write_bytes(partials.tobytes())
        partial_path.chmod(0o600)
        reference = references["stock_mlx"].reshape(x.shape[0], -1)[
            :, offset : offset + columns
        ]
        reference_path = output / f"source-{branch}-reference.f32"
        reference_path.write_bytes(reference.tobytes())
        reference_path.chmod(0o600)
        source_partials.append(
            {
                "branch": branch,
                "shape": list(partials.shape),
                "chunk": chunk,
                "partials": partial_path.name,
                "partials_sha256": digest(partial_path),
                "reference": reference_path.name,
                "reference_sha256": digest(reference_path),
            }
        )
        offset += columns
    for name, weights, scratch_dtype in policies:
        if scratch_dtype is None:
            projected = torch.nn.functional.linear(x, weights)
        else:
            projected = sum(
                torch.nn.functional.linear(x[:, i : i + 128], weights[:, i : i + 128])
                .to(scratch_dtype)
                .float()
                for i in range(0, width, 128)
            )
        result = projected.to(torch.bfloat16).float().numpy().reshape(-1)
        path = output / (name + ".f32")
        path.write_bytes(result.tobytes())
        path.chmod(0o600)
        comparisons = {}
        for reference_name, reference in references.items():
            error = result.astype(np.float64) - reference
            comparisons[reference_name] = {
                "different": int((result != reference).sum()),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "max_abs": float(np.abs(error).max()),
            }
        rows.append(
            {"policy": name, "output_sha256": digest(path), "comparisons": comparisons}
        )
    receipt = {
        "device": "cpu",
        "torch_version": torch.__version__,
        "tools_lock_sha256": digest(root / "tools/uv.lock"),
        "script_sha256": digest(Path(__file__)),
        "weights_sha256": digest(args.source / "model.safetensors"),
        "input_sha256": digest(args.fixture / "actual_qkv_input.f32"),
        "candidate_source_partials": source_partials,
        "rows": rows,
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "fixture", "native", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception:  # noqa: BLE001 - validation data and exception details stay private
        print(json.dumps({"passed": False, "error": "prefill_operand_control_failed"}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
