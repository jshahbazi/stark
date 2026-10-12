"""Qwen3 dispatch regression coverage for the independently qualified casts."""

import numpy as np
import pytest
from test_streaming import qwen_fixture, save_tensors, tensor_bytes

from stark_convert.qwen3 import build_qwen3_plan
from stark_convert.streaming import UnsupportedConversion, load_checkpoint, read_safetensors_header, write_plan


@pytest.mark.parametrize("source_dtype", ["BF16", "F16", "F32"])
@pytest.mark.parametrize("target", ["bfloat16", "float16", "float32"])
@pytest.mark.parametrize("tied", [False, True])
def test_qwen3_uniform_cast_dispatch_and_output(tmp_path, source_dtype, target, tied):
    import ml_dtypes

    source = tmp_path / "source"
    _, tensors = qwen_fixture(source, dtype=source_dtype, tied=tied, attention_bias=True, sharded=True)
    plan = build_qwen3_plan(source, dtype=target)
    inputs = {(info.path, info.offset): tensors[name][1] for name, info in load_checkpoint(source).items()}
    stats = write_plan(plan, tmp_path / "output.safetensors", chunk_bytes=96)
    converted = read_safetensors_header(tmp_path / "output.safetensors")[1]
    assert stats["peak_buffer_bytes"] <= 96
    target_dtype = {"bfloat16": ml_dtypes.bfloat16, "float16": "<f2", "float32": "<f4"}[target]
    for operation in plan.operations:
        if operation.dtype == "F32" and operation.name.endswith(".scales"):
            continue
        raw = []
        for info in operation.sources:
            original = inputs[(info.path, info.offset)]
            values = original.view(ml_dtypes.bfloat16) if source_dtype == "BF16" else original
            raw.append(values.astype(target_dtype).tobytes())
        assert tensor_bytes(converted[operation.name]) == b"".join(raw)


@pytest.mark.parametrize("fusion", ["qkv", "up_gate", "bias"])
def test_qwen3_mixed_fusion_fails_explicitly(tmp_path, fusion):
    source = tmp_path / "source"
    _, tensors = qwen_fixture(source, attention_bias=True)
    suffix = {"qkv": "self_attn.k_proj.weight", "up_gate": "mlp.gate_proj.weight", "bias": "self_attn.k_proj.bias"}[
        fusion
    ]
    name = "model.layers.0." + suffix
    tensors[name] = ("F16", np.zeros(tensors[name][1].shape, dtype="<f2"))
    save_tensors(source / "model.safetensors", tensors)
    with pytest.raises(UnsupportedConversion, match="Mixed-dtype"):
        build_qwen3_plan(source)
