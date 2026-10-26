import json

import numpy as np
import pytest
from test_qwen35 import qwen35_fixture
from test_streaming import save_tensors, tensor_bytes

from stark_convert.adapters.qwen35 import build_qwen35_plan
from stark_convert.streaming import ConversionError, UnsupportedConversion, read_safetensors_header, write_plan


def vision_fixture(path):
    config, tensors = qwen35_fixture(path, wrapper=True, dtype="BF16", sharded=False)
    vision = {
        "model_type": "qwen3_5",
        "hidden_size": 4,
        "intermediate_size": 6,
        "depth": 2,
        "num_heads": 2,
        "patch_size": 2,
        "temporal_patch_size": 1,
        "in_channels": 3,
        "num_position_embeddings": 5,
        "spatial_merge_size": 1,
        "out_hidden_size": 8,
        "hidden_act": "gelu_pytorch_tanh",
        "initializer_range": 0.02,
        "deepstack_visual_indexes": [],
    }
    config["vision_config"] = vision
    tensors.pop("model.visual.patch_embed.proj.weight")
    shapes = {
        "patch_embed.proj.weight": (4, 3, 1, 2, 2),
        "patch_embed.proj.bias": (4,),
        "pos_embed.weight": (5, 4),
        "merger.norm.weight": (4,),
        "merger.norm.bias": (4,),
        "merger.linear_fc1.weight": (4, 4),
        "merger.linear_fc1.bias": (4,),
        "merger.linear_fc2.weight": (8, 4),
        "merger.linear_fc2.bias": (8,),
    }
    for i in range(2):
        b = f"blocks.{i}"
        shapes.update(
            {
                f"{b}.norm1.weight": (4,),
                f"{b}.norm1.bias": (4,),
                f"{b}.norm2.weight": (4,),
                f"{b}.norm2.bias": (4,),
                f"{b}.attn.qkv.weight": (12, 4),
                f"{b}.attn.qkv.bias": (12,),
                f"{b}.attn.proj.weight": (4, 4),
                f"{b}.attn.proj.bias": (4,),
                f"{b}.mlp.linear_fc1.weight": (6, 4),
                f"{b}.mlp.linear_fc1.bias": (6,),
                f"{b}.mlp.linear_fc2.weight": (4, 6),
                f"{b}.mlp.linear_fc2.bias": (4,),
            }
        )
    for n, shape in shapes.items():
        tensors["model.visual." + n] = ("BF16", np.arange(np.prod(shape), dtype="<u2").reshape(shape) + len(tensors))
    (path / "config.json").write_text(json.dumps(config))
    save_tensors(path / "model.safetensors", tensors)
    return config, tensors, shapes


def test_vision_plan_consumes_all_tensors_and_preserves_shapes_and_bytes(tmp_path):
    source = tmp_path / "source"
    _, tensors, shapes = vision_fixture(source)
    plan = build_qwen35_plan(source, include_vision=True)
    vision_ops = {op.name: op for op in plan.operations if op.name.startswith("vision.")}
    assert set(vision_ops) == {"vision." + n for n in shapes}
    assert all(
        op.dtype == "BF16" and op.shape == shapes[op.name.removeprefix("vision.")] for op in vision_ops.values()
    )
    assert plan.source_tensors == len(tensors)
    write_plan(plan, tmp_path / "out.safetensors")
    _, output = read_safetensors_header(tmp_path / "out.safetensors")
    assert all(output["vision." + n].shape == shape for n, shape in shapes.items())
    assert (
        tensor_bytes(output["vision.patch_embed.proj.bias"])
        == tensors["model.visual.patch_embed.proj.bias"][1].tobytes()
    )
    assert (
        tensor_bytes(output["vision.blocks.1.mlp.linear_fc2.weight"])
        == tensors["model.visual.blocks.1.mlp.linear_fc2.weight"][1].tobytes()
    )


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_vision_requires_bf16_output(tmp_path, dtype):
    source = tmp_path / "source"
    vision_fixture(source)
    with pytest.raises(UnsupportedConversion):
        build_qwen35_plan(source, dtype=dtype, include_vision=True)


@pytest.mark.parametrize("case", ["missing", "extra", "shape", "dtype", "deepstack", "heads", "unknown"])
def test_vision_plan_rejects_invalid_inputs(tmp_path, case):
    source = tmp_path / "source"
    config, tensors, _ = vision_fixture(source)
    if case == "missing":
        del tensors["model.visual.blocks.1.mlp.linear_fc2.bias"]
    elif case == "extra":
        tensors["model.visual.extra.weight"] = ("BF16", np.zeros((1, 1), dtype="<u2"))
    elif case == "shape":
        tensors["model.visual.patch_embed.proj.weight"] = ("BF16", np.zeros((3, 3), dtype="<u2"))
    elif case == "dtype":
        tensors["model.visual.pos_embed.weight"] = ("F16", np.zeros((5, 4), dtype="<f2"))
    elif case == "deepstack":
        config["vision_config"]["deepstack_visual_indexes"] = [0]
    elif case == "heads":
        config["vision_config"]["num_heads"] = 3
    else:
        config["vision_config"]["unknown"] = True
    (source / "config.json").write_text(json.dumps(config))
    from test_streaming import save_tensors

    save_tensors(source / "model.safetensors", tensors)
    error = UnsupportedConversion if case == "unknown" else ConversionError
    with pytest.raises(error):
        build_qwen35_plan(source, include_vision=True)
