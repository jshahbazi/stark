"""Catalogue layout checks reuse converters without opening weight payloads."""

import json
import math
import shutil
from dataclasses import replace
from functools import partial
from pathlib import Path

import pytest
from test_dense_adapters import dense_fixture
from test_gemma import gemma_fixture
from test_gemma3_vision import config_for_size
from test_gemma4 import gemma4_fixture
from test_gemma4_image_profiles import FIXTURES as GEMMA4_FIXTURES
from test_gemma4_image_profiles import header_tensors as gemma4_headers
from test_gemma4_vision import inventory as gemma4_vision_headers
from test_granite_smollm import fixture as granite_fixture
from test_lfm2 import lfm2_fixture
from test_llamba import llamba_fixture
from test_mlx import mlx_qwen_fixture
from test_mlx_dense import mlx_dense_fixture
from test_mlx_qwen35 import mlx_qwen35_fixture
from test_olmo import olmo_fixture
from test_phi import phi_fixture
from test_qwen35 import qwen35_fixture
from test_qwen35_vision import vision_fixture
from test_qwen38 import fixture as qwen38_fixture
from test_streaming import qwen_fixture, save_tensors

from stark_convert.adapters.gemma3_vision import gemma3_config, vision_shapes
from stark_convert.checkpoint_metadata import CheckpointMetadata
from stark_convert.planning import plan_from_headers, planner_for
from stark_convert.streaming import ConversionError, TensorInfo, UnsupportedConversion, load_checkpoint

CASES = [
    ("qwen3", partial(qwen_fixture, tied=True), "text"),
    ("qwen2", partial(dense_fixture, family="qwen2", tied=True), "text"),
    ("llama", partial(dense_fixture, family="llama", tied=True), "text"),
    ("mistral", partial(dense_fixture, family="mistral", tied=True), "text"),
    ("gemma2", partial(gemma_fixture, family="gemma2"), "text"),
    ("gemma3_text", partial(gemma_fixture, family="gemma3_text"), "text"),
    ("gemma4_text", gemma4_fixture, "text"),
    ("olmo3", olmo_fixture, "text"),
    ("phi3", partial(phi_fixture, duplicate_head=True), "text"),
    ("granite", granite_fixture, "text"),
    ("smollm3", partial(granite_fixture, case="smollm3"), "text"),
    ("qwen3_5", partial(qwen35_fixture, tied=True), "text"),
    ("qwen3_5", vision_fixture, "vision-language"),
    ("lfm2", lfm2_fixture, "text"),
    ("llamba", llamba_fixture, "text"),
    ("qwen4_exp", qwen38_fixture, "text"),
    ("qwen3", partial(mlx_qwen_fixture, tied=True), "text"),
    ("llama", partial(mlx_dense_fixture, family="llama", tied=True), "text"),
    ("smollm3", partial(mlx_dense_fixture, family="smollm3", tied=True), "text"),
    ("qwen3_5", partial(mlx_qwen35_fixture, tied=True), "text"),
]


@pytest.mark.parametrize("family,fixture,mode", CASES)
def test_header_planner_matches_normal_layout_without_source_files(tmp_path, family, fixture, mode):
    source = tmp_path / "source"
    fixture(source)
    full = planner_for(family, mode=mode)(source)
    metadata = CheckpointMetadata(json.loads((source / "config.json").read_text()), load_checkpoint(source))
    shutil.rmtree(source)
    assert plan_from_headers(family, metadata, mode=mode) == full


@pytest.mark.parametrize("family,fixture,mode", CASES)
@pytest.mark.parametrize("corruption", ["missing", "shape", "dtype"])
def test_header_planner_rejects_real_tensor_layout_errors(tmp_path, family, fixture, mode, corruption):
    source = tmp_path / "source"
    fixture(source)
    tensors = load_checkpoint(source)
    name = next(iter(tensors))
    if corruption == "missing":
        del tensors[name]
    elif corruption == "shape":
        tensors[name] = replace(tensors[name], shape=(1,))
    else:
        tensors[name] = replace(tensors[name], dtype="U8")
    metadata = CheckpointMetadata(json.loads((source / "config.json").read_text()), tensors)
    shutil.rmtree(source)
    with pytest.raises((ConversionError, UnsupportedConversion)):
        plan_from_headers(family, metadata, mode=mode)


def test_header_admission_defers_tied_payload_equality_to_normal_conversion(tmp_path):
    source = tmp_path / "source"
    config, tensors = qwen_fixture(source, tied=True)
    dtype, values = tensors["lm_head.weight"]
    tensors["lm_head.weight"] = dtype, values + 1
    save_tensors(source / "model.safetensors", tensors)
    metadata = CheckpointMetadata(config, load_checkpoint(source))
    plan_from_headers("qwen3", metadata)
    with pytest.raises(ConversionError, match="Tied lm_head differs"):
        planner_for("qwen3")(source)


def test_gemma4_image_header_plan_checks_real_pinned_decoder_and_vision_inventory():
    config = json.loads((GEMMA4_FIXTURES / "e2b.config.json").read_bytes())
    tensors = {**gemma4_headers(config), **gemma4_vision_headers()}
    metadata = CheckpointMetadata(config, tensors)
    plan = plan_from_headers("gemma4_text", metadata, mode="vision-language")
    assert any(operation.name.startswith("decoder.") for operation in plan.operations)
    assert any(operation.name.startswith("vision.") for operation in plan.operations)
    del tensors["model.embed_vision.embedding_projection.weight"]
    with pytest.raises(UnsupportedConversion, match="inventory"):
        plan_from_headers("gemma4_text", metadata, mode="vision-language")


def test_gemma3_image_header_plan_checks_complete_source_storage():
    config = config_for_size(2560)
    config["text_config"]["vocab_size"] = 262208
    text, _ = gemma3_config(config)
    dim, hidden, vocab = (text[key] for key in ("hidden_size", "intermediate_size", "vocab_size"))
    width = text["head_dim"]
    query_width, kv_width = text["num_attention_heads"] * width, text["num_key_value_heads"] * width
    shapes = {
        "language_model.model.embed_tokens.weight": (vocab, dim),
        "language_model.model.norm.weight": (dim,),
        **vision_shapes(dim),
    }
    for layer in range(text["num_hidden_layers"]):
        prefix = f"language_model.model.layers.{layer}"
        for norm in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            shapes[f"{prefix}.{norm}.weight"] = (dim,)
        for norm in ("q_norm", "k_norm"):
            shapes[f"{prefix}.self_attn.{norm}.weight"] = (width,)
        for name, shape in (
            ("self_attn.q_proj", (query_width, dim)),
            ("self_attn.k_proj", (kv_width, dim)),
            ("self_attn.v_proj", (kv_width, dim)),
            ("self_attn.o_proj", (dim, query_width)),
            ("mlp.up_proj", (hidden, dim)),
            ("mlp.gate_proj", (hidden, dim)),
            ("mlp.down_proj", (dim, hidden)),
        ):
            shapes[f"{prefix}.{name}.weight"] = shape
    tensors = {
        name: TensorInfo(Path("unavailable.safetensors"), "BF16", shape, 0, math.prod(shape) * 2, (0, 0, 0, 0))
        for name, shape in shapes.items()
    }
    metadata = CheckpointMetadata(config, tensors)
    plan = plan_from_headers("gemma3", metadata, mode="vision-language")
    assert len(plan.input_tensors) == len(tensors)
    name = "multi_modal_projector.mm_input_projection_weight"
    tensors[name] = replace(tensors[name], dtype="F32")
    with pytest.raises((ConversionError, UnsupportedConversion)):
        plan_from_headers("gemma3", metadata, mode="vision-language")
