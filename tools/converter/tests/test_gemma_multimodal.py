"""Gemma 3 wrapper text exports and strictly accounted modal omissions."""

import hashlib
import json
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest
from test_gemma import gemma_fixture
from test_mlx import save_mlx_fixture
from test_mlx_gemma import mlx_gemma_fixture
from test_streaming import save_tensors, tensor_bytes

from stark_convert.adapters.gemma import build_gemma_plan
from stark_convert.streaming import (
    ConversionError,
    UnsupportedConversion,
    load_checkpoint,
    read_safetensors_header,
    write_plan,
)


def wrapper_fixture(
    path,
    *,
    bits=None,
    prefix="language_model.model",
    modal_prefix="",
    dtype="BF16",
    biased=False,
    sharded=False,
    duplicate_head=False,
    dense_embedding=False,
    dense_down=False,
    no_modal=False,
    mlx_patch_layout=True,
):
    if bits is None:
        text, weights = gemma_fixture(path, "gemma3_text", dtype=dtype, biased=biased)
        if duplicate_head:
            weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
    else:
        text, weights = mlx_gemma_fixture(
            path,
            bits=bits,
            dtype=dtype,
            attention_bias=biased,
            dense_embedding=dense_embedding,
            dense_down=dense_down,
            duplicate_head=duplicate_head,
        )
    tensors = {
        prefix + name[len("model") :]
        if name.startswith("model.")
        else ("model.language_model.lm_head" if prefix == "model.language_model" else "language_model.lm_head")
        + name[len("lm_head") :]: value
        for name, value in weights.items()
    }
    config = {
        "architectures": ["Gemma3ForConditionalGeneration"],
        "model_type": "gemma3",
        "text_config": {
            key: value for key, value in text.items() if key not in {"quantization", "quantization_config"}
        },
        "vision_config": {
            "model_type": "siglip_vision_model",
            "hidden_size": 8,
            "intermediate_size": 12,
            "num_attention_heads": 2,
            "num_hidden_layers": 2,
            "image_size": 8,
            "patch_size": 2,
            "vision_use_head": False,
            "skip_vision": True,
        },
        "boi_token_index": 6,
        "eoi_token_index": 7,
        "image_token_index": 8,
        "mm_tokens_per_image": 4,
        "eos_token_id": 1,
        "torch_dtype": text["torch_dtype"],
        "initializer_range": 0.02,
        "transformers_version": "4.50.0",
    }
    if bits is not None:
        config["quantization"] = text["quantization"]
    if not no_modal:

        def floating(name, shape):
            values = np.arange(np.prod(shape), dtype="<f4").reshape(shape) / 8 + len(tensors)
            tensors[modal_prefix + name] = "BF16", values.astype(ml_dtypes.bfloat16)

        root = "vision_tower.vision_model."
        shape = (8, 2, 2, 3) if mlx_patch_layout else (8, 3, 2, 2)
        floating(root + "embeddings.patch_embedding.weight", shape)
        floating(root + "embeddings.patch_embedding.bias", (8,))
        floating(root + "embeddings.position_embedding.weight", (16, 8))
        for layer in range(2):
            base = root + f"encoder.layers.{layer}."
            for norm in ("layer_norm1", "layer_norm2"):
                for plane in ("weight", "bias"):
                    floating(base + norm + "." + plane, (8,))
            for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
                floating(base + "self_attn." + projection + ".weight", (8, 8))
                floating(base + "self_attn." + projection + ".bias", (8,))
            for projection, rows, columns in (("fc1", 12, 8), ("fc2", 8, 12)):
                floating(base + "mlp." + projection + ".weight", (rows, columns))
                floating(base + "mlp." + projection + ".bias", (rows,))
        for plane in ("weight", "bias"):
            floating(root + "post_layernorm." + plane, (8,))
        floating("multi_modal_projector.mm_input_projection_weight", (8, text["hidden_size"]))
        floating("multi_modal_projector.mm_soft_emb_norm.weight", (8,))
    (path / "config.json").write_text(json.dumps(config))
    save_mlx_fixture(path, tensors, sharded=sharded)
    return config, tensors


@pytest.mark.parametrize("bits", [None, 4, 8])
@pytest.mark.parametrize("prefix", ["language_model.model", "model.language_model", "model"])
@pytest.mark.parametrize(
    "dtype,target", [(a, b) for a in ("BF16", "F16", "F32") for b in ("bfloat16", "float16", "float32")]
)
def test_wrapper_text_payloads_match_plain_gemma_and_account_for_all_sources(tmp_path, bits, prefix, dtype, target):
    source = tmp_path / "source"
    config, tensors = wrapper_fixture(source, bits=bits, prefix=prefix, dtype=dtype, sharded=True, biased=True)
    plan = build_gemma_plan(source, family="gemma3", dtype=target)
    assert plan.source_tensors == len(tensors)
    assert plan.input_bytes == sum(info.nbytes for info in load_checkpoint(source).values())
    assert set(plan.input_tensors) == set(load_checkpoint(source).values())
    assert all("vision" not in operation.name and "projector" not in operation.name for operation in plan.operations)
    assert not any("vision" in name or "projector" in name for name in plan.decoder_config)
    stats = write_plan(plan, tmp_path / "wrapper.safetensors", chunk_bytes=96)
    plain = tmp_path / "plain"
    plain.mkdir()
    text = dict(config["text_config"])
    if bits is not None:
        text["quantization"] = config["quantization"]
    (plain / "config.json").write_text(json.dumps(text))
    selected = {
        "model" + name[len(prefix) :]: value for name, value in tensors.items() if name.startswith(prefix + ".")
    }
    save_mlx_fixture(plain, selected)
    plain_plan = build_gemma_plan(plain, family="gemma3_text", dtype=target)
    write_plan(plain_plan, tmp_path / "plain.safetensors", chunk_bytes=96)
    assert plan.decoder_config == plain_plan.decoder_config
    assert (tmp_path / "wrapper.safetensors").read_bytes() == (tmp_path / "plain.safetensors").read_bytes()
    assert stats["peak_buffer_bytes"] <= 96


@pytest.mark.parametrize("bits", [None, 4, 8])
@pytest.mark.parametrize("no_modal,modal_prefix,patch_layout", [(False, "model.", False), (True, "", True)])
def test_wrapper_recognized_variants_and_tied_head(tmp_path, bits, no_modal, modal_prefix, patch_layout):
    source = tmp_path / "source"
    _, tensors = wrapper_fixture(
        source,
        bits=bits,
        prefix="model.language_model",
        modal_prefix=modal_prefix,
        duplicate_head=True,
        no_modal=no_modal,
        mlx_patch_layout=patch_layout,
    )
    assert build_gemma_plan(source, family="gemma3").source_tensors == len(tensors)


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "shape",
        "dtype",
        "unknown",
        "extra_layer",
        "head",
        "modal_alias",
        "text_alias",
        "head_alias",
        "head_root",
    ],
)
def test_wrapper_tensor_corruption_or_unaccounted_omissions_fail(tmp_path, case):
    source = tmp_path / "source"
    _, tensors = wrapper_fixture(source, duplicate_head=True)
    name = "vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight"
    if case == "missing":
        del tensors[name]
    elif case == "shape":
        tensors[name] = "BF16", np.zeros((8, 7), dtype=ml_dtypes.bfloat16)
    elif case == "dtype":
        tensors[name] = "U32", np.zeros((8, 8), dtype="<u4")
    elif case == "unknown":
        tensors["vision_tower.unknown.weight"] = "BF16", np.zeros(8, dtype=ml_dtypes.bfloat16)
    elif case == "extra_layer":
        tensors[name.replace("layers.0", "layers.2")] = tensors[name]
    elif case == "head":
        tensors["language_model.lm_head.weight"] = "BF16", np.zeros((16, 8), dtype=ml_dtypes.bfloat16)
    elif case == "modal_alias":
        tensors.update(
            {
                "model." + name: value
                for name, value in tensors.items()
                if name.startswith("vision_tower.") or name.startswith("multi_modal_projector.")
            }
        )
    elif case == "text_alias":
        tensors["model.embed_tokens.weight"] = tensors["language_model.model.embed_tokens.weight"]
    elif case == "head_alias":
        tensors["lm_head.weight"] = tensors["language_model.lm_head.weight"]
    else:
        tensors["model.language_model.lm_head.weight"] = tensors.pop("language_model.lm_head.weight")
    save_mlx_fixture(source, tensors)
    with pytest.raises(ConversionError):
        build_gemma_plan(source, family="gemma3")


@pytest.mark.parametrize(
    "changes",
    [
        {"unknown": True},
        {"architectures": ["Gemma3ForCausalLM"]},
        {"text_config": []},
        {"vision_config": None},
        {"boi_token_index": True},
        {"image_token_index": 17},
        {"mm_tokens_per_image": 0},
        {"tie_word_embeddings": False},
        {"add_cross_attention": True},
    ],
)
def test_wrapper_invalid_or_unimplemented_outer_config_fails(tmp_path, changes):
    source = tmp_path / "source"
    config, _ = wrapper_fixture(source)
    config.update(changes)
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises((ConversionError, UnsupportedConversion)):
        build_gemma_plan(source, family="gemma3")


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("text_config", "unknown", True),
        ("text_config", "model_type", "gemma4_text"),
        ("vision_config", "unknown", True),
        ("vision_config", "model_type", "clip_vision_model"),
        ("vision_config", "vision_use_head", True),
        ("vision_config", "image_size", 7),
        ("vision_config", "skip_vision", "true"),
    ],
)
def test_wrapper_unrecognized_nested_semantics_fail(tmp_path, section, key, value):
    source = tmp_path / "source"
    config, _ = wrapper_fixture(source)
    config[section][key] = value
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises((ConversionError, UnsupportedConversion)):
        build_gemma_plan(source, family="gemma3")


def test_wrapper_outer_nested_quantization_conflicts_fail(tmp_path):
    source = tmp_path / "source"
    config, _ = wrapper_fixture(source, bits=4)
    config["text_config"]["quantization"] = {"bits": 8, "group_size": 32}
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises(ConversionError, match="quantization descriptions disagree"):
        build_gemma_plan(source, family="gemma3")


def test_wrapper_omitted_modal_sources_are_rechecked_before_write(tmp_path):
    source = tmp_path / "source"
    _, tensors = wrapper_fixture(source)
    text = {name: value for name, value in tensors.items() if name.startswith("language_model.")}
    modal = {name: value for name, value in tensors.items() if name not in text}
    save_tensors(source / "text.safetensors", text)
    save_tensors(source / "modal.safetensors", modal)
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {**dict.fromkeys(text, "text.safetensors"), **dict.fromkeys(modal, "modal.safetensors")}}
        )
    )
    plan = build_gemma_plan(source, family="gemma3")
    assert not any(
        info.path.name == "modal.safetensors" for operation in plan.operations for info in operation.sources
    )
    with (source / "modal.safetensors").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ConversionError):
        write_plan(plan, tmp_path / "output.safetensors", chunk_bytes=96)


def test_wrapper_exports_match_independent_pinned_lalamo(tmp_path):
    fixture = json.loads((Path(__file__).parent / "fixtures" / "gemma3_multimodal_exports.json").read_text())
    for name, case in fixture["cases"].items():
        source = tmp_path / name
        wrapper_fixture(source, **case["parameters"])
        (source / "config.json").write_text(json.dumps(case["config"]))
        plan = build_gemma_plan(source, family="gemma3", dtype=case["target"])
        output = tmp_path / (name + ".safetensors")
        write_plan(plan, output, chunk_bytes=96)
        metadata, tensors = read_safetensors_header(output)
        assert plan.decoder_config == case["decoder_config"]
        assert metadata == case["metadata"]
        actual = {
            key: {
                "dtype": tensor.dtype,
                "shape": list(tensor.shape),
                "sha256": hashlib.sha256(tensor_bytes(tensor)).hexdigest(),
            }
            for key, tensor in sorted(tensors.items())
        }
        assert actual == case["tensors"]
