"""Gemma 4 native text topology, source binding and byte-exact payload tests."""

import json

import ml_dtypes
import numpy as np
import pytest
from test_streaming import save_tensors, tensor_bytes

from stark_convert.adapters.gemma4 import _activation_scalar, build_gemma4_plan, normalize_gemma4_config
from stark_convert.streaming import ConversionError, UnsupportedConversion, read_safetensors_header, write_plan


def gemma4_fixture(path, *, wrapper=True, double_wide=True, redundant=True, dim=16, sanitized=False):
    path.mkdir()
    config = {
        "model_type": "gemma4_text",
        "architectures": ["Gemma4ForCausalLM"],
        "hidden_size": dim,
        "intermediate_size": 32,
        "num_hidden_layers": 6,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "global_head_dim": 16,
        "vocab_size": 32,
        "vocab_size_per_layer_input": 32,
        "hidden_size_per_layer_input": 16,
        "num_kv_shared_layers": 2,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 1024,
        "sliding_window": 16,
        "tie_word_embeddings": True,
        "hidden_activation": "gelu_pytorch_tanh",
        "use_double_wide_mlp": double_wide,
        "eos_token_id": 1,
        "layer_types": ["sliding_attention", "full_attention"] * 3,
        "rope_parameters": {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "proportional", "rope_theta": 1000000.0, "partial_rotary_factor": 0.25},
        },
    }
    prefix = ("language_model.model" if sanitized else "model.language_model") if wrapper else "model"
    outer = (
        {
            "model_type": "gemma4",
            "architectures": ["Gemma4ForConditionalGeneration"],
            "text_config": config,
            "vision_config": {"model_type": "gemma4_vision"},
            "eos_token_id": [1, 106],
        }
        if wrapper
        else config
    )
    (path / "config.json").write_text(json.dumps(outer))
    tensors = {}

    def add(name, shape):
        values = (np.arange(np.prod(shape), dtype="<f4").reshape(shape) % 23 - 11) / 32
        tensors[name] = ("BF16", values.astype(ml_dtypes.bfloat16))

    add(prefix + ".embed_tokens.weight", (32, dim))
    add(prefix + ".embed_tokens_per_layer.weight", (32, 96))
    add(prefix + ".per_layer_model_projection.weight", (96, dim))
    add(prefix + ".per_layer_projection_norm.weight", (16,))
    add(prefix + ".norm.weight", (dim,))
    for layer in range(6):
        src = prefix + f".layers.{layer}"
        width = 8 if layer % 2 == 0 else 16
        intermediate = 64 if double_wide and layer >= 4 else 32
        for name in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
            "post_per_layer_input_norm",
        ):
            add(src + "." + name + ".weight", (dim,))
        add(src + ".layer_scalar", (1,))
        add(src + ".self_attn.q_norm.weight", (width,))
        add(src + ".self_attn.q_proj.weight", (2 * width, dim))
        if layer < 4 or redundant:
            add(src + ".self_attn.k_norm.weight", (width,))
            add(src + ".self_attn.k_proj.weight", (width, dim))
            add(src + ".self_attn.v_proj.weight", (width, dim))
        add(src + ".self_attn.o_proj.weight", (dim, 2 * width))
        add(src + ".mlp.up_proj.weight", (intermediate, dim))
        add(src + ".mlp.gate_proj.weight", (intermediate, dim))
        add(src + ".mlp.down_proj.weight", (dim, intermediate))
        add(src + ".per_layer_input_gate.weight", (16, dim))
        add(src + ".per_layer_projection.weight", (dim, 16))
    if wrapper:
        modality_root = "" if sanitized else "model."
        add(modality_root + "vision_tower.test.weight", (16, 16))
    save_tensors(path / "model.safetensors", tensors)
    return outer, tensors


@pytest.mark.parametrize("wrapper,redundant,double_wide", [(True, True, True), (False, False, False)])
def test_gemma4_native_topology_and_payloads(tmp_path, wrapper, redundant, double_wide):
    source = tmp_path / "source"
    _, tensors = gemma4_fixture(source, wrapper=wrapper, redundant=redundant, double_wide=double_wide)
    plan = build_gemma4_plan(source)
    layers = plan.decoder_config["transformer_config"]["layer_configs"]
    assert [layer["kv_source_layer_index"] for layer in layers] == [None, None, None, None, 2, 3]
    assert [layer["mixer_config"]["head_dim"] for layer in layers] == [8, 16, 8, 16, 8, 16]
    assert [layer["mixer_config"]["sliding_window_size"] for layer in layers] == [16, None] * 3
    assert layers[1]["rope_config"] == {
        "type": "ProportionalRoPEConfig",
        "base": 1000000.0,
        "head_dim": 16,
        "max_sequence_length": 1024,
        "rotary_dim": 4,
    }
    assert layers[4]["hidden_dim"] == (64 if double_wide else None)
    assert all(layer["has_post_layer_scalar"] and layer["ple_config"] for layer in layers)
    assert all(layer["ple_config"]["reference_bf16"] for layer in layers)
    assert plan.decoder_config["ple_model_config"]["reference_bf16"] is True
    assert plan.decoder_config["ple_model_config"]["input_scale"] == 0.7071067690849304
    assert all(layer["mixer_config"]["scale"] == 1.0 for layer in layers)
    assert all(layer["mixer_config"]["normalize_values"] for layer in layers)
    assert plan.decoder_config["ple_model_config"]["ple_embed_scale"] == 4.0
    assert plan.source_tensors == len(tensors)
    assert len(plan.input_tensors) == len(tensors)
    output = tmp_path / "output.safetensors"
    stats = write_plan(plan, output, chunk_bytes=96)
    assert stats["peak_buffer_bytes"] <= 96
    _, converted = read_safetensors_header(output)
    prefix = "model.language_model" if wrapper else "model"
    cases = {
        "decoder.per_layer_embedding.token_embedding.weights": [prefix + ".embed_tokens_per_layer.weight"],
        "decoder.per_layer_embedding.model_projection.weights.weights": [
            prefix + ".per_layer_model_projection.weight"
        ],
        "decoder.transformer.layers.0.mixer.qkvg_projection.weights.weights": [
            prefix + f".layers.0.self_attn.{branch}_proj.weight" for branch in ("q", "k", "v")
        ],
        "decoder.transformer.layers.4.mixer.qkvg_projection.weights.weights": [
            prefix + ".layers.4.self_attn.q_proj.weight"
        ],
        "decoder.transformer.layers.4.ple.gate.weights.weights": [prefix + ".layers.4.per_layer_input_gate.weight"],
    }
    for name, origins in cases.items():
        assert tensor_bytes(converted[name]) == b"".join(tensors[origin][1].tobytes() for origin in origins)
    assert not any("vision" in name for name in converted)
    assert tensor_bytes(converted["decoder.transformer.layers.0.mixer.query_norm.scales"]) == (
        tensors[prefix + ".layers.0.self_attn.q_norm.weight"][1].astype("<f4").tobytes()
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("model_type", "gemma4_unified"),
        ("enable_moe_block", True),
        ("attention_k_eq_v", True),
        ("hidden_size_per_layer_input", 0),
        ("rms_norm_eps", 1e-5),
        ("attention_bias", True),
        ("is_encoder_decoder", True),
        ("num_kv_shared_layers", 6),
        ("new_attention_semantics", True),
    ],
)
def test_gemma4_unimplemented_variants_fail_closed(tmp_path, key, value):
    source = tmp_path / "source"
    config, _ = gemma4_fixture(source, wrapper=False)
    config[key] = value
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises((ConversionError, UnsupportedConversion)):
        build_gemma4_plan(source)


def test_gemma4_embedding_scalars_are_explicitly_converted_to_activation_dtype():
    # Canonical HF's Gemma4TextScaledWordEmbedding explicitly casts these
    # buffers to the embedding weight dtype before multiplication.
    assert _activation_scalar(1536**0.5, "BF16") == 39.25
    assert _activation_scalar(256**0.5, "BF16") == 16.0


def test_gemma4_canonical_ple_scalars_retain_float32_opmath(tmp_path):
    source = tmp_path / "source"
    gemma4_fixture(source, dim=48)
    config = build_gemma4_plan(source).decoder_config
    assert config["embedding_config"]["input_scale"] == 6.9375
    assert config["ple_model_config"]["ple_embed_scale"] == 4.0
    assert config["ple_model_config"]["model_projection_scale"] == 0.14433756470680237
    assert config["ple_model_config"]["input_scale"] == 0.7071067690849304


def test_gemma4_wrapper_preserves_end_of_turn_tokens_and_tower_authority(tmp_path):
    source = tmp_path / "source"
    config, _ = gemma4_fixture(source)
    normalized = normalize_gemma4_config(config)
    assert normalized["eos_token_id"] == [1, 106]
    assert normalized["vocab_size"] == 32
    assert normalized["model_type"] == "gemma4_text"
    del config["vision_config"]
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises(ConversionError, match="unexpected"):
        build_gemma4_plan(source)


@pytest.mark.parametrize("invalid", ["undeclared", "mixed_namespace", "unrelated"])
def test_gemma4_sanitized_wrapper_modality_omissions_fail_closed(tmp_path, invalid):
    source = tmp_path / "source"
    config, tensors = gemma4_fixture(source, sanitized=True)
    if invalid == "undeclared":
        del config["vision_config"]
    else:
        unexpected = "model.vision_tower.extra.weight" if invalid == "mixed_namespace" else "other.weight"
        tensors[unexpected] = tensors["vision_tower.test.weight"]
    (source / "config.json").write_text(json.dumps(config))
    save_tensors(source / "model.safetensors", tensors)
    with pytest.raises(ConversionError, match="unexpected"):
        build_gemma4_plan(source)


def test_gemma4_shared_owner_schedule_is_validated(tmp_path):
    source = tmp_path / "source"
    config, _ = gemma4_fixture(source, wrapper=False)
    config["layer_types"] = ["sliding_attention"] * 4 + ["full_attention"] * 2
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises(ConversionError, match="owner"):
        build_gemma4_plan(source)


def test_gemma4_official_wrapper_exports_complete_native_metadata(tmp_path):
    from tokenizers import Tokenizer, models

    from stark_convert.cli import convert

    source = tmp_path / "source"
    config, _ = gemma4_fixture(source)
    config["eos_token_id"] = [1, 15]
    (source / "config.json").write_text(json.dumps(config))
    tokenizer = Tokenizer(models.WordLevel({f"t{index}": index for index in range(32)}, unk_token="t3"))
    tokenizer.save(str(source / "tokenizer.json"))
    (source / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "bos_token": "t2",
                "eos_token": "t1",
                "pad_token": "t0",
                "unk_token": "t3",
                "chat_template": "{{ bos_token }}{% for message in messages %}{{ message.content }}{% endfor %}",
            }
        )
    )
    output = tmp_path / "native"
    result = convert(source, output, repo=None, architecture="gemma4_text", metadata_profile="hf-generic")
    native = json.loads((output / "config.json").read_text())
    assert result["status"] == "ok"
    assert native["generation_config"]["stop_token_ids"] == [1, 15]
    assert native["decoder_config"]["vocab_size"] == 32
    assert native["decoder_config"]["ple_model_config"]["num_layers"] == 6
    assert result["artifact_contract"]["input_modalities"] == ["text"]


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("sanitized_wrapper", [False, True])
def test_gemma4_affine_matrices_preserve_packed_words_and_source_binding(tmp_path, bits, sanitized_wrapper):
    source = tmp_path / "source"
    config, tensors = gemma4_fixture(source, wrapper=sanitized_wrapper, sanitized=sanitized_wrapper, redundant=False)
    if sanitized_wrapper:
        config["audio_config"] = {"model_type": "gemma4_audio"}
        tensors["audio_tower.test.weight"] = tensors["vision_tower.test.weight"]
        tensors["embed_audio.embedding_projection.weight"] = tensors["vision_tower.test.weight"]
        tensors["embed_vision.embedding_projection.weight"] = tensors["vision_tower.test.weight"]
    config["quantization"] = {"bits": bits, "group_size": 16}
    (source / "config.json").write_text(json.dumps(config))
    for name, (_, array) in list(tensors.items()):
        if array.ndim != 2:
            continue
        rows, columns = array.shape
        packed = np.arange(rows * columns * bits // 32, dtype="<u4").reshape(rows, columns * bits // 32)
        packed = packed * np.uint32(2654435761)
        tensors[name] = ("U32", packed)
        for plane, value in (("scales", 0.125), ("biases", -0.25)):
            tensors[name.removesuffix(".weight") + "." + plane] = (
                "BF16",
                np.full((rows, columns // 16), value, dtype=ml_dtypes.bfloat16),
            )
    save_tensors(source / "model.safetensors", tensors)
    plan = build_gemma4_plan(source)
    output = tmp_path / "output.safetensors"
    stats = write_plan(plan, output, chunk_bytes=96)
    metadata, converted = read_safetensors_header(output)
    assert stats["peak_buffer_bytes"] <= 96
    assert plan.source_tensors == len(tensors)
    assert len(plan.input_tensors) == len(tensors)
    names = {
        (info.path, info.offset): name
        for info, name in (
            (info, name) for name, info in read_safetensors_header(source / "model.safetensors")[1].items()
        )
    }
    for operation in plan.operations:
        arrays = [tensors[names[(info.path, info.offset)]][1] for info in operation.sources]
        if operation.kind == "mlx_repack":
            assert tensor_bytes(converted[operation.name]) == b"".join(array.tobytes() for array in arrays)
        elif operation.kind == "mlx_params":
            planes = np.concatenate(arrays).T
            padded = np.zeros(operation.shape, dtype=ml_dtypes.bfloat16)
            padded[:, : planes.shape[1]] = planes
            assert tensor_bytes(converted[operation.name]) == padded.tobytes()
    assert json.loads(metadata["decoder.per_layer_embedding.token_embedding.spec"]) == {
        "type": "MLXSpec",
        "layout": "input_output",
        "bits": bits,
        "group_size": 16,
    }
