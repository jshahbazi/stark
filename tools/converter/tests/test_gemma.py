"""Complete Gemma text configs and independent bounded tensor expectations."""

import json
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest
from test_streaming import save_tensors, tensor_bytes

from stark_convert.adapters.gemma import build_gemma_plan
from stark_convert.streaming import ConversionError, UnsupportedConversion, read_safetensors_header, write_plan

BASELINES = json.loads((Path(__file__).parent / "fixtures" / "gemma_decoder_configs.json").read_text())
DTYPES = {"BF16": ml_dtypes.bfloat16, "F16": np.dtype("<f2"), "F32": np.dtype("<f4")}
TARGETS = {"BF16": "bfloat16", "F16": "float16", "F32": "float32"}


def gemma_fixture(path: Path, family: str, *, dtype="BF16", biased=False, sharded=False, head_dim=2):
    path.mkdir()
    config = {
        "model_type": family,
        "architectures": ["Gemma2ForCausalLM" if family == "gemma2" else "Gemma3ForCausalLM"],
        "hidden_size": 8,
        "intermediate_size": 12,
        "num_hidden_layers": 6,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": head_dim,
        "vocab_size": 16,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0 if family == "gemma2" else 1000000.0,
        "max_position_embeddings": 64,
        "tie_word_embeddings": True,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_activation": "gelu_pytorch_tanh",
        "attention_bias": biased,
        "attention_dropout": 0.0,
        "attn_logit_softcapping": 50.0 if family == "gemma2" else None,
        "final_logit_softcapping": 30.0 if family == "gemma2" else None,
        "query_pre_attn_scalar": 2.0,
        "sliding_window": 16,
        "cache_implementation": "hybrid",
        "eos_token_id": 1,
        "bos_token_id": 0,
        "pad_token_id": 0,
        "torch_dtype": TARGETS[dtype],
        "initializer_range": 0.02,
        "transformers_version": "4.50.0",
        "use_cache": True,
    }
    if family == "gemma3_text":
        config.update(sliding_window_pattern=3, rope_local_base_freq=10000.0, rope_scaling=None)
    (path / "config.json").write_text(json.dumps(config))
    tensors = {}

    def add(name, shape):
        values = np.arange(np.prod(shape), dtype="<f4").reshape(shape) / 8 + len(tensors)
        tensors[name] = (dtype, values.astype(DTYPES[dtype]))

    add("model.embed_tokens.weight", (16, 8))
    add("model.norm.weight", (8,))
    for layer in range(6):
        src = f"model.layers.{layer}"
        for name in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            add(src + "." + name + ".weight", (8,))
        if family == "gemma3_text":
            add(src + ".self_attn.q_norm.weight", (head_dim,))
            add(src + ".self_attn.k_norm.weight", (head_dim,))
        for name, width in (("q", 4 * head_dim), ("k", 2 * head_dim), ("v", 2 * head_dim)):
            add(src + f".self_attn.{name}_proj.weight", (width, 8))
            if biased:
                add(src + f".self_attn.{name}_proj.bias", (width,))
        add(src + ".self_attn.o_proj.weight", (8, 4 * head_dim))
        if biased and family == "gemma3_text":
            add(src + ".self_attn.o_proj.bias", (8,))
        add(src + ".mlp.up_proj.weight", (12, 8))
        add(src + ".mlp.gate_proj.weight", (12, 8))
        add(src + ".mlp.down_proj.weight", (8, 12))
    if sharded:
        weights = {}
        for shard in range(2):
            name = f"model-{shard}.safetensors"
            contents = dict(list(tensors.items())[shard::2])
            save_tensors(path / name, contents)
            weights.update(dict.fromkeys(contents, name))
        (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weights}))
    else:
        save_tensors(path / "model.safetensors", tensors)
    return config, tensors


def update_config(source, config, **changes):
    config.update(changes)
    (source / "config.json").write_text(json.dumps(config))


@pytest.mark.parametrize("case", BASELINES["cases"].values(), ids=BASELINES["cases"])
def test_complete_gemma_config_matches_frozen_external_reference(tmp_path, case):
    source = tmp_path / "source"
    gemma_fixture(source, case["family"], biased=case["config"]["attention_bias"])
    (source / "config.json").write_text(json.dumps(case["config"]))
    assert build_gemma_plan(source, family=case["family"]).decoder_config == case["decoder_config"]


@pytest.mark.parametrize("family", ["gemma2", "gemma3_text"])
@pytest.mark.parametrize("source_dtype", DTYPES)
@pytest.mark.parametrize("target_dtype", DTYPES)
@pytest.mark.parametrize("biased,sharded", [(False, False), (True, True)])
def test_gemma_payloads_are_exact_and_workspace_bounded(tmp_path, family, source_dtype, target_dtype, biased, sharded):
    source = tmp_path / "source"
    _, tensors = gemma_fixture(source, family, dtype=source_dtype, biased=biased, sharded=sharded, head_dim=4)
    plan = build_gemma_plan(source, family=family, dtype=TARGETS[target_dtype])
    output = tmp_path / "output.safetensors"
    stats = write_plan(plan, output, chunk_bytes=96)
    metadata, converted = read_safetensors_header(output)
    assert stats["peak_buffer_bytes"] <= 96
    expected = {}

    def projected(out, *names):
        expected[out] = b"".join(tensors[name][1].astype(DTYPES[target_dtype]).tobytes() for name in names)

    def norm(out, name):
        expected[out + ".scales"] = tensors[name][1].astype("<f4").tobytes()

    projected("decoder.embedding.embedding.weights", "model.embed_tokens.weight")
    norm("decoder.transformer.output_norm", "model.norm.weight")
    for layer in range(6):
        src, out = f"model.layers.{layer}", f"decoder.transformer.layers.{layer}"
        for destination, origin in (
            ("pre_mixer_norm", "input_layernorm"),
            ("post_mixer_norm", "post_attention_layernorm"),
            ("pre_mlp_norm", "pre_feedforward_layernorm"),
            ("post_mlp_norm", "post_feedforward_layernorm"),
        ):
            norm(out + "." + destination, src + "." + origin + ".weight")
        if family == "gemma3_text":
            norm(out + ".mixer.query_norm", src + ".self_attn.q_norm.weight")
            norm(out + ".mixer.key_norm", src + ".self_attn.k_norm.weight")
        projected(
            out + ".mixer.qkvg_projection.weights.weights",
            *(src + f".self_attn.{name}_proj.weight" for name in ("q", "k", "v")),
        )
        projected(out + ".mixer.out_projection.weights.weights", src + ".self_attn.o_proj.weight")
        projected(
            out + ".mlp.up_projection.weights.weights",
            src + ".mlp.up_proj.weight",
            src + ".mlp.gate_proj.weight",
        )
        projected(out + ".mlp.down_projection.weights.weights", src + ".mlp.down_proj.weight")
        if biased:
            projected(
                out + ".mixer.qkvg_projection.biases",
                *(src + f".self_attn.{name}_proj.bias" for name in ("q", "k", "v")),
            )
            if family == "gemma3_text":
                projected(out + ".mixer.out_projection.biases", src + ".self_attn.o_proj.bias")
    assert set(converted) == set(expected)
    assert all(tensor_bytes(converted[name]) == payload for name, payload in expected.items())
    assert len(metadata) == 25
    assert all(json.loads(spec)["type"] == "FullPrecisionSpec" for spec in metadata.values())
    assert json.loads(metadata["decoder.embedding.embedding.spec"])["layout"] == "input_output"


@pytest.mark.parametrize("family", ["gemma2", "gemma3_text"])
def test_tied_duplicate_is_consumed_and_verified(tmp_path, family):
    source = tmp_path / "source"
    _, tensors = gemma_fixture(source, family)
    tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"]
    save_tensors(source / "model.safetensors", tensors)
    assert build_gemma_plan(source, family=family).source_tensors == len(tensors)
    tensors["lm_head.weight"] = ("BF16", np.zeros((16, 8), dtype=ml_dtypes.bfloat16))
    save_tensors(source / "model.safetensors", tensors)
    with pytest.raises(ConversionError, match="^Tied Gemma lm_head differs"):
        build_gemma_plan(source, family=family)


@pytest.mark.parametrize("family", ["gemma2", "gemma3_text"])
@pytest.mark.parametrize("case", ["missing", "shape", "extra"])
def test_layout_corruption_is_rejected_before_unknown_semantics(tmp_path, family, case):
    source = tmp_path / "source"
    config, tensors = gemma_fixture(source, family)
    update_config(source, config, unknown_semantics=True)
    if case == "missing":
        del tensors["model.layers.5.post_feedforward_layernorm.weight"]
    elif case == "shape":
        tensors["model.layers.5.self_attn.q_proj.weight"] = ("BF16", np.zeros((2, 8), dtype=ml_dtypes.bfloat16))
    else:
        tensors["extra.weight"] = ("BF16", np.zeros(2, dtype=ml_dtypes.bfloat16))
    save_tensors(source / "model.safetensors", tensors)
    with pytest.raises(ConversionError):
        build_gemma_plan(source, family=family)


@pytest.mark.parametrize("family", ["gemma2", "gemma3_text"])
@pytest.mark.parametrize(
    "changes",
    [
        {"hidden_size": True},
        {"num_attention_heads": 3},
        {"head_dim": 3},
        {"query_pre_attn_scalar": 0.0},
        {"rms_norm_eps": False},
        {"tie_word_embeddings": "false"},
        {"attention_dropout": 2.0},
        {"attention_bias": "false"},
        {"sliding_window": 0},
        {"attn_logit_softcapping": True},
    ],
)
def test_invalid_gemma_config_fails_before_unknown_semantics(tmp_path, family, changes):
    source = tmp_path / "source"
    config, _ = gemma_fixture(source, family)
    update_config(source, config, unknown_semantics=True, **changes)
    with pytest.raises(ConversionError):
        build_gemma_plan(source, family=family)


@pytest.mark.parametrize("family", ["gemma2", "gemma3_text"])
@pytest.mark.parametrize(
    "changes",
    [
        {"hidden_activation": "gelu"},
        {"unknown_semantics": True},
        {"attention_dropout": 0.1},
        {"use_bidirectional_attention": True},
        {"tie_word_embeddings": False},
        {"rope_parameters": {"rope_type": "default"}},
    ],
)
def test_unqualified_gemma_config_fails_closed(tmp_path, family, changes):
    source = tmp_path / "source"
    config, _ = gemma_fixture(source, family)
    update_config(source, config, **changes)
    with pytest.raises(UnsupportedConversion):
        build_gemma_plan(source, family=family)


@pytest.mark.parametrize("family,error", [("gemma2", UnsupportedConversion), ("gemma3_text", ConversionError)])
def test_gemma_quantization_description_missing_group_is_rejected(tmp_path, family, error):
    source = tmp_path / "source"
    config, _ = gemma_fixture(source, family)
    update_config(source, config, quantization_config={"bits": 4})
    with pytest.raises(error):
        build_gemma_plan(source, family=family)


@pytest.mark.parametrize("family", ["gemma2", "gemma3_text"])
@pytest.mark.parametrize("mixed", ["self_attn.q_proj.weight", "mlp.up_proj.weight"])
def test_mixed_dtype_gemma_fusion_checks_complete_layout(tmp_path, family, mixed):
    source = tmp_path / "source"
    _, tensors = gemma_fixture(source, family)
    name = "model.layers.0." + mixed
    tensors[name] = ("F32", tensors[name][1].astype("<f4"))
    save_tensors(source / "model.safetensors", tensors)
    with pytest.raises(UnsupportedConversion, match="^Mixed-dtype Gemma fusion"):
        build_gemma_plan(source, family=family)
    del tensors["model.layers.5.post_feedforward_layernorm.weight"]
    save_tensors(source / "model.safetensors", tensors)
    with pytest.raises(ConversionError, match="^Gemma checkpoint is missing"):
        build_gemma_plan(source, family=family)


@pytest.mark.parametrize(
    "scaling",
    [
        {},
        {"rope_type": "linear", "factor": False},
        {"rope_type": "yarn", "factor": 2, "beta_fast": 1, "beta_slow": 32, "original_max_position_embeddings": 16},
        {"rope_type": "yarn", "factor": 2, "beta_fast": 32, "beta_slow": 1, "original_max_position_embeddings": 0},
    ],
)
def test_invalid_gemma3_global_rope_rejected(tmp_path, scaling):
    source = tmp_path / "source"
    config, _ = gemma_fixture(source, "gemma3_text")
    update_config(source, config, rope_scaling=scaling)
    with pytest.raises(ConversionError):
        build_gemma_plan(source, family="gemma3_text")


@pytest.mark.parametrize("annotation", [True, 0, "6"])
def test_malformed_private_window_annotation_rejected(tmp_path, annotation):
    source = tmp_path / "source"
    config, _ = gemma_fixture(source, "gemma3_text")
    update_config(source, config, _sliding_window_pattern=annotation)
    with pytest.raises(ConversionError):
        build_gemma_plan(source, family="gemma3_text")


@pytest.mark.parametrize(
    "changes",
    [
        {"_sliding_window_pattern": 6},
        {"layer_types": ["sliding_attention"] * 6},
    ],
)
def test_private_annotation_and_layer_types_must_match_reference_windows(tmp_path, changes):
    source = tmp_path / "source"
    config, _ = gemma_fixture(source, "gemma3_text")
    # This synthetic public pattern is 3. A private annotation of 6 changes
    # semantics, and a layer list that omits the globals contradicts both.
    update_config(source, config, **changes)
    with pytest.raises(UnsupportedConversion):
        build_gemma_plan(source, family="gemma3_text")


@pytest.mark.parametrize("family", ["gemma3", "gemma4"])
def test_relabelled_text_is_not_a_valid_wrapper_or_new_generation(tmp_path, family):
    source = tmp_path / "source"
    config, _ = gemma_fixture(source, "gemma3_text")
    update_config(source, config, model_type=family)
    with pytest.raises(UnsupportedConversion, match="^Checkpoint model_type does not match"):
        build_gemma_plan(source, family="gemma3_text")
    with pytest.raises((ConversionError, UnsupportedConversion)):
        build_gemma_plan(source, family=family)
