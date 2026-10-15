"""Qwen3-4B configuration planning and private trained-checkpoint conversion."""

import hashlib
import json
import math
from pathlib import Path

import pytest
from test_cli import source_tooling  # noqa: F401
from test_streaming import qwen_fixture
from tokenizers import Tokenizer, models

from stark_convert import cli
from stark_convert.contracts import ConversionRequest
from stark_convert.qwen3 import build_qwen3_plan
from stark_convert.sources import resolve_source
from stark_convert.streaming import ConversionError, TensorInfo, UnsupportedConversion, read_safetensors_header

REPO = "Qwen/Qwen3-4B-Instruct-2507"
FIXTURE = Path(__file__).parent / "fixtures" / "qwen3_4b_instruct_2507.json"


def qwen3_descriptors(config, path):
    """Describe the complete public 4B layout without allocating its weights."""
    hidden = config["hidden_size"]
    intermediate = config["intermediate_size"]
    head = config["head_dim"]
    query = config["num_attention_heads"] * head
    kv = config["num_key_value_heads"] * head
    shapes = {
        "model.embed_tokens.weight": (config["vocab_size"], hidden),
        "model.norm.weight": (hidden,),
    }
    for index in range(config["num_hidden_layers"]):
        layer = f"model.layers.{index}"
        shapes.update(
            {
                layer + ".input_layernorm.weight": (hidden,),
                layer + ".post_attention_layernorm.weight": (hidden,),
                layer + ".self_attn.q_norm.weight": (head,),
                layer + ".self_attn.k_norm.weight": (head,),
                layer + ".self_attn.q_proj.weight": (query, hidden),
                layer + ".self_attn.k_proj.weight": (kv, hidden),
                layer + ".self_attn.v_proj.weight": (kv, hidden),
                layer + ".self_attn.o_proj.weight": (hidden, query),
                layer + ".mlp.up_proj.weight": (intermediate, hidden),
                layer + ".mlp.gate_proj.weight": (intermediate, hidden),
                layer + ".mlp.down_proj.weight": (hidden, intermediate),
            }
        )
    return {
        name: TensorInfo(path, "BF16", shape, 0, math.prod(shape) * 2, (0, 0, 0, 0)) for name, shape in shapes.items()
    }


def test_qwen3_4b_instruct_2507_exact_public_config_plans_without_weights(tmp_path):
    fixture = json.loads(FIXTURE.read_text())
    config = fixture["config"]
    serialized = json.dumps(config, indent=2).encode()
    assert fixture["source"]["repo"] == REPO
    assert fixture["source"]["revision"] == "cdbee75f17c01a7cc42f958dc650907174af0554"
    assert hashlib.sha256(serialized).hexdigest() == fixture["source"]["sha256"]
    (tmp_path / "config.json").write_bytes(serialized)
    tensors = qwen3_descriptors(config, tmp_path / "unallocated.safetensors")
    plan = build_qwen3_plan(tmp_path, tensors=tensors)

    assert not (tmp_path / "unallocated.safetensors").exists()
    assert config["head_dim"] == 128 != config["hidden_size"] // config["num_attention_heads"]
    decoder = plan.decoder_config
    assert decoder["embedding_config"]["type"] == "TiedEmbeddingConfig"
    assert decoder["vocab_size"] == 151936
    transformer = decoder["transformer_config"]
    assert transformer["model_dim"] == 2560
    assert transformer["hidden_dim"] == 9728
    assert len(transformer["layer_configs"]) == 36
    for layer in transformer["layer_configs"]:
        mixer = layer["mixer_config"]
        assert (mixer["num_heads"], mixer["num_groups"], mixer["head_dim"]) == (32, 8, 128)
        assert mixer["sliding_window_size"] is None
        assert mixer["has_qkvg_biases"] is False
        assert layer["rope_config"] == {
            "type": "UnscaledRoPEConfig",
            "base": 5000000.0,
            "max_sequence_length": 262144,
            "head_dim": 128,
        }

    operations = {operation.name: operation for operation in plan.operations}
    assert operations["decoder.embedding.embedding.weights"].shape == (151936, 2560)
    assert not any("output_embedding" in name for name in operations)
    for index in range(36):
        prefix = f"decoder.transformer.layers.{index}"
        qkv = operations[prefix + ".mixer.qkvg_projection.weights.weights"]
        assert qkv.kind == "fuse"
        assert qkv.shape == (6144, 2560)
        assert tuple(source.shape for source in qkv.sources) == ((4096, 2560), (1024, 2560), (1024, 2560))
        assert operations[prefix + ".mixer.out_projection.weights.weights"].shape == (2560, 4096)
        up_gate = operations[prefix + ".mlp.up_projection.weights.weights"]
        assert up_gate.kind == "fuse"
        assert up_gate.shape == (19456, 2560)
        assert tuple(source.shape for source in up_gate.sources) == ((9728, 2560), (9728, 2560))
        assert operations[prefix + ".mlp.down_projection.weights.weights"].shape == (2560, 9728)
    capability = cli.capability(ConversionRequest(REPO))
    assert capability.conversion == "implemented"
    assert capability.native_loading == "supported_layout_unverified_tuple"
    assert capability.execution == "unverified"


@pytest.mark.parametrize("dual_declaration", [False, True])
def test_modern_unscaled_rope_matches_legacy_decoder_and_tensor_plan(tmp_path, dual_declaration):
    source = tmp_path / "source"
    config, _ = qwen_fixture(source)
    config["rope_theta"] = 5000000
    (source / "config.json").write_text(json.dumps(config))
    legacy = build_qwen3_plan(source)
    if not dual_declaration:
        config.pop("rope_theta")
    config["rope_parameters"] = {"rope_type": "default", "rope_theta": 5000000}
    config["rope_scaling"] = None
    (source / "config.json").write_text(json.dumps(config))
    modern = build_qwen3_plan(source)
    assert modern.decoder_config == legacy.decoder_config
    assert modern.operations == legacy.operations
    assert modern.metadata == legacy.metadata
    assert modern.input_tensors == legacy.input_tensors


@pytest.mark.parametrize("theta", [True, False, None, "5000000", 0, -1, float("inf"), float("nan"), 10**400])
@pytest.mark.parametrize("modern", [False, True])
def test_rope_theta_must_be_positive_finite_numeric(tmp_path, theta, modern):
    source = tmp_path / "source"
    config, _ = qwen_fixture(source)
    if modern:
        config.pop("rope_theta")
        config["rope_parameters"] = {"rope_type": "default", "rope_theta": theta}
    else:
        config["rope_theta"] = theta
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises(ConversionError):
        build_qwen3_plan(source)


@pytest.mark.parametrize(
    "parameters",
    [
        {},
        [],
        False,
        "default",
        {"rope_type": "default"},
        {"rope_theta": 5000000},
        {"rope_type": True, "rope_theta": 5000000},
        {"rope_type": "linear", "rope_theta": 5000000},
        {"rope_type": "yarn", "rope_theta": 5000000, "factor": 2},
        {"rope_type": "default", "rope_theta": 5000000, "factor": 1},
        {"rope_type": "default", "rope_theta": 5000000, "partial_rotary_factor": 1.0},
        {"rope_type": "default", "rope_theta": 5000000, "mrope_section": [16, 24, 24]},
        {"full_attention": {"rope_type": "default", "rope_theta": 5000000}},
    ],
)
def test_modern_rope_rejects_incomplete_scaled_or_custom_declarations(tmp_path, parameters):
    source = tmp_path / "source"
    config, _ = qwen_fixture(source)
    config.pop("rope_theta")
    config["rope_parameters"] = parameters
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises((ConversionError, UnsupportedConversion)):
        build_qwen3_plan(source)


@pytest.mark.parametrize("legacy_theta", [10000, True, None, float("inf")])
def test_modern_rope_does_not_override_conflicting_or_invalid_legacy_theta(tmp_path, legacy_theta):
    source = tmp_path / "source"
    config, _ = qwen_fixture(source)
    config["rope_theta"] = legacy_theta
    config["rope_parameters"] = {"rope_type": "default", "rope_theta": 5000000}
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises(ConversionError):
        build_qwen3_plan(source)


@pytest.mark.parametrize("scaling", [False, {}, {"rope_type": "default"}, {"rope_type": "yarn", "factor": 2}])
def test_modern_rope_does_not_enable_rope_scaling(tmp_path, scaling):
    source = tmp_path / "source"
    config, _ = qwen_fixture(source)
    config.pop("rope_theta")
    config["rope_parameters"] = {"rope_type": "default", "rope_theta": 5000000}
    config["rope_scaling"] = scaling
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises(UnsupportedConversion, match="RoPE"):
        build_qwen3_plan(source)


@pytest.mark.parametrize("modern_rope", [False, True])
def test_local_trained_qwen3_full_checkpoint_keeps_its_metadata_offline(tmp_path, modern_rope):
    source = tmp_path / "trained-checkpoint"
    config, _ = qwen_fixture(source, tied=True, sharded=True)
    config.update(
        eos_token_id=[2, 3],
        bos_token_id=4,
        rope_theta=5000000,
        max_position_embeddings=262144,
        use_sliding_window=False,
        sliding_window=None,
        max_window_layers=1,
    )
    if modern_rope:
        config["rope_parameters"] = {"rope_type": "default", "rope_theta": config.pop("rope_theta")}
    (source / "config.json").write_text(json.dumps(config))
    tokenizer = Tokenizer(
        models.WordLevel({f"[SYNTHETIC_{index}]": index for index in range(16)}, unk_token="[SYNTHETIC_0]")
    )
    tokenizer.save(str(source / "tokenizer.json"))
    template = "{% for message in messages %}{{ message.role }}: {{ message.content }}{% endfor %}"
    (source / "tokenizer_config.json").write_text(json.dumps({"chat_template": template}))
    (source / "generation_config.json").write_text(json.dumps({"eos_token_id": 7, "temperature": 0.31, "top_p": 0.87}))
    selected = resolve_source(source, metadata_profile="hf-generic", offline=True, hub=object())
    manifest = tmp_path / "source-manifest.json"
    selected.write_manifest(manifest)
    replay = resolve_source(source, source_manifest=manifest, architecture="qwen3", offline=True, hub=object())
    assert replay.receipt() == selected.receipt()
    assert replay.conversion_profile == {"architecture": "qwen3", "metadata_profile": "hf-generic"}
    output = tmp_path / "converted"
    receipt = cli.convert(replay, output, repo=None, chunk_bytes=4096)

    assert receipt["architecture"] == "qwen3"
    assert receipt["metadata_profile"] == "hf-generic"
    assert receipt["catalogued"] is False
    assert receipt["native_qualification"] is None
    assert receipt["capability"]["source_validation"] == "complete"
    assert receipt["capability"]["native_loading"] == "unverified"
    assert receipt["capability"]["execution"] == "unverified"
    assert receipt["transform_peak_bytes"] <= 4096
    converted = json.loads((output / "config.json").read_text())
    assert converted["token_codec_config"]["prompt_template"] == template
    assert converted["token_codec_config"]["reasoning_config"] is None
    assert converted["token_codec_config"]["output_parser_regex"] is None
    assert converted["token_codec_config"]["default_system_prompt"] is None
    assert converted["generation_config"]["stop_token_ids"] == [2, 3, 7]
    assert converted["generation_config"]["temperature"] == 0.31
    assert converted["generation_config"]["top_p"] == 0.87
    decoder = converted["decoder_config"]
    assert decoder["embedding_config"]["type"] == "TiedEmbeddingConfig"
    layer = decoder["transformer_config"]["layer_configs"][0]
    assert layer["mixer_config"]["sliding_window_size"] is None
    assert layer["rope_config"]["base"] == 5000000.0
    assert layer["rope_config"]["max_sequence_length"] == 262144
    _, tensors = read_safetensors_header(output / "model.safetensors")
    assert tensors["decoder.embedding.embedding.weights"].shape == (16, 8)
    assert tensors["decoder.transformer.layers.0.mixer.qkvg_projection.weights.weights"].shape == (16, 8)
