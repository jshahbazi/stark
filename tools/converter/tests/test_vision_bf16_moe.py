"""Official HF expert layout with bounded synthetic payloads; no model parity claim."""

import json

import numpy as np
import pytest
from test_cli import source_tooling  # noqa: F401
from test_streaming import save_tensors, tensor_bytes
from test_vision_family import FINE_TUNE, pinned_vision_source  # noqa: F401
from test_vision_family import family_source as _family_source
from test_vision_family_geometry import geometry

from stark_convert import cli, model_support, vision_family, vision_runtime
from stark_convert.adapters.qwen35 import build_qwen35_plan
from stark_convert.streaming import ConversionError, UnsupportedConversion, read_safetensors_header

family_source = _family_source


def moe_geometry():
    config = geometry(model=2048, hidden=512, layers=40, q=16, kv=2, linear=32, tied=False, tower=(27, 1152, 4304, 16))
    config.update(model_type="qwen3_5_moe", architectures=["Qwen3_5MoeForConditionalGeneration"])
    text = config["text_config"]
    text.update(
        model_type="qwen3_5_moe_text",
        num_experts=256,
        num_experts_per_tok=8,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
    )
    text.pop("intermediate_size")
    config["vision_config"]["model_type"] = "qwen3_5_moe"
    return config


@pytest.fixture
def bf16_moe_source(family_source):
    config = json.loads((family_source / "config.json").read_text())
    config.update(model_type="qwen3_5_moe", architectures=["Qwen3_5MoeForConditionalGeneration"])
    config["vision_config"]["model_type"] = "qwen3_5_moe"
    text = config["text_config"]
    hidden = text.pop("intermediate_size")
    dim = text["hidden_size"]
    text.update(
        model_type="qwen3_5_moe_text",
        architectures=["Qwen3_5MoeForCausalLM"],
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=hidden,
        shared_expert_intermediate_size=hidden,
    )
    _, infos = read_safetensors_header(family_source / "model.safetensors")
    tensors = {
        name: (
            info.dtype,
            np.frombuffer(tensor_bytes(info), dtype={"BF16": "<u2", "F32": "<f4"}[info.dtype])
            .copy()
            .reshape(info.shape),
        )
        for name, info in infos.items()
    }
    for layer in range(text["num_hidden_layers"]):
        prefix = f"model.language_model.layers.{layer}.mlp"
        for name in ("up_proj", "gate_proj", "down_proj"):
            tensors[f"{prefix}.shared_expert.{name}.weight"] = tensors.pop(f"{prefix}.{name}.weight")
        for name, shape in (
            ("experts.gate_up_proj", (4, 2 * hidden, dim)),
            ("experts.down_proj", (4, dim, hidden)),
            ("gate.weight", (4, dim)),
            ("shared_expert_gate.weight", (1, dim)),
        ):
            tensors[f"{prefix}.{name}"] = ("BF16", np.arange(np.prod(shape), dtype="<u2").reshape(shape))
    save_tensors(family_source / "model.safetensors", tensors)
    (family_source / "config.json").write_text(json.dumps(config))
    return family_source, config, tensors


def test_official_bf16_moe_geometry_is_image_only_and_never_reference_qualified():
    config = moe_geometry()
    text, vision = vision_family.validate_family_geometry(config)
    assert text["num_experts"] == 256 and vision["out_hidden_size"] == 2048
    assert model_support.preflight_model(config, mode="vision-language")["status"] == "conditional"
    for mode in (None, "text", "decision"):
        report = model_support.preflight_model(config, mode=mode)
        assert report["status"] == "unsupported"
        assert report["reasons"] == ["unsupported_bf16_moe_text"]
    assert vision_runtime.supported_image_family({"profile": vision_family.BF16_MOE_FAMILY_PROFILE, "version": 5})
    assert not vision_runtime.supported_image_family({"profile": vision_family.BF16_MOE_FAMILY_PROFILE, "version": 4})


def test_bf16_moe_conversion_reorders_each_gate_up_bank_and_preserves_source_identity(bf16_moe_source, tmp_path):
    source, source_config, tensors = bf16_moe_source
    output = tmp_path / "converted-bf16-moe"
    receipt = cli.convert(source, output, repo=FINE_TUNE, artifact_mode="vision-language", chunk_bytes=4096)
    contract = receipt["artifact_contract"]
    assert contract["schema_version"] == 3
    assert contract["family"] == {"profile": vision_family.BF16_MOE_FAMILY_PROFILE, "version": 5}
    assert contract["qualification"] == "unverified" and receipt["native_qualification"] is None
    assert (
        contract["source"]["weight_files"]["model.safetensors"]["size"]
        == (source / "model.safetensors").stat().st_size
    )
    published = json.loads((output / "config.json").read_text())
    decoder = published["decoder_config"]["transformer_config"]
    assert decoder["output_norm_config"]["scale_offset"] == 1.0
    _, infos = read_safetensors_header(output / "model.safetensors")
    hidden = source_config["text_config"]["moe_intermediate_size"]
    for layer, graph in enumerate(decoder["layer_configs"]):
        src = f"model.language_model.layers.{layer}.mlp"
        dst = f"decoder.transformer.layers.{layer}.mlp"
        fused = tensors[src + ".experts.gate_up_proj"][1]
        expected = np.concatenate([np.concatenate([bank[hidden:], bank[:hidden]]) for bank in fused])
        assert tensor_bytes(infos[dst + ".experts.up_projection.weights.weights"]) == expected.tobytes()
        assert (
            tensor_bytes(infos[dst + ".experts.down_projection.weights.weights"])
            == tensors[src + ".experts.down_proj"][1].tobytes()
        )
        assert infos[dst + ".shared_gate.weights.weights"].shape == (1, source_config["text_config"]["hidden_size"])
        assert graph["mixer_config"]["reference_bf16"] is True
        assert graph["pre_mlp_norm_config"]["upcast_mode"] == "full_layer"
        assert graph["mlp_config"]["num_active_routed_experts"] == 2
    assert all(info.dtype == "BF16" for name, info in infos.items() if ".mlp." in name)
    with pytest.raises(UnsupportedConversion, match="image artifact reference policy"):
        build_qwen35_plan(source)


@pytest.mark.parametrize("mutation", ["shape", "mixed_dtype", "unconsumed_alias", "missing", "unnormalized"])
def test_bf16_moe_conversion_rejects_unsupported_banks_without_publishing(bf16_moe_source, tmp_path, mutation):
    source, config, tensors = bf16_moe_source
    key = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    dtype, bank = tensors[key]
    if mutation == "shape":
        tensors[key] = (dtype, bank[:, :-1].copy())
    elif mutation == "mixed_dtype":
        tensors[key] = ("F16", bank.view("<f2"))
    elif mutation == "unconsumed_alias":
        tensors[key + ".weight"] = (dtype, bank)
    elif mutation == "missing":
        del tensors[key]
    else:
        config["text_config"]["norm_topk_prob"] = False
    save_tensors(source / "model.safetensors", tensors)
    (source / "config.json").write_text(json.dumps(config))
    output = tmp_path / "rejected"
    with pytest.raises((ConversionError, UnsupportedConversion)):
        cli.convert(source, output, repo=FINE_TUNE, artifact_mode="vision-language")
    assert not output.exists()
