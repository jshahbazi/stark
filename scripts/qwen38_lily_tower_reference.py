"""Bounded source-bound retained Lily BF16 tower component reference."""

import argparse
import hashlib
import importlib.util
import json
import resource
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import transformers
import transformers.modeling_outputs
import transformers.utils.generic
from safetensors import safe_open
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.utils.generic import get_max_seqlen

SOURCE_REVISION = "2e12b325d5c4532f3ec5b0ec54e1ff0b7e0db397"
SOURCE_SHA = "e57b3617e71e81ecb7e5ae3b420c45b5aee80d76a07aa8c88b797c2642f1f5df"
UPSTREAM_SHA = "79d37b1a81308db9da677d99506613383b6f3bf5190bf025c34f9ca25822ee92"
HELPER_HASHES = {
    "modeling_qwen3_5.py": "af05d0acfe79fff8aea1c75f93516a6a72bc80f6f901329a73e66a3866fe1e7c",
    "generate_reference.py": "0b7f9e3e3acfaa3841f62861bf0adb381855e9371444e2fe12572509490b58a3",
    "vision_utils.py": "bcecd5a92b3266b9926272a549d2b1a0f1fe7646c698c0fa19bd96f976085356",
    "image_processing_qwen2_vl.py": "ebdb484798a6aefebc5dbe5431d3e4af8f460404d95d4cb67165fdff624ed333",
}
ARTIFACT = "18c7bcc1e97d1fefdd361472c2daa59d81d89fcd6f3a61983d995a032b44bfb4"
PINNED_IDENTITY = {
    "config_sha256": "2a91155be52ebbb0f6dff20e61dcd8ae7484154a56b4afe067246c59da46f899",
    "index_sha256": "f7e35f06cb23dba09dea6197545111c092459f001715beead877c5f84959ccd4",
    "processor_sha256": "022f906548d5a89d3a18c61c0a87de760db8c81d618437501baa3dcdc14d4ecd",
    "template_sha256": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
    "tokenizer_sha256": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
}
PIXEL_MANIFEST = "4c49cf1bedb0188b6f7ef7f195cb03f31c5b910de99b5f6b1e91c962e598f468"
PHASE = "arguments"
SCAFFOLD_HASHES = {
    "transformers.modeling_outputs": "fef3993ec5b06b4aa488c50472fa0fe4013fb84e34a10595533a251afd6914c0",
    "transformers.utils.generic": "b60d9ecbbaeadb084134dda4a9b92419d35d22af6bc882e062fd530f4e47a4e1",
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class ConstructionOnlyBase(nn.Module):
    """No initialization: all333 parameters must be assigned from the source."""

    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.config = config

    def post_init(self):
        pass


def save(folder, name, value):
    value = value.detach().cpu().contiguous()
    assert value.dtype == torch.bfloat16 and value.numel() <= 256 * 4608
    data = value.view(torch.uint16).numpy().astype("<u2", copy=False).tobytes()
    filename = name + ".bf16"
    (folder / filename).write_bytes(data)
    return {
        "file": filename,
        "shape": list(value.shape),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def save_f32(folder, name, value):
    value = value.detach().cpu().contiguous()
    assert value.dtype == torch.float32 and value.numel() <= 256 * 72
    data = value.numpy().astype("<f4", copy=False).tobytes()
    filename = name + ".f32"
    (folder / filename).write_bytes(data)
    return {"file": filename, "shape": list(value.shape),
            "sha256": hashlib.sha256(data).hexdigest()}


def operator_capture(tower, namespace, folder, record):
    """Observe unchanged source operators on their own inputs, without recoding them."""
    handles = []
    block = tower.blocks[0]
    modules = {
        "patch": tower.patch_embed,
        "norm1": block.norm1,
        "qkv": block.attn.qkv,
        "projection": block.attn.proj,
        "norm2": block.norm2,
        "fc1": block.mlp.linear_fc1,
        "fc2": block.mlp.linear_fc2,
        "merger_norm": tower.merger.norm,
        "merger_fc1": tower.merger.linear_fc1,
        "merger_fc2": tower.merger.linear_fc2,
    }

    def capture(name):
        def hook(module, inputs, output):
            value = inputs[0]
            if name == "patch":
                value = value.reshape(-1, 1536)
            record.append({"operator": name,
                           "input": save(folder, name + "_input", value),
                           "output": save(folder, name + "_output", output)})
        return hook

    for name, module in modules.items():
        handles.append(module.register_forward_hook(capture(name)))
    original = namespace["eager_attention_forward"]

    def eager(*args, **kwargs):
        result = original(*args, **kwargs)
        _, query, key, value = args[:4]
        qkv = torch.cat([tensor[0].transpose(0, 1).reshape(-1, 1152)
                         for tensor in (query, key, value)], dim=-1)
        record.append({"operator": "attention",
                       "input": save(folder, "attention_input", qkv),
                       "output": save(folder, "attention_output", result[0].reshape(-1, 1152)),
                       "stages": {}})
        scores = torch.matmul(query, key.transpose(2, 3))
        record[-1]["stages"]["qk"] = save(folder, "attention_qk", scores[0])
        scores = scores * kwargs["scaling"]
        record[-1]["stages"]["scaled"] = save(folder, "attention_scaled", scores[0])
        probabilities = scores.softmax(-1, dtype=torch.float32).to(query.dtype)
        record[-1]["stages"]["probabilities"] = save(folder, "attention_probabilities", probabilities[0])
        return result

    namespace["eager_attention_forward"] = eager

    def close():
        namespace["eager_attention_forward"] = original
        for handle in handles:
            handle.remove()
    return close


def full_state_capture(tower, folder, records):
    """Observe full-depth norm and residual states under the fixture size cap."""
    handles = []
    for index, block in enumerate(tower.blocks):
        record = {"block_index": index}
        records.append(record)

        def capture(name, index, record):
            def hook(module, inputs, output):
                prefix = "block_" + str(index) + "_" + name
                record[name + "_input"] = save(folder, prefix + "_input", inputs[0])
                record[name + "_output"] = save(folder, prefix + "_output", output)
            return hook

        handles.append(block.norm1.register_forward_hook(capture("norm1", index, record)))
        handles.append(block.norm2.register_forward_hook(capture("norm2", index, record)))
        handles.append(block.register_forward_hook(capture("residual", index, record)))

    def close():
        for handle in handles:
            handle.remove()
    return close


def main(args):
    global PHASE
    assert sys.platform == "darwin" and torch.__version__ == "2.14.1"
    assert transformers.__version__ == "5.15.1"
    for name, expected in SCAFFOLD_HASHES.items():
        assert digest(Path(sys.modules[name].__file__)) == expected
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if args.gelu_domain:
        assert torch.version.git_version == "5c4886908584029761b579af026dcfb627c84070"
        bits = torch.arange(65536, dtype=torch.int32)
        bits = bits[(bits & 0x7F80) != 0x7F80].to(torch.uint16)
        values = bits.view(torch.bfloat16).reshape(255, 256)
        args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
        record = {"kind": "RetainedLilyGeluDomainReference",
                  "torch": torch.__version__, "torch_git": torch.version.git_version,
                  "finite_inputs": values.numel(),
                  "generator_sha256": digest(Path(__file__)),
                  "input": save(args.output, "input", values)}
        for name, approximate in (("erf", "none"), ("tanh", "tanh")):
            output = torch.nn.functional.gelu(values, approximate=approximate)
            assert torch.isfinite(output).all()
            record[name] = save(args.output, name, output)
        (args.output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
        total = sum(path.stat().st_size for path in args.output.iterdir())
        assert total < 1024**2
        print(json.dumps({"gelu_domain_complete": True,
                          "finite_inputs": values.numel(), "reference_bytes": total,
                          "manifest_sha256": digest(args.output / "manifest.json")}))
        return
    for name, expected in HELPER_HASHES.items():
        assert digest(args.helpers / name) == expected
    assert digest(args.sources / "modeling_qwen4_exp.py") == SOURCE_SHA
    assert digest(args.sources / "hf_vision_reference.py") == UPSTREAM_SHA
    assert digest(args.pixels / "reference-manifest.json") == PIXEL_MANIFEST
    contract = json.loads((args.model / "stark-image-artifact.json").read_bytes())
    assert contract["artifact_sha256"] == ARTIFACT
    assert all(contract[key] == value for key, value in PINNED_IDENTITY.items())
    assert contract["model_source"] == "fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl"
    assert contract["model_revision"] == "0f94842e3682b49acace4daeb4cf2e65a6f94f83"
    for filename, key in (
        ("config.json", "config_sha256"),
        ("model.safetensors.index.json", "index_sha256"),
        ("tokenizer.json", "tokenizer_sha256"),
        ("preprocessor_config.json", "processor_sha256"),
        ("chat_template.jinja", "template_sha256"),
    ):
        assert digest(args.model / filename) == contract[key]
    config_data = json.loads((args.model / "config.json").read_bytes())
    vision = config_data["vision_config"]
    assert [
        vision[key]
        for key in (
            "depth",
            "hidden_size",
            "num_heads",
            "intermediate_size",
            "out_hidden_size",
        )
    ] == [27, 1152, 16, 4304, 2560]
    assert [
        vision[key]
        for key in (
            "patch_size",
            "temporal_patch_size",
            "spatial_merge_size",
            "in_channels",
            "num_position_embeddings",
        )
    ] == [16, 2, 2, 3, 2304]
    assert (
        vision["hidden_act"] == "gelu_pytorch_tanh"
        and vision["deepstack_visual_indexes"] == []
    )
    config = SimpleNamespace(**vision)
    config.rope_parameters = {"rope_type": "axial", "rope_theta": 10000.0}
    config.num_attention_heads = config.num_heads
    config._attn_implementation = "eager"
    PHASE = "reference_namespace"
    spec = importlib.util.spec_from_file_location(
        "pinned_lily_tower_helpers", args.helpers / "generate_reference.py"
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    ns = helper.reference_namespace()
    ns.update(
        Qwen4ExpPreTrainedModel=ConstructionOnlyBase,
        BaseModelOutputWithPooling=BaseModelOutputWithPooling,
        get_max_seqlen=get_max_seqlen,
    )
    helper.selected_ast(
        args.sources / "modeling_qwen4_exp.py",
        {
            "rotate_half",
            "repeat_kv",
            "eager_attention_forward",
            "apply_rotary_pos_emb_vision",
            "Qwen4ExpVisionRotaryEmbedding",
            "Qwen4ExpVisionMLP",
            "Qwen4ExpVisionPatchEmbed",
            "Qwen4ExpVisionPatchMerger",
            "Qwen4ExpVisionAttention",
            "Qwen4ExpVisionBlock",
            "Qwen4ExpVisionModel",
        },
        ns,
    )
    helper.selected_ast(
        args.helpers / "vision_utils.py", {"get_vision_attention_seqlens"}, ns
    )
    if args.rotary_stages:
        PHASE = "rotary_stages"
        assert torch.version.git_version == "5c4886908584029761b579af026dcfb627c84070"
        rotary = ns["Qwen4ExpVisionRotaryEmbedding"](config)
        assert rotary.inv_freq.dtype == torch.float32
        args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
        record = {"kind": "RetainedLilyRotarySourceReference",
                  "source_revision": SOURCE_REVISION, "source_sha256": SOURCE_SHA,
                  "torch_git": torch.version.git_version,
                  "artifact_sha256": ARTIFACT,
                  "generator_sha256": digest(Path(__file__)),
                  "inv_freq": save_f32(args.output, "inv_freq", rotary.inv_freq),
                  "cases": []}
        for name in ("square_256", "stress_32"):
            case = json.loads((args.pixels / name / "metadata.json").read_bytes())
            grid = torch.tensor([case["grid"]], dtype=torch.long)
            positions = ns["get_vision_position_ids"](grid, 2, kwargs={})
            angles = positions[..., None].float() * rotary.inv_freq.float()
            cos, sin = rotary(torch.zeros(1, dtype=torch.bfloat16), positions)
            folder = args.output / name
            folder.mkdir(mode=0o700)
            record["cases"].append({"name": name, "grid": case["grid"],
                "angles": save_f32(folder, "angles", angles.reshape(-1, 36)),
                "cos": save_f32(folder, "cos", cos),
                "sin": save_f32(folder, "sin", sin)})
        (args.output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
        total = sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file())
        assert total < 1024**2
        print(json.dumps({"rotary_reference_complete": True,
                          "reference_bytes": total,
                          "manifest_sha256": digest(args.output / "manifest.json")}))
        return
    PHASE = "construct_tower"
    with torch.device("meta"):
        tower = ns["Qwen4ExpVisionModel"](config)
    index = json.loads((args.model / "model.safetensors.index.json").read_bytes())[
        "weight_map"
    ]
    names = sorted(name for name in index if name.startswith("model.visual."))
    assert len(names) == 333 and len({index[name] for name in names}) == 1
    shard = index[names[0]]
    PHASE = "hash_original_visual_shard"
    assert digest(args.model / shard) == contract["weight_files"][shard]["sha256"]
    tensors = {}
    tensor_hashes = {}
    PHASE = "load_only_visual_parameters"
    with safe_open(args.model / shard, framework="pt", device="cpu") as weights:
        for name in names:
            value = weights.get_tensor(name)
            assert value.dtype == torch.bfloat16
            suffix = name.removeprefix("model.visual.")
            tensors[suffix] = value
            tensor_hashes[suffix] = hashlib.sha256(
                value.contiguous().view(torch.uint16).numpy().tobytes()
            ).hexdigest()
        assert sum(value.numel() * 2 for value in tensors.values()) == 897862112
        tower.load_state_dict(tensors, strict=True, assign=True)
        del tensors
        rotary = tower.rotary_pos_emb
        rotary.inv_freq = rotary.compute_axial_rope_parameters(config, device="cpu")[0]
        rotary.original_inv_freq = rotary.inv_freq.clone()
        assert rotary.inv_freq.dtype == torch.float32
        tower.eval()
        args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
        manifest = {
            "schema_version": 1,
            "kind": "RetainedLilyTowerSourceReference",
            "artifact_contract": contract,
            "artifact_manifest_sha256": digest(
                args.model / "stark-image-artifact.json"
            ),
            "vision_config": vision,
            "source_revision": SOURCE_REVISION,
            "source_sha256": {
                "modeling_qwen4_exp.py": SOURCE_SHA,
                "upstream_hf_vision_reference.py": UPSTREAM_SHA,
                **HELPER_HASHES,
            },
            "generator_sha256": digest(Path(__file__)),
            "torch": torch.__version__,
            "transformers_scaffold": transformers.__version__,
            "scaffold_file_sha256": SCAFFOLD_HASHES,
            "torch_git": torch.version.git_version,
            "threads": 4,
            "device": "cpu",
            "dtype": "BF16",
            "rotary_buffer_dtype": "F32",
            "attention": "pinned_eager",
            "construction_scaffold": "nn.Module base and no post_init; all333 source parameters assigned strictly; framework decorators stripped; arithmetic bodies unchanged",
            "pixel_manifest_sha256": PIXEL_MANIFEST,
            "visual_shard": shard,
            "source_visual_tensor_sha256": tensor_hashes,
            "gate": {"relative_rms_max": 0.02, "cosine_min": 0.999},
            "cases": [],
        }
        original_blocks = tower.blocks
        PHASE = "source_forward"
        for name in ("square_256", "stress_32"):
            folder = args.output / name
            folder.mkdir(mode=0o700)
            case = json.loads((args.pixels / name / "metadata.json").read_bytes())
            pixel_record = case["patches"]
            data = (args.pixels / name / pixel_record["file"]).read_bytes()
            assert hashlib.sha256(data).hexdigest() == pixel_record["sha256"]
            pixels = torch.frombuffer(bytearray(data), dtype=torch.bfloat16).reshape(
                pixel_record["shape"]
            )
            assert pixels.shape[0] <= 256 and pixels.shape[1] == 1536
            rgb = (args.pixels / name / "rgb.u8").read_bytes()
            height, width = case["grid"][1] * 16, case["grid"][2] * 16
            assert rgb == helper.pixel_fixture(width, height).tobytes()
            image = (
                torch.from_numpy(
                    np.frombuffer(rgb, dtype=np.uint8).copy().reshape(height, width, 3)
                )
                .permute(2, 0, 1)
                .unsqueeze(0)
            )
            recomputed = helper.patchify(ns, (image.float() - 127.5) / 127.5).bfloat16()
            assert torch.equal(recomputed, pixels)
            (folder / "rgb.u8").write_bytes(rgb)
            record = {
                "name": name,
                "grid": case["grid"],
                "rgb_sha256": hashlib.sha256(rgb).hexdigest(),
                "pixels": save(folder, "pixels", pixels),
                "stages": [],
            }
            for depth in (0, 1, 27):
                tower.blocks = original_blocks[:depth]
                close = None
                if args.operators and depth == 1:
                    record["operators"] = []
                    close = operator_capture(tower, ns, folder, record["operators"])
                close_full = None
                if args.full_states and depth == 27:
                    record["full_states"] = []
                    close_full = full_state_capture(tower, folder, record["full_states"])
                with torch.inference_mode():
                    output = tower(
                        pixels, grid_thw=torch.tensor([case["grid"]], dtype=torch.long)
                    )
                if close:
                    close()
                    operators = {item["operator"]: item for item in record["operators"]}
                    for name, before, after in (
                        ("position", ("patch", "output"), ("norm1", "input")),
                        ("rotary", ("qkv", "output"), ("attention", "input")),
                        ("gelu_tanh", ("fc1", "output"), ("fc2", "input")),
                        ("gelu_erf", ("merger_fc1", "output"), ("merger_fc2", "input")),
                    ):
                        record["operators"].append({
                            "operator": name,
                            "input": operators[before[0]][before[1]],
                            "output": operators[after[0]][after[1]],
                        })
                if close_full:
                    close_full()
                record["stages"].append(
                    {
                        "blocks": depth,
                        "merged": save(
                            folder, "merged_" + str(depth), output.pooler_output
                        ),
                    }
                )
                assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= 3 * 1024**3
            manifest["cases"].append(record)
        tower.blocks = original_blocks
    manifest["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    total = sum(
        path.stat().st_size for path in args.output.rglob("*") if path.is_file()
    )
    assert total <= 256 * 1024**2
    print(
        json.dumps(
            {
                "reference_complete": True,
                "manifest_sha256": digest(args.output / "manifest.json"),
                "reference_bytes": total,
                "peak_rss_bytes": manifest["peak_rss_bytes"],
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("model", "sources", "helpers", "pixels", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--operators", action="store_true",
                        help="save same-input source operator controls at one block")
    parser.add_argument("--full-states", action="store_true",
                        help="save bounded full-depth source norm and residual states")
    parser.add_argument("--gelu-domain", action="store_true",
                        help="produce a model-free all-finite-BF16 GELU reference only")
    parser.add_argument("--rotary-stages", action="store_true",
                        help="produce bounded source F32 frequency/angle/cos/sin stages only")
    try:
        main(parser.parse_args())
    except BaseException as error:  # noqa: BLE001 - content-free private diagnostic
        print(json.dumps({"reference_failed": type(error).__name__, "phase": PHASE}))
        raise SystemExit(1)
