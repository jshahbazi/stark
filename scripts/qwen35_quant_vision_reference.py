"""Private source-bound 0.8B affine/BF16 full-model eager reference."""

import argparse
import ast
import hashlib
import importlib.util
import io
import json
import resource
import shutil
import struct
import sys
import zlib
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

PHASE = "initial"
SOURCE_REPO = "mlx-community/Qwen3.5-0.8B-MLX-4bit"
SOURCE_REVISION = "5d894f8cc4ef3e6c88537bf3746ed262f549da6a"
PINNED_SOURCES = {
    "modeling_qwen3_5.py": "af05d0acfe79fff8aea1c75f93516a6a72bc80f6f901329a73e66a3866fe1e7c",
    "modeling_utils.py": "251aaae7733affbb7cdb5aa05a84157c46cac02518d868898cd6d6282cc14571",
}
OPERATIONAL_TEXT_FIELDS = (
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "linear_num_key_heads",
    "linear_num_value_heads",
    "linear_key_head_dim",
    "linear_value_head_dim",
    "linear_conv_kernel_dim",
    "rms_norm_eps",
    "full_attention_interval",
    "rope_parameters",
    "tie_word_embeddings",
    "vocab_size",
    "attention_bias",
    "hidden_act",
    "layer_types",
)
OPERATIONAL_VISION_FIELDS = (
    "hidden_size",
    "intermediate_size",
    "depth",
    "num_heads",
    "num_position_embeddings",
    "out_hidden_size",
    "patch_size",
    "spatial_merge_size",
    "temporal_patch_size",
    "in_channels",
    "hidden_act",
    "deepstack_visual_indexes",
)
PIXEL_MANIFEST_SHA256 = (
    "e5dceac11aa5fd42bea86ca50b4aa1a9d48d8bfd99f08cd43c5fe950225330ae"
)


def load_pinned_reference(root):
    """Run pinned numerical classes with explicitly bound child constructors."""
    from transformers import modeling_utils

    for name, expected in PINNED_SOURCES.items():
        assert sha(root / name) == expected
    parsed = ast.parse((root / "modeling_utils.py").read_text())
    node = next(
        item
        for item in parsed.body
        if isinstance(item, ast.FunctionDef) and item.name == "slice_logits_to_keep"
    )
    namespace = {"torch": torch}
    code = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    # The complete source bytes were checked against the immutable pin above.
    exec(compile(code, str(root / "modeling_utils.py"), "exec"), namespace)  # noqa: S102
    modeling_utils.slice_logits_to_keep = namespace["slice_logits_to_keep"]
    name = "transformers.models.qwen3_5.stark_pinned_modeling"
    spec = importlib.util.spec_from_file_location(name, root / "modeling_qwen3_5.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.ROPE_INIT_FUNCTIONS["axial"] = (
        module.Qwen3_5VisionRotaryEmbedding.compute_axial_rope_parameters
    )

    class PinnedAutoModel:
        @staticmethod
        def from_config(config):
            if config.model_type == "qwen3_5_vision":
                config.rope_parameters = {"rope_type": "axial", "rope_theta": 10000.0}
                config.num_attention_heads = config.num_heads
                return module.Qwen3_5VisionModel(config)
            if config.model_type == "qwen3_5_text":
                return module.Qwen3_5TextModel(config)
            raise ValueError("unexpected pinned child configuration")

    module.AutoModel = PinnedAutoModel
    return module


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def fixed_fixture_rgb(encoded, width, height):
    """Decode only this hash-pinned corpus's RGB8, filter-zero PNG recipe."""
    assert encoded[:8] == b"\x89PNG\r\n\x1a\n"
    offset, compressed, header = 8, [], None
    while offset < len(encoded):
        length = struct.unpack(">I", encoded[offset : offset + 4])[0]
        kind = encoded[offset + 4 : offset + 8]
        payload = encoded[offset + 8 : offset + 8 + length]
        assert len(payload) == length
        crc = struct.unpack(">I", encoded[offset + 8 + length : offset + 12 + length])[
            0
        ]
        assert zlib.crc32(kind + payload) == crc
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", payload)
        elif kind == b"IDAT":
            compressed.append(payload)
        offset += length + 12
    assert header == (width, height, 8, 2, 0, 0, 0) and offset == len(encoded)
    expected = height * (width * 3 + 1)
    decoder = zlib.decompressobj()
    decoded = decoder.decompress(b"".join(compressed), expected + 1)
    assert (
        decoder.eof
        and not decoder.unused_data
        and not decoder.unconsumed_tail
        and len(decoded) == expected
    )
    rows = np.frombuffer(decoded, dtype=np.uint8).reshape(height, width * 3 + 1)
    assert np.all(rows[:, 0] == 0)
    return rows[:, 1:].reshape(height, width, 3).copy()


def dequant(source, name):
    packed = source.get_tensor(name)
    scales = source.get_tensor(name.removesuffix(".weight") + ".scales")
    biases = source.get_tensor(name.removesuffix(".weight") + ".biases")
    assert (
        packed.dtype == torch.uint32 and scales.dtype == biases.dtype == torch.bfloat16
    )
    assert packed.ndim == scales.ndim == 2 and scales.shape == biases.shape
    rows, columns = packed.shape[0], packed.shape[1] * 8
    assert scales.shape == (rows, columns // 64)
    output = torch.empty((rows, columns), dtype=torch.bfloat16)
    shifts = torch.arange(8, dtype=torch.int64) * 4
    for start in range(0, rows, 256):
        end = min(start + 256, rows)
        codes = (
            ((packed[start:end].to(torch.int64)[..., None] >> shifts) & 15)
            .reshape(end - start, columns)
            .float()
        )
        values = codes * scales[start:end].float().repeat_interleave(64, -1) + biases[
            start:end
        ].float().repeat_interleave(64, -1)
        output[start:end].copy_(values)
    return output


def main():
    global PHASE
    if not __debug__:
        raise RuntimeError("reference requires enabled validation assertions")
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in (
        "source",
        "model",
        "structure",
        "transformers-source",
        "fixtures",
        "output",
    ):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument(
        "--capture-layers",
        action="store_true",
        help="Save private BF16 layer outputs for drift isolation",
    )
    args = parser.parse_args()
    import transformers

    assert torch.__version__ == "2.14.1" and transformers.__version__ == "5.15.1"
    SOURCE, ARTIFACT, OUTPUT = args.source, args.model, args.output
    assert (
        sha(args.structure / "config.json")
        == "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204"
    )
    assert (
        sha(args.fixtures / "manifest.json")
        == "c32ee8e39b89a1bae59526be310848a9bc7983b9908ab745d0e4ca93dfae7002"
    )
    assert (
        sha(SOURCE / "model.safetensors")
        == "f5a0d9dd3efa73510542a8023d610ff26be2b4b020d181cfc4bedaa1fcc5dd9e"
    )
    OUTPUT.mkdir(mode=0o700, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    PHASE = "load_pinned_bf16_structure"
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        ns = load_pinned_reference(args.transformers_source)
        model = ns.Qwen3_5ForConditionalGeneration.from_pretrained(
            args.structure,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            local_files_only=True,
            trust_remote_code=False,
        ).eval()
    raw_config = json.loads((SOURCE / "config.json").read_bytes())
    text = raw_config["text_config"]
    for key in OPERATIONAL_TEXT_FIELDS:
        assert getattr(model.config.text_config, key) == text[key]
    vision = raw_config["vision_config"]
    assert vision["model_type"] in {"qwen3_5", "qwen3_5_vision"}
    for key in OPERATIONAL_VISION_FIELDS:
        assert getattr(model.config.vision_config, key) == vision[key]
    assert set(vision) <= set(OPERATIONAL_VISION_FIELDS) | {
        "model_type",
        "initializer_range",
        "dtype",
        "torch_dtype",
    }
    parameters = dict(model.named_parameters())
    consumed, converted = set(), set()
    PHASE = "assign_source_quantized_weights"
    with safe_open(
        SOURCE / "model.safetensors", framework="pt", device="cpu"
    ) as source:
        for name in source.keys():  # noqa: SIM118 - safetensors exposes keys(), not dict iteration.
            if name.endswith((".scales", ".biases")):
                continue
            target = name.replace(
                "language_model.model.", "model.language_model.", 1
            ).replace("vision_tower.", "model.visual.", 1)
            assert target in parameters
            parameter = parameters[target]
            value = source.get_tensor(name)
            if value.dtype == torch.uint32:
                value = dequant(source, name)
                consumed.update(
                    (
                        name.removesuffix(".weight") + ".scales",
                        name.removesuffix(".weight") + ".biases",
                    )
                )
            elif target == "model.visual.patch_embed.proj.weight":
                value = value.permute(0, 4, 1, 2, 3).contiguous()
            elif target.endswith(".conv1d.weight") and value.shape != parameter.shape:
                value = value.transpose(1, 2).contiguous()
            # The MLX export bakes zero-centered RMS gains. Recover the HF
            # delta only when it is exactly reversible at the source dtype.
            if target == "model.language_model.norm.weight" or target.endswith(
                (
                    ".input_layernorm.weight",
                    ".post_attention_layernorm.weight",
                    ".self_attn.q_norm.weight",
                    ".self_attn.k_norm.weight",
                )
            ):
                original = value.float()
                value = (original - 1).to(value.dtype)
                assert torch.equal(value.float() + 1, original)
            assert value.shape == parameter.shape and value.dtype in (
                torch.bfloat16,
                torch.float32,
            )
            parameter.data = value
            consumed.add(name)
            converted.add(target)
        assert consumed == set(source.keys()) and converted == set(parameters)
    del parameters
    assert (
        model.lm_head.weight.data_ptr()
        == model.model.language_model.embed_tokens.weight.data_ptr()
    )
    assert all(
        buffer.dtype == torch.float32
        for name, buffer in model.named_buffers()
        if name.endswith("inv_freq")
    )
    base = args.fixtures
    old = json.loads((base / "manifest.json").read_bytes())
    artifact_config = json.loads((ARTIFACT / "config.json").read_bytes())
    contract = artifact_config["artifact_contract"]
    assert (
        contract["source"]["repo"] == SOURCE_REPO
        and contract["source"]["revision"] == SOURCE_REVISION
    )
    assert sha(SOURCE / "config.json") == contract["source"]["hashes"]["config.json"]
    pixel_root = args.transformers_source / "eager_fp32_buffers"
    assert sha(pixel_root / "reference-manifest.json") == PIXEL_MANIFEST_SHA256
    pixel_manifest = json.loads((pixel_root / "reference-manifest.json").read_bytes())
    identity = {
        "artifact_schema_version": contract["schema_version"],
        "config_sha256": sha(ARTIFACT / "config.json"),
        "artifact_sha256": sha(ARTIFACT / "model.safetensors"),
        "tokenizer_sha256": sha(ARTIFACT / "tokenizer.json"),
        "processor_sha256": sha(ARTIFACT / "preprocessor_config.json"),
        "template_sha256": sha(ARTIFACT / "chat_template.jinja"),
    }
    manifest = {
        "schema_version": 2,
        "artifact_identity": identity,
        "source": {key: contract["source"][key] for key in ("repo", "revision")},
        "source_weight_manifest": contract["source"],
        "family": contract["family"],
        "sampling": {"mode": "greedy"},
        "mtp": False,
        "threads": 4,
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "transformers": transformers.__version__,
        "structure_trained_weights_discarded": True,
        "operational_text_config_fields_checked": list(OPERATIONAL_TEXT_FIELDS),
        "operational_vision_config_fields_checked": list(OPERATIONAL_VISION_FIELDS),
        "source_config_sha256": sha(SOURCE / "config.json"),
        "pixel_reference_manifest_sha256": PIXEL_MANIFEST_SHA256,
        "vision_fixed_operations": {
            "rope_type": "axial",
            "rope_theta": 10000.0,
            "layer_norm_epsilon": 1e-6,
        },
        "reference": "actual pinned Transformers full model with independently BF16-dequantized source affine weights",
        "reference_weight_policy": "group64 unsignedQ4 F32(code*BF16scale+BF16bias), round effective weight to BF16",
        "source_weight_sha256": sha(SOURCE / "model.safetensors"),
        "pinned_transformers_revision": "2e12b325d5c4532f3ec5b0ec54e1ff0b7e0db397",
        "input_fixture_manifest_sha256": sha(base / "manifest.json"),
        "reference_sources_sha256": {
            name: sha(args.transformers_source / name) for name in PINNED_SOURCES
        },
        "structure_config_sha256": sha(args.structure / "config.json"),
        "generator_sha256": sha(__file__),
        "vocab_size": model.config.text_config.vocab_size,
        "context_capacity": 512,
        "captures_decoder_layers": args.capture_layers,
        "cases": [],
    }
    for old_case in old["cases"]:
        PHASE = "execute_" + old_case["name"]
        folder = OUTPUT / old_case["name"]
        folder.mkdir(mode=0o700)
        case = json.loads(json.dumps(old_case))
        item = case["image"]
        shutil.copyfile(base / case["name"] / item["file"], folder / item["file"])
        for key, item in case["files"].items():
            if key != "logits":
                shutil.copyfile(
                    base / case["name"] / item["file"], folder / item["file"]
                )
        tokens = np.frombuffer(
            (folder / case["files"]["tokens"]["file"]).read_bytes(), dtype="<u4"
        ).astype(np.int64)
        ids = torch.tensor(tokens[None, :], dtype=torch.long)
        grid = torch.tensor([case["grid"]], dtype=torch.long)
        tower_name = (
            "tower_256x256"
            if case["name"] == "square_256"
            else "tower_processed_rgb_47x81"
        )
        tower_folder = pixel_root / tower_name
        tower_meta = next(
            item for item in pixel_manifest["tower"] if item["name"] == tower_name
        )
        assert tower_meta["grid"] == case["grid"]
        patch_record = tower_meta["patches"]
        patch_bytes = (tower_folder / patch_record["file"]).read_bytes()
        assert hashlib.sha256(patch_bytes).hexdigest() == patch_record["sha256"]
        assert patch_record["dtype"] == "bf16" and len(patch_bytes) == 2 * int(
            np.prod(patch_record["shape"])
        )
        image_bytes = (folder / case["image"]["file"]).read_bytes()
        assert hashlib.sha256(image_bytes).hexdigest() == case["image"]["sha256"]
        rgb_record = tower_meta["source_rgb"]
        rgb = fixed_fixture_rgb(image_bytes, rgb_record["width"], rgb_record["height"])
        assert rgb.shape == (rgb_record["height"], rgb_record["width"], 3)
        assert hashlib.sha256(rgb.tobytes()).hexdigest() == rgb_record["sha256"]
        (folder / "pixel_values.bf16").write_bytes(patch_bytes)
        case["files"]["pixel_values"] = {**patch_record, "file": "pixel_values.bf16"}
        pixels = torch.frombuffer(
            bytearray(patch_bytes),
            dtype=torch.bfloat16,
        ).reshape(patch_record["shape"])
        mask = torch.ones_like(ids)
        types = (ids == 248056).to(torch.int32)
        handles, layer_files = [], []
        if args.capture_layers:
            for index, layer in enumerate(model.model.language_model.layers):

                def capture(
                    _module,
                    _arguments,
                    output,
                    *,
                    layer_index=index,
                    rows=ids.shape[1],
                    case_folder=folder,
                    files=layer_files,
                ):
                    hidden = output[0] if isinstance(output, tuple) else output
                    assert hidden.dtype == torch.bfloat16 and hidden.shape == (
                        1,
                        rows,
                        text["hidden_size"],
                    )
                    data = (
                        hidden.detach()
                        .cpu()
                        .contiguous()
                        .view(torch.uint16)
                        .numpy()
                        .astype("<u2", copy=False)
                        .tobytes()
                    )
                    filename = f"layer_{layer_index}.bf16"
                    (case_folder / filename).write_bytes(data)
                    files.append(
                        {
                            "layer": layer_index,
                            "file": filename,
                            "sha256": hashlib.sha256(data).hexdigest(),
                        }
                    )

                handles.append(layer.register_forward_hook(capture))
        with (
            torch.no_grad(),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            positions, delta = model.model.get_rope_index(
                ids, image_grid_thw=grid, attention_mask=mask, mm_token_type_ids=types
            )
            expected = np.frombuffer(
                (folder / case["files"]["positions"]["file"]).read_bytes(), dtype="<u4"
            ).reshape(-1, 3)
            assert (
                np.array_equal(positions[:, 0, :].T.numpy(), expected)
                and int(delta.item()) == case["position_delta"]
            )
            model.model.rope_deltas = None
            result = model(
                input_ids=ids,
                attention_mask=mask,
                pixel_values=pixels,
                image_grid_thw=grid,
                mm_token_type_ids=types,
                use_cache=False,
                logits_to_keep=1,
                return_dict=True,
            )
        for handle in handles:
            handle.remove()
        if args.capture_layers:
            assert len(layer_files) == text["num_hidden_layers"]
            case["decoder_layer_outputs"] = layer_files
        logits = result.logits[0, -1].detach().cpu()
        assert logits.dtype == torch.bfloat16 and torch.isfinite(logits).all()
        data = logits.float().numpy().astype("<f4", copy=False).tobytes()
        (folder / "logits.f32").write_bytes(data)
        case["files"]["logits"] = {
            "file": "logits.f32",
            "shape": [logits.numel()],
            "dtype": "f32",
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        manifest["cases"].append(case)
        del result
    encoded = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    (OUTPUT / "manifest.json").write_bytes(encoded)
    print(
        json.dumps(
            {
                "reference_ready": True,
                "cases": len(manifest["cases"]),
                "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
                "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:  # noqa: BLE001 - preserve content-free failures, including interruption.
        print(
            json.dumps(
                {
                    "reference_ready": False,
                    "phase": PHASE,
                    "failure_category": type(error).__name__,
                }
            ),
            flush=True,
        )
        raise SystemExit(1)
