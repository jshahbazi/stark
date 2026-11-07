"""Create private canonical HF Gemma 4 image references on CPU.

Uses the actual pinned Transformers image processor, vision model, multimodal
embedder and text class. The three RGB fixture PNGs are copied byte-for-byte
from an existing private fixture manifest. Only aggregates and hashes escape
the ignored output directory; no model payloads or exception dumps are printed.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from gemma4_hf_reference import digest, metrics

MODEL_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
PHASE = "startup"


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    path.chmod(0o600)


def run(args) -> dict:
    global PHASE
    import numpy as np
    import torch
    import transformers
    from PIL import Image
    from safetensors import safe_open
    from transformers.initialization import no_init_weights
    from transformers.models.gemma4 import image_processing_gemma4, modeling_gemma4
    from transformers.models.gemma4.configuration_gemma4 import (
        Gemma4TextConfig,
        Gemma4VisionConfig,
    )

    if (
        torch.__version__.split("+")[0] != "2.14.1"
        or transformers.__version__ != "5.15.1"
    ):
        raise ValueError("canonical_reference_versions_changed")
    torch.set_num_threads(8)
    source, existing, output = (
        args.source.resolve(),
        args.fixtures.resolve(),
        args.output.resolve(),
    )
    root = Path(__file__).resolve().parents[1]
    if (
        source.name != MODEL_REVISION
        or output.exists()
        or subprocess.run(
            ["git", "check-ignore", "--quiet", "--", str(output)], cwd=root, check=False
        ).returncode
        != 0
    ):
        raise ValueError("invalid_reference_source_or_destination")
    original = json.loads((existing / "manifest.json").read_text())
    if [case["name"] for case in original["cases"]] != ["red", "blue", "pattern"]:
        raise ValueError("unexpected_rgb_fixture_cases")
    config = json.loads((source / "config.json").read_text())
    vision_config = Gemma4VisionConfig(**config["vision_config"])
    vision_config._attn_implementation = "sdpa"
    text_config = Gemma4TextConfig(**config["text_config"])
    text_config._attn_implementation = "sdpa"
    processor_config = json.loads((source / "processor_config.json").read_text())[
        "image_processor"
    ]
    processor = image_processing_gemma4.Gemma4ImageProcessor.from_dict(processor_config)

    def construct(cls, *config_args):
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with no_init_weights():
                return cls(*config_args)
        finally:
            torch.set_default_dtype(previous)

    PHASE = "vision_construction"
    vision = construct(modeling_gemma4.Gemma4VisionModel, vision_config)
    projector = construct(
        modeling_gemma4.Gemma4MultimodalEmbedder, vision_config, text_config
    )

    def load(module, prefix, *, tied=False):
        expected = set(module.state_dict())
        state = {}
        with safe_open(
            str(source / "model.safetensors"), framework="pt", device="cpu"
        ) as weights:
            for key in weights.keys():  # noqa: SIM118 - safetensors is not a mapping
                if key.startswith(prefix):
                    name = key.removeprefix(prefix)
                    name = "model." + name if tied else name
                    if name in expected:
                        state[name] = weights.get_tensor(key)
        if tied:
            state["lm_head.weight"] = state["model.embed_tokens.weight"]
        if set(state) != expected:
            raise ValueError("canonical_weight_inventory_changed")
        module.load_state_dict(state, strict=True, assign=True)
        module.eval()

    load(vision, "model.vision_tower.")
    load(projector, "model.embed_vision.")
    output.mkdir(parents=True, mode=0o700)
    definitions, prepared, feature_values, controls = [], [], [], []
    source_identity = {
        "reference_profile": "transformers-canonical",
        "reference_version": transformers.__version__,
        "reference_revision": "transformers-" + transformers.__version__,
        "source_sha256": digest(Path(modeling_gemma4.__file__)),
        "image_processor_sha256": digest(Path(image_processing_gemma4.__file__)),
        "model_revision": MODEL_REVISION,
        "reference_device": "cpu",
        "attention": "sdpa",
        "torch": torch.__version__,
        "torchvision": importlib.metadata.version("torchvision"),
        "pillow": importlib.metadata.version("pillow"),
        "reference_lock_sha256": digest(args.reference_lock.resolve()),
        "processor_config_sha256": digest(source / "processor_config.json"),
    }
    with torch.inference_mode():
        for index, old in enumerate(original["cases"]):
            PHASE = f"image_{index}_canonical_processor"
            image_path = existing / old["image"]
            if digest(image_path) != old["image_sha256"]:
                raise ValueError("rgb_fixture_bytes_changed")
            target = output / f"case-{index}.png"
            shutil.copyfile(image_path, target)
            target.chmod(0o600)
            image = Image.open(image_path).convert("RGB")
            batch = processor(images=image, return_tensors="pt")
            pixels = batch["pixel_values"]
            positions = batch["image_position_ids"]
            valid = (positions[0] >= 0).all(dim=-1)
            real_positions = positions[0, valid]
            grid_width = int(real_positions[:, 0].max()) + 1
            grid_height = int(real_positions[:, 1].max()) + 1
            image_tokens = int(batch["num_soft_tokens_per_image"][0])
            if image_tokens != old["image_tokens"]:
                raise ValueError("canonical_processor_changes_placeholder_geometry")
            normalized = (
                (2 * (pixels[0, valid] - 0.5)).bfloat16().float().numpy().astype("<f4")
            )
            patch_path = output / f"case-{index}-patches.f32"
            patch_path.write_bytes(normalized.tobytes())
            patch_path.chmod(0o600)
            position_path = output / f"case-{index}-positions.i64"
            position_path.write_bytes(positions.numpy().astype("<i8").tobytes())
            position_path.chmod(0o600)
            PHASE = f"image_{index}_canonical_bf16_vision"
            started = time.perf_counter()
            features = projector(
                vision(
                    pixel_values=pixels, pixel_position_ids=positions
                ).last_hidden_state
            )
            values = features.float().numpy().astype("<f4")
            if values.shape != (image_tokens, 1536) or not np.isfinite(values).all():
                raise ValueError("canonical_feature_output_invalid")
            feature_path = output / f"case-{index}-features.f32"
            feature_path.write_bytes(values.tobytes())
            feature_path.chmod(0o600)
            feature_values.append(values)
            prepared.append((pixels, positions))
            old_patches = np.frombuffer(
                (existing / old["patches"]).read_bytes(), dtype="<f4"
            ).reshape(normalized.shape)
            old_features = np.frombuffer(
                (existing / old["features"]).read_bytes(), dtype="<f4"
            ).reshape(values.shape)
            definitions.append(
                {
                    **old,
                    "image": target.name,
                    "patches": patch_path.name,
                    "positions": position_path.name,
                    "features": feature_path.name,
                    "logits": f"case-{index}-logits.f32",
                    "grid": [1, grid_height, grid_width],
                    "processed_dimensions": [grid_width * 16, grid_height * 16],
                    "image_sha256": digest(target),
                    "patches_sha256": digest(patch_path),
                    "features_sha256": digest(feature_path),
                    "positions_sha256": digest(position_path),
                }
            )
            controls.append(
                {
                    "case_index": index,
                    "vision_seconds": time.perf_counter() - started,
                    "patches_vs_mlx_profile": {
                        "different_elements": int(
                            np.count_nonzero(normalized != old_patches)
                        ),
                        "max_abs_error": float(np.abs(normalized - old_patches).max()),
                    },
                    "features_vs_mlx_profile": metrics(
                        values.reshape(-1), old_features.reshape(-1)
                    ),
                }
            )
        write_json(
            output / "tower-manifest.json",
            {
                **source_identity,
                "feature_references_complete": True,
                "cases": definitions,
                "controls": controls,
            },
        )

        PHASE = "vision_float32_control"
        vision.to(dtype=torch.float32)
        projector.to(dtype=torch.float32)
        for index, ((pixels, positions), bf16_values) in enumerate(
            zip(prepared, feature_values, strict=True)
        ):
            full = projector(
                vision(
                    pixel_values=pixels, pixel_position_ids=positions
                ).last_hidden_state
            )
            values = full.float().numpy().astype("<f4")
            path = output / f"case-{index}-features-fp32.f32"
            path.write_bytes(values.tobytes())
            path.chmod(0o600)
            controls[index]["features_bf16_to_float32"] = metrics(
                bf16_values.reshape(-1), values.reshape(-1)
            )
        del vision, projector

        PHASE = "canonical_text_construction"
        text = construct(modeling_gemma4.Gemma4ForCausalLM, text_config)
        load(text, "model.language_model.", tied=True)
        for index, case in enumerate(definitions):
            PHASE = f"image_{index}_canonical_language_forward"
            tokens = torch.tensor([case["tokens"]], dtype=torch.long)
            masked_ids = tokens.clone()
            masked_ids[tokens == config["image_token_id"]] = text_config.pad_token_id
            embeddings = text.model.embed_tokens(masked_ids)
            ple_inputs = text.model.get_per_layer_inputs(masked_ids, embeddings)
            start = case["token_start"]
            features = torch.from_numpy(feature_values[index].copy()).bfloat16()
            embeddings[:, start : start + case["image_tokens"]] = features
            if len(case["tokens"]) > 1024 or start + case["image_tokens"] >= len(
                case["tokens"]
            ):
                raise ValueError("canonical_image_prefill_geometry_changed")
            prefix = text(
                inputs_embeds=embeddings[:, :-1],
                per_layer_inputs=ple_inputs[:, :-1],
                use_cache=True,
                logits_to_keep=1,
            )
            result = text(
                input_ids=tokens[:, -1:],
                past_key_values=prefix.past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
            logits = result.logits[0, -1]
            values = logits.float().numpy().astype("<f4")
            logit_path = output / case["logits"]
            logit_path.write_bytes(values.tobytes())
            logit_path.chmod(0o600)
            generated = []
            for step in range(16):
                token = int(logits.argmax())
                if token in (1, 106):
                    break
                generated.append(token)
                if step + 1 < 16:
                    result = text(
                        torch.tensor([[token]], dtype=torch.long),
                        past_key_values=result.past_key_values,
                        use_cache=True,
                        logits_to_keep=1,
                    )
                    logits = result.logits[0, -1]
            case["reference_tokens"] = generated
            case["logits_sha256"] = digest(logit_path)
            old = original["cases"][index]
            old_logits = np.frombuffer(
                (existing / old["logits"]).read_bytes(), dtype="<f4"
            )
            controls[index]["logits_vs_mlx_profile"] = metrics(values, old_logits)
    manifest = {
        **source_identity,
        "forward_profile": "native-prefix-then-last-token",
        "source_weights_sha256": digest(source / "model.safetensors"),
        "cases": definitions,
        "controls": controls,
        "complete": True,
    }
    write_json(output / "manifest.json", manifest)
    return {
        "passed": True,
        "reference_profile": source_identity["reference_profile"],
        "reference_version": transformers.__version__,
        "cases": len(definitions),
        "manifest_sha256": digest(output / "manifest.json"),
        "controls": controls,
    }


def main() -> int:
    os.umask(0o077)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-lock", type=Path, required=True)
    args = parser.parse_args()
    try:
        with (
            open(os.devnull, "w") as quiet,
            contextlib.redirect_stdout(quiet),
            contextlib.redirect_stderr(quiet),
        ):
            report = run(args)
    except Exception as error:  # noqa: BLE001 - no model payloads or exception dumps
        print(
            json.dumps(
                {"passed": False, "phase": PHASE, "error_type": type(error).__name__}
            )
        )
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
