#!/usr/bin/env python3
"""Create private source-bound Gemma 4 image references from unchanged MLX-VLM.

The loader imports selected forward modules from a verified clean pinned source
tree without executing MLX-VLM's unrelated top-level server/UI entry points.
The model forward and its native MLX kernels are unmodified. Audio is excluded
from construction and source loading. Only aggregate identities are printed.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
import types
from pathlib import Path

REVISION = "53ebd1647cb83549b482a031ea70ba9d9ea60a04"
MODEL_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
PHASE = "startup"


class CheckFailed(RuntimeError):
    pass


def require(value, code):
    if not value:
        raise CheckFailed(code)


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    path.chmod(0o600)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install_forward_packages(root):
    check = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, check=False
    )
    require(
        check.returncode == 0 and check.stdout.decode().strip() == REVISION,
        "reference_revision_changed",
    )
    check = subprocess.run(
        ["git", "-C", str(root), "diff", "--exit-code", "HEAD", "--", "mlx_vlm"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    require(check.returncode == 0, "reference_sources_changed")
    # Import machinery still loads each requested .py file normally. Skipping
    # only package __init__ avoids importing unrelated chat/server/audio APIs.
    for path in [root / "mlx_vlm", *(root / "mlx_vlm").rglob("__init__.py")]:
        folder = path if path.is_dir() else path.parent
        name = ".".join(folder.relative_to(root).parts)
        package = types.ModuleType(name)
        package.__package__ = name
        package.__path__ = [str(folder)]
        sys.modules[name] = package
    return importlib.import_module("mlx_vlm.models.gemma4.gemma4")


def run(args):
    global PHASE
    import mlx.core as mx
    import numpy as np
    from PIL import Image
    from transformers import AutoTokenizer
    from transformers.utils import logging

    logging.set_verbosity_error()
    require(
        importlib.metadata.version("mlx") == "0.32.4", "reference_mlx_version_changed"
    )
    require(
        importlib.metadata.version("transformers") == "5.14.0",
        "reference_transformers_version_changed",
    )
    PHASE = "source_import"
    module = install_forward_packages(args.mlx_vlm_source.resolve(strict=True))
    configs = importlib.import_module("mlx_vlm.models.gemma4.config")
    source = args.source.resolve(strict=True)
    require(source.name == MODEL_REVISION, "source_model_revision_changed")
    config = json.loads((source / "config.json").read_text())
    settings = {
        k: v
        for k, v in config.items()
        if k not in {"text_config", "vision_config", "audio_config"}
    }
    model_config = configs.ModelConfig.from_dict(settings)
    model_config.text_config = configs.TextConfig.from_dict(config["text_config"])
    model_config.vision_config = configs.VisionConfig.from_dict(config["vision_config"])
    model_config.audio_config = None
    PHASE = "model_construction"
    model = module.Model(model_config)
    PHASE = "weight_loading"
    weights = mx.load(str(source / "model.safetensors"))
    weights = model.language_model.sanitize(model.sanitize(weights))
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    PHASE = "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(
        source, local_files_only=True, trust_remote_code=False
    )
    tokenizer.chat_template = (source / "chat_template.jinja").read_text()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    definitions = []
    for index, (name, width, height) in enumerate(
        (("red", 64, 48), ("blue", 48, 64), ("pattern", 96, 96))
    ):
        pixels = np.zeros((height, width, 3), dtype=np.uint8)
        if name == "red":
            pixels[:, :, 0] = 255
        elif name == "blue":
            pixels[:, :, 2] = 255
        else:
            yy, xx = np.indices((height, width))
            pixels[:, :, 0] = ((xx // 8 + yy // 8) % 2 * 255).astype(np.uint8)
            pixels[:, :, 2] = 255 - pixels[:, :, 0]
            pixels[height // 3 : height // 3 + 8, :, 1] = 255
        image = Image.fromarray(pixels)
        image_path = output / f"case-{index}.png"
        image.save(image_path)
        image_path.chmod(0o600)
        factor = (2520 * 16**2 / (height * width)) ** 0.5
        target_h = int(factor * height // 48) * 48
        target_w = int(factor * width // 48) * 48
        require(target_h > 0 and target_w > 0, "reference_image_geometry_invalid")
        resized = np.asarray(
            image.resize((target_w, target_h), Image.Resampling.BICUBIC), dtype=np.uint8
        )
        raw = resized.astype(np.float32) * np.float32(1 / 255)
        gh, gw = target_h // 16, target_w // 16
        patch_pixels = (
            raw.reshape(gh, 16, gw, 16, 3)
            .transpose(0, 2, 1, 3, 4)
            .reshape(gh * gw, 768)
        )
        patch_tensor = (2 * (mx.array(patch_pixels) - 0.5)).astype(mx.bfloat16)
        patch_path = output / f"case-{index}-patches.f32"
        patch_path.write_bytes(
            np.asarray(patch_tensor.astype(mx.float32)).astype("<f4").tobytes()
        )
        # Float32 container holds exactly representable BF16 values; do not
        # rely on NumPy supporting an independent BF16 dtype.
        patch_path.chmod(0o600)
        tensor = mx.array(raw.transpose(2, 0, 1)[None])
        PHASE = f"image_{index}_vision_forward"
        features = model.encode_image(tensor)
        mx.eval(features)
        feature_values = np.asarray(features.astype(mx.float32))[0]
        require(bool(np.isfinite(feature_values).all()), "reference_features_nonfinite")
        image_tokens = gh * gw // 9
        require(
            feature_values.shape == (image_tokens, 1536),
            "reference_feature_shape_invalid",
        )
        feature_path = output / f"case-{index}-features.f32"
        feature_path.write_bytes(feature_values.astype("<f4").tobytes())
        feature_path.chmod(0o600)
        PHASE = f"image_{index}_template"
        rendered = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": "What is the dominant color? Answer with one color name.",
                        },
                    ],
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        require(
            rendered.count(tokenizer.image_token) == 1,
            "reference_placeholder_count_invalid",
        )
        expanded = rendered.replace(
            tokenizer.image_token,
            tokenizer.boi_token
            + tokenizer.image_token * image_tokens
            + tokenizer.eoi_token,
        )
        tokens = tokenizer.encode(expanded, add_special_tokens=False)
        unexpanded = tokenizer.encode(rendered, add_special_tokens=False)
        start = tokens.index(config["image_token_id"])
        cache = model.make_cache()
        # Native score/stream caches the prefix and then reads one final token.
        PHASE = f"image_{index}_language_forward"
        result = model(
            mx.array([tokens[:-1]], dtype=mx.int32),
            pixel_values=tensor,
            cached_image_features=features,
            cache=cache,
        )
        mx.eval(result.logits)
        result = model.language_model(
            mx.array([tokens[-1:]], dtype=mx.int32), cache=cache
        )
        mx.eval(result.logits)
        logits = result.logits[0, -1]
        logit_values = np.asarray(logits.astype(mx.float32))
        require(bool(np.isfinite(logit_values).all()), "reference_logits_nonfinite")
        logit_path = output / f"case-{index}-logits.f32"
        logit_path.write_bytes(logit_values.astype("<f4").tobytes())
        logit_path.chmod(0o600)
        generated = []
        for step in range(16):
            token = int(mx.argmax(logits).item())
            if token in {1, 106}:
                break
            generated.append(token)
            if step + 1 < 16:
                result = model.language_model(
                    mx.array([[token]], dtype=mx.int32), cache=cache
                )
                mx.eval(result.logits)
                logits = result.logits[0, -1]
        definitions.append(
            {
                "name": name,
                "image": image_path.name,
                "patches": patch_path.name,
                "features": feature_path.name,
                "logits": logit_path.name,
                "rendered_prompt": rendered,
                "unexpanded_tokens": unexpanded,
                "tokens": tokens,
                "token_start": start,
                "grid": [1, gh, gw],
                "image_tokens": image_tokens,
                "processed_dimensions": [target_w, target_h],
                "reference_tokens": generated,
                "context_tokens": 1024,
                "max_new_tokens": 16,
                "image_sha256": sha(image_path),
                "patches_sha256": sha(patch_path),
                "features_sha256": sha(feature_path),
                "logits_sha256": sha(logit_path),
            }
        )
    source_hashes = {}
    root = args.mlx_vlm_source.resolve()
    for name, imported in sys.modules.copy().items():
        file = getattr(imported, "__file__", None)
        if name.startswith("mlx_vlm") and file and Path(file).is_relative_to(root):
            source_hashes[str(Path(file).relative_to(root))] = sha(Path(file))
    manifest = {
        "reference_revision": REVISION,
        "model_revision": MODEL_REVISION,
        "mlx": importlib.metadata.version("mlx"),
        "transformers": importlib.metadata.version("transformers"),
        "forward_profile": "native-prefix-then-last-token",
        "vision_attention": "source-fused-sdpa",
        "source_files_sha256": source_hashes,
        "cases": definitions,
    }
    write_json(output / "manifest.json", manifest)
    return {
        "passed": True,
        "cases": len(definitions),
        "reference_revision": REVISION,
        "model_revision": MODEL_REVISION,
        "mlx": manifest["mlx"],
        "manifest_sha256": sha(output / "manifest.json"),
        "image_tokens": [case["image_tokens"] for case in definitions],
        "source_forward_files": len(source_hashes),
    }


def main():
    os.umask(0o077)
    for key in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_HUB_DISABLE_PROGRESS_BARS",
        "HF_HUB_DISABLE_TELEMETRY",
    ):
        os.environ[key] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--mlx-vlm-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        with (
            open(os.devnull, "w") as quiet,
            contextlib.redirect_stdout(quiet),
            contextlib.redirect_stderr(quiet),
        ):
            report = run(args)
        print(json.dumps(report, sort_keys=True))
        return 0
    except CheckFailed as failure:
        print(json.dumps({"passed": False, "check": str(failure)}))
    except Exception as failure:  # noqa: BLE001 - no model payloads or exception dumps
        print(
            json.dumps(
                {
                    "passed": False,
                    "check": "gemma_image_reference_failed",
                    "exception_type": type(failure).__name__,
                    "phase": PHASE,
                }
            )
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
