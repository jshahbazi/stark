"""Create private encoder-free Gemma 4 12B vision references from pinned extracted weights.

This runs only the official image tower and projector on CPU. It provides no
language-model or generation qualification. Raw images and vectors stay ignored.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

REVISION = "43c9d014b231474fa7212dbb48a762566c2b255d"
SOURCE = "c52d9b27fc83b6ced166aee816b235e1db057fde10df6c1eff64bbc91fb16f39"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def run(args):
    import numpy as np
    import torch
    import transformers
    from PIL import Image
    from safetensors.torch import load_file
    from transformers.initialization import no_init_weights
    from transformers.models.gemma4_unified import (
        image_processing_gemma4_unified as image_processing_gemma4,
    )
    from transformers.models.gemma4_unified import (
        modeling_gemma4_unified as modeling_gemma4,
    )
    from transformers.models.gemma4_unified.configuration_gemma4_unified import (
        Gemma4UnifiedTextConfig as Gemma4TextConfig,
    )
    from transformers.models.gemma4_unified.configuration_gemma4_unified import (
        Gemma4UnifiedVisionConfig as Gemma4VisionConfig,
    )

    if (
        torch.__version__.split("+")[0] != "2.14.1"
        or transformers.__version__ != "5.15.1"
    ):
        raise ValueError("reference_versions_changed")
    if digest(modeling_gemma4.__file__) != SOURCE:
        raise ValueError("reference_source_changed")
    source, output, fixtures = (
        args.source.resolve(),
        args.output.resolve(),
        args.fixtures.resolve(),
    )
    root = Path(__file__).resolve().parents[1]
    if (
        output.exists()
        or subprocess.run(
            ["git", "check-ignore", "--quiet", "--", str(output)], cwd=root, check=False
        ).returncode
    ):
        raise ValueError("output_must_be_new_and_ignored")
    receipt = json.loads((source / "receipt.json").read_text())
    if receipt["repo"] != "google/gemma-4-12B-it" or receipt["revision"] != REVISION:
        raise ValueError("source_identity_changed")
    if digest(source / "vision.safetensors") != receipt["sha256"]["vision.safetensors"]:
        raise ValueError("source_weights_changed")
    torch.set_num_threads(4)
    config = json.loads((source / "config.json").read_text())
    vision_config = Gemma4VisionConfig(**config["vision_config"])
    vision_config._attn_implementation = "sdpa"
    text_config = Gemma4TextConfig(**config["text_config"])
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with no_init_weights():
            tower = modeling_gemma4.Gemma4UnifiedVisionEmbedder(
                vision_config, text_config
            )
    finally:
        torch.set_default_dtype(previous)
    state = load_file(source / "vision.safetensors", device="cpu")
    # The pinned 5.10 checkpoint separates vision_embedder and embed_vision;
    # 5.15.1 nests the final projector inside the unified vision embedder.
    selected = {}
    for name, value in state.items():
        if name.startswith("model.vision_embedder."):
            selected[name.removeprefix("model.vision_embedder.")] = value
        elif name.startswith("model.embed_vision."):
            selected[
                "multimodal_embedder." + name.removeprefix("model.embed_vision.")
            ] = value
        else:
            raise ValueError("unexpected_source_tensor")
    tower.load_state_dict(selected, strict=True, assign=True)
    tower.eval()
    del selected
    del state
    processor = image_processing_gemma4.Gemma4UnifiedImageProcessor.from_dict(
        json.loads((source / "processor_config.json").read_text())["image_processor"]
    )
    original = json.loads((fixtures / "manifest.json").read_text())
    if [case["name"] for case in original["cases"]] != ["red", "blue", "pattern"]:
        raise ValueError("fixture_cases_changed")
    output.mkdir(mode=0o700)
    cases = []
    with torch.inference_mode():
        for index, case in enumerate(original["cases"]):
            image = fixtures / case["image"]
            if digest(image) != case["image_sha256"]:
                raise ValueError("fixture_image_changed")
            image_name = f"case-{index}.png"
            shutil.copyfile(image, output / image_name)
            batch = processor(
                images=Image.open(image).convert("RGB"), return_tensors="pt"
            )
            positions = batch["image_position_ids"]
            valid = (positions[0] >= 0).all(dim=-1)
            width = int(positions[0, valid, 0].max()) + 1
            height = int(positions[0, valid, 1].max()) + 1
            patches = (
                batch["pixel_values"][0, valid].bfloat16().float().numpy().astype("<f4")
            )
            features = tower(
                pixel_values=batch["pixel_values"], image_position_ids=positions
            )[0, valid]
            values = features.float().numpy().astype("<f4")
            count = int(batch["num_soft_tokens_per_image"][0])
            if values.shape != (count, 3840) or not np.isfinite(values).all():
                raise ValueError("feature_output_invalid")
            patch_name, feature_name = (
                f"case-{index}-patches.f32",
                f"case-{index}-features.f32",
            )
            (output / patch_name).write_bytes(patches.tobytes())
            (output / feature_name).write_bytes(values.tobytes())
            cases.append(
                {
                    "name": case["name"],
                    "image": image_name,
                    "image_sha256": digest(output / image_name),
                    "patches": patch_name,
                    "patches_sha256": digest(output / patch_name),
                    "features": feature_name,
                    "features_sha256": digest(output / feature_name),
                    "grid": [1, height * 3, width * 3],
                    "image_tokens": count,
                }
            )
    manifest = {
        "scope": "vision-features-only",
        "model_revision": REVISION,
        "model_dim": 3840,
        "reference_profile": "transformers-canonical",
        "reference_version": transformers.__version__,
        "source_sha256": SOURCE,
        "source_receipt_sha256": digest(source / "receipt.json"),
        "reference_lock_sha256": digest(args.reference_lock),
        "cases": cases,
        "complete": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
    return {
        "passed": True,
        "scope": "vision-features-only",
        "cases": len(cases),
        "manifest_sha256": digest(output / "manifest.json"),
    }


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "fixtures", "output", "reference-lock"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        with (
            open(os.devnull, "w") as quiet,
            contextlib.redirect_stdout(quiet),
            contextlib.redirect_stderr(quiet),
        ):
            result = run(args)
    except Exception as error:  # noqa: BLE001 - keep private framework failures content-free
        print(json.dumps({"passed": False, "error_type": type(error).__name__}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
