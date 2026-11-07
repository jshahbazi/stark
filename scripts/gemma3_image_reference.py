#!/usr/bin/env python3
"""Create private, original-source Gemma 3 CPU image reference fixtures.

Uses unchanged Transformers Gemma3/SigLIP forward functions. All pixels,
features, prompts, tokens and logits remain inside an ignored private directory.
Only source identities and aggregate counts are printed. This fixture checks
numerical implementation and request lifecycle; it does not measure quality.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import json
import os
import subprocess
from pathlib import Path

REVISION = "093f9f388b31de276ce2de164bdc2081324b9767"
TEMPLATE_HASH = "7de1c58e208eda46e9c7f86397df37ec49883aeece39fb961e0a6b24088dd3c4"
PHASE = "startup"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    path.chmod(0o600)


def run(args):
    global PHASE
    import numpy as np
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration
    from transformers.models.gemma3 import modeling_gemma3
    from transformers.models.siglip import modeling_siglip

    if torch.__version__.split("+")[0] != "2.14.1" or transformers.__version__ != "5.15.1":
        raise ValueError("reference_versions_changed")
    source, output = args.source.resolve(strict=True), args.output.resolve()
    repo = Path(__file__).resolve().parents[1]
    if source.name != REVISION or output.exists() or subprocess.run(
        ["git", "check-ignore", "--quiet", "--", str(output)], cwd=repo, check=False
    ).returncode:
        raise ValueError("source_or_private_output_identity_invalid")
    template = json.loads((source / "tokenizer_config.json").read_text())["chat_template"]
    if hashlib.sha256(template.encode()).hexdigest() != TEMPLATE_HASH:
        raise ValueError("reference_template_changed")
    torch.set_num_threads(4)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    PHASE = "model_load"
    transformers.utils.logging.set_verbosity_error()
    model = Gemma3ForConditionalGeneration.from_pretrained(
        source, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager"
    ).eval()
    processor = AutoProcessor.from_pretrained(source, local_files_only=True)
    definitions = []
    prompt = "Describe the image briefly."
    for name, width, height in (("red", 64, 48), ("blue", 48, 64), ("pattern", 111, 79)):
        PHASE = "fixture_" + name
        pixels = np.zeros((height, width, 3), dtype=np.uint8)
        if name == "red":
            pixels[:, :, 0] = 255
        elif name == "blue":
            pixels[:, :, 2] = 255
        else:
            yy, xx = np.indices((height, width))
            pixels[:, :, 0] = (xx * 17 + yy * 3) % 256
            pixels[:, :, 1] = (xx * 5 + yy * 19) % 256
            pixels[:, :, 2] = (xx * 11 + yy * 7) % 256
        image = Image.fromarray(pixels)
        image_path = output / f"{name}.png"
        image.save(image_path)
        image_path.chmod(0o600)
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        rendered = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        raw_tokens = processor.tokenizer.encode(rendered, add_special_tokens=False)
        inputs = processor(text=rendered, images=image, return_tensors="pt", do_pan_and_scan=False, add_special_tokens=False)
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        if len(inputs["input_ids"][0]) - len(raw_tokens) != 258:
            raise ValueError("processor_expansion_accounting_changed")
        patches = inputs["pixel_values"].reshape(1, 3, 64, 14, 64, 14).permute(0, 2, 4, 1, 3, 5).reshape(4096, 588)
        patch_path = output / f"{name}-patches.f32"
        patches.float().numpy().astype("<f4").tofile(patch_path)
        patch_path.chmod(0o600)
        with torch.inference_mode():
            PHASE = "vision_" + name
            features = model.model.get_image_features(inputs["pixel_values"])
            if not isinstance(features, torch.Tensor):
                features = features.pooler_output
            features = features.reshape(256, 2560)
            features_path = output / f"{name}-features.f32"
            features.float().numpy().astype("<f4").tofile(features_path)
            features_path.chmod(0o600)
            PHASE = "generation_" + name
            # The full unchanged multimodal forward independently recomputes
            # its image features and builds the official bidirectional masks.
            result = model(**inputs, use_cache=True, logits_to_keep=1)
            logits = []
            generated = []
            cache = result.past_key_values
            for step in range(args.tokens):
                row = result.logits[0, -1].float()
                logits.append(row.numpy())
                token = int(row.argmax().item())
                generated.append(token)
                if step + 1 < args.tokens:
                    next_token = torch.tensor([[token]], dtype=torch.long)
                    result = model(input_ids=next_token, past_key_values=cache, use_cache=True, logits_to_keep=1)
                    cache = result.past_key_values
            logits_path = output / f"{name}-logits.f32"
            np.stack(logits).astype("<f4").tofile(logits_path)
            logits_path.chmod(0o600)
        write_json(output / f"{name}-tokens.json", {"raw": raw_tokens, "expanded": inputs["input_ids"][0].tolist(),
            "generated": generated})
        definitions.append({"name": name, "image": image_path.name, "patches": patch_path.name,
            "features": features_path.name, "logits": logits_path.name, "tokens": f"{name}-tokens.json",
            "hashes": {path.name: digest(path) for path in (image_path, patch_path, features_path, logits_path,
                output / f"{name}-tokens.json")}})
    PHASE = "manifest"
    manifest = {"schema_version": 1, "source_repo": "google/gemma-3-4b-it", "source_revision": REVISION,
        "reference_profile": "transformers-canonical-eager-bf16", "torch": torch.__version__.split("+")[0],
        "transformers": transformers.__version__, "tokens_per_case": args.tokens,
        "source_hashes": {path.name: digest(path) for path in (source / "config.json", source / "model.safetensors.index.json")},
        "reference_hashes": {"modeling_gemma3.py": digest(Path(inspect.getfile(modeling_gemma3))),
            "modeling_siglip.py": digest(Path(inspect.getfile(modeling_siglip)))}, "cases": definitions}
    write_json(output / "manifest.json", manifest)
    return {"passed": True, "cases": len(definitions), "tokens_per_case": args.tokens,
        "manifest_sha256": digest(output / "manifest.json"), "reference_hashes": manifest["reference_hashes"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokens", default=8, type=int, choices=range(1, 33))
    args = parser.parse_args()
    os.umask(0o077)
    try:
        result = run(args)
    except Exception:
        print(json.dumps({"passed": False, "phase": PHASE}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
