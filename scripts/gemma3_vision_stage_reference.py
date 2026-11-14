#!/usr/bin/env python3
"""Extract private first-block diagnostics from the unchanged source SigLIP.

Loads only vision weights, retains bounded row prefixes and emits aggregate
identities. This diagnostic producer does not qualify whole-model execution.
"""

import argparse
import inspect
import json
import os
import subprocess
from pathlib import Path

from gemma3_image_reference import PROCESSOR_HASHES, REFERENCE_HASHES, REVISION, digest, write_json

PHASE = "admission"


def run(args):
    global PHASE
    import torch
    import transformers
    from PIL import Image
    from safetensors import safe_open
    from transformers import AutoConfig, AutoProcessor, SiglipVisionModel, image_processing_backends
    from transformers.models.gemma3 import image_processing_gemma3
    from transformers.models.siglip import modeling_siglip

    source, fixture, output = args.source.resolve(strict=True), args.fixture.resolve(strict=True), args.output.resolve()
    if source.name != REVISION or output.exists() or subprocess.run(
        ["git", "check-ignore", "--quiet", "--", str(output)], cwd=Path(__file__).resolve().parents[1], check=False
    ).returncode:
        raise ValueError("private_stage_paths_invalid")
    if torch.__version__.split("+")[0] != "2.14.1" or transformers.__version__ != "5.15.1":
        raise ValueError("stage_framework_versions_changed")
    modules = {"modeling_siglip.py": modeling_siglip, "image_processing_gemma3.py": image_processing_gemma3,
               "image_processing_backends.py": image_processing_backends}
    hashes = {name: digest(Path(inspect.getfile(module))) for name, module in modules.items()}
    if hashes != {"modeling_siglip.py": REFERENCE_HASHES["modeling_siglip.py"], **PROCESSOR_HASHES}:
        raise ValueError("stage_reference_code_changed")
    full = json.loads((fixture / "manifest.json").read_text())
    if digest(fixture / "resume-inputs.json") != full["resume_inputs_sha256"]:
        raise ValueError("stage_source_receipt_changed")
    resume = json.loads((fixture / "resume-inputs.json").read_text())
    for name, expected in resume["verified_source_files"].items():
        if Path(name).name != name or digest(source / name) != expected:
            raise ValueError("stage_original_source_changed")
    case = next(case for case in full["cases"] if case["name"] == "pattern")
    if any(digest(fixture / name) != checksum for name, checksum in case["hashes"].items()):
        raise ValueError("stage_fixture_changed")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    torch.set_num_threads(4)
    PHASE = "vision_weight_load"
    config = AutoConfig.from_pretrained(source, local_files_only=True, trust_remote_code=False).vision_config
    config._attn_implementation = "eager"
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        model = SiglipVisionModel(config).eval()
    finally:
        torch.set_default_dtype(previous_dtype)
    index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    state = {}
    for shard in sorted({value for key, value in index.items() if key.startswith("vision_tower.")}):
        with safe_open(source / shard, framework="pt", device="cpu") as handle:
            for name in sorted(key for key, value in index.items() if key.startswith("vision_tower.") and value == shard):
                state[name.removeprefix("vision_tower.vision_model.")] = handle.get_tensor(name)
    model.load_state_dict(state, strict=True)
    del state
    processor = AutoProcessor.from_pretrained(source, local_files_only=True, trust_remote_code=False)
    if type(processor.image_processor) is not image_processing_gemma3.Gemma3ImageProcessor:
        raise ValueError("stage_processor_backend_changed")
    values = processor.image_processor(images=Image.open(fixture / case["image"]), return_tensors="pt",
                                        do_pan_and_scan=False)["pixel_values"].to(torch.bfloat16)
    stages, captured = [], {}

    def save(name, value):
        if isinstance(value, tuple):
            value = value[0]
        value = value.detach()
        if value.ndim == 4 and value.shape[1] == 1152:
            value = value.flatten(2).transpose(1, 2)
        value = value.reshape(-1, value.shape[-1])
        captured[name] = value
        selected = value[:64].contiguous().float().numpy().astype("<f4")
        filename = name + ".f32"
        selected.tofile(output / filename)
        stages.append({"name": name, "filename": filename, "shape": list(value.shape),
                       "elements": int(selected.size), "sha256": digest(output / filename)})

    layer = model.encoder.layers[0]
    for name, module in [("patch", model.embeddings.patch_embedding), ("embeddings", model.embeddings),
                         ("norm1", layer.layer_norm1), ("query", layer.self_attn.q_proj), ("key", layer.self_attn.k_proj),
                         ("value", layer.self_attn.v_proj), ("projection", layer.self_attn.out_proj),
                         ("norm2", layer.layer_norm2), ("fc1", layer.mlp.fc1), ("activation", layer.mlp.activation_fn),
                         ("fc2", layer.mlp.fc2), ("block", layer)]:
        module.register_forward_hook(lambda _module, _inputs, value, name=name: save(name, value))
    layer.self_attn.out_proj.register_forward_pre_hook(lambda _module, arguments: save("attention", arguments[0]))
    PHASE = "original_first_block"
    with torch.inference_mode():
        hidden = model.embeddings(values)
        layer(hidden, attention_mask=None)
        save("qkv", torch.cat([captured[name] for name in ("query", "key", "value")], dim=-1))
        queries = captured["query"].reshape(1, 4096, 16, 72).transpose(1, 2)
        keys = captured["key"].reshape(1, 4096, 16, 72).transpose(1, 2)
        scores = torch.matmul(queries, keys.transpose(-1, -2))
        save("scores", scores[0, 0])
        scaled = scores * layer.self_attn.scale
        save("scaled_scores", scaled[0, 0])
        probabilities = torch.softmax(scaled, dim=-1, dtype=torch.float32).to(torch.bfloat16)
        save("probabilities", probabilities[0, 0])
    manifest = {"source_repo": "google/gemma-3-4b-it", "source_revision": source.name,
        "reference_scope": "diagnostic-original-SigLIP-first-block-with-observer-hooks",
        "derived_stages": ["qkv", "scores", "scaled_scores", "probabilities"], "retained_rows": 64,
        "producer_sha256": digest(Path(__file__).resolve()),
        "image_sha256": digest(fixture / case["image"]), "source_code_hashes": hashes,
        "full_reference_manifest_sha256": digest(fixture / "manifest.json"), "stages": stages}
    write_json(output / "manifest.json", manifest)
    return {"produced": True, "stages": len(stages), "manifest_sha256": digest(output / "manifest.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "fixture", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    try:
        result = run(args)
    except Exception:
        print(json.dumps({"produced": False, "phase": PHASE}), flush=True)
        return 1
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
