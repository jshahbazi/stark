"""Create private larger Gemma 4 vision references from pinned extracted weights.

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

REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"
SOURCE = "4f874549f79deda4bb9ce7fb7d7b8e7349122be1711f733f7a4157e46ed10cc3"


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
    from transformers.models.gemma4 import image_processing_gemma4, modeling_gemma4
    from transformers.models.gemma4.configuration_gemma4 import (
        Gemma4TextConfig,
        Gemma4VisionConfig,
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
    if receipt["repo"] != "google/gemma-4-31B-it" or receipt["revision"] != REVISION:
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
            tower = modeling_gemma4.Gemma4VisionModel(vision_config)
            projector = modeling_gemma4.Gemma4MultimodalEmbedder(
                vision_config, text_config
            )
    finally:
        torch.set_default_dtype(previous)
    state = load_file(source / "vision.safetensors", device="cpu")
    for module, prefix in [
        (tower, "model.vision_tower."),
        (projector, "model.embed_vision."),
    ]:
        selected = {
            name.removeprefix(prefix): value
            for name, value in state.items()
            if name.startswith(prefix)
        }
        module.load_state_dict(selected, strict=True, assign=True)
        module.eval()
    del state
    processor = image_processing_gemma4.Gemma4ImageProcessor.from_dict(
        json.loads((source / "processor_config.json").read_text())["image_processor"]
    )
    original = json.loads((fixtures / "manifest.json").read_text())
    if [case["name"] for case in original["cases"]] != ["red", "blue", "pattern"]:
        raise ValueError("fixture_cases_changed")
    output.mkdir(mode=0o700)
    cases = []
    stage_bytes = [0]
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
                (2 * (batch["pixel_values"][0, valid] - 0.5))
                .bfloat16()
                .float()
                .numpy()
                .astype("<f4")
            )
            stages = {}
            hooks = []
            focus_case = args.capture_full_attention_case
            focus_mode = "attention"
            if args.capture_mlp_down_case is not None:
                focus_case, focus_mode = args.capture_mlp_down_case, "mlp"
            focused = focus_case is not None
            capture = args.capture_stages and (not focused or index == focus_case)
            original_rotary = modeling_gemma4.apply_multidimensional_rope
            rotary_calls = [0]

            def save_stage(
                tag,
                value,
                *,
                full=False,
                head0=False,
                _valid=valid,
                _index=index,
                _stages=stages,
                _focused=focus_mode if focused else None,
            ):
                if _focused == "attention":
                    if tag not in {
                        "prepared_queries_0",
                        "prepared_keys_0",
                        "prepared_values_0",
                        "attention_0",
                    }:
                        return
                    full, head0 = True, False
                elif _focused == "mlp":
                    allowed = {
                        "pre_ff_norm_0",
                        "gate_pre_gelu_0",
                        "gate_post_gelu_0",
                        "up_projection_0",
                        "mlp_product_0",
                        "mlp_down_input_0",
                        "mlp_down_output_0",
                    }
                    if tag not in allowed:
                        return
                    full = tag in {"mlp_down_input_0", "mlp_down_output_0"}
                value = value.detach()
                if value.ndim >= 3:
                    value = value[0, _valid]
                if value.ndim == 3:
                    value = (
                        value[:, 0, :] if head0 else value.reshape(value.shape[0], -1)
                    )
                if not full:
                    value = value[:8]
                values = value.float().numpy().astype("<f4")
                data = values.tobytes()
                stage_bytes[0] += len(data)
                if stage_bytes[0] > 64 * 1024 * 1024:
                    raise ValueError("stage_capture_budget_exceeded")
                name = f"case-{_index}-stage-{tag}.f32"
                (output / name).write_bytes(data)
                _stages[tag] = {
                    "file": name,
                    "sha256": digest(output / name),
                    "shape": list(values.shape),
                }

            def rotated(
                value, *parts, _rotary=original_rotary, _calls=rotary_calls, **kwargs
            ):
                result = _rotary(value, *parts, **kwargs)
                call = _calls[0]
                _calls[0] += 1
                if call < 2:
                    save_stage(
                        "prepared_queries_0" if call == 0 else "prepared_keys_0",
                        result,
                        full=True,
                        head0=True,
                    )
                return result

            def capture_values(_module, _inputs, value):
                save_stage("v_norm_0", value)
                save_stage("prepared_values_0", value, full=True, head0=True)

            if capture:
                hooks.append(
                    tower.patch_embedder.input_proj.register_forward_hook(
                        lambda _m, _i, value: save_stage("patch_projection", value)
                    )
                )
                hooks.append(
                    tower.patch_embedder.register_forward_hook(
                        lambda _m, _i, value: save_stage("patch_positions", value)
                    )
                )
                first = tower.encoder.layers[0]
                if args.capture_mlp_down_case is not None:
                    for tag, module in [
                        ("pre_ff_norm_0", first.pre_feedforward_layernorm),
                        ("gate_pre_gelu_0", first.mlp.gate_proj),
                        ("gate_post_gelu_0", first.mlp.act_fn),
                        ("up_projection_0", first.mlp.up_proj),
                        ("mlp_down_output_0", first.mlp.down_proj),
                    ]:
                        hooks.append(
                            module.register_forward_hook(
                                lambda _m, _i, value, tag=tag: save_stage(tag, value)
                            )
                        )

                    def capture_down(_module, inputs):
                        save_stage("mlp_down_input_0", inputs[0], full=True)
                        save_stage("mlp_product_0", inputs[0])

                    hooks.append(
                        first.mlp.down_proj.register_forward_pre_hook(capture_down)
                    )
                hooks.append(
                    first.input_layernorm.register_forward_hook(
                        lambda _m, _i, value: save_stage("input_norm_0", value)
                    )
                )
                for tag, module in [
                    ("q_raw_0", first.self_attn.q_proj),
                    ("k_raw_0", first.self_attn.k_proj),
                    ("v_raw_0", first.self_attn.v_proj),
                    ("q_norm_0", first.self_attn.q_norm),
                    ("k_norm_0", first.self_attn.k_norm),
                ]:
                    hooks.append(
                        module.register_forward_hook(
                            lambda _m, _i, value, tag=tag: save_stage(tag, value)
                        )
                    )
                hooks.append(
                    first.self_attn.v_norm.register_forward_hook(capture_values)
                )
                hooks.append(
                    first.self_attn.o_proj.register_forward_pre_hook(
                        lambda _m, inputs: save_stage("attention_0", inputs[0])
                    )
                )
                for layer, module in enumerate(tower.encoder.layers):
                    hooks.append(
                        module.register_forward_hook(
                            lambda _m, _i, value, layer=layer: save_stage(
                                f"block_{layer}", value
                            )
                        )
                    )
                hooks.append(
                    tower.encoder.register_forward_hook(
                        lambda _m, _i, value: save_stage(
                            "encoder_final", value.last_hidden_state, full=True
                        )
                    )
                )
                hooks.append(
                    tower.pooler.register_forward_hook(
                        lambda _m, _i, value: save_stage(
                            "pool_scaled", value[0][value[1]], full=True
                        )
                    )
                )
                hooks.append(
                    projector.embedding_pre_projection_norm.register_forward_hook(
                        lambda _m, _i, value: save_stage(
                            "projector_input", value, full=True
                        )
                    )
                )
                modeling_gemma4.apply_multidimensional_rope = rotated
            try:
                encoded = tower(
                    pixel_values=batch["pixel_values"], pixel_position_ids=positions
                ).last_hidden_state
                if capture:
                    save_stage("standardized", encoded, full=True)
                features = projector(encoded)
            finally:
                modeling_gemma4.apply_multidimensional_rope = original_rotary
                for hook in hooks:
                    hook.remove()

            values = features.float().numpy().astype("<f4")
            count = int(batch["num_soft_tokens_per_image"][0])
            if values.shape != (count, 5376) or not np.isfinite(values).all():
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
                    "grid": [1, height, width],
                    "image_tokens": count,
                }
            )
            if args.capture_stages:
                cases[-1]["stages"] = stages
    manifest = {
        "scope": "vision-features-only",
        "model_revision": REVISION,
        "model_dim": 5376,
        "reference_profile": "transformers-canonical",
        "reference_version": transformers.__version__,
        "source_sha256": SOURCE,
        "source_receipt_sha256": digest(source / "receipt.json"),
        "reference_lock_sha256": digest(args.reference_lock),
        "cases": cases,
        "complete": True,
    }
    if args.capture_full_attention_case is not None:
        manifest["stage_focus"] = "first-layer-full-attention"
        manifest["stage_case_index"] = args.capture_full_attention_case
    if args.capture_mlp_down_case is not None:
        manifest["stage_focus"] = "first-layer-mlp-down"
        manifest["stage_case_index"] = args.capture_mlp_down_case
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
    parser.add_argument("--capture-stages", action="store_true")
    focus = parser.add_mutually_exclusive_group()
    focus.add_argument("--capture-full-attention-case", type=int, choices=(0, 1, 2))
    focus.add_argument("--capture-mlp-down-case", type=int, choices=(0, 1, 2))
    args = parser.parse_args()
    if (
        args.capture_full_attention_case is not None
        or args.capture_mlp_down_case is not None
    ) and not args.capture_stages:
        parser.error("focused captures require --capture-stages")
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
