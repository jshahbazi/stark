"""Prepare the pinned Base Labs Elephant 1 decision/image checkpoint."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

from tokenizers import Tokenizer

from ._build_metadata import file_hash as _hash
from .adapters.qwen35 import build_qwen35_plan
from .checkpoint_metadata import CheckpointMetadata
from .elephant_protocol import SLOT_LABELS
from .streaming import (
    ConversionError,
    TensorOperation,
    UnsupportedConversion,
    authorize_source_path,
    load_checkpoint,
    read_json,
    read_regular_bytes,
    write_plan,
)

SOURCE_REPO = "baselabs/elephant-1"
SOURCE_REVISION = "b3f8a93462fb1051a9de031e94a7ff380e254072"
MAX_INPUT_TOKENS = 16384
MAX_PIXELS = 16777216
SOURCE_LICENSE = "MIT"
_METADATA_FILES = (
    "README.md",
    "chat_template.jinja",
    "config.json",
    "constraints.txt",
    "engine.py",
    "generation_config.json",
    "processor_config.json",
    "readout.json",
    "serve.patch",
    "setup.sh",
    "tokenizer.json",
    "tokenizer_config.json",
)
_EXPECTED_PROCESSOR = {
    "do_convert_rgb": True,
    "do_normalize": True,
    "do_rescale": True,
    "do_resize": True,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "merge_size": 2,
    "patch_size": 16,
    "resample": 3,
    "rescale_factor": 1 / 255,
    "size": {"longest_edge": MAX_PIXELS, "shortest_edge": 65536},
    "temporal_patch_size": 2,
    "image_processor_type": "Qwen2VLImageProcessor",
}


def validate_source_metadata(config, readout, processor):
    overrides = config.get("modeling_overrides")
    text = config.get("text_config")
    vision = config.get("vision_config")
    if (
        config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]
        or config.get("model_type") != "qwen3_5"
        or config.get("tie_word_embeddings") is not False
        or not isinstance(text, dict)
        or not isinstance(vision, dict)
        or text.get("model_type") != "qwen3_5_text"
        or text.get("max_position_embeddings") != 262144
        or overrides
        != {
            "attention": {"full_attention": "block_bidirectional", "linear_attention": "causal"},
            "readout": "slot_lm",
            "head_dim": 256,
            "n_slots": 256,
        }
        or readout != {"readout": "slot_lm", "head_dim": 256, "n_slots": 256}
        or processor.get("processor_class") != "Qwen3VLProcessor"
        or processor.get("image_processor") != _EXPECTED_PROCESSOR
        or not isinstance(config.get("image_token_id"), int)
        or not isinstance(config.get("vision_start_token_id"), int)
        or not isinstance(config.get("vision_end_token_id"), int)
    ):
        raise UnsupportedConversion("Unsupported Elephant checkpoint protocol or metadata")
    rope = text.get("rope_parameters")
    if (
        not isinstance(rope, dict)
        or rope.get("mrope_interleaved") is not True
        or rope.get("mrope_section") != [11, 11, 10]
        or rope.get("rope_theta") != 10000000
    ):
        raise UnsupportedConversion("Elephant images require the pinned Qwen3.5 M-RoPE profile")
    if vision.get("rope_parameters") != {"rope_theta": 10000.0, "rope_type": "axial"}:
        raise UnsupportedConversion("Elephant vision rotary configuration is unsupported")


def _slot_token_ids(tokenizer):
    token_ids = []
    for label in SLOT_LABELS:
        ids = tokenizer.encode(label, add_special_tokens=False).ids
        if len(ids) != 1 or tokenizer.id_to_token(ids[0]) != label or tokenizer.decode(ids) != label:
            raise ConversionError("Elephant answer slots are not unique single round-trip tokens")
        token_ids.append(ids[0])
    if len(set(token_ids)) != len(token_ids):
        raise ConversionError("Elephant answer slot token IDs collide")
    return token_ids


def build_elephant_plan(source):
    source = Path(source).resolve()
    config = read_json(source / "config.json")
    readout = read_json(source / "readout.json")
    processor = read_json(source / "processor_config.json")
    validate_source_metadata(config, readout, processor)
    tensors = load_checkpoint(source)
    if any(not key.startswith(("model.language_model.", "model.visual.", "lm_head.")) for key in tensors):
        raise ConversionError("Elephant backbone has unexpected tensor namespaces")
    planned_config = dict(config)
    planned_config.pop("modeling_overrides")
    planned_vision = dict(config["vision_config"])
    planned_vision.pop("rope_parameters")
    planned_config["vision_config"] = planned_vision
    plan = build_qwen35_plan(
        source, include_vision=True, checkpoint_metadata=CheckpointMetadata(planned_config, tensors)
    )
    if plan.mtp_config is not None:
        raise UnsupportedConversion("Elephant does not implement a generative MTP path")
    decoder = plan.decoder_config
    sections = config["text_config"]["rope_parameters"]["mrope_section"]
    for layer in decoder["transformer_config"]["layer_configs"]:
        if layer["mixer_config"]["type"] == "AttentionConfig":
            layer["mixer_config"]["is_causal"] = True
            rope = layer["rope_config"]
            half_width = rope["head_dim"] // 2
            if sum(sections) != half_width or any(
                sections[axis] and (sections[axis] - 1) * 3 + axis >= half_width for axis in (1, 2)
            ):
                raise UnsupportedConversion("Elephant M-RoPE sections disagree with the rotary width")
            layer["rope_config"] = {
                **rope,
                "type": "Qwen35MRoPEConfig",
                "mrope_section": list(sections),
                "interleaved": True,
            }
    tokenizer = Tokenizer.from_file(str(source / "tokenizer.json"))
    slot_ids = _slot_token_ids(tokenizer)
    output = next(
        (
            operation
            for operation in plan.operations
            if operation.name == "decoder.embedding.output_embedding.weights"
        ),
        None,
    )
    if (
        output is None
        or output.kind != "copy"
        or output.dtype != "BF16"
        or len(output.sources) != 1
        or len(output.shape) != 2
    ):
        raise ConversionError("Elephant requires a full BF16 untied LM head")
    head = output.sources[0]
    vocab, dim = output.shape
    if head.shape != output.shape or vocab != plan.decoder_config["vocab_size"] or max(slot_ids) >= vocab:
        raise ConversionError("Elephant LM head geometry disagrees with its decoder")
    row_bytes = dim * head.itemsize
    rows = tuple(
        replace(head, shape=(1, dim), offset=head.offset + token_id * row_bytes, nbytes=row_bytes)
        for token_id in slot_ids
    )
    plan = replace(
        plan,
        operations=(
            *plan.operations,
            TensorOperation("decision.readout.weights", rows, "BF16", (len(slot_ids), dim), "copy"),
        ),
        metadata={
            **plan.metadata,
            "decision.readout.spec": json.dumps({"type": "FullPrecisionSpec", "layout": "output_input"}),
        },
        input_bytes=plan.input_bytes + len(slot_ids) * row_bytes,
        source_tensors=plan.source_tensors + len(slot_ids),
        input_tensors=(*plan.input_tensors, *rows),
    )
    return plan, config, slot_ids


def prepare(source, output, *, revision=SOURCE_REVISION):
    source, output = Path(source).resolve(), Path(output).absolute()
    if output.exists():
        raise ConversionError("Elephant output already exists")
    if revision != SOURCE_REVISION:
        raise ConversionError("Elephant source revision must match the pinned revision")
    plan, config, slot_ids = build_elephant_plan(source)
    payloads = {
        name: read_regular_bytes(authorize_source_path(source, source / name)) for name in _METADATA_FILES
    }
    tokenizer = Tokenizer.from_str(payloads["tokenizer.json"].decode("utf-8"))
    if _slot_token_ids(tokenizer) != slot_ids:
        raise ConversionError("Elephant tokenizer changed while preparing")
    token_config = read_json(source / "tokenizer_config.json")
    template = payloads["chat_template.jinja"].decode("utf-8")
    if token_config.get("chat_template") not in (None, template):
        raise ConversionError("Elephant tokenizer templates disagree")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".elephant-", dir=output.parent))
    try:
        for name, payload in payloads.items():
            if name != "config.json":
                (staging / name).write_bytes(payload)
        stats = write_plan(plan, staging / "model.safetensors")
        provenance = {
            "repo": SOURCE_REPO,
            "revision": revision,
            "license": SOURCE_LICENSE,
            "metadata_sha256": {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()},
            "weight_files": {
                str(path.relative_to(source)): {"sha256": _hash(path), "size": path.stat().st_size}
                for path in sorted({info.path for info in plan.input_tensors})
            },
        }
        decision = {
            "protocol": "onejev-slot-lm-v2",
            "protocol_version": 5,
            "num_options": len(SLOT_LABELS),
            "readout": "slot_lm",
            "attention": "block_bidirectional",
            "slot_labels": list(SLOT_LABELS),
            "slot_token_ids": slot_ids,
            "temperature": 1.0,
            "max_input_tokens": MAX_INPUT_TOKENS,
            "max_pixels": MAX_PIXELS,
            "image_token_id": config["image_token_id"],
            "vision_start_token_id": config["vision_start_token_id"],
            "vision_end_token_id": config["vision_end_token_id"],
            "mrope_section": config["text_config"]["rope_parameters"]["mrope_section"],
        }
        root = {
            "type": "DeciderModelConfig",
            "decoder_config": plan.decoder_config,
            "vision_config": {**config["vision_config"], "layer_norm_eps": 1e-6},
            "decision_config": decision,
            "source": provenance,
            "capabilities": {
                "decision": True,
                "images": True,
                "max_images": 1,
                "locate": False,
                "prefix_reuse": False,
                "mtp": False,
                "qualification": "unverified",
            },
        }
        (staging / "config.json").write_text(json.dumps(root, indent=2) + "\n")
        (staging / "conversion.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "source": provenance,
                    "output_sha256": _hash(staging / "model.safetensors"),
                    "output_bytes": (staging / "model.safetensors").stat().st_size,
                    "config_sha256": _hash(staging / "config.json"),
                    "source_mtp": "config declaration without predictor tensors; non-generative checkpoint",
                    "prefix_reuse_blocker": (
                        "Block-bidirectional full-attention states depend on the complete question suffix."
                    ),
                    "precision": (
                        "BF16 weights/activations/KV; F32 normalization, recurrent state and slot readout"
                    ),
                    "stats": stats,
                },
                indent=2,
            )
            + "\n"
        )
        staging.rename(output)
        return {
            "prepared": True,
            "type": root["type"],
            "images": True,
            "output_sha256": _hash(output / "model.safetensors"),
        }
    finally:
        if staging.exists():
            shutil.rmtree(staging)
