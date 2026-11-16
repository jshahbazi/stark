"""Prepare and run the native Decider Flash v1.3 decision/image path.

The reference implementation is used only by external qualification tools.
Run ``python -m stark_convert.decider --help`` for the explicit CLI. Prepared
artifacts retain source identities; they do not claim numerical qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import selectors
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path

from tokenizers import Tokenizer

from ._build_metadata import file_hash as _hash
from .adapters.qwen35 import build_qwen35_plan
from .checkpoint_metadata import CheckpointMetadata
from .decider_protocol import PROTOCOL, DeciderError, Encoder, answer, probabilities
from .streaming import (
    ConversionError,
    TensorOperation,
    UnsupportedConversion,
    load_checkpoint,
    read_json,
    read_regular_bytes,
    read_safetensors_header,
    write_plan,
)
from .template_runtime import IsolatedTemplate

SOURCE_REPO = "perplexity-ai/pplx-decider-flash-v1.3"
SOURCE_REVISION = "d06fb3f2d97e0a52325a0d463c450cf0338f44fe"
_METADATA_FILES = (
    "config.json",
    "decision_config.json",
    "joint_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "LICENSE",
    "NOTICE",
)


def validate_decision_metadata(config, joint, tokenizer=None):
    """Reject causal/old-protocol checkpoints and preserve the trained option rows."""
    codes, token_ids = config.get("codes"), config.get("token_ids")
    temperature = config.get("temperature")
    if (
        config.get("format_version") != 2
        or config.get("base_model") != "Qwen/Qwen3.5-4B"
        or config.get("model_type") != "qwen3_5"
        or config.get("attention_mode") != "noncausal_full_attention"
        or config.get("pooling") != "last"
        or not isinstance(codes, list)
        or len(codes) != 255
        or any(not isinstance(code, str) or not code for code in codes)
        or len(set(codes)) != 255
        or not isinstance(token_ids, list)
        or len(token_ids) != 255
        or any(type(token_id) is not int or token_id < 0 for token_id in token_ids)
        or len(set(token_ids)) != 255
        or type(temperature) not in (int, float)
        or not math.isfinite(temperature)
        or temperature <= 0
        or config.get("max_pixels") != 1638400
        or joint != PROTOCOL
    ):
        raise UnsupportedConversion("Unsupported Decider checkpoint protocol or decision metadata")
    if tokenizer is not None and any(
        tokenizer.encode(code, add_special_tokens=False).ids != [token_id] or tokenizer.id_to_token(token_id) != code
        for code, token_id in zip(codes, token_ids, strict=True)
    ):
        raise ConversionError("Decider option token IDs disagree with the tokenizer")


def build_decider_plan(source):
    """Map the decision backbone through the existing BF16 Qwen3.5 operators.

    Source architecture and namespaces are adapted only in the planner's local
    view. TensorInfo retains the original file/offset identities for every byte.
    The emitted config has its own type and cannot enter the causal LM loader.
    """
    source = Path(source).resolve()
    config = read_json(source / "config.json")
    decision = read_json(source / "decision_config.json")
    joint = read_json(source / "joint_config.json")
    validate_decision_metadata(decision, joint)
    if (
        config.get("architectures") != ["Qwen3_5Model"]
        or config.get("model_type") != "qwen3_5"
        or not isinstance(config.get("text_config"), dict)
        or not isinstance(config.get("vision_config"), dict)
        or config.get("tie_word_embeddings") is not True
        or config.get("auto_map") is not None
    ):
        raise UnsupportedConversion("Decider requires the complete dense Flash vision backbone")
    vision = dict(config["vision_config"])
    if vision.pop("rope_parameters", None) != {"rope_theta": 10000.0, "rope_type": "axial"}:
        raise UnsupportedConversion("Decider vision rotary configuration is unsupported")
    tensors = load_checkpoint(source)
    if any(not key.startswith(("language_model.", "visual.")) for key in tensors):
        raise ConversionError("Decider backbone has unexpected tensor namespaces")
    # These aliases restore the official conditional-model namespaces expected
    # by the existing Qwen3.5 planner, without touching any source files.
    mapped = {"model." + key: value for key, value in tensors.items()}
    planned_config = {**config, "architectures": ["Qwen3_5ForConditionalGeneration"], "vision_config": vision}
    plan = build_qwen35_plan(
        source, include_vision=True, checkpoint_metadata=CheckpointMetadata(planned_config, mapped)
    )
    if plan.mtp_config is not None:
        raise UnsupportedConversion("Decider does not implement a generative MTP path")
    _, readout_tensors = read_safetensors_header(source / "readout.safetensors")
    readout = readout_tensors.get("weight")
    dim = plan.decoder_config["transformer_config"]["model_dim"]
    if set(readout_tensors) != {"weight"} or readout.dtype != "BF16" or readout.shape != (255, dim):
        raise ConversionError("Decider requires its complete BF16 255-option readout")
    decoder = plan.decoder_config
    position = config["text_config"].get("rope_parameters", {})
    sections = position.get("mrope_section")
    if (
        position.get("mrope_interleaved") is not True
        or not isinstance(sections, list)
        or len(sections) != 3
        or any(type(section) is not int or section < 0 for section in sections)
    ):
        raise UnsupportedConversion("Decider images require explicit interleaved M-RoPE sections")
    for layer in decoder["transformer_config"]["layer_configs"]:
        if layer["mixer_config"]["type"] == "AttentionConfig":
            layer["mixer_config"]["is_causal"] = False
            rope = layer["rope_config"]
            half_width = rope["head_dim"] // 2
            if sum(sections) != half_width or any(
                sections[axis] and (sections[axis] - 1) * 3 + axis >= half_width for axis in (1, 2)
            ):
                raise UnsupportedConversion("Decider M-RoPE sections disagree with the rotary width")
            # The ordinary vision writer performs this replacement after the
            # Qwen3.5 planner. This dedicated writer must preserve the same
            # image-position contract instead of leaving scalar text RoPE.
            layer["rope_config"] = {
                **rope,
                "type": "Qwen35MRoPEConfig",
                "mrope_section": list(sections),
                "interleaved": True,
            }
    plan = replace(
        plan,
        operations=(*plan.operations, TensorOperation("decision.readout.weights", (readout,), "BF16", (255, dim))),
        metadata={
            **plan.metadata,
            "decision.readout.spec": json.dumps({"type": "FullPrecisionSpec", "layout": "output_input"}),
        },
        input_bytes=plan.input_bytes + readout.nbytes,
        source_tensors=plan.source_tensors + 1,
        input_tensors=(*plan.input_tensors, readout),
    )
    return plan, config, decision, joint


def prepare(source, output, *, revision=SOURCE_REVISION):
    """Atomically prepare an image-enabled BF16 artifact from a local snapshot."""
    source, output = Path(source).resolve(), Path(output).absolute()
    if output.exists():
        raise ConversionError("Decider output already exists")
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ConversionError("Decider source revision must be immutable")
    plan, config, decision, joint = build_decider_plan(source)
    payloads = {name: read_regular_bytes(source / name) for name in _METADATA_FILES}
    tokenizer = Tokenizer.from_str(payloads["tokenizer.json"].decode("utf-8"))
    validate_decision_metadata(decision, joint, tokenizer)
    marker, separator = (tokenizer.token_to_id(joint[name]) for name in ("marker", "separator"))
    if marker is None or separator is None or marker == separator:
        raise ConversionError("Decider tokenizer is missing protocol markers")
    vocab_size = plan.decoder_config["vocab_size"]
    if any(not 0 <= token_id < vocab_size for token_id in tokenizer.get_vocab().values()):
        raise ConversionError("Decider tokenizer IDs exceed its embedding table")
    if any(
        type(config.get(field)) is not int
        or tokenizer.token_to_id(token) != config[field]
        or tokenizer.encode(token, add_special_tokens=False).ids != [config[field]]
        for field, token in (
            ("image_token_id", "<|image_pad|>"),
            ("vision_start_token_id", "<|vision_start|>"),
            ("vision_end_token_id", "<|vision_end|>"),
        )
    ):
        raise ConversionError("Decider image token IDs disagree with the tokenizer")
    token_config = read_json(source / "tokenizer_config.json")
    template = payloads["chat_template.jinja"].decode("utf-8")
    if token_config.get("chat_template") not in (None, template):
        raise ConversionError("Decider tokenizer templates disagree")
    processor = read_json(source / "processor_config.json")
    image_processor = processor.get("image_processor", {})
    expected_processor = {
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
        "size": {"longest_edge": 1638400, "shortest_edge": 65536},
        "temporal_patch_size": 2,
        "image_processor_type": "Qwen2VLImageProcessor",
    }
    if processor.get("processor_class") != "Qwen3VLProcessor" or image_processor != expected_processor:
        raise UnsupportedConversion("Decider image processor is outside the implemented profile")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".decider-", dir=output.parent))
    try:
        for name, payload in payloads.items():
            if name != "config.json":
                (staging / name).write_bytes(payload)
        stats = write_plan(plan, staging / "model.safetensors")
        provenance = {
            "repo": SOURCE_REPO,
            "revision": revision,
            "metadata_sha256": {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()},
            "weight_files": {
                str(path.relative_to(source)): {"sha256": _hash(path), "size": path.stat().st_size}
                for path in sorted({info.path for info in plan.input_tensors})
            },
        }
        root = {
            "type": "DeciderModelConfig",
            "decoder_config": plan.decoder_config,
            "vision_config": {**config["vision_config"], "layer_norm_eps": 1e-6},
            "decision_config": {
                "num_options": 255,
                "marker_token_id": marker,
                "separator_token_id": separator,
                "temperature": decision["temperature"],
                "codes": decision["codes"],
                "token_ids": decision["token_ids"],
                "protocol_version": 4,
                "max_pixels": decision["max_pixels"],
                "image_token_id": config["image_token_id"],
                "vision_start_token_id": config["vision_start_token_id"],
                "vision_end_token_id": config["vision_end_token_id"],
                "mrope_section": config["text_config"]["rope_parameters"]["mrope_section"],
            },
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
                    "source_mtp": "config declaration without predictor tensors; non-generative checkpoint",
                    "prefix_reuse_blocker": (
                        "Full-attention states depend on future question tokens; causal prefix reuse is invalid."
                    ),
                    "precision": ("BF16 weights and activations; F32 normalization, accumulation and recurrent state"),
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


class Decider:
    """One serialized native worker. Request content is never sent to diagnostics."""

    def __init__(self, model_path, worker_path, *, max_length=4096, timeout=300):
        self.model_path = Path(model_path).resolve()
        self.config = read_json(self.model_path / "config.json")
        if self.config.get("type") != "DeciderModelConfig":
            raise DeciderError("A prepared Decider artifact is required.")
        self.temperature = self.config["decision_config"]["temperature"]
        self.max_length, self.timeout = max_length, timeout
        self._lock = threading.Lock()
        self._buffer = bytearray()
        self._process = None
        tokenizer = Tokenizer.from_file(str(self.model_path / "tokenizer.json"))
        template = IsolatedTemplate((self.model_path / "chat_template.jinja").read_text())
        self.encoder = Encoder(tokenizer, template, self.config["decision_config"]["codes"])
        self._process = subprocess.Popen(
            [str(Path(worker_path).resolve()), "--model-path", str(self.model_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
            bufsize=0,
        )
        try:
            self.ready = self._read_reply()
            if not self.ready.get("ready") or self.ready.get("mode") != "decider":
                raise DeciderError("Native decision worker did not become ready.")
            self.max_length = min(max_length, self.ready["max_input_tokens"])
        except Exception:
            self.close()
            raise DeciderError("Native decision worker did not become ready.") from None

    def _read_reply(self):
        deadline = time.monotonic() + self.timeout
        process = self._process
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while True:
                if b"\n" in self._buffer:
                    line, _, self._buffer = self._buffer.partition(b"\n")
                    try:
                        result = json.loads(line)
                    except (ValueError, UnicodeError):
                        raise DeciderError("Invalid native decision reply.") from None
                    if not isinstance(result, dict):
                        raise DeciderError("Invalid native decision reply.")
                    return result
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise DeciderError("Native decision worker timed out.")
                chunk = os.read(process.stdout.fileno(), 65536)
                if not chunk:
                    raise DeciderError("Native decision worker stopped.")
                self._buffer.extend(chunk)
                if len(self._buffer) > 8 * 1024 * 1024:
                    raise DeciderError("Native decision reply exceeded its limit.")

    def _request(self, request):
        if self._process is None:
            raise DeciderError("Native decision worker is closed.")
        data = json.dumps(request, ensure_ascii=False, allow_nan=False).encode() + b"\n"
        if len(data) > 8 * 1024 * 1024:
            raise DeciderError("Decision request exceeded its limit.")
        try:
            # A FileIO write may be partial, and a stalled worker must not hold
            # the request lock indefinitely before the read deadline begins.
            deadline = time.monotonic() + self.timeout
            descriptor = self._process.stdin.fileno()
            os.set_blocking(descriptor, False)
            written = 0
            with selectors.DefaultSelector() as selector:
                selector.register(descriptor, selectors.EVENT_WRITE)
                while written < len(data):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not selector.select(remaining):
                        raise DeciderError("Native decision worker timed out.")
                    try:
                        written += os.write(descriptor, data[written:])
                    except BlockingIOError:
                        continue
            result = self._read_reply()
            if result.get("ok") is not True:
                raise DeciderError("Native decision request failed.")
            return result
        except Exception:
            self.close()
            raise DeciderError("Native decision request failed; the worker was cleared.") from None

    def decide(self, request, *, temperature=None):
        with self._lock:
            try:
                images = request.get("images", []) if isinstance(request, dict) else []
                if (
                    not isinstance(images, list)
                    or len(images) > 1
                    or any(
                        not isinstance(image, (str, Path))
                        or not Path(image).is_absolute()
                        or not Path(image).is_file()
                        for image in images
                    )
                ):
                    raise DeciderError("Images must contain at most one absolute local PNG or JPEG path.")
                plan = self.encoder.plan(request, self.max_length)
                selected_temperature = self.temperature if temperature is None else temperature
                # Validate the override before computing any question.
                probabilities([0.0], selected_temperature)
                answers, input_tokens = {}, 0
                for encoded in plan:
                    result = self._request(
                        {
                            "operation": "decide",
                            "input_token_ids": list(encoded.input_ids),
                            "readout_positions": [len(encoded.input_ids) - 1],
                            "option_counts": [encoded.option_count],
                            "images": [str(image) for image in images],
                            "context_tokens": self.max_length,
                        }
                    )
                    rows = result.get("logits")
                    if not isinstance(rows, list) or len(rows) != 1 or len(rows[0]) != encoded.option_count:
                        raise DeciderError("Invalid native decision result.")
                    answers[encoded.key] = answer(
                        request["questions"][encoded.key], probabilities(rows[0], selected_temperature)
                    )
                    input_tokens += result["input_tokens"]
                return {
                    "answers": answers,
                    "usage": {"input_tokens": input_tokens, "output_tokens": 0, "sequences": len(plan)},
                }
            except Exception:
                # Clear any transient model state on validation and inference
                # failures; no subsequent request inherits partial decisions.
                if self._process is not None:
                    try:
                        self._request({"operation": "reset_cache"})
                    except DeciderError:
                        pass
                raise

    def close(self):
        process, self._process = self._process, None
        if process is not None:
            process.kill()
            process.wait()
            process.stdin.close()
            process.stdout.close()
        self._buffer.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare or run native Decider Flash v1.3 with images.")
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--source", type=Path, required=True)
    preparation.add_argument("--output", type=Path, required=True)
    preparation.add_argument("--revision", default=SOURCE_REVISION)
    run = commands.add_parser("decide", help="Read requests and write responses as JSON lines.")
    run.add_argument("--model-path", type=Path, required=True)
    run.add_argument("--worker", type=Path, required=True)
    run.add_argument("--max-length", type=int, default=4096)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            print(json.dumps(prepare(args.source, args.output, revision=args.revision)))
        else:
            with Decider(args.model_path, args.worker, max_length=args.max_length) as decider:
                for line in sys.stdin:
                    try:
                        result = decider.decide(json.loads(line))
                    except Exception:
                        result = {"error": "decision_request_failed"}
                    print(json.dumps(result, ensure_ascii=False), flush=True)
    except Exception:
        print("Decider preparation or startup failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
