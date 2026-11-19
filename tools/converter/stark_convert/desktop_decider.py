"""Native Decider backend for the desktop's ordered-answer Decision workflow.

Swift sends opaque attachment references to DesktopAttachments. Only claimed
bytes enter this adapter; private worker paths never come from app payloads.
"""

from __future__ import annotations

import math
import os
import subprocess
import threading
from contextlib import nullcontext
from pathlib import Path

from tokenizers import Tokenizer

from .chat import ChatCancelled, ChatError, Worker, _popen_with_gpu_memory, read_model_config
from .decider_protocol import DeciderError, Encoder, answer, probabilities
from .elephant_protocol import Encoder as ElephantEncoder
from .elephant_protocol import answer as elephant_answer
from .decision import DecisionInputError, DecisionProtocolError, validate_inputs
from .media import PrivateMediaStore
from .resource_budget import REQUEST_BUDGET_MIB
from .template_runtime import IsolatedTemplate, TemplateCancelled

IMAGE_LIMITS = {
    "max_source_pixels": 32_000_000,
    "max_processed_pixels": 1_638_400,
    "max_patches": 6400,
    "max_image_tokens": 1600,
    "max_images": 1,
}


class _DecisionTemplate:
    def __init__(self, template, cancelled):
        self.template, self.cancelled = template, cancelled

    def render(self, **arguments):
        return self.template.render_with_cancellation(self.cancelled, **arguments)


def is_decider(config):
    return isinstance(config, dict) and config.get("type") == "DeciderModelConfig"

def _decision_protocol(config):
    return config["decision_config"].get("protocol_version", 4)


def prepared_decider(path, config=None):
    """Cheap structural import admission; the native worker still validates weights."""
    try:
        path = Path(path)
        config = read_model_config(path) if config is None else config
        if not is_decider(config) or not all(
            (path / name).is_file()
            for name in ("model.safetensors", "tokenizer.json", "chat_template.jinja", "config.json")
        ):
            return False
        decision = config["decision_config"]
        protocol = _decision_protocol(config)
        temperature = decision["temperature"]
        if (
            protocol not in {4, 5}
            or type(temperature) not in (int, float)
            or not math.isfinite(temperature)
            or temperature <= 0
            or not isinstance(config.get("vision_config"), dict)
        ):
            return False
        if protocol == 4:
            codes, tokens = decision["codes"], decision["token_ids"]
            if (
                decision.get("num_options") != 255
                or decision.get("max_pixels") != IMAGE_LIMITS["max_processed_pixels"]
                or not isinstance(codes, list)
                or len(codes) != 255
                or any(not isinstance(code, str) or not code for code in codes)
                or len(set(codes)) != 255
                or not isinstance(tokens, list)
                or len(tokens) != 255
                or any(type(token) is not int or token < 0 for token in tokens)
                or len(set(tokens)) != 255
            ):
                return False
        else:
            labels, tokens = decision["slot_labels"], decision["slot_token_ids"]
            if (
                decision.get("protocol") != "onejev-slot-lm-v2"
                or decision.get("readout") != "slot_lm"
                or decision.get("attention") != "block_bidirectional"
                or decision.get("num_options") != 256
                or decision.get("max_pixels") != 16_777_216
                or decision.get("max_input_tokens") != 16_384
                or not isinstance(labels, list)
                or len(labels) != 256
                or any(not isinstance(label, str) or not label for label in labels)
                or not isinstance(tokens, list)
                or len(tokens) != 256
                or any(type(token) is not int or token < 0 for token in tokens)
                or len(set(tokens)) != 256
            ):
                return False
        layers = config["decoder_config"]["transformer_config"]["layer_configs"]
        full = [layer for layer in layers if layer["mixer_config"]["type"] == "AttentionConfig"]
        return bool(full) and all(
            layer["mixer_config"].get("is_causal") is (protocol == 5)
            and layer["rope_config"].get("type") == "Qwen35MRoPEConfig"
            and layer["rope_config"].get("mrope_section") == [11, 11, 10]
            and layer["rope_config"].get("interleaved") is True
            for layer in full
        )
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError):
        return False


def validate_desktop_inputs(text, question, answers, *, image=False):
    # The generic app input limits stay the same. Empty state is meaningful
    # when the state consists of the attached image; no placeholder is sent.
    if image and isinstance(text, str) and not text.strip():
        return validate_inputs("image", question, answers)
    return validate_inputs(text, question, answers)


class DesktopDeciderWorker(Worker):
    backend = "decision-native"

    def __init__(self, executable, model, timeout, gpu_memory_callback=None):
        self.timeout = timeout
        self.pending = b""
        self._closed = False
        self._decision_lock = threading.RLock()
        self._decision_cancelled = threading.Event()
        self.model = Path(model)
        self.executable = Path(executable)
        config = read_model_config(self.model)
        if not prepared_decider(self.model, config):
            raise DecisionInputError("This Decider artifact is incomplete or lacks image position support.")
        decision = config["decision_config"]
        self.protocol = _decision_protocol(config)
        self.num_options = decision["num_options"]
        self.temperature = decision["temperature"]
        tokenizer = Tokenizer.from_file(str(self.model / "tokenizer.json"))
        template = IsolatedTemplate((self.model / "chat_template.jinja").read_text())
        template = _DecisionTemplate(template, self._decision_cancelled)
        if self.protocol == 4:
            if any(
                tokenizer.encode(code, add_special_tokens=False).ids != [token]
                for code, token in zip(decision["codes"], decision["token_ids"], strict=True)
            ):
                raise DecisionInputError("The Decider option mapping does not match its tokenizer.")
            self.encoder = Encoder(tokenizer, template, decision["codes"])
            self.answer = answer
        else:
            if any(
                tokenizer.encode(label, add_special_tokens=False).ids != [token]
                for label, token in zip(decision["slot_labels"], decision["slot_token_ids"], strict=True)
            ):
                raise DecisionInputError("The Elephant option mapping does not match its tokenizer.")
            self.encoder = ElephantEncoder(tokenizer, template)
            self.answer = elephant_answer
        self.images = False
        self.image_input = {
            **IMAGE_LIMITS,
            "max_processed_pixels": decision["max_pixels"],
            "max_patches": decision["max_pixels"] // 1024,
            "max_image_tokens": decision["max_pixels"] // 1024,
        }
        self.image_verification = "unverified"
        self.media = PrivateMediaStore()
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("STARK_CAPTURE_", "UZU_CAPTURE_"))
            and key not in {"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONINSPECT"}
        }
        environment["STARK_REQUEST_BUDGET_MIB"] = str(REQUEST_BUDGET_MIB)
        try:
            self.process, descriptor = _popen_with_gpu_memory(
                [str(executable), "--model-path", str(model)],
                callback=gpu_memory_callback,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
                start_new_session=True,
                env=environment,
            )
            self._start_gpu_memory_reader(descriptor, gpu_memory_callback)
        except BaseException:
            if hasattr(self, "process"):
                self.close()
            else:
                self.media.close()
            raise

    def read(self, *args, **kwargs):
        reply = super().read(*args, **kwargs)
        if reply.get("ready") is True:
            if (
                reply.get("mode") != "decider"
                or reply.get("images") is not True
                or type(reply.get("max_images")) is not int
                or reply["max_images"] != 1
                or reply.get("options") != self.num_options
                or type(reply.get("max_input_tokens")) is not int
                or reply["max_input_tokens"] != (
                    4096 if self.protocol == 4 else 16384
                )
            ):
                self.close()
                raise ChatError("The native Decider worker did not advertise its required image capabilities.")
            self.max_input_tokens = reply["max_input_tokens"]
            self.images = True
            self.image_input = {
                **IMAGE_LIMITS,
                "max_processed_pixels": read_model_config(self.model)["decision_config"]["max_pixels"],
                "max_patches": read_model_config(self.model)["decision_config"]["max_pixels"] // 1024,
                "max_image_tokens": read_model_config(self.model)["decision_config"]["max_pixels"] // 1024,
            }
            self.image_verification = "unverified"
        return reply

    def reset_decision(self):
        with self._decision_lock:
            reply = self.request({"operation": "reset_cache"})
            if reply.get("ok") is not True or reply.get("cache_cleared") is not True:
                raise DecisionProtocolError

    def close(self):
        self._decision_cancelled.set()
        super().close()

    def cancel(self, request_id):
        # Decider has no native cancellation pipe. Reaping the worker fences
        # its unfinished reply and releases every request's transient state.
        self.close()
        raise ChatCancelled("Request cancelled; the decision worker was closed.")

    def decide(self, text, question, answers, image=None):
        labels = validate_desktop_inputs(text, question, answers, image=image is not None)
        result = self.decide_request(
            {
                "state": text,
                "questions": {
                    "decision": {"type": "choice", "instructions": question, "criteria": dict.fromkeys(labels)}
                },
            },
            image=image,
        )
        values = list(result["answers"]["decision"]["probabilities"].values())
        return {
            "choice_index": max(range(len(labels)), key=values.__getitem__),
            "answers": labels,
            "probabilities": values,
            "input_tokens": result["usage"]["input_tokens"],
            "calibration": "checkpoint_temperature",
            "temperature": self.temperature,
            "backend": self.backend,
        }

    def decide_request(self, request, *, image=None, cancelled=None):
        """Run all separate-layout questions as one serialized worker transaction."""
        # One complete decision, including image staging and encoding, owns the
        # worker until its final reply. Stop may still kill the process directly.
        with self._decision_lock:
            if cancelled is not None and cancelled.is_set():
                raise ChatCancelled("Request cancelled.")
            request = dict(request)
            if image is not None:
                if not self.images:
                    raise DecisionInputError("Image decisions are unavailable for this worker.")
                request["images"] = ["private attachment"]  # Encoder uses only the image count.
            self.encoder.template.cancelled = cancelled if cancelled is not None else self._decision_cancelled
            try:
                try:
                    plan = self.encoder.plan(request, self.max_input_tokens)
                except TemplateCancelled:
                    raise ChatCancelled("Request cancelled.") from None
                except DeciderError:
                    raise DecisionInputError(
                        "The decision is invalid or exceeds its input limits. Use shorter text or fewer answers."
                    ) from None
            finally:
                self.encoder.template.cancelled = self._decision_cancelled
            lease = self.media.stage(image) if image is not None else nullcontext(None)
            with lease as reference:
                answers, input_tokens = {}, 0
                for encoded in plan:
                    values, count = self._decide_question(encoded, reference, image is not None, cancelled)
                    answers[encoded.key] = self.answer(request["questions"][encoded.key], values)
                    input_tokens += count
            return {
                "answers": answers,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0, "sequences": len(plan)},
            }

    def _decide_question(self, encoded, reference, has_image, cancelled):
        reply = self.request(
            {
                "operation": "decide",
                "input_token_ids": list(encoded.input_ids),
                "readout_positions": [len(encoded.input_ids) - 1],
                "option_counts": [encoded.option_count],
                "block_boundaries": (
                    [encoded.block_boundary] if self.protocol == 5 else []
                ),
                "context_tokens": self.max_input_tokens,
                "images": [str(self.media.root / reference["id"])] if reference is not None else [],
            },
            cancelled=cancelled,
        )
        if reply.get("ok") is not True:
            if reply.get("error") in {"invalid_decision_request", "decider_context_overflow"}:
                raise DecisionInputError(
                    f"The decision and image exceed the {self.max_input_tokens}-token limit. Use a smaller input."
                )
            if reply.get("error") == "decider_image_preparation_failed":
                raise DecisionInputError("The image could not be prepared. Choose a supported PNG or JPEG image.")
            if reply.get("error") == "decider_resource_limit":
                raise DecisionInputError("The decision exceeds its context or memory limit. Use a smaller input.")
            raise DecisionProtocolError
        rows, count = reply.get("logits"), reply.get("input_tokens")
        if (
            not isinstance(rows, list)
            or len(rows) != 1
            or not isinstance(rows[0], list)
            or len(rows[0]) != encoded.option_count
            or type(count) is not int
            or not len(encoded.input_ids) <= count <= self.max_input_tokens
            or not has_image
            and count != len(encoded.input_ids)
            or reply.get("cache_hit") is not False
        ):
            raise DecisionProtocolError
        try:
            values = probabilities(rows[0], self.temperature)
        except DeciderError:
            raise DecisionProtocolError from None
        return values, count
