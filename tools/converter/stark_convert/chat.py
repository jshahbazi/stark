"""Terminal chat client. Conversation content stays in memory and private pipes."""

import argparse
import hashlib
import json
import os
import re
import secrets
import select
import signal
import stat
import subprocess
import sys
import threading
import time
from contextlib import ExitStack
from pathlib import Path

from jinja2.exceptions import TemplateError

from . import inkling_prepare
from ._build_metadata import ConversionError, file_hash, open_regular
from .api_tools import tool_parser_for
from .generation_eta import progress_frame
from .inkling_chat import PARSER as INKLING_PARSER
from .inkling_chat import parse_response as parse_inkling_response
from .media import MAX_IMAGE_BYTES, ImageData, MediaError, PrivateMediaStore
from .mtp import default_mtp_enabled, mtp_capabilities, resolve_mtp
from .reasoning import assistant_message, prompt_opens_reasoning, reasoning_options, reasoning_tags
from .resource_budget import REQUEST_BUDGET_MIB, WIRE_BYTES
from .template_runtime import (
    SYSTEM_ROLE_FOLD,
    SYSTEM_ROLE_POLICY,
    IsolatedTemplate,
    SystemRoleError,
    TemplateCancelled,
    TemplateHelperMissing,
    TemplateRuntimeError,
    TemplateSizeLimit,
)
from .tool_protocols import runtime_template_source
from .vision_runtime import (
    CONFIGURED_FAMILY_PROFILE,
    HY4_STQ_VISION_PROFILE,
    HY4_VISION_PROFILE,
    LILY_FAMILY_PROFILE,
    VISION_MODEL_TYPES,
    family_image_input_contract,
    image_input_contract,
    model_image_family,
    qualified_image_worker,
    supported_image_family,
)

_MAX_GPU_MEMORY_LINE = 128
_MAX_GPU_MEMORY_BYTES = (1 << 63) - 1
_LEGACY_MAX_REQUEST_BYTES = 1024 * 1024
# Inkling needs room for the channel transition and at least one answer token.
MIN_THINKING_BUDGET_OUTPUT_TOKENS = 4


def _popen_with_gpu_memory(command, *, callback, **kwargs):
    """Launch a worker with an optional private telemetry pipe."""
    if callback is None:
        environment = kwargs.get("env")
        if environment is None:
            environment = os.environ
        if "STARK_GPU_MEMORY_FD" in environment:
            environment = dict(environment)
            environment.pop("STARK_GPU_MEMORY_FD", None)
            kwargs["env"] = environment
        return subprocess.Popen(command, **kwargs), None
    read_fd, write_fd = os.pipe()
    try:
        os.set_blocking(read_fd, False)
        os.set_blocking(write_fd, False)
        environment = dict(kwargs.get("env") or os.environ)
        environment["STARK_GPU_MEMORY_FD"] = str(write_fd)
        kwargs["env"] = environment
        kwargs["pass_fds"] = (*kwargs.get("pass_fds", ()), write_fd)
        process = subprocess.Popen(command, **kwargs)
    except BaseException:
        os.close(read_fd)
        raise
    finally:
        os.close(write_fd)
    return process, read_fd


def _parse_gpu_memory_line(line):
    if len(line) > _MAX_GPU_MEMORY_LINE:
        return None
    try:
        value = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    amount = value.get("gpu_memory_bytes")
    if type(amount) is not int or not 0 <= amount <= _MAX_GPU_MEMORY_BYTES:
        return None
    return amount


class ChatError(Exception):
    """Content-free user-facing failure."""


class ChatCancelled(ChatError):
    """A cancelled request whose terminal response has been consumed."""


class ChatRequestTooLarge(ChatError):
    """A request rejected for exceeding the worker's resource budget."""


class ChatBudgetUnavailable(ChatError):
    """An older worker cannot yet plan automatic memory-bounded execution."""


def automatic_generation_budget(
    worker,
    input_tokens,
    context_tokens,
    *,
    context_capacity_known=False,
    cancelled=None,
    input_token_ids=None,
    cache_scope=None,
    mtp_drafts=None,
    sampling=None,
):
    """Ask the loaded backend to admit execution after tokenization."""
    if not getattr(worker, "generation_budget", False):
        raise ChatBudgetUnavailable("Update Stark's worker to enable automatic output for this model.")
    request = {"operation": "plan_generation", "input_tokens": input_tokens, "context_tokens": context_tokens}
    if getattr(worker, "generation_budget_inputs", False) is True:
        if input_token_ids is not None:
            request["input_token_ids"] = input_token_ids
        if cache_scope is not None:
            request["cache_scope"] = cache_scope
        if mtp_drafts is not None:
            request["mtp_drafts"] = mtp_drafts
        request["sampling"] = sampling or {"mode": "greedy"}
    reply = worker.request(
        request,
        **({"cancelled": cancelled} if cancelled is not None else {}),
    )
    if reply.get("error") == "context_planning_unavailable":
        raise ChatBudgetUnavailable("This worker cannot plan automatic execution. Rebuild or reinstall Stark.")
    context, maximum = reply.get("context_tokens"), reply.get("max_new_tokens")
    if (
        type(context) is not int
        or not input_tokens < context < 0xFFFFFFFF
        or context > context_tokens
        or type(maximum) is not int
        or maximum != context - input_tokens
        or maximum > getattr(worker, "max_new_tokens", 0xFFFFFFFE)
        or reply.get("context_capacity_known") is not context_capacity_known
        or reply.get("budget_kind") != "memory"
    ):
        raise ChatError("Invalid automatic execution budget; restart the model.")
    return context, maximum


class ChatImageError(ChatError):
    """A recoverable, content-free image request rejection."""

    def __init__(self, code):
        self.code = code
        message = {
            "invalid_image_request": "This image could not be decoded or exceeds the model's image limits.",
            "context_overflow": (
                "The image and conversation exceed the context budget. Start a new chat or reduce the output limit."
            ),
            "thinking_budget_output_too_small": "Leave at least four output tokens when Thinking Budget is enabled.",
            "invalid_image_reference": "The image attachment is no longer available. Remove it and attach it again.",
            "image_read_failed": "The image attachment could not be read. Remove it and attach it again.",
        }.get(code, "The image request could not be prepared or executed.")
        super().__init__(message)


_SYSTEM_ROLE_MESSAGES = {
    "system_message_repeated": "This model's chat template has no system role; send at most one system message.",
    "system_message_not_first": "This model's chat template has no system role; the system message must come first.",
    "system_content_not_text": "This model's chat template has no system role; the system message must be plain text.",
    "system_message_without_user": (
        "This model's chat template has no system role; a user message must follow the system message."
    ),
    "user_content_not_text": (
        "This model's chat template has no system role; the first user message must be plain text."
    ),
}


class _ChatTemplate(IsolatedTemplate):
    def __init__(self, source, *, system_role=None):
        super().__init__(runtime_template_source(source), system_role=system_role)

    def render_with_cancellation(self, cancelled, **arguments):
        try:
            return super().render_with_cancellation(cancelled, **arguments)
        except TemplateCancelled:
            raise ChatCancelled("Request cancelled.") from None
        except TemplateSizeLimit:
            raise ChatRequestTooLarge("Conversation exceeds the template request limit; use /reset.") from None
        except SystemRoleError as error:
            raise ChatError(_SYSTEM_ROLE_MESSAGES[error.reason]) from None
        except TemplateRuntimeError:
            raise ChatError("The model's chat template is unsupported or exceeded its resource limits.") from None


def render_template(template, *, cancelled=None, **arguments):
    if isinstance(template, IsolatedTemplate):
        return template.render_with_cancellation(cancelled, **arguments)
    return template.render(**arguments)


_RECOVERABLE_IMAGE_ERRORS = {
    "invalid_request",
    "request_too_large",
    "resource_budget_exceeded",
    "images_not_supported",
    "invalid_image_reference",
    "image_read_failed",
    "invalid_image_request",
    "invalid_prepared_image",
    "image_request_setup_failed",
    "invalid_generation_request",
    "context_overflow",
    "invalid_sampling_request",
    "generation_cancelled",
    "tokenization_failed",
}
_FATAL_WORKER_ERRORS = {
    "native_generation_failed",
    "invalid_generation_output",
    "invalid_tokenization_output",
    "decoding_failed",
    "cancel_channel_failed",
    "invalid_image_preparation_output",
}


def terminal_text(text):
    # Model output must not execute terminal control sequences.
    return "".join(c for c in text if c in "\n\t" or (c.isprintable() and c != "\x1b"))


def _chat_file(path, maximum):
    with open_regular(path.resolve(strict=True)) as stream:
        before = os.fstat(stream.fileno())
        if not 0 < before.st_size <= maximum:
            raise ValueError
        data = stream.read(before.st_size + 1)
        after = os.fstat(stream.fileno())
        if len(data) != before.st_size or any(
            getattr(before, key) != getattr(after, key)
            for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        ):
            raise ValueError
        return data


def read_model_config(model):
    value = json.loads(_chat_file(model / "config.json", 2 * 1024 * 1024))
    if not isinstance(value, dict):
        raise ValueError
    return value


def load_template(model, *, expected_hashes=None):
    try:
        data = _chat_file(model / "config.json", 2 * 1024 * 1024)
        config = json.loads(data)
        image_family = model_image_family(config, model)
        image_profile = (image_family or {}).get("profile")
        hy4 = config.get("model_type") == "hy_v4_flash" and image_profile in (
            HY4_VISION_PROFILE,
            HY4_STQ_VISION_PROFILE,
        )
        if hy4:
            from .vision_runtime import _HY4_METADATA_HASHES, _HY4_STQ_METADATA_HASHES

            pinned = _HY4_STQ_METADATA_HASHES if image_profile == HY4_STQ_VISION_PROFILE else _HY4_METADATA_HASHES
            metadata = {"config_sha256": hashlib.sha256(data).hexdigest()}
            tokenizer_bytes = _chat_file(model / "tokenizer.json", 64 * 1024 * 1024)
            metadata["tokenizer_sha256"] = hashlib.sha256(tokenizer_bytes).hexdigest()
            metadata["processor_sha256"] = hashlib.sha256(
                _chat_file(model / "processor_config.json", 64 * 1024)
            ).hexdigest()
            template_bytes = _chat_file(model / "chat_template.jinja", 256 * 1024)
            metadata["template_sha256"] = hashlib.sha256(template_bytes).hexdigest()
            if metadata != pinned or (
                expected_hashes is not None
                and any(expected_hashes.get(key) != value for key, value in metadata.items())
            ):
                raise ValueError
            source = template_bytes.decode("utf-8")
            from tokenizers import Tokenizer

            tokenizer = Tokenizer.from_str(tokenizer_bytes.decode("utf-8"))
            text_config = config.get("text_config", config)

            def special_token(name):
                value = text_config.get(name)
                if isinstance(value, list):
                    value = value[0] if value else None
                if type(value) is not int or not 0 <= value <= 1_000_000:
                    return None
                token = tokenizer.decode([value], skip_special_tokens=False)
                return token if isinstance(token, str) and 0 < len(token.encode("utf-8")) <= 8192 else None

            codec = {
                "type": "ChatCodecConfig",
                "prompt_template": source,
            }
            for field, key in (("bos_token", "bos_token_id"), ("eos_token", "eos_token_id")):
                token = special_token(key)
                if token is not None:
                    codec[field] = token
        else:
            if expected_hashes is not None and hashlib.sha256(data).hexdigest() != expected_hashes["config_sha256"]:
                raise ValueError
            codec = config["token_codec_config"]
            source = codec["prompt_template"]
            if codec.get("type") != "ChatCodecConfig" or not isinstance(source, str) or not source:
                raise ValueError
        if config.get("type") in VISION_MODEL_TYPES or image_family is not None:
            # Only the recognized program may reach Jinja compilation. An
            # immutable sandbox does not bound arbitrary template computation.
            template_hashes = {
                "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
                "7f0e529032c25183bcd66c7f238da2d377f43be754a94e2725a58c4e16d2ed67",
            }
            from .vision_family import _OFFICIAL_QWEN38_TEMPLATE_SHA256

            if config.get("type") == "Gemma4VisionLanguageModelConfig":
                descriptor = config.get("artifact_contract", {}).get("family", {})
                template_hashes = (
                    {"ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4"}
                    if descriptor
                    in (
                        {"id": profile, "version": 1}
                        for profile in ("gemma4-31b-bf16-v1", "gemma4-12b-bf16-v1", "gemma4-26b-a4b-bf16-v1")
                    )
                    else {"0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5"}
                )
            elif config.get("type") == "Gemma3VisionLanguageModelConfig":
                template_hashes = {"7de1c58e208eda46e9c7f86397df37ec49883aeece39fb961e0a6b24088dd3c4"}
            elif image_family == {"profile": LILY_FAMILY_PROFILE, "version": 1}:
                from . import qwen38_retained

                template_hashes = qwen38_retained.SUPPORTED_TEMPLATE_HASHES
            elif image_family == {"profile": inkling_prepare.VISION_PROFILE, "version": 1}:
                template_hashes = {inkling_prepare.TEMPLATE_SHA}
            elif hy4:
                template_hashes = {pinned["template_sha256"]}
            contract = config.get("artifact_contract")
            family = contract.get("family") if isinstance(contract, dict) else None
            if (
                config.get("type") == "Qwen35VisionLanguageModelConfig"
                and supported_image_family(family)
                and family["profile"]
                in {
                    CONFIGURED_FAMILY_PROFILE,
                    "qwen35-dense-affine-merged-v3",
                    "qwen35-moe-affine-merged-v4",
                    "qwen35-moe-bf16-merged-v5",
                }
            ):
                template_hashes.add("a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715")
                template_hashes.add(_OFFICIAL_QWEN38_TEMPLATE_SHA256)
                if family["profile"] in {
                    "qwen35-dense-affine-merged-v3",
                    "qwen35-moe-affine-merged-v4",
                    "qwen35-moe-bf16-merged-v5",
                }:
                    template_hashes.add("c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041")
            copied = _chat_file(model / "chat_template.jinja", 256 * 1024)
            copied_digest = hashlib.sha256(copied).hexdigest()
            copied_matches_source = copied == source.encode("utf-8")
            if copied_digest == _OFFICIAL_QWEN38_TEMPLATE_SHA256:
                from .template_compat import normalize_chat_template

                copied_matches_source = normalize_chat_template(copied.decode("utf-8")) == source
            if (
                len(data) > 1024 * 1024
                or len(copied) > 256 * 1024
                or not copied_matches_source
                or copied_digest not in template_hashes
                or (expected_hashes is not None and copied_digest != expected_hashes["template_sha256"])
            ):
                raise ValueError
        from .api_tools import codec_with_template_review
        from .model_inventory import template_review_receipt

        review = template_review_receipt(model)
        # The converter recorded that this program refuses the system role.
        # Only the exact policy is honoured; any other recorded value fails.
        policy = review.get("chat_template_system_role")
        if policy is not None and policy != SYSTEM_ROLE_POLICY:
            raise ValueError
        runtime_source = _hy4_runtime_template(source) if hy4 else source
        template = _ChatTemplate(runtime_source, system_role=SYSTEM_ROLE_FOLD if policy is not None else None)
        # A template selected from a named list keeps tool calling unavailable.
        return codec_with_template_review(codec, review), template
    except TemplateHelperMissing:
        raise ChatError("Native chat-template helper is missing; rebuild or reinstall Stark.") from None
    except (OSError, ValueError, KeyError, TypeError, UnicodeError, ConversionError, TemplateError, SyntaxError):
        raise ChatError("Cannot read chat metadata; use a converted language model with a chat template.") from None


_HY4_FORMAT_TOKEN = re.compile(r"'([^'\\]*)\{\}([^'\\]*)'\.format\(HYTK\)")


def _hy4_runtime_template(source):
    """Normalize the pinned Hy4 template's bounded Python string formatting for MiniJinja."""
    from .vision_runtime import _HY4_METADATA_HASHES, _HY4_STQ_METADATA_HASHES

    if hashlib.sha256(source.encode("utf-8")).hexdigest() not in {
        _HY4_METADATA_HASHES["template_sha256"],
        _HY4_STQ_METADATA_HASHES["template_sha256"],
    }:
        raise ValueError("unrecognized Hy4 chat template")
    normalized, replacements = _HY4_FORMAT_TOKEN.subn(r"('\1' ~ HYTK ~ '\2')", source)
    if replacements != 22:
        raise ValueError("unrecognized Hy4 chat template formatting")
    return normalized


class Worker:
    def _init_protocol(self, timeout):
        self.timeout = timeout
        self.pending = b""
        self.cancellation = False
        self.mtp = mtp_capabilities(None)
        self.request_sequence = 0
        self._transaction_lock = threading.RLock()
        self.cancelled_id = None
        self.image_input = None
        self.images = False
        self.image_verification = "unverified"
        self.image_check_cached = False
        self.image_check_dir = None
        self.media = None
        self.prefix_cache = False
        self.prefix_chunk_tokens = None
        self.prefix_subchunk_tokens = None
        self.generation_budget_inputs = False
        self._ready_seen = False

    def __init__(self, executable, model, timeout, gpu_memory_callback=None):
        self._init_protocol(timeout)
        self.model_path = Path(model)
        self._closed = False
        self.hy4_lanes = ()
        self._pending_hy4_lanes = []
        self._hy4_lane_lock = threading.Lock()
        self.executable = Path(executable)
        try:
            # Bind later image qualification to the exact executable observed
            # before launch, so a concurrent replacement cannot qualify the
            # different file now present at this path.
            self._launch_worker_sha256 = file_hash(self.executable)
        except (OSError, ValueError, TypeError, ConversionError):
            self._launch_worker_sha256 = None
        command = [str(executable), "--model-path", str(model)]
        try:
            config = read_model_config(Path(model))
        except (OSError, ValueError, ConversionError):
            config = {}
        converted_vision = isinstance(config, dict) and config.get("type") in VISION_MODEL_TYPES
        family = model_image_family(config, Path(model)) if isinstance(config, dict) and not converted_vision else None
        if converted_vision or family is not None:
            from .image_checks import ImageCheckError, preflight_family_memory

            try:
                preflight_family_memory(Path(model), config)
            except ImageCheckError as error:
                raise ChatError(str(error)) from None
            self.media = PrivateMediaStore()
            command.extend(["--vision-staging-root", str(self.media.root)])
        # Readiness enables reuse only when the loaded decoder can restore an
        # exact text prefix, including text requests on image-capable models.
        command.append("--prefix-cache")
        read_fd, self.cancel_fd = os.pipe()
        os.set_blocking(self.cancel_fd, False)
        telemetry_fd = None
        try:
            with ExitStack() as child_pipes:
                env = dict(
                    os.environ,
                    STARK_CANCEL_FD=str(read_fd),
                    STARK_REQUEST_BUDGET_MIB=str(REQUEST_BUDGET_MIB),
                )
                env.pop("STARK_HY4_LANE_FDS", None)
                inherited = (read_fd,)
                slots = env.get("STARK_HY4_CONCURRENT_SLOTS", "4")
                lane_fds = ()
                if (family or {}).get("profile") == HY4_STQ_VISION_PROFILE and slots in {"2", "3", "4"}:
                    for _ in range(int(slots) - 1):
                        lane, descriptors = _Hy4Lane.open(timeout, child_pipes)
                        self._pending_hy4_lanes.append(lane)
                        lane_fds += descriptors
                if lane_fds:
                    inherited += lane_fds
                    env["STARK_HY4_LANE_FDS"] = ",".join(map(str, lane_fds))
                self.process, telemetry_fd = _popen_with_gpu_memory(
                    command,
                    callback=gpu_memory_callback,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    bufsize=0,
                    start_new_session=True,
                    pass_fds=inherited,
                    env=env,
                )
        except BaseException:
            for lane in self._pending_hy4_lanes:
                lane.close()
            os.close(self.cancel_fd)
            if self.media is not None:
                self.media.close()
            raise
        finally:
            os.close(read_fd)
        try:
            self._start_gpu_memory_reader(telemetry_fd, gpu_memory_callback)
        except BaseException:
            self.close()
            raise

    def _start_gpu_memory_reader(self, descriptor, callback):
        self._gpu_memory_fd = descriptor
        self._gpu_memory_stop = threading.Event() if descriptor is not None else None
        self._gpu_memory_thread = None
        if descriptor is None:
            return

        def read_memory():
            pending = bytearray()
            discarding = False
            unexpected_eof = False
            try:
                while not self._gpu_memory_stop.is_set():
                    if self._gpu_memory_stop.is_set():
                        break
                    try:
                        readable = select.select([descriptor], [], [], 0.2)[0]
                        chunk = os.read(descriptor, 1024) if readable else None
                    except (OSError, ValueError):
                        break
                    if chunk is None:
                        continue
                    if not chunk:
                        unexpected_eof = True
                        break
                    for byte in chunk:
                        if discarding:
                            if byte == 10:
                                discarding = False
                            continue
                        if byte == 10:
                            amount = _parse_gpu_memory_line(bytes(pending))
                            pending.clear()
                            if amount is not None:
                                try:
                                    callback(self, amount)
                                except Exception:
                                    pass
                        elif len(pending) < _MAX_GPU_MEMORY_LINE:
                            pending.append(byte)
                        else:
                            pending.clear()
                            discarding = True
            finally:
                self._gpu_memory_fd = None
                if unexpected_eof and not self._gpu_memory_stop.is_set() and not getattr(self, "_closed", False):
                    try:
                        callback(self, None)
                    except Exception:
                        pass
                try:
                    os.close(descriptor)
                except OSError:
                    pass

        self._gpu_memory_thread = threading.Thread(target=read_memory, name="stark-gpu-memory-reader", daemon=True)
        try:
            self._gpu_memory_thread.start()
        except BaseException:
            self._gpu_memory_stop.set()
            descriptor, self._gpu_memory_fd = self._gpu_memory_fd, None
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            self._gpu_memory_thread = None
            self._gpu_memory_stop = None
            raise

    def _close_gpu_memory_reader(self):
        stop = getattr(self, "_gpu_memory_stop", None)
        if stop is None:
            return
        stop.set()
        thread = self._gpu_memory_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=0.5)

    def cancel(self, request_id):
        if self.cancelled_id == request_id:
            return
        if not self.cancellation:
            raise ChatError("Worker cannot cancel safely; restart the model.")
        try:
            os.write(self.cancel_fd, request_id.to_bytes(8, "little"))
            self.cancelled_id = request_id
        except OSError:
            raise ChatError("Cannot cancel worker; restart the model.") from None

    def read(self, cancelled=None, request_id=None):
        try:
            reply = self._read_reply(cancelled, request_id)
            if reply.get("ready") is False:
                self.mtp = mtp_capabilities(None)
            return reply
        except BaseException:
            self.mtp = mtp_capabilities(None)
            raise

    def _read_reply(self, cancelled=None, request_id=None):
        deadline = getattr(self, "_image_deadline", None)
        if deadline is None:
            deadline = time.monotonic() + self.timeout
        while b"\n" not in self.pending:
            remaining = deadline - time.monotonic()
            if cancelled is not None and cancelled.is_set() and request_id is not None:
                self.cancel(request_id)
            if remaining <= 0:
                raise ChatError("Worker timed out; chat closed.")
            output = self._output_pipe
            if not select.select([output], [], [], min(remaining, 0.1) if cancelled is not None else remaining)[0]:
                continue
            chunk = os.read(output.fileno(), 65536)
            if not chunk:
                raise ChatError("Worker stopped; chat closed.")
            self.pending += chunk
            if len(self.pending) > WIRE_BYTES:
                raise ChatError("Worker reply exceeded the size limit; chat closed.")
        if getattr(self, "_image_deadline", None) is not None and time.monotonic() >= deadline:
            raise ChatError("Worker timed out; chat closed.")
        line, self.pending = self.pending.split(b"\n", 1)
        try:
            reply = json.loads(line)
            if not isinstance(reply, dict):
                raise ValueError
            if reply.get("ready") is True:
                if getattr(self, "_ready_seen", False):
                    raise ChatError("Unexpected worker readiness reply; restart the model.")
                self._ready_seen = True
                advertised_limit = reply.get("max_request_bytes")
                if advertised_limit is not None and (type(advertised_limit) is not int or advertised_limit < 1):
                    raise ChatError("Invalid worker readiness reply; restart the model.")
                self.max_request_bytes = (
                    min(WIRE_BYTES, advertised_limit) if advertised_limit is not None else _LEGACY_MAX_REQUEST_BYTES
                )
                maximum_output = reply.get("max_new_tokens")
                if type(maximum_output) is int and maximum_output > 0:
                    self.max_new_tokens = maximum_output
                self.generation_budget = reply.get("generation_budget") == "memory-context-v1"
                self.generation_budget_inputs = reply.get("generation_budget_inputs") == "token-ids-v1"
                self.mtp = mtp_capabilities(reply)
                self.prefix_cache = prefix_cache_enabled(reply)
                prefix_settings = reply["prefix_cache"] if self.prefix_cache else {}
                self.prefix_chunk_tokens = prefix_settings.get("chunk_tokens")
                self.prefix_subchunk_tokens = prefix_settings.get("subchunk_tokens")
                self.cancellation = reply.get("cancellation") == "pipe-v1"
                self.image_input = image_input_contract(reply.get("image_input"))
                self._read_hy4_lanes_ready(reply)
                launch_hash = getattr(self, "_launch_worker_sha256", None)
                self.images = (
                    getattr(self, "media", None) is not None
                    and self.image_input is not None
                    and isinstance(launch_hash, str)
                    and qualified_image_worker(
                        self.image_input,
                        self.executable,
                        expected_worker_sha256=launch_hash,
                    )
                )
                self.image_verification = "reference_verified" if self.images else "unverified"
                if (
                    not self.images
                    and getattr(self, "media", None) is not None
                    and family_image_input_contract(self.image_input)
                ):
                    from .image_checks import ImageCheckError, verify_images

                    try:
                        progress = getattr(self, "image_check_progress", None)
                        if progress is not None:
                            progress()
                        self.image_check_cached = verify_images(self, reply, directory=self.image_check_dir)
                        self.images = True
                        self.image_verification = "locally_tested"
                    except BaseException as error:
                        try:
                            self.invalidate_image_checks()
                        except BaseException:
                            pass
                        try:
                            self.close()
                        except BaseException:
                            pass
                        if isinstance(error, ImageCheckError):
                            raise ChatError(str(error)) from None
                        raise ChatError(
                            "The model did not pass local image checks. Check its compatibility and memory."
                        ) from None
            return reply
        except (ValueError, UnicodeError):
            raise ChatError("Invalid worker reply; chat closed.") from None

    @property
    def _input_pipe(self):
        return self.process.stdin

    @property
    def _output_pipe(self):
        return self.process.stdout

    def _read_hy4_lanes_ready(self, primary):
        pending = tuple(getattr(self, "_pending_hy4_lanes", ()))
        if not pending:
            return
        capacities = primary.get("hy4_lane_context_tokens", [])
        if "hy4_lane_context_tokens" not in primary and len(pending) == 1:
            legacy_capacity = primary.get("hy4_secondary_context_tokens")
            if legacy_capacity is not None:
                capacities = [legacy_capacity]
        if (
            not isinstance(capacities, list)
            or len(capacities) > len(pending)
            or any(type(capacity) is not int or not 2 <= capacity < 0xFFFFFFFF for capacity in capacities)
        ):
            raise ChatError("Invalid Hy4 auxiliary worker capacity; restart the model.")
        for lane, capacity in zip(pending, capacities, strict=False):
            self._read_hy4_lane_ready(primary, lane, capacity)
        with self._hy4_lane_lock:
            if self._closed:
                raise ChatError("Worker stopped; chat closed.")
            for lane in pending[len(capacities) :]:
                lane.close()
            self.hy4_lanes = pending[: len(capacities)]
            self._pending_hy4_lanes = []

    def _read_hy4_lane_ready(self, primary, lane, capacity):
        ready = lane.read()
        if (
            ready.get("ready") is not True
            or ready.get("mode") != "text-generation-v1"
            or ready.get("max_context_tokens") != capacity
            or type(ready.get("max_new_tokens")) is not int
            or ready["max_new_tokens"] < 1
            or ready.get("vocab_size") != primary.get("vocab_size")
            or lane.mtp != self.mtp
            or not lane.cancellation
            or lane.prefix_cache
            or lane.image_input is not None
        ):
            raise ChatError("Invalid Hy4 auxiliary worker readiness; restart the model.")
        lane.context_limit = capacity
        lane.streaming = ready.get("streaming") == "text-delta-v1"
        lane.stop_sequences = ready.get("stop_sequences") is True
        if lane.streaming != (primary.get("streaming") == "text-delta-v1") or lane.stop_sequences != (
            primary.get("stop_sequences") is True
        ):
            raise ChatError("Invalid Hy4 auxiliary worker features; restart the model.")

    def request(self, payload, on_delta=None, cancelled=None, *, on_progress=None):
        lock = getattr(self, "_transaction_lock", None)
        if lock is None:
            lock = self._transaction_lock = threading.RLock()
        with lock:
            try:
                return self._request(payload, on_delta=on_delta, cancelled=cancelled, on_progress=on_progress)
            except (ChatImageError, ChatCancelled, ChatRequestTooLarge):
                raise
            except BaseException:
                self.mtp = mtp_capabilities(None)
                try:
                    self.invalidate_image_checks()
                except BaseException:
                    pass  # Preserve the primary failure; this process rejects its cache key.
                raise

    def _request(self, payload, on_delta=None, cancelled=None, *, on_progress=None):
        if on_progress is not None and on_delta is None and payload.get("stream") is True:

            def on_delta(_text):
                pass

        if cancelled is not None and cancelled.is_set():
            raise ChatCancelled("Request cancelled.")
        cancellable_operation = payload.get("operation") in {"generate", "prepare_image", "generate_image", "decide"}
        request_id = 0 if cancellable_operation else None
        if cancellable_operation and getattr(self, "cancellation", False):
            self.request_sequence += 1
            request_id = self.request_sequence
            payload = {**payload, "request_id": request_id}
        data = json.dumps(payload, ensure_ascii=False).encode() + b"\n"
        if len(data) > min(WIRE_BYTES, getattr(self, "max_request_bytes", WIRE_BYTES)):
            raise ChatRequestTooLarge("Conversation exceeds the worker request limit; use /reset.")
        if getattr(self, "_image_deadline", None) is not None:
            self._write_image_frame(data)
        else:
            # FileIO writes can be partial for large prompts.
            view = memoryview(data)
            while view:
                count = self._input_pipe.write(view)
                if not count:
                    raise ChatError("Cannot send request to worker.")
                view = view[count:]
        chunks = []
        progress = None
        control = cancelled
        while True:
            if control is not None and control.is_set() and request_id is not None:
                self.cancel(request_id)
            reply = self.read(control, request_id) if control is not None else self.read()
            if "event" not in reply:
                if isinstance(reply.get("error"), str) and reply["error"] in _FATAL_WORKER_ERRORS:
                    raise ChatError("Native worker failed; restart the model.")
                if reply.get("error") in {"request_too_large", "resource_budget_exceeded"}:
                    raise ChatRequestTooLarge("The worker rejected this request for exceeding its resource budget.")
                if (control is not None and control.is_set()) or reply.get("error") == "generation_cancelled":
                    raise ChatCancelled("Request cancelled.")
                if on_delta is not None and "error" not in reply and reply.get("text") != "".join(chunks):
                    raise ChatError("Invalid worker stream; restart the model.")
                return reply
            if reply.get("event") == "progress":
                accepted = progress_frame(reply, progress)
                if (
                    payload.get("stream") is not True
                    or payload.get("operation") not in {"generate", "generate_image"}
                    or accepted is None
                ):
                    raise ChatError("Invalid worker stream; restart the model.")
                progress = accepted
                if control is None or not control.is_set():
                    if (
                        payload.get("operation") == "generate_image"
                        and getattr(self, "_image_deadline", None) is not None
                    ):
                        self._renew_image_deadline()
                    if on_progress is not None:
                        try:
                            on_progress(accepted)
                        except OSError:
                            control = control or threading.Event()
                            control.set()
                continue
            if (
                on_delta is None
                or set(reply) != {"event", "text"}
                or reply["event"] != "delta"
                or not isinstance(reply["text"], str)
            ):
                raise ChatError("Invalid worker stream; restart the model.")
            chunks.append(reply["text"])
            if control is None or not control.is_set():
                if payload.get("operation") == "generate_image" and getattr(self, "_image_deadline", None) is not None:
                    self._renew_image_deadline()
                try:
                    on_delta(reply["text"])
                except OSError:
                    control = control or threading.Event()
                    control.set()

    def _renew_image_deadline(self):
        # Only validated generation activity renews an active transaction. Image
        # verification still has an absolute cap, and cancellation cannot renew.
        deadline = time.monotonic() + self.timeout
        verification_deadline = getattr(self, "_verification_deadline", None)
        self._image_deadline = min(deadline, verification_deadline) if verification_deadline is not None else deadline

    def request_image(self, prompt, image, options, on_delta=None, cancelled=None, *, on_progress=None):
        """Own one complete prepare/generate/discard transaction on the serialized worker."""
        with self._transaction_lock:
            self._last_image_tokens = None
            self._renew_image_deadline()
            try:
                return self._request_image(
                    prompt, image, options, on_delta=on_delta, cancelled=cancelled, on_progress=on_progress
                )
            except (ChatImageError, ChatCancelled, ChatRequestTooLarge):
                raise
            except BaseException:
                try:
                    self.invalidate_image_checks()
                except BaseException:
                    pass  # Cleanup must still kill/reap the native child.
                # Fatal protocol/transport errors leave alignment unknown. Kill
                # immediately instead of issuing another RPC or waiting again.
                try:
                    if self.process.poll() is None:
                        self.process.kill()
                        self.process.wait()
                except (OSError, AttributeError):
                    pass
                try:
                    self.close()
                except BaseException:
                    pass  # Preserve the original request failure.
                raise
            finally:
                self._image_deadline = None

    def _write_image_frame(self, data):
        """Bound pipe backpressure by the same image transaction deadline."""
        descriptor = self.process.stdin.fileno()
        was_blocking = os.get_blocking(descriptor)
        os.set_blocking(descriptor, False)
        try:
            view = memoryview(data)
            while view:
                remaining = self._image_deadline - time.monotonic()
                if remaining <= 0:
                    raise ChatError("Worker timed out; chat closed.")
                if not select.select([], [descriptor], [], remaining)[1]:
                    continue
                try:
                    written = os.write(descriptor, view)
                except BlockingIOError:
                    continue
                if not written:
                    raise ChatError("Cannot send request to worker.")
                view = view[written:]
        except OSError:
            raise ChatError("Cannot send request to worker.") from None
        finally:
            try:
                os.set_blocking(descriptor, was_blocking)
            except OSError:
                pass  # A concurrent close already invalidated this descriptor.

    def _request_image(self, prompt, image, options, on_delta=None, cancelled=None, *, on_progress=None):
        if self.media is None or self.image_input is None:
            raise ChatImageError("images_not_supported")
        primary_fatal = None
        try:
            with self.media.stage(image) as reference:
                aligned = True
                try:
                    requested_maximum = options["max_new_tokens"]
                    automatic_maximum = requested_maximum in (None, 0)
                    prepare_maximum = 1 if automatic_maximum else requested_maximum
                    prepared = self.request(
                        {
                            "operation": "prepare_image",
                            "image": reference,
                            "rendered_prompt": prompt,
                            "context_tokens": options["context_tokens"],
                            "max_new_tokens": prepare_maximum,
                        },
                        cancelled=cancelled,
                    )
                    self._check_image_reply(prepared)
                    identifier = prepared.get("prepared_id")
                    if (
                        not isinstance(identifier, str)
                        or len(identifier) != 32
                        or any(c not in "0123456789abcdef" for c in identifier)
                        or type(prepared.get("input_tokens")) is not int
                        or type(prepared.get("image_tokens")) is not int
                        or not 0 < prepared["image_tokens"] <= self.image_input["max_image_tokens"]
                        or not prepared["image_tokens"] < prepared["input_tokens"]
                        or prepared["input_tokens"] + prepare_maximum > options["context_tokens"]
                    ):
                        raise ChatError("Invalid image preparation reply; restart the model.")
                    self._last_image_tokens = prepared["image_tokens"]
                    if automatic_maximum:
                        maximum = min(
                            getattr(self, "max_new_tokens", options["context_tokens"] - 1),
                            options["context_tokens"] - prepared["input_tokens"],
                        )
                        if maximum < 1:
                            raise ChatImageError("context_overflow")
                    else:
                        maximum = requested_maximum
                    if "thinking_budget_tokens" in options and maximum < MIN_THINKING_BUDGET_OUTPUT_TOKENS:
                        raise ChatImageError("thinking_budget_output_too_small")
                    generate_options = {**options, "max_new_tokens": maximum}
                    reply = self.request(
                        {**generate_options, "operation": "generate_image", "prepared_id": identifier},
                        on_delta=on_delta,
                        cancelled=cancelled,
                        **({"on_progress": on_progress} if on_progress is not None else {}),
                    )
                    self._check_image_reply(reply)
                    if (
                        not isinstance(reply.get("text"), str)
                        or type(reply.get("input_tokens")) is not int
                        or reply["input_tokens"] != prepared["input_tokens"]
                        or type(reply.get("output_tokens")) is not int
                        or not 0 <= reply["output_tokens"] <= maximum
                        or reply.get("finish_reason")
                        not in {"eos", "stop_token", "stop_sequence", "length", "context"}
                    ):
                        raise ChatError("Invalid image generation reply; restart the model.")
                    return reply
                except (ChatImageError, ChatCancelled, ChatRequestTooLarge):
                    raise
                except BaseException as error:
                    primary_fatal = error
                    aligned = False
                    raise
                finally:
                    # Consume the discard reply even after cancellation, keeping
                    # the next request aligned with the private JSON protocol.
                    if aligned and self.process.poll() is None:
                        if time.monotonic() >= self._image_deadline:
                            raise ChatError("Worker timed out; chat closed.")
                        cleared = self.request({"operation": "discard_image"})
                        if type(cleared.get("discarded")) is not bool:
                            raise ChatError("Image cleanup failed; restart the model.")
        except MediaError:
            if primary_fatal is not None:
                raise primary_fatal from None
            raise ChatError("Private image staging failed; restart the model.") from None

    @staticmethod
    def _check_image_reply(reply):
        if "error" not in reply:
            return
        if isinstance(reply["error"], str) and reply["error"] in _RECOVERABLE_IMAGE_ERRORS:
            raise ChatImageError(reply["error"])
        raise ChatError("Native image generation failed; restart the model.")

    def invalidate_image_checks(self):
        from .image_checks import invalidate_checks

        self.images = False
        self.image_verification = "unverified"
        self.image_check_cached = False
        invalidate_checks(self)

    def close(self):
        self.mtp = mtp_capabilities(None)
        was_closed = getattr(self, "_closed", False)
        self._closed = True
        self._close_gpu_memory_reader()
        if self.process.poll() not in (None, 0) and not was_closed:
            try:
                self.invalidate_image_checks()
            except BaseException:
                pass  # A metadata failure cannot bypass child/resource cleanup.
        self.images = False
        self.image_verification = "unverified"

        def signal_worker(signal_number):
            if self.process.poll() is not None:
                return
            try:
                os.killpg(self.process.pid, signal_number)
            except ProcessLookupError:
                pass  # It exited between poll and signal; still reap it.
            except PermissionError:
                try:
                    self.process.send_signal(signal_number)
                except ProcessLookupError:
                    pass

        try:
            signal_worker(signal.SIGTERM)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                signal_worker(signal.SIGKILL)
                self.process.wait()
        finally:
            lane_lock = getattr(self, "_hy4_lane_lock", None)
            if lane_lock is not None:
                with lane_lock:
                    for lane in (*self.hy4_lanes, *self._pending_hy4_lanes):
                        lane.close()
                    self.hy4_lanes = ()
                    self._pending_hy4_lanes = []
            for pipe in (self.process.stdin, self.process.stdout):
                try:
                    pipe.close()
                except OSError:
                    pass
            cancel_fd = getattr(self, "cancel_fd", None)
            if cancel_fd is not None:
                self.cancel_fd = None
                try:
                    os.close(cancel_fd)
                except OSError:
                    pass
            media, self.media = getattr(self, "media", None), None
            if media is not None:
                media.close()


class _Hy4Lane(Worker):
    """A text protocol connection owned by the primary Hy4 process."""

    @classmethod
    def open(cls, timeout, child_pipes):
        lane = cls.__new__(cls)
        lane._init_protocol(timeout)
        with ExitStack() as host_pipes:
            child_in, host_out = os.pipe()
            child_pipes.callback(os.close, child_in)
            lane._lane_input = host_pipes.enter_context(os.fdopen(host_out, "wb", buffering=0))
            host_in, child_out = os.pipe()
            child_pipes.callback(os.close, child_out)
            lane._lane_output = host_pipes.enter_context(os.fdopen(host_in, "rb", buffering=0))
            child_cancel, lane.cancel_fd = os.pipe()
            child_pipes.callback(os.close, child_cancel)
            host_pipes.callback(os.close, lane.cancel_fd)
            os.set_blocking(lane.cancel_fd, False)
            lane._pipes = host_pipes.pop_all()
        return lane, (child_in, child_out, child_cancel)

    @property
    def _input_pipe(self):
        return self._lane_input

    @property
    def _output_pipe(self):
        return self._lane_output

    def invalidate_image_checks(self):
        pass  # Image readiness and verification belong to the primary lane.

    def close(self):
        self.cancel_fd = None
        self._pipes.close()


def _read_image_file(path: Path) -> ImageData:
    """Read one bounded regular image file without exposing its path in errors."""
    descriptor = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        flags |= getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_IMAGE_BYTES:
            raise OSError
        payload = bytearray()
        while len(payload) <= MAX_IMAGE_BYTES:
            remaining = MAX_IMAGE_BYTES + 1 - len(payload)
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            payload.extend(chunk)
        if not payload or len(payload) > MAX_IMAGE_BYTES:
            raise ValueError
    except (OSError, TypeError, ValueError):
        raise ChatError("Image file is invalid or unavailable.") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)

    data = bytes(payload)
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        media_type = "image/png"
    elif data.startswith(b"\xff\xd8\xff") and data.endswith(b"\xff\xd9"):
        media_type = "image/jpeg"
    else:
        raise ChatError("Image file is invalid or unavailable.")
    return ImageData(data, media_type)


def prefix_cache_enabled(ready):
    """Accept consecutive prefix reuse only from a complete readiness contract."""
    settings = ready.get("prefix_cache")
    if not isinstance(settings, dict):
        return False
    chunk = settings.get("chunk_tokens")
    subchunk = settings.get("subchunk_tokens")
    return (
        ready.get("state_policy") == "consecutive-prefix-v1"
        and settings.get("enabled") is True
        and type(chunk) is int
        and 1024 <= chunk <= 65536
        and chunk % 1024 == 0
        and (subchunk is None or (type(subchunk) is int and 1 <= subchunk < chunk and chunk % subchunk == 0))
        and "reset_cache" in (ready.get("operations") or [])
    )


def run_chat(args, worker):
    ready = worker.read()
    if ready.get("ready") is not True or ready.get("mode") != "text-generation-v1":
        raise ChatError("Generation worker could not load this model.")
    image_path = getattr(args, "image", None)
    mtp_enabled = getattr(args, "mtp", None)
    if mtp_enabled and image_path is not None:
        raise ChatError("MTP is unavailable for image requests.")
    capabilities = mtp_capabilities(ready)
    if mtp_enabled is None:
        # Draft by default when the loaded worker does so (Inkling's multi-row
        # verifier); image requests and older workers stay off.
        mtp_enabled = image_path is None and default_mtp_enabled(capabilities)
    try:
        drafts = resolve_mtp(mtp_enabled, capabilities)
    except ValueError as error:
        if str(error) == "mtp_not_supported":
            raise ChatError("MTP is unavailable for this model and worker.") from None
        raise ChatError("The MTP option must be a boolean.") from None
    codec, template = load_template(args.model_path, expected_hashes=getattr(worker, "image_input", None))
    context_capacity_known = ready["max_context_tokens"] != 0xFFFFFFFF
    context = args.context_tokens if args.context_tokens is not None else ready["max_context_tokens"]
    if context > ready["max_context_tokens"] or (
        args.max_new_tokens is not None and args.max_new_tokens > ready["max_new_tokens"]
    ):
        raise ChatError("Requested token limits exceed this worker's limits.")
    worker.generation_budget = ready.get("generation_budget") == "memory-context-v1"
    streaming = ready.get("streaming") == "text-delta-v1"
    image = None
    if image_path is not None:
        if not context_capacity_known:
            raise ChatError("Image requests require a model with a declared finite context capacity.")
        if getattr(worker, "images", False) is not True:
            raise ChatError("Image support is unavailable for this model and worker.")
        image = _read_image_file(image_path)
    system = args.system if args.system is not None else codec.get("default_system_prompt")
    initial = [{"role": "system", "content": system}] if system else []
    messages = list(initial)
    image_attached = False
    cache_scope = secrets.token_hex(16)
    print("Stark chat. /reset clears history; /exit quits. Ctrl-D or Ctrl-C exits.")
    while True:
        try:
            text = input("You> ")
        except EOFError:
            print()
            return 0
        if not text.strip():
            continue
        if text.strip() in ("/exit", "/quit"):
            return 0
        if text.strip() == "/reset":
            messages = list(initial)
            image_attached = False
            cache_scope = secrets.token_hex(16)
            print("Conversation cleared.")
            continue
        if text.strip() == "/help":
            print("Enter one message per line. Commands: /reset, /exit, /quit, /help.")
            continue
        content = text
        if image is not None and not image_attached:
            content = [{"type": "image"}, {"type": "text", "text": text}]
        candidate = [*messages, {"role": "user", "content": content}]
        try:
            prompt = template.render(
                messages=candidate,
                tools=None,
                add_generation_prompt=True,
                **reasoning_options(codec, thinking=args.thinking),
                bos_token=codec.get("bos_token") or "",
                eos_token=codec.get("eos_token") or "",
            )
            if image is not None:
                reply = worker.request_image(
                    prompt,
                    image,
                    {
                        "context_tokens": context,
                        "max_new_tokens": args.max_new_tokens or 0,
                        "sampling": {"mode": "greedy"},
                        **({"stream": True} if streaming else {}),
                    },
                    **({"on_delta": lambda _text: None} if streaming else {}),
                )
            else:
                tokens = worker.request({"operation": "tokenize", "text": prompt, "add_special_tokens": False})
                count = tokens.get("token_count")
                if type(count) is not int or count < 0:
                    raise ChatError("Worker could not tokenize the conversation.")
                if count >= context:
                    print("Conversation exceeds the context budget; use /reset or a larger --context-tokens.")
                    continue
                maximum = args.max_new_tokens
                request_context = context
                if maximum is None:
                    if context_capacity_known and not worker.generation_budget:
                        maximum = min(ready["max_new_tokens"], context - count)
                    else:
                        request_context, maximum = automatic_generation_budget(
                            worker,
                            count,
                            context,
                            context_capacity_known=context_capacity_known,
                            input_token_ids=tokens.get("token_ids"),
                            cache_scope=cache_scope if getattr(worker, "prefix_cache", False) else None,
                            mtp_drafts=drafts,
                        )
                if maximum < 1 or count + maximum > context:
                    print("Conversation exceeds the context budget; use /reset or a larger --context-tokens.")
                    continue
                request = {
                    "operation": "generate",
                    "rendered_prompt": prompt,
                    "context_tokens": request_context if context_capacity_known else count + maximum,
                    "max_new_tokens": maximum,
                    "sampling": {"mode": "greedy"},
                }
                if capabilities["supported"]:
                    request["mtp_drafts"] = drafts
                if getattr(worker, "prefix_cache", False) is True:
                    request["cache_scope"] = cache_scope
                if streaming:
                    request["stream"] = True
                    reply = worker.request(request, on_delta=lambda _text: None)
                else:
                    reply = worker.request(request)
            if not isinstance(reply.get("text"), str) or "error" in reply:
                raise ChatError("Generation failed; chat closed.")
        except ChatError as error:
            if isinstance(error, (ChatRequestTooLarge, ChatBudgetUnavailable)):
                print(str(error))
                continue
            if str(error).startswith("Conversation exceeds"):
                print(str(error))
                continue
            raise
        parser = tool_parser_for(codec)
        if parser == INKLING_PARSER:
            assistant, _, _ = parse_inkling_response(reply["text"])
            displayed = assistant["content"]
        else:
            tags = reasoning_tags(parser)
            assistant = assistant_message(
                reply["text"], initial_thinking=prompt_opens_reasoning(prompt, tags=tags), tags=tags
            )
            displayed = reply["text"]
        print("Stark> " + terminal_text(displayed) + "\n")
        messages = [*candidate, assistant]
        if image is not None:
            image_attached = True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, prog="stark chat")
    parser.add_argument("--model-path", type=Path, required=True, help="Converted Stark model directory")
    parser.add_argument("--image", type=Path, help="Attach one local PNG or JPEG image to terminal chat")
    parser.add_argument("--worker", type=Path, default=Path("stark-generate"), help=argparse.SUPPRESS)
    parser.add_argument("--system", help="System message (default: model metadata)")
    parser.add_argument("--context-tokens", type=int, help="Context budget (default: loaded model's maximum)")
    parser.add_argument("--max-new-tokens", type=int, help="Reply budget (default: all remaining context)")
    parser.add_argument(
        "--thinking", action="store_true", help="Enable reasoning where supported; GLM uses max effort (default: low)"
    )
    mtp = parser.add_mutually_exclusive_group()
    mtp.add_argument(
        "--mtp", dest="mtp", action="store_true", help="Enable exact MTP (default if the worker drafts by default)"
    )
    mtp.add_argument("--no-mtp", dest="mtp", action="store_false", help="Disable MTP")
    parser.set_defaults(mtp=None)
    parser.add_argument(
        "--timeout",
        type=float,
        default=300,
        help="Worker inactivity timeout in seconds; progress renews it (default: 300)",
    )
    args = parser.parse_args(argv)
    if (
        (args.context_tokens is not None and args.context_tokens < 1)
        or (args.max_new_tokens is not None and args.max_new_tokens < 1)
        or not 0 < args.timeout < float("inf")
    ):
        parser.error("token limits and timeout must be positive and finite")
    worker = None
    try:
        if read_model_config(args.model_path).get("type") not in VISION_MODEL_TYPES:
            load_template(args.model_path)
        print("Loading model...", file=sys.stderr)
        worker = Worker(args.worker, args.model_path, args.timeout)
        return run_chat(args, worker)
    except KeyboardInterrupt:
        print("\nChat closed.")
        return 130
    except ChatError as error:
        print("stark chat: " + str(error), file=sys.stderr)
        return 1
    except Exception:
        # Never expose template, IPC, filesystem or native exception payloads.
        print("stark chat: unable to complete chat; check the model and worker installation.", file=sys.stderr)
        return 1
    finally:
        if worker is not None:
            worker.close()


if __name__ == "__main__":
    raise SystemExit(main())
