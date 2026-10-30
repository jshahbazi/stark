"""Private JSON-lines controller for Stark's native menu bar app.

Only model metadata is persisted. Conversation content stays in memory and pipes.
"""

import argparse
import hashlib
import json
import math
import os
import re
import select
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from jinja2.exceptions import TemplateError

from .api_request import parse_chat_request
from .chat import (
    ChatCancelled,
    ChatError,
    ChatImageError,
    ChatRequestTooLarge,
    Worker,
    load_template,
    read_model_config,
)
from .decision import (
    DecisionInputError,
    DecisionProtocolError,
    ScoreWorker,
    decide,
    validate_inputs,
    worker_context_limit,
)
from .desktop_media import DesktopAttachments
from .hybrid_prepare import prepared_identity
from .local_api import APIError, LocalAPI
from .media import MAX_IMAGE_BYTES, MediaError
from .model_inventory import model_display
from .model_removal import RemovalError, prepare_removal, removal_info, remove_model
from .model_support import support_matrix
from .registry import get_model
from .sources import huggingface_repository
from .vision_runtime import supported_image_family

MAX_COMMAND_BYTES = 1024 * 1024
MAX_PROGRESS_BYTES = 4096
MAX_SUPPORT_BYTES = 65536
MAX_CATALOG_BYTES = 256 * 1024
PROGRESS_PHASES = ("download", "conversion")
MODEL_FILES = ("config.json", "tokenizer.json", "model.safetensors")


class DesktopError(Exception):
    """A fixed, content-free user-facing failure."""


def cancellation_error(cancelled):
    if getattr(cancelled, "reason", None) == "deadline":
        return APIError(504, "Request deadline exceeded.", "request_timeout")
    return APIError(499, "Request cancelled.", "request_cancelled")


def terminate(process):
    """Reap a private child and all descendants in its dedicated process group."""
    if process is None:
        return
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _qwen38_source(path):
    """A source export needs conversion even when a legacy receipt is present."""
    try:
        return read_model_config(path).get("model_type") == "qwen4_exp"
    except Exception:
        return False


def model_valid(path):
    if _qwen38_source(path):
        return False
    identity = prepared_identity(path)
    if identity is not None:
        if identity["architecture"] == "qwen4_exp":
            return False
        try:
            load_template(path)
            return True
        except Exception:
            return False
    if not path.is_dir() or not all((path / name).is_file() for name in MODEL_FILES):
        return False
    try:
        config = read_model_config(path)
        if not isinstance(config.get("decoder_config"), dict):
            return False
        if config.get("type") == "Qwen35VisionLanguageModelConfig":
            codec = config.get("token_codec_config", {})
            contract = config.get("artifact_contract")
            if (
                not isinstance(contract, dict)
                or type(contract.get("schema_version")) is not int
                or contract["schema_version"] not in (2, 3)
            ):
                return False
            return (
                all((path / name).is_file() for name in ("preprocessor_config.json", "chat_template.jinja"))
                and isinstance(codec, dict)
                and codec.get("type") == "ChatCodecConfig"
                and isinstance(codec.get("prompt_template"), str)
                and bool(codec["prompt_template"])
            )
        load_template(path)
        return True
    except Exception:
        return False


def metric(reply, key):
    value = reply.get(key)
    valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
    return value if valid else None


def metrics(reply):
    return {
        key: metric(reply, key)
        for key in (
            "input_tokens",
            "output_tokens",
            "input_tokens_per_second",
            "output_tokens_per_second",
            "prefill_seconds",
            "decode_seconds",
            "generation_seconds",
        )
    }


def converter_progress(value):
    """Accept only the content-free converter pipe contract."""
    if not isinstance(value, dict) or set(value) != {"phase", "state", "completed_bytes", "total_bytes"}:
        return None
    if (
        not isinstance(value["phase"], str)
        or value["phase"] not in PROGRESS_PHASES
        or not isinstance(value["state"], str)
        or value["state"] not in {"waiting", "running", "complete", "skipped"}
    ):
        return None
    completed, total = value["completed_bytes"], value["total_bytes"]
    if type(completed) is not int or not 0 <= completed <= 2**63 - 1:
        return None
    if total is not None and (type(total) is not int or not completed <= total <= 2**63 - 1):
        return None
    if value["state"] == "complete" and total is not None and completed != total:
        return None
    return {key: value[key] for key in ("phase", "state", "completed_bytes", "total_bytes")}


def api_model_id(model):
    """Expose source names while preserving stable private library identifiers."""
    source = model["source"]
    return source.removeprefix("huggingface:") if source.startswith("huggingface:") else model["name"]


class Desktop:
    """Backend for the menu bar app: model library, worker lifecycle and the loopback API.

    The app sends one JSON command per line on stdin (``list``, ``install``, ``run``,
    ``stop``, ``quit``, ``generate``, ``decide``, ``reset``, ``configure``, ``support``,
    ``catalog``, ``catalog_cancel``, ``install_catalog``, ``remove_model``) and receives
    one JSON event per line on ``output`` (``models``, ``status``, ``progress``,
    ``installed``, ``ready``, ``configured``, ``reply``, ``decision``, ``connection``,
    ``error``, ``model_families``, ``model_support``, ``model_catalog``,
    ``model_removed``, ``model_remove_error``). One worker is loaded at a
    time, under ``self.lock``. ``self.epoch`` increases whenever the model is
    stopped or replaced; commands, queued API
    requests and in-flight replies carry the epoch they started under and are
    discarded when it no longer matches, so a late reply from a previous worker
    can never reach the app or a client. App chat turns and API completions share
    one FIFO queue (``api_queue``) in front of the single worker. See
    docs/TECHNICAL_OVERVIEW.md for the control protocol and docs/LOCAL_API.md for
    the HTTP contract.
    """

    def __init__(self, executable, data_dir, output=None, timeout=300, score_executable=None, attachment_root=None):
        self.executable = executable
        self.score_executable = score_executable or Path(executable).with_name("stark-score")
        self.data_dir = data_dir.expanduser().absolute()
        self.output = output or sys.stdout
        self.timeout = timeout
        self.lock = threading.RLock()
        self.write_lock = threading.Lock()
        self._completion_events = {}
        self.output_failed = False
        self.epoch = 0
        self.job = None
        self.worker = None
        self.converter = None
        self.catalog_process = None
        self.catalog_generation = 0
        self.catalog_request_id = None
        self.catalog_family_id = None
        self.catalog_entries = {}
        self.catalog_termination_lock = threading.Lock()
        self.install_epoch = None
        self.install_progress = {}
        self.active_model = None
        self.active_mode = None
        self.vocab_size = None
        self.template = None
        self.codec = None
        self.messages = []
        self.image_identity = None
        self.attachments = None
        self.context = 8192
        self.maximum = 512
        self.system_prompt = None
        self.thinking = False
        self.context_limit = 8192
        self.output_limit = 512
        self.models = []
        self.api = None
        self.api_closing = False
        self.api_condition = threading.Condition(self.lock)
        self.api_queue = []
        self.api_queue_limit = 8
        self.api_queue_timeout = 30
        self.api_context = 16384
        self.api_maximum = 4096
        self.api_vocab = 0
        self.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.data_dir.chmod(0o700)
        self.model_dir = self.data_dir / "models"
        self.model_dir.mkdir(exist_ok=True, mode=0o700)
        self.model_dir.chmod(0o700)
        self.registry = self.data_dir / "models.json"
        if self.registry.exists():
            self.registry.chmod(0o600)
            value = json.loads(self.registry.read_text())
            if not isinstance(value, list):
                raise DesktopError("Cannot read the model library.")
            ids = set()
            for model in value:
                if (
                    not isinstance(model, dict)
                    or not all(isinstance(model.get(k), str) and model[k] for k in ("id", "name", "path", "source"))
                    or not Path(model["path"]).is_absolute()
                    or model["id"] in ids
                ):
                    raise DesktopError("Cannot read the model library.")
                ids.add(model["id"])
            self.models = value
        self.attachments = DesktopAttachments(attachment_root) if attachment_root is not None else None

    def emit(self, event, epoch=None, **fields):
        with self.lock:
            if epoch is not None and epoch != self.epoch:
                return
            operation = getattr(self, "_completion_events", {}).get(threading.current_thread())
            if operation is not None and (
                event in {"ready", "reply", "decision", "error", "installed", "model_removed", "model_remove_error"}
                or (event == "status" and fields.get("state") in {"ready", "idle", "error"})
            ):
                # The next command must be admissible when the UI sees Ready.
                # Publish terminal events only after this operation releases its slot.
                operation_epoch, completion = operation
                completion.append((event, operation_epoch if epoch is None else epoch, fields))
                return
            with self.write_lock:
                try:
                    serialized = json.dumps({"event": event, **fields}, ensure_ascii=False, allow_nan=False)
                    self.output.write(serialized + "\n")
                    self.output.flush()
                except (OSError, ValueError):
                    # A disappearing GUI must never trigger a thread traceback.
                    self.output_failed = True

    def status(self, state, message, epoch=None):
        self.emit("status", epoch, state=state, message=message)

    def library(self):
        with self.lock:
            rows = []
            for model in self.models:
                display = model_display(model)
                rows.append(
                    {
                        **{key: model[key] for key in ("id", "name", "path")},
                        "api_id": api_model_id(model),
                        "display": display,
                        "artifact_mode": display.get("artifact_mode", "text") if display else "text",
                        "removal": removal_info(model, self.model_dir, self.models),
                    }
                )
            self.emit(
                "models",
                models=rows,
            )

    def families(self):
        self.emit("model_families", report=support_matrix())

    @staticmethod
    def request_id(value):
        return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value) is not None

    def cancel_catalog(self, request_id=None):
        with self.lock:
            if request_id is not None and request_id != getattr(self, "catalog_request_id", None):
                return
            self.catalog_generation = getattr(self, "catalog_generation", 0) + 1
            process = getattr(self, "catalog_process", None)
            self.catalog_process = None
            self.catalog_request_id = None
            self.catalog_family_id = None
            self.catalog_entries = {}
        self.terminate_catalog(process)

    def terminate_catalog(self, process):
        if process is None:
            return
        with self.catalog_termination_lock:
            try:
                terminate(process)
            except PermissionError:
                if process.poll() is None:
                    raise

    def start_catalog(self, family, request_id):
        allowed = {item["family_id"] for item in support_matrix()["families"] if item["model_kind"] == "language"}
        if family not in allowed or not self.request_id(request_id):
            raise DesktopError("Choose a model family to browse.")
        self.cancel_catalog()
        with self.lock:
            self.catalog_request_id, self.catalog_family_id = request_id, family
            generation = self.catalog_generation
        threading.Thread(
            target=self.load_catalog, args=(family, request_id, generation), name="stark-model-catalog", daemon=True
        ).start()

    def run_catalog_check(self, command, environment, generation):
        process = None
        try:
            with self.lock:
                if generation != self.catalog_generation:
                    return None
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    env=environment,
                    start_new_session=True,
                )
                self.catalog_process = process
            pending = bytearray()
            deadline = time.monotonic() + 120
            while True:
                with self.lock:
                    if generation != self.catalog_generation:
                        return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ValueError()
                if not select.select([process.stdout], [], [], min(remaining, 0.1))[0]:
                    continue
                chunk = os.read(process.stdout.fileno(), 65536)
                if not chunk:
                    break
                pending.extend(chunk)
                if len(pending) > MAX_CATALOG_BYTES:
                    raise ValueError()
            with self.lock:
                if generation != self.catalog_generation:
                    return None
            remaining = deadline - time.monotonic()
            if remaining <= 0 or process.wait(timeout=remaining) != 0:
                raise ValueError()
            return json.loads(pending)
        finally:
            if process is not None:
                self.terminate_catalog(process)
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    if process.poll() is None:
                        raise
                if process.stdout is not None:
                    process.stdout.close()
                with self.lock:
                    if self.catalog_process is process:
                        self.catalog_process = None

    def catalog_report(self, value, family):
        if (
            not isinstance(value, dict)
            or type(value.get("schema_version")) is not int
            or value["schema_version"] != 1
            or value.get("family_id") != family
            or value.get("status") not in {"ready", "offline", "error"}
            or not isinstance(value.get("message"), str)
            or not 0 < len(value["message"]) <= 2000
            or not isinstance(value.get("items"), list)
            or len(value["items"]) > 30
            or type(value.get("has_more")) is not bool
        ):
            raise ValueError()
        if value["status"] != "ready" and value["items"]:
            raise ValueError()
        ids = set()
        for item in value["items"]:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("id"), str)
                or re.fullmatch(r"[0-9a-f]{32}", item["id"]) is None
                or item["id"] in ids
                or not isinstance(item.get("repo"), str)
                or huggingface_repository(item["repo"]) != item["repo"]
                or not isinstance(item.get("revision"), str)
                or re.fullmatch(r"[0-9a-f]{40}", item["revision"]) is None
                or not isinstance(item.get("name"), str)
                or not 0 < len(item["name"].encode()) <= 256
                or any(ord(char) < 32 or ord(char) == 127 for char in item["name"])
                or not isinstance(item.get("quantization"), str)
                or not 0 < len(item["quantization"]) <= 64
                or item.get("downloadable") is not True
                or not isinstance(item.get("artifact_modes"), list)
                or not item["artifact_modes"]
                or len(item["artifact_modes"]) != len(set(item["artifact_modes"]))
                or not set(item["artifact_modes"]) <= {"text", "vision-language"}
                or item.get("default_artifact_mode") not in item["artifact_modes"]
            ):
                raise ValueError()
            group = item.get("model_group")
            if group is not None and (
                not isinstance(group, dict)
                or set(group) != {"id", "name", "publisher"}
                or not isinstance(group.get("id"), str)
                or re.fullmatch(r"[0-9a-f]{32}", group["id"]) is None
                or not isinstance(group.get("name"), str)
                or not 0 < len(group["name"].encode()) <= 256
                or any(ord(char) < 32 or ord(char) == 127 for char in group["name"])
                or not isinstance(group.get("publisher"), str)
                or re.fullmatch(r"[A-Za-z0-9_.-]{1,96}", group["publisher"]) is None
            ):
                raise ValueError()
            for key in ("parameter_count", "weight_bytes"):
                if item.get(key) is not None and (type(item[key]) is not int or not 0 < item[key] <= 2**63 - 1):
                    raise ValueError()
            if item.get("size_label") is not None and (
                not isinstance(item["size_label"], str) or not 0 < len(item["size_label"]) <= 64
            ):
                raise ValueError()
            ids.add(item["id"])
        return value

    def catalog_installed_ids(self, item, artifact_mode=None, descriptions=None):
        identifiers = []
        descriptions = (
            descriptions
            if descriptions is not None
            else [(model["id"], model_display(model)) for model in self.models]
        )
        for identifier, display in descriptions:
            if (
                display
                and display["group_id"] == "repo:" + item["repo"]
                and display["source_revision"] == item["revision"]
                and (artifact_mode is None or display["artifact_mode"] == artifact_mode)
            ):
                identifiers.append(identifier)
        return identifiers

    def load_catalog(self, family, request_id, generation):
        command = [sys.executable, "-I", "-B", "-m", "stark_convert.cli", "--family-models", family]
        environment = dict(os.environ, HF_HUB_DISABLE_PROGRESS_BARS="1", HF_HUB_DISABLE_TELEMETRY="1")
        try:
            report = self.run_catalog_check(command, environment, generation)
            if report is None:
                return
            report = self.catalog_report(report, family)
        except Exception:
            report = {
                "schema_version": 1,
                "family_id": family,
                "status": "error",
                "items": [],
                "has_more": False,
                "message": "Models could not be loaded. Check your connection and try again.",
            }
        with self.lock:
            if generation != self.catalog_generation or request_id != self.catalog_request_id:
                return
            self.catalog_entries = {item["id"]: dict(item) for item in report["items"]}
            descriptions = [(model["id"], model_display(model)) for model in self.models]
            report = {
                **report,
                "items": [
                    {**item, "installed_ids": self.catalog_installed_ids(item, descriptions=descriptions)}
                    for item in report["items"]
                ],
            }
            self.emit("model_catalog", request_id=request_id, family_id=family, report=report)

    def install_catalog(self, payload):
        with self.lock:
            if self.job is not None and self.job.is_alive():
                raise DesktopError("Wait for the current operation or choose Stop.")
            if payload.get("catalog_request_id") != self.catalog_request_id:
                raise DesktopError("Refresh this family before downloading a model.")
            identifier = payload.get("entry_id")
            item = self.catalog_entries.get(identifier) if isinstance(identifier, str) else None
            request_id = payload.get("support_request_id")
            if item is None or not self.request_id(request_id):
                raise DesktopError("Choose an available model to download.")
            mode = payload.get("artifact_mode", item["default_artifact_mode"])
            if mode not in item["artifact_modes"]:
                raise DesktopError("Choose an available model version.")
            installed = self.catalog_installed_ids(item, mode)
            previous = next((model for model in self.models if model["id"] in installed), None)
            if previous is not None and model_valid(Path(previous["path"])):
                self.library()
                self.emit(
                    "installed",
                    model={key: previous[key] for key in ("id", "name", "path")},
                    support_request_id=request_id,
                )
                self.status("ready" if self.worker else "idle", "Model already downloaded")
                return
            self.start(
                self.install,
                "installing",
                "Checking model support…",
                "huggingface",
                item["repo"],
                mode,
                mode,
                request_id,
                item["revision"],
                request_id=request_id,
            )

    def remove_saved_model(self, epoch, identifier, action, request_id):
        stopped = False
        try:
            with self.lock:
                if epoch != self.epoch:
                    return
                model = next((entry for entry in self.models if entry["id"] == identifier), None)
                if model is None:
                    raise RemovalError("This model is no longer in your library.")
                plan = prepare_removal(model, self.model_dir, self.models)
                if plan.action != action:
                    raise RemovalError("The model folder changed. Refresh your library before removing it.")
                if self.active_model == identifier:
                    self.close_worker()
                    stopped = True
                original = list(self.models)

                def commit():
                    self.models[:] = [entry for entry in original if entry["id"] != identifier]
                    try:
                        self.persist()
                    except BaseException:
                        self.models[:] = original
                        raise

                result = remove_model(plan, commit)
                self.library()
                self.emit(
                    "model_removed",
                    epoch,
                    request_id=request_id,
                    stopped_model_id=identifier if stopped else None,
                    **result,
                )
                self.status("ready" if self.worker else "idle", "Model removed", epoch)
        except Exception as error:
            with self.lock:
                if epoch != self.epoch:
                    return
                message = (
                    str(error) if isinstance(error, RemovalError) else "The model could not be removed. Try again."
                )
                self.emit(
                    "model_remove_error",
                    epoch,
                    request_id=request_id,
                    model_id=identifier,
                    message=message,
                    stopped_model_id=identifier if stopped else None,
                )
                self.status("ready" if self.worker else "idle", "Ready" if self.worker else "Model stopped", epoch)

    def persist(self):
        temporary = self.data_dir / (".models-" + uuid.uuid4().hex)
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w") as output:
                json.dump(self.models, output, ensure_ascii=False)
                output.flush()
                os.fsync(output.fileno())
            temporary.replace(self.registry)
        finally:
            temporary.unlink(missing_ok=True)

    def close_worker(self):
        with self.lock:
            worker, self.worker = self.worker, None
            self.active_model = None
            self.active_mode = None
            self.vocab_size = None
            self.messages = []
            self.image_identity = None
            self.codec = None
            self.template = None
        if worker is not None:
            worker.close()

    def report_gpu_memory(self, epoch, identifier, mode, source, amount):
        with self.lock:
            if epoch != self.epoch or self.worker is not source:
                return
            self.emit(
                "gpu_memory",
                epoch,
                model_id=identifier,
                mode=mode,
                gpu_memory_bytes=amount,
            )

    def progress(self, epoch, phase, state, completed_bytes=0, total_bytes=None):
        with self.lock:
            if epoch != self.epoch or self.install_epoch != epoch:
                return
            value = {"state": state, "completed_bytes": completed_bytes, "total_bytes": total_bytes}
            previous = self.install_progress.get(phase)
            if previous and previous["state"] in {"complete", "skipped", "cancelled", "failed"}:
                return
            if previous and completed_bytes < previous["completed_bytes"]:
                return
            self.install_progress[phase] = value
            self.emit("progress", epoch, phase=phase, **value)

    def begin_install_progress(self, epoch, kind):
        with self.lock:
            if epoch != self.epoch:
                return
            self.install_epoch = epoch
            self.install_progress = {}
            self.progress(epoch, "download", "waiting" if kind == "huggingface" else "skipped")
            self.progress(epoch, "conversion", "waiting")

    def finish_install_progress(self, epoch, skipped=False):
        for phase in PROGRESS_PHASES:
            previous = self.install_progress.get(phase, {})
            self.progress(
                epoch,
                phase,
                "skipped" if skipped else "complete",
                previous.get("completed_bytes", 0),
                previous.get("total_bytes"),
            )

    def fail_install_progress(self, epoch):
        for phase in PROGRESS_PHASES:
            previous = self.install_progress.get(phase, {})
            self.progress(epoch, phase, "failed", previous.get("completed_bytes", 0), previous.get("total_bytes"))

    def stop(self, quiet=False):
        self.cancel_catalog()
        with self.lock:
            cancelled = self.install_progress.copy() if self.install_epoch == self.epoch else {}
            self.epoch += 1
            self.api_condition.notify_all()
            self.install_epoch = None
            converter, job = self.converter, self.job
            for phase, previous in cancelled.items():
                if previous["state"] in {"waiting", "running"}:
                    self.emit("progress", phase=phase, **{**previous, "state": "cancelled"})
        terminate(converter)
        self.close_worker()
        if job is not None and job is not threading.current_thread():
            job.join(timeout=5)
        with self.lock:
            # A cancelled job cannot publish events or acquire a new child.
            if self.job is job and (job is None or not job.is_alive()):
                self.job = None
            if self.converter is converter:
                self.converter = None
        if not quiet:
            self.status("idle", "Stopped")
        attachments = getattr(self, "attachments", None)
        if attachments is not None:
            try:
                attachments.clear()
            except MediaError:
                raise DesktopError("Cannot clear private image storage. Quit and reopen Stark.") from None

    def start(self, function, state, message, *args, request_id=None):
        with self.lock:
            if self.job is not None and self.job.is_alive():
                self.emit(
                    "error",
                    message="Wait for the current operation or choose Stop.",
                    **({"request_id": request_id} if request_id else {}),
                )
                return
            epoch = self.epoch
            installing = state == "installing"
            if installing:
                self.begin_install_progress(epoch, args[0])
            self.status(state, message, epoch)

            def execute():
                current = threading.current_thread()
                completion = []
                with self.lock:
                    self._completion_events[current] = (epoch, completion)
                try:
                    function(epoch, *args)
                except (DesktopError, ChatRequestTooLarge) as error:
                    if epoch == self.epoch:
                        if installing:
                            self.fail_install_progress(epoch)
                        self.emit(
                            "error", epoch, message=str(error), **({"request_id": request_id} if request_id else {})
                        )
                        self.status(
                            "ready" if self.worker else "error", "Ready" if self.worker else "Operation failed", epoch
                        )
                except BaseException:
                    if epoch == self.epoch:
                        if installing:
                            self.fail_install_progress(epoch)
                        self.close_worker()
                        self.emit(
                            "error",
                            epoch,
                            message="Unable to complete the operation. Check the model and try again.",
                            **({"request_id": request_id} if request_id else {}),
                        )
                        self.status("error", "Operation failed", epoch)
                finally:
                    with self.lock:
                        self._completion_events.pop(current, None)
                        if installing and self.install_epoch == epoch:
                            self.install_epoch = None
                        if self.job is current:
                            self.job = None
                            self.api_condition.notify_all()
                        for event, event_epoch, fields in completion:
                            self.emit(event, event_epoch, **fields)

            self.job = threading.Thread(target=execute, name="stark-desktop-operation", daemon=True)
            self.job.start()

    def read_converter_progress(self, descriptor, process, epoch):
        pending = b""
        discarding = False
        exited_at = None
        while True:
            if process.poll() is not None:
                exited_at = exited_at or time.monotonic()
                if time.monotonic() - exited_at >= 0.5:
                    break
            if not select.select([descriptor], [], [], 0.1)[0]:
                if exited_at is not None:
                    break
                continue
            chunk = os.read(descriptor, 65536)
            if not chunk:
                break
            pending += chunk
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                if discarding:
                    discarding = False
                    continue
                if len(line) > MAX_PROGRESS_BYTES:
                    continue
                try:
                    event = converter_progress(json.loads(line))
                except (ValueError, UnicodeError):
                    continue
                if event is not None:
                    # The app still validates/publishes the model and registry.
                    if event["phase"] == "conversion" and event["state"] == "complete":
                        event["state"] = "running"
                    self.progress(epoch, **event)
            if len(pending) > MAX_PROGRESS_BYTES:
                pending = b""
                discarding = True

    def run_converter(self, command, environment, epoch):
        read_descriptor, write_descriptor = os.pipe()
        process = None
        try:
            with self.lock:
                if epoch != self.epoch:
                    return None
                process = subprocess.Popen(
                    [*command, "--progress-fd", str(write_descriptor)],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=environment,
                    start_new_session=True,
                    pass_fds=(write_descriptor,),
                )
                self.converter = process
            os.close(write_descriptor)
            write_descriptor = None
            self.read_converter_progress(read_descriptor, process, epoch)
            return process.wait()
        finally:
            os.close(read_descriptor)
            if write_descriptor is not None:
                os.close(write_descriptor)
            if process is not None:
                terminate(process)
                # A child must not leave descendants holding its private pipe.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                with self.lock:
                    if self.converter is process:
                        self.converter = None

    def run_support_check(self, command, environment, epoch):
        """Read one bounded metadata-only report from an isolated cancellable child."""
        process = None
        try:
            with self.lock:
                if epoch != self.epoch:
                    return None
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    env=environment,
                    start_new_session=True,
                )
                self.converter = process
            pending = bytearray()
            deadline = time.monotonic() + min(self.timeout, 120)
            while True:
                with self.lock:
                    if epoch != self.epoch:
                        return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise DesktopError("The model support check timed out. Check Hugging Face access and retry.")
                if not select.select([process.stdout], [], [], min(remaining, 0.1))[0]:
                    continue
                chunk = os.read(process.stdout.fileno(), 65536)
                if not chunk:
                    break
                pending.extend(chunk)
                if len(pending) > MAX_SUPPORT_BYTES:
                    raise DesktopError("Cannot read the model support report. Update Stark and retry.")
            remaining = deadline - time.monotonic()
            if remaining <= 0 or process.wait(timeout=remaining) != 0:
                raise DesktopError("Cannot check this model's family. Check the source and Hugging Face access.")
            report = json.loads(pending)
            if (
                not isinstance(report, dict)
                or type(report.get("schema_version")) is not int
                or report["schema_version"] != 1
                or report.get("status") not in {"conditional", "unsupported", "inconclusive"}
                or not isinstance(report.get("message"), str)
                or not 0 < len(report["message"]) <= 2000
                or not isinstance(report.get("source"), dict)
            ):
                raise ValueError
            return report
        except DesktopError:
            raise
        except (OSError, ValueError, UnicodeError, TypeError, subprocess.SubprocessError):
            raise DesktopError("Cannot check this model's family. Check the source and Hugging Face access.") from None
        finally:
            if process is not None:
                terminate(process)
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                if process.stdout is not None:
                    process.stdout.close()
                with self.lock:
                    if self.converter is process:
                        self.converter = None

    def check_install_support(
        self, epoch, kind, source, selected_mode, support_mode, request_id, expected_revision=None
    ):
        command = [
            sys.executable,
            "-I",
            "-B",
            "-m",
            "stark_convert.cli",
            source,
            "--preflight",
            "--support-mode",
            support_mode,
            "--artifact-mode",
            selected_mode,
        ]
        if expected_revision is not None:
            command.extend(["--revision", expected_revision])
        if kind == "local" or get_model(source) is None:
            command.extend(["--metadata-profile", "hf-generic"])
        if kind == "local":
            command.append("--offline")
        environment = dict(os.environ, HF_HUB_DISABLE_PROGRESS_BARS="1", HF_HUB_DISABLE_TELEMETRY="1")
        self.status("installing", "Checking model family before downloading weights…", epoch)
        report = self.run_support_check(command, environment, epoch)
        with self.lock:
            if epoch != self.epoch:
                return None
            if report is None:
                raise DesktopError("Cannot check this model's family. Retry the support check.")
            if report.get("mode") != support_mode:
                raise DesktopError("The model support report does not match this installation. Retry.")
            identity = report["source"]
            revision = identity.get("revision")
            if report["status"] == "conditional" and expected_revision is not None and revision != expected_revision:
                raise DesktopError("The model version changed. Refresh this family and try again.")
            if kind == "huggingface" and (
                identity.get("repo") != source
                or not isinstance(revision, str)
                or re.fullmatch(r"[0-9a-f]{40}", revision) is None
            ):
                if report["status"] == "conditional":
                    raise DesktopError("Cannot pin this model's source revision. Retry the support check.")
            self.emit("model_support", epoch, request_id=request_id, report=report)
            if report["status"] != "conditional":
                raise DesktopError(report["message"])
            return revision

    def install(
        self,
        epoch,
        kind,
        source,
        artifact_mode=None,
        support_mode=None,
        support_request_id=None,
        expected_revision=None,
    ):
        if expected_revision is not None and (
            kind != "huggingface"
            or not isinstance(expected_revision, str)
            or re.fullmatch(r"[0-9a-f]{40}", expected_revision) is None
        ):
            raise DesktopError("Choose a pinned model version to download.")
        if artifact_mode not in (None, "text", "vision-language"):
            raise DesktopError("Choose text or vision-language conversion.")
        selected_mode = artifact_mode or "text"
        if support_mode not in (None, "text", "decision", "vision-language"):
            raise DesktopError("Choose text, decision or image support.")
        if support_request_id is not None and (
            not isinstance(support_request_id, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", support_request_id) is None
        ):
            raise DesktopError("Invalid model support request.")
        if artifact_mode == "vision-language" and support_mode not in (None, "vision-language"):
            raise DesktopError("Image artifacts support Chat mode only. Choose image input or a text artifact.")
        if kind == "local":
            path = Path(source).expanduser().resolve()
            if not path.is_dir():
                raise DesktopError("Choose a complete local model folder.")
            source_key = "local:" + str(path)
            name = path.name
            converted = model_valid(path)
            try:
                local_config = read_model_config(path)
                is_stark = "decoder_config" in local_config or (
                    "stark_native_hybrid" in local_config and local_config.get("model_type") != "qwen4_exp"
                )
            except Exception:
                is_stark = False
            if is_stark and not converted:
                raise DesktopError("This Stark folder is incomplete or has no supported chat template.")
            if converted:
                config = read_model_config(path)
                stored_mode = "vision-language" if config.get("type") == "Qwen35VisionLanguageModelConfig" else "text"
                if artifact_mode is not None and selected_mode != stored_mode:
                    raise DesktopError("The selected conversion mode does not match this Stark artifact.")
                selected_mode = stored_mode
        elif kind == "huggingface":
            try:
                source = huggingface_repository(source)
            except ValueError:
                raise DesktopError("Enter a model name or a https://huggingface.co/owner/model URL.") from None
            source_key, name, converted = "huggingface:" + source, source.split("/")[-1], False
        else:
            raise DesktopError("Choose a local folder or Hugging Face repository.")
        purpose = support_mode or selected_mode
        if converted and selected_mode == "vision-language" and purpose == "text":
            purpose = "vision-language"  # Ordinary Chat can import an existing image artifact.
        if selected_mode == "vision-language" and purpose != "vision-language":
            raise DesktopError(
                "This vision artifact supports Chat mode only. Choose Chat or a text artifact for decisions."
            )
        if selected_mode != "vision-language" and purpose == "vision-language":
            raise DesktopError("Enable image input when converting this checkpoint.")
        with self.lock:
            if epoch != self.epoch:
                return
            previous = next(
                (
                    m
                    for m in self.models
                    if m["source"] == source_key
                    and m.get("artifact_mode", "text") == selected_mode
                    and (
                        expected_revision is None
                        or (model_display(m) or {}).get("source_revision") == expected_revision
                    )
                ),
                None,
            )
        legacy_previous = None
        if previous is not None and _qwen38_source(Path(previous["path"])):
            # Re-import a legacy source through the ordinary converter. Keep
            # its registration until the replacement is published atomically.
            legacy_previous, previous = previous, None
        if previous is not None:
            if not model_valid(Path(previous["path"])):
                raise DesktopError("The installed model folder is missing or incomplete.")
            verification = self.check_family_install(epoch, Path(previous["path"]))
            with self.lock:
                if epoch != self.epoch:
                    return
                if verification is not None:
                    previous["image_verification"] = verification
                    self.persist()
                self.finish_install_progress(epoch, skipped=True)
                self.emit(
                    "installed",
                    epoch,
                    model={k: previous[k] for k in ("id", "name", "path")},
                    support_request_id=support_request_id,
                )
                self.status("ready" if self.worker else "idle", "Model already installed", epoch)
                return
        identifier = uuid.uuid4().hex
        staging = self.model_dir / (".install-" + identifier)
        managed = None
        source_revision = None
        try:
            if not converted:
                source_revision = self.check_install_support(
                    epoch,
                    kind,
                    str(path) if kind == "local" else source,
                    selected_mode,
                    purpose,
                    support_request_id,
                    **({"expected_revision": expected_revision} if expected_revision is not None else {}),
                )
                with self.lock:
                    if epoch != self.epoch:
                        return
                staging.mkdir(mode=0o700)
                output = staging / "model"
                command = [
                    sys.executable,
                    "-I",
                    "-B",
                    "-m",
                    "stark_convert.cli",
                    str(path) if kind == "local" else source,
                    "--output",
                    str(output),
                    "--support-mode",
                    purpose,
                ]
                if kind == "local" or get_model(source) is None:
                    command.extend(["--metadata-profile", "hf-generic"])
                if kind == "local":
                    command.append("--offline")
                if selected_mode == "vision-language":
                    command.extend(["--artifact-mode", "vision-language"])
                if source_revision is not None:
                    command.extend(["--revision", source_revision])
                environment = dict(os.environ, HF_HUB_DISABLE_PROGRESS_BARS="1", HF_HUB_DISABLE_TELEMETRY="1")
                result = self.run_converter(command, environment, epoch)
                with self.lock:
                    if epoch != self.epoch:
                        return
                if result != 0 or not model_valid(output):
                    raise DesktopError(
                        "Model installation failed. Check model support, free disk space and Hugging Face access."
                    )
                with self.lock:
                    if epoch != self.epoch:
                        return
                    managed = self.model_dir / identifier
                    output.replace(managed)
                    path = managed
            verification = self.check_family_install(epoch, path)
            model = {"id": identifier, "name": name, "path": str(path), "source": source_key}
            if selected_mode == "vision-language":
                model["artifact_mode"] = selected_mode
            if verification is not None:
                model["image_verification"] = verification
            with self.lock:
                if epoch != self.epoch:
                    return
                previous_index = self.models.index(legacy_previous) if legacy_previous in self.models else None
                if previous_index is None:
                    self.models.append(model)
                else:
                    self.models[previous_index] = model
                try:
                    self.persist()
                except BaseException:
                    if previous_index is None:
                        self.models.remove(model)
                    else:
                        self.models[previous_index] = legacy_previous
                    raise
                managed = None  # The registry now owns the managed folder.
                self.finish_install_progress(epoch, skipped=converted)
                self.library()
                self.emit(
                    "installed",
                    epoch,
                    model={k: model[k] for k in ("id", "name", "path")},
                    support_request_id=support_request_id,
                )
                self.status("ready" if self.worker else "idle", "Model installed", epoch)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            if managed is not None:
                shutil.rmtree(managed, ignore_errors=True)

    def check_family_install(self, epoch, path):
        """Check a family artifact before publishing it, with one loaded worker."""
        config = read_model_config(path)
        contract = config.get("artifact_contract", {})
        if (
            config.get("type") != "Qwen35VisionLanguageModelConfig"
            or not isinstance(contract, dict)
            or contract.get("schema_version") != 3
        ):
            return None
        family = contract.get("family")
        if not supported_image_family(family):
            raise DesktopError("This image model family is not supported.")
        with self.lock:
            if epoch != self.epoch:
                raise DesktopError("Image checks were interrupted by a model change.")
        self.emit("image_check", epoch, state="checking")
        self.status("installing", "Checking image support on this Mac…", epoch)
        with self.lock:
            if epoch != self.epoch:
                raise DesktopError("Image checks were interrupted by a model change.")
            self.close_worker()
        worker = None
        try:
            with self.lock:
                if epoch != self.epoch:
                    raise DesktopError("Image checks were interrupted by a model change.")
                worker = Worker(self.executable, path, self.timeout)
                worker.image_check_dir = self.data_dir / "image-checks"
                self.worker = worker
            ready = worker.read()
            with self.lock:
                if epoch != self.epoch or self.worker is not worker:
                    raise DesktopError("Image checks were interrupted by a model change.")
                if (
                    ready.get("ready") is not True
                    or ready.get("mode") != "text-generation-v1"
                    or worker.images is not True
                    or worker.image_verification != "locally_tested"
                ):
                    raise DesktopError("This model did not pass local image execution checks.")
                return "locally_tested"
        except ChatError as error:
            raise DesktopError(str(error)) from None
        finally:
            with self.lock:
                current = self.worker is worker
                if current:
                    self.close_worker()
            if not current and worker is not None:
                worker.close()

    def run(self, epoch, identifier, mode="chat"):
        model = next((m for m in self.models if m["id"] == identifier), None)
        if model is not None and _qwen38_source(Path(model["path"])):
            raise DesktopError("Import this Qwen3.8 source again to create a converted Stark model.")
        if model is None or not model_valid(Path(model["path"])):
            raise DesktopError("Select an installed model whose folder is still available.")
        vision = read_model_config(Path(model["path"])).get("type") == "Qwen35VisionLanguageModelConfig"
        codec, template = (None, None) if vision else load_template(Path(model["path"]))
        if mode == "decision" and vision:
            raise DesktopError("Vision-language artifacts support Chat mode only.")
        with self.lock:
            if epoch != self.epoch:
                return

            def on_gpu_memory(source, amount):
                self.report_gpu_memory(epoch, identifier, mode, source, amount)

            worker = (
                ScoreWorker(
                    self.score_executable,
                    Path(model["path"]),
                    self.timeout,
                    gpu_memory_callback=on_gpu_memory,
                )
                if mode == "decision"
                else Worker(
                    self.executable,
                    Path(model["path"]),
                    self.timeout,
                    gpu_memory_callback=on_gpu_memory,
                )
            )
            self.worker = worker
            if mode == "chat":
                worker.image_check_dir = self.data_dir / "image-checks"
                worker.image_check_progress = lambda: self.status(
                    "loading", "Checking image support on this Mac…", epoch
                )
        ready = worker.read()
        if vision:
            codec, template = load_template(Path(model["path"]), expected_hashes=worker.image_input)
        if mode == "decision":
            try:
                decision_context_limit = worker_context_limit(ready, legacy=True)
            except DecisionProtocolError:
                self.close_worker()
                raise DesktopError(
                    "Stark could not load this model for decisions. Check model support and memory."
                ) from None
            if (
                ready.get("ready") is not True
                or ready.get("mode") != "exact-token-scoring-cache-v1"
                or type(ready.get("vocab_size")) is not int
                or not 0 < ready["vocab_size"] <= 2**32 - 1
            ):
                self.close_worker()
                raise DesktopError("Stark could not load this model for decisions. Check model support and memory.")
            with self.lock:
                if epoch != self.epoch:
                    return
                self.codec, self.template = codec, template
                self.active_model, self.active_mode = identifier, mode
                self.vocab_size = ready["vocab_size"]
                worker.max_context_tokens = decision_context_limit
                self.messages = []
                self.emit("ready", epoch, model_id=identifier, mode=mode)
                self.status("ready", "Ready", epoch)
            return
        if (
            ready.get("ready") is not True
            or ready.get("mode") != "text-generation-v1"
            or type(ready.get("max_context_tokens")) is not int
            or not 2 <= ready["max_context_tokens"] <= 65536
            or type(ready.get("max_new_tokens")) is not int
            or not 1 <= ready["max_new_tokens"] <= 4096
        ):
            self.close_worker()
            raise DesktopError("Stark could not load this model. Check model support and available memory.")
        with self.lock:
            if epoch != self.epoch:
                return
            worker.streaming = ready.get("streaming") == "text-delta-v1"
            worker.stop_sequences = ready.get("stop_sequences") is True
            self.codec, self.template = codec, template
            self.context_limit = ready["max_context_tokens"]
            self.output_limit = min(4096, ready["max_new_tokens"], self.context_limit - 1)
            self.context = min(self.context, self.context_limit)
            self.maximum = min(self.maximum, self.output_limit, self.context - 1)
            self.api_context = self.context_limit
            self.api_maximum = min(self.output_limit, self.api_context - 1)
            self.api_vocab = ready.get("vocab_size", 0)
            self.active_model = identifier
            self.active_mode = mode
            self.reset()
            self.emit("ready", epoch, model_id=identifier, mode=mode, **self.settings_summary())
            self.status("ready", "Ready", epoch)

    def settings_summary(self):
        # The system prompt stays in memory; acknowledgements contain only limits.
        return {
            "context_tokens": self.context,
            "max_new_tokens": self.maximum,
            "max_context_tokens": self.context_limit,
            "max_output_tokens": self.output_limit,
            "images": bool(self.attachments is not None and self.worker and getattr(self.worker, "images", False)),
            "image_attachment_transport": "private-file-v1",
            "max_image_bytes": MAX_IMAGE_BYTES,
            "image_verification": getattr(self.worker, "image_verification", "unverified"),
        }

    def configure(self, payload):
        with self.lock:
            if self.job is not None:
                raise DesktopError("Wait for the current operation or choose Stop.")
            if self.worker is None:
                raise DesktopError("Run a model before changing advanced options.")
            if self.active_mode != "chat":
                raise DesktopError("Advanced chat options are available in Chat mode.")
            context, maximum = payload.get("context_tokens"), payload.get("max_new_tokens")
            system, thinking = payload.get("system_prompt"), payload.get("thinking")
            if (
                set(payload) != {"command", "context_tokens", "max_new_tokens", "system_prompt", "thinking"}
                or type(context) is not int
                or not 2 <= context <= self.context_limit
                or type(maximum) is not int
                or not 1 <= maximum <= min(self.output_limit, context - 1)
                or (system is not None and not isinstance(system, str))
                or type(thinking) is not bool
            ):
                raise DesktopError("Invalid advanced options. Check the model's token limits.")
            self.context, self.maximum = context, maximum
            self.system_prompt, self.thinking = system, thinking
            self.reset()
            self.emit("configured", **self.settings_summary())

    def reset(self):
        with self.lock:
            system = self.system_prompt
            if system is None:
                system = (self.codec or {}).get("default_system_prompt")
            self.messages = [{"role": "system", "content": system}] if system else []
            self.image_identity = None

    def generate(self, epoch, text, reference=None):
        with self.lock:
            if epoch != self.epoch:
                return
            if self.active_mode != "chat":
                raise DesktopError("Choose Chat mode and run a model before sending a message.")
            attachments = self.attachments
            previous_image = self.image_identity
            if reference is not None and (
                attachments is None or self.worker is None or not getattr(self.worker, "images", False)
            ):
                raise DesktopError("Image support has not passed checks for this model.")
            if previous_image is not None and reference is None:
                raise DesktopError("Resupply the conversation image or start a new chat.")
            history = list(self.messages)
            maximum, thinking = self.maximum, self.thinking
        image = None
        identity = None
        try:
            if reference is not None:
                self.emit("image_progress", epoch, phase="preparing")
                image = attachments.claim(reference)
                identity = (image.media_type, hashlib.sha256(image.data).digest())
                if previous_image is not None and identity != previous_image:
                    raise DesktopError("Start a new chat to use a different image.")
            content = (
                [{"type": "text", "text": text}, {"type": "image"}]
                if image is not None and previous_image is None
                else text
            )
            candidate = [*history, {"role": "user", "content": content}]
            if image is not None:
                self.emit("image_progress", epoch, phase="generating")
            reply = self.infer(candidate, maximum, epoch=epoch, thinking=thinking, image=image)
        except (MediaError, ChatImageError) as error:
            raise DesktopError(str(error)) from None
        with self.lock:
            if epoch != self.epoch:
                return
            self.messages = [*candidate, {"role": "assistant", "content": reply["text"]}]
            self.image_identity = identity
            fields = metrics(reply)
            self.emit("metrics", epoch, **fields)
            self.emit("reply", epoch, text=reply["text"], finish_reason=reply.get("finish_reason"), **fields)
            self.status("ready", "Ready", epoch)

    def infer(
        self,
        messages,
        maximum,
        on_delta=None,
        *,
        context=None,
        sampling=None,
        seed=0,
        stop=None,
        cancelled=None,
        epoch=None,
        thinking=False,
        image=None,
    ):
        with self.lock:
            if epoch is not None and epoch != self.epoch:
                raise APIError(503, "Model changed or stopped before this request started.", "model_changed")
            worker, template, codec = self.worker, self.template, self.codec
            dispatch_epoch, identifier = self.epoch, self.active_model
            context = context or self.context
        if worker is None or template is None:
            raise APIError(503, "Run a model before sending a prompt.", "model_not_loaded")
        if on_delta is not None and not worker.streaming:
            raise APIError(
                503, "This worker needs an update for streaming. Rebuild or reinstall Stark.", "streaming_unavailable"
            )
        if stop and not worker.stop_sequences:
            raise APIError(503, "This worker needs an update for stop sequences.", "stop_sequences_unavailable")
        if on_delta is not None:
            deliver = on_delta

            def on_delta(text):
                # Keep delivery atomic with Stop/model replacement. The HTTP
                # writer has a bounded timeout. OSError asks Worker to cancel
                # and consume the terminal reply before the next request.
                with self.lock:
                    if self.epoch != dispatch_epoch or self.worker is not worker or self.active_model != identifier:
                        raise OSError("Request no longer belongs to the running model.")
                    deliver(text)

        try:
            prompt = template.render(
                messages=messages,
                tools=None,
                add_generation_prompt=True,
                enable_thinking=thinking,
                bos_token=codec.get("bos_token") or "",
                eos_token=codec.get("eos_token") or "",
            )
        except (TemplateError, ChatError):
            raise APIError(400, "The model's chat template rejected these messages.", "invalid_messages") from None
        if image is not None:
            if not getattr(worker, "images", False):
                raise APIError(400, "Image support has not passed checks for this model.", "images_unsupported")
            options = {
                "context_tokens": context,
                "max_new_tokens": maximum,
                "sampling": sampling or {"mode": "greedy"},
                "seed": seed,
                "stop": stop or [],
                "stream": on_delta is not None,
            }
            reply = worker.request_image(prompt, image, options, on_delta=on_delta, cancelled=cancelled)
            if not isinstance(reply.get("text"), str) or "error" in reply:
                raise ValueError
            return reply
        tokens = worker.request(
            {"operation": "tokenize", "text": prompt, "add_special_tokens": False}, cancelled=cancelled
        )
        count = tokens.get("token_count")
        if type(count) is not int:
            raise ValueError
        if count + maximum > context:
            raise DesktopError("Conversation is full. Start a new conversation and try again.")
        token_ids = tokens.get("token_ids")
        if (
            not isinstance(token_ids, list)
            or len(token_ids) != count
            or any(type(token) is not int for token in token_ids)
        ):
            raise ValueError
        request = {
            "operation": "generate",
            "input_token_ids": token_ids,
            "context_tokens": context,
            "max_new_tokens": maximum,
            "sampling": sampling or {"mode": "greedy"},
            "seed": seed,
        }
        if stop:
            request["stop"] = stop
        if on_delta is not None:
            request["stream"] = True
            reply = worker.request(request, on_delta=on_delta, cancelled=cancelled)
        else:
            reply = worker.request(request, cancelled=cancelled)
        if not isinstance(reply.get("text"), str) or "error" in reply:
            raise ValueError
        return reply

    def start_api(self):
        try:
            self.api = LocalAPI(self.api_models, self.api_complete, status=self.api_status, cancellable=True)
            details = self.api.start()
            # Credentials travel only through the private app pipe; never persist them.
            self.emit("connection", **details)
        except OSError:
            self.api = None
            self.emit("connection", message="Cannot start the local API. Quit and reopen Stark to retry.")

    def api_models(self):
        with self.lock:
            if self.active_model is None or self.active_mode != "chat":
                return []
            model = next(m for m in self.models if m["id"] == self.active_model)
            return [
                {
                    "id": api_model_id(model),
                    "object": "model",
                    "created": 0,
                    "owned_by": "stark",
                    "name": model["name"],
                    "context_length": self.api_context,
                    "max_output_tokens": self.api_maximum,
                }
            ]

    def api_status(self):
        with self.lock:
            return {
                "status": "ok",
                "model_loaded": self.active_model is not None and self.active_mode == "chat",
                "busy": self.job is not None,
                "queued_requests": len(self.api_queue),
                "queue_limit": self.api_queue_limit,
                "queue_timeout_seconds": self.api_queue_timeout,
                "models": self.api_models(),
                "image_verification": (
                    getattr(self.worker, "image_verification", "unverified")
                    if self.active_mode == "chat" and self.worker and getattr(self.worker, "images", False)
                    else "unverified"
                ),
                "capabilities": {
                    "chat_completions": True,
                    "streaming": True,
                    "sampling": ["temperature", "top_p", "top_k", "min_p", "seed"],
                    "stop_sequences": bool(self.active_mode == "chat" and self.worker and self.worker.stop_sequences),
                    "cancellation": bool(self.active_mode == "chat" and self.worker and self.worker.cancellation),
                    "tool_calling": False,
                    "structured_output": False,
                    "images": bool(
                        self.active_mode == "chat" and self.worker and getattr(self.worker, "images", False)
                    ),
                    "embeddings": False,
                    "responses": False,
                },
            }

    def api_complete(self, payload, on_delta=None, cancelled=None):
        """Run one chat completion for the HTTP API and return the response body.

        ``payload`` is the raw request object; it is validated under the lock, then
        the request takes a FIFO ticket and waits (at most ``api_queue_timeout``
        seconds) for the worker. ``on_delta`` receives streamed text when the client
        asked for streaming. ``cancelled`` is the handler's event: it is set when
        the client disconnects or the request deadline passes, and its optional
        ``reason`` attribute (``"deadline"``) selects the 504 response. While the
        completion runs, ``self.job`` is the serving thread, which ``stop`` uses to
        fence it. Failures map to ``APIError``: a cancelled request, a context
        overflow or oversized worker request (400), or a worker fault (503, after which the worker is closed
        and the app shows the model as stopped).
        """
        cancelled = cancelled or threading.Event()
        with self.lock:
            request = parse_chat_request(
                payload,
                self.api_maximum,
                self.api_vocab,
                allow_images=bool(self.worker and getattr(self.worker, "images", False)),
            )
            maximum = request["maximum"]
            messages = request["messages"]
            if self.api_closing:
                raise APIError(503, "Stark is shutting down.", "server_stopped")
            if self.active_model is None or self.worker is None or self.active_mode != "chat":
                raise APIError(503, "Run a model in Stark first.", "model_not_loaded")
            active = next(m for m in self.models if m["id"] == self.active_model)
            public_id = api_model_id(active)
            if request["model"] not in (None, public_id, self.active_model, active["name"]):
                raise APIError(
                    404, "Select the running model ID from Connection Details or /v1/models.", "model_not_found"
                )
            epoch, identifier, worker = self.epoch, self.active_model, self.worker
            if len(self.api_queue) >= self.api_queue_limit:
                raise APIError(429, "Stark's request queue is full. Retry shortly.", "queue_full")
            ticket = object()
            self.api_queue.append(ticket)
            deadline = time.monotonic() + self.api_queue_timeout
            try:
                while self.job is not None or self.api_queue[0] is not ticket:
                    if cancelled.is_set():
                        raise cancellation_error(cancelled)
                    if self.api_closing or self.epoch != epoch:
                        raise APIError(503, "Model changed or stopped while this request waited.", "model_changed")
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise APIError(429, "Request queue wait expired. Retry shortly.", "queue_timeout")
                    self.api_condition.wait(min(0.1, remaining))
                if cancelled.is_set():
                    raise cancellation_error(cancelled)
                if self.api_closing or self.epoch != epoch or self.active_model != identifier:
                    raise APIError(503, "Model changed or stopped while this request waited.", "model_changed")
            finally:
                self.api_queue.remove(ticket)
                self.api_condition.notify_all()
            self.job = threading.current_thread()
            candidate = [dict(m) for m in messages]
            system = (self.codec or {}).get("default_system_prompt")
            if system and not any(m["role"] == "system" for m in candidate):
                candidate.insert(0, {"role": "system", "content": system})
            self.status("generating", "Generating for connected application…", epoch)
        try:
            reply = self.infer(
                candidate,
                maximum,
                on_delta=on_delta,
                context=self.api_context,
                sampling=request["sampling"],
                seed=request["seed"],
                stop=request["stop"],
                cancelled=cancelled,
                epoch=epoch,
                image=request["image"],
            )
            with self.lock:
                if epoch != self.epoch or self.worker is not worker or self.active_model != identifier:
                    raise APIError(503, "Request stopped. Run a model and retry.", "request_stopped")
                self.emit("metrics", epoch, **metrics(reply))
            prompt_tokens = metric(reply, "input_tokens") or 0
            completion_tokens = metric(reply, "output_tokens") or 0
            return {
                "id": "chatcmpl-" + uuid.uuid4().hex,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": public_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": reply["text"]},
                        "finish_reason": "length" if reply.get("finish_reason") in ("length", "context") else "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
        except ChatCancelled:
            raise cancellation_error(cancelled) from None
        except ChatImageError as error:
            if error.code == "context_overflow":
                raise APIError(400, "Image and text exceed the context budget.", "context_length_exceeded") from None
            raise APIError(
                400, "The image request was rejected. Check its format and token budget.", error.code
            ) from None
        except ChatRequestTooLarge:
            raise APIError(
                400,
                "Rendered request exceeds the worker size limit. Shorten the conversation.",
                "request_too_large",
            ) from None
        except DesktopError:
            raise APIError(
                400,
                "Conversation exceeds the context budget. Shorten messages or the reply limit.",
                "context_length_exceeded",
            ) from None
        except APIError:
            raise
        except BaseException:
            with self.lock:
                if epoch == self.epoch and self.worker is worker and self.active_model == identifier:
                    self.close_worker()
            raise APIError(503, "Generation failed. Run the model again in Stark.", "generation_failed") from None
        finally:
            with self.lock:
                if self.job is threading.current_thread():
                    self.job = None
                    self.api_condition.notify_all()
                if epoch == self.epoch:
                    self.status(
                        "ready" if self.worker else "error", "Ready" if self.worker else "Model stopped", epoch
                    )

    def decide(self, epoch, text, question, answers):
        with self.lock:
            if self.active_mode != "decision" or self.worker is None:
                raise DesktopError("Choose Decision mode and run a model before evaluating a decision.")
            worker, codec, template, vocab_size = self.worker, self.codec, self.template, self.vocab_size
        try:
            reply = decide(worker, codec, template, text, question, answers, vocab_size)
        except DecisionInputError as error:
            raise DesktopError(str(error)) from None
        with self.lock:
            if epoch != self.epoch:
                return
            self.emit("decision", epoch, **reply)
            self.status("ready", "Ready", epoch)

    def monitor(self):
        with self.lock:
            if self.worker is None or self.job is not None or self.worker.process.poll() is None:
                return
        self.close_worker()
        self.emit("error", message="The model stopped unexpectedly. Choose Run to restart it.")
        self.status("error", "Model stopped")

    def dispatch(self, payload):
        if not isinstance(payload, dict) or not isinstance(payload.get("command"), str):
            raise DesktopError("Invalid app command.")
        command = payload["command"]
        if command == "list":
            self.library()
        elif command == "support":
            self.families()
        elif command == "catalog":
            self.start_catalog(payload.get("family_id"), payload.get("request_id"))
        elif command == "catalog_cancel":
            if not self.request_id(payload.get("request_id")):
                raise DesktopError("Invalid model catalogue request.")
            self.cancel_catalog(payload["request_id"])
        elif command == "install_catalog":
            try:
                self.install_catalog(payload)
            except DesktopError as error:
                self.emit("error", message=str(error), request_id=payload.get("support_request_id"))
        elif command == "remove_model":
            identifier, action, request_id = payload.get("model_id"), payload.get("action"), payload.get("request_id")
            if (
                not isinstance(identifier, str)
                or action not in {"trash", "unregister"}
                or not self.request_id(request_id)
            ):
                raise DesktopError("Choose a model and removal action.")
            self.start(
                self.remove_saved_model,
                "removing",
                "Removing model…",
                identifier,
                action,
                request_id,
                request_id=request_id,
            )
        elif command in ("stop", "quit"):
            self.stop()
            return command != "quit"
        elif command == "run" and isinstance(payload.get("model_id"), str):
            mode = payload.get("mode", "chat")
            if mode not in ("chat", "decision"):
                raise DesktopError("Choose Chat or Decision mode.")
            self.stop(quiet=True)
            self.start(self.run, "loading", "Loading model…", payload["model_id"], mode)
        elif command == "install" and isinstance(payload.get("source"), str) and isinstance(payload.get("kind"), str):
            artifact_mode = payload.get("artifact_mode")
            if artifact_mode not in (None, "text", "vision-language"):
                raise DesktopError("Choose text or vision-language conversion.")
            if "support_mode" in payload or "support_request_id" in payload:
                self.start(
                    self.install,
                    "installing",
                    "Checking model support…",
                    payload["kind"],
                    payload["source"].strip(),
                    artifact_mode,
                    payload.get("support_mode"),
                    payload.get("support_request_id"),
                )
            elif artifact_mode is None:
                self.start(self.install, "installing", "Installing model…", payload["kind"], payload["source"].strip())
            else:
                self.start(
                    self.install,
                    "installing",
                    "Installing model…",
                    payload["kind"],
                    payload["source"].strip(),
                    artifact_mode,
                )
        elif (
            command == "generate"
            and isinstance(payload.get("text"), str)
            and (payload["text"].strip() or "image" in payload)
        ):
            if set(payload) not in ({"command", "text"}, {"command", "text", "image"}):
                raise DesktopError("Invalid app command.")
            if "image" in payload:
                if not isinstance(payload["image"], dict):
                    raise DesktopError("Invalid image attachment.")
                self.start(self.generate, "generating", "Generating…", payload["text"], payload["image"])
            else:
                self.start(self.generate, "generating", "Generating…", payload["text"])
        elif command == "decide":
            try:
                answers = validate_inputs(payload.get("text"), payload.get("question"), payload.get("answers"))
            except DecisionInputError as error:
                raise DesktopError(str(error)) from None
            self.start(self.decide, "deciding", "Evaluating decision…", payload["text"], payload["question"], answers)
        elif command == "reset":
            with self.lock:
                if self.job is not None:
                    raise DesktopError("Wait for the current operation or choose Stop.")
                self.reset()
        elif command == "configure":
            self.configure(payload)
        else:
            raise DesktopError("Invalid app command.")
        return True


def serve(desktop):
    desktop.library()
    desktop.families()
    desktop.status("idle", "Choose a model to run")
    desktop.start_api()
    pending = b""
    discarding = False
    while True:
        if desktop.output_failed:
            return
        desktop.monitor()
        if not select.select([sys.stdin], [], [], 0.25)[0]:
            continue
        chunk = os.read(sys.stdin.fileno(), 65536)
        if not chunk:
            return
        pending += chunk
        while b"\n" in pending:
            line, pending = pending.split(b"\n", 1)
            if discarding:
                discarding = False
                continue
            try:
                if len(line) > MAX_COMMAND_BYTES:
                    raise DesktopError("App command is too large.")
                if not desktop.dispatch(json.loads(line)):
                    return
            except DesktopError as error:
                desktop.emit("error", message=str(error))
            except Exception:
                desktop.emit("error", message="Invalid app command.")
        if len(pending) > MAX_COMMAND_BYTES:
            pending = b""
            if not discarding:
                desktop.emit("error", message="App command is too large.")
            discarding = True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--score-worker", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--attachment-root", type=Path)
    args = parser.parse_args(argv)
    desktop = None

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        desktop = Desktop(
            args.worker, args.data_dir, score_executable=args.score_worker, attachment_root=args.attachment_root
        )
        serve(desktop)
        return 0
    except (KeyboardInterrupt, BrokenPipeError):
        return 0
    except BaseException:
        try:
            print(
                json.dumps(
                    {"event": "error", "message": "Cannot open Stark's model library. Check the app installation."}
                ),
                flush=True,
            )
        except BaseException:
            pass
        return 1
    finally:
        if desktop is not None:
            with desktop.lock:
                desktop.api_closing = True
            # Every resource gets a cleanup attempt even if an earlier one fails.
            # Exceptions here must not bypass main's content-free error boundary.
            for cleanup in (
                lambda: desktop.stop(quiet=True),
                lambda: desktop.api.close() if desktop.api is not None else None,
                lambda: desktop.attachments.close() if desktop.attachments is not None else None,
            ):
                try:
                    cleanup()
                except BaseException:
                    desktop.emit("error", message="Cannot finish cleanup. Quit and reopen Stark.")


if __name__ == "__main__":
    raise SystemExit(main())
