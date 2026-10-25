"""Private JSON-lines controller for Stark's native menu bar app.

Only model metadata is persisted. Conversation content stays in memory and pipes.
"""

import argparse
import json
import math
import os
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
from .chat import ChatCancelled, ChatError, Worker, load_template
from .decision import DecisionInputError, ScoreWorker, decide, validate_inputs
from .local_api import APIError, LocalAPI
from .registry import get_model
from .sources import huggingface_repository

MAX_COMMAND_BYTES = 1024 * 1024
MAX_PROGRESS_BYTES = 4096
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


def model_valid(path):
    if not path.is_dir() or not all((path / name).is_file() for name in MODEL_FILES):
        return False
    try:
        config = json.loads((path / "config.json").read_text())
        if not isinstance(config.get("decoder_config"), dict):
            return False
        load_template(path)
        return True
    except Exception:
        return False


def metric(reply, key):
    value = reply.get(key)
    valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
    return value if valid else None


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
    ``stop``, ``quit``, ``generate``, ``decide``, ``reset``, ``configure``) and receives
    one JSON event per line on ``output`` (``models``, ``status``, ``progress``,
    ``installed``, ``ready``, ``configured``, ``reply``, ``decision``, ``connection``,
    ``error``). One worker is loaded at a time, under ``self.lock``. ``self.epoch``
    increases whenever the model is stopped or replaced; commands, queued API
    requests and in-flight replies carry the epoch they started under and are
    discarded when it no longer matches, so a late reply from a previous worker
    can never reach the app or a client. App chat turns and API completions share
    one FIFO queue (``api_queue``) in front of the single worker. See
    docs/TECHNICAL_OVERVIEW.md for the control protocol and docs/LOCAL_API.md for
    the HTTP contract.
    """

    def __init__(self, executable, data_dir, output=None, timeout=300, score_executable=None):
        self.executable = executable
        self.score_executable = score_executable or Path(executable).with_name("stark-score")
        self.data_dir = data_dir.expanduser().absolute()
        self.output = output or sys.stdout
        self.timeout = timeout
        self.lock = threading.RLock()
        self.write_lock = threading.Lock()
        self.output_failed = False
        self.epoch = 0
        self.job = None
        self.worker = None
        self.converter = None
        self.install_epoch = None
        self.install_progress = {}
        self.active_model = None
        self.active_mode = None
        self.vocab_size = None
        self.template = None
        self.codec = None
        self.messages = []
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

    def emit(self, event, epoch=None, **fields):
        with self.lock:
            if epoch is not None and epoch != self.epoch:
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
        self.emit("models", models=[
            {**{k: m[k] for k in ("id", "name", "path")}, "api_id": api_model_id(m)} for m in self.models
        ])

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
            self.codec = None
            self.template = None
        if worker is not None:
            terminate(worker.process)
            worker.close()

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
            self.progress(epoch, "download", "running" if kind == "huggingface" else "skipped")
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

    def start(self, function, state, message, *args):
        with self.lock:
            if self.job is not None and self.job.is_alive():
                self.emit("error", message="Wait for the current operation or choose Stop.")
                return
            epoch = self.epoch
            installing = state == "installing"
            if installing:
                self.begin_install_progress(epoch, args[0])
            self.status(state, message, epoch)

            def execute():
                try:
                    function(epoch, *args)
                except DesktopError as error:
                    if epoch == self.epoch:
                        if installing:
                            self.fail_install_progress(epoch)
                        self.emit("error", epoch, message=str(error))
                        self.status(
                            "ready" if self.worker else "error", "Ready" if self.worker else "Operation failed", epoch
                        )
                except BaseException:
                    if epoch == self.epoch:
                        if installing:
                            self.fail_install_progress(epoch)
                        self.close_worker()
                        self.emit(
                            "error", epoch, message="Unable to complete the operation. Check the model and try again."
                        )
                        self.status("error", "Operation failed", epoch)
                finally:
                    with self.lock:
                        if installing and self.install_epoch == epoch:
                            self.install_epoch = None
                        if self.job is threading.current_thread():
                            self.job = None

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

    def install(self, epoch, kind, source):
        if kind == "local":
            path = Path(source).expanduser().resolve()
            if not path.is_dir():
                raise DesktopError("Choose a complete local model folder.")
            source_key = "local:" + str(path)
            name = path.name
            converted = model_valid(path)
            try:
                is_stark = "decoder_config" in json.loads((path / "config.json").read_text())
            except Exception:
                is_stark = False
            if is_stark and not converted:
                raise DesktopError("This Stark folder is incomplete or has no supported chat template.")
        elif kind == "huggingface":
            try:
                source = huggingface_repository(source)
            except ValueError:
                raise DesktopError("Enter a model name or a https://huggingface.co/owner/model URL.") from None
            source_key, name, converted = "huggingface:" + source, source.split("/")[-1], False
        else:
            raise DesktopError("Choose a local folder or Hugging Face repository.")
        with self.lock:
            if epoch != self.epoch:
                return
            previous = next((m for m in self.models if m["source"] == source_key), None)
            if previous is not None:
                if not model_valid(Path(previous["path"])):
                    raise DesktopError("The installed model folder is missing or incomplete.")
                self.finish_install_progress(epoch, skipped=True)
                self.emit("installed", epoch, model={k: previous[k] for k in ("id", "name", "path")})
                self.status("ready" if self.worker else "idle", "Model already installed", epoch)
                return
        identifier = uuid.uuid4().hex
        staging = self.model_dir / (".install-" + identifier)
        managed = None
        try:
            if not converted:
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
                ]
                if kind == "local" or get_model(source) is None:
                    command.extend(["--metadata-profile", "hf-generic"])
                if kind == "local":
                    command.append("--offline")
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
            model = {"id": identifier, "name": name, "path": str(path), "source": source_key}
            with self.lock:
                if epoch != self.epoch:
                    return
                self.models.append(model)
                try:
                    self.persist()
                except BaseException:
                    self.models.remove(model)
                    raise
                managed = None  # The registry now owns the managed folder.
                self.finish_install_progress(epoch, skipped=converted)
                self.library()
                self.emit("installed", epoch, model={k: model[k] for k in ("id", "name", "path")})
                self.status("ready" if self.worker else "idle", "Model installed", epoch)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            if managed is not None:
                shutil.rmtree(managed, ignore_errors=True)

    def run(self, epoch, identifier, mode="chat"):
        model = next((m for m in self.models if m["id"] == identifier), None)
        if model is None or not model_valid(Path(model["path"])):
            raise DesktopError("Select an installed model whose folder is still available.")
        codec, template = load_template(Path(model["path"]))
        with self.lock:
            if epoch != self.epoch:
                return
            worker = (
                ScoreWorker(self.score_executable, Path(model["path"]), self.timeout)
                if mode == "decision"
                else Worker(self.executable, Path(model["path"]), self.timeout)
            )
            self.worker = worker
        ready = worker.read()
        if mode == "decision":
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
                self.messages = []
                self.emit("ready", epoch, model_id=identifier, mode=mode)
                self.status("ready", "Ready", epoch)
            return
        if (
            ready.get("ready") is not True
            or ready.get("mode") != "text-generation-v1"
            or not isinstance(ready.get("max_context_tokens"), int)
            or ready["max_context_tokens"] < 2
            or not isinstance(ready.get("max_new_tokens"), int)
            or ready["max_new_tokens"] < 1
        ):
            self.close_worker()
            raise DesktopError("Stark could not load this model. Check model support and available memory.")
        with self.lock:
            if epoch != self.epoch:
                return
            worker.streaming = ready.get("streaming") == "text-delta-v1"
            worker.stop_sequences = ready.get("stop_sequences") is True
            self.codec, self.template = codec, template
            self.context_limit = min(16384, ready["max_context_tokens"])
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

    def generate(self, epoch, text):
        with self.lock:
            if self.active_mode != "chat":
                raise DesktopError("Choose Chat mode and run a model before sending a message.")
            candidate = [*self.messages, {"role": "user", "content": text}]
        reply = self.infer(candidate, self.maximum, epoch=epoch, thinking=self.thinking)
        with self.lock:
            if epoch != self.epoch:
                return
            self.messages = [*candidate, {"role": "assistant", "content": reply["text"]}]
            self.emit(
                "reply",
                epoch,
                text=reply["text"],
                input_tokens=metric(reply, "input_tokens"),
                output_tokens=metric(reply, "output_tokens"),
                finish_reason=reply.get("finish_reason"),
                **{
                    key: metric(reply, key)
                    for key in (
                        "input_tokens_per_second",
                        "output_tokens_per_second",
                        "prefill_seconds",
                        "decode_seconds",
                        "generation_seconds",
                    )
                },
            )
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
    ):
        with self.lock:
            if epoch is not None and epoch != self.epoch:
                raise APIError(503, "Model changed or stopped before this request started.", "model_changed")
            worker, template, codec = self.worker, self.template, self.codec
            context = context or self.context
        if worker is None or template is None:
            raise APIError(503, "Run a model before sending a prompt.", "model_not_loaded")
        if on_delta is not None and not worker.streaming:
            raise APIError(
                503, "This worker needs an update for streaming. Rebuild or reinstall Stark.", "streaming_unavailable"
            )
        if stop and not worker.stop_sequences:
            raise APIError(503, "This worker needs an update for stop sequences.", "stop_sequences_unavailable")
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
        tokens = worker.request(
            {"operation": "tokenize", "text": prompt, "add_special_tokens": False}, cancelled=cancelled
        )
        count = tokens.get("token_count")
        if not isinstance(count, int):
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
                "capabilities": {
                    "chat_completions": True,
                    "streaming": True,
                    "sampling": ["temperature", "top_p", "top_k", "min_p", "seed"],
                    "stop_sequences": bool(self.active_mode == "chat" and self.worker and self.worker.stop_sequences),
                    "cancellation": bool(self.active_mode == "chat" and self.worker and self.worker.cancellation),
                    "tool_calling": False,
                    "structured_output": False,
                    "images": False,
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
        overflow (400) or a worker fault (503, after which the worker is closed
        and the app shows the model as stopped).
        """
        cancelled = cancelled or threading.Event()
        with self.lock:
            request = parse_chat_request(payload, self.api_maximum, self.api_vocab)
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
            epoch, identifier = self.epoch, self.active_model
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
            )
            with self.lock:
                if epoch != self.epoch:
                    raise APIError(503, "Request stopped. Run a model and retry.", "request_stopped")
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
                if epoch == self.epoch:
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
            self.start(self.install, "installing", "Installing model…", payload["kind"], payload["source"].strip())
        elif command == "generate" and isinstance(payload.get("text"), str) and payload["text"].strip():
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
    args = parser.parse_args(argv)
    desktop = None

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        desktop = Desktop(args.worker, args.data_dir, score_executable=args.score_worker)
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
            desktop.stop(quiet=True)
            if desktop.api is not None:
                desktop.api.close()


if __name__ == "__main__":
    raise SystemExit(main())
