"""Private JSON-lines controller for Stark's native menu bar app.

Only model metadata is persisted. Conversation content stays in memory and pipes.
"""

import argparse
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
from .chat import ChatCancelled, ChatError, Worker, load_template
from .local_api import APIError, LocalAPI
from .registry import get_model

MAX_COMMAND_BYTES = 1024 * 1024
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


def api_model_id(model):
    """Expose source names while preserving stable private library identifiers."""
    source = model["source"]
    return source.removeprefix("huggingface:") if source.startswith("huggingface:") else model["name"]


class Desktop:
    def __init__(self, executable, data_dir, output=None, timeout=300):
        self.executable = executable
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
        self.active_model = None
        self.template = None
        self.codec = None
        self.messages = []
        self.context = 8192
        self.maximum = 512
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
            self.messages = []
            self.codec = None
            self.template = None
        if worker is not None:
            terminate(worker.process)
            worker.close()

    def stop(self, quiet=False):
        with self.lock:
            self.epoch += 1
            self.api_condition.notify_all()
            converter, job = self.converter, self.job
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
            self.status(state, message, epoch)

            def execute():
                try:
                    function(epoch, *args)
                except DesktopError as error:
                    if epoch == self.epoch:
                        self.emit("error", epoch, message=str(error))
                        self.status(
                            "ready" if self.worker else "error", "Ready" if self.worker else "Operation failed", epoch
                        )
                except BaseException:
                    if epoch == self.epoch:
                        self.close_worker()
                        self.emit(
                            "error", epoch, message="Unable to complete the operation. Check the model and try again."
                        )
                        self.status("error", "Operation failed", epoch)
                finally:
                    with self.lock:
                        if self.job is threading.current_thread():
                            self.job = None

            self.job = threading.Thread(target=execute, name="stark-desktop-operation", daemon=True)
            self.job.start()

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
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", source):
                raise DesktopError("Enter a Hugging Face repository as owner/model.")
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
                with self.lock:
                    if epoch != self.epoch:
                        return
                    process = subprocess.Popen(
                        command,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        env=environment,
                        start_new_session=True,
                    )
                    self.converter = process
                result = process.wait()
                with self.lock:
                    if self.converter is process:
                        self.converter = None
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
                self.library()
                self.emit("installed", epoch, model={k: model[k] for k in ("id", "name", "path")})
                self.status("ready" if self.worker else "idle", "Model installed", epoch)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            if managed is not None:
                shutil.rmtree(managed, ignore_errors=True)

    def run(self, epoch, identifier):
        model = next((m for m in self.models if m["id"] == identifier), None)
        if model is None or not model_valid(Path(model["path"])):
            raise DesktopError("Select an installed model whose folder is still available.")
        codec, template = load_template(Path(model["path"]))
        with self.lock:
            if epoch != self.epoch:
                return
            worker = Worker(self.executable, Path(model["path"]), self.timeout)
            self.worker = worker
        ready = worker.read()
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
            self.context = min(8192, ready["max_context_tokens"])
            self.maximum = min(512, ready["max_new_tokens"], self.context - 1)
            self.api_context = min(16384, ready["max_context_tokens"])
            self.api_maximum = min(4096, ready["max_new_tokens"], self.api_context - 1)
            self.api_vocab = ready.get("vocab_size", 0)
            self.active_model = identifier
            self.reset()
            self.emit("ready", epoch, model_id=identifier,
                      max_context_tokens=self.api_context, max_new_tokens=self.api_maximum)
            self.status("ready", "Ready", epoch)

    def reset(self):
        with self.lock:
            system = (self.codec or {}).get("default_system_prompt")
            self.messages = [{"role": "system", "content": system}] if system else []

    def generate(self, epoch, text):
        with self.lock:
            candidate = [*self.messages, {"role": "user", "content": text}]
        reply = self.infer(candidate, self.maximum, epoch=epoch)
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
                enable_thinking=False,
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
            if self.active_model is None:
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
                "model_loaded": self.active_model is not None,
                "busy": self.job is not None,
                "queued_requests": len(self.api_queue),
                "queue_limit": self.api_queue_limit,
                "queue_timeout_seconds": self.api_queue_timeout,
                "models": self.api_models(),
                "capabilities": {
                    "chat_completions": True,
                    "streaming": True,
                    "sampling": ["temperature", "top_p", "top_k", "min_p", "seed"],
                    "stop_sequences": bool(self.worker and self.worker.stop_sequences),
                    "cancellation": bool(self.worker and self.worker.cancellation),
                    "tool_calling": False,
                    "structured_output": False,
                    "images": False,
                    "embeddings": False,
                    "responses": False,
                },
            }

    def api_complete(self, payload, on_delta=None, cancelled=None):
        cancelled = cancelled or threading.Event()
        with self.lock:
            request = parse_chat_request(payload, self.api_maximum, self.api_vocab)
            maximum = request["maximum"]
            messages = request["messages"]
            if self.api_closing:
                raise APIError(503, "Stark is shutting down.", "server_stopped")
            if self.active_model is None or self.worker is None:
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
            self.stop(quiet=True)
            self.start(self.run, "loading", "Loading model…", payload["model_id"])
        elif command == "install" and isinstance(payload.get("source"), str) and isinstance(payload.get("kind"), str):
            self.start(self.install, "installing", "Installing model…", payload["kind"], payload["source"].strip())
        elif command == "generate" and isinstance(payload.get("text"), str) and payload["text"].strip():
            self.start(self.generate, "generating", "Generating…", payload["text"])
        elif command == "reset":
            with self.lock:
                if self.job is not None:
                    raise DesktopError("Wait for the current operation or choose Stop.")
                self.reset()
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
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    desktop = None

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        desktop = Desktop(args.worker, args.data_dir)
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
