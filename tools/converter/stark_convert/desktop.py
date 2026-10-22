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
import uuid
from pathlib import Path

from .chat import Worker, load_template
from .registry import get_model

MAX_COMMAND_BYTES = 1024 * 1024
MODEL_FILES = ("config.json", "tokenizer.json", "model.safetensors")


class DesktopError(Exception):
    """A fixed, content-free user-facing failure."""


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
        self.emit("models", models=[{k: m[k] for k in ("id", "name", "path")} for m in self.models])

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
            self.codec, self.template = codec, template
            self.context = min(8192, ready["max_context_tokens"])
            self.maximum = min(512, ready["max_new_tokens"], self.context - 1)
            self.active_model = identifier
            self.reset()
            self.emit("ready", epoch, model_id=identifier)
            self.status("ready", "Ready", epoch)

    def reset(self):
        with self.lock:
            system = (self.codec or {}).get("default_system_prompt")
            self.messages = [{"role": "system", "content": system}] if system else []

    def generate(self, epoch, text):
        with self.lock:
            worker, template, codec = self.worker, self.template, self.codec
            candidate = [*self.messages, {"role": "user", "content": text}]
        if worker is None or template is None:
            raise DesktopError("Run a model before sending a prompt.")
        prompt = template.render(
            messages=candidate,
            tools=None,
            add_generation_prompt=True,
            enable_thinking=False,
            bos_token=codec.get("bos_token") or "",
            eos_token=codec.get("eos_token") or "",
        )
        tokens = worker.request({"operation": "tokenize", "text": prompt, "add_special_tokens": False})
        count = tokens.get("token_count")
        if not isinstance(count, int):
            raise ValueError
        if count + self.maximum > self.context:
            raise DesktopError("Conversation is full. Start a new conversation and try again.")
        reply = worker.request(
            {
                "operation": "generate",
                "rendered_prompt": prompt,
                "context_tokens": self.context,
                "max_new_tokens": self.maximum,
                "sampling": {"mode": "greedy"},
            }
        )
        if not isinstance(reply.get("text"), str) or "error" in reply:
            raise ValueError
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
            desktop.stop(quiet=True)


if __name__ == "__main__":
    raise SystemExit(main())
