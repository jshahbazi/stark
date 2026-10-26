"""Terminal chat client. Conversation content stays in memory and private pipes."""

import argparse
import json
import os
import select
import subprocess
import sys
import threading
import time
from pathlib import Path

from jinja2.sandbox import ImmutableSandboxedEnvironment


class ChatError(Exception):
    """Content-free user-facing failure."""


class ChatCancelled(ChatError):
    """A cancelled request whose terminal response has been consumed."""


class ChatRequestTooLarge(ChatError):
    """An oversized request rejected before writing any bytes to the worker."""


def terminal_text(text):
    # Model output must not execute terminal control sequences.
    return "".join(c for c in text if c in "\n\t" or (c.isprintable() and c != "\x1b"))


def load_template(model):
    try:
        codec = json.loads((model / "config.json").read_text())["token_codec_config"]
        source = codec["prompt_template"]
        if codec.get("type") != "ChatCodecConfig" or not isinstance(source, str) or not source:
            raise ValueError
        environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)

        def raise_exception(_message):
            raise ChatError("The model's chat template rejected this conversation.")

        environment.globals["raise_exception"] = raise_exception
        template = environment.from_string(source)
        return codec, template
    except (OSError, ValueError, KeyError, TypeError):
        raise ChatError("Cannot read chat metadata; use a converted language model with a chat template.") from None


class Worker:
    def __init__(self, executable, model, timeout):
        self.timeout = timeout
        self.pending = b""
        self.cancellation = False
        self.request_sequence = 0
        self.cancelled_id = None
        read_fd, self.cancel_fd = os.pipe()
        os.set_blocking(self.cancel_fd, False)
        try:
            self.process = subprocess.Popen(
                [str(executable), "--model-path", str(model)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
                start_new_session=True,
                pass_fds=(read_fd,),
                env=dict(os.environ, STARK_CANCEL_FD=str(read_fd)),
            )
        except BaseException:
            os.close(self.cancel_fd)
            raise
        finally:
            os.close(read_fd)

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
        deadline = time.monotonic() + self.timeout
        while b"\n" not in self.pending:
            remaining = deadline - time.monotonic()
            if cancelled is not None and cancelled.is_set() and request_id is not None:
                self.cancel(request_id)
            if remaining <= 0:
                raise ChatError("Worker timed out; chat closed.")
            if not select.select(
                [self.process.stdout], [], [], min(remaining, 0.1) if cancelled is not None else remaining
            )[0]:
                continue
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                raise ChatError("Worker stopped; chat closed.")
            self.pending += chunk
            if len(self.pending) > 16 * 1024 * 1024:
                raise ChatError("Worker reply exceeded the size limit; chat closed.")
        line, self.pending = self.pending.split(b"\n", 1)
        try:
            reply = json.loads(line)
            if not isinstance(reply, dict):
                raise ValueError
            if reply.get("ready") is True:
                self.cancellation = reply.get("cancellation") == "pipe-v1"
            return reply
        except (ValueError, UnicodeError):
            raise ChatError("Invalid worker reply; chat closed.") from None

    def request(self, payload, on_delta=None, cancelled=None):
        if cancelled is not None and cancelled.is_set():
            raise ChatCancelled("Request cancelled.")
        request_id = 0 if payload.get("operation") == "generate" else None
        if payload.get("operation") == "generate" and getattr(self, "cancellation", False):
            self.request_sequence += 1
            request_id = self.request_sequence
            payload = {**payload, "request_id": request_id}
        data = json.dumps(payload, ensure_ascii=False).encode() + b"\n"
        if len(data) > 1024 * 1024:
            raise ChatRequestTooLarge("Conversation exceeds the worker request limit; use /reset.")
        # FileIO writes can be partial for large prompts.
        view = memoryview(data)
        while view:
            count = self.process.stdin.write(view)
            if not count:
                raise ChatError("Cannot send request to worker.")
            view = view[count:]
        chunks = []
        control = cancelled
        while True:
            if control is not None and control.is_set() and request_id is not None:
                self.cancel(request_id)
            reply = self.read(control, request_id) if control is not None else self.read()
            if "event" not in reply:
                if (control is not None and control.is_set()) or reply.get("error") == "generation_cancelled":
                    raise ChatCancelled("Request cancelled.")
                if on_delta is not None and "error" not in reply and reply.get("text") != "".join(chunks):
                    raise ChatError("Invalid worker stream; restart the model.")
                return reply
            if (
                on_delta is None
                or set(reply) != {"event", "text"}
                or reply["event"] != "delta"
                or not isinstance(reply["text"], str)
            ):
                raise ChatError("Invalid worker stream; restart the model.")
            chunks.append(reply["text"])
            if control is None or not control.is_set():
                try:
                    on_delta(reply["text"])
                except OSError:
                    control = control or threading.Event()
                    control.set()

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process.stdin.close()
        self.process.stdout.close()
        cancel_fd = getattr(self, "cancel_fd", None)
        if cancel_fd is not None:
            self.cancel_fd = None
            os.close(cancel_fd)


def run_chat(args, worker):
    codec, template = load_template(args.model_path)
    ready = worker.read()
    if ready.get("ready") is not True or ready.get("mode") != "text-generation-v1":
        raise ChatError("Generation worker could not load this model.")
    context = args.context_tokens or min(8192, ready["max_context_tokens"])
    if context > ready["max_context_tokens"] or args.max_new_tokens > ready["max_new_tokens"]:
        raise ChatError("Requested token limits exceed this worker's limits.")
    system = args.system if args.system is not None else codec.get("default_system_prompt")
    initial = [{"role": "system", "content": system}] if system else []
    messages = list(initial)
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
            print("Conversation cleared.")
            continue
        if text.strip() == "/help":
            print("Enter one message per line. Commands: /reset, /exit, /quit, /help.")
            continue
        candidate = [*messages, {"role": "user", "content": text}]
        try:
            prompt = template.render(
                messages=candidate,
                tools=None,
                add_generation_prompt=True,
                enable_thinking=args.thinking,
                bos_token=codec.get("bos_token") or "",
                eos_token=codec.get("eos_token") or "",
            )
            tokens = worker.request({"operation": "tokenize", "text": prompt, "add_special_tokens": False})
            count = tokens.get("token_count")
            if not isinstance(count, int):
                raise ChatError("Worker could not tokenize the conversation.")
            if count + args.max_new_tokens > context:
                print("Conversation exceeds the context budget; use /reset or a larger --context-tokens.")
                continue
            reply = worker.request(
                {
                    "operation": "generate",
                    "rendered_prompt": prompt,
                    "context_tokens": context,
                    "max_new_tokens": args.max_new_tokens,
                    "sampling": {"mode": "greedy"},
                }
            )
            if not isinstance(reply.get("text"), str) or "error" in reply:
                raise ChatError("Generation failed; chat closed.")
        except ChatError as error:
            if str(error).startswith("Conversation exceeds"):
                print(str(error))
                continue
            raise
        print("Stark> " + terminal_text(reply["text"]) + "\n")
        messages = [*candidate, {"role": "assistant", "content": reply["text"]}]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, prog="stark chat")
    parser.add_argument("--model-path", type=Path, required=True, help="Converted Stark model directory")
    parser.add_argument("--worker", type=Path, default=Path("stark-generate"), help=argparse.SUPPRESS)
    parser.add_argument("--system", help="System message (default: model metadata)")
    parser.add_argument("--context-tokens", type=int, help="Context budget (default: up to 8192)")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Reply budget (default: 512)")
    parser.add_argument("--thinking", action="store_true", help="Enable reasoning for templates that support it")
    parser.add_argument("--timeout", type=float, default=300, help="Worker reply timeout in seconds (default: 300)")
    args = parser.parse_args(argv)
    if (
        (args.context_tokens is not None and args.context_tokens < 1)
        or args.max_new_tokens < 1
        or not 0 < args.timeout < float("inf")
    ):
        parser.error("token limits and timeout must be positive and finite")
    worker = None
    try:
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
