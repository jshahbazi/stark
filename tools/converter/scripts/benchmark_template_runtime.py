"""Compare disposable renderers on public templates and synthetic input only."""

import argparse
import hashlib
import importlib.util
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

from stark_convert.decision import SYSTEM
from stark_convert.template_runtime import IsolatedTemplate

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def baseline(project):
    path = project / "stark_convert/template_runtime.py"
    spec = importlib.util.spec_from_file_location("_baseline_template_runtime", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    python = project / ".venv/bin/python"

    def spawn(command, **kwargs):
        return subprocess.Popen([str(python), *command[1:]], **kwargs)

    # Preserve the baseline's -I -B -m launch and its own locked environment.
    module.Popen = spawn
    return module.IsolatedTemplate


def summary(values):
    return {
        "samples": len(values),
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def measure(functions, blocks):
    for _ in range(3):
        for function in functions.values():
            function()
    timings = {name: [] for name in functions}
    names = list(functions)
    for _ in range(blocks):
        for name in [*names, *reversed(names)]:
            started = time.perf_counter()
            functions[name]()
            timings[name].append((time.perf_counter() - started) * 1000)
    return {name: summary(values) for name, values in timings.items()}


def run(options):
    fixture = PROJECT / "tests/fixtures/qwen3_4b_chat_template.jinja"
    source = fixture.read_text()
    previous_class = baseline(options.baseline_project.resolve())
    previous, native = previous_class(source), IsolatedTemplate(source)
    cases = {}
    for name, count, length in [("short", 2, 64), ("history_64k", 32, 2048)]:
        arguments = {
            "messages": [
                {"role": "user" if index % 2 == 0 else "assistant", "content": "x" * length} for index in range(count)
            ],
            "tools": None,
            "add_generation_prompt": True,
            "enable_thinking": False,
            "bos_token": "",
            "eos_token": "",
        }
        expected = previous.render(**arguments)

        def render(renderer, arguments=arguments, expected=expected):
            if renderer.render(**arguments) != expected:
                raise RuntimeError("render_mismatch")

        cases[name] = {
            "message_content_bytes": count * length,
            **measure({"python": lambda: render(previous), "native": lambda: render(native)}, options.blocks),
        }
    payload = {
        "text": "x" * 64,
        "question": "synthetic",
        "answers": [{"letter": "A", "label": "synthetic-a"}, {"letter": "B", "label": "synthetic-b"}],
    }
    arguments = {
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(payload)}],
        "tools": None,
        "enable_thinking": False,
        "bos_token": "",
        "eos_token": "",
    }

    def pair(renderer):
        return tuple(renderer.render(**arguments, add_generation_prompt=flag) for flag in (False, True))

    expected_pair = pair(previous)

    def checked_pair(renderer):
        if pair(renderer) != expected_pair:
            raise RuntimeError("render_mismatch")

    cases["decision_pair"] = measure(
        {"python": lambda: checked_pair(previous), "native": lambda: checked_pair(native)}, options.blocks
    )
    cases["construction"] = measure(
        {"python": lambda: previous_class(source), "native": lambda: IsolatedTemplate(source)}, options.blocks
    )
    inputs = [
        fixture,
        PROJECT / "stark_convert/template_runtime.py",
        Path(__file__),
        ROOT / "Cargo.lock",
        *sorted((ROOT / "crates/stark-template").rglob("*.rs")),
    ]
    return {
        "status": "ok",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "warmup_pairs": 3,
        "abba_blocks": options.blocks,
        "all_outputs_match": True,
        "fixture_sha256": digest(fixture),
        "baseline_runtime_sha256": digest(options.baseline_project / "stark_convert/template_runtime.py"),
        "native_worker_sha256": digest(native._executable),
        "source_sha256": {path.relative_to(ROOT).as_posix(): digest(path) for path in inputs},
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-project", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=10)
    options = parser.parse_args()
    if not 1 <= options.blocks <= 100:
        parser.error("blocks must be between 1 and 100")
    try:
        result = run(options)
    except Exception:
        print('{"status":"error","error":"template_benchmark_failed"}')
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
