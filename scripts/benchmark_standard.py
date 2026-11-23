"""Stark standard benchmark: one protocol for every model Stark supports.

The protocol is defined in docs/BENCHMARK.md. This tool runs it: server-level
timing through the real worker, against a reference engine, with one client,
one scenario set, one set of validity checks and aggregate-only receipts.
Everything model-specific (artifacts, context limits, chat adapter, reference
engines and their launch arguments) comes from a declarative profile in
scripts/benchmark_profiles/<key>.toml. No model's artifacts, limits or launch
arguments are in this file.

Scenarios (client: llmprobe 0.6.18, greedy, 192 output tokens):

- initial coding, sustained coding and novel prose decode;
- decode on the context ladder (about 512, 4K, 8K and 16K input tokens);
- long-prompt prefill through the client, and an exact 8,800-ID unseen-text
  prefill control with matching input-ID hashes;
- a 16-user burst, 300 tokens each, total tokens over wall time, draft off.
  It is reported and does not count toward a standing, because Stark
  serializes one worker by design.

Arms are (engine build, model artifact) pairs, never configuration flags. A
Stark arm is one `stark-generate` binary serving one artifact the profile
declares: `--stark-binary` is the build under test, `--stark-build NAME=PATH`
adds another build (for example an earlier commit), and `--stark-artifact
NAME[=PATH]` selects artifacts; every build runs every selected artifact. A
reference arm is one pack served by a reference engine (`--reference-binary`,
`--reference-artifact`). Arms whose artifacts share the profile's `match`
class are a matched pair, and only matched pairs compare engines.

Each engine runs its own defaults. The Stark gateway forwards what the client
asked for and nothing else, so the worker's drafting default applies; a
reference engine starts without a draft flag. The burst forces drafting off
on every engine. Environment inherited from the shell is removed before a
worker starts. Any forced setting, diagnostic environment variable or
undeclared artifact is recorded and takes the run or the arm off the
scoreboard.

Every figure states its level. Scenario rows are server-level: timed by the
client over HTTP, or the engine's own phase time read through the server. An
engine-only diagnostic number never enters a scenario row; the quality slot
is where one belongs. Speed without quality is not a result: each Stark arm
has a quality slot, filled by attaching reports from a pluggable probe
(`quality attach`). The result is pooled over the rows of the corpus and
judged pass, fail or not assessed against one bar; an arm without a pass is
speed only and holds no standing.

Receipts are aggregate only: counts, timings, SHA-256 hashes and engine notes.
Prompt text, completions, token vectors and probabilities are never written.
Raw client output is parsed in memory and discarded; its JSON report is
reduced to the benchmark section. The client runs with `--no-save` and a
temporary HOME.

Timing runs refuse to start unless `pmset -g batt` reports AC power, Low Power
Mode is off and no competing model server or compiler is running. They hold
`caffeinate`, take an exclusive lock file (one GPU job at a time) and record
power source, adapter wattage, thermal state, macOS version, hardware model
and sleep events in the run manifest.

Commands, from the repository root (`BENCH` stands for
`uv run --project tools/converter --locked --python 3.13.13 python
scripts/benchmark_standard.py`):

    BENCH profiles
    BENCH plan --profile <key> [arm options]
    BENCH run --profile <key> [arm options] --llmprobe <checkout>
    BENCH summarize <run> [<run> ...]
    BENCH quality attach --run <run> --arm <arm> --report <file> \\
      [--report <file> ...] --candidate-artifact <dir> \\
      --corpus-kind natural-text

`plan` (the same as `run --dry-run`) validates the profile and every path,
then prints exactly what a run would start, in order: each arm's binary,
artifact and matching class, every server session with its launch arguments,
the client command, the planned samples per scenario, whether the run counts
as a standard run, and the host state. It starts no server, worker or client
and needs no GPU.

A failed arm does not stop the others: its failure is recorded in the manifest
under `arm_failures`, the arm is skipped for the rest of the run, the command
exits with status 3, and `summarize` lists it. When a reference server fails
to start, the last lines of its own startup output (before any benchmark
request) are kept in the private run directory. `plan` also checks that every
weight shard a model's `model.safetensors.index.json` names is present. To
fill in one arm later, run it alone with `--arms NAME` and pass both run
directories to `summarize`.

`summarize` joins run directories into one table: medians with ranges, the
ratio of medians against the reference arm, paired per-repeat ratios against
the reference arm and against the matched reference arm, a standing (ahead or
behind only when every block agrees and there are at least three), validity
findings, and the quality verdict of each arm.
It also reads run directories written by the earlier single-model harness.

Prerequisites: `cargo +1.98.1 build --release --locked -p stark-generate` and
`-p stark-template` in this checkout; the artifacts the profile names; the
reference engine binary and packs; an llmprobe 0.6.18 checkout with
node_modules and the aggregate-receipt patch bundled; Node.js 22.19 or newer.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import shutil
import socket
import statistics
import struct
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SCHEMA = "stark-standard-benchmark-v1"
SUMMARY_SCHEMA = "stark-standard-benchmark-summary-v1"
# Run directories written before the harness took profiles carry none. Every
# one of them measured the profile named here.
LEGACY_SCHEMAS = {"stark-inkling-engine-matrix-v2": "inkling-2-mini"}
PROFILE_SCHEMA = "stark-benchmark-profile-v1"
QUALITY_SCHEMA = "stark-benchmark-quality-v1"
REPOSITORY = Path(__file__).resolve().parents[1]
PROFILE_DIRECTORY = Path(__file__).resolve().parent / "benchmark_profiles"
HOST = "127.0.0.1"

# Documented tool defaults. Every location is ignored by Git. Artifact and
# reference-engine locations come from the profile.
DEFAULTS = {
    "stark_binary": ("STARKBENCH_STARK_BINARY", "target/release/stark-generate"),
    "llmprobe": ("STARKBENCH_LLMPROBE", "runtime/llmprobe-0.6.18"),
    "output": ("STARKBENCH_OUTPUT", "runtime/benchmark"),
    "lock_file": ("STARKBENCH_LOCK_FILE", "runtime/model-scoring.lock"),
    "node": ("STARKBENCH_NODE", "node"),
}

ARM_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,31}")
ENV_KEY = re.compile(r"[A-Z_][A-Z0-9_]{0,127}")
# Environment removed before launching a Stark worker, so nothing inherited
# from the shell changes what a build does. --stark-env is applied afterwards.
REMOVED_ENV_PREFIXES = ("STARK_", "UZU_")
CLIENT_ENV_PREFIXES = ("LLMPROBE_", "MLX_SERVE_")
SECRET_ENV = frozenset(
    {"OPENAI_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "ANTHROPIC_API_KEY"}
)

# Context sizing only (the worker's initial context): requests up to the
# profile's minimum are planned at that size, larger ones are rounded up to
# this step and capped at the profile's maximum. Drafting is the worker's
# decision.
CONTEXT_STEP = 1024
# A fixed, non-secret scope string; it only groups prefix-cache reuse.
STARK_CACHE_SCOPE = "e7f2eb4a0632c6db61fd28c88138bb28"
DRAFT_MODES = ("engine", "off")
REASONING_CLIENT_DEFAULT = "client-default"
SESSION_KINDS = ("probe", "prefill", "burst")
# blocks: every repeat is one block of fresh sessions, arm order reversed on
# odd repeats, so each arm's figures are adjacent in time to the others'.
# legacy: the first recorded runs' layout (probe repeats, then one prefill
# session and one burst session per arm, fixed arm order).
LAYOUTS = ("blocks", "legacy")
MIN_STANDARD_REPEATS = 3

PREFILL_METRIC = "vllm:request_prefill_time_seconds_sum"
DECODE_METRIC = "vllm:request_decode_time_seconds_sum"
FINISH_REASONS = frozenset(
    {"length", "stop", "max_tokens", "max_output_tokens", "end_turn"}
)

LLMPROBE_VERSION = "0.6.18"
# The receipt patch's marker and environment names belong to the pinned client
# build. Their names are historical; they are the same for every model.
LLMPROBE_RECEIPT_MARKER = "[inkling-bench-sample]"
# Hashes of the bundle and benchmark source used for the 2026-11-22 run, with
# the aggregate-receipt patch (patch SHA-256 599a9ea7...) applied. A different
# build is allowed but recorded as unpinned.
LLMPROBE_PINNED = {
    "bin/dist/llmprobe.mjs": "5013dcfadcee02e6cdcd9dc210697acc90ad15e0f9c9984de41a1b2c3dcf0276",
    "src/bench/index.ts": "0936b4756578f713b85af025f78a0b00f78dc11e709e4041f929238f18b03600",
}
LLMPROBE_IDENTITY_FILES = (
    "package.json",
    "bin/dist/llmprobe.mjs",
    "src/bench/index.ts",
    "src/bench/corpus.ts",
)
TELEMETRY = re.compile(r"^\[inkling-bench-sample\] (\{.*\})\s*$")
SAMPLE_FIELDS = {
    "label": str,
    "warmup": bool,
    "failed": bool,
    "inputTokens": int,
    "outputTokens": int,
    "cachedInputTokens": int,
    "ttftMs": (int, float),
    "wallMs": (int, float),
    "decodeTokPerSec": (int, float),
    "prefillTokPerSec": (int, float),
    "tokensPerStep": (int, float),
    "streamCoalesced": bool,
    "finishReason": str,
    "inputUtf8Sha256": str,
}
BENCH_FIELDS = (
    "decodeTokPerSec",
    "streamCaveat",
    "decodeLengthNote",
    "samplingNote",
    "runsNote",
    "reasoningNote",
    "ttftMs",
    "prefillTokPerSec",
    "prefillPromptTokens",
    "prefillUnseenTokPerSec",
    "prefillUnseenPromptTokens",
    "speculative",
    "prefixCache",
    "batching",
    "batchingSweep",
    "loadDrift",
    "contextScaling",
)
MLX_PERF = re.compile(
    r" <- (\d+)\+(\d+) tokens streamed(?: \((\d+)ms\)|, \d+ frames)? "
    r"\[([^\[\]]{0,160})\] \[(length|stop|max_tokens|end_turn)\]"
)
MLX_SPEC_FIELDS = frozenset(
    {
        "attempts",
        "accepts",
        "avg_per_round",
        "stage_rounds",
        "copy_rounds",
        "copy_drafted",
        "copy_accepted",
        "group_rounds",
        "per_draft_pct",
        "drafted",
        "depth",
    }
)
NUMBER = r"[0-9]+(?:\.[0-9]+)?"

EXACT_PREFILL_TOKENS = 8800
EXACT_PREFILL_CORPUS_BYTES = 80000
# A protocol constant with a historical name: one seed for every model and
# run, so the unseen text is the same text in every record.
EXACT_PREFILL_SEED = "inkling-m5-max-exact-unseen"
EXACT_PREFILL_CAP = 8
BURST_WIDTH = 16
BURST_CAP = 300
DECODE_CAP = 192
OUTLIER_FACTOR = 3.0
COMPETING_PROCESSES = (
    "stark-generate",
    "stark-score",
    "stark-decide",
    "mlx-serve",
    "llama-server",
    "ollama",
    "lms",
    "vllm",
    "cargo",
    "rustc",
    "swift-build",
    "xcodebuild",
    "metal",
)


class BenchError(RuntimeError):
    """A failure whose message is a fixed code, safe to print and persist."""


# --------------------------------------------------------------------------
# Scenario catalog
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    source: str  # "llmprobe", "prefill" or "burst"
    metric: str
    unit: str
    higher_is_better: bool
    cap: int | None
    per_invocation: str
    description: str
    comparability: str


SCENARIOS = (
    Scenario(
        "decode",
        "Initial coding (llmprobe decode)",
        "llmprobe",
        "decodeTokPerSec",
        "tok/s",
        True,
        DECODE_CAP,
        "1 warmup + K (K=3) measured",
        "One user turn asking for a TypeScript retryWithBackoff function (a short prompt, under "
        "100 rendered tokens), greedy, 192-token cap. Every request carries a unique cache-bust "
        "lead-in, so each measured run starts from a cold prefix.",
        "llmprobe first sends a length probe with ignore_eos and min_tokens=192. An engine that "
        "honours it (MLX-Serve does) is forced to 192 tokens. The Stark gateway rejects the "
        "override (the worker has no EOS override), so Stark runs follow checkpoint EOS; samples "
        "that stop short of 192 are excluded and counted.",
    ),
    Scenario(
        "prefill_repeated",
        "Repeated-sentence prefill (llmprobe prefill)",
        "llmprobe",
        "prefillTokPerSec",
        "tok/s",
        True,
        None,
        "1 warmup + K measured",
        "About 40 KB of one repeated archive sentence (roughly 8.8K tokens; the count depends on "
        "the tokenizer and is recorded) plus a one-word reply instruction, 8-token cap. Rate is "
        "input tokens over client time to first token.",
        "Cache-busted at the front only; the body is the same sentence repeated, which is not "
        "representative of real text.",
    ),
    Scenario(
        "unseen_prefill",
        "Unseen-text prefill (llmprobe)",
        "llmprobe",
        "prefillTokPerSec",
        "tok/s",
        True,
        None,
        "K measured, no warmup",
        "About 40 KB of seeded random everyday words (the token count depends on the tokenizer "
        "and is recorded), 8-token cap. Rate is input tokens over client time to first token.",
        "The token count varies with the tokenizer; the exact 8,800-ID control below is the "
        "matched prefill cell.",
    ),
    Scenario(
        "spec_predictable",
        "Speculation ceiling, echo (llmprobe spec:predictable)",
        "llmprobe",
        "decodeTokPerSec",
        "tok/s",
        True,
        DECODE_CAP,
        "1 warmup + K measured",
        "System message 'text normaliser' plus a fixed passage to echo (a few hundred rendered "
        "tokens), 192-token cap. Only runs that actually echoed are counted.",
        "Draft-friendly output; the cache-bust tag sits on the system message.",
    ),
    Scenario(
        "spec_novel",
        "Novel prose (llmprobe spec:novel)",
        "llmprobe",
        "decodeTokPerSec",
        "tok/s",
        True,
        DECODE_CAP,
        "1 warmup + K measured",
        "One short user turn asking for original stream-of-consciousness prose (a few dozen "
        "rendered tokens), 192-token cap, cold prefix.",
        "Low draft acceptance by design.",
    ),
    Scenario(
        "context_512",
        "Context ladder ~512",
        "llmprobe",
        "decodeTokPerSec",
        "tok/s",
        True,
        DECODE_CAP,
        "1 cold run per rung (llmprobe default)",
        "A fresh cache-busted synthetic TypeScript archive with a planted constant, sized from "
        "the engine's own token counts to about 512 input tokens, then an instruction to write "
        "code that uses the constant. 192-token cap, single turn, cold prefix.",
        "The filler is fitted per engine, so the two engines see slightly different input lengths.",
    ),
    Scenario(
        "context_512_ceiling",
        "Context ladder ~512, counting ceiling",
        "llmprobe",
        "decodeTokPerSec",
        "tok/s",
        True,
        DECODE_CAP,
        "1 run per rung, after the coding run",
        "The same archive as the coding run, then 'count from 1 to 200'. 192-token cap.",
        "Shares the coding run's archive prefix by design, so an engine with prefix reuse "
        "serves most of the prompt from cache. Not a cold number.",
    ),
    Scenario(
        "context_4k",
        "Context ladder ~4K",
        "llmprobe",
        "decodeTokPerSec",
        "tok/s",
        True,
        DECODE_CAP,
        "1 cold run per rung",
        "As ~512 with about 4,020 input tokens.",
        "Filler fitted per engine.",
    ),
    Scenario(
        "context_4k_ceiling",
        "Context ladder ~4K, counting ceiling",
        "llmprobe",
        "decodeTokPerSec",
        "tok/s",
        True,
        DECODE_CAP,
        "1 run per rung",
        "Counting on the ~4K archive.",
        "Prefix shared with the coding run by design.",
    ),
    Scenario(
        "context_8k",
        "Context ladder ~8K",
        "llmprobe",
        "decodeTokPerSec",
        "tok/s",
        True,
        DECODE_CAP,
        "1 cold run per rung",
        "As ~512 with about 8,250 input tokens.",
        "Filler fitted per engine.",
    ),
    Scenario(
        "context_8k_ceiling",
        "Context ladder ~8K, counting ceiling",
        "llmprobe",
        "decodeTokPerSec",
        "tok/s",
        True,
        DECODE_CAP,
        "1 run per rung",
        "Counting on the ~8K archive.",
        "Prefix shared with the coding run by design.",
    ),
    Scenario(
        "context_16k",
        "Context ladder ~16K",
        "llmprobe",
        "decodeTokPerSec",
        "tok/s",
        True,
        DECODE_CAP,
        "1 cold run per rung",
        "As ~512 with about 16,340 input tokens.",
        "Filler fitted per engine.",
    ),
    Scenario(
        "context_16k_ceiling",
        "Context ladder ~16K, counting ceiling",
        "llmprobe",
        "decodeTokPerSec",
        "tok/s",
        True,
        DECODE_CAP,
        "1 run per rung",
        "Counting on the ~16K archive.",
        "Prefix shared with the coding run by design.",
    ),
    Scenario(
        "sustained_decode",
        "Sustained coding (llmprobe sustained decode)",
        "llmprobe",
        "decodeTokPerSec",
        "tok/s",
        True,
        DECODE_CAP,
        "K measured, no warmup, last in the suite",
        "Exactly the initial-coding request again (one user turn, unique cache-bust lead-in, "
        "cold prefix, 192-token cap), run after every other scenario and a 5 s idle. It is not "
        "multi-turn and not long-context.",
        "Its value depends on everything that ran before it: minutes of load, thermal state and "
        "engine warmth (allocators, kernels, drafts). It is not a cold single-turn number.",
    ),
    Scenario(
        "prefill_8800_native",
        "Exact 8,800-ID unseen prefill, engine phase rate",
        "prefill",
        "native_prefill_tok_s",
        "tok/s",
        True,
        None,
        "1 warmup + R measured (one session)",
        "Seeded unseen text from llmprobe's corpus builder, tokenized by the engine and cut to "
        "exactly 8,800 IDs, sent as raw token IDs to /v1/completions with an 8-token cap and no "
        "draft field. Prefix storage is off on the reference engine; Stark completions never "
        "use prefix reuse.",
        "Input IDs are hashed (little-endian u32) per trial; the summary checks per seed that "
        "every arm received identical IDs.",
    ),
    Scenario(
        "prefill_8800_client",
        "Exact 8,800-ID unseen prefill, client rate",
        "prefill",
        "client_prompt_tok_s",
        "tok/s",
        True,
        None,
        "1 warmup + R measured",
        "8,800 over client time to first token, for the same trials.",
        "End-to-end common measure.",
    ),
    Scenario(
        "burst_16x300",
        "16 users x 300 tokens, draft off",
        "burst",
        "aggregate_tok_s",
        "tok/s",
        True,
        None,
        "1 warmup + R measured bursts (one session)",
        "Sixteen distinct raw counting prompts submitted together behind a barrier, 300-token "
        "cap, normal EOS, prefix storage off, drafting forced off on both engines. Rate is "
        "total emitted tokens over the time from release to the last completion. Bursts where "
        "any stream stops short of 300 tokens or reuses cached input are recorded and excluded.",
        "A batching engine serves the 16 slots together; Stark serializes the burst through one "
        "worker by design.",
    ),
    Scenario(
        "burst_16x300_makespan",
        "16 users x 300 tokens, makespan",
        "burst",
        "wall_s",
        "s",
        False,
        None,
        "1 warmup + R measured bursts",
        "Wall time of each measured burst.",
        "Lower is better.",
    ),
)
SCENARIO_BY_KEY = {scenario.key: scenario for scenario in SCENARIOS}
# The standard set: the cells a standing is read from. Every other cell is
# supplementary: measured and reported, never used for a standing. The
# 16-user burst is supplementary because Stark serializes one worker by
# design, so its row shows the cost of that choice and decides nothing.
STANDARD_SCENARIOS = (
    "decode",
    "sustained_decode",
    "spec_novel",
    "context_512",
    "context_4k",
    "context_8k",
    "context_16k",
    "prefill_repeated",
    "prefill_8800_client",
    "prefill_8800_native",
)
# Cells whose rendered input must be identical on every arm of one model. The
# context ladder is fitted by the client per engine and is not in this set.
MATCHED_INPUT_SCENARIOS = (
    "decode",
    "sustained_decode",
    "spec_novel",
    "spec_predictable",
    "prefill_repeated",
    "unseen_prefill",
    "prefill_8800_native",
    "prefill_8800_client",
)
# Every figure states its level. Scenario cells are server-level; an engine
# diagnostic (one process, no server) never enters a scenario cell.
LEVEL_CLIENT = "server, client-timed"
LEVEL_PHASE = "server, engine phase time"
LEVEL_DIAGNOSTIC = "engine diagnostic"
# The fewest per-block ratios a standing is read from.
MIN_STANDING_PAIRS = 3


def scenario_level(key: str) -> str:
    return LEVEL_PHASE if key == "prefill_8800_native" else LEVEL_CLIENT


def scenario_tier(key: str) -> str:
    return "standard" if key in STANDARD_SCENARIOS else "supplementary"


LLMPROBE_REPORT_ONLY = (
    (
        "prefix_cache",
        (
            "Four requests over one ~40 KB document: cold, exact repeat, a different question "
            "under the same document (96-token cap) and a second turn that carries the model's "
            "own answer. Deliberate prefix-cache hits; llmprobe reports TTFT ratios only."
        ),
    ),
    (
        "batching_sweep",
        (
            "One stream, then bursts of 2, 4, 6 and 8 identical echo prompts (96-token cap) "
            "with cache-busted system messages. Aggregate only, kept from the report."
        ),
    ),
    (
        "decode_length_probe",
        (
            "One request with ignore_eos and min_tokens=192 that decides whether decode and "
            "sustained runs are forced to 192 tokens. Not measured."
        ),
    ),
)
RUNG_KEYS = {
    "512": "512",
    "4.1k": "4k",
    "8.2k": "8k",
    "16.4k": "16k",
    "32.8k": "32k",
    "65.5k": "64k",
}
LABEL_PATTERNS = (
    (re.compile(r"^decode length probe$"), "decode_length_probe"),
    (re.compile(r"^decode (?:warmup|\d+/\d+)$"), "decode"),
    (re.compile(r"^prefill (?:warmup|\d+/\d+)$"), "prefill_repeated"),
    (re.compile(r"^unseen prefill \d+/\d+$"), "unseen_prefill"),
    (re.compile(r"^spec:predictable included \d+$"), "spec_predictable"),
    (re.compile(r"^spec:predictable (?:warmup|\d+/\d+)$"), "spec_predictable_all"),
    (re.compile(r"^spec:novel (?:warmup|\d+/\d+)$"), "spec_novel"),
    (re.compile(r"^sustained decode \d+/\d+$"), "sustained_decode"),
)
CONTEXT_LABEL = re.compile(r"^context ~([0-9.]+k?)( ceiling)?(?: \d+/\d+)?$")


def scenario_for_label(label: str) -> str | None:
    """Map one llmprobe receipt label to a scenario key."""
    for pattern, key in LABEL_PATTERNS:
        if pattern.match(label):
            return key
    match = CONTEXT_LABEL.match(label)
    if match:
        rung = RUNG_KEYS.get(match[1], match[1].replace(".", "_"))
        return f"context_{rung}" + ("_ceiling" if match[2] else "")
    return None


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def ids_digest(ids) -> str:
    return hashlib.sha256(struct.pack(f"<{len(ids)}I", *ids)).hexdigest()


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def now_utc() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class Receipts:
    """Append-only, private JSONL receipts under one directory."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.lock = threading.Lock()

    def write(self, stream: str, record: dict) -> None:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,80}", stream):
            raise BenchError("invalid_receipt_stream")
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.lock, (self.directory / f"{stream}.jsonl").open("a") as output:
            print(json.dumps(record, sort_keys=True), file=output, flush=True)


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@contextlib.contextmanager
def scoped_environ(updates: dict[str, str], remove_prefixes=()):
    """Temporarily replace the process environment; restore it afterwards."""
    saved = dict(os.environ)
    try:
        for key in tuple(os.environ):
            if key.startswith(tuple(remove_prefixes)):
                del os.environ[key]
        os.environ.update(updates)
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


def client_env() -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(CLIENT_ENV_PREFIXES) and key not in SECRET_ENV
    }
    env["NO_COLOR"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    return env


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind((HOST, 0))
        return sock.getsockname()[1]


def numeric_tree(value, depth=0):
    """Keep numbers, booleans, short codes and nested numeric containers only."""
    if value is None or isinstance(value, bool) or number(value):
        return value
    if isinstance(value, str):
        return value if re.fullmatch(r"[A-Za-z0-9._-]{1,64}", value) else None
    if depth >= 3:
        return None
    if isinstance(value, list):
        return [numeric_tree(item, depth + 1) for item in value[:64]]
    if isinstance(value, dict):
        return {
            str(key)[:64]: numeric_tree(item, depth + 1)
            for key, item in list(value.items())[:64]
        }
    return None


# --------------------------------------------------------------------------
# Host preflight
# --------------------------------------------------------------------------


def run_text(command, timeout=30) -> str | None:
    try:
        return subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=True
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def parse_pmset_batt(text: str | None) -> dict:
    """Parse `pmset -g batt`; only the power source and battery state are kept."""
    if not text:
        return {"source": None, "on_ac": False, "batteries": []}
    source = None
    match = re.search(r"Now drawing from '([^']+)'", text)
    if match:
        source = match[1]
    batteries = []
    for line in text.splitlines():
        battery = re.search(r"(\d{1,3})%;\s*([A-Za-z ]+?)(?:;|$)", line)
        if battery and "InternalBattery" in line:
            batteries.append({"percent": int(battery[1]), "state": battery[2].strip()})
    return {"source": source, "on_ac": source == "AC Power", "batteries": batteries}


def parse_pmset_therm(text: str | None) -> dict:
    """Parse `pmset -g therm` notes and limits into scalar fields."""
    result = {
        "thermal_warning": None,
        "performance_warning": None,
        "cpu_power_status": None,
        "limits": {},
    }
    if not text:
        result["available"] = False
        return result
    result["available"] = True
    for line in text.splitlines():
        lowered = line.strip().lower()
        for key, phrase in (
            ("thermal_warning", "thermal warning level"),
            ("performance_warning", "performance warning level"),
            ("cpu_power_status", "cpu power status"),
        ):
            if phrase in lowered:
                none_recorded = "no " in lowered and "recorded" in lowered
                result[key] = (
                    "none_recorded"
                    if none_recorded
                    else line.split(":", 1)[-1].strip()[:64]
                )
        limit = re.match(r"\s*([A-Za-z_]+)\s*=\s*(-?\d+)\s*$", line)
        if limit:
            result["limits"][limit[1]] = int(limit[2])
    return result


def parse_low_power_mode(text: str | None) -> bool | None:
    if not text:
        return None
    # `lowpowermode` is a flag. `powermode` is an enumeration on Apple Silicon:
    # 0 automatic, 1 low power, 2 high power. Only 1 is Low Power Mode.
    flag = re.search(r"^\s*lowpowermode\s+(\d+)\s*$", text, re.MULTILINE)
    if flag is not None:
        return flag[1] != "0"
    mode = re.search(r"^\s*powermode\s+(\d+)\s*$", text, re.MULTILINE)
    return None if mode is None else mode[1] == "1"


def parse_pmset_ac(text: str | None) -> dict:
    """Parse `pmset -g ac`: negotiated and rated adapter wattage only."""
    result = {"watts": None, "rated_watts": None}
    if not text:
        return result
    watts = re.search(r"^\s*Wattage\s*=\s*(\d{1,4})W\s*$", text, re.MULTILINE)
    rated = re.search(r"^\s*Adapter Name\s*=.*?(\d{2,4})W", text, re.MULTILINE)
    if watts:
        result["watts"] = int(watts[1])
    if rated:
        result["rated_watts"] = int(rated[1])
    return result


def host_warnings(snapshot: dict) -> list[str]:
    """Host conditions that move timings without making a run invalid.

    They are recorded with the run and shown in the plan and the summary. A
    power-limited or warm host shifts absolute rates by 10 percent or more,
    which is why comparisons are read from paired blocks.
    """
    warnings = []
    for battery in (snapshot.get("power") or {}).get("batteries") or []:
        if battery.get("state") == "charging":
            warnings.append(
                f"battery charging at {battery.get('percent')} percent, which lowers the power "
                "available to the GPU"
            )
    adapter = snapshot.get("adapter") or {}
    if (
        adapter.get("watts")
        and adapter.get("rated_watts")
        and adapter["watts"] < adapter["rated_watts"]
    ):
        warnings.append(
            f"power adapter negotiated {adapter['watts']} W of a rated {adapter['rated_watts']} W"
        )
    state = snapshot.get("process_thermal_state")
    if state:
        warnings.append(f"process thermal state {state} at start (0 is nominal)")
    return warnings


def process_thermal_state() -> int | None:
    """NSProcessInfo.thermalState: 0 nominal, 1 fair, 2 serious, 3 critical."""
    try:
        ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation")
        objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        send = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(
            ("objc_msgSend", objc)
        )
        info = send(
            objc.objc_getClass(b"NSProcessInfo"), objc.sel_registerName(b"processInfo")
        )
        scalar = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)(
            ("objc_msgSend", objc)
        )
        return int(scalar(info, objc.sel_registerName(b"thermalState")))
    except Exception:  # noqa: BLE001 - reported as unavailable
        return None


def competing_processes(ps_text: str | None, own_pids=()) -> list[str]:
    """Names (never arguments) of running processes that would perturb timing."""
    if ps_text is None:
        return ["unavailable"]
    found = set()
    for line in ps_text.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit() or int(parts[0]) in own_pids:
            continue
        name = Path(parts[1].strip()).name
        if name in COMPETING_PROCESSES:
            found.add(name)
    return sorted(found)


def host_snapshot(runner=run_text) -> dict:
    """Power, thermal and identity facts; reads system state only."""
    hardware = (runner(["/usr/sbin/sysctl", "-n", "hw.model"]) or "").strip() or None
    chip = (
        runner(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"]) or ""
    ).strip() or None
    memory = (runner(["/usr/sbin/sysctl", "-n", "hw.memsize"]) or "").strip()
    return {
        "captured_at": now_utc(),
        "power": parse_pmset_batt(runner(["/usr/bin/pmset", "-g", "batt"])),
        "low_power_mode": parse_low_power_mode(runner(["/usr/bin/pmset", "-g"])),
        "adapter": parse_pmset_ac(runner(["/usr/bin/pmset", "-g", "ac"])),
        "thermal": parse_pmset_therm(runner(["/usr/bin/pmset", "-g", "therm"])),
        "process_thermal_state": process_thermal_state(),
        "macos_version": (runner(["/usr/bin/sw_vers", "-productVersion"]) or "").strip()
        or None,
        "macos_build": (runner(["/usr/bin/sw_vers", "-buildVersion"]) or "").strip()
        or None,
        "hardware_model": hardware,
        "chip": chip,
        "memory_bytes": int(memory) if memory.isdigit() else None,
        "competing_processes": competing_processes(
            runner(["/bin/ps", "-axo", "pid=,comm="]),
            own_pids=(os.getpid(), os.getppid()),
        ),
    }


def require_timing_host(snapshot: dict, *, allow_busy_host=False) -> None:
    """Refuse a timing run unless the host is on AC power and otherwise quiet."""
    if not snapshot["power"]["on_ac"]:
        raise BenchError("not_on_ac_power")
    if snapshot["low_power_mode"]:
        raise BenchError("low_power_mode_enabled")
    if snapshot["competing_processes"] and not allow_busy_host:
        raise BenchError(
            "competing_processes_running:" + ",".join(snapshot["competing_processes"])
        )


def sleep_events(start: float, end: float, runner=run_text) -> dict:
    """Count system sleep entries inside the measurement window."""
    text = runner(["/usr/bin/pmset", "-g", "log"], timeout=60)
    if text is None:
        return {"checked": False, "sleep_events": None}
    count = 0
    for line in text.splitlines():
        if "Entering Sleep" not in line:
            continue
        try:
            stamp = dt.datetime.strptime(line[:25], "%Y-%m-%d %H:%M:%S %z").timestamp()
        except ValueError:
            continue
        if start <= stamp <= end:
            count += 1
    return {"checked": True, "sleep_events": count}


# --------------------------------------------------------------------------
# Model profiles
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Artifact:
    """One model artifact an engine can serve."""

    key: str
    description: str
    quantization: str
    match: str  # arms whose artifacts share this value are a matched pair
    path: str | None = None
    repo: str | None = None
    revision: str | None = None
    record: str | None = None
    declared: bool = True


@dataclass(frozen=True)
class LogEvent:
    """A reference-engine console line kept as a fixed event and state word."""

    event: str
    all: tuple[str, ...]
    any: tuple[str, ...]
    states: tuple[str, ...]


@dataclass(frozen=True)
class Reference:
    """One reference engine and the artifacts it serves for this model."""

    engine: str
    label: str
    arm_prefix: str
    binary: str | None
    version: str | None
    flags: tuple[str, ...]
    draft_off_flags: tuple[str, ...]
    draft_force_flags: tuple[str, ...]
    log_events: tuple[LogEvent, ...]
    artifacts: tuple[Artifact, ...]


@dataclass(frozen=True)
class Profile:
    """Everything model-specific the standard benchmark needs for one model."""

    key: str
    title: str
    status: str
    reference_arm: str | None
    source: dict
    reasoning: str
    cache_tag_base: str
    served_model_id: str
    model_type: str | None
    vocab_size: int
    chat_adapter: str
    template_sha256: str | None
    image_capable: bool
    drafting: str
    min_context: int
    max_context: int
    burst_context: int
    artifacts: tuple[Artifact, ...]
    references: tuple[Reference, ...]
    quality: dict
    notes: dict
    path: Path | None = None
    sha256: str | None = None
    # The Stark artifact a run uses when none is named; the first when unset.
    default_artifact: str = ""

    def default_stark_artifacts(self) -> list[Artifact]:
        key = self.default_artifact or self.artifacts[0].key
        return [item for item in self.artifacts if item.key == key]

    def reference(self, engine: str) -> Reference:
        for reference in self.references:
            if reference.engine == engine:
                return reference
        raise BenchError(f"profile_has_no_reference:{engine[:32]}")

    def facts(self) -> dict:
        """What a manifest records about the profile: identity and limits."""
        return {
            "key": self.key,
            "title": self.title,
            "status": self.status,
            "sha256": self.sha256,
            "source": dict(self.source),
            "reference_arm": self.reference_arm,
            "reasoning": self.reasoning,
            "served_model_id": self.served_model_id,
            "chat_adapter": self.chat_adapter,
            "image_capable": self.image_capable,
            "drafting": self.drafting,
            "default_artifact": self.default_artifact or self.artifacts[0].key,
            "context": {
                "minimum": self.min_context,
                "maximum": self.max_context,
                "burst": self.burst_context,
            },
            "references": [
                {"engine": item.engine, "version": item.version}
                for item in self.references
            ],
            "notes": dict(self.notes),
        }


PROFILE_KEY = re.compile(r"[a-z0-9][a-z0-9.-]{0,47}")
ARTIFACT_KEY = re.compile(r"[a-z0-9][a-z0-9-]{0,23}")
CODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/ -]{0,95}")
HEX64 = re.compile(r"[0-9a-f]{64}")
HEX40 = re.compile(r"[0-9a-f]{40}")
PROFILE_STATUSES = ("complete", "draft")
CHAT_ADAPTERS = ("inkling2-json-v1", "think-tags-v1")
DRAFT_POLICIES = ("engine-default",)
PROFILE_TABLES = {
    "": {
        "schema",
        "key",
        "title",
        "status",
        "reference_arm",
        "source",
        "client",
        "stark",
        "reference",
        "quality",
        "notes",
    },
    "source": {"repo", "revision"},
    "client": {"reasoning", "cache_tag_base"},
    "stark": {
        "served_model_id",
        "model_type",
        "vocab_size",
        "chat_adapter",
        "template_sha256",
        "image_capable",
        "drafting",
        "default_artifact",
        "context",
        "artifacts",
    },
    "stark.context": {"minimum", "maximum", "burst"},
    "artifact": {
        "description",
        "quantization",
        "match",
        "path",
        "repo",
        "revision",
        "record",
    },
    "reference": {
        "engine",
        "label",
        "arm_prefix",
        "binary",
        "version",
        "flags",
        "draft_off_flags",
        "draft_force_flags",
        "log_event",
        "artifacts",
    },
    "log_event": {"event", "all", "any", "states"},
    "quality": {
        "reference_artifact",
        "reference_path",
        "probe",
        "probe_format",
        "probe_command",
    },
}


def _table(value, where: str, allowed: str) -> dict:
    """A TOML table with no key the format does not define (catches typos)."""
    if not isinstance(value, dict):
        raise BenchError(f"invalid_profile:{where}")
    unknown = sorted(set(value) - PROFILE_TABLES[allowed])
    if unknown:
        raise BenchError(f"invalid_profile:{where}.{str(unknown[0])[:32]}")
    return value


def _field(table: dict, key: str, where: str, kind, *, required=True, pattern=None):
    value = table.get(key)
    if value is None:
        if required:
            raise BenchError(f"invalid_profile:{where}.{key}")
        return None
    good = isinstance(value, kind) and not (kind is int and isinstance(value, bool))
    if good and kind is str:
        good = 0 < len(value) <= 400 and (pattern is None or pattern.fullmatch(value))
    if not good:
        raise BenchError(f"invalid_profile:{where}.{key}")
    return value


def _strings(table: dict, key: str, where: str) -> tuple[str, ...]:
    value = table.get(key, [])
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not 0 < len(item) <= 200 for item in value
    ):
        raise BenchError(f"invalid_profile:{where}.{key}")
    return tuple(value)


def _artifacts(tables, where: str) -> tuple[Artifact, ...]:
    if not isinstance(tables, dict) or not tables:
        raise BenchError(f"invalid_profile:{where}.artifacts")
    result = []
    for key, raw in tables.items():
        here = f"{where}.artifacts.{str(key)[:24]}"
        if not ARTIFACT_KEY.fullmatch(str(key)):
            raise BenchError(f"invalid_profile:{here}")
        table = _table(raw, here, "artifact")
        result.append(
            Artifact(
                key=key,
                description=_field(table, "description", here, str),
                quantization=_field(table, "quantization", here, str),
                match=_field(table, "match", here, str, pattern=ARTIFACT_KEY),
                path=_field(table, "path", here, str, required=False),
                repo=_field(table, "repo", here, str, required=False, pattern=CODE),
                revision=_field(
                    table, "revision", here, str, required=False, pattern=HEX40
                ),
                record=_field(table, "record", here, str, required=False),
            )
        )
    return tuple(result)


def parse_profile(data: dict, *, key: str | None = None) -> Profile:
    """Validate one parsed profile document. Raises BenchError on any defect."""
    root = _table(data, "profile", "")
    if root.get("schema") != PROFILE_SCHEMA:
        raise BenchError("invalid_profile:schema")
    name = _field(root, "key", "profile", str, pattern=PROFILE_KEY)
    if key is not None and name != key:
        raise BenchError("invalid_profile:key_differs_from_file_name")
    status = _field(root, "status", "profile", str)
    if status not in PROFILE_STATUSES:
        raise BenchError("invalid_profile:profile.status")
    source = _table(root.get("source", {}), "source", "source")
    client = _table(root.get("client"), "client", "client")
    stark = _table(root.get("stark"), "stark", "stark")
    context = _table(stark.get("context"), "stark.context", "stark.context")
    limits = [
        _field(context, item, "stark.context", int)
        for item in ("minimum", "maximum", "burst")
    ]
    if not (0 < limits[0] <= limits[1] and 0 < limits[2] <= limits[1]):
        raise BenchError("invalid_profile:stark.context")
    adapter = _field(stark, "chat_adapter", "stark", str)
    if adapter not in CHAT_ADAPTERS:
        raise BenchError("invalid_profile:stark.chat_adapter")
    drafting = _field(stark, "drafting", "stark", str)
    if drafting not in DRAFT_POLICIES:
        raise BenchError("invalid_profile:stark.drafting")
    vocab = _field(stark, "vocab_size", "stark", int)
    if vocab <= 0:
        raise BenchError("invalid_profile:stark.vocab_size")
    artifacts = _artifacts(stark.get("artifacts"), "stark")
    default_artifact = _field(
        stark, "default_artifact", "stark", str, required=False, pattern=ARTIFACT_KEY
    )
    if default_artifact is not None and default_artifact not in {
        item.key for item in artifacts
    }:
        raise BenchError("invalid_profile:stark.default_artifact")
    references = []
    raw_references = root.get("reference", [])
    if not isinstance(raw_references, list):
        raise BenchError("invalid_profile:reference")
    for index, raw in enumerate(raw_references):
        here = f"reference[{index}]"
        table = _table(raw, here, "reference")
        engine = _field(table, "engine", here, str)
        if engine == "stark" or engine not in ENGINES:
            raise BenchError(f"invalid_profile:{here}.engine")
        events = []
        raw_events = table.get("log_event", [])
        if not isinstance(raw_events, list):
            raise BenchError(f"invalid_profile:{here}.log_event")
        for raw_event in raw_events:
            event = _table(raw_event, f"{here}.log_event", "log_event")
            events.append(
                LogEvent(
                    event=_field(
                        event,
                        "event",
                        f"{here}.log_event",
                        str,
                        pattern=re.compile(r"[a-z][a-z0-9_]{0,47}"),
                    ),
                    all=_strings(event, "all", f"{here}.log_event"),
                    any=_strings(event, "any", f"{here}.log_event"),
                    states=_strings(event, "states", f"{here}.log_event"),
                )
            )
        references.append(
            Reference(
                engine=engine,
                label=_field(table, "label", here, str),
                arm_prefix=_field(table, "arm_prefix", here, str, pattern=ARTIFACT_KEY),
                binary=_field(table, "binary", here, str, required=False),
                version=_field(table, "version", here, str, required=False),
                flags=_strings(table, "flags", here),
                draft_off_flags=_strings(table, "draft_off_flags", here),
                draft_force_flags=_strings(table, "draft_force_flags", here),
                log_events=tuple(events),
                artifacts=_artifacts(table.get("artifacts"), here),
            )
        )
    if len({item.engine for item in references}) != len(references):
        raise BenchError("invalid_profile:reference.engine_repeated")
    arm_names = [f"stark-{item.key}" for item in artifacts] + [
        f"{reference.arm_prefix}-{item.key}"
        for reference in references
        for item in reference.artifacts
    ]
    if len(set(arm_names)) != len(arm_names) or any(
        not ARM_NAME.fullmatch(item) for item in arm_names
    ):
        raise BenchError("invalid_profile:arm_names")
    # The reference arm is a reference engine's arm (by policy its load of the
    # original checkpoint), never one of Stark's own.
    reference_arm = _field(root, "reference_arm", "profile", str, required=False)
    if reference_arm is not None and reference_arm not in arm_names[len(artifacts) :]:
        raise BenchError("invalid_profile:profile.reference_arm")
    quality = dict(_table(root.get("quality", {}), "quality", "quality"))
    for item in ("reference_artifact", "reference_path", "probe", "probe_format"):
        _field(quality, item, "quality", str, required=False, pattern=CODE)
    if quality.get("reference_artifact") not in (
        None,
        *(item.key for item in artifacts),
    ):
        raise BenchError("invalid_profile:quality.reference_artifact")
    if quality.get("probe_format") not in (None, *QUALITY_FORMATS):
        raise BenchError("invalid_profile:quality.probe_format")
    quality["probe_command"] = list(_strings(quality, "probe_command", "quality"))
    notes = root.get("notes", {})
    if not isinstance(notes, dict) or any(
        key not in SCENARIO_BY_KEY or not isinstance(value, str) or len(value) > 400
        for key, value in notes.items()
    ):
        raise BenchError("invalid_profile:notes")
    template = _field(stark, "template_sha256", "stark", str, required=False)
    if template is not None and not HEX64.fullmatch(template):
        raise BenchError("invalid_profile:stark.template_sha256")
    reasoning = _field(client, "reasoning", "client", str)
    tag = _field(client, "cache_tag_base", "client", str)
    if not re.fullmatch(r"[a-z-]{1,16}", reasoning) or not re.fullmatch(
        r"[a-z0-9]{1,12}", tag
    ):
        raise BenchError("invalid_profile:client")
    return Profile(
        key=name,
        title=_field(root, "title", "profile", str),
        status=status,
        reference_arm=reference_arm,
        source={
            "repo": _field(source, "repo", "source", str, required=False, pattern=CODE),
            "revision": _field(
                source, "revision", "source", str, required=False, pattern=HEX40
            ),
        },
        reasoning=reasoning,
        cache_tag_base=tag,
        served_model_id=_field(
            stark,
            "served_model_id",
            "stark",
            str,
            pattern=re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}"),
        ),
        model_type=_field(stark, "model_type", "stark", str, required=False),
        vocab_size=vocab,
        chat_adapter=adapter,
        template_sha256=template,
        image_capable=_field(stark, "image_capable", "stark", bool),
        drafting=drafting,
        min_context=limits[0],
        max_context=limits[1],
        burst_context=limits[2],
        artifacts=artifacts,
        references=tuple(references),
        quality=quality,
        notes=dict(notes),
        default_artifact=default_artifact or "",
    )


def load_profile(source: str | Path, *, directory: Path | None = None) -> Profile:
    """Load a profile by key (from the profile directory) or by file path."""
    directory = directory or PROFILE_DIRECTORY
    text = str(source)
    if text.endswith(".toml") or "/" in text:
        path = resolve_path(text)
        key = None
    else:
        if not PROFILE_KEY.fullmatch(text):
            raise BenchError("invalid_profile_name")
        path = directory / f"{text}.toml"
        key = text
    if not path.is_file():
        raise BenchError(f"profile_not_found:{path.name[:64]}")
    try:
        data = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError, UnicodeDecodeError):
        raise BenchError(f"profile_unreadable:{path.name[:64]}") from None
    profile = parse_profile(data, key=key)
    return replace(profile, path=path, sha256=sha256_file(path))


def available_profiles(directory: Path | None = None) -> list[tuple[str, object]]:
    """Every profile file with either its Profile or the BenchError it raises."""
    directory = directory or PROFILE_DIRECTORY
    result = []
    for path in sorted(directory.glob("*.toml")) if directory.is_dir() else []:
        try:
            result.append((path.stem, load_profile(path.stem, directory=directory)))
        except BenchError as error:
            result.append((path.stem, error))
    return result


def require_profile(config) -> Profile:
    profile = getattr(config, "profile", None)
    if profile is None:
        raise BenchError("profile_required")
    return profile


# --------------------------------------------------------------------------
# Configuration, validation and the planned matrix
# --------------------------------------------------------------------------


@dataclass
class Arm:
    name: str
    engine: str  # "stark" or a reference engine key such as "mlx-serve"
    binary: Path
    model: Path
    description: str = ""
    env: dict[str, str] = field(default_factory=dict)  # diagnostic only
    model_label: str = ""  # set when the artifact is not the run's shared one
    artifact: str = ""  # artifact key; empty when the caller named none
    quantization: str = ""
    match: str = ""  # matching class; empty for an artifact the profile lacks
    build: str = ""  # Stark build name: "current" or a --stark-build name
    declared: bool = True  # False for an artifact the profile does not declare

    def identity(self) -> dict:
        identity = {
            "name": self.name,
            "engine": self.engine,
            "binary_name": self.binary.name,
            "description": self.description,
            "diagnostic_env": dict(self.env),
        }
        if self.model_label:
            identity["model_label"] = self.model_label
        for key in ("artifact", "quantization", "match", "build"):
            if getattr(self, key):
                identity[key] = getattr(self, key)
        if not self.declared:
            identity["declared"] = False
        return identity


@dataclass
class Config:
    engine: str  # which engines were selected: "stark", "mlx-serve", "both", "all"
    arms: list[Arm]
    llmprobe: Path
    node: str
    output: Path
    lock_file: Path
    repeats: int
    warmups: int
    sessions: tuple[str, ...]
    probe_runs: int | None
    reasoning: str
    cache_tag_base: str
    allow_busy_host: bool
    allow_unpinned_llmprobe: bool
    stark_drafts: str = "engine"
    stark_worker_timeout: float = 600.0
    stark_image_check_dir: Path | None = None
    stark_effort_aliases: dict = field(default_factory=dict)
    stark_min_context: int | None = None  # None: the profile's minimum
    mlx_force_mtp: bool = False  # force drafting on in reference engines
    profile: Profile | None = None
    layout: str = "legacy"  # "blocks" or "legacy"; see run_matrix
    run_label: str = ""  # run directory suffix; the engine selection when empty
    entry: Path | None = None  # the script that was invoked, when not this file

    def min_context(self) -> int | None:
        if self.stark_min_context is not None:
            return self.stark_min_context
        return self.profile.min_context if self.profile else None

    def protocol(self) -> dict:
        """Settings that change what is measured; repeats and sessions do not."""
        return {
            "probe_runs": self.probe_runs,
            "reasoning": self.reasoning,
            "stark_drafts": self.stark_drafts,
            "stark_effort_aliases": self.stark_effort_aliases,
            "stark_min_context": self.min_context(),
            "mlx_force_mtp": self.mlx_force_mtp,
        }


def resolve_path(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPOSITORY / path


def default_for(name: str) -> str | None:
    variable, fallback = DEFAULTS[name]
    return os.environ.get(variable, fallback)


def interleave(first: list, second: list) -> list:
    """Alternate two lists: a0, b0, a1, b1, then the remainder."""
    merged = []
    for index in range(max(len(first), len(second))):
        merged.extend(item[index] for item in (first, second) if index < len(item))
    return merged


def parse_sessions(text: str) -> tuple[str, ...]:
    sessions = tuple(part.strip() for part in text.split(",") if part.strip())
    if not sessions or any(part not in SESSION_KINDS for part in sessions):
        raise BenchError("invalid_sessions")
    return sessions


def parse_effort_aliases(specs) -> dict:
    aliases = {}
    for spec in specs or ():
        source, _, target = spec.partition("=")
        if not re.fullmatch(r"[a-z]{1,16}", source) or not re.fullmatch(
            r"[a-z]{1,16}", target
        ):
            raise BenchError("invalid_effort_alias")
        aliases[source] = target
    return aliases


def apply_stark_env(by_name: dict[str, Arm], specs, *, prefix: str = "") -> None:
    """Diagnostic environment variables for single Stark arms (NAME=KEY=VALUE)."""
    for spec in specs or ():
        name, _, assignment = spec.partition("=")
        key, _, value = assignment.partition("=")
        arm = by_name.get(prefix + name)
        if (
            arm is None
            or arm.engine != "stark"
            or not ENV_KEY.fullmatch(key)
            or len(value) > 256
        ):
            raise BenchError("invalid_stark_env")
        arm.env[key] = value


def select_arms(arms: list[Arm], selection: str | None) -> list[Arm]:
    selected = [part.strip() for part in (selection or "").split(",") if part.strip()]
    if not selected:
        return arms
    unknown = sorted(set(selected) - {arm.name for arm in arms})
    if unknown:
        raise BenchError("unknown_arms:" + ",".join(name[:32] for name in unknown))
    return [arm for arm in arms if arm.name in selected]


def check_run_arguments(args) -> None:
    if args.repeats < 1 or args.warmups < 0:
        raise BenchError("invalid_repeat_count")
    if not re.fullmatch(r"[a-z0-9]{1,12}", args.cache_tag_base):
        raise BenchError("invalid_cache_tag_base")
    if not re.fullmatch(r"[a-z-]{1,16}", args.llmprobe_reasoning):
        raise BenchError("invalid_reasoning_mode")


def artifact_choices(
    declared: tuple[Artifact, ...], specs, *, default: list[Artifact], code: str
) -> list[tuple[Artifact, Path]]:
    """Artifacts named NAME or NAME=PATH, or the default set when none is named."""
    by_key = {item.key: item for item in declared}
    chosen: list[tuple[Artifact, Path]] = []
    for spec in specs or ():
        name, _, path = spec.partition("=")
        if not ARTIFACT_KEY.fullmatch(name) or any(name == a.key for a, _ in chosen):
            raise BenchError(code)
        artifact = by_key.get(name)
        if artifact is None:
            if not path:
                raise BenchError(code)
            artifact = Artifact(
                key=name,
                description="Artifact not declared in the profile",
                quantization="undeclared",
                match="",
                declared=False,
            )
        location = path or artifact.path
        if not location:
            raise BenchError(f"artifact_path_required:{name}")
        chosen.append((artifact, resolve_path(location)))
    if not specs:
        for artifact in default:
            if not artifact.path:
                raise BenchError(f"artifact_path_required:{artifact.key}")
            chosen.append((artifact, resolve_path(artifact.path)))
    return chosen


def profile_arms(profile: Profile, args) -> list[Arm]:
    """Arms for a profile: (Stark build, artifact) pairs and reference packs."""
    wanted = [part.strip() for part in (args.engines or "").split(",") if part.strip()]
    known = ["stark", *(reference.engine for reference in profile.references)]
    if any(item not in known for item in wanted):
        raise BenchError("unknown_engine")
    wanted = wanted or known
    stark: list[Arm] = []
    if "stark" in wanted:
        builds = [("current", resolve_path(args.stark_binary))]
        for spec in args.stark_build or ():
            name, _, path = spec.partition("=")
            if (
                not ARTIFACT_KEY.fullmatch(name)
                or not path
                or any(name == seen for seen, _ in builds)
            ):
                raise BenchError("invalid_stark_build")
            builds.append((name, resolve_path(path)))
        chosen = artifact_choices(
            profile.artifacts,
            args.stark_artifact,
            default=profile.default_stark_artifacts(),
            code="invalid_stark_artifact",
        )
        for build, binary in builds:
            for artifact, path in chosen:
                suffix = "" if build == "current" else f"-{build}"
                stark.append(
                    Arm(
                        f"stark-{artifact.key}{suffix}",
                        "stark",
                        binary,
                        path,
                        f"Stark build {build}, {artifact.description}",
                        artifact=artifact.key,
                        quantization=artifact.quantization,
                        match=artifact.match,
                        build=build,
                        declared=artifact.declared,
                    )
                )
    binaries: dict[str, str] = {}
    for spec in args.reference_binary or ():
        engine, separator, path = spec.partition("=")
        if not separator and len(profile.references) == 1:
            engine, path = profile.references[0].engine, spec
        if engine not in known[1:] or not path or engine in binaries:
            raise BenchError("invalid_reference_binary")
        binaries[engine] = path
    packs: dict[str, list[str]] = {}
    for spec in args.reference_artifact or ():
        engine, separator, rest = spec.partition(":")
        if not separator and len(profile.references) == 1:
            engine, rest = profile.references[0].engine, spec
        if engine not in known[1:] or not rest:
            raise BenchError("invalid_reference_artifact")
        packs.setdefault(engine, []).append(rest)
    references: list[Arm] = []
    for reference in profile.references:
        if reference.engine not in wanted:
            continue
        location = binaries.get(reference.engine) or reference.binary
        if not location:
            raise BenchError(f"reference_binary_required:{reference.engine}")
        chosen = artifact_choices(
            reference.artifacts,
            packs.get(reference.engine),
            default=[item for item in reference.artifacts if item.path],
            code="invalid_reference_artifact",
        )
        for artifact, path in chosen:
            references.append(
                Arm(
                    f"{reference.arm_prefix}-{artifact.key}",
                    reference.engine,
                    resolve_path(location),
                    path,
                    f"{reference.label}, {artifact.description}",
                    artifact=artifact.key,
                    quantization=artifact.quantization,
                    match=artifact.match,
                    declared=artifact.declared,
                )
            )
    return interleave(stark, references)


def build_config(args) -> Config:
    """Configuration for the profile-driven command line (run and plan)."""
    profile = load_profile(args.profile)
    if args.llmprobe_reasoning is None:
        args.llmprobe_reasoning = profile.reasoning
    if args.cache_tag_base is None:
        args.cache_tag_base = profile.cache_tag_base
    check_run_arguments(args)
    if args.layout not in LAYOUTS:
        raise BenchError("invalid_layout")
    if args.stark_min_context is not None and not (
        0 < args.stark_min_context <= profile.max_context
    ):
        raise BenchError("invalid_min_context")
    arms = profile_arms(profile, args)
    by_name = {arm.name: arm for arm in arms}
    if len(by_name) != len(arms) or any(
        not ARM_NAME.fullmatch(name) for name in by_name
    ):
        raise BenchError("duplicate_arm_names")
    apply_stark_env(by_name, args.stark_env)
    arms = select_arms(arms, args.arms)
    if not arms:
        raise BenchError("no_arms")
    return Config(
        engine=args.engines or "all",
        arms=arms,
        llmprobe=resolve_path(args.llmprobe),
        node=args.node,
        output=resolve_path(args.output),
        lock_file=resolve_path(args.lock_file),
        repeats=args.repeats,
        warmups=args.warmups,
        sessions=parse_sessions(args.sessions),
        probe_runs=args.probe_runs,
        reasoning=args.llmprobe_reasoning,
        cache_tag_base=args.cache_tag_base,
        allow_busy_host=args.allow_busy_host,
        allow_unpinned_llmprobe=args.allow_unpinned_llmprobe,
        stark_drafts=args.stark_drafts,
        stark_worker_timeout=args.stark_worker_timeout,
        stark_image_check_dir=resolve_path(args.stark_image_check_dir),
        stark_effort_aliases=parse_effort_aliases(args.stark_effort_alias),
        stark_min_context=args.stark_min_context
        if args.stark_min_context is not None
        else profile.min_context,
        mlx_force_mtp=args.reference_force_drafts,
        profile=profile,
        layout=args.layout,
        run_label=profile.key,
    )


def node_version(node: str) -> tuple[int, int, int] | None:
    text = run_text([node, "--version"], timeout=15)
    match = re.match(r"v(\d+)\.(\d+)\.(\d+)", text or "")
    return None if match is None else tuple(int(part) for part in match.groups())


def template_helper() -> Path | None:
    """Where stark_convert will find the native chat-template helper."""
    candidate = REPOSITORY / "target/release/stark-template"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return candidate
    found = shutil.which("stark-template")
    return Path(found) if found else None


def llmprobe_identity(root: Path) -> dict:
    identity = {
        "root_present": root.is_dir(),
        "files": {},
        "version": None,
        "receipt_patch": False,
    }
    if not root.is_dir():
        return identity
    for name in LLMPROBE_IDENTITY_FILES:
        path = root / name
        identity["files"][name] = sha256_file(path) if path.is_file() else None
    with contextlib.suppress(OSError, ValueError):
        identity["version"] = json.loads((root / "package.json").read_text()).get(
            "version"
        )
    bundle = root / "bin/dist/llmprobe.mjs"
    if bundle.is_file():
        identity["receipt_patch"] = (
            LLMPROBE_RECEIPT_MARKER.encode() in bundle.read_bytes()
        )
    identity["pinned"] = all(
        identity["files"].get(name) == digest
        for name, digest in LLMPROBE_PINNED.items()
    )
    return identity


def model_identity(model: Path, *, small_limit=16 * 1024 * 1024) -> dict:
    """Hashes of small metadata files and sizes of weight files."""
    result = {"files": {}, "weights": {}}
    if not model.is_dir():
        return result
    for path in sorted(model.iterdir()):
        if not path.is_file():
            continue
        size = path.stat().st_size
        if path.suffix in (".json", ".jinja", ".txt", ".model") and size <= small_limit:
            result["files"][path.name] = sha256_file(path)
        elif path.suffix == ".safetensors":
            result["weights"][path.name] = size
    return result


def fingerprint(identity: dict) -> str:
    """One hash of an artifact identity, used to bind a quality report to it."""
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def weight_problem(model: Path) -> str | None:
    """Whether the weight shards a model directory names are actually present.

    A pack whose shards were removed still has its index, config and draft
    head, and an engine then fails at load time with little explanation.
    """
    index = model / "model.safetensors.index.json"
    if index.is_file():
        try:
            names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return "model.safetensors.index.json is unreadable"
        missing = [
            name
            for name in names
            if not (model / name).is_file() or (model / name).stat().st_size == 0
        ]
        if missing:
            return (
                f"{len(missing)} of {len(names)} weight shards named in model.safetensors.index.json "
                f"are missing (first: {missing[0]})"
            )
        return None
    shards = [
        path for path in model.glob("*.safetensors") if path.name != "mtp.safetensors"
    ]
    return (
        None
        if shards
        else "no weight shards (*.safetensors other than mtp.safetensors)"
    )


def artifact_facts(model: Path) -> dict:
    """Model type and chat-template hash from an artifact's config.json."""
    facts = {"model_type": None, "template_sha256": None}
    with contextlib.suppress(OSError, ValueError, TypeError, AttributeError):
        config = json.loads((model / "config.json").read_text())
        if isinstance(config.get("model_type"), str):
            facts["model_type"] = config["model_type"][:64]
        template = (config.get("token_codec_config") or {}).get("prompt_template")
        if isinstance(template, str):
            facts["template_sha256"] = text_digest(template)
    return facts


def output_problem(output: Path, runner=None) -> str | None:
    """Receipts are private: inside the repository they must be ignored by Git."""
    runner = runner or run_text
    try:
        output.resolve().relative_to(REPOSITORY)
    except ValueError:
        return None
    ignored = runner(
        ["git", "-C", str(REPOSITORY), "check-ignore", str(output.resolve())]
    )
    if ignored and ignored.strip():
        return None
    return (
        f"output directory is inside the repository and not ignored by Git: {output} "
        "(use a location under runtime/)"
    )


def validate(
    config: Config,
    *,
    node_check=None,
    template_check=None,
    strict=False,
    ignore_check=None,
) -> tuple[list[str], dict]:
    """Return (problems, facts). Reads files only; starts no model process.

    strict adds the checks a standard run needs and an older command line did
    not make: the profile is complete, each Stark artifact is the profile's
    model with the profile's chat template, and the output is private.
    """
    node_check = node_check or node_version
    template_check = template_check or template_helper
    problems: list[str] = []
    facts: dict = {"binary_sha256": {}}
    for arm in config.arms:
        if not (arm.binary and arm.binary.is_file() and os.access(arm.binary, os.X_OK)):
            problems.append(
                f"{arm.name}: engine binary missing or not executable: {arm.binary}"
            )
        else:
            facts["binary_sha256"][arm.name] = sha256_file(arm.binary)
        if not (arm.model and (arm.model / "config.json").is_file()):
            problems.append(
                f"{arm.name}: model directory has no config.json: {arm.model}"
            )
        elif problem := weight_problem(arm.model):
            problems.append(f"{arm.name}: {problem}: {arm.model}")
    stark = [arm for arm in config.arms if arm.engine == "stark"]
    pairs = [
        (facts["binary_sha256"].get(arm.name), str(arm.model))
        for arm in stark
        if facts["binary_sha256"].get(arm.name)
    ]
    if len(pairs) != len(set(pairs)):
        problems.append("two Stark arms point at byte-identical binaries")
    if stark:
        helper = template_check()
        facts["template_helper"] = str(helper) if helper else None
        if helper is None:
            problems.append(
                "native chat-template helper missing: build target/release/stark-template"
            )
    probe = (
        llmprobe_identity(config.llmprobe)
        if config.llmprobe
        else {"root_present": False}
    )
    facts["llmprobe"] = probe
    if "probe" in config.sessions or "prefill" in config.sessions:
        if not probe.get("root_present"):
            problems.append(f"llmprobe checkout missing: {config.llmprobe}")
        else:
            if probe.get("version") != LLMPROBE_VERSION:
                problems.append(
                    f"llmprobe version is {probe.get('version')}, expected {LLMPROBE_VERSION}"
                )
            if "probe" in config.sessions and not probe.get("receipt_patch"):
                problems.append("llmprobe bundle lacks the aggregate-receipt patch")
            if not probe.get("pinned") and not config.allow_unpinned_llmprobe:
                problems.append(
                    "llmprobe bundle or benchmark source differs from the pinned build "
                    "(pass --allow-unpinned-llmprobe to record it as unpinned)"
                )
            if not (config.llmprobe / "node_modules").is_dir():
                problems.append("llmprobe node_modules missing")
        version = node_check(config.node)
        facts["node_version"] = ".".join(map(str, version)) if version else None
        if version is None or version < (22, 19, 0):
            problems.append(
                f"Node.js 22.19 or newer required; found {facts['node_version']}"
            )
    if strict:
        profile = require_profile(config)
        if profile.status != "complete":
            problems.append(
                f"profile {profile.key} is marked {profile.status}; only a complete profile is run"
            )
        for arm in stark:
            found = artifact_facts(arm.model)
            if profile.model_type and found["model_type"] != profile.model_type:
                problems.append(
                    f"{arm.name}: config.json model_type is {found['model_type']}, "
                    f"the profile expects {profile.model_type}"
                )
            if (
                profile.template_sha256
                and found["template_sha256"] != profile.template_sha256
            ):
                problems.append(
                    f"{arm.name}: chat template differs from the profile's; "
                    "the rendered prompts would not match"
                )
        if problem := output_problem(config.output, ignore_check):
            problems.append(problem)
        for kind, reason in unavailable_sessions(config).items():
            problems.append(f"session {kind} cannot run for this profile: {reason}")
    return problems, facts


def unavailable_sessions(config) -> dict[str, str]:
    """Requested sessions the profile's limits cannot hold, with the reason."""
    profile = getattr(config, "profile", None)
    result = {}
    if profile is None:
        return result
    need = EXACT_PREFILL_TOKENS + EXACT_PREFILL_CAP
    if "prefill" in config.sessions and need > profile.max_context:
        result["prefill"] = (
            f"{EXACT_PREFILL_TOKENS:,} input tokens exceed the {profile.max_context:,}-token "
            "context limit"
        )
    return result


def standard_deviations(config: Config, facts: dict | None = None) -> list[str]:
    """Why a run is not a standard scoreboard run. Empty for a standard run."""
    profile = getattr(config, "profile", None)
    reasons = []
    if profile is None:
        return ["no profile"]
    if config.reasoning != profile.reasoning:
        reasons.append(
            f"reasoning mode {config.reasoning} differs from the profile's {profile.reasoning}"
        )
    if config.cache_tag_base != profile.cache_tag_base:
        reasons.append("cache tag base differs from the profile's (inputs differ)")
    if config.probe_runs:
        reasons.append("client run count overridden (--probe-runs)")
    if config.stark_drafts != "engine":
        reasons.append("Stark drafting forced off")
    if config.mlx_force_mtp:
        reasons.append("reference-engine drafting forced on")
    if config.stark_effort_aliases:
        reasons.append("reasoning effort alias in use")
    if config.min_context() != profile.min_context:
        reasons.append("worker minimum context overridden")
    if config.layout != "blocks":
        reasons.append(f"session layout {config.layout} is not the block layout")
    if config.repeats < MIN_STANDARD_REPEATS:
        reasons.append(f"fewer than {MIN_STANDARD_REPEATS} repeats")
    if config.warmups != 1:
        reasons.append("warmup count differs from 1")
    if config.allow_busy_host:
        reasons.append("busy host allowed")
    if facts is not None and not (facts.get("llmprobe") or {}).get("pinned", True):
        reasons.append("client build differs from the pinned build")
    return reasons


def arm_disqualifications(arm: Arm) -> list[str]:
    """Why one arm is diagnostic even inside a standard run."""
    reasons = []
    if arm.env:
        reasons.append("diagnostic environment: " + ", ".join(sorted(arm.env)))
    if not arm.declared:
        reasons.append("artifact not declared in the profile")
    return reasons


def session_order(config: Config, repeat: int) -> list[Arm]:
    """Arm order for one repeat: reversed on odd repeats in the block layout."""
    if config.layout == "blocks" and repeat % 2:
        return list(reversed(config.arms))
    return list(config.arms)


def matched_counterparts(arms: list[Arm]) -> dict[str, str]:
    """For each Stark arm, the reference arm serving a matched artifact."""
    result = {}
    for arm in arms:
        if arm.engine != "stark" or not arm.match:
            continue
        for other in arms:
            if other.engine != "stark" and other.match == arm.match:
                result[arm.name] = other.name
                break
    return result


def reasoning_label(mode: str | None) -> str:
    if mode in (None, REASONING_CLIENT_DEFAULT):
        return "client default (reasoning_effort medium)"
    return f"llmprobe --reasoning {mode}"


def planned_cells(config: Config) -> list[dict]:
    """Planned measured samples per scenario and arm."""
    k = config.probe_runs or 3
    ladder = config.probe_runs or 1
    rows = []
    for scenario in SCENARIOS:
        if scenario.source == "llmprobe" and "probe" in config.sessions:
            per = ladder if scenario.key.startswith("context_") else k
            note = (
                "upper bound; only echoed runs count"
                if scenario.key == "spec_predictable"
                else ""
            )
            rows.append(
                {
                    "scenario": scenario.key,
                    "samples": per * config.repeats,
                    "note": note,
                }
            )
        elif scenario.source in config.sessions:
            rows.append(
                {
                    "scenario": scenario.key,
                    "samples": config.repeats,
                    "note": f"{config.warmups} warmup",
                }
            )
    return rows


def drafting_plan(config: Config) -> str:
    """One line saying how drafting is set on each engine."""
    profile = getattr(config, "profile", None)
    keys = ["stark"] + (
        [reference.engine for reference in profile.references]
        if profile
        else [key for key in ENGINES if key != "stark"]
    )
    parts = [
        f"{ENGINES[key].label} {ENGINES[key].drafting_plan(config)}" for key in keys
    ]
    scope = "both" if len(keys) == 2 else "every engine"
    return f"Drafting: {'; '.join(parts)}; burst forced off on {scope}"


def print_plan(
    config: Config,
    problems: list[str],
    facts: dict,
    host: dict,
    out=sys.stdout,
    *,
    title: str | None = None,
) -> None:
    def write(text=""):
        print(text, file=out)

    probe = facts.get("llmprobe", {})
    profile = getattr(config, "profile", None)
    title = title or (
        f"Stark standard benchmark, profile {profile.key}"
        if profile
        else "Stark standard benchmark"
    )
    write(f"{title} ({SCHEMA}), engines: {config.engine}")
    write(
        f"llmprobe: {config.llmprobe} (version {probe.get('version')}, pinned {probe.get('pinned')})"
    )
    write(f"Output: {config.output}   lock: {config.lock_file}")
    write(
        f"Repeats per cell: {config.repeats}   warmups: {config.warmups}   sessions: {','.join(config.sessions)}"
    )
    write(f"Reasoning mode: {reasoning_label(config.reasoning)}")
    write(f"llmprobe runs flag: {config.probe_runs or 'default (K=3, ladder 1)'}")
    write(drafting_plan(config))
    write()
    write("Arms (session order):")
    for arm in config.arms:
        digest = facts.get("binary_sha256", {}).get(arm.name)
        write(f"  {arm.name}: {arm.description}")
        write(
            f"    binary {arm.binary} (sha256 {digest[:16] + '...' if digest else 'unavailable'})"
        )
        write(f"    model {arm.model}")
        if arm.env:
            write(
                "    diagnostic env: "
                + ", ".join(f"{k}={v}" for k, v in sorted(arm.env.items()))
            )
    write()
    write("Sessions per arm (each llmprobe invocation gets a fresh server):")
    blocks = config.layout == "blocks"
    if "probe" in config.sessions:
        write(
            f"  probe: {config.repeats} llmprobe --bench-only invocations, arms alternated within each repeat"
        )
    if "prefill" in config.sessions:
        write(
            f"  prefill: {config.repeats} sessions, each {config.warmups} warmup + 1 exact 8,800-ID trial"
            if blocks
            else f"  prefill: one session, {config.warmups} warmup + {config.repeats} exact 8,800-ID trials"
        )
    if "burst" in config.sessions:
        write(
            (
                f"  burst: {config.repeats} sessions, each {config.warmups} warmup + 1 burst of "
                if blocks
                else f"  burst: one session, {config.warmups} warmup + {config.repeats} bursts of "
            )
            + f"{BURST_WIDTH} x {BURST_CAP} tokens, drafting off"
        )
    write()
    write("Planned measured samples per arm:")
    for row in planned_cells(config):
        scenario = SCENARIO_BY_KEY[row["scenario"]]
        note = f" ({row['note']})" if row["note"] else ""
        write(f"  {scenario.key:<22} n={row['samples']:<3} {scenario.title}{note}")
    write(
        "  Report-only llmprobe sections: "
        + ", ".join(name for name, _ in LLMPROBE_REPORT_ONLY)
    )
    write()
    power = host["power"]
    write(
        f"Host: {host.get('hardware_model')} {host.get('chip')}, macOS {host.get('macos_version')} "
        f"({host.get('macos_build')})"
    )
    write(
        f"Power: {power.get('source')}; low power mode {host.get('low_power_mode')}; thermal "
        f"{host['thermal'].get('thermal_warning')}, process thermal state {host.get('process_thermal_state')}"
    )
    write(f"Competing processes: {', '.join(host['competing_processes']) or 'none'}")
    blockers = []
    if not power.get("on_ac"):
        blockers.append("not on AC power")
    if host.get("low_power_mode"):
        blockers.append("Low Power Mode is on")
    if host["competing_processes"] and not config.allow_busy_host:
        blockers.append("competing processes running")
    write(
        "Timing run would "
        + ("be refused: " + "; ".join(blockers) if blockers else "be allowed")
    )
    write()
    if problems:
        write("Configuration problems:")
        for problem in problems:
            write(f"  - {problem}")
    else:
        write("Configuration valid.")
    write("Dry run: no server, worker or client was started.")


def print_plan_details(config: Config, facts: dict, host: dict, out=sys.stdout) -> None:
    """Everything a standard run would launch, in order, without launching it."""

    def write(text=""):
        print(text, file=out)

    profile = require_profile(config)
    write()
    write(f"Profile: {profile.key} ({profile.title}), status {profile.status}")
    write(f"  file {profile.path} (sha256 {(profile.sha256 or '')[:16]}...)")
    write(f"  source {profile.source.get('repo')} at {profile.source.get('revision')}")
    write(
        f"  Stark context: minimum {config.min_context():,}, maximum {profile.max_context:,}, "
        f"burst {profile.burst_context:,}; served model id {profile.served_model_id}"
    )
    write(
        f"  chat adapter {profile.chat_adapter}; images "
        + ("must be loaded" if profile.image_capable else "not applicable (text-only)")
    )
    deviations = standard_deviations(config, facts)
    write()
    if deviations:
        write("Standard run: no. This run is diagnostic and stays off the scoreboard:")
        for reason in deviations:
            write(f"  - {reason}")
    else:
        write("Standard run: yes.")
    counterparts = matched_counterparts(config.arms)
    write()
    write("Arms, artifacts and matching:")
    for arm in config.arms:
        write(
            f"  {arm.name}: artifact {arm.artifact or '-'} [{arm.quantization or '-'}], "
            f"match class {arm.match or 'none'}"
        )
        if arm.engine == "stark":
            other = counterparts.get(arm.name)
            write(
                f"    matched reference arm: {other}"
                if other
                else "    matched reference arm: none in this run (ratios against other "
                "arms compare artifacts as well as engines)"
            )
        for reason in arm_disqualifications(arm):
            write(f"    off the scoreboard: {reason}")
    write()
    write(f"Layout: {config.layout}. Session order:")
    for step, (kind, repeat, arm) in enumerate(planned_sessions(config), 1):
        write(f"  {step:>3}. repeat {repeat}  {kind:<7} {arm.name}")
    write()
    write("Launch per session (ports are chosen at run time):")
    for arm in config.arms:
        engine = ENGINES[arm.engine]
        for kind in config.sessions:
            write(f"  {arm.name} / {kind}:")
            for line in engine.launch_plan(config, arm, kind):
                write(f"    {line}")
    if "probe" in config.sessions:
        write()
        write("Client per probe session (model id is the served id of the arm):")
        command = llmprobe_command(
            config, 0, "<model-id>", Path("<scratch>/report.json")
        )
        command[2] = f"http://{HOST}:<port>"
        write("    " + " ".join(command))
        write(
            "    environment: HOME=<scratch>, LLMPROBE_INKLING_LOCAL_TELEMETRY=1, "
            f"LLMPROBE_INKLING_LOCAL_CACHE_TAG={config.cache_tag_base}<repeat, two digits>; "
            "LLMPROBE_*, MLX_SERVE_* and API keys removed"
        )
    if "prefill" in config.sessions:
        write()
        write(
            f"Prefill control: {EXACT_PREFILL_TOKENS:,} token IDs from seed "
            f"{EXACT_PREFILL_SEED}-<trial>, cap {EXACT_PREFILL_CAP}, raw IDs to /v1/completions, "
            "no draft field; input-ID hashes are compared across arms"
        )
    if "burst" in config.sessions:
        write(
            f"Burst: {BURST_WIDTH} streams x {BURST_CAP} tokens behind one barrier, drafting off"
        )
    write()
    quality = profile.quality
    write("Quality slot (engine diagnostic, attached after the run):")
    stark = [arm.name for arm in config.arms if arm.engine == "stark"]
    write(f"  required for: {', '.join(stark) or 'no Stark arm in this run'}")
    write(
        f"  reference: artifact {quality.get('reference_artifact') or '-'}, "
        f"path {quality.get('reference_path') or '-'}; probe {quality.get('probe') or 'not configured'}"
    )
    if quality.get("probe_command"):
        write(
            "  probe command (not run by this tool): "
            + " ".join(quality["probe_command"])
        )
    write(
        f"  pass: next-token accuracy not more than {QUALITY_MAX_ACCURACY_SHORTFALL * 100:.1f} "
        f"points below the reference's on the pooled rows, and top-1 agreement at least "
        f"{QUALITY_MIN_CONFIDENT_AGREEMENT:.0%} where the reference's top probability is at least "
        f"{QUALITY_CONFIDENT_PROBABILITY:g}"
    )
    write(
        f"  not assessed: not natural text, or the reference predicts under "
        f"{QUALITY_MIN_REFERENCE_ACCURACY:.0%} of next tokens. Without a pass an arm is speed "
        "only and holds no standing"
    )
    write(
        "  attach: quality attach --run <run> --arm <arm> --report <file> [--report <file> ...] "
        "--candidate-artifact <dir> --corpus-kind natural-text"
    )
    warnings = host_warnings(host)
    if warnings:
        write()
        write("Host notes (recorded, not refused):")
        for warning in warnings:
            write(f"  - {warning}")


def planned_sessions(config: Config) -> list[tuple[str, int, Arm]]:
    """(kind, repeat, arm) for every server session a run starts, in order."""
    plan = []
    if config.layout == "blocks":
        for repeat in range(config.repeats):
            for kind in SESSION_KINDS:
                if kind in config.sessions:
                    plan.extend(
                        (kind, repeat, arm) for arm in session_order(config, repeat)
                    )
        return plan
    if "probe" in config.sessions:
        for repeat in range(config.repeats):
            plan.extend(("probe", repeat, arm) for arm in config.arms)
    for kind in ("prefill", "burst"):
        if kind in config.sessions:
            plan.extend((kind, 0, arm) for arm in config.arms)
    return plan


# --------------------------------------------------------------------------
# Shared OpenAI-compatible client pieces (every engine)
# --------------------------------------------------------------------------


def http_request(port, route, body=None, timeout=900):
    return urlopen(
        Request(
            f"http://{HOST}:{port}{route}",
            data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        ),
        timeout=timeout,
    )


def fetch(port, route, body=None):
    with http_request(port, route, body) as response:
        return json.load(response)


def native_metrics(port) -> dict:
    with http_request(port, "/metrics") as response:
        pairs = [
            line.split()
            for line in response.read().decode().splitlines()
            if line.startswith((PREFILL_METRIC + " ", DECODE_METRIC + " "))
        ]
    if len(pairs) != 2:
        raise BenchError("native_metrics_missing")
    return {name: float(value) for name, value in pairs}


def completion(port, model, prompt, cap, draft, salt, ignore_eos=False) -> dict:
    """One streamed /v1/completions request; timings and counts only.

    draft None sends no draft field (engine default); True or False sends
    enable_mtp explicitly.
    """
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": cap,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0,
        "top_p": 1,
        "top_k": 0,
        "min_p": 0,
        "repetition_penalty": 1,
        "enable_pld": False,
        "enable_drafter": False,
        "cache_salt": salt,
    }
    if draft is not None:
        body["enable_mtp"] = draft
    if ignore_eos:
        body["ignore_eos"] = True
    start = time.perf_counter()
    first = last = usage = finish = None
    done = False
    with http_request(port, "/v1/completions", body) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            stamp = time.perf_counter()
            item = line[6:].rstrip()
            if item == b"[DONE]":
                done = True
                break
            event = json.loads(item)
            if event.get("usage") is not None:
                usage = event["usage"]
            for choice in event.get("choices", []):
                finish = choice.get("finish_reason") or finish
                if choice.get("text"):
                    first = first or stamp
                    last = stamp
    wall = time.perf_counter() - start
    if usage is None or not done or first is None:
        raise BenchError("incomplete_streaming_usage")
    generated = usage["completion_tokens"]
    return {
        "prompt_tokens": usage["prompt_tokens"],
        "generated_tokens": generated,
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get(
            "cached_tokens"
        ),
        "finish_reason": finish if finish in FINISH_REASONS else None,
        "client_ttft_s": first - start,
        "client_wall_s": wall,
        "client_delivery_s": last - first,
        "client_useful_decode_tok_s": (generated - 1) / (last - first)
        if last > first
        else None,
        "drafts_requested": "engine_default"
        if draft is None
        else ("on" if draft else 0),
        # Only the Stark gateway reports this; MLX-Serve reports through its log.
        "engine_mtp": numeric_tree(usage.get("stark_mtp")),
    }


def unseen_corpus(
    node: str, llmprobe: Path, seed: str, size: int = EXACT_PREFILL_CORPUS_BYTES
) -> str:
    """llmprobe's own unseen-text builder, evaluated by Node in memory."""
    script = (
        "import { buildUnseenText } from './src/bench/corpus.ts'; "
        "process.stdout.write(buildUnseenText(process.argv[1], Number(process.argv[2])))"
    )
    return subprocess.check_output(
        [node, "--input-type=module", "-e", script, seed, str(size)],
        cwd=llmprobe,
        env=client_env(),
        text=True,
        stderr=subprocess.DEVNULL,
        timeout=120,
    )


def exact_prefill(
    port,
    model,
    receipts: Receipts,
    *,
    corpus,
    warmups=1,
    repeats=3,
    seed_prefix=EXACT_PREFILL_SEED,
    tokens=EXACT_PREFILL_TOKENS,
    first=0,
) -> None:
    """Warmup trials, then measured trials numbered from warmups + first.

    The trial number selects the seed, so a session holding one measured
    trial (the block layout) uses the same inputs as one holding all of them.
    """
    for step in range(warmups + repeats):
        index = step if step < warmups else step + first
        text = corpus(f"{seed_prefix}-{index}")
        ids = fetch(port, "/tokenize", {"content": text})["tokens"][:tokens]
        del text
        if len(ids) != tokens:
            raise BenchError("unseen_corpus_too_short")
        before = native_metrics(port)
        result = completion(
            port, model, ids, EXACT_PREFILL_CAP, None, f"exact-unseen-{index}"
        )
        after = native_metrics(port)
        prefill = after[PREFILL_METRIC] - before[PREFILL_METRIC]
        decode = after[DECODE_METRIC] - before[DECODE_METRIC]
        valid = result["prompt_tokens"] == tokens and result["cached_tokens"] == 0
        receipts.write(
            "prefill-8800",
            {
                "trial": index,
                "warmup": step < warmups,
                "valid": valid,
                "seed": f"{seed_prefix}-{index}",
                "input_ids_le_u32_sha256": ids_digest(ids),
                **result,
                "native_prefill_s": prefill,
                "native_decode_s": decode,
                "native_prefill_tok_s": result["prompt_tokens"] / prefill
                if prefill > 0
                else None,
                "client_prompt_tok_s": result["prompt_tokens"] / result["client_ttft_s"]
                if result["client_ttft_s"] > 0
                else None,
            },
        )


def burst_prompts(width=BURST_WIDTH) -> list[str]:
    """The matched burst deck of the first recorded comparison (slot 6 replaced)."""
    prompts = [
        f"[load trial request {i:02d}] Begin now with the numbers. Count in increasing order from {1000 + i * 321}. "
        "Write one number on each line for at least 600 lines. Use no explanation or headings."
        for i in range(width)
    ]
    if width > 6:
        prompts[6] = (
            "[load trial request 06] Write the complete list of consecutive integers "
            "from 7001 through 8000, one integer per line. Begin with 7001. "
            "Write every integer in order and stop after 8000. Do not abbreviate the list."
        )
    return prompts


def burst(
    port,
    model,
    receipts: Receipts,
    *,
    width=BURST_WIDTH,
    cap=BURST_CAP,
    warmups=1,
    repeats=3,
    first=0,
) -> None:
    """The published 16-user protocol: drafting explicitly off on every request."""
    prompts = burst_prompts(width)
    for step in range(warmups + repeats):
        index = step if step < warmups else step + first
        gate = threading.Barrier(width + 1)

        def one(slot, gate=gate, index=index):
            gate.wait()
            result = completion(
                port, model, prompts[slot], cap, False, f"burst-{index}-{slot}"
            )
            result["input_utf8_sha256"] = text_digest(prompts[slot])
            return result

        before = native_metrics(port)
        with ThreadPoolExecutor(max_workers=width) as executor:
            futures = [executor.submit(one, slot) for slot in range(width)]
            gate.wait()
            start = time.perf_counter()
            requests = [future.result() for future in futures]
            wall = time.perf_counter() - start
        after = native_metrics(port)
        total = sum(item["generated_tokens"] for item in requests)
        valid = all(
            item["generated_tokens"] == cap
            and item["finish_reason"] == "length"
            and item["cached_tokens"] == 0
            for item in requests
        )
        receipts.write(
            "burst",
            {
                "trial": index,
                "warmup": step < warmups,
                "valid": valid,
                "streams": width,
                "cap": cap,
                "total_generated_tokens": total,
                "wall_s": wall,
                "aggregate_tok_s": total / wall,
                "length_finished": sum(
                    item["finish_reason"] == "length" for item in requests
                ),
                "native_prefill_request_seconds_sum": after[PREFILL_METRIC]
                - before[PREFILL_METRIC],
                "native_decode_request_seconds_sum": after[DECODE_METRIC]
                - before[DECODE_METRIC],
                "requests": requests,
            },
        )


# --------------------------------------------------------------------------
# llmprobe runner
# --------------------------------------------------------------------------


def sanitize_sample(raw: dict) -> dict | None:
    """Keep the whitelisted scalar receipt fields; drop everything else."""
    if not isinstance(raw, dict) or not isinstance(raw.get("label"), str):
        return None
    sample = {}
    for key, kind in SAMPLE_FIELDS.items():
        value = raw.get(key)
        if (
            value is not None
            and isinstance(value, kind)
            and (kind is bool or not isinstance(value, bool))
        ):
            sample[key] = value
        else:
            sample[key] = None
    if sample["inputUtf8Sha256"] is not None and not re.fullmatch(
        r"[0-9a-f]{64}", sample["inputUtf8Sha256"]
    ):
        sample["inputUtf8Sha256"] = None
    if sample["finishReason"] not in (None, *FINISH_REASONS, "stop_sequence"):
        sample["finishReason"] = None
    scenario = scenario_for_label(sample["label"])
    if scenario is None:
        sample["label"] = "unrecognized"
    sample["scenario"] = scenario
    return sample


def sanitize_report(report: dict) -> dict:
    bench = report.get("bench") or {}
    target = report.get("target") or {}
    return {
        "bench": {key: bench.get(key) for key in BENCH_FIELDS},
        "target": {
            "engine": target.get("engine"),
            "model": target.get("model"),
            "engineSettings": target.get("engineSettings"),
        },
        "machine": report.get("machine"),
        "run": {
            key: (report.get("run") or {}).get(key)
            for key in ("depth", "mode", "startedAt")
        },
    }


def llmprobe_command(config: Config, port: int, model_id: str, save: Path) -> list[str]:
    command = [
        config.node,
        str(config.llmprobe / "bin/dist/llmprobe.mjs"),
        f"http://{HOST}:{port}",
        "--model",
        model_id,
        "--bench-only",
        "--no-save",
        "--save",
        str(save),
    ]
    if config.probe_runs:
        command.extend(["--runs", str(config.probe_runs)])
    if config.reasoning and config.reasoning != REASONING_CLIENT_DEFAULT:
        command.extend(["--reasoning", config.reasoning])
    return command


def run_llmprobe(
    config: Config, port: int, model_id: str, directory: Path, cache_tag: str
) -> dict:
    """Run one llmprobe --bench-only invocation; keep sanitized receipts only."""
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    receipts = Receipts(directory)
    with tempfile.TemporaryDirectory(prefix="llmprobe-", dir=directory) as scratch:
        save = Path(scratch) / "report.json"
        env = client_env()
        env["HOME"] = scratch
        env["LLMPROBE_INKLING_LOCAL_TELEMETRY"] = "1"
        env["LLMPROBE_INKLING_LOCAL_CACHE_TAG"] = cache_tag
        count = 0
        with subprocess.Popen(
            llmprobe_command(config, port, model_id, save),
            env=env,
            cwd=scratch,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        ) as process:
            for line in process.stdout:
                match = TELEMETRY.match(line)
                if match is None:
                    continue
                with contextlib.suppress(ValueError):
                    sample = sanitize_sample(json.loads(match[1]))
                    if sample is not None:
                        receipts.write("samples", sample)
                        count += 1
        if process.returncode or not save.is_file():
            raise BenchError(f"llmprobe_failed:{process.returncode}")
        report = json.loads(save.read_text())
        if not report.get("bench"):
            raise BenchError("llmprobe_bench_missing")
        sanitized = sanitize_report(report)
    (directory / "bench.json").write_text(
        json.dumps(sanitized, indent=2, sort_keys=True) + "\n"
    )
    return {"samples": count, "cache_tag": cache_tag}


# --------------------------------------------------------------------------
# Stark gateway
# --------------------------------------------------------------------------


def choose_context(
    burst_mode: bool,
    total: int,
    min_context: int,
    max_context: int,
    burst_context: int,
) -> int:
    """Context sizing only; it does not decide drafting."""
    if burst_mode:
        return burst_context
    if total <= min_context:
        return min_context
    return min(max_context, (total + CONTEXT_STEP - 1) // CONTEXT_STEP * CONTEXT_STEP)


def draft_fields(mode: str, *, burst_mode: bool, enable_mtp) -> tuple[dict, object]:
    """Worker request fields for drafting, and the label recorded per request.

    engine: forward what the client asked for. No client field means no worker
      field, so the worker's own default applies; enable_mtp true becomes
      "mtp": true (the worker's recommended drafts); false becomes 0.
    off: always 0. The burst session is always off (published protocol).
    """
    if burst_mode or mode == "off" or enable_mtp is False:
        return {"mtp_drafts": 0}, 0
    if mode != "engine":
        raise BenchError("unknown_draft_mode")
    if enable_mtp is True:
        return {"mtp": True}, "on"
    return {}, "engine_default"


def engine_mtp_report(reply: dict) -> dict | None:
    """What the worker says it did: effective drafts, fallback and draft counts."""
    mtp = reply.get("mtp")
    if not isinstance(mtp, dict):
        return None
    stats = mtp.get("stats") if isinstance(mtp.get("stats"), dict) else {}
    speculative = (
        stats.get("speculative") if isinstance(stats.get("speculative"), dict) else {}
    )
    return numeric_tree(
        {
            "enabled": mtp.get("enabled"),
            "drafts": mtp.get("drafts"),
            "requested_drafts": mtp.get("requested_drafts"),
            "fallback_reason": mtp.get("fallback_reason"),
            "verification": mtp.get("verification"),
            "proposed": stats.get("proposed"),
            "accepted": stats.get("accepted"),
            "rounds": speculative.get("rounds"),
        }
    )


class Gateway(ThreadingHTTPServer):
    """Loopback OpenAI-compatible surface over one serialized Stark worker."""

    daemon_threads = True
    request_queue_size = 64

    def __init__(
        self,
        worker,
        *,
        receipts: Receipts,
        burst_mode: bool,
        cache: bool,
        renderer,
        stream_factory,
        parse_reply,
        model_id: str,
        min_context: int,
        max_context: int,
        burst_context: int,
        vocab_size: int,
        draft_mode="engine",
        effort_aliases=None,
        prompt_aware=False,
    ):
        self.worker = worker
        self.receipts = receipts
        self.burst_mode = burst_mode
        self.cache = cache
        self.draft_mode = draft_mode
        self.renderer = renderer
        self.stream_factory = stream_factory
        self.parse_reply = parse_reply
        self.effort_aliases = dict(effort_aliases or {})
        self.model_id = model_id
        self.min_context = min_context
        self.max_context = max_context
        self.burst_context = burst_context
        self.vocab_size = vocab_size
        self.prompt_aware = prompt_aware
        self.metric_lock = threading.Lock()
        self.totals = {PREFILL_METRIC: 0.0, DECODE_METRIC: 0.0}
        super().__init__((HOST, 0), GatewayHandler)

    def handle_error(self, request, client_address):
        self.receipts.write("adapter-errors", {"error": "transport_failure"})

    def record_generation(
        self, surface, reply, ids, cap, requested, engine_mtp, context
    ):
        with self.metric_lock:
            self.totals[PREFILL_METRIC] += reply.get("prefill_seconds") or 0.0
            self.totals[DECODE_METRIC] += reply.get("decode_seconds") or 0.0
        generated = reply.get("generated_token_ids") or []
        self.receipts.write(
            "native",
            {
                "event": "native_generation",
                "surface": surface,
                "input_tokens": reply.get("input_tokens"),
                "input_ids_le_u32_sha256": ids_digest(ids),
                "output_tokens": reply.get("output_tokens"),
                "sampled_tokens": reply.get("sampled_tokens"),
                "output_ids_le_u32_sha256": ids_digest(generated)
                if generated
                else None,
                "cache_reuse": reply.get("reused_input_tokens", 0),
                "cap": cap,
                "drafts_requested": requested,
                "engine_mtp": engine_mtp,
                "context": context,
                "finish": reply.get("finish_reason"),
                "prefill_s": reply.get("prefill_seconds"),
                "decode_s": reply.get("decode_seconds"),
                "native_prefill_tok_s": reply.get("input_tokens_per_second"),
                "native_decode_tok_s": reply.get("output_tokens_per_second"),
            },
        )


class GatewayHandler(BaseHTTPRequestHandler):
    server: Gateway

    def setup(self):
        super().setup()
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def log_message(self, *_):
        return

    def respond(self, value, code=200):
        data = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def error(self, code, reason):
        self.respond(
            {
                "error": {
                    "code": reason,
                    "type": "invalid_request_error",
                    "message": reason,
                }
            },
            code,
        )

    def do_GET(self):
        if self.path in ("/v1/models", "/models"):
            self.respond(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.server.model_id,
                            "object": "model",
                            "created": 0,
                            "owned_by": "stark",
                        }
                    ],
                }
            )
        elif self.path in ("/health", "/v1/health"):
            self.respond({"status": "ok"})
        elif self.path == "/metrics":
            with self.server.metric_lock:
                lines = "".join(
                    f"{name} {value:.12f}\n"
                    for name, value in self.server.totals.items()
                ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(lines)))
            self.end_headers()
            self.wfile.write(lines)
        else:
            self.error(404, "not_found")

    def tokenize(self, text):
        reply = self.server.worker.request(
            {"operation": "tokenize", "text": text, "add_special_tokens": False}
        )
        if reply.get("error") or "token_ids" not in reply:
            raise ValueError("tokenization_failed")
        return reply["token_ids"]

    def do_POST(self):
        if self.path not in ("/tokenize", "/v1/chat/completions", "/v1/completions"):
            self.error(404, "not_found")
            return
        try:
            size = int(self.headers.get("Content-Length", 0))
            if not 0 < size <= 2 * 1024 * 1024:
                raise ValueError("invalid_body_size")
            body = json.loads(self.rfile.read(size))
            if self.path == "/tokenize":
                self.respond({"tokens": self.tokenize(body["content"])})
                return
            self.complete(body)
        except (BrokenPipeError, ConnectionResetError):
            return
        except (ValueError, KeyError, TypeError, RuntimeError):
            self.error(400, "invalid_benchmark_request")

    def complete(self, body):
        server = self.server
        if body.get("ignore_eos") or body.get("min_tokens"):
            # The worker has no EOS override; llmprobe then measures to checkpoint EOS.
            self.error(400, "unsupported_stop_override")
            return
        chat = self.path == "/v1/chat/completions"
        if chat:
            effort = body.get("reasoning_effort")
            effort = server.effort_aliases.get(effort, effort)
            kwargs = body.get("chat_template_kwargs") or {}
            off = kwargs.get("enable_thinking") is False or effort in (None, "none")
            prompt = server.renderer(
                body["messages"], thinking=not off, effort=None if off else effort
            )
        else:
            prompt = body["prompt"]
        ids = self.tokenize(prompt) if isinstance(prompt, str) else prompt
        if (
            not isinstance(ids, list)
            or not ids
            or any(
                type(token) is not int or not 0 <= token < server.vocab_size
                for token in ids
            )
        ):
            raise ValueError("invalid_prompt_tokens")
        cap = body.get("max_completion_tokens", body.get("max_tokens", DECODE_CAP))
        if type(cap) is not int or cap <= 0:
            raise ValueError("invalid_output_cap")
        context = choose_context(
            server.burst_mode,
            len(ids) + cap,
            server.min_context,
            server.max_context,
            server.burst_context,
        )
        if len(ids) + cap > context:
            self.error(400, "context_overflow")
            return
        stream = body.get("stream") is True
        temperature = body.get("temperature", 0) or 0
        sampling = {"mode": "greedy"}
        if temperature:
            sampling = {
                "mode": "stochastic",
                "temperature": temperature,
                "top_p": body.get("top_p", 1.0),
            }
        enable_mtp = body.get("enable_mtp")
        fields, requested = draft_fields(
            server.draft_mode,
            burst_mode=server.burst_mode,
            enable_mtp=enable_mtp if isinstance(enable_mtp, bool) else None,
        )
        request = {
            "operation": "generate",
            "input_token_ids": ids,
            "context_tokens": context,
            "max_new_tokens": cap,
            "sampling": sampling,
            "stream": stream,
            "seed": body.get("seed") or 0,
            **fields,
        }
        if (
            chat
            and not server.burst_mode
            and server.cache
            and getattr(server.worker, "prefix_cache", False)
        ):
            request["cache_scope"] = STARK_CACHE_SCOPE
        if body.get("stop"):
            request["stop"] = (
                [body["stop"]] if isinstance(body["stop"], str) else body["stop"]
            )

        shell = {
            "id": "stark-benchmark",
            "created": 0,
            "model": server.model_id,
            "object": "chat.completion.chunk"
            if chat and stream
            else "chat.completion"
            if chat
            else "text_completion",
        }
        header_sent = False

        def event(value):
            nonlocal header_sent
            if not header_sent:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                header_sent = True
            data = "data: " + (
                value
                if isinstance(value, str)
                else json.dumps(value, ensure_ascii=False)
            )
            self.wfile.write((data + "\n\n").encode())
            self.wfile.flush()

        def delta(kind, text):
            payload = {"delta": {kind: text}} if chat else {"text": text}
            event(
                {**shell, "choices": [{"index": 0, **payload, "finish_reason": None}]}
            )

        # A prompt-aware adapter needs the rendered prompt to know whether a
        # reply opens inside a reasoning block. It is used in memory only.
        aware = {"prompt": prompt} if chat and server.prompt_aware else {}
        parser = (
            server.stream_factory(
                lambda text: delta("content", text),
                lambda text: delta("reasoning_content", text),
                **aware,
            )
            if chat
            else None
        )
        callback = (
            parser.feed if parser is not None else (lambda text: delta("content", text))
        )
        reply = server.worker.request(request, on_delta=callback if stream else None)
        if reply.get("error"):
            # Worker errors are fixed codes; anything else is reduced to a generic one.
            raw = reply["error"]
            code = (
                raw
                if isinstance(raw, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", raw)
                else "worker_error"
            )
            if header_sent:
                event({"error": {"type": "invalid_request_error", "message": code}})
                event("[DONE]")
            else:
                self.error(400, code)
            return
        engine_mtp = engine_mtp_report(reply)
        server.record_generation(
            "chat" if chat else "completion",
            reply,
            ids,
            cap,
            requested,
            engine_mtp,
            context,
        )
        finish = (
            "length" if reply.get("finish_reason") in ("length", "context") else "stop"
        )
        usage = {
            "prompt_tokens": reply["input_tokens"],
            "completion_tokens": reply["output_tokens"],
            "total_tokens": reply["input_tokens"] + reply["output_tokens"],
            "prompt_tokens_details": {
                "cached_tokens": reply.get("reused_input_tokens", 0)
            },
            "stark_mtp": {"requested": requested, **(engine_mtp or {})},
        }
        message = server.parse_reply(reply["text"], **aware) if chat else None
        if stream:
            if parser is not None:
                parser.finish(message, **({"text": reply["text"]} if aware else {}))
            end = {"delta": {}} if chat else {"text": ""}
            event({**shell, "choices": [{"index": 0, **end, "finish_reason": finish}]})
            event({**shell, "choices": [], "usage": usage})
            event("[DONE]")
        else:
            result = {"message": message} if chat else {"text": reply["text"]}
            self.respond(
                {
                    **shell,
                    "choices": [{"index": 0, **result, "finish_reason": finish}],
                    "usage": usage,
                }
            )


class ThinkTagStream:
    """Stream splitter for families that mark reasoning with <think> tags."""

    def __init__(self, content, reasoning, *, prompt=""):
        from stark_convert.reasoning import ReasoningStream, prompt_opens_reasoning

        self.inner = ReasoningStream(
            content, reasoning, initial_thinking=prompt_opens_reasoning(prompt)
        )

    def feed(self, text):
        self.inner.feed(text)

    def finish(self, message, *, text=""):
        self.inner.finish(text)


def chat_adapters(model: Path, adapter: str):
    """Production renderer, stream splitter and parser from stark_convert.

    Returns (renderer, stream factory, reply parser, prompt_aware). The
    profile names the adapter, and it is checked against the artifact's own
    chat template so a profile cannot split replies the wrong way unnoticed.
    """
    from stark_convert.api_tools import tool_parser_for
    from stark_convert.chat import load_template, render_template
    from stark_convert.inkling_chat import PARSER as INKLING_PARSER
    from stark_convert.reasoning import reasoning_options

    codec, template = load_template(model)
    if (tool_parser_for(codec) == INKLING_PARSER) != (adapter == INKLING_PARSER):
        raise BenchError("chat_adapter_does_not_match_template")

    def renderer(messages, *, thinking, effort):
        return render_template(
            template,
            messages=messages,
            tools=None,
            add_generation_prompt=True,
            bos_token=codec.get("bos_token") or "",
            eos_token=codec.get("eos_token") or "",
            **reasoning_options(codec, thinking=thinking, effort=effort),
        )

    if adapter == INKLING_PARSER:
        from stark_convert.inkling_chat import ResponseStream, parse_response

        return renderer, ResponseStream, (lambda text: parse_response(text)[0]), False
    if adapter == "think-tags-v1":
        from stark_convert.reasoning import assistant_message, prompt_opens_reasoning

        def parse_reply(text, *, prompt=""):
            return assistant_message(
                text, initial_thinking=prompt_opens_reasoning(prompt)
            )

        return renderer, ThinkTagStream, parse_reply, True
    raise BenchError("unknown_chat_adapter")


def model_vocab(model: Path, fallback: int) -> int:
    with contextlib.suppress(OSError, ValueError, TypeError, AttributeError):
        config = json.loads((model / "config.json").read_text())
        for source in (config, config.get("text_config") or {}):
            if type(source.get("vocab_size")) is int and source["vocab_size"] > 0:
                return source["vocab_size"]
    return fallback


def default_worker_factory(
    binary: Path, model: Path, timeout: float, image_check_dir: Path | None
):
    from stark_convert.chat import Worker

    worker = Worker(binary, model, timeout)
    worker.image_check_dir = image_check_dir
    return worker


def readiness_mtp(ready: dict) -> dict | None:
    """The worker's own advertised drafting defaults, from its readiness reply."""
    mtp = ready.get("mtp")
    if not isinstance(mtp, dict):
        return None
    keys = (
        "supported",
        "max_drafts",
        "recommended_drafts",
        "default_drafts",
        "verification",
        "default_policy",
    )
    return numeric_tree({key: mtp.get(key) for key in keys})


def worker_min_context(config, profile: Profile) -> int:
    """The smallest planned worker context: the profile's unless overridden."""
    return getattr(config, "stark_min_context", None) or profile.min_context


@contextlib.contextmanager
def stark_session(
    config,
    arm: Arm,
    receipts: Receipts,
    kind: str,
    *,
    profile: Profile | None = None,
    worker_factory=default_worker_factory,
    adapters=None,
):
    """One Stark worker behind one gateway. kind is probe, prefill or burst."""
    profile = profile or require_profile(config)
    adapters = adapters or (lambda model: chat_adapters(model, profile.chat_adapter))
    removed = sorted(
        {key for key in os.environ if key.startswith(REMOVED_ENV_PREFIXES)}
    )
    try:
        with scoped_environ(dict(arm.env), REMOVED_ENV_PREFIXES):
            worker = worker_factory(
                arm.binary,
                arm.model,
                config.stark_worker_timeout,
                config.stark_image_check_dir,
            )
    except Exception as error:
        write_startup_failure(receipts, kind, failure_code(error), [])
        raise
    try:
        ready = worker.read()
        parsed = getattr(worker, "mtp", None) or {}
        receipts.write(
            "native",
            {
                "event": "worker_ready",
                "session": kind,
                "ready": bool(ready.get("ready")),
                "image_loaded": bool(getattr(worker, "images", False)),
                "image_verification": getattr(worker, "image_verification", None),
                "readiness_mtp": readiness_mtp(ready),
                "client_mtp_supported": bool(parsed.get("supported")),
                "prefix_cache": bool(getattr(worker, "prefix_cache", False)),
                "removed_environment_names": removed,
                "diagnostic_env": dict(arm.env),
                "draft_mode": "off" if kind == "burst" else config.stark_drafts,
            },
        )
        if not ready.get("ready"):
            raise BenchError("stark_worker_not_ready")
        # An image-capable model is measured with images resident: text speed
        # with images loaded is the number that ships.
        if profile.image_capable and not getattr(worker, "images", False):
            raise BenchError("stark_images_not_loaded")
        renderer, stream_factory, parse_reply, *aware = adapters(arm.model)
        gateway = Gateway(
            worker,
            receipts=receipts,
            burst_mode=kind == "burst",
            cache=kind == "probe",
            renderer=renderer,
            stream_factory=stream_factory,
            parse_reply=parse_reply,
            model_id=profile.served_model_id,
            min_context=worker_min_context(config, profile),
            max_context=profile.max_context,
            burst_context=profile.burst_context,
            vocab_size=model_vocab(arm.model, profile.vocab_size),
            draft_mode=config.stark_drafts,
            effort_aliases=config.stark_effort_aliases,
            prompt_aware=bool(aware and aware[0]),
        )
    except Exception as error:
        # The worker's own stderr is not captured by stark_convert, so a failed
        # start is recorded by its fixed failure code only.
        write_startup_failure(receipts, kind, failure_code(error), [])
        worker.close()
        raise
    try:
        with gateway:
            driver = threading.Thread(target=gateway.serve_forever, daemon=True)
            driver.start()
            try:
                yield gateway.server_port, profile.served_model_id
            finally:
                gateway.shutdown()
                driver.join(timeout=5)
    finally:
        worker.close()


# --------------------------------------------------------------------------
# Reference engines
# --------------------------------------------------------------------------


class StartupTail:
    """A bounded tail of a server's console output from before it is ready.

    Only startup lines are kept (arguments, model and weight loading, load
    errors); collection stops once the server answers, before any benchmark
    request is sent, so no prompt or completion text can enter it. It is
    written to the private run directory only when the start fails.
    """

    LIMIT = 60
    WIDTH = 300

    def __init__(self):
        self.lock = threading.Lock()
        self.active = True
        self.lines: list[str] = []

    def add(self, line: str) -> None:
        with self.lock:
            if not self.active:
                return
            clean = "".join(
                ch if ch.isprintable() else " " for ch in line.rstrip("\n")
            )[: self.WIDTH]
            self.lines = (self.lines + [clean])[-self.LIMIT :]

    def stop(self) -> None:
        with self.lock:
            self.active = False
            self.lines = []

    def snapshot(self) -> list[str]:
        with self.lock:
            return list(self.lines)


def write_startup_failure(
    receipts: Receipts, kind: str, code: str, lines: list[str]
) -> None:
    """Persist a failed start: the tail file plus a receipt naming it."""
    name = f"{kind}-server-startup-tail.txt"
    receipts.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    (receipts.directory / name).write_text("\n".join(lines) + ("\n" if lines else ""))
    errors = [
        line[:200]
        for line in lines
        if re.search(r"error|failed|fatal|panic|not found", line, re.IGNORECASE)
    ]
    receipts.write(
        "native",
        {
            "event": "server_start_failed",
            "session": kind,
            "failure": code,
            "tail_file": name,
            "tail_lines": len(lines),
            "error_lines": errors[-5:],
        },
    )


def mlx_server_records(
    lines, receipts: Receipts, tail: StartupTail | None = None, log_events=()
) -> None:
    """Keep only numeric MLX-Serve telemetry from its console stream."""
    for line in lines:
        if tail is not None:
            tail.add(line)
        if "[spec-stats]" in line:
            suffix = line.partition("[spec-stats]")[2]
            mode = re.search(r"\bmode=([a-zA-Z0-9_-]{1,32})", suffix)
            fields = {
                key: float(value)
                for key, value in re.findall(rf"\b([a-z_]+)=({NUMBER})(?:\b|%)", suffix)
                if key in MLX_SPEC_FIELDS
            }
            if mode:
                receipts.write(
                    "native", {"event": "spec", "mode": mode[1], "fields": fields}
                )
        match = MLX_PERF.search(line)
        if match:
            prompt, generated, wall, phase, finish = match.groups()
            safe = phase if re.fullmatch(r"[ a-zA-Z0-9.,:=/%+\-]*", phase) else None
            receipts.write(
                "native",
                {
                    "event": "server_perf",
                    "prompt": int(prompt),
                    "generated": int(generated),
                    "wall_ms": int(wall) if wall is not None else None,
                    "phase_summary": safe,
                    "finish": finish,
                },
            )
        # Lines the profile asks to keep, reduced to a fixed event and a state.
        for spec in log_events:
            if all(part in line for part in spec.all) and (
                not spec.any or any(part in line for part in spec.any)
            ):
                state = next(
                    (word for word in spec.states if word in line), "mentioned"
                )
                receipts.write("native", {"event": spec.event, "state": state})


def mlx_command(
    binary: Path,
    model: Path,
    port: int,
    kind: str,
    force_mtp=False,
    *,
    max_context: int,
    burst_context: int,
    flags=(),
    draft_off_flags=(),
    draft_force_flags=(),
) -> list[str]:
    """MLX-Serve arguments. Drafting is its own default except in the burst."""
    if kind == "burst":
        context, concurrency, entries = burst_context, BURST_WIDTH, 0
    elif kind == "prefill":
        context, concurrency, entries = max_context, 1, 0
    else:
        context, concurrency, entries = max_context, 1, None
    command = [
        str(binary),
        "--model",
        str(model),
        "--serve",
        "--host",
        HOST,
        "--port",
        str(port),
        "--ctx-size",
        str(context),
        "--max-concurrent",
        str(concurrency),
        *flags,
    ]
    if kind == "burst":
        command.extend(draft_off_flags)
    elif force_mtp:
        command.extend(draft_force_flags)
    if entries is not None:
        command.extend(["--prefix-cache-entries", str(entries)])
    return command


class Engine:
    """One way to serve a model on a loopback OpenAI-compatible endpoint.

    `session` is a context manager that yields (port, model id) once the
    endpoint answers. The endpoint must provide /v1/models, /tokenize
    ({"content": text} to {"tokens": [...]}), the chat completions route the
    client uses, streamed /v1/completions that accepts token-ID prompts and
    reports usage, and /metrics with the prefill and decode phase-time
    counters. `launch_plan` describes the same launch for `plan`.
    """

    key = ""
    label = ""

    def session(self, config, arm: Arm, receipts: Receipts, kind: str):
        raise NotImplementedError

    def launch_plan(self, config, arm: Arm, kind: str) -> list[str]:
        return []

    def drafting_plan(self, config) -> str:
        return "engine default"


class StarkEngine(Engine):
    key = "stark"
    label = "Stark"

    def session(self, config, arm, receipts, kind):
        return stark_session(config, arm, receipts, kind)

    def launch_plan(self, config, arm, kind):
        profile = require_profile(config)
        burst = kind == "burst"
        worker = [str(arm.binary), "--model-path", str(arm.model)]
        if profile.image_capable:
            worker += ["--vision-staging-root", "<private temporary directory>"]
        worker.append("--prefix-cache")
        drafting = (
            "forced off"
            if burst or config.stark_drafts == "off"
            else "worker default (no draft field unless the client sends one)"
        )
        context = (
            f"{profile.burst_context:,} per stream"
            if burst
            else f"{worker_min_context(config, profile):,} to {profile.max_context:,} by request size"
        )
        environment = ", ".join(f"{prefix}*" for prefix in REMOVED_ENV_PREFIXES)
        if arm.env:
            environment += " removed; set " + ", ".join(
                f"{key}={value}" for key, value in sorted(arm.env.items())
            )
        else:
            environment += " removed"
        reuse = "on for chat requests" if kind == "probe" else "off"
        return [
            "worker (through stark_convert.chat.Worker): " + " ".join(worker),
            f"environment: {environment}",
            f"gateway: served id {profile.served_model_id}, context {context}, "
            + f"drafting {drafting}, prefix reuse {reuse}",
        ]

    def drafting_plan(self, config):
        return "engine default" if config.stark_drafts == "engine" else "forced off"


class ServerEngine(Engine):
    """A reference engine started as one server process per session.

    A subclass supplies `command` (the full argv) and, optionally, `console`
    (which console lines to keep as numeric receipts). Everything else is
    shared: a private HOME, startup polling, the startup tail kept on a
    failed start, and termination.
    """

    failure_prefix = "server"
    startup_timeout = 300

    def command(self, config, arm: Arm, port: int, kind: str) -> list[str]:
        raise NotImplementedError

    def console(self, config, lines, receipts: Receipts, tail: StartupTail) -> None:
        for line in lines:
            tail.add(line)

    def launch_plan(self, config, arm, kind):
        return [" ".join(self.command(config, arm, "<port>", kind))]

    @contextlib.contextmanager
    def session(self, config, arm, receipts, kind, *, startup_timeout=None):
        port = free_port()
        command = self.command(config, arm, port, kind)
        receipts.write(
            "native",
            {"event": "server_start", "session": kind, "arguments": command[3:]},
        )
        timeout = startup_timeout or self.startup_timeout
        with tempfile.TemporaryDirectory(
            prefix=f"{self.key}-home-", dir=receipts.directory
        ) as home:
            env = client_env()
            env["HOME"] = home
            with subprocess.Popen(
                command,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
            ) as process:
                tail = StartupTail()
                reader = threading.Thread(
                    target=self.console,
                    args=(config, process.stdout, receipts, tail),
                    daemon=True,
                )
                reader.start()
                try:
                    deadline = time.monotonic() + timeout
                    try:
                        while True:
                            if process.poll() is not None:
                                raise BenchError(
                                    f"{self.failure_prefix}_exited:{process.returncode}"
                                )
                            with contextlib.suppress(URLError, OSError, ValueError):
                                models = fetch(port, "/v1/models")["data"]
                                if len(models) == 1:
                                    break
                            if time.monotonic() >= deadline:
                                raise BenchError(
                                    f"{self.failure_prefix}_startup_timeout"
                                )
                            time.sleep(0.5)
                    except BenchError as error:
                        if process.poll() is None:
                            process.terminate()
                            with contextlib.suppress(subprocess.TimeoutExpired):
                                process.wait(15)
                        reader.join(timeout=5)
                        write_startup_failure(
                            receipts, kind, str(error), tail.snapshot()
                        )
                        raise
                    tail.stop()
                    yield port, models[0]["id"]
                finally:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(15)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                    reader.join(timeout=5)
                    receipts.write(
                        "native",
                        {
                            "event": "server_exit",
                            "session": kind,
                            "exit_code": process.returncode,
                        },
                    )


class MlxServeEngine(ServerEngine):
    key = "mlx-serve"
    label = "MLX-Serve"
    failure_prefix = "mlx_serve"

    def command(self, config, arm, port, kind):
        profile = require_profile(config)
        reference = profile.reference(self.key)
        return mlx_command(
            arm.binary,
            arm.model,
            port,
            kind,
            config.mlx_force_mtp,
            max_context=profile.max_context,
            burst_context=profile.burst_context,
            flags=reference.flags,
            draft_off_flags=reference.draft_off_flags,
            draft_force_flags=reference.draft_force_flags,
        )

    def console(self, config, lines, receipts, tail):
        events = require_profile(config).reference(self.key).log_events
        mlx_server_records(lines, receipts, tail, events)

    def drafting_plan(self, config):
        return "--mtp forced" if config.mlx_force_mtp else "engine default"


# Adding a reference engine: subclass ServerEngine (command, console), register
# it here, and declare a [[reference]] block for it in a model profile.
ENGINES: dict[str, Engine] = {
    engine.key: engine for engine in (StarkEngine(), MlxServeEngine())
}


def mlx_session(
    config, arm: Arm, receipts: Receipts, kind: str, *, startup_timeout=300
):
    return ENGINES["mlx-serve"].session(
        config, arm, receipts, kind, startup_timeout=startup_timeout
    )


def engine_session(config, arm, receipts, kind):
    engine = ENGINES.get(arm.engine)
    if engine is None:
        raise BenchError(f"unknown_engine:{arm.engine[:32]}")
    return engine.session(config, arm, receipts, kind)


# --------------------------------------------------------------------------
# Run orchestration
# --------------------------------------------------------------------------


def git_identity() -> dict:
    head = (
        run_text(["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"]) or ""
    ).strip() or None
    status = run_text(["git", "-C", str(REPOSITORY), "status", "--porcelain"])
    return {
        "commit": head,
        "dirty_paths": None if status is None else len(status.splitlines()),
    }


def write_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def failure_code(error: BaseException) -> str:
    """A persistable failure description. BenchError messages are fixed codes;
    stark_convert's ChatError messages are fixed user-facing sentences."""
    if isinstance(error, BenchError):
        return str(error)[:200]
    if type(error).__name__ == "ChatError":
        return f"ChatError: {str(error)[:160]}"
    return type(error).__name__


def harness_files(config) -> dict:
    """SHA-256 of every file that defines what a run does."""
    files = {Path(__file__).name: sha256_file(Path(__file__))}
    entry = getattr(config, "entry", None)
    if entry is not None and Path(entry).is_file():
        files[Path(entry).name] = sha256_file(Path(entry))
    profile = getattr(config, "profile", None)
    if profile is not None and profile.path is not None:
        files[f"benchmark_profiles/{profile.path.name}"] = profile.sha256
    return files


def run_matrix(
    config: Config,
    facts: dict,
    *,
    session_factory=engine_session,
    host=host_snapshot,
    sleep_check=sleep_events,
) -> Path:
    start_snapshot = host()
    require_timing_host(start_snapshot, allow_busy_host=config.allow_busy_host)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = config.output / f"{stamp}-{config.run_label or config.engine}"
    run_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    helper = facts.get("template_helper")
    profile = getattr(config, "profile", None)
    deviations = standard_deviations(config, facts)
    manifest = {
        "schema": SCHEMA,
        "engine": config.engine,
        "status": "running",
        "started_at": now_utc(),
        "harness_sha256": sha256_file(Path(__file__)),
        "harness_files": harness_files(config),
        "repository": git_identity(),
        "profile": profile.facts() if profile else None,
        "llmprobe": facts.get("llmprobe"),
        "node_version": facts.get("node_version"),
        "template_helper_sha256": sha256_file(Path(helper)) if helper else None,
        "config": {
            "repeats": config.repeats,
            "warmups": config.warmups,
            "sessions": list(config.sessions),
            "cache_tag_base": config.cache_tag_base,
            "allow_busy_host": config.allow_busy_host,
            "layout": config.layout,
            **config.protocol(),
        },
        # A standard run follows the protocol without overrides. Anything else
        # is diagnostic: recorded in full, kept off the scoreboard.
        "standard": {"eligible": not deviations, "deviations": deviations},
        "arms": [],
        "host_start": start_snapshot,
        "host_warnings": host_warnings(start_snapshot),
        "power_checks": [],
        "sessions": [],
    }
    for arm in config.arms:
        identity = model_identity(arm.model)
        reasons = arm_disqualifications(arm)
        manifest["arms"].append(
            {
                **arm.identity(),
                "binary_sha256": sha256_file(arm.binary),
                "model": identity,
                "artifact_fingerprint": fingerprint(identity),
                "scoreboard": {
                    "eligible": not deviations and not reasons,
                    "reasons": reasons,
                },
            }
        )
    write_manifest(run_dir / "manifest.json", manifest)
    started = time.time()

    def checkpoint(label):
        snapshot = host()
        manifest["power_checks"].append(
            {
                "before": label,
                "source": snapshot["power"]["source"],
                "process_thermal_state": snapshot["process_thermal_state"],
                "thermal_warning": snapshot["thermal"]["thermal_warning"],
                "captured_at": snapshot["captured_at"],
            }
        )
        if not snapshot["power"]["on_ac"]:
            raise BenchError("power_source_changed")

    failed: set[str] = set()
    manifest["arm_failures"] = []

    def session(arm, kind, directory, body):
        """One server session. A failed arm is recorded and skipped from then on;
        the other arms still run. Losing AC power still stops the whole run."""
        if arm.name in failed:
            return
        receipts = Receipts(directory)
        checkpoint(f"{arm.name}:{kind}")
        begun = time.time()
        try:
            with session_factory(config, arm, receipts, kind) as (port, model_id):
                body(port, model_id, receipts)
        except Exception as error:  # noqa: BLE001 - recorded per arm, the run continues
            failed.add(arm.name)
            manifest["arm_failures"].append(
                {
                    "arm": arm.name,
                    "kind": kind,
                    "directory": str(directory.relative_to(run_dir)),
                    "failure": failure_code(error),
                    "seconds": round(time.time() - begun, 3),
                }
            )
            write_manifest(run_dir / "manifest.json", manifest)
            return
        manifest["sessions"].append(
            {"arm": arm.name, "kind": kind, "seconds": round(time.time() - begun, 3)}
        )
        write_manifest(run_dir / "manifest.json", manifest)

    blocks = config.layout == "blocks"
    # In the block layout a prefill or burst session holds one measured trial,
    # numbered by its repeat so every arm sees the same input in one block.
    trials = 1 if blocks else config.repeats

    def corpus(seed):
        return unseen_corpus(config.node, config.llmprobe, seed)

    try:
        for kind, repeat, arm in planned_sessions(config):
            if kind == "probe":
                tag = f"{config.cache_tag_base}{repeat:02d}"
                directory = run_dir / arm.name / f"probe-{repeat:02d}"
                # Gateway receipts share the probe directory so samples can be joined to them.
                session(
                    arm,
                    "probe",
                    directory,
                    lambda port, model_id, _r, d=directory, t=tag: run_llmprobe(
                        config, port, model_id, d, t
                    ),
                )
            elif kind == "prefill":
                session(
                    arm,
                    "prefill",
                    run_dir / arm.name,
                    lambda port, model_id, receipts, first=repeat: exact_prefill(
                        port,
                        model_id,
                        receipts,
                        warmups=config.warmups,
                        repeats=trials,
                        corpus=corpus,
                        first=first if blocks else 0,
                    ),
                )
            else:
                session(
                    arm,
                    "burst",
                    run_dir / arm.name,
                    lambda port, model_id, receipts, first=repeat: burst(
                        port,
                        model_id,
                        receipts,
                        warmups=config.warmups,
                        repeats=trials,
                        first=first if blocks else 0,
                    ),
                )
        manifest["status"] = "partial" if failed else "complete"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["failure"] = failure_code(error)
        raise
    finally:
        manifest["finished_at"] = now_utc()
        manifest["host_end"] = host()
        manifest["sleep"] = sleep_check(started, time.time())
        write_manifest(run_dir / "manifest.json", manifest)
    return run_dir


@contextlib.contextmanager
def exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise BenchError("benchmark_lock_held") from None
        yield


@contextlib.contextmanager
def caffeinated():
    process = None
    with contextlib.suppress(OSError):
        process = subprocess.Popen(
            ["/usr/bin/caffeinate", "-dims", "-w", str(os.getpid())],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    try:
        yield process is not None
    finally:
        if process is not None:
            process.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(5)


# --------------------------------------------------------------------------
# Quality slot
# --------------------------------------------------------------------------

# Speed without quality is not a result. Every Stark arm has a slot for one
# quality result: an engine-level diagnostic comparing the arm's artifact and
# default execution path with a reference artifact or path on natural text.
# The probe is pluggable and is not part of this tool; the slot fixes what a
# report must contain, binds it to the artifact that was timed, and judges it
# against one bar that is the same for every model.
QUALITY_FORMATS = (QUALITY_SCHEMA, "inkling-slice-quality-compare")
QUALITY_CORPUS_KINDS = ("natural-text", "synthetic")
QUALITY_PATHS = ("exact", "default")
QUALITY_VERDICTS = ("pass", "fail", "not assessed")
# The bar. A result passes when the arm's next-token accuracy is not more
# than QUALITY_MAX_ACCURACY_SHORTFALL below the reference's and the two agree
# on the top token in at least QUALITY_MIN_CONFIDENT_AGREEMENT of the rows
# where the reference's top probability is at least
# QUALITY_CONFIDENT_PROBABILITY. The accuracy criterion is one-sided: an arm
# that scores higher than the reference never fails on it.
# Accuracy is compared on the pooled rows of the whole corpus, not per input:
# single inputs of a few hundred rows move by more than the bar. A corpus on
# which the reference predicts fewer next tokens than the floor is close to
# unpredictable for the model; agreement on it says little, so the result is
# not assessed and is never a pass.
QUALITY_MIN_REFERENCE_ACCURACY = 0.10
QUALITY_MAX_ACCURACY_SHORTFALL = 0.02
QUALITY_CONFIDENT_PROBABILITY = 0.5
QUALITY_MIN_CONFIDENT_AGREEMENT = 0.98
# Rates in a report are rounded by the probe; a difference this small is not
# a difference.
QUALITY_TOLERANCE = 1e-9


def _rate(value, where: str) -> float:
    if not number(value) or not 0 <= value <= 1:
        raise BenchError(f"invalid_quality_report:{where}")
    return float(value)


def _count(value, where: str, minimum=0) -> int:
    if type(value) is not int or value < minimum:
        raise BenchError(f"invalid_quality_report:{where}")
    return value


def normalize_quality(raw, fmt: str = QUALITY_SCHEMA) -> dict:
    """Reduce a probe's report to the slot schema: numbers and fixed codes only.

    Anything else in the probe's output (file names, free text) is dropped.
    """
    if not isinstance(raw, dict):
        raise BenchError("invalid_quality_report:not_an_object")
    if fmt == "inkling-slice-quality-compare":
        # The printed report of `inkling_slice --quality-compare`: one input.
        sides = {
            side: "exact" if (raw.get(side) or {}).get("exact_prefill") else "default"
            for side in ("reference", "candidate")
        }
        raw = {
            "schema": QUALITY_SCHEMA,
            "corpus": {
                key: raw.get(key)
                for key in ("prompt_tokens", "context", "window", "rows")
            },
            "reference": {"path": sides["reference"]},
            "candidate": {"path": sides["candidate"]},
            "metrics": {
                "next_token_accuracy": {
                    "reference": raw.get("next_token_accuracy_reference"),
                    "candidate": raw.get("next_token_accuracy_candidate"),
                },
                "top1_agreement": {
                    "all": raw.get("top1_agreement"),
                    "strata": [
                        {
                            "reference_top1_probability_min": threshold,
                            "rows": raw.get(f"confident_rows_{name}"),
                            "agreement": raw.get(f"top1_agreement_{name}"),
                        }
                        for threshold, name in ((0.5, "p50"), (0.9, "p90"))
                    ],
                },
                "kl_reference_to_candidate": {
                    "mean": raw.get("kl_reference_to_candidate_mean"),
                    "max": raw.get("kl_reference_to_candidate_max"),
                },
                "next_token_logprob_abs_change_mean": raw.get(
                    "next_token_logprob_abs_change_mean"
                ),
            },
        }
    elif fmt != QUALITY_SCHEMA:
        raise BenchError("unknown_quality_format")
    if raw.get("schema") != QUALITY_SCHEMA:
        raise BenchError("invalid_quality_report:schema")
    corpus, metrics = raw.get("corpus") or {}, raw.get("metrics") or {}
    accuracy = metrics.get("next_token_accuracy") or {}
    agreement = metrics.get("top1_agreement") or {}
    divergence = metrics.get("kl_reference_to_candidate") or {}
    strata = []
    for item in agreement.get("strata") or []:
        rows = _count((item or {}).get("rows"), "strata.rows")
        if rows == 0:
            continue  # a stratum with no rows has no agreement rate
        strata.append(
            {
                "reference_top1_probability_min": _rate(
                    item.get("reference_top1_probability_min"), "strata.threshold"
                ),
                "rows": rows,
                "agreement": _rate(item.get("agreement"), "strata.agreement"),
            }
        )
    for key in ("mean", "max"):
        if not number(divergence.get(key)) or divergence[key] < 0:
            raise BenchError(f"invalid_quality_report:kl.{key}")
    change = metrics.get("next_token_logprob_abs_change_mean")
    paths = {}
    for side in ("reference", "candidate"):
        path = (raw.get(side) or {}).get("path")
        if path not in QUALITY_PATHS:
            raise BenchError(f"invalid_quality_report:{side}.path")
        paths[side] = path
    inputs = corpus.get("inputs")
    return {
        "schema": QUALITY_SCHEMA,
        "level": LEVEL_DIAGNOSTIC,
        "corpus": {
            "rows": _count(corpus.get("rows"), "corpus.rows", 1),
            "inputs": 1 if inputs is None else _count(inputs, "corpus.inputs", 1),
            **{
                key: _count(corpus[key], f"corpus.{key}", 1)
                for key in ("prompt_tokens", "context", "window")
                if corpus.get(key) is not None
            },
        },
        "reference": {"path": paths["reference"]},
        "candidate": {"path": paths["candidate"]},
        "metrics": {
            "next_token_accuracy": {
                "reference": _rate(accuracy.get("reference"), "accuracy.reference"),
                "candidate": _rate(accuracy.get("candidate"), "accuracy.candidate"),
            },
            "top1_agreement": {
                "all": _rate(agreement.get("all"), "agreement.all"),
                "strata": strata,
            },
            "kl_reference_to_candidate": {
                "unit": "nats",
                "mean": float(divergence["mean"]),
                "max": float(divergence["max"]),
            },
            "next_token_logprob_abs_change_mean": float(change)
            if number(change) and change >= 0
            else None,
        },
    }


def pool_quality(reports: list[dict]) -> dict:
    """Pool normalized reports, one per input, over their rows.

    Every rate becomes the rate over all rows of all inputs. That pooled
    figure, not any single input's, is what the bar is applied to.
    """
    if not reports:
        raise BenchError("no_quality_reports")
    if len(reports) == 1:
        return reports[0]
    for side in ("reference", "candidate"):
        if len({report[side]["path"] for report in reports}) != 1:
            raise BenchError(f"quality_reports_differ:{side}.path")
    rows = sum(report["corpus"]["rows"] for report in reports)

    def pooled(*keys):
        total = 0.0
        for report in reports:
            value = report["metrics"]
            for key in keys:
                value = value[key]
            total += value * report["corpus"]["rows"]
        return total / rows

    strata: dict[float, list] = {}
    for report in reports:
        for item in report["metrics"]["top1_agreement"]["strata"]:
            entry = strata.setdefault(item["reference_top1_probability_min"], [0, 0.0])
            entry[0] += item["rows"]
            entry[1] += item["agreement"] * item["rows"]
    corpus = {
        "rows": rows,
        "inputs": sum(report["corpus"]["inputs"] for report in reports),
    }
    for key in ("prompt_tokens", "context", "window"):
        values = {report["corpus"].get(key) for report in reports}
        if len(values) == 1 and None not in values:
            corpus[key] = values.pop()
    changes = [
        report["metrics"]["next_token_logprob_abs_change_mean"] for report in reports
    ]
    return {
        "schema": QUALITY_SCHEMA,
        "level": LEVEL_DIAGNOSTIC,
        "corpus": corpus,
        "reference": {"path": reports[0]["reference"]["path"]},
        "candidate": {"path": reports[0]["candidate"]["path"]},
        "metrics": {
            "next_token_accuracy": {
                "reference": pooled("next_token_accuracy", "reference"),
                "candidate": pooled("next_token_accuracy", "candidate"),
            },
            "top1_agreement": {
                "all": pooled("top1_agreement", "all"),
                "strata": [
                    {
                        "reference_top1_probability_min": threshold,
                        "rows": count,
                        "agreement": agreed / count,
                    }
                    for threshold, (count, agreed) in sorted(strata.items())
                ],
            },
            "kl_reference_to_candidate": {
                "unit": "nats",
                "mean": pooled("kl_reference_to_candidate", "mean"),
                "max": max(
                    report["metrics"]["kl_reference_to_candidate"]["max"]
                    for report in reports
                ),
            },
            "next_token_logprob_abs_change_mean": pooled(
                "next_token_logprob_abs_change_mean"
            )
            if None not in changes
            else None,
        },
    }


def quality_verdict(report: dict) -> tuple[str, str | None, dict]:
    """pass, fail or not assessed; the reason; and the figures it was judged on."""
    metrics = report["metrics"]
    reference = metrics["next_token_accuracy"]["reference"]
    # Positive when the arm is below the reference, negative when above.
    shortfall = reference - metrics["next_token_accuracy"]["candidate"]
    confident = next(
        (
            item
            for item in metrics["top1_agreement"]["strata"]
            if abs(
                item["reference_top1_probability_min"] - QUALITY_CONFIDENT_PROBABILITY
            )
            < QUALITY_TOLERANCE
        ),
        None,
    )
    corpus = report.get("corpus") or {}
    criteria = {
        "pooled_rows": corpus.get("rows"),
        "pooled_inputs": corpus.get("inputs"),
        "reference_accuracy": reference,
        "reference_accuracy_floor": QUALITY_MIN_REFERENCE_ACCURACY,
        "accuracy_shortfall_points": shortfall * 100,
        "accuracy_shortfall_limit_points": QUALITY_MAX_ACCURACY_SHORTFALL * 100,
        "confident_probability": QUALITY_CONFIDENT_PROBABILITY,
        "confident_rows": confident["rows"] if confident else 0,
        "confident_agreement": confident["agreement"] if confident else None,
        "confident_agreement_floor": QUALITY_MIN_CONFIDENT_AGREEMENT,
    }
    if corpus.get("kind") != "natural-text":
        return "not assessed", "corpus is not natural text", criteria
    if reference < QUALITY_MIN_REFERENCE_ACCURACY - QUALITY_TOLERANCE:
        return (
            "not assessed",
            f"the reference predicts {reference:.0%} of next tokens on this corpus",
            criteria,
        )
    if confident is None:
        level = f"{QUALITY_CONFIDENT_PROBABILITY:g}"
        reason = f"no rows where the reference's top probability is at least {level}"
        return "not assessed", reason, criteria
    failures = []
    if shortfall > QUALITY_MAX_ACCURACY_SHORTFALL + QUALITY_TOLERANCE:
        failures.append(
            f"next-token accuracy is {shortfall * 100:.1f} points below the reference's on "
            f"the pooled rows (limit {QUALITY_MAX_ACCURACY_SHORTFALL * 100:.1f})"
        )
    if confident["agreement"] < QUALITY_MIN_CONFIDENT_AGREEMENT - QUALITY_TOLERANCE:
        failures.append(
            f"top-1 agreement is {confident['agreement']:.1%} where the reference's top "
            f"probability is at least {QUALITY_CONFIDENT_PROBABILITY:g} "
            f"(floor {QUALITY_MIN_CONFIDENT_AGREEMENT:.0%})"
        )
    if failures:
        return "fail", "; ".join(failures), criteria
    return "pass", None, criteria


def judge_quality(report: dict) -> dict:
    report["verdict"], report["verdict_reason"], report["criteria"] = quality_verdict(
        report
    )
    return report


def attach_quality(
    run_dir: Path,
    arm_name: str,
    report: Path | list[Path],
    candidate_artifact: Path,
    *,
    corpus_kind: str,
    fmt: str = QUALITY_SCHEMA,
    reference_artifact: Path | None = None,
    corpus_sha256: str | None = None,
    probe_binary: Path | None = None,
    probe_name: str | None = None,
    replace_existing: bool = False,
) -> dict:
    """Validate, pool, judge and store a quality result beside an arm's receipts.

    Several reports, one per input, are pooled over their rows before the bar
    is applied. The result is bound to the timed artifact: the artifact given
    here must have the fingerprint the run recorded for the arm, so a quality
    result cannot be carried to another artifact.
    """
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if (
        manifest.get("schema") != SCHEMA
        and manifest.get("schema") not in LEGACY_SCHEMAS
    ):
        raise BenchError("unknown_manifest_schema")
    arm = next((item for item in manifest["arms"] if item["name"] == arm_name), None)
    if arm is None:
        raise BenchError("unknown_arm")
    if arm.get("engine") != "stark":
        raise BenchError("quality_slot_is_for_stark_arms")
    if corpus_kind not in QUALITY_CORPUS_KINDS:
        raise BenchError("invalid_corpus_kind")
    if corpus_sha256 is not None and not HEX64.fullmatch(corpus_sha256):
        raise BenchError("invalid_corpus_sha256")
    if probe_name is not None and not CODE.fullmatch(probe_name):
        raise BenchError("invalid_probe_name")
    recorded = arm.get("artifact_fingerprint") or fingerprint(arm.get("model") or {})
    found = fingerprint(model_identity(candidate_artifact))
    if found != recorded:
        raise BenchError("quality_artifact_mismatch")
    target = run_dir / arm_name / "quality.json"
    if target.exists() and not replace_existing:
        raise BenchError("quality_report_already_attached")
    sources = [report] if isinstance(report, Path) else list(report)
    if not sources or len(sources) > 256:
        raise BenchError("no_quality_reports")
    parts = []
    for source in sources:
        try:
            raw = json.loads(source.read_text())
        except (OSError, ValueError):
            raise BenchError("quality_report_unreadable") from None
        parts.append(normalize_quality(raw, fmt))
    clean = pool_quality(parts)
    clean["corpus"]["kind"] = corpus_kind
    clean["corpus"]["sha256"] = corpus_sha256
    clean["probe"] = {
        "name": probe_name,
        "format": fmt,
        "binary_sha256": sha256_file(probe_binary) if probe_binary else None,
    }
    clean["candidate"]["artifact_fingerprint"] = found
    clean["reference"]["artifact_fingerprint"] = (
        fingerprint(model_identity(reference_artifact)) if reference_artifact else None
    )
    clean["bound_to"] = {
        "arm": arm_name,
        "worker_binary_sha256": arm.get("binary_sha256"),
        "source_report_sha256": [sha256_file(source) for source in sources],
    }
    clean["attached_at"] = now_utc()
    judge_quality(clean)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    target.write_text(json.dumps(clean, indent=2, sort_keys=True) + "\n")
    return clean


def read_quality(path: Path, artifact_fingerprint: str | None = None) -> dict | None:
    """An attached result, judged again from its figures, or None when the slot
    is empty or unreadable. A result bound to another artifact is not assessed."""
    with contextlib.suppress(
        OSError, ValueError, BenchError, KeyError, TypeError, AttributeError
    ):
        stored = json.loads(path.read_text())
        clean = normalize_quality(stored)

        def code(value, pattern=CODE):
            return (
                value if isinstance(value, str) and pattern.fullmatch(value) else None
            )

        corpus, probe = stored.get("corpus") or {}, stored.get("probe") or {}
        kind = corpus.get("kind")
        clean["corpus"]["kind"] = kind if kind in QUALITY_CORPUS_KINDS else None
        clean["corpus"]["sha256"] = code(corpus.get("sha256"), HEX64)
        clean["probe"] = {
            "name": code(probe.get("name")),
            "format": code(probe.get("format")),
            "binary_sha256": code(probe.get("binary_sha256"), HEX64),
        }
        for side in ("reference", "candidate"):
            clean[side]["artifact_fingerprint"] = code(
                (stored.get(side) or {}).get("artifact_fingerprint"), HEX64
            )
        clean["bound_to"] = numeric_tree(stored.get("bound_to"))
        clean["attached_at"] = code(
            stored.get("attached_at"), re.compile(r"[0-9TZ:-]{20}")
        )
        judge_quality(clean)
        bound = clean["candidate"]["artifact_fingerprint"]
        if artifact_fingerprint is not None and bound != artifact_fingerprint:
            clean["verdict"] = "not assessed"
            clean["verdict_reason"] = "the result is bound to another artifact"
        return clean
    return None


# --------------------------------------------------------------------------
# Summarizer
# --------------------------------------------------------------------------


def stats(values: list[float]) -> dict | None:
    if not values:
        return None
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def join_native(samples: list[dict], natives: list[dict]) -> None:
    """Attach the gateway's per-request draft record to each llmprobe sample.

    Every client request passes through the gateway in order; receipts exist
    for a subset. A sample matches the next generation with the same input and
    output token counts.
    """
    generations = [row for row in natives if row.get("event") == "native_generation"]
    cursor = 0
    for sample in samples:
        if sample.get("failed") or sample.get("inputTokens") is None:
            continue
        for index in range(cursor, len(generations)):
            row = generations[index]
            if row.get("input_tokens") == sample["inputTokens"] and row.get(
                "output_tokens"
            ) == sample.get("outputTokens"):
                sample["drafts_requested"] = row.get("drafts_requested")
                sample["engine_mtp"] = row.get("engine_mtp")
                cursor = index + 1
                break


def draft_summary(rows: list[dict]) -> dict:
    """Requested draft settings and what the engine reported, for one cell."""
    requested = sorted(
        {str(row.get("drafts_requested")) for row in rows if "drafts_requested" in row}
    )
    reports = [
        row["engine_mtp"] for row in rows if isinstance(row.get("engine_mtp"), dict)
    ]
    drafts = sorted({str(report.get("drafts")) for report in reports})
    proposed = sum(report.get("proposed") or 0 for report in reports)
    accepted = sum(report.get("accepted") or 0 for report in reports)
    return {
        "requested": requested,
        "engine_drafts": drafts,
        "engine_reports": len(reports),
        "proposed": proposed,
        "accepted": accepted,
        "acceptance": accepted / proposed if proposed else None,
    }


def included_samples(samples: list[dict], scenario: Scenario):
    """The samples a cell is computed from, and counts of the excluded ones."""
    included, excluded = [], {"failed": 0, "short_output": 0, "no_rate": 0}
    for sample in samples:
        if sample.get("warmup"):
            continue
        if sample.get("failed"):
            excluded["failed"] += 1
            continue
        if scenario.cap is not None and sample.get("outputTokens") != scenario.cap:
            excluded["short_output"] += 1
            continue
        if not number(sample.get(scenario.metric)):
            excluded["no_rate"] += 1
            continue
        included.append(sample)
    return included, excluded


def llmprobe_cell(samples: list[dict], scenario: Scenario) -> dict:
    """Median, range and exclusions for one llmprobe scenario and arm."""
    included, excluded = included_samples(samples, scenario)
    values = [sample[scenario.metric] for sample in included]
    walls = [sample["wallMs"] for sample in included if number(sample.get("wallMs"))]
    outliers = 0
    kept = values
    if len(walls) >= 2:
        typical = statistics.median(walls)
        flags = [
            number(s.get("wallMs")) and s["wallMs"] > OUTLIER_FACTOR * typical
            for s in included
        ]
        outliers = sum(flags)
        kept = [value for value, flag in zip(values, flags, strict=True) if not flag]
    steps = [s["tokensPerStep"] for s in included if number(s.get("tokensPerStep"))]
    return {
        "metric": scenario.metric,
        "unit": scenario.unit,
        "stats": stats(values),
        "excluded": excluded,
        "wall_outliers": outliers,
        "median_without_outliers": statistics.median(kept)
        if kept and outliers
        else None,
        "tokens_per_step_median": statistics.median(steps) if steps else None,
        "input_tokens": sorted(
            {s["inputTokens"] for s in included if s.get("inputTokens") is not None}
        ),
        "cached_tokens": sorted(
            {
                s["cachedInputTokens"]
                for s in included
                if s.get("cachedInputTokens") is not None
            }
        ),
        "ttft_ms": stats([s["ttftMs"] for s in included if number(s.get("ttftMs"))]),
        "wall_ms": stats(walls),
        "drafting": draft_summary(included),
    }


def valid_trials(rows: list[dict], metric: str) -> list[dict]:
    return [
        row
        for row in rows
        if not row.get("warmup") and row.get("valid") and number(row.get(metric))
    ]


def trial_cell(rows: list[dict], metric: str, unit: str) -> dict:
    measured = [row for row in rows if not row.get("warmup")]
    valid = valid_trials(rows, metric)
    cell = {
        "metric": metric,
        "unit": unit,
        "stats": stats([row[metric] for row in valid]),
        "excluded": {"invalid": len(measured) - len(valid)},
        "wall_outliers": 0,
        "median_without_outliers": None,
        "tokens_per_step_median": None,
    }
    if rows and "prompt_tokens" in rows[0]:
        cell["input_tokens"] = sorted({row["prompt_tokens"] for row in valid})
        cell["cached_tokens"] = sorted(
            {row["cached_tokens"] for row in valid if row["cached_tokens"] is not None}
        )
        cell["input_ids_by_seed"] = {
            row.get("seed") or str(row.get("trial")): row["input_ids_le_u32_sha256"]
            for row in rows
            if row.get("input_ids_le_u32_sha256")
        }
        cell["client_ttft_s"] = stats([row["client_ttft_s"] for row in valid])
        cell["drafting"] = draft_summary(valid)
    else:
        requests = [req for row in valid for req in row.get("requests", [])]
        cell["input_tokens"] = sorted({req["prompt_tokens"] for req in requests})
        cell["cached_tokens"] = sorted(
            {
                req["cached_tokens"]
                for req in requests
                if req.get("cached_tokens") is not None
            }
        )
        cell["total_tokens"] = [row["total_generated_tokens"] for row in valid]
        cell["drafting"] = draft_summary(requests)
    return cell


def load_runs(run_dirs: list[Path], failed_arms: list | None = None) -> dict:
    """Arms keyed by name, merging identical arms from several run directories."""
    arms: dict[str, dict] = {}
    for run_dir in run_dirs:
        manifest = json.loads((run_dir / "manifest.json").read_text())
        schema = manifest.get("schema")
        if schema != SCHEMA and schema not in LEGACY_SCHEMAS:
            raise BenchError("unknown_manifest_schema")
        # A run made before profiles existed measured the profile its schema
        # names; its layout was the legacy one.
        recorded = manifest.get("profile") or {}
        profile_key = recorded.get("key") or LEGACY_SCHEMAS.get(schema)
        if not recorded and profile_key:
            with contextlib.suppress(BenchError):
                recorded = load_profile(profile_key).facts()
        layout = (manifest.get("config") or {}).get("layout") or "legacy"
        protocol = {
            key: (manifest.get("config") or {}).get(key)
            for key in (
                "probe_runs",
                "reasoning",
                "stark_drafts",
                "stark_effort_aliases",
                "stark_min_context",
                "mlx_force_mtp",
            )
        }
        failures = {
            item["arm"]: item["failure"] for item in manifest.get("arm_failures") or []
        }
        for arm in manifest["arms"]:
            arm_dir = run_dir / arm["name"]
            data = {
                "samples": [],
                "natives": [],
                "benches": [],
                "prefill": read_jsonl(arm_dir / "prefill-8800.jsonl"),
                "burst": read_jsonl(arm_dir / "burst.jsonl"),
            }
            # The block a value was measured in: arms are compared within it.
            for row in (*data["prefill"], *data["burst"]):
                row["_pair"] = f"{run_dir.name}/trial-{row.get('trial')}"
            for probe in sorted(arm_dir.glob("probe-*")):
                samples = read_jsonl(probe / "samples.jsonl")
                natives = read_jsonl(probe / "native.jsonl")
                join_native(samples, natives)
                for row in samples:
                    row["_pair"] = f"{run_dir.name}/{probe.name}"
                data["samples"].extend(samples)
                data["natives"].extend(natives)
                if (probe / "bench.json").is_file():
                    data["benches"].append(
                        json.loads((probe / "bench.json").read_text())
                    )
            if not (data["samples"] or data["prefill"] or data["burst"]):
                # An arm that produced nothing (for example a server that never
                # started) is listed as failed and does not occupy its name, so a
                # later run of the same arm merges cleanly.
                (failed_arms if failed_arms is not None else []).append(
                    {
                        "run": run_dir.name,
                        "arm": arm["name"],
                        "failure": failures.get(arm["name"])
                        or manifest.get("failure")
                        or "no_data",
                    }
                )
                continue
            identity = json.dumps(
                {
                    "binary_sha256": arm.get("binary_sha256"),
                    "model": arm.get("model"),
                    "diagnostic_env": arm.get("diagnostic_env"),
                    "protocol": protocol,
                    "layout": layout,
                    "profile": profile_key,
                },
                sort_keys=True,
            )
            name = arm["name"]
            if name in arms and arms[name]["identity"] != identity:
                name = f"{name}@{run_dir.name}"
            entry = arms.setdefault(
                name,
                {
                    "identity": identity,
                    "engine": arm["engine"],
                    "runs": [],
                    "samples": [],
                    "benches": [],
                    "prefill": [],
                    "burst": [],
                    "native": [],
                    "quality": [],
                },
            )
            entry["runs"].append(
                {
                    "run": run_dir.name,
                    "status": manifest.get("status"),
                    "host_start": manifest.get("host_start"),
                    "host_end": manifest.get("host_end"),
                    "sleep": manifest.get("sleep"),
                    "binary_sha256": arm.get("binary_sha256"),
                    "description": arm.get("description"),
                    "diagnostic_env": arm.get("diagnostic_env"),
                    "llmprobe": manifest.get("llmprobe"),
                    "reasoning": protocol["reasoning"],
                    "protocol": protocol,
                    "layout": layout,
                    "profile": profile_key,
                    "profile_title": recorded.get("title"),
                    "reference_arm": recorded.get("reference_arm"),
                    "scenario_notes": recorded.get("notes"),
                    "standard": manifest.get("standard"),
                    "scoreboard": arm.get("scoreboard"),
                    "artifact": arm.get("artifact"),
                    "quantization": arm.get("quantization"),
                    "match": arm.get("match"),
                    "artifact_fingerprint": arm.get("artifact_fingerprint"),
                    "host_warnings": manifest.get("host_warnings"),
                }
            )
            entry["runs"][-1]["arm_failure"] = failures.get(arm["name"])
            entry["samples"].extend(data["samples"])
            entry["native"].extend(data["natives"])
            entry["benches"].extend(data["benches"])
            entry["prefill"].extend(data["prefill"])
            entry["burst"].extend(data["burst"])
            entry["native"].extend(read_jsonl(arm_dir / "native.jsonl"))
            # A quality result counts only for the artifact this arm timed.
            report = read_quality(
                arm_dir / "quality.json",
                arm.get("artifact_fingerprint") or fingerprint(arm.get("model") or {}),
            )
            if report is not None:
                entry["quality"].append(report)
    return arms


def speed_ratio(value, reference, higher_is_better=True):
    if not (number(value) and number(reference)) or value <= 0 or reference <= 0:
        return None
    return value / reference if higher_is_better else reference / value


def drafting_condition(arm: dict) -> str:
    """How drafting was configured for an arm, from the engine's own reports."""
    if arm["engine"] == "stark":
        defaults = sorted(
            {
                str((row.get("readiness_mtp") or {}).get("default_drafts"))
                for row in arm["native"]
                if row.get("event") == "worker_ready" and row.get("session") != "burst"
            }
        )
        modes = sorted(
            {
                row.get("draft_mode")
                for row in arm["native"]
                if row.get("event") == "worker_ready" and row.get("session") != "burst"
            }
            - {None}
        )
        return f"{'/'.join(modes) or '-'}; worker default_drafts {'/'.join(defaults) or '-'}"
    settings = [bench["target"].get("engineSettings") or {} for bench in arm["benches"]]
    default_on = sorted(
        {str((s.get("mtp") or {}).get("default_on")) for s in settings if s}
    )
    forced = sorted(
        {str((run.get("protocol") or {}).get("mlx_force_mtp")) for run in arm["runs"]}
    )
    return f"server mtp default_on {'/'.join(default_on) or '-'}; --mtp forced {'/'.join(forced)}"


def pair_values(rows: list[dict], metric: str) -> dict[str, list[float]]:
    """Included values grouped by the block they were measured in."""
    groups: dict[str, list[float]] = {}
    for row in rows:
        if row.get("_pair") is not None and number(row.get(metric)):
            groups.setdefault(row["_pair"], []).append(row[metric])
    return groups


def paired_ratio(
    values, reference, higher_is_better=True, adjacent=True, counts=True
) -> dict | None:
    """Per-block ratios of one arm against another, and what they resolve.

    Each block contributes one ratio (the arm's median over the reference's
    median in that block). The ratio and its range are always reported. The
    standing has no numeric threshold: an arm is ahead only when every block
    agrees and there are at least MIN_STANDING_PAIRS blocks, behind likewise,
    and unresolved otherwise. Blocks that were not adjacent in time resolve
    nothing, and a cell outside the standard set (counts false) is reported
    only.
    """
    if not values or not reference:
        return None
    ratios = []
    for pair in sorted(set(values) & set(reference)):
        ratio = speed_ratio(
            statistics.median(values[pair]),
            statistics.median(reference[pair]),
            higher_is_better,
        )
        if ratio is not None:
            ratios.append(ratio)
    if not ratios:
        return {"n": 0, "standing": "unpaired"}
    if not counts:
        standing = "reported only"
    elif not adjacent:
        standing = "not time-paired"
    elif len(ratios) < MIN_STANDING_PAIRS:
        standing = "too few pairs"
    elif min(ratios) > 1:
        standing = "ahead"
    elif max(ratios) < 1:
        standing = "behind"
    else:
        standing = "unresolved"
    return {
        "median": statistics.median(ratios),
        "min": min(ratios),
        "max": max(ratios),
        "n": len(ratios),
        "wins": sum(ratio > 1 for ratio in ratios),
        "standing": standing,
    }


def arm_field(arm: dict, key: str):
    """A per-run arm fact that should agree across the arm's runs."""
    values = [run.get(key) for run in arm["runs"] if run.get(key) is not None]
    return values[0] if values else None


def matched_arms(arms: dict) -> dict[str, str]:
    """For each Stark arm, the reference arm serving a matched artifact."""
    result = {}
    for name, arm in arms.items():
        match = arm_field(arm, "match")
        if arm["engine"] != "stark" or not match:
            continue
        for other_name, other in arms.items():
            if other["engine"] != "stark" and arm_field(other, "match") == match:
                result[name] = other_name
                break
    return result


def quality_summary(arm: dict) -> dict:
    """The quality slot of one arm across its runs: one verdict.

    A failing result in any run fails the arm; a pass needs no failure beside
    it; anything else is not assessed. An arm with no result is missing.
    """
    if arm["engine"] != "stark":
        return {"status": "not applicable", "reason": "reference engine", "reports": []}
    reports = arm.get("quality") or []
    if not reports:
        return {"status": "missing", "reason": "no result attached", "reports": []}
    verdicts = {report["verdict"] for report in reports}
    status = next(
        (verdict for verdict in ("fail", "pass") if verdict in verdicts), "not assessed"
    )
    chosen = [report for report in reports if report["verdict"] == status]
    return {
        "status": status,
        "reason": chosen[0].get("verdict_reason"),
        "reports": chosen,
    }


def speed_only(arm: dict, quality: dict) -> bool:
    """A Stark arm without a passing quality result holds no standing."""
    return arm["engine"] == "stark" and quality["status"] != "pass"


def scoreboard_standing(
    name: str, arm: dict, quality: dict, findings: list[dict]
) -> dict:
    """Whether an arm's figures count as standard results, and why not."""
    reasons: list[str] = []
    assessed = True
    for run in arm["runs"]:
        standard = run.get("standard")
        if standard is None:
            assessed = False
            continue
        reasons.extend(standard.get("deviations") or [])
        reasons.extend((run.get("scoreboard") or {}).get("reasons") or [])
    for finding in findings:
        if finding["blocking"] and name in finding["arms"]:
            reasons.append(finding["detail"])
    if speed_only(arm, quality):
        reasons.append(f"speed only: quality {quality['status']}")
    reasons = sorted(set(reasons))
    if not assessed:
        return {
            "eligible": None,
            "reasons": ["recorded before the standard was defined", *reasons],
        }
    return {"eligible": not reasons, "reasons": reasons}


def validity_findings(arms: dict, cells: dict, prefill_match, modes) -> list[dict]:
    """Conditions that limit what the table can be used for.

    A blocking finding means the affected arms were not measured on matched
    work, so their figures are not comparable with the other arms'.
    """
    findings: list[dict] = []

    def add(code, detail, names, blocking=False, scenario=None):
        for known in findings:
            if known["code"] == code and known["detail"] == detail:
                known["arms"] = sorted({*known["arms"], *names})
                return
        findings.append(
            {
                "code": code,
                "scenario": scenario,
                "arms": sorted(names),
                "blocking": blocking,
                "detail": detail,
            }
        )

    if len(modes) > 1:
        add(
            "mixed_reasoning_modes",
            "reasoning modes differ between arms: " + ", ".join(modes),
            arms,
            blocking=True,
        )
    profiles = sorted(
        {str(run.get("profile")) for arm in arms.values() for run in arm["runs"]}
    )
    if len(profiles) > 1:
        add(
            "mixed_profiles",
            "runs of different profiles are joined: " + ", ".join(profiles),
            arms,
            blocking=True,
        )
    for key in MATCHED_INPUT_SCENARIOS:
        row = cells.get(key) or {}
        seen = {
            name: frozenset(cell.get("input_tokens") or [])
            for name, cell in row.items()
            if cell.get("stats")
        }
        sets = [item for item in seen.values() if item]
        # Arms measured over more repeats see more inputs; a subset is a match.
        if any(not (a <= b or b <= a) for a in sets for b in sets):
            add(
                "rendered_inputs_differ",
                f"{key}: input token counts differ between arms ("
                + "; ".join(
                    f"{name} {span(sorted(item))}" for name, item in seen.items()
                )
                + "), so the arms did not render the same prompt",
                seen,
                blocking=True,
                scenario=key,
            )
    if prefill_match is False:
        add(
            "prefill_inputs_differ",
            "exact prefill input-ID hashes differ between arms for the same seed",
            cells.get("prefill_8800_native") or {},
            blocking=True,
            scenario="prefill_8800_native",
        )
    for key, row in cells.items():
        scenario = SCENARIO_BY_KEY[key]
        for name, cell in row.items():
            short = (cell.get("excluded") or {}).get("short_output")
            if short:
                add(
                    "short_outputs_excluded",
                    f"{key}: {name} stopped before the {scenario.cap}-token cap in {short} "
                    "samples; they are excluded so the remaining samples did equal work",
                    [name],
                    scenario=key,
                )
            reused = [count for count in cell.get("cached_tokens") or [] if count]
            if reused and not key.endswith("_ceiling"):
                add(
                    "prefix_reuse_in_cold_cell",
                    f"{key}: {name} reused {span(reused)} cached input tokens in a cell "
                    "defined as cold",
                    [name],
                    scenario=key,
                )
    for name, arm in arms.items():
        for run in arm["runs"]:
            if (run.get("sleep") or {}).get("sleep_events"):
                add(
                    "host_slept",
                    f"the host slept during run {run['run']}",
                    [name],
                    blocking=True,
                )
            for warning in run.get("host_warnings") or []:
                add("host_condition", f"run {run['run']}: {warning}", [name])
    return findings


def summarize(
    run_dirs: list[Path], reference_arm: str | None = None, *, title: str | None = None
) -> dict:
    failed_arms: list[dict] = []
    arms = load_runs(run_dirs, failed_arms)
    declared = [
        run.get("reference_arm")
        for arm in arms.values()
        for run in arm["runs"]
        if run.get("reference_arm")
    ]
    reference_arm = reference_arm or (declared[0] if declared else "mlx-original")
    titles = sorted(
        {
            run["profile_title"]
            for arm in arms.values()
            for run in arm["runs"]
            if run.get("profile_title")
        }
    )
    title = title or (
        f"Stark standard benchmark: {', '.join(titles)}"
        if titles
        else "Stark standard benchmark"
    )
    matched = matched_arms(arms)
    cells: dict[str, dict] = {}
    for scenario in SCENARIOS:
        row = {}
        pairs: dict[str, dict] = {}
        for name, arm in arms.items():
            if scenario.source == "llmprobe":
                samples = [
                    s for s in arm["samples"] if s.get("scenario") == scenario.key
                ]
                if samples:
                    row[name] = llmprobe_cell(samples, scenario)
                    pairs[name] = pair_values(
                        included_samples(samples, scenario)[0], scenario.metric
                    )
            elif arm[scenario.source]:
                rows = arm[scenario.source]
                row[name] = trial_cell(rows, scenario.metric, scenario.unit)
                pairs[name] = pair_values(
                    valid_trials(rows, scenario.metric), scenario.metric
                )
        reference = (row.get(reference_arm) or {}).get("stats") or {}

        def adjacent(*names, scenario=scenario):
            # Client scenarios are adjacent by construction; prefill and burst
            # trials are adjacent only in the block layout.
            return scenario.source == "llmprobe" or all(
                run.get("layout") == "blocks"
                for name in names
                for run in arms[name]["runs"]
            )

        # Only the standard set is read for a standing; other cells are reported.
        counts = scenario_tier(scenario.key) == "standard"
        for name, cell in row.items():
            median = (cell.get("stats") or {}).get("median")
            cell["speed_vs_reference"] = speed_ratio(
                median, reference.get("median"), scenario.higher_is_better
            )
            cell["level"] = scenario_level(scenario.key)
            cell["tier"] = scenario_tier(scenario.key)
            cell["paired_vs_reference"] = (
                paired_ratio(
                    pairs.get(name),
                    pairs.get(reference_arm),
                    scenario.higher_is_better,
                    adjacent(name, reference_arm),
                    counts,
                )
                if name != reference_arm and reference_arm in row
                else None
            )
            other = matched.get(name)
            cell["paired_vs_matched"] = (
                {
                    "arm": other,
                    **(
                        paired_ratio(
                            pairs.get(name),
                            pairs.get(other),
                            scenario.higher_is_better,
                            adjacent(name, other),
                            counts,
                        )
                        or {"n": 0, "standing": "unpaired"}
                    ),
                }
                if other in row
                else None
            )
        if row:
            cells[scenario.key] = row
    best = {}
    decode_keys = [s.key for s in SCENARIOS if s.metric == "decodeTokPerSec"]
    for name in arms:
        candidates = [
            (cells[key][name]["stats"]["median"], key)
            for key in decode_keys
            if name in cells.get(key, {}) and cells[key][name]["stats"]
        ]
        cold = [item for item in candidates if not item[1].endswith("_ceiling")]
        best[name] = {
            "any": max(candidates) if candidates else None,
            "cold_prefix": max(cold) if cold else None,
        }
    by_seed: dict[str, set] = {}
    for cell in cells.get("prefill_8800_native", {}).values():
        for seed, digest in (cell.get("input_ids_by_seed") or {}).items():
            by_seed.setdefault(seed, set()).add(digest)
    prefill_match = all(len(d) == 1 for d in by_seed.values()) if by_seed else None
    modes = sorted(
        {str(run["reasoning"]) for arm in arms.values() for run in arm["runs"]}
    )
    notes = {}
    for name, arm in arms.items():
        benches = [bench["bench"] for bench in arm["benches"]]
        notes[name] = {
            "reasoning_modes": sorted({str(run["reasoning"]) for run in arm["runs"]}),
            "drafting": drafting_condition(arm),
            "decode_length_notes": sorted(
                {
                    b.get("decodeLengthNote")
                    for b in benches
                    if b.get("decodeLengthNote")
                }
            ),
            "reasoning_notes": sorted(
                {b.get("reasoningNote") for b in benches if b.get("reasoningNote")}
            ),
            "runs_notes": sorted(
                {b.get("runsNote") for b in benches if b.get("runsNote")}
            ),
            "stream_caveats": sorted(
                {b.get("streamCaveat") for b in benches if b.get("streamCaveat")}
            ),
            "load_drift": [b.get("loadDrift") for b in benches],
            "prefix_cache": [b.get("prefixCache") for b in benches],
            "batching_sweep": [b.get("batchingSweep") for b in benches],
            "engine_settings": [
                bench["target"].get("engineSettings") for bench in arm["benches"]
            ],
        }
    findings = validity_findings(arms, cells, prefill_match, modes)
    quality = {name: quality_summary(arm) for name, arm in arms.items()}
    # Speed without quality is not a result: the ratio and its range stay, the
    # standing is withheld.
    for row in cells.values():
        for name, cell in row.items():
            if not speed_only(arms[name], quality[name]):
                continue
            for key in ("paired_vs_reference", "paired_vs_matched"):
                paired = cell.get(key)
                if paired and paired.get("standing") in ("ahead", "behind"):
                    paired["standing"] = "speed only"
    for name, arm in arms.items():
        if arm["engine"] == "stark" and reference_arm in arms and name not in matched:
            findings.append(
                {
                    "code": "artifact_not_matched",
                    "scenario": None,
                    "arms": [name],
                    "blocking": False,
                    "detail": f"{name} has no reference arm on a matched artifact here; its "
                    f"ratios against {reference_arm} compare artifacts as well as engines"
                    if arm_field(arm, "match")
                    else f"{name}: the run does not record which artifact class it served",
                }
            )
    scenario_notes = {}
    for arm in arms.values():
        for run in arm["runs"]:
            scenario_notes.update(run.get("scenario_notes") or {})
    return {
        "schema": SUMMARY_SCHEMA,
        "title": title,
        "generated_at": now_utc(),
        "reference_arm": reference_arm,
        "runs": [run_dir.name for run_dir in run_dirs],
        "arms": {
            name: {"engine": arm["engine"], "runs": arm["runs"]}
            for name, arm in arms.items()
        },
        "scenarios": {
            s.key: {
                "title": s.title,
                "metric": s.metric,
                "unit": s.unit,
                "higher_is_better": s.higher_is_better,
                "description": s.description,
                "comparability": s.comparability,
                "level": scenario_level(s.key),
                "tier": scenario_tier(s.key),
                "profile_note": scenario_notes.get(s.key),
            }
            for s in SCENARIOS
        },
        "cells": cells,
        "best_single_user_decode": best,
        "prefill_input_ids_match_across_arms": prefill_match,
        "reasoning_modes": modes,
        "mixed_reasoning_modes": len(modes) > 1,
        "arm_notes": notes,
        "matched_arms": matched,
        "quality": quality,
        "validity": findings,
        "scoreboard": {
            name: scoreboard_standing(name, arm, quality[name], findings)
            for name, arm in arms.items()
        },
        "failed_arms": [
            {**item, "measured_in_another_run": item["arm"] in arms}
            for item in failed_arms
        ],
    }


def fmt(value, digits=1):
    if not number(value):
        return "-"
    return f"{value:,.{digits}f}"


def span(values: list) -> str:
    if not values:
        return "-"
    return str(values[0]) if len(values) == 1 else f"{min(values)}-{max(values)}"


def drafting_text(cell: dict) -> str:
    drafting = cell.get("drafting") or {}
    if not drafting.get("requested") and not drafting.get("engine_reports"):
        return "-"
    text = "req " + "/".join(drafting.get("requested") or ["-"])
    if drafting.get("engine_reports"):
        text += "; engine " + "/".join(drafting["engine_drafts"])
        if drafting.get("acceptance") is not None:
            text += f", accept {drafting['acceptance']:.2f}"
    return text


def paired_text(paired: dict | None, *, named=False) -> str:
    if not paired:
        return "-"
    prefix = f"{paired['arm']}: " if named and paired.get("arm") else ""
    if not paired.get("n"):
        return prefix + paired.get("standing", "-")
    return (
        f"{prefix}{fmt(paired['median'], 2)} ({fmt(paired['min'], 2)}-{fmt(paired['max'], 2)}, "
        f"n={paired['n']}) {paired['standing']}"
    )


def render_markdown(summary: dict) -> str:
    reference = summary["reference_arm"]
    lines = [f"# {summary.get('title') or 'Stark standard benchmark'}", ""]
    if summary["mixed_reasoning_modes"]:
        lines += [
            "**Warning: this table mixes reasoning modes ("
            + ", ".join(summary["reasoning_modes"])
            + "). Rows from different modes are not comparable.**",
            "",
        ]
    for item in summary.get("failed_arms") or []:
        note = (
            "; measured in another run"
            if item["measured_in_another_run"]
            else "; no measurements"
        )
        lines += [
            f"Failed arm: `{item['arm']}` in run {item['run']}: {item['failure']}{note}.",
            "",
        ]
    lines += [
        (
            f"Runs: {', '.join(summary['runs'])}. Reasoning mode: {', '.join(summary['reasoning_modes'])}. "
            f"Reference arm for the speed column: `{reference}` (above 1.00 means faster than the reference)."
        ),
        "",
        (
            "Every row is server-level, measured through the real worker or server. Paired columns give "
            "the median of per-block ratios with its range; a standing is ahead or behind only when "
            f"every block agrees and there are at least {MIN_STANDING_PAIRS} blocks. Supplementary rows, "
            "the burst among them, are reported only. A Stark arm without a passing quality result is "
            "speed only and holds no standing."
        ),
        "",
        (
            "| Scenario | Arm | Median | Range | n | Excluded | Tok/step | Drafting | Input tokens "
            "| Cached tokens | Speed vs ref | Level | Set | Paired vs ref | Paired vs matched arm |"
        ),
        "| --- | --- | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    for key, row in summary["cells"].items():
        scenario = summary["scenarios"][key]
        unit = scenario["unit"]
        for name, cell in row.items():
            stat = cell.get("stats") or {}
            digits = 3 if unit == "s" else 1
            excluded = ", ".join(f"{k} {v}" for k, v in cell["excluded"].items() if v)
            if cell.get("wall_outliers"):
                excluded = (
                    excluded + ", " if excluded else ""
                ) + f"wall outliers {cell['wall_outliers']} (kept)"
            lines.append(
                f"| {scenario['title']} | {name} | {fmt(stat.get('median'), digits)} {unit} | "
                f"{fmt(stat.get('min'), digits)}-{fmt(stat.get('max'), digits)} | {stat.get('n', 0)} | "
                f"{excluded or '-'} | {fmt(cell.get('tokens_per_step_median'), 2)} | {drafting_text(cell)} | "
                f"{span(cell.get('input_tokens') or [])} | {span(cell.get('cached_tokens') or [])} | "
                f"{fmt(cell.get('speed_vs_reference'), 2)} | {cell.get('level', '-')} | "
                f"{cell.get('tier', '-')} | {paired_text(cell.get('paired_vs_reference'))} | "
                f"{paired_text(cell.get('paired_vs_matched'), named=True)} |"
            )
    lines += [
        "",
        "## Best single-user decode",
        "",
        "| Arm | Best scenario (any) | Median | Best cold-prefix scenario | Median |",
        "| --- | --- | ---: | --- | ---: |",
    ]
    for name, best in summary["best_single_user_decode"].items():
        anyone, cold = best["any"], best["cold_prefix"]
        lines.append(
            f"| {name} | {anyone[1] if anyone else '-'} | {fmt(anyone[0]) if anyone else '-'} | "
            f"{cold[1] if cold else '-'} | {fmt(cold[0]) if cold else '-'} |"
        )
    lines += [
        "",
        "## Quality and standing",
        "",
        (
            "Quality figures are an engine diagnostic on natural text, pooled over all rows of the corpus, "
            "not a server measurement. A result passes when next-token accuracy is not more than "
            f"{QUALITY_MAX_ACCURACY_SHORTFALL * 100:.1f} points below the reference's and top-1 agreement is at "
            f"least {QUALITY_MIN_CONFIDENT_AGREEMENT:.0%} where the reference's top probability is at "
            f"least {QUALITY_CONFIDENT_PROBABILITY:g}. A corpus on which the reference predicts fewer than "
            f"{QUALITY_MIN_REFERENCE_ACCURACY:.0%} of next tokens is not assessed."
        ),
        "",
        (
            "| Arm | Artifact | Quantization | Matched arm | Quality | Next-token accuracy (ref / arm) "
            "| Top-1 agreement (all; by reference confidence) | KL ref to arm, nats (mean / max) "
            "| Standard result |"
        ),
        "| --- | --- | --- | --- | --- | ---: | --- | ---: | --- |",
    ]
    for name, arm in summary["arms"].items():
        quality = (summary.get("quality") or {}).get(name) or {
            "status": "-",
            "reports": [],
        }
        report = quality["reports"][0] if quality["reports"] else None
        metrics = (report or {}).get("metrics") or {}
        accuracy = metrics.get("next_token_accuracy") or {}
        agreement = metrics.get("top1_agreement") or {}
        divergence = metrics.get("kl_reference_to_candidate") or {}
        strata = "; ".join(
            f"p>={item['reference_top1_probability_min']:g}: {item['agreement']:.4f} of {item['rows']}"
            for item in agreement.get("strata") or []
        )
        standing = (summary.get("scoreboard") or {}).get(name) or {}
        verdict = (
            "not judged"
            if standing.get("eligible") is None
            else "yes"
            if standing["eligible"]
            else "no"
        )
        if standing.get("reasons"):
            verdict += ": " + "; ".join(standing["reasons"])
        runs = arm["runs"]
        first = runs[0] if runs else {}
        lines.append(
            f"| {name} | {first.get('artifact') or '-'} | {first.get('quantization') or '-'} | "
            f"{(summary.get('matched_arms') or {}).get(name, '-')} | "
            f"{quality['status']}{' (' + quality['reason'] + ')' if quality.get('reason') else ''} | "
            f"{fmt(accuracy.get('reference'), 4)} / {fmt(accuracy.get('candidate'), 4)} | "
            f"{fmt(agreement.get('all'), 4)}{'; ' + strata if strata else ''} | "
            f"{fmt(divergence.get('mean'), 5)} / {fmt(divergence.get('max'), 5)} | {verdict} |"
        )
    findings = summary.get("validity") or []
    lines += ["", "## Validity", ""]
    if findings:
        lines += [
            ("Not comparable: " if item["blocking"] else "Note: ")
            + item["detail"]
            + "."
            for item in findings
        ]
    else:
        lines.append("No findings.")
    match = summary["prefill_input_ids_match_across_arms"]
    lines += [
        "",
        "## Conditions",
        "",
        f"Exact 8,800-ID prefill inputs identical across arms: {'-' if match is None else match}.",
        "",
        (
            "| Arm | Engine | Binary | Runs | Reasoning mode | Drafting | Power | Thermal | macOS "
            "| Hardware | Sleep events | Reasoning note | Decode length note |"
        ),
        "| --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for name, arm in summary["arms"].items():
        runs = arm["runs"]
        hosts = [run.get("host_start") or {} for run in runs]
        power = ", ".join(
            sorted({(h.get("power") or {}).get("source") or "-" for h in hosts})
        )
        thermal = ", ".join(
            sorted({str(h.get("process_thermal_state")) for h in hosts})
        )
        macos = ", ".join(sorted({str(h.get("macos_version")) for h in hosts}))
        hardware = ", ".join(
            sorted({f"{h.get('hardware_model')} {h.get('chip')}" for h in hosts})
        )
        binaries = ", ".join(
            sorted({(run.get("binary_sha256") or "-")[:12] for run in runs})
        )
        sleeps = sum((run.get("sleep") or {}).get("sleep_events") or 0 for run in runs)
        notes = summary["arm_notes"][name]
        modes = ", ".join(
            reasoning_label(None if m == "None" else m)
            for m in notes["reasoning_modes"]
        )
        lines.append(
            f"| {name} | {arm['engine']} | {binaries} | {len(runs)} | {modes} | {notes['drafting']} | {power} | "
            f"{thermal} | {macos} | {hardware} | {sleeps} | {'; '.join(notes['reasoning_notes']) or '-'} | "
            f"{'; '.join(notes['decode_length_notes']) or '-'} |"
        )
    lines += [
        "",
        "Scenario definitions and comparability notes are in the JSON aggregate under `scenarios`.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        required=True,
        help="Model profile: a key in scripts/benchmark_profiles/ or a .toml path",
    )
    parser.add_argument(
        "--engines",
        default=None,
        help="Comma list of engines to run (stark and the profile's reference engines; default all)",
    )
    parser.add_argument(
        "--stark-binary",
        default=default_for("stark_binary"),
        help="stark-generate build under test, build name current (STARKBENCH_STARK_BINARY)",
    )
    parser.add_argument(
        "--stark-build",
        action="append",
        metavar="NAME=PATH",
        help="Another stark-generate build, for example a baseline; it runs every selected "
        "artifact as arm stark-ARTIFACT-NAME (repeatable)",
    )
    parser.add_argument(
        "--stark-artifact",
        action="append",
        metavar="NAME[=PATH]",
        help="Stark artifact to run, by its profile name, optionally at PATH instead of the "
        "profile's default (repeatable; default: the profile's default artifact)",
    )
    parser.add_argument(
        "--reference-binary",
        action="append",
        metavar="[ENGINE=]PATH",
        help="Reference engine executable in place of the profile's default (repeatable)",
    )
    parser.add_argument(
        "--reference-artifact",
        action="append",
        metavar="[ENGINE:]NAME[=PATH]",
        help="Reference pack to run, by its profile name, optionally at PATH (repeatable; "
        "default: every pack the profile gives a path)",
    )
    parser.add_argument(
        "--llmprobe",
        default=default_for("llmprobe"),
        help="llmprobe 0.6.18 checkout with the receipt patch bundled (STARKBENCH_LLMPROBE)",
    )
    parser.add_argument(
        "--node",
        default=default_for("node"),
        help="Node.js executable (STARKBENCH_NODE)",
    )
    parser.add_argument(
        "--output",
        default=default_for("output"),
        help="Directory for run directories (STARKBENCH_OUTPUT; default under ignored runtime/)",
    )
    parser.add_argument(
        "--lock-file",
        default=default_for("lock_file"),
        help="Exclusive lock shared with other GPU benchmarks (STARKBENCH_LOCK_FILE)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Measured repeats per cell (default 3, the fewest a standing can be read from)",
    )
    parser.add_argument(
        "--sessions",
        default=",".join(SESSION_KINDS),
        help="Comma list of probe, prefill and burst (default all)",
    )
    parser.add_argument(
        "--arms",
        default=None,
        help="Comma list of arm names to run, to fill in one arm in a new run directory "
        "that summarize merges with an earlier run",
    )
    parser.add_argument(
        "--layout",
        default="blocks",
        help="blocks (default): each repeat is one block of fresh sessions, arm order "
        "reversed on odd repeats. legacy: the first recorded runs' layout",
    )
    deviations = parser.add_argument_group(
        "diagnostic overrides",
        "Each of these is recorded and takes the run or the arm off the scoreboard.",
    )
    deviations.add_argument(
        "--warmups",
        type=int,
        default=1,
        help="Discarded prefill and burst trials per session (standard: 1)",
    )
    deviations.add_argument(
        "--probe-runs",
        type=int,
        default=None,
        help="Pass --runs to llmprobe (changes K and the ladder)",
    )
    deviations.add_argument(
        "--llmprobe-reasoning",
        default=None,
        help="Reasoning mode in place of the profile's: client-default or an llmprobe "
        "--reasoning value",
    )
    deviations.add_argument(
        "--cache-tag-base",
        default=None,
        help="Cache-bust tag base in place of the profile's (changes the rendered inputs)",
    )
    deviations.add_argument(
        "--allow-busy-host",
        action="store_true",
        help="Run even when competing model servers or compilers are running",
    )
    deviations.add_argument(
        "--allow-unpinned-llmprobe",
        action="store_true",
        help="Accept an llmprobe build whose hashes differ from the pinned one",
    )
    deviations.add_argument(
        "--stark-env",
        action="append",
        metavar="ARM=KEY=VALUE",
        help="Diagnostic environment variable for one Stark arm, by arm name (repeatable)",
    )
    deviations.add_argument(
        "--stark-drafts",
        choices=DRAFT_MODES,
        default="engine",
        help="off forces drafting off on every Stark request (the burst is always off)",
    )
    deviations.add_argument(
        "--stark-effort-alias",
        action="append",
        metavar="FROM=TO",
        help="Map a client reasoning effort the template rejects, for example medium=high",
    )
    deviations.add_argument(
        "--stark-min-context",
        type=int,
        default=None,
        help="Smallest planned worker context in place of the profile's; sizing only",
    )
    deviations.add_argument(
        "--reference-force-drafts",
        action="store_true",
        help="Start reference engines with drafting forced on (not in the burst)",
    )
    parser.add_argument(
        "--stark-worker-timeout",
        type=float,
        default=600.0,
        help="Worker reply timeout in seconds",
    )
    parser.add_argument(
        "--stark-image-check-dir",
        default=None,
        help="Optional persistent image-check store",
    )


def parser() -> argparse.ArgumentParser:
    raw = argparse.RawDescriptionHelpFormatter
    root = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0], epilog=__doc__, formatter_class=raw
    )
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "profiles", help="List the model profiles and what each can run"
    )
    plan = commands.add_parser(
        "plan",
        help="Validate a profile and print exactly what a run would start (no GPU)",
    )
    add_run_arguments(plan)
    run = commands.add_parser("run", help="Run the standard benchmark for one profile")
    add_run_arguments(run)
    run.add_argument(
        "--dry-run", action="store_true", help="The same as the plan command"
    )
    summary = commands.add_parser(
        "summarize", help="Join run directories into one table"
    )
    summary.add_argument(
        "runs", nargs="+", help="Run directories written by the run command"
    )
    summary.add_argument(
        "--reference-arm",
        default=None,
        help="Arm the speed ratios are taken against (default: the profile's)",
    )
    summary.add_argument(
        "--markdown",
        default=None,
        help="Markdown output (default <first run>/../summary.md)",
    )
    summary.add_argument(
        "--json", default=None, help="JSON output (default <first run>/../summary.json)"
    )
    quality = commands.add_parser(
        "quality", help="The quality slot beside each Stark arm's speed result"
    )
    actions = quality.add_subparsers(dest="quality_command", required=True)
    actions.add_parser("schema", help="Print the report schema the slot accepts")
    attach = actions.add_parser(
        "attach",
        help="Validate, pool and judge probe reports and bind the result to one arm of a run",
    )
    attach.add_argument("--run", required=True, help="Run directory")
    attach.add_argument("--arm", required=True, help="Stark arm name in that run")
    attach.add_argument(
        "--report",
        required=True,
        action="append",
        help="A probe's JSON report; repeat it for one report per input, and the "
        "reports are pooled over their rows before the bar is applied",
    )
    attach.add_argument(
        "--candidate-artifact",
        required=True,
        help="Artifact directory the probe measured; it must be the one the arm timed",
    )
    attach.add_argument(
        "--reference-artifact",
        default=None,
        help="Artifact directory the probe compared against",
    )
    attach.add_argument(
        "--format",
        default=QUALITY_SCHEMA,
        choices=QUALITY_FORMATS,
        help="Report format: the slot schema or a known probe's own output",
    )
    attach.add_argument(
        "--corpus-kind",
        required=True,
        choices=QUALITY_CORPUS_KINDS,
        help="What the probe ran on; only natural text gives a usable result",
    )
    attach.add_argument(
        "--corpus-sha256", default=None, help="SHA-256 of the private token file"
    )
    attach.add_argument(
        "--probe-binary", default=None, help="Probe executable (hashed)"
    )
    attach.add_argument("--probe-name", default=None, help="Short probe name")
    attach.add_argument(
        "--replace", action="store_true", help="Replace a report already attached"
    )
    return root


QUALITY_SCHEMA_EXAMPLE = {
    "schema": QUALITY_SCHEMA,
    "corpus": {
        "rows": 1024,
        "inputs": 1,
        "prompt_tokens": 4096,
        "context": 8192,
        "window": 1024,
    },
    "reference": {"path": "exact"},
    "candidate": {"path": "default"},
    "metrics": {
        "next_token_accuracy": {"reference": 0.0, "candidate": 0.0},
        "top1_agreement": {
            "all": 0.0,
            "strata": [
                {"reference_top1_probability_min": 0.5, "rows": 0, "agreement": 0.0},
                {"reference_top1_probability_min": 0.9, "rows": 0, "agreement": 0.0},
            ],
        },
        "kl_reference_to_candidate": {"mean": 0.0, "max": 0.0},
        "next_token_logprob_abs_change_mean": 0.0,
    },
}


def print_profiles(out=sys.stdout, directory: Path | None = None) -> int:
    status = 0
    for key, profile in available_profiles(directory):
        if isinstance(profile, BenchError):
            print(f"{key}: invalid ({profile})", file=out)
            status = 2
            continue
        print(f"{key}: {profile.title} [{profile.status}]", file=out)
        print(
            f"  source {profile.source.get('repo')} at {profile.source.get('revision')}",
            file=out,
        )
        print(
            "  Stark artifacts: "
            + ", ".join(
                f"{item.key} (match {item.match})" for item in profile.artifacts
            )
            + f"; default {profile.default_stark_artifacts()[0].key}",
            file=out,
        )
        for reference in profile.references:
            print(
                f"  reference {reference.engine}: "
                + ", ".join(
                    f"{reference.arm_prefix}-{item.key} (match {item.match})"
                    for item in reference.artifacts
                ),
                file=out,
            )
        print(f"  reference arm: {profile.reference_arm}", file=out)
    return status


def main(argv=None, *, host=host_snapshot, out=sys.stdout) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "profiles":
            return print_profiles(out)
        if args.command == "summarize":
            runs = [resolve_path(path) for path in args.runs]
            result = summarize(runs, args.reference_arm)
            markdown = (
                resolve_path(args.markdown)
                if args.markdown
                else runs[0].parent / "summary.md"
            )
            output = (
                resolve_path(args.json)
                if args.json
                else runs[0].parent / "summary.json"
            )
            markdown.write_text(render_markdown(result))
            output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            print(f"Wrote {markdown} and {output}", file=out)
            return 0
        if args.command == "quality":
            if args.quality_command == "schema":
                print(json.dumps(QUALITY_SCHEMA_EXAMPLE, indent=2), file=out)
                return 0
            report = attach_quality(
                resolve_path(args.run),
                args.arm,
                [resolve_path(path) for path in args.report],
                resolve_path(args.candidate_artifact),
                corpus_kind=args.corpus_kind,
                fmt=args.format,
                reference_artifact=resolve_path(args.reference_artifact),
                corpus_sha256=args.corpus_sha256,
                probe_binary=resolve_path(args.probe_binary),
                probe_name=args.probe_name,
                replace_existing=args.replace,
            )
            reason = (
                f" ({report['verdict_reason']})" if report["verdict_reason"] else ""
            )
            print(
                f"Quality result attached: {report['verdict']}{reason}; pooled over "
                f"{report['corpus']['rows']} rows of {report['corpus']['inputs']} input(s)",
                file=out,
            )
            return 0
        config = build_config(args)
        problems, facts = validate(config, strict=True)
        if args.command == "plan" or args.dry_run:
            snapshot = host()
            print_plan(config, problems, facts, snapshot, out=out)
            print_plan_details(config, facts, snapshot, out=out)
            return 2 if problems else 0
        if problems:
            for problem in problems:
                print(f"configuration problem: {problem}", file=sys.stderr)
            return 2
        # Refuse before taking the lock or holding caffeinate.
        require_timing_host(host(), allow_busy_host=config.allow_busy_host)
        os.umask(0o077)
        with exclusive_lock(config.lock_file), caffeinated():
            run_dir = run_matrix(config, facts, host=host)
        return report_run(run_dir, out)
    except HTTPError as error:
        print(f"HTTP status {error.code}", file=sys.stderr)
    except BenchError as error:
        print(str(error), file=sys.stderr)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ) as error:
        print(type(error).__name__, file=sys.stderr)
    return 1


def report_run(run_dir: Path, out=sys.stdout) -> int:
    """Say how a finished run ended; 3 when an arm failed."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    standard = manifest.get("standard") or {}
    if standard.get("deviations"):
        print(
            "Diagnostic run (off the scoreboard): " + "; ".join(standard["deviations"]),
            file=out,
        )
    if manifest.get("arm_failures"):
        print(f"Run finished with failed arms: {run_dir}", file=out)
        for item in manifest["arm_failures"]:
            print(
                f"  {item['arm']} ({item['kind']}): {item['failure']}; see {item['directory']}",
                file=out,
            )
        return 3
    print(f"Run complete: {run_dir}", file=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
