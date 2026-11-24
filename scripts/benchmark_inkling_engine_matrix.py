"""Inkling 2 Mini engine scoreboard: Stark against MLX-Serve through one client.

This is the original command line of the Inkling 2 Mini scoreboard. It is now
an entry point over the Stark standard benchmark
(scripts/benchmark_standard.py, protocol in docs/BENCHMARK.md) with the
inkling-2-mini profile and the session layout of the first recorded runs, so
those commands and their run directories keep working. New measurements
should use the standard tool, which adds profiles for other models, the block
layout, paired comparisons and the quality slot.

This harness reproduces MLX-Serve's published Inkling 2 Mini protocol on the
local Mac and measures Stark through the same client:

- every llmprobe 0.6.18 single-user benchmark scenario (greedy, 192 output
  tokens), including the context ladder (~512, ~4K, ~8K, ~16K) and the
  "sustained coding" cell that produced MLX-Serve's best published number;
- an exact 8,800-ID unseen-text prefill control with matching input-ID hashes;
- a 16-user burst, 300 tokens each, total tokens over wall time, draft off.

Arms are engine builds, not configuration flags. A Stark arm is one
`stark-generate` binary: `--stark-binary` is the current build (arm
`stark-current`) and each `--stark-arm NAME=PATH` adds another build (arm
`stark-NAME`), for example a build of an earlier commit as a baseline. An
MLX-Serve arm is one model pack served by the MLX-Serve binary.

Drafting is left to each engine. The Stark gateway forwards what the client
asked for and nothing else: no draft field when the client sends none, so the
worker's own default applies. MLX-Serve starts without a draft flag, so its
own default applies. The 16-user burst forces drafting off on both engines, as
its published protocol requires. Each Stark request records what was
requested and what the worker reported (effective drafts, proposals and
acceptances); the summary shows both per cell.

Reasoning defaults to a matched mode: llmprobe runs with `--reasoning off` on
both engines. `--llmprobe-reasoning client-default` restores llmprobe's own
default (`reasoning_effort: "medium"`, which MLX-Serve accepts and the Inkling
template rejects), optionally with `--stark-effort-alias medium=high`. The
mode is part of each arm's identity and is printed in the conditions table,
and the summary warns when one table mixes modes.

Receipts are aggregate only: counts, timings, SHA-256 hashes and engine notes.
Prompt text, completions, token vectors and probabilities are never written.
Raw llmprobe output is parsed in memory and discarded; its JSON report is
reduced to the benchmark section. llmprobe runs with `--no-save` and a
temporary HOME, so nothing is added to `~/.llmprobe`.

Prerequisites (all paths are arguments or `INKBENCH_*` environment variables;
the documented defaults point at ignored locations under `runtime/`):

- `cargo +1.98.1 build --release --locked -p stark-generate` and
  `cargo +1.98.1 build --release --locked -p stark-template` in this checkout
  (the gateway renders chat messages with the native template helper);
- a prepared Inkling 2 Mini Stark model directory;
- the MLX-Serve release binary and its model directories;
- an llmprobe 0.6.18 source checkout with node_modules installed and the
  aggregate-receipt patch applied and bundled (`bin/dist/llmprobe.mjs` must
  emit `[inkling-bench-sample]` lines);
- Node.js 22.19 or newer.

Timing runs refuse to start unless `pmset -g batt` reports AC power, Low Power
Mode is off and no competing model server or compiler is running. They hold
`caffeinate`, take an exclusive lock file and record power source, thermal
state, macOS version, hardware model and sleep events in the run manifest.
`--dry-run` validates the configuration, prints the planned matrix and exits
without starting a server, a worker or llmprobe.

Commands for the AC-power runs, from the repository root. One engine per
command:

    uv run --project tools/converter --locked --python 3.13.13 \\
      python scripts/benchmark_inkling_engine_matrix.py stark \\
      --stark-binary target/release/stark-generate \\
      --stark-model <prepared-inkling-dir> \\
      --llmprobe <llmprobe-0.6.18-checkout> \\
      --output runtime/inkling-engine-matrix

    uv run --project tools/converter --locked --python 3.13.13 \\
      python scripts/benchmark_inkling_engine_matrix.py mlx-serve \\
      --mlx-serve-binary <mlx-serve> \\
      --mlx-original-model <original-hf-dir> \\
      --mlx-author-8bit-model <author-8bit-dir> \\
      --mlx-community-4bit-model <community-4bit-dir> \\
      --llmprobe <llmprobe-0.6.18-checkout> \\
      --output runtime/inkling-engine-matrix

Or both engines in one command, alternating engines within every repeat:

    uv run --project tools/converter --locked --python 3.13.13 \\
      python scripts/benchmark_inkling_engine_matrix.py both \\
      --stark-binary target/release/stark-generate \\
      --stark-model <prepared-inkling-dir> \\
      --mlx-serve-binary <mlx-serve> \\
      --mlx-original-model <original-hf-dir> \\
      --llmprobe <llmprobe-0.6.18-checkout> \\
      --output runtime/inkling-engine-matrix

Then join any number of run directories into one gap table:

    uv run --project tools/converter --locked --python 3.13.13 \\
      python scripts/benchmark_inkling_engine_matrix.py summarize \\
      runtime/inkling-engine-matrix/<run> [runtime/inkling-engine-matrix/<run> ...]

A failed arm does not stop the others: its failure is recorded in the
manifest under `arm_failures`, the arm is skipped for the rest of the run, the
command exits with status 3, and `summarize` lists it. When a server fails to
start, the last lines of its own startup output (before any benchmark request)
are kept in `<arm>/.../<session>-server-startup-tail.txt` in the private run
directory. The dry run also checks that every weight shard a model's
`model.safetensors.index.json` names is present. To fill in one arm later,
run it alone into a new run directory with `--arms NAME` and pass both run
directories to `summarize`.

Append `--dry-run` to an engine command to check it on battery. Add a baseline
build with `--stark-arm baseline=<path-to-earlier-stark-generate>`.
`--stark-env NAME=KEY=VALUE` sets a diagnostic environment variable for one
Stark arm (NAME is `current` or a `--stark-arm` name); it is recorded in the
manifest and is not meant for scoreboard runs. `--repeats` (default 3) sets
the measured repeats per cell: the number of fresh-server llmprobe invocations
and the number of measured prefill and burst trials.
"""

from __future__ import annotations

import argparse
import datetime as dt  # noqa: F401 - part of the module's original surface
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError

import benchmark_standard as core
from benchmark_standard import (  # noqa: F401 - the original module surface
    ARM_NAME,
    BURST_CAP,
    BURST_WIDTH,
    DECODE_CAP,
    DECODE_METRIC,
    DRAFT_MODES,
    ENV_KEY,
    EXACT_PREFILL_SEED,
    EXACT_PREFILL_TOKENS,
    HOST,
    PREFILL_METRIC,
    REASONING_CLIENT_DEFAULT,
    SCENARIO_BY_KEY,
    SCENARIOS,
    SCHEMA,
    STARK_CACHE_SCOPE,
    SUMMARY_SCHEMA,
    Arm,
    BenchError,
    Receipts,
    Scenario,
    StartupTail,
    burst,
    burst_prompts,
    caffeinated,
    completion,
    draft_fields,
    engine_mtp_report,
    exact_prefill,
    exclusive_lock,
    fetch,
    host_snapshot,
    http_request,
    ids_digest,
    interleave,
    join_native,
    llmprobe_command,
    llmprobe_identity,
    model_identity,
    native_metrics,
    node_version,
    numeric_tree,
    parse_low_power_mode,
    parse_pmset_batt,
    parse_pmset_therm,
    read_jsonl,
    render_markdown,
    require_timing_host,
    resolve_path,
    run_llmprobe,
    scenario_for_label,
    sha256_file,
    sleep_events,
    template_helper,
    text_digest,
    weight_problem,
)

# Everything model-specific comes from the profile, as in the standard tool.
INKLING = core.load_profile("inkling-2-mini")
MLX_SERVE = INKLING.reference("mlx-serve")
STARK_MODEL_ID = INKLING.served_model_id
STARK_MIN_CONTEXT = INKLING.min_context
STARK_MAX_CONTEXT = INKLING.max_context
STARK_BURST_CONTEXT = INKLING.burst_context
INKLING_VOCAB = INKLING.vocab_size
TITLE = "Inkling 2 Mini engine matrix"

# Documented defaults. Every location is ignored by Git.
DEFAULTS = {
    "stark_binary": ("INKBENCH_STARK_BINARY", "target/release/stark-generate"),
    "stark_model": ("INKBENCH_STARK_MODEL", "runtime/inkling-2-mini-stark"),
    "mlx_serve_binary": ("INKBENCH_MLX_SERVE_BINARY", "runtime/mlx-serve/mlx-serve"),
    "mlx_original_model": (
        "INKBENCH_MLX_ORIGINAL_MODEL",
        "runtime/inkling-2-mini-original",
    ),
    "mlx_author_8bit_model": ("INKBENCH_MLX_AUTHOR_8BIT_MODEL", None),
    "mlx_community_4bit_model": ("INKBENCH_MLX_COMMUNITY_4BIT_MODEL", None),
    "llmprobe": ("INKBENCH_LLMPROBE", "runtime/llmprobe-0.6.18"),
    "output": ("INKBENCH_OUTPUT", "runtime/inkling-engine-matrix"),
    "lock_file": ("INKBENCH_LOCK_FILE", "runtime/model-scoring.lock"),
    "node": ("INKBENCH_NODE", "node"),
}


def default_for(name: str) -> str | None:
    variable, fallback = DEFAULTS[name]
    return os.environ.get(variable, fallback)


@dataclass
class Config(core.Config):
    """The standard configuration with this command line's fixed choices."""

    stark_min_context: int | None = STARK_MIN_CONTEXT
    profile: core.Profile | None = INKLING
    layout: str = "legacy"
    entry: Path | None = Path(__file__)


def with_profile(config):
    """Callers of the original functions pass configurations without a profile."""
    if getattr(config, "profile", None) is not None:
        return config
    return SimpleNamespace(**{**vars(config), "profile": INKLING})


# --------------------------------------------------------------------------
# The original function signatures, bound to the Inkling 2 Mini profile
# --------------------------------------------------------------------------


def choose_context(
    burst_mode: bool,
    total: int,
    min_context=STARK_MIN_CONTEXT,
    max_context=STARK_MAX_CONTEXT,
) -> int:
    return core.choose_context(
        burst_mode, total, min_context, max_context, STARK_BURST_CONTEXT
    )


class Gateway(core.Gateway):
    def __init__(
        self,
        worker,
        *,
        model_id=STARK_MODEL_ID,
        min_context=STARK_MIN_CONTEXT,
        vocab_size=INKLING_VOCAB,
        **options,
    ):
        super().__init__(
            worker,
            model_id=model_id,
            min_context=min_context,
            max_context=STARK_MAX_CONTEXT,
            burst_context=STARK_BURST_CONTEXT,
            vocab_size=vocab_size,
            **options,
        )


def stark_adapters(model: Path):
    return core.chat_adapters(model, INKLING.chat_adapter)[:3]


def model_vocab(model: Path) -> int:
    return core.model_vocab(model, INKLING_VOCAB)


def stark_session(
    config,
    arm: Arm,
    receipts: Receipts,
    kind: str,
    *,
    worker_factory=core.default_worker_factory,
    adapters=stark_adapters,
):
    return core.stark_session(
        config,
        arm,
        receipts,
        kind,
        profile=getattr(config, "profile", None) or INKLING,
        worker_factory=worker_factory,
        adapters=adapters,
    )


def mlx_command(
    binary: Path, model: Path, port: int, kind: str, force_mtp=False
) -> list[str]:
    return core.mlx_command(
        binary,
        model,
        port,
        kind,
        force_mtp,
        max_context=STARK_MAX_CONTEXT,
        burst_context=STARK_BURST_CONTEXT,
        flags=MLX_SERVE.flags,
        draft_off_flags=MLX_SERVE.draft_off_flags,
        draft_force_flags=MLX_SERVE.draft_force_flags,
    )


def mlx_server_records(lines, receipts: Receipts, tail=None) -> None:
    core.mlx_server_records(lines, receipts, tail, MLX_SERVE.log_events)


def mlx_session(
    config, arm: Arm, receipts: Receipts, kind: str, *, startup_timeout=300
):
    return core.mlx_session(
        with_profile(config), arm, receipts, kind, startup_timeout=startup_timeout
    )


def engine_session(config, arm, receipts, kind):
    return core.engine_session(with_profile(config), arm, receipts, kind)


def run_matrix(
    config,
    facts: dict,
    *,
    session_factory=engine_session,
    host=host_snapshot,
    sleep_check=sleep_events,
) -> Path:
    return core.run_matrix(
        with_profile(config),
        facts,
        session_factory=session_factory,
        host=host,
        sleep_check=sleep_check,
    )


def summarize(run_dirs: list[Path], reference_arm="mlx-original") -> dict:
    return core.summarize(run_dirs, reference_arm, title=TITLE)


# --------------------------------------------------------------------------
# Configuration: the original arm flags
# --------------------------------------------------------------------------


def stark_arms(args) -> tuple[list[Arm], dict]:
    model = resolve_path(args.stark_model)
    arms = [
        Arm(
            "stark-current",
            "stark",
            resolve_path(args.stark_binary),
            model,
            "Stark build from --stark-binary",
        )
    ]
    for spec in args.stark_arm or ():
        name, _, path = spec.partition("=")
        if not ARM_NAME.fullmatch(name) or name == "current" or not path:
            raise BenchError("invalid_stark_arm")
        arms.append(
            Arm(
                f"stark-{name}",
                "stark",
                resolve_path(path),
                model,
                f"Stark build {name}",
            )
        )
    by_name = {arm.name: arm for arm in arms}
    for spec in getattr(args, "stark_arm_model", None) or ():
        name, _, path = spec.partition("=")
        arm = by_name.get(f"stark-{name}")
        if arm is None or not path:
            raise BenchError("invalid_stark_arm_model")
        arm.model = resolve_path(path)
        arm.model_label = arm.model.name
    core.apply_stark_env(by_name, args.stark_env, prefix="stark-")
    return arms, core.parse_effort_aliases(args.stark_effort_alias)


def mlx_arms(args) -> list[Arm]:
    binary = resolve_path(args.mlx_serve_binary)
    packs = (
        ("original", args.mlx_original_model, "Original checkpoint"),
        ("author-8bit", args.mlx_author_8bit_model, "Author 8-bit pack"),
        ("community-4bit", args.mlx_community_4bit_model, "Community 4-bit pack"),
    )
    return [
        Arm(f"mlx-{name}", "mlx-serve", binary, resolve_path(path), description)
        for name, path, description in packs
        if path
    ]


def build_config(args) -> Config:
    core.check_run_arguments(args)
    sessions = core.parse_sessions(args.sessions)
    stark, aliases = stark_arms(args) if args.engine in ("stark", "both") else ([], {})
    mlx = mlx_arms(args) if args.engine in ("mlx-serve", "both") else []
    arms = interleave(stark, mlx)
    selected = core.select_arms(arms, args.arms)
    if selected is arms and args.engine == "both" and not (stark and mlx):
        raise BenchError("no_arms")
    arms = selected
    if not arms:
        raise BenchError("no_arms")
    if len({arm.name for arm in arms}) != len(arms):
        raise BenchError("duplicate_arm_names")
    return Config(
        engine=args.engine,
        arms=arms,
        llmprobe=resolve_path(args.llmprobe),
        node=args.node,
        output=resolve_path(args.output),
        lock_file=resolve_path(args.lock_file),
        repeats=args.repeats,
        warmups=args.warmups,
        sessions=sessions,
        probe_runs=args.probe_runs,
        reasoning=args.llmprobe_reasoning,
        cache_tag_base=args.cache_tag_base,
        allow_busy_host=args.allow_busy_host,
        allow_unpinned_llmprobe=args.allow_unpinned_llmprobe,
        stark_drafts=getattr(args, "stark_drafts", "engine"),
        stark_worker_timeout=getattr(args, "stark_worker_timeout", 600.0),
        stark_image_check_dir=resolve_path(
            getattr(args, "stark_image_check_dir", None)
        ),
        stark_effort_aliases=aliases,
        stark_min_context=getattr(args, "stark_min_context", STARK_MIN_CONTEXT),
        mlx_force_mtp=getattr(args, "mlx_force_mtp", False),
    )


def validate(
    config: Config, *, node_check=None, template_check=None
) -> tuple[list[str], dict]:
    """Return (problems, facts). Reads files only; starts no model process."""
    return core.validate(
        config,
        node_check=node_check or node_version,
        template_check=template_check or template_helper,
    )


def print_plan(
    config: Config, problems: list[str], facts: dict, host: dict, out=sys.stdout
) -> None:
    core.print_plan(config, problems, facts, host, out, title="Inkling engine matrix")


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--llmprobe",
        default=default_for("llmprobe"),
        help="llmprobe 0.6.18 checkout with the receipt patch bundled (INKBENCH_LLMPROBE)",
    )
    parser.add_argument(
        "--node", default=default_for("node"), help="Node.js executable (INKBENCH_NODE)"
    )
    parser.add_argument(
        "--output",
        default=default_for("output"),
        help="Directory for run directories (INKBENCH_OUTPUT; default under ignored runtime/)",
    )
    parser.add_argument(
        "--lock-file",
        default=default_for("lock_file"),
        help="Exclusive lock shared with other GPU benchmarks (INKBENCH_LOCK_FILE)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Measured repeats per cell: llmprobe invocations and prefill/burst trials (default 3)",
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=1,
        help="Discarded prefill and burst trials (default 1)",
    )
    parser.add_argument(
        "--sessions",
        default="probe,prefill,burst",
        help="Comma list of probe, prefill and burst (default all)",
    )
    parser.add_argument(
        "--probe-runs",
        type=int,
        default=None,
        help="Pass --runs to llmprobe (changes K and the ladder; not the published protocol)",
    )
    parser.add_argument(
        "--llmprobe-reasoning",
        default="off",
        help="Reasoning mode: off (default, matched on both engines), client-default "
        "(llmprobe's reasoning_effort medium), or another llmprobe --reasoning value",
    )
    parser.add_argument(
        "--cache-tag-base",
        default="inkbench",
        help="Cache-bust tag base; invocation i uses BASE plus i, identical across engines",
    )
    parser.add_argument(
        "--allow-busy-host",
        action="store_true",
        help="Run even when competing model servers or compilers are running (recorded)",
    )
    parser.add_argument(
        "--allow-unpinned-llmprobe",
        action="store_true",
        help="Accept an llmprobe build whose hashes differ from the pinned one (recorded)",
    )
    parser.add_argument(
        "--arms",
        default=None,
        help=(
            "Comma list of arm names to run (for example mlx-author-8bit), to fill in one "
            "arm in a new run directory that summarize merges with an earlier run"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate, print the planned matrix and exit without starting anything",
    )


def add_stark(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--stark-binary",
        default=default_for("stark_binary"),
        help="Current stark-generate build, arm stark-current (INKBENCH_STARK_BINARY)",
    )
    parser.add_argument(
        "--stark-model",
        default=default_for("stark_model"),
        help="Prepared Inkling 2 Mini model directory (INKBENCH_STARK_MODEL)",
    )
    parser.add_argument(
        "--stark-arm",
        action="append",
        metavar="NAME=PATH",
        help="Another stark-generate build as arm stark-NAME, for example a baseline (repeatable)",
    )
    parser.add_argument(
        "--stark-arm-model",
        action="append",
        metavar="NAME=PATH",
        help="Prepared model directory for one Stark arm (NAME current or a --stark-arm name) "
        "in place of --stark-model, for example another artifact of the same model (repeatable)",
    )
    parser.add_argument(
        "--stark-env",
        action="append",
        metavar="NAME=KEY=VALUE",
        help="Diagnostic environment variable for one Stark arm (NAME current or a --stark-arm "
        "name); recorded, not for scoreboard runs",
    )
    parser.add_argument(
        "--stark-drafts",
        choices=DRAFT_MODES,
        default="engine",
        help="engine (default): forward the client's request, worker default otherwise; "
        "off: force drafting off on every Stark request. The burst is always off",
    )
    parser.add_argument(
        "--stark-effort-alias",
        action="append",
        metavar="FROM=TO",
        help="Map a client reasoning effort the template rejects, for example medium=high "
        "(only relevant with --llmprobe-reasoning client-default)",
    )
    parser.add_argument(
        "--stark-min-context",
        type=int,
        default=STARK_MIN_CONTEXT,
        help="Smallest planned worker context (default 4352); sizing only",
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


def add_mlx(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mlx-serve-binary",
        default=default_for("mlx_serve_binary"),
        help="MLX-Serve executable (INKBENCH_MLX_SERVE_BINARY)",
    )
    parser.add_argument(
        "--mlx-original-model",
        default=default_for("mlx_original_model"),
        help="Original checkpoint directory (INKBENCH_MLX_ORIGINAL_MODEL)",
    )
    parser.add_argument(
        "--mlx-author-8bit-model",
        default=default_for("mlx_author_8bit_model"),
        help="Optional author 8-bit pack (INKBENCH_MLX_AUTHOR_8BIT_MODEL)",
    )
    parser.add_argument(
        "--mlx-community-4bit-model",
        default=default_for("mlx_community_4bit_model"),
        help="Optional community 4-bit pack (INKBENCH_MLX_COMMUNITY_4BIT_MODEL)",
    )
    parser.add_argument(
        "--mlx-force-mtp",
        action="store_true",
        help="Start MLX-Serve with --mtp instead of its own default (not in the burst)",
    )


def parser() -> argparse.ArgumentParser:
    raw = argparse.RawDescriptionHelpFormatter
    root = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0], epilog=__doc__, formatter_class=raw
    )
    commands = root.add_subparsers(dest="command", required=True)
    stark = commands.add_parser(
        "stark", help="Run the Stark arms", formatter_class=raw, epilog=__doc__
    )
    add_stark(stark)
    add_common(stark)
    mlx = commands.add_parser(
        "mlx-serve", help="Run the MLX-Serve arms", formatter_class=raw, epilog=__doc__
    )
    add_mlx(mlx)
    add_common(mlx)
    both = commands.add_parser(
        "both",
        help="Run both engines, alternating within each repeat",
        formatter_class=raw,
        epilog=__doc__,
    )
    add_stark(both)
    add_mlx(both)
    add_common(both)
    summary = commands.add_parser(
        "summarize", help="Join run directories into one gap table"
    )
    summary.add_argument(
        "runs", nargs="+", help="Run directories written by the engine commands"
    )
    summary.add_argument("--reference-arm", default="mlx-original")
    summary.add_argument(
        "--markdown",
        default=None,
        help="Markdown output (default <first run>/../summary.md)",
    )
    summary.add_argument(
        "--json", default=None, help="JSON output (default <first run>/../summary.json)"
    )
    return root


def main(argv=None, *, host=host_snapshot, out=sys.stdout) -> int:
    args = parser().parse_args(argv)
    try:
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
        args.engine = args.command
        config = build_config(args)
        problems, facts = validate(config)
        if args.dry_run:
            print_plan(config, problems, facts, host(), out=out)
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
        manifest = json.loads((run_dir / "manifest.json").read_text())
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


if __name__ == "__main__":
    raise SystemExit(main())
