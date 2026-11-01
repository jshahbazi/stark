"""Ephemeral locked native logit probes and aggregate-only numerical comparison.

The probe is generated from Stark's existing private GGUF experiment worker;
the only runtime changes are its scoped 40,960 context cap and an absolute
protocol source path. It is never installed or added to a product API. Full
vocabulary vectors travel through private pipes and stay in Python memory.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import subprocess
import tomllib
from pathlib import Path

import benchmark_native_prefill as native
import gguf_experiment as common
import qwen3_prefill_experiment as existing

MAX_CONTEXT = 40960


def json_file(path, value):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600
    )
    with os.fdopen(descriptor, "w") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def dependency_version(lock, name, prefix):
    versions = {
        package["version"]
        for package in lock["package"]
        if package["name"] == name and package["version"].startswith(prefix)
    }
    common.require(len(versions) == 1, "ambiguous_probe_dependency")
    return next(iter(versions))


def external_packages(lock):
    return {
        (p["name"], p["version"], p["source"], p.get("checksum"))
        for p in lock["package"]
        if "source" in p
    }


def prepare_probe(args, runner=subprocess.run):
    root = args.source_root.resolve(strict=True)
    project = args.project.absolute()
    common.require(
        not project.exists() and not project.is_symlink(), "probe_project_exists"
    )
    source_path = root / "crates/stark-score/probes/gguf_experiment.rs"
    protocol_path = root / "crates/stark-score/src/protocol.rs"
    source = source_path.read_text()
    old_path, old_cap = '#[path = "../src/protocol.rs"]', "context > 16384"
    common.require(
        source.count(old_path) == source.count(old_cap) == 1,
        "probe_source_contract_changed",
    )
    source = source.replace(
        old_path, "#[path = " + json.dumps(str(protocol_path), ensure_ascii=False) + "]"
    )
    source = source.replace(old_cap, f"context > {MAX_CONTEXT}")
    lock_bytes = (root / "Cargo.lock").read_bytes()
    lock = tomllib.loads(lock_bytes.decode())
    engine_path = json.dumps(str(root / "crates/stark-engine"), ensure_ascii=False)
    versions = {
        name: dependency_version(lock, name, prefix)
        for name, prefix in (
            ("serde", "1."),
            ("serde_json", "1."),
            ("libc", "0.2."),
            ("sha2", "0.10."),
        )
    }
    cargo = (
        '[package]\nname = "stark-private-prefill-probe"\nversion = "0.0.0"\nedition = "2024"\n'
        "[workspace]\n[dependencies]\nstark-engine = { path = "
        + engine_path
        + ', default-features = false, features = ["metal"] }\n'
        'serde = { version = "='
        + versions["serde"]
        + '", features = ["derive"] }\n'
        + "".join(
            name + ' = "=' + versions[name] + '"\n'
            for name in ("serde_json", "libc", "sha2")
        )
    )
    workspace = tomllib.loads((root / "Cargo.toml").read_text())
    for registry, dependencies in workspace.get("patch", {}).items():
        cargo += "\n[patch." + json.dumps(registry) + "]\n"
        for name, entry in dependencies.items():
            common.require(
                isinstance(entry, dict) and set(entry) == {"path"},
                "unsupported_probe_patch",
            )
            cargo += (
                json.dumps(name)
                + " = { path = "
                + json.dumps(str(root / entry["path"]), ensure_ascii=False)
                + " }\n"
            )
    cargo += "\n[profile.release]\n"
    for name, value in workspace.get("profile", {}).get("release", {}).items():
        common.require(
            type(value) in (str, bool, int), "unsupported_probe_release_profile"
        )
        cargo += json.dumps(name) + " = " + json.dumps(value) + "\n"
    project.mkdir(mode=0o700, parents=True)
    (project / "src").mkdir(mode=0o700)
    (project / "Cargo.toml").write_text(cargo)
    (project / "Cargo.lock").write_bytes(lock_bytes)
    (project / "src/main.rs").write_text(source)
    seal = native.source_seal(root)
    report = {
        "schema_version": 1,
        "scope": "private_native_prefill_logit_probe",
        "maximum_context": MAX_CONTEXT,
        "source_tree_sha256": seal,
        "origin_probe_sha256": common.digest_file(source_path),
        "protocol_sha256": common.digest_file(protocol_path),
        "helper_source_sha256": common.digest_file(project / "src/main.rs"),
        "generator_sha256": common.digest_file(Path(__file__)),
        "origin_lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        "built": False,
    }
    if args.build:
        executable = Path.home() / ".cargo/bin/cargo"
        command = [str(executable) if executable.is_file() else "cargo", "+1.98.1"]

        def checked(arguments):
            result = runner(
                arguments,
                cwd=project,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=1800,
                check=False,
            )
            common.require(result.returncode == 0, "probe_build_failed")

        checked(
            [
                *command,
                "metadata",
                "--offline",
                "--format-version",
                "1",
                "--manifest-path",
                str(project / "Cargo.toml"),
            ]
        )
        resolved = tomllib.loads((project / "Cargo.lock").read_text())
        common.require(
            external_packages(resolved) <= external_packages(lock),
            "probe_dependency_lock_drift",
        )
        checked(
            [
                *command,
                "build",
                "--release",
                "--locked",
                "--offline",
                "--manifest-path",
                str(project / "Cargo.toml"),
                "--target-dir",
                str(args.target.resolve()),
            ]
        )
        report.update(
            built=True,
            external_dependencies_from_repository_lock=True,
            helper_binary_sha256=common.digest_file(
                args.target.resolve() / "release/stark-private-prefill-probe"
            ),
            resolved_lock_sha256=common.digest_file(project / "Cargo.lock"),
        )
    report["source_unchanged"] = seal == native.source_seal(root)
    report["generator_unchanged"] = report["generator_sha256"] == common.digest_file(
        Path(__file__)
    )
    common.require(report["source_unchanged"], "probe_source_changed")
    common.require(report["generator_unchanged"], "probe_generator_changed")
    json_file(project / "probe-receipt.json", report)
    return report


class ProbeTokenizer:
    """Adapt native fixture tokenization to the probe's fixed-false protocol."""

    def __init__(self, worker):
        self.worker = worker

    def request(self, request):
        common.require(
            isinstance(request, dict)
            and set(request) == {"operation", "text", "add_special_tokens"}
            and request["operation"] == "tokenize"
            and isinstance(request["text"], str)
            and request["add_special_tokens"] is False,
            "probe_tokenization_requires_no_special_tokens",
        )
        # Request::Tokenize rejects unknown fields and explicitly encodes false.
        return self.worker.request({"operation": "tokenize", "text": request["text"]})


def prepare_probe_fixtures(workers, model, target_tokens, context, output_tokens):
    return native.prepare_fixtures(
        {label: ProbeTokenizer(worker) for label, worker in workers.items()},
        model,
        target_tokens,
        context,
        output_tokens,
    )


def compare(
    args, *, worker_factory=common.Worker, fixture_factory=prepare_probe_fixtures
):
    common.require(
        1 <= args.steps <= 8
        and args.target_tokens >= 64
        and args.target_tokens + args.steps <= args.context <= MAX_CONTEXT,
        "comparison_limits_invalid",
    )
    binding = {
        "baseline": common.digest_file(args.baseline),
        "candidate": common.digest_file(args.candidate),
        "configuration": common.digest_file(args.model / "config.json"),
        "tokenizer": common.digest_file(args.model / "tokenizer.json"),
        "weights": common.digest_file(args.model / "model.safetensors"),
        "harness": common.digest_file(Path(__file__)),
        "fixture_harness": common.digest_file(Path(native.__file__)),
        "comparison_harness": common.digest_file(Path(common.__file__)),
        "score_validation_harness": common.digest_file(Path(existing.__file__)),
        "model_and_rendering": native.input_bindings(args.model),
    }
    rows = []
    with contextlib.ExitStack() as stack:
        workers = {}
        for label in ("baseline", "candidate"):
            worker = worker_factory(
                getattr(args, label).resolve(), args.model.resolve(), 120
            )
            stack.callback(worker.close)
            workers[label] = worker
        vocab = workers["baseline"].ready.get("vocab_size")
        common.require(
            type(vocab) is int
            and vocab == workers["candidate"].ready.get("vocab_size")
            and vocab >= 32,
            "vocabulary_mismatch",
        )
        fixtures = fixture_factory(
            workers, args.model, args.target_tokens, args.context, args.steps
        )
        report = {
            "schema_version": 1,
            "scope": "full_vocabulary_native_prefill_numerics",
            "binding": binding,
            "context_tokens": args.context,
            "vocabulary": vocab,
            "teacher_forced_steps": args.steps,
            "probe_tokenization_add_special_tokens": False,
            "fixtures": [{"name": f["name"], **f["metadata"]} for f in fixtures],
            "by_case": rows,
            "native_seeded_generation_compared_here": False,
            "numeric_acceptance": "unqualified",
        }
        cases = [(f["name"], list(f["ids"]), args.steps) for f in fixtures]
        for count in args.prefix_lengths:
            common.require(
                1 <= count <= len(fixtures[-1]["ids"]),
                "diagnostic_prefix_length_invalid",
            )
            cases.append(
                ("diagnostic_synthetic_prefix", fixtures[-1]["ids"][:count], 1)
            )
        for name, ids, steps in cases:
            for step in range(steps):
                values, hashes, repeatable, metrics = {}, {}, {}, {}
                request = {
                    "operation": "score",
                    "input_token_ids": ids,
                    "context_tokens": args.context,
                }
                for label, worker in workers.items():
                    response = worker.request(request)
                    values[label] = existing.validate_score(response, ids, vocab)
                    hashes[label] = response["logits_sha256"]
                    metrics[label] = {
                        key: value
                        for key, value in response.get("metrics", {}).items()
                        if key in existing.SCORE_METRICS and native.finite(value)
                    }
                    if args.check_repeats:
                        again = worker.request(request)
                        existing.validate_score(again, ids, vocab)
                        repeatable[label] = again["logits_sha256"] == hashes[label]
                rows.append(
                    {
                        "fixture": name,
                        "position": step,
                        "input_tokens": len(ids),
                        "input_sha256": common.digest_ids(ids),
                        "exact_logits": hashes["baseline"] == hashes["candidate"],
                        "repeat_exact": repeatable,
                        "numerics": common.compare_logits(
                            values["baseline"], values["candidate"]
                        ),
                        "score_metrics": metrics,
                    }
                )
                # Shared reference continuation, never each candidate's divergent
                # sequence: later rows compare the same complete input on both sides.
                next_token = max(range(vocab), key=values["baseline"].__getitem__)
                ids = [*ids, next_token]
    report["workers_unchanged"] = all(
        common.digest_file(getattr(args, label)) == binding[label]
        for label in ("baseline", "candidate")
    )
    report["model_and_rendering_unchanged"] = binding[
        "model_and_rendering"
    ] == native.input_bindings(args.model)
    report["harnesses_unchanged"] = all(
        common.digest_file(Path(path)) == binding[key]
        for key, path in (
            ("harness", __file__),
            ("fixture_harness", native.__file__),
            ("comparison_harness", common.__file__),
            ("score_validation_harness", existing.__file__),
        )
    )
    report["passed"] = (
        report["workers_unchanged"]
        and report["model_and_rendering_unchanged"]
        and report["harnesses_unchanged"]
        and all(all(row["repeat_exact"].values()) for row in rows)
    )
    report["exact_logit_cases"] = sum(row["exact_logits"] for row in rows)
    report["greedy_agreement_cases"] = sum(
        row["numerics"]["top1_agreement"] for row in rows
    )
    report["case_count"] = len(rows)
    # Successful finite comparisons are measurements, not a tolerance contract.
    # Native greedy/fixed-seed sampled behavior is checked by the paired
    # generation harness; no substitute RNG or sampling repair is invented here.
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    prepare = sub.add_parser("prepare-probe")
    prepare.add_argument("--source-root", type=Path, required=True)
    prepare.add_argument("--project", type=Path, required=True)
    prepare.add_argument(
        "--target",
        type=Path,
        required=True,
        help="Use a distinct target per source checkout.",
    )
    prepare.add_argument("--build", action="store_true")
    measure = sub.add_parser("compare")
    measure.add_argument("--baseline", type=Path, required=True)
    measure.add_argument("--candidate", type=Path, required=True)
    measure.add_argument("--model", type=Path, required=True)
    measure.add_argument("--output", type=Path, required=True)
    measure.add_argument("--target-tokens", type=int, default=18544)
    measure.add_argument("--context", type=int, default=40960)
    measure.add_argument("--steps", type=int, default=2)
    measure.add_argument(
        "--prefix-lengths", type=int, nargs="*", default=[1023, 1024, 1025]
    )
    measure.add_argument("--check-repeats", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.operation == "prepare-probe":
            report = prepare_probe(args)
        else:
            common.require(
                not args.output.exists() and not args.output.is_symlink(),
                "output_exists",
            )
            report = compare(args)
            json_file(args.output, report)
        print(json.dumps(report, allow_nan=False), flush=True)
        return 0 if report.get("passed", report.get("source_unchanged", False)) else 1
    except common.ExperimentFailure as error:
        code = str(error)
        if not re.fullmatch(r"[a-z0-9_]{1,80}", code):
            code = "private_prefill_logit_validation_failed"
        print(json.dumps({"passed": False, "stage": code}), flush=True)
        return 1
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ):
        print(
            json.dumps(
                {"passed": False, "stage": "private_prefill_logit_validation_failed"}
            ),
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
