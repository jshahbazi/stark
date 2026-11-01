from __future__ import annotations

import json
import tomllib
from argparse import Namespace

import benchmark_native_prefill as native
import gguf_experiment as common
import pytest
import qwen3_prefill_experiment as existing
import validate_native_prefill_logits as validation


def test_fixture_tokenization_matches_the_probes_strict_false_only_protocol():
    class StrictProbe:
        def __init__(self):
            self.requests = []

        def request(self, request):
            assert set(request) == {"operation", "text"}
            assert request["operation"] == "tokenize"
            self.requests.append(request)
            return {"token_ids": [3, 7]}

    probe = StrictProbe()
    adapter = validation.ProbeTokenizer(probe)
    assert native.tokenize(adapter, "synthetic tokenizer contract") == [3, 7]
    assert probe.requests == [
        {"operation": "tokenize", "text": "synthetic tokenizer contract"}
    ]
    for unsupported in [
        {"operation": "tokenize", "text": "synthetic", "add_special_tokens": True},
        {"operation": "tokenize", "text": "synthetic"},
        {
            "operation": "tokenize",
            "text": "synthetic",
            "add_special_tokens": False,
            "unknown": 1,
        },
    ]:
        with pytest.raises(
            common.ExperimentFailure,
            match="probe_tokenization_requires_no_special_tokens",
        ):
            adapter.request(unsupported)
    assert len(probe.requests) == 1


def write_source_root(path):
    path.mkdir()
    probe = path / "crates/stark-score/probes/gguf_experiment.rs"
    probe.parent.mkdir(parents=True)
    probe.write_text(
        '#[path = "../src/protocol.rs"]\nmod protocol;\n// context > 16384\n'
    )
    protocol = path / "crates/stark-score/src/protocol.rs"
    protocol.parent.mkdir()
    protocol.write_text("// fixed private protocol fixture\n")
    (path / "Cargo.toml").write_text(
        '[patch."https://example.test/cxx"]\ncxx = { path = "vendor/cxx" }\n[profile.release]\nlto = "thin"\n'
    )
    versions = [
        ("serde", "1.0.228"),
        ("serde_json", "1.0.150"),
        ("libc", "0.2.183"),
        ("sha2", "0.10.9"),
    ]
    lock = "version = 4\n" + "".join(
        f'\n[[package]]\nname = "{name}"\nversion = "{version}"\nsource = "registry+https://example.test/index"\nchecksum = "'
        + "a" * 64
        + '"\n'
        for name, version in versions
    )
    (path / "Cargo.lock").write_text(lock)


def test_prepare_probe_preserves_source_and_features_and_pins_dependencies(
    tmp_path, monkeypatch
):
    root = tmp_path / "source"
    write_source_root(root)
    monkeypatch.setattr(native, "source_seal", lambda _root: "f" * 64)
    original = (root / "crates/stark-score/probes/gguf_experiment.rs").read_bytes()
    args = Namespace(
        source_root=root,
        project=tmp_path / "private-project",
        target=tmp_path / "target",
        build=False,
    )
    report = validation.prepare_probe(args)
    generated = (args.project / "src/main.rs").read_text()
    assert "context > 40960" in generated and "context > 16384" not in generated
    assert str(root / "crates/stark-score/src/protocol.rs") in generated
    assert (
        root / "crates/stark-score/probes/gguf_experiment.rs"
    ).read_bytes() == original
    cargo = tomllib.loads((args.project / "Cargo.toml").read_text())
    assert cargo["dependencies"]["stark-engine"]["default-features"] is False
    assert cargo["dependencies"]["stark-engine"]["features"] == ["metal"]
    assert cargo["dependencies"]["sha2"] == "=0.10.9"
    assert cargo["patch"]["https://example.test/cxx"]["cxx"]["path"] == str(
        root / "vendor/cxx"
    )
    assert cargo["profile"]["release"]["lto"] == "thin"
    assert report["source_unchanged"] is True and report["built"] is False
    with pytest.raises(common.ExperimentFailure, match="probe_project_exists"):
        validation.prepare_probe(args)


def test_probe_build_refuses_any_dependency_outside_the_repository_lock(
    tmp_path, monkeypatch
):
    root = tmp_path / "source"
    write_source_root(root)
    monkeypatch.setattr(native, "source_seal", lambda _root: "f" * 64)
    args = Namespace(
        source_root=root,
        project=tmp_path / "private-project",
        target=tmp_path / "target",
        build=True,
    )
    commands = []

    def runner(command, **_options):
        commands.append(command)
        lock = args.project / "Cargo.lock"
        lock.write_text(
            lock.read_text()
            + '\n[[package]]\nname = "unexpected"\nversion = "9.0.0"\nsource = "registry+https://example.test/index"\n'
        )
        return Namespace(returncode=0)

    with pytest.raises(common.ExperimentFailure, match="probe_dependency_lock_drift"):
        validation.prepare_probe(args, runner=runner)
    assert len(commands) == 1 and "metadata" in commands[0]


def test_aggregate_comparison_keeps_logits_private_and_marks_numeric_acceptance_unqualified(
    tmp_path,
):
    model = tmp_path / "model"
    model.mkdir()
    for name in ("config.json", "tokenizer.json", "model.safetensors"):
        (model / name).write_bytes(b"private-model-fixture")
    (model / "conversion.json").write_text(
        json.dumps({"source_manifest_sha256": "a" * 64})
    )
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    baseline.write_bytes(b"baseline")
    candidate.write_bytes(b"candidate")
    workers = []

    class FakeProbe:
        def __init__(self, path, _model, _timeout):
            self.candidate = path.name == "candidate"
            self.ready = {"vocab_size": 64}
            self.closed = False
            self.requests = []
            workers.append(self)

        def request(self, request):
            self.requests.append(list(request["input_token_ids"]))
            logits = [
                index / 32 + (0.125 if self.candidate else 0) for index in range(64)
            ]
            return {
                "logits": logits,
                "logits_sha256": existing.logits_hash(logits),
                "input_tokens_sha256": common.digest_ids(request["input_token_ids"]),
                "metrics": {
                    "state_creation_seconds": 0.1,
                    "prefill_seconds": 1.0,
                    "readout_seconds": 0.1,
                    "total_seconds": 1.2,
                },
            }

        def close(self):
            self.closed = True

    def fixtures(*_args):
        return [
            {
                "name": "long_tools_history",
                "ids": [2] * 64,
                "metadata": {
                    "input_tokens": 64,
                    "tools": 8,
                    "messages": 5,
                    "tool_results": 1,
                },
            }
        ]

    args = Namespace(
        model=model,
        baseline=baseline,
        candidate=candidate,
        target_tokens=64,
        context=256,
        steps=2,
        prefix_lengths=[31],
        check_repeats=True,
    )
    report = validation.compare(
        args, worker_factory=FakeProbe, fixture_factory=fixtures
    )
    assert report["passed"] is True
    assert (
        report["numeric_acceptance"] == "unqualified"
        and report["native_seeded_generation_compared_here"] is False
    )
    assert report["case_count"] == report["greedy_agreement_cases"] == 3
    assert report["exact_logit_cases"] == 0
    assert report["by_case"][0]["numerics"]["maximum_absolute_error"] == 0.125
    assert report["by_case"][0]["numerics"]["reference_to_candidate_kl"] < 1e-12
    assert all(worker.closed for worker in workers)
    # Reference continuation is shared, even if a future candidate's argmax differs.
    assert workers[0].requests == workers[1].requests
    assert workers[0].requests[2] == [2] * 64 + [63]
    serialized = json.dumps(report)
    for forbidden in (
        '"logits": [',
        '"input_token_ids":',
        "private-model-fixture",
        str(model),
    ):
        assert forbidden not in serialized


def test_report_writer_refuses_symlink(tmp_path):
    existing = tmp_path / "existing"
    existing.write_text("preserve")
    link = tmp_path / "link"
    link.symlink_to(existing)
    with pytest.raises(OSError):
        validation.json_file(link, {"passed": True})
    assert existing.read_text() == "preserve"
