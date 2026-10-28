"""Private byte progress, cached/resumed downloads and publication boundaries."""

import hashlib
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from huggingface_hub.utils._xet_progress_reporting import XetDownloadProgressReporter
from test_cli import REPOS, run, source_fixture, source_tooling  # noqa: F401
from test_sources import PIN, PRIMARY, Hub, files
from test_streaming import save_tensors

from stark_convert import cli, sources
from stark_convert.progress import DownloadProgress, FDProgressSink, ProgressReporter
from stark_convert.sources import resolve_source
from stark_convert.streaming import (
    ConversionError,
    ConversionPlan,
    TensorOperation,
    read_safetensors_header,
    write_plan,
)


def assert_private(events):
    for event in events:
        assert set(event) == {"phase", "state", "completed_bytes", "total_bytes"}
        assert event["phase"] in {"download", "conversion"}
        assert event["state"] in {"waiting", "running", "complete", "skipped"}
        assert type(event["completed_bytes"]) is int and event["completed_bytes"] >= 0
        total = event["total_bytes"]
        assert total is None or type(total) is int and event["completed_bytes"] <= total


def test_reporter_throttles_counts_and_always_emits_state_and_final_counts():
    events, now = [], [0.0]
    progress = ProgressReporter(events.append, clock=lambda: now[0])
    progress.emit("download", "waiting")
    progress.emit("download", "running", 0, 100)
    for count in range(1, 20):
        now[0] += 0.004
        progress.emit("download", "running", count, 100)
    assert len(events) == 2
    now[0] = 0.11
    progress.emit("download", "running", 20, 100)
    progress.emit("download", "running", 100, 100, force=True)
    progress.emit("download", "complete", 100, 100)
    assert [event["completed_bytes"] for event in events] == [0, 0, 20, 100, 100]
    assert_private(events)
    with pytest.raises(ValueError, match="invalid_progress_state"):
        progress.emit("private-model-name", "running")
    for completed, total in ((True, 5), (1, False), (-1, 1), (2, 1)):
        with pytest.raises(ValueError, match="invalid_progress_counts"):
            progress.emit("download", "running", completed, total)


def test_fd_sink_is_private_owns_duplicate_and_tolerates_closed_reader(capsys):
    read_fd, write_fd = os.pipe()
    sink = FDProgressSink(write_fd)
    os.close(write_fd)
    progress = ProgressReporter(sink)
    progress.emit("conversion", "running", 5, 9)
    progress.emit("conversion", "complete", 9, 9)
    sink.close()
    try:
        events = [json.loads(line) for line in os.read(read_fd, 4096).splitlines()]
    finally:
        os.close(read_fd)
    assert_private(events)
    assert events[-1]["state"] == "complete"
    assert capsys.readouterr() == ("", "")
    with pytest.raises(ValueError, match="invalid_progress_descriptor"):
        FDProgressSink(1)
    read_fd, write_fd = os.pipe()
    sink = FDProgressSink(write_fd)
    os.close(read_fd)
    os.close(write_fd)
    sink(events[-1])
    assert sink.descriptor is None


class ProgressHub(Hub):
    def __init__(self, repositories, *, cached=(), resume=()):
        super().__init__(repositories)
        self.cached = set(cached)
        self.resume = set(resume)
        self.network = []
        self.before_weight = None

    def hf_hub_download(self, *, repo_id, filename, revision, local_files_only, dry_run=False, tqdm_class=None):
        assert revision == PIN and local_files_only is False
        path = self.repositories[repo_id] / filename
        if not path.is_file():
            raise FileNotFoundError()
        if dry_run:
            self.events.append(("size", repo_id, filename))
            return SimpleNamespace(file_size=path.stat().st_size, is_cached=filename in self.cached)
        self.events.append(("download", repo_id, filename))
        if filename in self.cached:
            return str(path)
        if self.before_weight is not None and filename.endswith(".safetensors"):
            self.before_weight()
        self.network.append(filename)
        size = path.stat().st_size
        with tqdm_class(total=size, initial=size // 2 if filename in self.resume else 0, disable=True) as bar:
            if filename in self.resume:
                # A server ignoring Range restarts the transfer. Already
                # satisfied bytes are not counted for a second time.
                bar.update(-size // 2)
            bar.update(size // 2)
            bar.update(size - size // 2)
            bar.update_transfer(size * 3)
        return str(path)


def test_selected_file_totals_include_cache_resume_and_exclude_unselected_files(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    root = files(tmp_path / "private-model")
    (root / "unrelated.safetensors").write_bytes(b"excluded" * 100)
    events = []
    progress = ProgressReporter(events.append, interval=0)
    hub = ProgressHub({PRIMARY: root}, cached={"tokenizer.json"}, resume={"model.safetensors"})

    def before_weight():
        known = [event for event in events if event["total_bytes"] is not None]
        assert known and known[-1]["total_bytes"] == sum(
            path.stat().st_size for path in root.iterdir() if path.name != "unrelated.safetensors"
        )

    hub.before_weight = before_weight
    snapshot = resolve_source(PRIMARY, hub=hub, progress=progress)
    total = sum(record["size"] for record in snapshot.manifest["files"])
    assert events[-1] == {"phase": "download", "state": "complete", "completed_bytes": total, "total_bytes": total}
    assert "tokenizer.json" not in hub.network and "unrelated.safetensors" not in hub.network
    assert [event["completed_bytes"] for event in events] == sorted(event["completed_bytes"] for event in events)
    assert all(event["total_bytes"] in {None, total} for event in events)
    assert_private(events)
    assert str(root) not in json.dumps(events) and PRIMARY not in json.dumps(events)
    assert capsys.readouterr() == ("", "")


def test_xet_counts_reconstruction_once_with_progress_globally_disabled(monkeypatch, capsys):
    monkeypatch.setenv("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    events = []
    download = DownloadProgress(ProgressReporter(events.append, interval=0))
    key = ("private-repository", PIN, "private-file")
    download.sizes[key] = 100
    download.selected = {key}
    with XetDownloadProgressReporter(
        reconstruction_desc="private-file", total=100, log_level=0, tqdm_class=download.bar_class(key)
    ) as xet:
        assert xet.reconstruction_bar is xet.transfer_bar
        for reconstructed, transferred in ((20, 60), (40, 120), (100, 180)):
            xet.update_progress(
                SimpleNamespace(
                    total_bytes_completed=reconstructed,
                    total_transfer_bytes_completed=transferred,
                    total_bytes_completion_rate=1.0,
                    total_transfer_bytes_completion_rate=2.0,
                    total_bytes=100,
                )
            )
    assert [event["completed_bytes"] for event in events] == [0, 0, 20, 40, 100]
    assert events[-1]["state"] == "running"
    download.complete()
    assert events[-1]["state"] == "complete"
    assert_private(events)
    assert capsys.readouterr() == ("", "")


def test_sharded_download_total_is_known_before_weight_payloads(tmp_path):
    root = files(tmp_path / "source", weights="part-1.safetensors")
    (root / "part-2.safetensors").write_bytes(b"second shard")
    (root / "unrelated.safetensors").write_bytes(b"excluded")
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"first": "part-1.safetensors", "second": "part-2.safetensors"}})
    )
    events = []
    expected = sum(path.stat().st_size for path in root.iterdir() if path.name != "unrelated.safetensors")
    hub = ProgressHub({PRIMARY: root})

    def before_weight():
        assert events[-1]["total_bytes"] == expected

    hub.before_weight = before_weight
    snapshot = resolve_source(PRIMARY, hub=hub, progress=ProgressReporter(events.append, interval=0))
    assert sum(record["size"] for record in snapshot.manifest["files"]) == expected
    assert events[-1]["completed_bytes"] == expected
    assert "unrelated.safetensors" not in hub.network


def test_configless_checkpoint_sizes_external_metadata_before_weight_payload(tmp_path):
    primary = "cartesia-ai/Llamba-1B-4bit-mlx"
    external = "cartesia-ai/Llamba-1B"
    tokenizer = "meta-llama/Llama-3.2-1B-Instruct"
    roots = {
        primary: files(tmp_path / "primary"),
        external: files(tmp_path / "external", model_type="llamba"),
        tokenizer: files(tmp_path / "tokenizer"),
    }
    (roots[primary] / "config.json").unlink()
    rules = sources.required_files(sources.get_model(primary))
    expected = (roots[primary] / "model.safetensors").stat().st_size + sum(
        (roots[rule["repo"]] / rule["filename"]).stat().st_size for rule in rules
    )
    events = []
    hub = ProgressHub(roots)

    def before_weight():
        assert events[-1]["total_bytes"] == expected

    hub.before_weight = before_weight
    snapshot = resolve_source(primary, hub=hub, progress=ProgressReporter(events.append, interval=0))
    assert snapshot.root == roots[primary]
    assert events[-1]["completed_bytes"] == events[-1]["total_bytes"] == expected


def test_download_completion_follows_source_validation(tmp_path, monkeypatch):
    root = files(tmp_path / "source")
    events = []

    def fail_validation(_paths):
        assert not any(event["state"] == "complete" for event in events)
        raise ConversionError("synthetic validation failure")

    monkeypatch.setattr(sources, "_verified_hashes", fail_validation)
    with pytest.raises(ConversionError):
        resolve_source(PRIMARY, hub=ProgressHub({PRIMARY: root}), progress=events.append)
    assert not any(event["state"] == "complete" for event in events)


def test_local_and_offline_sources_skip_download(tmp_path):
    root = files(tmp_path / "source")
    events = []
    snapshot = resolve_source(root, repo=PRIMARY, revision=PIN, progress=events.append)
    manifest = tmp_path / "source-manifest.json"
    snapshot.write_manifest(manifest)
    replay = resolve_source(PRIMARY, source_manifest=manifest, offline=True, hub=object(), progress=events.append)
    assert replay.receipt() == snapshot.receipt()
    assert events == [{"phase": "download", "state": "skipped", "completed_bytes": 0, "total_bytes": None}] * 2


@pytest.mark.parametrize("kind", ["transpose", "transpose_f32"])
def test_conversion_counts_actual_random_writes_header_and_unchanged_hash(tmp_path, kind):
    source = tmp_path / "source.safetensors"
    data = np.arange(35, dtype="<f2").reshape(5, 7)
    save_tensors(source, {"input": ("F16", data)})
    _, tensors = read_safetensors_header(source)
    operation = TensorOperation(
        "output", (tensors["input"],), "F32" if kind == "transpose_f32" else "F16", (7, 5), kind
    )
    plan = ConversionPlan({}, (operation,), {}, tensors["input"].nbytes, 1)
    events = []
    output = tmp_path / "progress.safetensors"
    result = write_plan(plan, output, chunk_bytes=32, progress=ProgressReporter(events.append, interval=0))
    normal = tmp_path / "default.safetensors"
    default = write_plan(plan, normal, chunk_bytes=32)
    size = output.stat().st_size
    assert output.read_bytes() == normal.read_bytes()
    assert result["output_sha256"] == default["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert events[0]["completed_bytes"] == 0
    assert events[-1] == {"phase": "conversion", "state": "running", "completed_bytes": size, "total_bytes": size}
    assert [event["completed_bytes"] for event in events] == sorted(event["completed_bytes"] for event in events)
    assert (
        sum(
            after["completed_bytes"] - before["completed_bytes"]
            for before, after in zip(events, events[1:], strict=False)
        )
        == size
    )
    # Seeking across a transpose must not advance the byte counter over output
    # that has not been written. Payload updates obey the transform workspace.
    assert all(
        after["completed_bytes"] - before["completed_bytes"] <= 32
        for before, after in zip(events[2:], events[3:], strict=False)
    )
    assert all(event["total_bytes"] == size and event["state"] == "running" for event in events)
    assert_private(events)


def test_conversion_completes_only_after_receipt_validation_and_atomic_publication(tmp_path, monkeypatch):
    source = source_fixture(tmp_path / "source")
    output = tmp_path / "converted"
    events = []
    original_publish = cli._publish

    def publish(staging, destination):
        assert not any(event["state"] == "complete" for event in events)
        assert (staging / "conversion.json").is_file()
        original_publish(staging, destination)

    def receive(event):
        if event["phase"] == "conversion" and event["state"] == "complete":
            assert (output / "conversion.json").is_file()
        events.append(event)

    monkeypatch.setattr(cli, "_publish", publish)
    result = run(source, output, progress=receive)
    monkeypatch.setattr(cli, "_publish", original_publish)
    normal = run(source, tmp_path / "normal")
    assert result.keys() == normal.keys()
    for key in ("output_sha256", "config_sha256", "tokenizer_sha256", "source_manifest_sha256", "capability"):
        assert result[key] == normal[key]
    assert events[-1]["state"] == "complete"
    assert events[-1]["completed_bytes"] == events[-1]["total_bytes"] == result["output_bytes"]
    assert_private(events)


def test_failed_publication_never_reports_conversion_complete(tmp_path, monkeypatch):
    source = source_fixture(tmp_path / "source")
    events = []

    def fail_publication(_staging, _destination):
        raise ConversionError("synthetic publication failure")

    monkeypatch.setattr(cli, "_publish", fail_publication)
    with pytest.raises(ConversionError):
        run(source, tmp_path / "converted", progress=events.append)
    assert not any(event["state"] == "complete" for event in events)
    assert not list(tmp_path.glob(".stark-convert-*"))


def test_cli_hidden_private_fd_keeps_stdout_as_single_receipt(tmp_path):
    source = source_fixture(tmp_path / "source")
    output = tmp_path / "converted"
    read_fd, write_fd = os.pipe()
    command = [
        sys.executable,
        "-m",
        "stark_convert.cli",
        str(source),
        "--repo",
        REPOS["qwen3"],
        "--offline",
        "--output",
        str(output),
        "--progress-fd",
        str(write_fd),
    ]
    try:
        process = subprocess.Popen(command, pass_fds=(write_fd,), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        os.close(write_fd)
        write_fd = None
        stdout, stderr = process.communicate(timeout=30)
        progress = os.read(read_fd, 32768)
    finally:
        os.close(read_fd)
        if write_fd is not None:
            os.close(write_fd)
    assert process.returncode == 0 and stderr == b""
    receipt = json.loads(stdout)
    assert receipt["status"] == "ok"
    events = [json.loads(line) for line in progress.splitlines()]
    assert [event["state"] for event in events[:2]] == ["waiting", "waiting"]
    assert events[-1]["state"] == "complete"
    assert_private(events)
    assert str(source) not in progress.decode() and PRIMARY not in progress.decode()
    help_result = subprocess.run(command[:3] + ["--help"], capture_output=True, check=True)
    assert b"--progress-fd" not in help_result.stdout
