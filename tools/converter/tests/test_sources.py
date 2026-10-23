"""Source pinning, exact file selection, and private offline replay."""

import hashlib
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from stark_convert import sources
from stark_convert.sources import resolve_source
from stark_convert.streaming import ConversionError, UnsupportedConversion

PRIMARY = "Qwen/Qwen3-0.6B"
PIN = "1" * 40


def files(root, *, weights="model.safetensors"):
    root.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        (root / name).write_text("{}")
    (root / weights).write_bytes(b"synthetic checkpoint")
    return root


class Hub:
    def __init__(self, repositories):
        self.repositories = repositories
        self.events = []

    def HfApi(self):
        return self

    def model_info(self, *, repo_id, revision=None):
        self.events.append(("pin", repo_id, revision))
        return SimpleNamespace(sha=PIN)

    def list_repo_files(self, *, repo_id, revision):
        self.events.append(("list", repo_id, revision))
        return [
            path.relative_to(self.repositories[repo_id]).as_posix()
            for path in self.repositories[repo_id].rglob("*")
            if path.is_file()
        ]

    def hf_hub_download(self, *, repo_id, filename, revision, local_files_only):
        assert revision == PIN and local_files_only is False
        self.events.append(("download", repo_id, filename))
        path = self.repositories[repo_id] / filename
        if not path.is_file():
            raise FileNotFoundError()
        return str(path)


def test_local_receipt_hashes_all_consumed_files_without_paths(tmp_path):
    root = files(tmp_path / "source")
    snapshot = resolve_source(root, repo=PRIMARY, revision=PIN)
    assert len(snapshot.manifest["files"]) == 5
    assert snapshot.resolve_file("config.json") == root / "config.json"
    snapshot.assert_unchanged()
    serialized = json.dumps(snapshot.receipt())
    assert str(root) not in serialized and "tokenizer.json" not in serialized and "signature" not in serialized
    assert all(len(record["sha256"]) == 64 for record in snapshot.receipt()["source_files"])


def test_manifest_replays_without_network_and_is_private(tmp_path):
    root = files(tmp_path / "source")
    snapshot = resolve_source(root, repo=PRIMARY, revision=PIN)
    manifest = tmp_path / "private-manifest.json"
    snapshot.write_manifest(manifest)
    assert manifest.stat().st_mode & 0o777 == 0o600
    replay = resolve_source(PRIMARY, source_manifest=manifest, offline=True, hub=object())
    assert replay.receipt() == snapshot.receipt()
    assert replay.root == root
    with pytest.raises(FileExistsError):
        snapshot.write_manifest(manifest)


def test_canonical_weights_skip_unrelated_containers_and_extra_tensors(tmp_path):
    root = files(tmp_path / "source")
    (root / "unrelated.safetensors").write_bytes(b"not used")
    (root / "pytorch_model.bin").write_bytes(b"not used")
    snapshot = resolve_source(root, repo=PRIMARY)
    assert [record["filename"] for record in snapshot.manifest["files"] if record["role"] == "weights"] == [
        "model.safetensors"
    ]


def test_shard_index_selects_only_its_weight_files(tmp_path):
    root = files(tmp_path / "source", weights="part-1.safetensors")
    (root / "part-2.safetensors").write_bytes(b"part 2")
    (root / "unrelated.safetensors").write_bytes(b"not used")
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"one": "part-1.safetensors", "two": "part-2.safetensors"}})
    )
    snapshot = resolve_source(root, repo=PRIMARY)
    assert {record["filename"] for record in snapshot.manifest["files"] if record["role"] == "weights"} == {
        "part-1.safetensors",
        "part-2.safetensors",
    }
    assert any(record["role"] == "weight_index" for record in snapshot.manifest["files"])


@pytest.mark.parametrize("filename", ["../escape.safetensors", "/escape.safetensors", "missing.safetensors"])
def test_invalid_index_is_rejected_before_conversion(tmp_path, filename):
    root = files(tmp_path / "source")
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"one": filename}}))
    with pytest.raises(ConversionError):
        resolve_source(root, repo=PRIMARY)


def test_online_pins_every_repository_before_any_download(tmp_path):
    primary = "mistralai/Devstral-Small-2505"
    external = "mistralai/Mistral-Small-3.1-24B-Base-2503"
    chat_repo = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    primary_root = files(tmp_path / "primary")
    (primary_root / "SYSTEM_PROMPT.txt").write_text("synthetic fixture")
    external_root = files(tmp_path / "external")
    chat_root = files(tmp_path / "chat")
    (chat_root / "chat_template.json").write_text("{}")
    hub = Hub({primary: primary_root, external: external_root, chat_repo: chat_root})
    snapshot = resolve_source(primary, hub=hub)
    first_download = next(index for index, event in enumerate(hub.events) if event[0] == "download")
    assert {event[1] for event in hub.events[:first_download] if event[0] == "pin"} == {primary, external, chat_repo}
    assert snapshot.resolve_file("tokenizer.json", repo=external) == external_root / "tokenizer.json"
    manifest = tmp_path / "manifest.json"
    snapshot.write_manifest(manifest)
    assert resolve_source(primary, source_manifest=manifest, hub=object()).receipt() == snapshot.receipt()
    # Every cross-repository metadata rule remains mandatory during replay.
    replay_data = json.loads(manifest.read_text())
    replay_data["files"] = [
        record
        for record in replay_data["files"]
        if not (record["repo"] == external and record["filename"] == "tokenizer.json")
    ]
    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text(json.dumps(replay_data))
    with pytest.raises(ConversionError):
        resolve_source(primary, source_manifest=incomplete, hub=object())
    (external_root / "tokenizer_config.json").write_text('{"modified":true}')
    with pytest.raises(ConversionError, match="Source changed during conversion"):
        snapshot.assert_unchanged(verify_content=False)


@pytest.mark.parametrize(
    "value",
    [PRIMARY, f"https://huggingface.co/{PRIMARY}", f" https://huggingface.co/{PRIMARY}/?download=true#model-card "],
)
def test_repository_names_and_urls_resolve_to_the_same_pinned_source(tmp_path, value):
    hub = Hub({PRIMARY: files(tmp_path / "source")})
    snapshot = resolve_source(value, repo=PRIMARY, hub=hub)
    assert snapshot.repo == PRIMARY
    assert snapshot.revision == PIN
    assert all(event[1] == PRIMARY for event in hub.events)
    assert snapshot.receipt() == resolve_source(PRIMARY, repo=f"https://huggingface.co/{PRIMARY}", hub=hub).receipt()


@pytest.mark.parametrize(
    "value",
    [
        "owner/model/extra",
        "https://example.com/Qwen/Qwen3-0.6B",
        "https://huggingface.co.example.com/Qwen/Qwen3-0.6B",
        "https://huggingface.co@evil.example/Qwen/Qwen3-0.6B",
        "https://user:password@huggingface.co/Qwen/Qwen3-0.6B",
        "http://huggingface.co/Qwen/Qwen3-0.6B",
        "https://huggingface.co/datasets/owner/model",
        "https://huggingface.co/spaces/owner/model",
        "https://huggingface.co/Qwen/Qwen3-0.6B/tree/main",
        "https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json",
        "https://huggingface.co/Qwen%2FQwen3-0.6B",
        "https://huggingface.co/Qwen/\nQwen3-0.6B",
    ],
)
def test_invalid_repository_urls_are_rejected_before_hub_access(value):
    with pytest.raises(ValueError, match="^invalid_huggingface_repository$"):
        resolve_source(value, hub=object())


def test_repository_url_cannot_override_an_explicit_repository():
    with pytest.raises(ValueError, match="^conflicting_repository$"):
        resolve_source(f"https://huggingface.co/{PRIMARY}", repo="example/other-model", hub=object())


def test_offline_external_sources_require_complete_manifest(tmp_path):
    root = files(tmp_path / "source")
    with pytest.raises(ConversionError):
        resolve_source(root, repo="mistralai/Devstral-Small-2505", offline=True, hub=object())
    with pytest.raises(ConversionError):
        resolve_source(PRIMARY, offline=True, hub=object())


def test_replay_detects_mutation_even_if_timestamp_is_restored(tmp_path):
    import os

    root = files(tmp_path / "source")
    snapshot = resolve_source(root, repo=PRIMARY)
    manifest = tmp_path / "manifest.json"
    snapshot.write_manifest(manifest)
    path = root / "model.safetensors"
    stat = path.stat()
    path.write_bytes(b"x" * stat.st_size)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    snapshot.assert_unchanged(verify_content=False)
    with pytest.raises(ConversionError):
        resolve_source(PRIMARY, source_manifest=manifest, hub=object())
    with pytest.raises(ConversionError):
        snapshot.assert_unchanged()


def test_signature_only_check_skips_payload_hashing_but_final_default_hashes_every_file(tmp_path, monkeypatch):
    root = files(tmp_path / "source")
    snapshot = resolve_source(root, repo=PRIMARY)
    hashes = []
    original_hash = sources._hash

    def tracked_hash(path):
        hashes.append(path)
        return original_hash(path)

    monkeypatch.setattr(sources, "_hash", tracked_hash)
    snapshot.assert_unchanged(verify_content=False)
    assert hashes == []
    snapshot.assert_unchanged()
    assert hashes == [Path(record["path"]) for record in snapshot.manifest["files"]]


@pytest.mark.parametrize(
    "filename",
    ["config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json", "model.safetensors"],
)
def test_signature_only_check_rejects_mutation_of_every_consumed_file(tmp_path, filename, monkeypatch):
    root = files(tmp_path / "source")
    snapshot = resolve_source(root, repo=PRIMARY)
    (root / filename).write_bytes(b"changed source bytes")
    monkeypatch.setattr(sources, "_hash", lambda path: pytest.fail("signature-only check read payload content"))
    with pytest.raises(ConversionError, match="Source changed during conversion"):
        snapshot.assert_unchanged(verify_content=False)


def test_signature_only_check_validates_signature_again_after_initial_check(tmp_path, monkeypatch):
    root = files(tmp_path / "source")
    snapshot = resolve_source(root, repo=PRIMARY)
    original_signature = sources._signature
    tracked_path = root / "tokenizer.json"
    calls = []

    def changed_signature(path):
        signature = original_signature(path)
        if path == tracked_path:
            calls.append(path)
            if len(calls) == 2:
                return (*signature[:3], signature[3] + 1)
        return signature

    monkeypatch.setattr(sources, "_signature", changed_signature)
    with pytest.raises(ConversionError, match="Source changed during conversion"):
        snapshot.assert_unchanged(verify_content=False)
    assert calls == [tracked_path, tracked_path]


@pytest.mark.parametrize("verify_content", [None, 0, 1, "false"])
def test_content_verification_mode_requires_boolean(tmp_path, verify_content):
    snapshot = resolve_source(files(tmp_path / "source"), repo=PRIMARY)
    with pytest.raises(ValueError, match="invalid_content_verification_mode"):
        snapshot.assert_unchanged(verify_content=verify_content)


@pytest.mark.parametrize("mutation", ["missing", "revision", "wrong_path", "wrong_size"])
def test_manifest_rejects_missing_dependencies_and_forged_metadata(tmp_path, mutation):
    root = files(tmp_path / "source")
    snapshot = resolve_source(root, repo=PRIMARY)
    manifest = snapshot.manifest
    record = next(record for record in manifest["files"] if record["filename"] == "config.json")
    if mutation == "missing":
        manifest["files"].remove(record)
    elif mutation == "revision":
        record["revision"] = "2" * 40
    elif mutation == "wrong_path":
        alternate = tmp_path / "other.json"
        alternate.write_text("{}")
        record["path"] = str(alternate)
    else:
        record["size"] += 1
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ConversionError):
        resolve_source(PRIMARY, source_manifest=path, hub=object())


def test_unknown_repository_is_not_downloaded():
    with pytest.raises(UnsupportedConversion):
        resolve_source("unregistered/model", hub=object())


def test_local_revision_must_be_immutable(tmp_path):
    root = files(tmp_path / "source")
    with pytest.raises(ConversionError):
        resolve_source(root, repo=PRIMARY, revision="main")


@pytest.mark.parametrize("verify_content", [False, True])
def test_new_index_cannot_redirect_planner_to_unpinned_weights(tmp_path, verify_content):
    root = files(tmp_path / "source")
    snapshot = resolve_source(root, repo=PRIMARY)
    (root / "untracked.safetensors").write_bytes(b"untracked")
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"one": "untracked.safetensors"}}))
    with pytest.raises(ConversionError):
        snapshot.assert_unchanged(verify_content=verify_content)


def test_forged_config_role_cannot_bypass_root_binding(tmp_path):
    root = files(tmp_path / "source")
    snapshot = resolve_source(root, repo=PRIMARY)
    manifest = snapshot.manifest
    record = next(record for record in manifest["files"] if record["filename"] == "config.json")
    record["role"] = "other"
    alternate = tmp_path / "alternate.json"
    alternate.write_text("{}")
    record["path"] = str(alternate)
    alternate_stat = alternate.stat()
    record["signature"] = [
        alternate_stat.st_dev,
        alternate_stat.st_ino,
        alternate_stat.st_size,
        alternate_stat.st_mtime_ns,
    ]
    record["size"] = alternate_stat.st_size
    record["sha256"] = hashlib.sha256(alternate.read_bytes()).hexdigest()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ConversionError, match="Catalog source descriptors disagree with the declared profile"):
        resolve_source(PRIMARY, source_manifest=path, hub=object())


@pytest.mark.parametrize("size", [0, 1, 256 * 1024 - 1, 256 * 1024, 256 * 1024 + 1, 2 * 256 * 1024 + 17])
def test_reusable_hash_buffer_preserves_exact_digest_across_read_boundaries(tmp_path, size):
    payload = (b"independent hash fixture\x00\xff" * (size // 26 + 1))[:size]
    path = tmp_path / "payload.bin"
    path.write_bytes(payload)
    assert sources._hash(path) == hashlib.sha256(payload).hexdigest()


def test_parallel_hashes_are_exact_ordered_and_bounded_to_four_readers(tmp_path, monkeypatch):
    paths = [tmp_path / f"shard-{index}.bin" for index in range(7)]
    metadata = tmp_path / "metadata.json"
    for index, path in enumerate(paths):
        path.write_bytes(bytes([index]) * 32)
    metadata.write_bytes(b"{}")
    paths.insert(2, metadata)
    monkeypatch.setattr(sources, "_PARALLEL_HASH_MIN_BYTES", 16)
    original_hash = sources._hash
    lock = threading.Lock()
    first_wave = threading.Event()
    active = 0
    peak = 0
    seen = []

    def tracked_hash(path):
        nonlocal active, peak
        if path == metadata:
            assert active == 0
            assert threading.current_thread() is threading.main_thread()
            seen.append(path)
            return original_hash(path)
        with lock:
            active += 1
            peak = max(peak, active)
            seen.append(path)
            if active == 4:
                first_wave.set()
        try:
            assert first_wave.wait(5), "large hashes did not run concurrently"
            return original_hash(path)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(sources, "_hash", tracked_hash)
    _, actual = sources._verified_hashes(paths)
    assert actual == [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    assert 2 <= peak <= 4
    assert active == 0 and len(seen) == len(paths) and set(seen) == set(paths)


@pytest.mark.parametrize("failure", [OSError("private checkpoint path and exception details"), KeyboardInterrupt()])
def test_parallel_hash_failure_joins_all_started_readers(tmp_path, monkeypatch, failure):
    paths = [tmp_path / f"shard-{index}.bin" for index in range(4)]
    for path in paths:
        path.write_bytes(b"payload")
    monkeypatch.setattr(sources, "_PARALLEL_HASH_MIN_BYTES", 1)
    barrier = threading.Barrier(4)
    finished = set()
    lock = threading.Lock()

    def failing_hash(path):
        barrier.wait(timeout=5)
        try:
            if path == paths[0]:
                raise failure
            return hashlib.sha256(path.read_bytes()).hexdigest()
        finally:
            with lock:
                finished.add(path)

    monkeypatch.setattr(sources, "_hash", failing_hash)
    expected = [sources._signature(path) for path in paths]
    if isinstance(failure, OSError):
        with pytest.raises(ConversionError, match="^Source unavailable during conversion$") as caught:
            sources._verified_hashes(paths, expected_signatures=expected)
        assert "private" not in str(caught.value)
    else:
        with pytest.raises(KeyboardInterrupt):
            sources._verified_hashes(paths, expected_signatures=expected)
    assert finished == set(paths)
    assert not any(thread.name.startswith("stark-source-hash") for thread in threading.enumerate())


def test_completed_earlier_file_is_rechecked_after_later_hashes(tmp_path, monkeypatch):
    paths = [tmp_path / "first.bin", tmp_path / "second.bin"]
    for path in paths:
        path.write_bytes(b"payload")
    original_hash = sources._hash
    first_stat = paths[0].stat()

    def mutating_hash(path):
        digest = original_hash(path)
        if path == paths[-1]:
            paths[0].write_bytes(b"changed")
            os.utime(paths[0], ns=(first_stat.st_atime_ns, first_stat.st_mtime_ns + 1))
        return digest

    monkeypatch.setattr(sources, "_hash", mutating_hash)
    with pytest.raises(ConversionError, match="Source changed while hashing"):
        sources._verified_hashes(paths)


def test_checkpoint_selection_is_rechecked_after_full_hash_batch(tmp_path, monkeypatch):
    root = files(tmp_path / "source")
    snapshot = resolve_source(root, repo=PRIMARY)
    original_hash = sources._hash

    def mutating_hash(path):
        digest = original_hash(path)
        if path.name == "model.safetensors":
            (root / "untracked.safetensors").write_bytes(b"untracked")
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"one": "untracked.safetensors"}})
            )
        return digest

    monkeypatch.setattr(sources, "_hash", mutating_hash)
    with pytest.raises(ConversionError, match="Source checkpoint selection changed during conversion"):
        snapshot.assert_unchanged()


def test_changed_signature_prevents_entire_hash_batch(tmp_path, monkeypatch):
    root = files(tmp_path / "source")
    snapshot = resolve_source(root, repo=PRIMARY)
    (root / "tokenizer_config.json").write_bytes(b"changed")
    monkeypatch.setattr(sources, "_hash", lambda path: pytest.fail("changed source started a hash batch"))
    with pytest.raises(ConversionError, match="Source changed during conversion"):
        snapshot.assert_unchanged()
