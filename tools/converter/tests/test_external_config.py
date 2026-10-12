"""Declared external model configs stay distinct from checkpoint weight roots."""

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest
from test_cli import source_tooling  # noqa: F401
from test_llamba import llamba_fixture
from test_sources import PIN, Hub
from tokenizers import Tokenizer, models

from stark_convert import cli
from stark_convert.contracts import ConversionRequest
from stark_convert.sources import resolve_source
from stark_convert.streaming import ConversionError

DENSE = "cartesia-ai/Llamba-1B"
MLX = "cartesia-ai/Llamba-1B-4bit-mlx"
TOKENIZER = "meta-llama/Llama-3.2-1B-Instruct"


def source_repositories(tmp_path, *, repo=MLX, primary_config=False, real_tensors=False):
    primary = tmp_path / "primary"
    external = tmp_path / "external-config"
    tokenizer = tmp_path / "external-tokenizer"
    tokenizer.mkdir()
    if real_tensors:
        config, _ = llamba_fixture(primary, "mlx4_group16" if repo == MLX else "backbone_tied")
    else:
        primary.mkdir()
        (primary / "model.safetensors").write_bytes(b"synthetic checkpoint")
        config = {"model_type": "llamba", "synthetic": True}
    if repo == MLX:
        external.mkdir()
        config.pop("quantization", None)
        (external / "config.json").write_text(json.dumps(config))
        (primary / "config.json").unlink(missing_ok=True)
        if primary_config:
            (primary / "config.json").write_text('{"model_type":"unselected-primary-decoy"}')
        repositories = {MLX: primary, DENSE: external, TOKENIZER: tokenizer}
    else:
        (primary / "config.json").write_text(json.dumps(config))
        repositories = {DENSE: primary, TOKENIZER: tokenizer}
    if real_tensors:
        Tokenizer(models.WordLevel({"[SYNTHETIC]": 0}, unk_token="[SYNTHETIC]")).save(
            str(tokenizer / "tokenizer.json")
        )
    else:
        (tokenizer / "tokenizer.json").write_text("{}")
    (tokenizer / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages }}"}))
    (tokenizer / "generation_config.json").write_text(json.dumps({"eos_token_id": 128009, "temperature": 0.6}))
    return Hub(repositories), primary, repositories[DENSE], tokenizer


def rebind_record(record, path):
    """Make an adversarial descriptor internally consistent with its bytes."""
    stat = path.stat()
    record.update(
        path=str(path),
        size=stat.st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        signature=[stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns],
    )


def test_configless_remote_checkpoint_root_comes_from_primary_weights(tmp_path):
    hub, primary, external, _ = source_repositories(tmp_path)
    snapshot = resolve_source(MLX, hub=hub)
    assert snapshot.root == primary
    assert snapshot.checkpoint == primary
    assert snapshot.config_path == external / "config.json"
    assert snapshot.root != snapshot.config_path.parent
    assert not (snapshot.root / "config.json").exists()
    assert snapshot.resolve_file("model.safetensors") == primary / "model.safetensors"
    assert ("download", MLX, "config.json") not in hub.events
    snapshot.assert_unchanged()


def test_unselected_primary_config_cannot_override_declared_external_config(tmp_path):
    hub, primary, external, _ = source_repositories(tmp_path, primary_config=True)
    snapshot = resolve_source(MLX, hub=hub)
    assert snapshot.config_path == external / "config.json"
    assert json.loads(snapshot.config_path.read_text())["model_type"] == "llamba"
    assert not any(
        record["repo"] == MLX and record["filename"] == "config.json" for record in snapshot.manifest["files"]
    )
    assert ("download", MLX, "config.json") not in hub.events
    (primary / "config.json").write_text('{"changed":"still unselected"}')
    snapshot.assert_unchanged()


@pytest.mark.parametrize("repo", [DENSE, MLX])
def test_all_declared_config_and_tokenizer_repositories_are_pinned_before_download(tmp_path, repo):
    hub, _, _, _ = source_repositories(tmp_path, repo=repo)
    snapshot = resolve_source(repo, hub=hub)
    first = next(index for index, event in enumerate(hub.events) if event[0] == "download")
    expected_repos = {DENSE, TOKENIZER, *([MLX] if repo == MLX else [])}
    assert {event[1] for event in hub.events[:first] if event[0] == "pin"} == expected_repos
    assert {record["repo"] for record in snapshot.manifest["sources"]} == expected_repos
    assert len(snapshot.manifest["files"]) == 5
    for record in snapshot.manifest["files"]:
        path = Path(record["path"])
        stat = path.stat()
        assert record["revision"] == PIN
        assert record["size"] == stat.st_size
        assert record["signature"] == [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns]
        assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    config = next(record for record in snapshot.manifest["files"] if record["role"] == "model_config")
    assert (config["repo"], config["filename"]) == (DENSE, "config.json")
    assert snapshot.config_path == Path(config["path"])


@pytest.mark.parametrize("repo", [DENSE, MLX])
def test_external_sources_replay_schema_one_privately_without_network(tmp_path, repo):
    hub, primary, external, _ = source_repositories(tmp_path, repo=repo)
    snapshot = resolve_source(repo, hub=hub)
    manifest = tmp_path / "private-source.json"
    snapshot.write_manifest(manifest)
    assert manifest.stat().st_mode & 0o777 == 0o600
    assert snapshot.manifest["schema_version"] == 1
    assert "conversion_profile" not in snapshot.manifest
    replay = resolve_source(repo, source_manifest=manifest, offline=True, hub=object())
    assert replay.receipt() == snapshot.receipt()
    assert replay.root == primary
    assert replay.config_path == external / "config.json"
    assert str(tmp_path) not in json.dumps(replay.receipt())


@pytest.mark.parametrize("role", ["model_config", "tokenizer", "tokenizer_config", "generation_config", "weights"])
@pytest.mark.parametrize("restored_timestamp", [False, True])
def test_every_selected_external_file_is_bound_to_signatures_and_content(tmp_path, role, restored_timestamp):
    hub, _, _, _ = source_repositories(tmp_path)
    snapshot = resolve_source(MLX, hub=hub)
    manifest = tmp_path / "manifest.json"
    snapshot.write_manifest(manifest)
    record = next(record for record in snapshot.manifest["files"] if record["role"] == role)
    path = Path(record["path"])
    stat = path.stat()
    path.write_bytes(b"x" * stat.st_size)
    if restored_timestamp:
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        snapshot.assert_unchanged(verify_content=False)
    else:
        with pytest.raises(ConversionError):
            snapshot.assert_unchanged(verify_content=False)
    with pytest.raises(ConversionError):
        snapshot.assert_unchanged()
    with pytest.raises(ConversionError):
        resolve_source(MLX, source_manifest=manifest, offline=True, hub=object())


@pytest.mark.parametrize(
    "forgery",
    [
        "config_role",
        "config_primary_repo",
        "config_filename",
        "tokenizer_repo",
        "extra_primary_config",
        "weight_role",
        "weight_path",
        "duplicate_descriptor",
    ],
)
def test_rehashed_manifest_cannot_forge_declared_source_descriptors(tmp_path, forgery):
    hub, primary, external, tokenizer = source_repositories(tmp_path, primary_config=True)
    snapshot = resolve_source(MLX, hub=hub)
    manifest = copy.deepcopy(snapshot.manifest)
    config = next(record for record in manifest["files"] if record["role"] == "model_config")
    weight = next(record for record in manifest["files"] if record["role"] == "weights")
    if forgery == "config_role":
        config["role"] = "tokenizer_config"
        rebind_record(config, external / "config.json")
    elif forgery == "config_primary_repo":
        config["repo"] = MLX
        rebind_record(config, primary / "config.json")
    elif forgery == "config_filename":
        config["filename"] = "alternate.json"
        alternate = external / "alternate.json"
        alternate.write_bytes((external / "config.json").read_bytes())
        rebind_record(config, alternate)
    elif forgery == "tokenizer_repo":
        record = next(record for record in manifest["files"] if record["role"] == "tokenizer")
        record["repo"] = DENSE
        alternate = external / "tokenizer.json"
        alternate.write_bytes((tokenizer / "tokenizer.json").read_bytes())
        rebind_record(record, alternate)
    elif forgery == "extra_primary_config":
        record = {**config, "repo": MLX}
        rebind_record(record, primary / "config.json")
        manifest["files"].append(record)
    elif forgery == "weight_role":
        weight["role"] = "model_config"
        rebind_record(weight, primary / "model.safetensors")
    elif forgery == "weight_path":
        alternate = external / "model.safetensors"
        alternate.write_bytes((primary / "model.safetensors").read_bytes())
        rebind_record(weight, alternate)
    else:
        manifest["files"].append(copy.deepcopy(config))
    forged = tmp_path / "forged.json"
    forged.write_text(json.dumps(manifest))
    with pytest.raises(ConversionError):
        resolve_source(MLX, source_manifest=forged, offline=True, hub=object())


@pytest.mark.parametrize(
    "route",
    [
        "primary_config",
        "primary_subdirectory",
        "external_symlink_to_primary",
        "primary_symlink_to_external",
        "normalized_parent_alias_to_primary_symlink",
    ],
)
@pytest.mark.parametrize("replay", [False, True])
def test_rehashed_external_config_cannot_be_routed_into_primary_checkpoint(tmp_path, route, replay):
    hub, primary, external, _ = source_repositories(tmp_path, primary_config=True)
    snapshot = resolve_source(MLX, hub=hub)
    record = next(record for record in snapshot.manifest["files"] if record["role"] == "model_config")
    if route == "primary_config":
        alternate = primary / "config.json"
    elif route == "primary_subdirectory":
        alternate = primary / "another-config" / "config.json"
        alternate.parent.mkdir()
        alternate.write_bytes((external / "config.json").read_bytes())
    elif route == "external_symlink_to_primary":
        alternate = external / "primary-config-alias.json"
        alternate.symlink_to(primary / "config.json")
    else:
        alias = primary / "config.json"
        alias.unlink()
        alias.symlink_to(external / "config.json")
        alternate = alias if route == "primary_symlink_to_external" else external / ".." / primary.name / "config.json"
        # These paths resolve outside the primary tree. Their normalized
        # lexical location still places the selected config inside that tree.
        assert alternate.resolve() == external / "config.json"
    # Keep the external repository, immutable revision, filename and role.
    # Matching new hashes/signatures must not override the declared source route.
    rebind_record(record, alternate)
    if replay:
        manifest = tmp_path / "redirected-source.json"
        manifest.write_text(json.dumps(snapshot.manifest))
        with pytest.raises(ConversionError):
            resolve_source(MLX, source_manifest=manifest, offline=True, hub=object())
    else:
        with pytest.raises(ConversionError):
            snapshot.assert_unchanged()


@pytest.mark.parametrize("repo", [DENSE, MLX])
def test_complete_llamba_conversion_uses_the_declared_config_with_mocked_sources(tmp_path, repo):
    hub, primary, external, _ = source_repositories(tmp_path, repo=repo, primary_config=True, real_tensors=True)
    snapshot = resolve_source(repo, hub=hub)
    assert snapshot.config_path == external / "config.json"
    planned = cli.plan_conversion(snapshot, ConversionRequest(repo, "bfloat16"))
    assert planned.model_config["model_type"] == "llamba"
    assert planned.tensors.decoder_config["transformer_config"]["model_dim"] == 32
    if repo == MLX:
        assert "quantization" not in planned.model_config
        assert not any(
            record["repo"] == MLX and record["filename"] == "config.json" for record in snapshot.manifest["files"]
        )
        assert ("download", MLX, "config.json") not in hub.events
    output = tmp_path / "converted"
    receipt = cli.convert(snapshot, output, repo=repo, chunk_bytes=4096)
    assert receipt["capability"]["source_validation"] == "complete"
    assert receipt["capability"]["native_loading"] == "unverified"
    assert receipt["capability"]["execution"] == "unverified"
    assert receipt["source_manifest_sha256"] == snapshot.receipt()["source_manifest_sha256"]
    root_config = json.loads((output / "config.json").read_text())
    assert root_config["decoder_config"] == planned.tensors.decoder_config
    assert root_config["generation_config"]["stop_token_ids"] == [128001, 128008, 128009]
    assert not list(tmp_path.glob(".stark-convert-*"))
