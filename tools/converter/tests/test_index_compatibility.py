"""Pinned stale indexes keep strict source selection and complete header accounting."""

import copy
import hashlib
import json
import os
import sys

import numpy as np
import pytest
from test_cli import source_tooling  # noqa: F401 - isolated receipt provenance fixture
from test_gemma_multimodal import wrapper_fixture
from test_sources import PIN, PRIMARY, Hub
from test_streaming import save_tensors
from tokenizers import Tokenizer, models

from stark_convert import cli, registry, sources
from stark_convert.sources import _weights, resolve_source
from stark_convert.streaming import ConversionError, checkpoint_tensor_names_sha256, load_checkpoint


def fixture(root, *, shards=1, indexed_available=False):
    root.mkdir()
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        (root / name).write_text(json.dumps({"model_type": "qwen3"}) if name == "config.json" else "{}")
    index = {"weight_map": {"first.weight": "old-1.safetensors", "second": "old-2.safetensors"}}
    index_path = root / "model.safetensors.index.json"
    index_path.write_text(json.dumps(index))
    tensors = {name: ("F32", np.arange(2, dtype="<f4")) for name in ("first.weight", "first.scales", "second")}
    filenames = ["model.safetensors"] if shards == 1 else ["part-1.safetensors", "part-2.safetensors"]
    for shard, filename in enumerate(filenames):
        save_tensors(root / filename, dict(list(tensors.items())[shard::shards]))
    if indexed_available:
        index_path.write_text(json.dumps({"weight_map": {"first.weight": filenames[-1], "second": filenames[0]}}))
    policy = {
        "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "weight_filenames": filenames,
        "tensor_names_sha256": checkpoint_tensor_names_sha256(tensors),
    }
    return tensors, policy


def names(root):
    return tuple(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())


def patch_catalog(monkeypatch, policy):
    entry = copy.deepcopy(registry.get_model(PRIMARY))
    entry["weight_index_compatibility"] = policy
    original = sources.get_model
    monkeypatch.setattr(sources, "get_model", lambda repo: entry if repo == PRIMARY else original(repo))


@pytest.mark.parametrize("shards", [1, 2])
@pytest.mark.parametrize("remote", [False, True])
def test_pinned_selection_hashes_original_index_and_replays_all_actual_weights(tmp_path, monkeypatch, shards, remote):
    root = tmp_path / "source"
    tensors, policy = fixture(root, shards=shards)
    patch_catalog(monkeypatch, policy)
    hub = Hub({PRIMARY: root})
    snapshot = resolve_source(PRIMARY, hub=hub) if remote else resolve_source(root, repo=PRIMARY, revision=PIN)
    records = snapshot.manifest["files"]
    assert (
        sorted(record["filename"] for record in records if record["role"] == "weights") == policy["weight_filenames"]
    )
    index = next(record for record in records if record["role"] == "weight_index")
    assert index["filename"] == "model.safetensors.index.json"
    assert index["sha256"] == policy["index_sha256"]
    assert set(load_checkpoint(root, index_compatibility=policy)) == set(tensors)
    snapshot.assert_unchanged()
    manifest = tmp_path / "source-manifest.json"
    snapshot.write_manifest(manifest)
    replay = resolve_source(PRIMARY, source_manifest=manifest, offline=True, hub=object())
    assert replay.receipt() == snapshot.receipt()
    assert set(load_checkpoint(replay.root, index_compatibility=policy)) == set(tensors)


def test_unregistered_and_default_source_selection_remain_strict(tmp_path):
    root = tmp_path / "source"
    fixture(root)
    with pytest.raises(ConversionError, match="unavailable shard"):
        resolve_source(root, repo=PRIMARY)
    with pytest.raises(ConversionError, match="unavailable shard"):
        resolve_source(root, metadata_profile="hf-generic", offline=True)
    with pytest.raises(ConversionError, match="unavailable shard"):
        load_checkpoint(root)


@pytest.mark.parametrize(
    "filename",
    [
        "../escape.safetensors",
        "/escape.safetensors",
        "sub/../escape.safetensors",
        "sub//escape.safetensors",
        "./escape.safetensors",
        "sub\\escape.safetensors",
        "model.bin",
        "",
    ],
)
def test_invalid_index_paths_fail_before_compatibility_fallback(tmp_path, filename):
    root = tmp_path / "source"
    _, policy = fixture(root)
    index = root / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"first.weight": filename}}))
    policy["index_sha256"] = hashlib.sha256(index.read_bytes()).hexdigest()
    with pytest.raises(ConversionError):
        _weights(names(root), ".safetensors", lambda name: root / name, compatibility=policy)
    with pytest.raises(ConversionError):
        load_checkpoint(root, index_compatibility=policy)


@pytest.mark.parametrize(
    "raw",
    [
        '{"weight_map":{"a":"old.safetensors","a":"other.safetensors"}}',
        '{"weight_map":{}}',
        '{"weight_map":{"":"old.safetensors"}}',
        '{"weight_map":{"a":false}}',
        '{"weight_map":{"a":"old.safetensors"},"bad":NaN}',
        "[]",
        "{",
    ],
)
def test_malformed_index_cannot_be_excused_by_an_exact_hash(tmp_path, raw):
    root = tmp_path / "source"
    _, policy = fixture(root)
    index = root / "model.safetensors.index.json"
    index.write_text(raw)
    policy["index_sha256"] = hashlib.sha256(index.read_bytes()).hexdigest()
    with pytest.raises(ConversionError):
        load_checkpoint(root, index_compatibility=policy)


def test_index_encoding_must_remain_utf8(tmp_path):
    root = tmp_path / "source"
    _, policy = fixture(root)
    index = root / "model.safetensors.index.json"
    index.write_bytes(index.read_text().encode("utf-16"))
    policy["index_sha256"] = hashlib.sha256(index.read_bytes()).hexdigest()
    with pytest.raises(ConversionError):
        load_checkpoint(root, index_compatibility=policy)


@pytest.mark.parametrize(
    "case",
    [
        "extra_field",
        "wrong_hash",
        "uppercase_hash",
        "bad_inventory",
        "missing_hash",
        "empty_files",
        "nested_file",
        "duplicate_file",
        "unsorted_files",
        "boolean_file",
    ],
)
def test_compatibility_policy_requires_exact_hashes_and_flat_sorted_filenames(tmp_path, case):
    root = tmp_path / "source"
    _, policy = fixture(root, shards=2)
    if case == "extra_field":
        policy["other"] = True
    elif case == "wrong_hash":
        policy["index_sha256"] = "0" * 64
    elif case == "uppercase_hash":
        policy["index_sha256"] = policy["index_sha256"].upper()
    elif case == "bad_inventory":
        policy["tensor_names_sha256"] = True
    elif case == "missing_hash":
        del policy["tensor_names_sha256"]
    elif case == "empty_files":
        policy["weight_filenames"] = []
    elif case == "nested_file":
        policy["weight_filenames"] = ["sub/part-1.safetensors"]
    elif case == "duplicate_file":
        policy["weight_filenames"] = ["part-1.safetensors"] * 2
    elif case == "unsorted_files":
        policy["weight_filenames"].reverse()
    else:
        policy["weight_filenames"] = [True]
    with pytest.raises(ConversionError):
        load_checkpoint(root, index_compatibility=policy)


@pytest.mark.parametrize("case", ["extra", "nested_extra", "missing"])
@pytest.mark.parametrize("verify_content", [False, True])
def test_checkpoint_file_set_changes_reject_source_recheck_and_offline_replay(
    tmp_path, monkeypatch, case, verify_content
):
    root = tmp_path / "source"
    _, policy = fixture(root)
    patch_catalog(monkeypatch, policy)
    snapshot = resolve_source(root, repo=PRIMARY)
    manifest = tmp_path / "manifest.json"
    snapshot.write_manifest(manifest)
    if case == "missing":
        (root / "model.safetensors").unlink()
    else:
        path = root / ("nested/extra.safetensors" if case == "nested_extra" else "extra.safetensors")
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"unexpected")
    with pytest.raises(ConversionError):
        snapshot.assert_unchanged(verify_content=verify_content)
    with pytest.raises(ConversionError):
        resolve_source(PRIMARY, source_manifest=manifest, offline=True)
    with pytest.raises(ConversionError):
        load_checkpoint(root, index_compatibility=policy)


@pytest.mark.parametrize("verify_content", [False, True])
def test_original_index_mutation_is_rejected_even_with_restored_signature(tmp_path, monkeypatch, verify_content):
    root = tmp_path / "source"
    _, policy = fixture(root)
    patch_catalog(monkeypatch, policy)
    snapshot = resolve_source(root, repo=PRIMARY)
    manifest = tmp_path / "manifest.json"
    snapshot.write_manifest(manifest)
    index = root / "model.safetensors.index.json"
    stat = index.stat()
    index.write_bytes(index.read_bytes().replace(b"old-1", b"old-3"))
    os.utime(index, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert index.stat().st_size == stat.st_size
    with pytest.raises(ConversionError, match="compatibility pin"):
        snapshot.assert_unchanged(verify_content=verify_content)
    with pytest.raises(ConversionError):
        resolve_source(PRIMARY, source_manifest=manifest, offline=True)


def test_index_changed_between_selection_and_hashing_is_not_returned_as_a_valid_snapshot(tmp_path, monkeypatch):
    root = tmp_path / "source"
    _, policy = fixture(root)
    patch_catalog(monkeypatch, policy)
    original = sources._verified_hashes

    def mutate_before_hashing(paths, **kwargs):
        index = root / "model.safetensors.index.json"
        index.write_bytes(index.read_bytes() + b" ")
        return original(paths, **kwargs)

    monkeypatch.setattr(sources, "_verified_hashes", mutate_before_hashing)
    with pytest.raises(ConversionError, match="compatibility pin"):
        resolve_source(root, repo=PRIMARY)


def test_duplicates_across_compatible_shards_are_rejected(tmp_path):
    root = tmp_path / "source"
    tensors, policy = fixture(root, shards=2)
    save_tensors(
        root / "part-2.safetensors", {"first.weight": tensors["first.weight"], "first.scales": tensors["first.scales"]}
    )
    with pytest.raises(ConversionError, match="more than one shard"):
        load_checkpoint(root, index_compatibility=policy)


@pytest.mark.parametrize("case", ["extra_name", "missing_name", "missing_indexed_name"])
def test_fallback_requires_exact_actual_inventory_and_all_original_index_keys(tmp_path, case):
    root = tmp_path / "source"
    tensors, policy = fixture(root)
    if case == "extra_name":
        tensors["unexpected"] = tensors["second"]
    elif case == "missing_name":
        del tensors["first.scales"]
    else:
        del tensors["second"]
        # Even an exact inventory pin cannot excuse missing original index keys.
        policy["tensor_names_sha256"] = checkpoint_tensor_names_sha256(tensors)
    save_tensors(root / "model.safetensors", tensors)
    with pytest.raises(ConversionError, match="tensor inventory"):
        load_checkpoint(root, index_compatibility=policy)


def test_available_shards_with_unknown_index_contents_keep_strict_header_validation(tmp_path):
    root = tmp_path / "source"
    tensors, policy = fixture(root)
    save_tensors(root / "old-1.safetensors", {"first.weight": tensors["first.weight"]})
    save_tensors(root / "old-2.safetensors", {"second": tensors["second"]})
    # Unrecognized index contents use strict indexed loading even when a catalog
    # policy exists for a different published index.
    index_path = root / "model.safetensors.index.json"
    index_path.write_bytes(index_path.read_bytes() + b" ")
    selected, index = _weights(names(root), ".safetensors", lambda name: root / name, compatibility=policy)
    assert selected == ("old-1.safetensors", "old-2.safetensors")
    assert index == "model.safetensors.index.json"
    assert set(load_checkpoint(root, index_compatibility=policy)) == {"first.weight", "second"}
    save_tensors(root / "old-1.safetensors", {"not_indexed": tensors["first.weight"]})
    with pytest.raises(ConversionError, match="index disagrees"):
        load_checkpoint(root, index_compatibility=policy)


@pytest.mark.parametrize("shards", [1, 2])
def test_exact_known_index_with_available_files_uses_its_complete_inventory_pin(tmp_path, monkeypatch, shards):
    root = tmp_path / "source"
    tensors, policy = fixture(root, shards=shards, indexed_available=True)
    patch_catalog(monkeypatch, policy)
    snapshot = resolve_source(root, repo=PRIMARY)
    assert set(load_checkpoint(root, index_compatibility=policy)) == set(tensors)
    snapshot.assert_unchanged()
    manifest = tmp_path / "manifest.json"
    snapshot.write_manifest(manifest)
    assert resolve_source(PRIMARY, source_manifest=manifest, offline=True).receipt() == snapshot.receipt()
    with pytest.raises(ConversionError):
        load_checkpoint(root)


def test_changed_index_hash_with_available_files_cannot_use_stale_assignment_exception(tmp_path):
    root = tmp_path / "source"
    _, policy = fixture(root, shards=2, indexed_available=True)
    index = root / "model.safetensors.index.json"
    index.write_bytes(index.read_bytes() + b" ")
    with pytest.raises(ConversionError, match="index disagrees"):
        load_checkpoint(root, index_compatibility=policy)


@pytest.mark.parametrize("case", ["unexpected_file", "unexpected_tensor", "non_subset"])
def test_available_paths_do_not_relax_pinned_filename_or_inventory_bounds(tmp_path, case):
    root = tmp_path / "source"
    tensors, policy = fixture(root, shards=2, indexed_available=True)
    if case == "unexpected_file":
        save_tensors(root / "unexpected.safetensors", {"extra": tensors["second"]})
    elif case == "unexpected_tensor":
        tensors["extra"] = tensors["second"]
        save_tensors(
            root / "part-1.safetensors", {key: value for key, value in tensors.items() if key != "first.scales"}
        )
    else:
        del tensors["second"]
        policy["tensor_names_sha256"] = checkpoint_tensor_names_sha256(tensors)
        save_tensors(root / "part-1.safetensors", {"first.weight": tensors["first.weight"]})
    with pytest.raises(ConversionError):
        load_checkpoint(root, index_compatibility=policy)


@pytest.mark.parametrize("generic", [False, True])
def test_actual_cli_handles_pinned_available_paths_with_stale_map_and_affine_inventory(
    tmp_path, monkeypatch, capsys, generic
):
    source = tmp_path / "source"
    _, tensors = wrapper_fixture(source, bits=8)
    filenames = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    for shard, filename in enumerate(filenames):
        save_tensors(source / filename, dict(list(tensors.items())[shard::2]))
    (source / "model.safetensors").unlink()
    original_names = [name for name in tensors if not name.endswith((".scales", ".biases"))]
    index = source / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": dict.fromkeys(original_names, filenames[1])}))
    policy = {
        "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
        "weight_filenames": filenames,
        "tensor_names_sha256": checkpoint_tensor_names_sha256(tensors),
    }
    repo = "mlx-community/gemma-3-4b-it-8bit"
    catalog = registry.load_catalog()
    entry = next(row for row in catalog["models"] if row["repo"] == repo)
    entry["weight_index_compatibility"] = policy
    # Keep this synthetic end-to-end fixture entirely local. Production profiles
    # retain their declared external generation metadata and exact source pins.
    for descriptor in entry["files"].values():
        descriptor["repo"] = repo
    monkeypatch.setattr(registry, "load_catalog", lambda: copy.deepcopy(catalog))
    tokenizer = Tokenizer(models.WordLevel({f"[SYNTHETIC_{i}]": i for i in range(17)}, unk_token="[SYNTHETIC_0]"))
    tokenizer.save(str(source / "tokenizer.json"))
    (source / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages }}"}))
    (source / "generation_config.json").write_text(json.dumps({"eos_token_id": 1, "temperature": 0.6}))
    output = tmp_path / "output"
    args = ["stark-convert", str(source), "--offline", "--output", str(output)]
    args.extend(["--metadata-profile", "hf-generic"] if generic else ["--repo", repo])
    monkeypatch.setattr(sys, "argv", args)
    assert cli.main() == (1 if generic else 0)
    captured = capsys.readouterr()
    if generic:
        assert json.loads(captured.err)["error_category"] == "ConversionError"
        assert not output.exists()
    else:
        assert json.loads(captured.out)["backend"] == "streaming"
        assert (output / "model.safetensors").is_file()
        receipt = json.loads((output / "conversion.json").read_text())
        assert receipt["source_manifest_sha256"]
