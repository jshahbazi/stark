"""Model-free tests for exact-artifact resident-memory policy binding."""

import hashlib
import json

import pytest

from stark_convert import guided_memory_profiles as resident
from stark_convert import guided_models as guided

GIB = guided.GIB


def _artifact(tmp_path, monkeypatch, *, extra=False, link=False, root_link=False, metadata_link=False):
    """Create a sparse exact-identity fixture without allocating model weights."""
    root = tmp_path / "model"
    root.mkdir()
    metadata = {
        "config.json": b"fixture config",
        "native-source.json": b"fixture source receipt",
        "model.safetensors.index.json": b"fixture index",
    }
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in metadata.items()}
    for name, data in metadata.items():
        (root / name).write_bytes(data)
    if metadata_link:
        (root / "native-source.json").unlink()
        (root / "native-source.json").symlink_to(root / "config.json")
    payloads = {
        "one.safetensors": 48_000_000_000,
        "two.safetensors": 27_303_252_216,
        "ngram_table.bin": 32_000_153_976,
    }
    for name, size in payloads.items():
        with (root / name).open("wb") as stream:
            stream.truncate(size)
    if extra:
        (root / "unexpected.safetensors").write_bytes(b"x")
    if link:
        (root / "linked.safetensors").symlink_to(root / "one.safetensors")
    if root_link:
        linked = tmp_path / "linked-model"
        linked.symlink_to(root)
        root = linked
    profile = {
        "schema_version": 1,
        "policy": "qwen38-paged-text-v1",
        "maximum_context_tokens": 32768,
        "maximum_mtp_drafts": 2,
        "request_headroom_divisor": 1,
        "paged_working_set_bytes": GIB,
        "host_reserve_bytes": 8 * GIB,
        "profiles": {
            "qwen38-mlx-serve-mixed": {
                "repo": "ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit",
                "revision": "788c07a8725642bccf13a0a3b8a21954ab84b4f2",
                "weight_bytes": sum(payloads.values()),
                "paged_weight_bytes": 32_000_153_976,
                "files_sha256": hashes,
                "payloads": payloads,
            }
        },
    }
    monkeypatch.setattr(resident, "profiles", lambda: profile)
    return root, profile


def test_profile_requires_exact_metadata_payload_inventory_and_regular_files(tmp_path, monkeypatch):
    root, _ = _artifact(tmp_path, monkeypatch)
    choice = "qwen38-mlx-serve-mixed"
    assert resident.qualified_profile(root, choice) is not None

    (root / "config.json").write_bytes(b"altered config")
    assert resident.qualified_profile(root, choice) is None
    (root / "config.json").write_bytes(b"fixture config")
    (root / "two.safetensors").unlink()
    assert resident.qualified_profile(root, choice) is None

    (tmp_path / "resized").mkdir()
    root, _ = _artifact(tmp_path / "resized", monkeypatch)
    with (root / "one.safetensors").open("r+b") as stream:
        stream.truncate(47_999_999_999)
    assert resident.qualified_profile(root, choice) is None


@pytest.mark.parametrize("variant", ["extra", "payload_symlink", "root_symlink", "metadata_symlink"])
def test_profile_rejects_extra_or_linked_payloads(tmp_path, monkeypatch, variant):
    root, _ = _artifact(
        tmp_path,
        monkeypatch,
        extra=variant == "extra",
        link=variant == "payload_symlink",
        root_link=variant == "root_symlink",
        metadata_link=variant == "metadata_symlink",
    )
    assert resident.qualified_profile(root, "qwen38-mlx-serve-mixed") is None


def test_profile_policy_json_contract_fails_closed(monkeypatch):
    raw = {
        "schema_version": 1,
        "policy": "qwen38-paged-text-v1",
        "maximum_context_tokens": 32768,
        "maximum_mtp_drafts": 2,
        "request_headroom_divisor": 3,
        "paged_working_set_bytes": GIB,
        "host_reserve_bytes": 8 * GIB,
        "profiles": {},
    }
    resident.profiles.cache_clear()
    monkeypatch.setattr(resident.Path, "read_text", lambda _self: json.dumps(raw))
    with pytest.raises(ValueError, match="bounds"):
        resident.profiles()
    resident.profiles.cache_clear()


def test_qualified_and_unqualified_plans_share_recommendation_start_gate(tmp_path, monkeypatch):
    root, fixture = _artifact(tmp_path, monkeypatch)
    recipe = next(item for item in guided.native_recipes() if item.native_choice == "qwen38-mlx-serve-mixed")
    # Keep the profile identity tied to this fixture's source and full sparse payload inventory.
    row = fixture["profiles"][recipe.native_choice]
    row.update(repo=recipe.repo, revision=recipe.revision, weight_bytes=recipe.weight_bytes)
    row["native_profile_sha256"] = hashlib.sha256(
        json.dumps(recipe.native_profile, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    # The production receipt totals and profile payload manifest must agree.
    payload_total = sum(row["payloads"].values())
    assert payload_total == row["weight_bytes"]
    profiled = guided.with_resident_profile(recipe, root)
    assert profiled.resident_profile is not None
    fallback_plan = guided.with_resident_profile(recipe, tmp_path / "no-artifact")
    assert fallback_plan.runtime_memory == 111 * GIB
    assert profiled.runtime_memory == 88 * GIB

    host = {
        "supported": True,
        "physical_memory_bytes": 128 * GIB,
        "available_memory_bytes": profiled.runtime_memory,
        "gpu_available": True,
        "unified_memory": True,
        "gpu_memory_budget_bytes": 107_520_000_000,
        "free_disk_bytes": 0,
        "download_free_disk_bytes": 0,
    }
    assert guided.execution_allowed(profiled, host)
    assert not guided.execution_allowed(profiled, {**host, "available_memory_bytes": profiled.runtime_memory - 1})
    report = guided.recommendation_report(
        host, {recipe.id: ["installed-copy"]}, candidates=[profiled], benchmarks=guided.load_registry()
    )
    assert [item["id"] for item in report["items"]] == [recipe.id]
    start_host = {**host, "available_memory_bytes": profiled.runtime_memory}
    assert guided.execution_allowed(profiled, start_host)

    # A missing or changed artifact uses the conservative full-file charge.
    (root / "one.safetensors").unlink()
    fallback = guided.with_resident_profile(recipe, root)
    assert fallback.resident_profile is None
    assert fallback.runtime_memory == fallback_plan.runtime_memory
    assert not guided.execution_allowed(fallback, host)


def test_payload_sum_and_native_paging_identity_mismatches_fail_closed(tmp_path, monkeypatch):
    root, fixture = _artifact(tmp_path, monkeypatch)
    choice = "qwen38-mlx-serve-mixed"
    row = fixture["profiles"][choice]
    row["weight_bytes"] += 1
    assert resident.qualified_profile(root, choice) is None

    row["weight_bytes"] -= 1
    recipe = next(item for item in guided.native_recipes() if item.native_choice == choice)
    row.update(repo=recipe.repo, revision=recipe.revision, weight_bytes=recipe.weight_bytes)
    row["native_profile_sha256"] = hashlib.sha256(
        json.dumps(recipe.native_profile, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert resident.qualified_profile(root, choice) is not None
    row["paged_weight_bytes"] -= 1
    assert guided.with_resident_profile(recipe, root).resident_profile is None


@pytest.mark.parametrize(
    "change",
    ["weight", "revision", "paged", "geometry", "template"],
)
def test_source_profile_binds_canonical_native_recipe_revision_and_layout(change):
    recipe = next(item for item in guided.native_recipes() if item.native_choice == "qwen38-mlx-serve-mixed")
    planned = resident.source_profile(recipe.native_choice, recipe.native_profile)
    assert planned is not None
    assert (planned.repo, planned.revision, planned.weight_bytes, planned.paged_weight_bytes) == (
        recipe.repo,
        recipe.revision,
        recipe.weight_bytes,
        recipe.native_profile["paged_weight_bytes"],
    )
    changed = json.loads(json.dumps(recipe.native_profile))
    if change == "weight":
        changed["weight_bytes"] += 1
    elif change == "revision":
        changed["weights_source"]["revision"] = "f" * 40
    elif change == "paged":
        changed["paged_weight_bytes"] += 1
    elif change == "geometry":
        changed["geometry"]["num_hidden_layers"] += 1
    else:
        changed["template_sha256"] = "f" * 64
    assert resident.source_profile(recipe.native_choice, changed) is None
