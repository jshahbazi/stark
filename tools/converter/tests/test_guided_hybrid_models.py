"""Native source binding, recurrent planning and the actual guided install route."""

import hashlib
import io
import json
import os
import struct
import threading
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
from test_guided_models import artifact
from test_model_catalog import Hub
from test_vision_family_geometry import geometry

from stark_convert import desktop as module
from stark_convert import guided_memory as memory
from stark_convert import guided_models as guided
from stark_convert import hybrid_prepare as hybrid
from stark_convert import model_catalog, native_catalog
from stark_convert.desktop import Desktop

GIB = guided.GIB


def host_resources(**changes):
    return {
        "supported": True,
        "physical_memory_bytes": 128 * GIB,
        "available_memory_bytes": 120 * GIB,
        "free_disk_bytes": 63_720_000_000,
        "download_free_disk_bytes": 63_720_000_000,
        "gpu_available": True,
        "unified_memory": True,
        "gpu_memory_budget_bytes": 107_520_000_000,
        **changes,
    }


def test_every_fixed_native_source_has_a_separate_bound_plan():
    recipes = guided.native_recipes()
    assert {recipe.native_choice for recipe in recipes} == set(hybrid.SOURCES)
    assert len({recipe.id for recipe in recipes}) == len(recipes)
    for recipe in recipes:
        source = hybrid.SOURCES[recipe.native_choice]
        assert recipe.repo == source["weights_repo"] and recipe.revision == source["weights_revision"]
        assert recipe.runtime_memory > recipe.weight_bytes
        assert recipe.setup_bytes == recipe.installed_bytes + 4 * GIB
        assert recipe.context_tokens == 32768
        assert recipe.gpu_runtime_memory <= recipe.runtime_memory
        assert memory.native_request_bytes(
            recipe.native_choice, recipe.native_profile["geometry"], 8192
        ) <= memory.native_request_bytes(recipe.native_choice, recipe.native_profile["geometry"], 32768)


def test_native_128gb_fit_distinguishes_gpu_paging_disk_and_installed_reuse():
    host = host_resources()
    recipes = {recipe.native_choice: recipe for recipe in guided.native_recipes()}
    assert guided.installation_allowed(recipes["flint-int4"], host)
    for choice in ("qwen38-lily-q4-xl", "qwen38-mlx-serve-mixed"):
        recipe = recipes[choice]
        assert recipe.runtime_memory > host["gpu_memory_budget_bytes"]
        assert recipe.gpu_runtime_memory < host["gpu_memory_budget_bytes"] * 85 // 100
        assert not guided.installation_allowed(recipe, host)
        assert guided.installation_allowed(recipe, host, installed=True)
        assert guided.installation_allowed(recipe, {**host, "free_disk_bytes": 256 * GIB})
        assert not guided.installation_allowed(recipe, {**host, "gpu_memory_budget_bytes": 48 * GIB}, installed=True)
    for choice in ("glm53-q2", "qwen38-mixed"):
        assert not guided.installation_allowed(recipes[choice], host, installed=True)
        larger = {
            **host,
            "physical_memory_bytes": 192 * GIB,
            "gpu_memory_budget_bytes": 144 * GIB,
            "free_disk_bytes": 256 * GIB,
        }
        assert guided.installation_allowed(recipes[choice], larger)
    for recipe in recipes.values():
        assert not guided.installation_allowed(recipe, {**host, "physical_memory_bytes": 16 * GIB}, installed=True)


def test_native_uses_only_library_disk_and_keeps_missing_gpu_fail_closed():
    recipe = guided.native_recipes()[0]
    assert guided.installation_allowed(recipe, host_resources(download_free_disk_bytes=0))
    assert not guided.installation_allowed(recipe, host_resources(free_disk_bytes=recipe.setup_bytes - 1))
    assert not guided.installation_allowed(recipe, host_resources(gpu_available=False), installed=True)
    assert not guided.installation_allowed(
        recipe, {"supported": True, "physical_memory_bytes": 128 * GIB}, installed=True
    )


def test_native_retry_credits_only_owned_allocated_payload_and_rechecks(tmp_path):
    recipe = guided.native_recipes()[0]
    source = hybrid.SOURCES[recipe.native_choice]
    with module.native_download_stage(tmp_path, recipe.native_choice) as directory:
        stage = directory / "model.native-download"
        stage.mkdir()
        marker = stage / "stark-download.json"
        identity = {"choice": recipe.native_choice, "source": source}
        marker.write_text(json.dumps(identity))
        fake = stage / "unrelated.safetensors"
        fake.write_bytes(os.urandom(8192))
        payload_entry, sparse_entry = recipe.native_profile["payload_files"][:2]
        payload = stage / payload_entry["incomplete_path"]
        payload.parent.mkdir(parents=True)
        payload.write_bytes(os.urandom(8192))
        # Another installer holding the lease never supplies credit.
        assert module.native_setup_credit(tmp_path, recipe) == 0
    physical = min(payload.stat().st_size, payload.stat().st_blocks * 512)
    assert module.native_setup_credit(tmp_path, recipe) == physical > 0
    sparse = stage / sparse_entry["incomplete_path"]
    sparse.parent.mkdir(parents=True, exist_ok=True)
    with sparse.open("wb") as stream:
        stream.truncate(sparse_entry["bytes"])
    expected = min(recipe.weight_bytes, physical + min(sparse.stat().st_size, sparse.stat().st_blocks * 512))
    assert module.native_setup_credit(tmp_path, recipe) == expected
    host = host_resources(
        free_disk_bytes=recipe.setup_bytes - expected, native_setup_credit_bytes={recipe.native_choice: expected}
    )
    assert guided.installation_allowed(recipe, host)
    report = guided.recommendation_report(host, {}, candidates=[recipe])
    assert report["items"][0]["setup_disk_bytes_remaining_estimate"] == recipe.setup_bytes - expected
    assert str(tmp_path) not in json.dumps(report)
    marker.write_text(json.dumps({**identity, "choice": "different"}))
    assert module.native_setup_credit(tmp_path, recipe) == 0
    marker.write_text(json.dumps(identity))
    link = stage / "linked.safetensors"
    link.symlink_to(payload)
    assert module.native_setup_credit(tmp_path, recipe) == 0
    link.unlink()
    payload.unlink()
    sparse.unlink()
    fresh = module.native_setup_credit(tmp_path, recipe)
    assert fresh == 0
    assert not guided.installation_allowed(
        recipe, {**host, "native_setup_credit_bytes": {recipe.native_choice: fresh}}
    )
    # Neither a plausible unrelated payload nor a nonregular ownership marker
    # can make discovery block or claim reusable source bytes.
    marker.unlink()
    os.mkfifo(marker)
    result = []
    check = threading.Thread(target=lambda: result.append(module.native_setup_credit(tmp_path, recipe)), daemon=True)
    check.start()
    check.join(1)
    assert not check.is_alive() and result == [0]


def test_changed_source_does_not_inherit_a_native_profile(monkeypatch):
    choice = "qwen38-lily-q4-xl"
    monkeypatch.setitem(hybrid.SOURCES, choice, {**hybrid.SOURCES[choice], "metadata_revision": "f" * 40})
    assert choice not in {recipe.native_choice for recipe in guided.native_recipes()}
    with pytest.raises(ValueError, match="unknown"):
        memory.native_request_bytes("new-unplanned-format", guided.native_recipes()[0].native_profile["geometry"])


def test_native_sources_are_considered_offline_without_catalog_candidates(tmp_path, monkeypatch):
    host = Desktop(tmp_path / "unused-worker", tmp_path / "library", io.StringIO())
    monkeypatch.setattr(module, "host_resources", lambda _path: host_resources())
    hardware = {
        "gpu_available": True,
        "gpu_name": "Apple M5 Max",
        "unified_memory": True,
        "gpu_memory_budget_bytes": 107_520_000_000,
        "cpu_cores": 18,
    }
    try:
        host.publish_guided_models(
            "code",
            "balanced",
            hardware,
            {"excluded_recipe_ids": set(), "excluded_model_ids": set()},
            "offer",
            {"items": [], "incomplete": True},
        )
        report = [json.loads(line) for line in host.output.getvalue().splitlines()][-1]["report"]
        assert report["items"][0]["repo"] == hybrid.SOURCES["flint-int4"]["weights_repo"]
        assert "Native hybrid" in report["message"]
        assert report["catalog_search"]["candidates_considered"] == len(guided.RECIPES) + len(hybrid.SOURCES)
    finally:
        host.stop(quiet=True)


def small_native_recipe(monkeypatch):
    recipe = next(recipe for recipe in guided.native_recipes() if recipe.native_choice == "flint-int4")
    source = {**hybrid.SOURCES[recipe.native_choice], "weight_bytes": 5}
    monkeypatch.setitem(hybrid.SOURCES, recipe.native_choice, source)
    config = {"model_type": "flint", "text_config": {"vocab_size": 8, "eos_token_id": 1}}
    template = (Path(__file__).parent / "fixtures" / "glm53_chat_template.jinja").read_text()
    profile = {
        **recipe.native_profile,
        "config_identity_sha256": memory.config_identity(config),
        "template_sha256": hashlib.sha256(template.encode()).hexdigest(),
    }
    prepared = {
        **deepcopy(config),
        "token_codec_config": {"type": "ChatCodecConfig", "prompt_template": template},
        "stark_native_hybrid": {"schema_version": 1, "source": recipe.repo, "revision": recipe.revision},
    }
    prepared["text_config"]["eos_token_id"] = [1, 2]
    profile["prepared_config_sha256"] = hashlib.sha256(
        json.dumps(prepared, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    recipe = replace(recipe, estimated_weight_bytes=5, minimum_memory=16 * GIB, native_profile=profile)

    def snapshot(_repo, **options):
        root = Path(options["local_dir"])
        assert options["revision"] == source["weights_revision"]
        for i in range(5):
            (root / f"model-{i}.safetensors").write_bytes(b"x")
        (root / "config.json").write_text(json.dumps(config))
        (root / "tokenizer_config.json").write_text(json.dumps({"chat_template": template}))
        (root / "tokenizer.json").write_text("{}")
        (root / "generation_config.json").write_text(json.dumps({"eos_token_id": [1, 2]}))

    monkeypatch.setattr(hybrid, "snapshot_download", snapshot)
    return recipe


def test_guided_native_install_uses_preparation_reuses_and_rejects_receipt_drift(tmp_path, monkeypatch):
    recipe = small_native_recipe(monkeypatch)
    host = Desktop(tmp_path / "unused-worker", tmp_path / "library", io.StringIO())
    monkeypatch.setattr(module, "host_resources", lambda _path: host_resources())
    host.guided_reports["offer"] = {recipe.id: recipe}
    host.guided_hardware["offer"] = host_resources()
    calls = []

    def prepare(command, _environment, _epoch):
        assert command[4:6] == ["stark_convert.hybrid_prepare", recipe.native_choice]
        calls.append(command)
        hybrid.prepare(recipe.native_choice, Path(command[command.index("--output") + 1]))
        return 0

    monkeypatch.setattr(host, "run_converter", prepare)
    command = {
        "command": "install_guided",
        "recommendation_request_id": "offer",
        "recipe_id": recipe.id,
        "support_request_id": "install",
    }
    try:
        host.install_guided(command)
        job = host.job
        if job is not None:
            job.join(5)
            assert not job.is_alive()
        assert len(host.models) == len(calls) == 1
        path = Path(host.models[0]["path"])
        assert guided.matches_recipe(path, recipe)
        host.install_guided(command)
        assert len(calls) == 1
        receipt_path = path / "native-source.json"
        receipt = json.loads(receipt_path.read_bytes())
        for field, value in (
            ("metadata_source", {"repo": recipe.repo, "revision": "f" * 40}),
            ("quantization", "F32"),
        ):
            receipt_path.write_text(json.dumps({**receipt, field: value}))
            assert not guided.matches_recipe(path, recipe)
        receipt_path.write_text(json.dumps(receipt))
        changed = json.loads((path / "config.json").read_bytes())
        changed["text_config"]["vocab_size"] = 9
        encoded = json.dumps(changed).encode()
        (path / "config.json").write_bytes(encoded)
        receipt["config_sha256"] = hashlib.sha256(encoded).hexdigest()
        receipt_path.write_text(json.dumps(receipt))
        assert not guided.matches_recipe(path, recipe)
    finally:
        host.stop(quiet=True)


def test_flint_is_an_explicit_native_catalog_family(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    item = native_catalog.catalog_item("flint-int4")
    assert native_catalog.source_choice("flint", item) == "flint-int4"
    assert native_catalog.family_definition("flint")["conversion"]["containers"] == [".safetensors"]
    assert model_catalog.family_catalog("flint")["items"] == [item]


def test_recurrent_catalog_plan_counts_attention_mtp_and_f32_state(tmp_path, monkeypatch):
    config = geometry()
    config["text_config"]["mtp_num_hidden_layers"] = 1
    hub = Hub()
    template = (Path(__file__).parent / "fixtures" / "qwen3_4b_chat_template.jinja").read_text()
    hub.add(
        tmp_path,
        "example/Hybrid",
        config,
        names={"model.safetensors": 1_234_567_890, "config.json": 100, "tokenizer_config.json": 100},
        sidecars={"tokenizer_config.json": {"chat_template": template}},
    )
    hub.discovered["qwen3_5"] = ["example/Hybrid"]
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(model_catalog, "_catalog_seeds", lambda _family: [])
    monkeypatch.setattr(model_catalog, "get_model", lambda _repo: None)
    item = model_catalog.family_catalog("qwen3_5", hub=hub, planning=True)["items"][0]
    assert item["planning"]["status"] == "available"
    assert item["planning"]["attention_layers"] == 7  # Six target layers and the MTP head.
    recipe = guided.recipe_from_catalog(item)
    assert recipe is not None and recipe.recurrent_state_bytes > 0
    without_recurrence = replace(recipe, recurrent_state_bytes=0, recurrent_workspace_bytes=0)
    assert recipe.runtime_memory > without_recurrence.runtime_memory


def test_recurrent_recipe_accepts_only_its_explicit_f32_control_tensors(tmp_path):
    recipe = replace(guided.RECIPES[1], architecture="qwen3_5")
    root = artifact(tmp_path / "hybrid", recipe)
    weights = root / "model.safetensors"
    original = weights.read_bytes()
    length = struct.unpack("<Q", original[:8])[0]
    header = json.loads(original[8 : 8 + length])
    payload = original[8 + length :]

    def write_tensor(name, shape):
        nonlocal payload
        size = 4
        for value in shape:
            size *= value
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [len(payload), len(payload) + size]}
        payload += b"\0" * size
        encoded = json.dumps(header).encode()
        weights.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
        receipt_path = root / "conversion.json"
        receipt = json.loads(receipt_path.read_bytes())
        receipt["output_bytes"] = weights.stat().st_size
        receipt_path.write_text(json.dumps(receipt))

    for suffix, shape in (("a_log", [2]), ("dt_bias", [2]), ("conv.weights", [2, 4]), ("norm.scales", [2])):
        write_tensor("decoder.transformer.layers.0.mixer." + suffix, shape)
    assert guided.matches_recipe(root, recipe)
    write_tensor("decoder.transformer.layers.0.mixer.out_proj.weights", [2, 2])
    assert not guided.matches_recipe(root, recipe)


@pytest.mark.parametrize(
    "change",
    [
        {"layer_types": [["linear_attention"]]},
        {"linear_num_key_heads": 0},
        {"linear_key_head_dim": 64},
        {"full_attention_interval": 0},
    ],
)
def test_recurrent_geometry_fails_closed(change):
    config = {
        "num_hidden_layers": 4,
        "full_attention_interval": 4,
        "linear_num_key_heads": 4,
        "linear_num_value_heads": 8,
        "linear_conv_kernel_dim": 4,
        **change,
    }
    assert memory.qwen35_geometry(config) is None
