"""Beginner recipes, source binding, scoped cancellation, and context policy."""

import hashlib
import io
import json
import struct
import sys
import threading
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from stark_convert import desktop as module
from stark_convert import guided_models as guided
from stark_convert.desktop import Desktop


def resources(**changes):
    return {
        "supported": True,
        "physical_memory_bytes": 16 * guided.GIB,
        "available_memory_bytes": 12 * guided.GIB,
        "free_disk_bytes": 32 * guided.GIB,
        "download_free_disk_bytes": 32 * guided.GIB,
        **changes,
    }


def artifact(path, recipe, *, dtype="BF16", parameter_dtype="BF16", native_loading="unverified"):
    """Synthetic output plus a source receipt; never real model payloads."""
    path.mkdir(parents=True, exist_ok=True)
    config = json.dumps({"type": "LanguageModelConfig", "decoder_config": {}}).encode()
    (path / "config.json").write_bytes(config)
    (path / "tokenizer.json").write_text("{}")
    header, metadata, offset = {}, {}, 0

    def tensor(name, shape, dtype):
        nonlocal offset
        elements = 1
        for value in shape:
            elements *= value
        size = elements * {"BF16": 2, "F16": 2, "F32": 4, "U32": 4}[dtype]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size

    base = "decoder.embedding.embedding"
    if recipe.quantization.endswith("-bit"):
        metadata[base + ".spec"] = json.dumps(
            {
                "type": "MLXSpec",
                "layout": "output_input",
                "bits": int(recipe.quantization.removesuffix("-bit")),
                "group_size": 64,
            }
        )
        tensor(base + ".weights", [2, 1], "U32")
        tensor(base + ".scales", [1, 2], parameter_dtype)
        tensor(base + ".biases", [1, 2], parameter_dtype)
    else:
        metadata[base + ".spec"] = json.dumps({"type": "FullPrecisionSpec", "layout": "output_input"})
        tensor(base + ".weights", [2, 4], dtype)
    tensor("decoder.transformer.output_norm.scales", [2], "F32")
    header["__metadata__"] = metadata
    encoded = json.dumps(header).encode()
    weights = path / "model.safetensors"
    weights.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\0" * offset)
    sources = [{"repo": recipe.repo, "revision": recipe.revision}]
    selection = recipe.files or (("model.safetensors", recipe.weight_bytes, "a" * 64),)
    files = [
        {"repo": recipe.repo, "revision": recipe.revision, "size": size, "sha256": digest}
        for _name, size, digest in selection
    ]
    identity = {"sources": sources, "files": files}
    if recipe.metadata_profile is not None:
        identity["conversion_profile"] = {
            "architecture": recipe.architecture,
            "metadata_profile": recipe.metadata_profile,
        }
    (path / "conversion.json").write_text(
        json.dumps(
            {
                **identity,
                "source_files": files,
                "source_manifest_sha256": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
                "config_sha256": hashlib.sha256(config).hexdigest(),
                "status": "ok",
                "output_bytes": weights.stat().st_size,
                "capability": {
                    "adapter": recipe.architecture,
                    "model_kind": "language",
                    "conversion": "implemented",
                    "source_validation": "complete",
                    "native_loading": native_loading,
                },
            }
        )
    )
    return path


def partitioned_artifact(path):
    weights = path / "model.safetensors"
    raw = weights.read_bytes()
    header_length = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + header_length])
    metadata = {}
    parent = "decoder.embedding.embedding"
    header = {}
    offset = 0
    for index, (rows, bits) in enumerate(((2, 4), (2, 6))):
        base = f"{parent}.segment_{index}"
        code_bytes = 64 * bits // 8
        header[base + ".weights"] = {
            "dtype": "U8",
            "shape": [rows, code_bytes],
            "data_offsets": [offset, offset + rows * code_bytes],
        }
        offset += rows * code_bytes
        for plane in ("scales", "biases"):
            header[base + "." + plane] = {"dtype": "BF16", "shape": [1, 4], "data_offsets": [offset, offset + 8]}
            offset += 8
    metadata[parent + ".spec"] = json.dumps(
        {
            "type": "MLXRowPartitionedSpec",
            "layout": "output_input",
            "segments": [
                {"rows": 2, "bits": 4, "group_size": 64},
                {"rows": 2, "bits": 6, "group_size": 64},
            ],
        }
    )
    header["decoder.transformer.output_norm.scales"] = {
        "dtype": "F32",
        "shape": [2],
        "data_offsets": [offset, offset + 8],
    }
    offset += 8
    header["__metadata__"] = metadata
    encoded = json.dumps(header).encode()
    weights.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\0" * offset)
    receipt_path = path / "conversion.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["output_bytes"] = weights.stat().st_size
    receipt_path.write_text(json.dumps(receipt))
    return path


class ExistingWorker:
    closed = False

    def close(self):
        self.closed = True


@pytest.fixture
def host(tmp_path, monkeypatch):
    value = Desktop(tmp_path / "unused-worker", tmp_path / "library", io.StringIO(), timeout=2)
    monkeypatch.setattr(module, "host_resources", lambda _path: resources())
    monkeypatch.setattr(module, "model_valid", lambda path: (path / "config.json").is_file())
    yield value
    value.stop(quiet=True)


def events(host):
    return [json.loads(line) for line in host.output.getvalue().splitlines()]


def finish(host):
    job = host.job
    if job is not None:
        job.join(3)
        assert not job.is_alive()
    assert host.job is None


def offer(host, *, request_id="offer", **changes):
    host.dispatch({"command": "guided_models", "request_id": request_id, "task": "everyday", **changes})
    return events(host)[-1]["report"]


def install_command(recipe, *, offer_id="offer", request_id="install"):
    return {
        "command": "install_guided",
        "recommendation_request_id": offer_id,
        "recipe_id": recipe.id,
        "support_request_id": request_id,
    }


def add_model(host, recipe, identifier="installed"):
    path = artifact(host.model_dir / identifier, recipe)
    model = {"id": identifier, "name": recipe.name, "path": str(path), "source": "huggingface:" + recipe.repo}
    host.models.append(model)
    return model


def test_fixed_recipes_bind_complete_real_source_manifests():
    assert len({recipe.id for recipe in guided.RECIPES}) == 2
    for recipe in guided.RECIPES:
        assert len(recipe.revision) == 40 and int(recipe.revision, 16)
        assert recipe.download_bytes == sum(file[1] for file in recipe.files)
        assert recipe.setup_bytes > recipe.download_bytes + recipe.installed_bytes
        assert len({file[0] for file in recipe.files}) == len(recipe.files)
        assert all(len(file[2]) == 64 and int(file[2], 16) for file in recipe.files)


def test_resources_select_starting_choices_and_installed_reuse():
    normal = guided.recommendation_report(resources(), {})
    assert [item["id"] for item in normal["items"]] == [recipe.id for recipe in guided.RECIPES]
    small = guided.recommendation_report(resources(physical_memory_bytes=8 * guided.GIB), {})
    assert small["items"] == [] and "16 GB" in small["message"]
    assert [item["context_tokens"] for item in normal["items"]] == [32768, 32768]
    assert [item["runtime_memory_bytes_estimate"] for item in normal["items"]] == [11 * guided.GIB, 9 * guided.GIB]
    pressure = resources(available_memory_bytes=0, free_disk_bytes=0, download_free_disk_bytes=0)
    assert guided.recommendation_report(pressure, {})["items"] == []
    reused = guided.recommendation_report(pressure, {guided.RECIPES[0].id: ["same-copy"]})
    assert reused["items"][0]["installed_ids"] == ["same-copy"]
    assert "planning estimates" in reused["items"][0]["caveats"][0]


def catalog_candidate(number=1, *, weights=2 * guided.GIB, parameters=2_000_000_000, tags=()):
    return {
        "id": f"{number:032x}",
        "repo": f"example/Model-{number}",
        "revision": f"{number:040x}",
        "name": f"Model {number}",
        "weight_bytes": weights,
        "parameter_count": parameters,
        "quantization": "4-bit",
        "artifact_modes": ["text"],
        "downloadable": True,
        "planning": {
            "status": "available",
            "architecture": "llama",
            "context_limit": 32768,
            "attention_layers": 8,
            "kv_heads": 2,
            "query_heads": 8,
            "head_dim": 64,
            "hidden_size": 1024,
            "download_bytes": weights + 32 * guided.MIB,
            "task_tags": list(tags),
            "chat_template_available": True,
            "supports_tools": True,
            "tool_parser": "qwen3-hermes-v1",
        },
    }


def gpu_resources(**changes):
    return resources(
        physical_memory_bytes=64 * guided.GIB,
        available_memory_bytes=48 * guided.GIB,
        free_disk_bytes=256 * guided.GIB,
        download_free_disk_bytes=256 * guided.GIB,
        gpu_available=True,
        gpu_name="Fixture Apple GPU",
        gpu_memory_budget_bytes=48 * guided.GIB,
        unified_memory=True,
        **changes,
    )


def test_gpu_budget_changes_catalog_recommendation():
    candidates = [
        guided.recipe_from_catalog(catalog_candidate(1)),
        guided.recipe_from_catalog(catalog_candidate(2, weights=6 * guided.GIB, parameters=8_000_000_000)),
        guided.recipe_from_catalog(catalog_candidate(3, weights=20 * guided.GIB, parameters=32_000_000_000)),
    ]
    for budget, expected in ((48, candidates[2]), (16, candidates[1]), (10, candidates[0])):
        host = gpu_resources()
        host["gpu_memory_budget_bytes"] = budget * guided.GIB
        report = guided.recommendation_report(host, {}, preference="answers", candidates=candidates)
        assert report["items"][0]["id"] == expected.id
        assert "GPU budget" in report["items"][0]["recommendation_reason"]
        assert report["items"][0]["evidence"] == "catalogue_metadata"


def test_preferences_and_explicit_task_metadata_change_order():
    small = guided.recipe_from_catalog(catalog_candidate(1))
    large = guided.recipe_from_catalog(catalog_candidate(2, weights=6 * guided.GIB, parameters=8_000_000_000))
    coding = guided.recipe_from_catalog(catalog_candidate(3, tags=("code",)))
    candidates = [small, large, coding]
    balanced = guided.recommendation_report(gpu_resources(), {}, candidates=candidates)
    assert balanced["items"][0]["id"] == large.id
    speed = guided.recommendation_report(gpu_resources(), {}, preference="speed", candidates=candidates)
    assert speed["items"][0]["id"] == small.id
    assert "has not been measured" in speed["items"][0]["recommendation_reason"]
    code = guided.recommendation_report(gpu_resources(), {}, task="code", candidates=candidates)
    assert code["items"][0]["id"] == coding.id
    assert "intended use" in code["items"][0]["recommendation_reason"]


def test_balanced_uses_a_memory_cost_knee_instead_of_largest_checkpoint():
    small = guided.recipe_from_catalog(catalog_candidate(1))
    medium = guided.recipe_from_catalog(catalog_candidate(2, weights=6 * guided.GIB, parameters=8_000_000_000))
    large = guided.recipe_from_catalog(catalog_candidate(3, weights=20 * guided.GIB, parameters=32_000_000_000))
    candidates = [small, medium, large]
    for preference, expected in (("balanced", medium), ("speed", small), ("answers", large)):
        report = guided.recommendation_report(gpu_resources(), {}, preference=preference, candidates=candidates)
        assert report["items"][0]["id"] == expected.id


def test_large_query_head_count_includes_attention_prefill_workspace():
    item = catalog_candidate()
    basic = guided.recipe_from_catalog(item)
    item["planning"]["query_heads"] = 128
    wide = guided.recipe_from_catalog(item)
    assert wide.runtime_memory > basic.runtime_memory
    assert wide.minimum_memory == wide.runtime_memory + 2 * guided.GIB


def test_installed_copy_does_not_override_hardware_or_capacity_ranking():
    small = guided.recipe_from_catalog(catalog_candidate(1))
    large = guided.recipe_from_catalog(catalog_candidate(2, weights=6 * guided.GIB, parameters=8_000_000_000))
    report = guided.recommendation_report(gpu_resources(), {small.id: ["installed"]}, candidates=[small, large])
    assert report["items"][0]["id"] == large.id
    host = gpu_resources()
    host["gpu_memory_budget_bytes"] = 10 * guided.GIB
    report = guided.recommendation_report(host, {large.id: ["installed"]}, candidates=[small, large])
    assert [item["id"] for item in report["items"]] == [small.id]


@pytest.mark.parametrize("task", sorted(guided.TASKS))
@pytest.mark.parametrize("preference", ["balanced", "speed", "answers"])
def test_every_task_respects_explicit_model_priority(task, preference):
    small = guided.recipe_from_catalog(catalog_candidate(1, weights=guided.GIB, parameters=1_000_000_000))
    coding = guided.recipe_from_catalog(
        catalog_candidate(2, weights=6 * guided.GIB, parameters=8_000_000_000, tags=("code",))
    )
    large = guided.recipe_from_catalog(catalog_candidate(3, weights=20 * guided.GIB, parameters=32_000_000_000))
    report = guided.recommendation_report(
        gpu_resources(), {}, task=task, preference=preference, candidates=[coding, large, small]
    )
    if preference == "speed":
        assert report["items"][0]["id"] == small.id
    elif preference == "answers":
        assert report["items"][0]["id"] == large.id
    else:
        assert report["items"][0]["id"] == coding.id
    assert len(report["items"]) == 3


def installed_metadata():
    return {
        "repo": "fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl",
        "revision": "0f94842e3682b49acace4daeb4cf2e65a6f94f83",
        "architecture": "qwen4_exp",
        "weight_bytes": 106_905_155_192,
        "quantization": "Lily q4-xl",
        "context_tokens": 32768,
        "runtime_memory": 116 * guided.GIB,
        "logical_parameters": 120_795_955_200,
        "capacity_is_lower_bound": True,
        "config_sha256": "a" * 64,
    }


def test_installed_display_name_cannot_invalidate_the_native_report(monkeypatch):
    from stark_convert import guided_installed

    monkeypatch.setattr(guided_installed, "inspect_installed", lambda _path: installed_metadata())
    model = {"id": "local", "name": "x" * 201, "path": "/unused"}
    recipe = guided.recipe_from_installed(model)
    assert recipe.name == "Qwen3.8-Flash-Next-lily-q4-xl"
    model["name"] = "Custom installed name"
    assert guided.recipe_from_installed(model).name == model["name"]


def test_installed_flash_next_is_ranked_offline_and_reused_without_download(host, monkeypatch):
    from stark_convert import guided_installed

    path = host.model_dir / "local-flash"
    path.mkdir()
    (path / "config.json").write_text("{}")
    model = {"id": "local-flash", "name": "Qwen3.8 Flash Next", "path": str(path), "source": "local:flash"}
    host.models.append(model)
    monkeypatch.setattr(guided_installed, "inspect_installed", lambda _path: installed_metadata())
    monkeypatch.setattr(
        module,
        "host_resources",
        lambda _path: resources(
            physical_memory_bytes=256 * guided.GIB,
            available_memory_bytes=200 * guided.GIB,
            free_disk_bytes=1,
            download_free_disk_bytes=1,
        ),
    )
    report = offer(host, preference="answers")
    item = report["items"][0]
    assert item["name"] == model["name"]
    assert item["installed_ids"] == [model["id"]]
    assert item["download_bytes"] == 0
    assert item["evidence"] == "installed_metadata"
    assert "lower bound" in item["recommendation_reason"]
    recipe = host.guided_reports["offer"][item["id"]]
    assert not guided.installation_allowed(recipe, resources(physical_memory_bytes=256 * guided.GIB))
    monkeypatch.setattr(host, "start", lambda *_args, **_kwargs: pytest.fail("must reuse local install"))
    host.install_guided(install_command(recipe))
    assert (
        next(event for event in reversed(events(host)) if event["event"] == "installed")["model"]["id"] == model["id"]
    )
    excluded = offer(host, request_id="exclude", preference="answers", excluded_model_ids=[model["id"]])
    assert excluded["items"] == []
    monkeypatch.setattr(guided_installed, "inspect_installed", lambda _path: None)
    assert not guided.matches_recipe(path, recipe)
    with pytest.raises(module.DesktopError, match="changed"):
        host.run_guided(
            {
                "command": "run_guided",
                "request_id": "start",
                "recommendation_request_id": "offer",
                "recipe_id": recipe.id,
                "model_id": model["id"],
            }
        )


def test_unplanned_installed_models_are_disclosed_in_the_guide(host, monkeypatch):
    from stark_convert import guided_installed

    path = host.model_dir / "unplanned"
    path.mkdir()
    (path / "config.json").write_text("{}")
    host.models.append({"id": "unplanned", "name": "Unplanned", "path": str(path)})
    monkeypatch.setattr(guided_installed, "inspect_installed", lambda _path: None)
    assert "1 installed model(s) could not be assessed" in offer(host)["message"]


def test_identical_local_imports_share_one_offer_and_different_model_exclusion(host, monkeypatch):
    from stark_convert import guided_installed

    for identifier in ("first-copy", "second-copy"):
        path = host.model_dir / identifier
        path.mkdir()
        (path / "config.json").write_text("{}")
        host.models.append({"id": identifier, "name": "Same source", "path": str(path), "source": "local:copy"})
    monkeypatch.setattr(guided_installed, "inspect_installed", lambda _path: installed_metadata())
    monkeypatch.setattr(module, "host_resources", lambda _path: resources(physical_memory_bytes=256 * guided.GIB))
    report = offer(host, preference="answers")
    item = report["items"][0]
    assert item["installed_ids"] == ["first-copy", "second-copy"]
    assert len([item for item in report["items"] if item["evidence"] == "installed_metadata"]) == 1
    excluded = offer(host, request_id="different", preference="answers", excluded_model_ids=["second-copy"])
    assert all("first-copy" not in item["installed_ids"] for item in excluded["items"])


def test_paged_weights_have_separate_physical_and_gpu_memory_admission():
    recipe = replace(
        guided.RECIPES[0],
        installed_only=True,
        minimum_memory=114 * guided.GIB,
        estimated_runtime_bytes=112 * guided.GIB,
        estimated_gpu_runtime_bytes=40 * guided.GIB,
    )
    host = gpu_resources()
    host["physical_memory_bytes"] = 128 * guided.GIB
    host["gpu_memory_budget_bytes"] = 107 * guided.GIB
    assert guided.installation_allowed(recipe, host, installed=True)
    assert guided._fit_tier(recipe, host) == 1
    host["physical_memory_bytes"] = 96 * guided.GIB
    assert not guided.installation_allowed(recipe, host, installed=True)
    host["physical_memory_bytes"] = 128 * guided.GIB
    host["gpu_memory_budget_bytes"] = 32 * guided.GIB
    assert not guided.installation_allowed(recipe, host, installed=True)


def test_current_worker_memory_pressure_does_not_remove_static_fit():
    candidate = guided.recipe_from_catalog(catalog_candidate())
    host = gpu_resources()
    host["available_memory_bytes"] = 2 * guided.GIB
    report = guided.recommendation_report(host, {}, candidates=[candidate])
    assert report["items"][0]["id"] == candidate.id
    assert "when starting" in report["items"][0]["recommendation_reason"]


@pytest.mark.parametrize(
    "changes",
    [
        {"gpu_available": False},
        {"gpu_memory_budget_bytes": None},
        {"gpu_memory_budget_bytes": True},
        {"gpu_memory_budget_bytes": 0},
        {"unified_memory": False},
    ],
)
def test_incomplete_or_unusable_gpu_does_not_offer_a_model(changes):
    host = {**gpu_resources(), **changes}
    assert guided.recommendation_report(host, {})["status"] == "unavailable"


@pytest.mark.parametrize(
    "key,value",
    [
        ("architecture", "qwen3_5"),
        ("architecture", []),
        ("context_limit", 16384),
        ("attention_layers", None),
        ("kv_heads", True),
        ("query_heads", 3),
        ("head_dim", 0),
        ("hidden_size", 0),
        ("download_bytes", 1),
        ("status", "unavailable"),
        ("task_tags", [{}]),
        ("chat_template_available", False),
        ("supports_tools", False),
        ("tool_parser", "unknown-v1"),
    ],
)
def test_unknown_or_unsupported_catalog_plans_are_excluded(key, value):
    item = catalog_candidate()
    item["planning"][key] = value
    assert guided.recipe_from_catalog(item) is None


@pytest.mark.parametrize("parser,eligible", [("mistral-json-v1", True), ("functiongemma-v1", False)])
def test_new_protocols_preserve_general_assistant_recommendation_boundary(parser, eligible):
    item = catalog_candidate()
    item["planning"]["architecture"] = "gemma3_text" if parser == "functiongemma-v1" else "mistral"
    item["planning"]["tool_parser"] = parser
    assert (guided.recipe_from_catalog(item) is not None) == eligible


def test_dynamic_recipe_snapshots_source_and_checks_config_receipt(tmp_path):
    item = catalog_candidate()
    candidate = guided.recipe_from_catalog(item)
    assert candidate.minimum_memory == candidate.runtime_memory + 2 * guided.GIB
    assert candidate.context_tokens == 32768
    assert candidate.installed_bytes > item["weight_bytes"]
    assert candidate.setup_bytes > candidate.download_bytes + candidate.installed_bytes
    original_item = deepcopy(candidate.catalog_item)
    item["revision"] = "f" * 40
    assert candidate.catalog_item == original_item
    path = artifact(tmp_path / "dynamic", candidate)
    assert guided.matches_recipe(path, candidate)
    receipt_path = path / "conversion.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["source_files"][0]["revision"] = "f" * 40
    receipt_path.write_text(json.dumps(receipt))
    assert not guided.matches_recipe(path, candidate)


def test_catalog_results_are_bounded_and_unknown_parameter_count_is_honest():
    candidates = [guided.recipe_from_catalog(catalog_candidate(number)) for number in range(1, 6)]
    report = guided.recommendation_report(gpu_resources(), {}, candidates=candidates)
    assert len(report["items"]) == 3
    item = catalog_candidate()
    item["parameter_count"] = None
    candidate = guided.recipe_from_catalog(item)
    report = guided.recommendation_report(gpu_resources(), {}, candidates=[candidate])
    assert "capacity is unknown" in report["items"][0]["recommendation_reason"]


@pytest.mark.parametrize("item", [None, [], {}, {"planning": []}])
def test_malformed_catalog_document_does_not_raise(item):
    assert guided.recipe_from_catalog(item) is None


@pytest.mark.parametrize("key,value", [("name", "bad\ud800"), ("repo", "bad\ud800/model"), ("quantization", "bad\n")])
def test_malformed_catalog_display_metadata_does_not_raise(key, value):
    item = catalog_candidate()
    item[key] = value
    assert guided.recipe_from_catalog(item) is None


def test_explicit_coding_tag_aliases_and_declared_quantized_capacity():
    item = catalog_candidate(tags=("coding", "code-generation"))
    item["parameter_count"] = None
    item["size_label"] = "7B"
    candidate = guided.recipe_from_catalog(item)
    assert candidate.logical_parameters == 7_000_000_000
    assert candidate.task_tags == ("code",)


@pytest.mark.parametrize(
    "changes",
    [
        {"supported": False},
        {"physical_memory_bytes": None},
        {"physical_memory_bytes": True},
        {"physical_memory_bytes": 4 * guided.GIB},
        {"free_disk_bytes": 1},
        {"download_free_disk_bytes": 1},
        {"available_memory_bytes": None},
    ],
)
def test_unknown_or_insufficient_resources_fail_closed(changes):
    report = guided.recommendation_report(resources(**changes), {})
    assert report["status"] == "unavailable" and report["items"] == []


def test_exact_receipt_match_rejects_changed_config_source_and_profile(tmp_path):
    recipe = guided.RECIPES[0]
    path = artifact(tmp_path / "model", recipe)
    original = (path / "conversion.json").read_text()
    assert guided.matches_recipe(path, recipe)
    for key, value in [("conversion_profile", {}), ("sources", []), ("source_files", [])]:
        receipt = json.loads(original)
        receipt[key] = value
        (path / "conversion.json").write_text(json.dumps(receipt))
        assert not guided.matches_recipe(path, recipe)
    (path / "conversion.json").write_text(original)
    (path / "config.json").write_text("{}")
    assert not guided.matches_recipe(path, recipe)


def test_recommendation_excludes_the_rejected_exact_model_and_keeps_worker(host):
    recipe = guided.RECIPES[0]
    add_model(host, recipe)
    worker = host.worker = ExistingWorker()
    messages = host.messages = [{"role": "user", "content": "synthetic retained draft"}]
    report = offer(host, excluded_model_ids=["installed"])
    assert [item["id"] for item in report["items"]] == [guided.RECIPES[1].id]
    assert host.worker is worker and not worker.closed and host.messages is messages
    report = offer(host, excluded_recipe_ids=[item.id for item in guided.RECIPES])
    assert report["items"] == [] and report["status"] == "unavailable"


@pytest.mark.parametrize("change", [{"task": []}, {"task": "other"}, {"excluded_model_ids": "bad"}, {"extra": True}])
def test_malformed_chooser_requests_are_tagged_errors(host, change):
    host.dispatch({"command": "guided_models", "request_id": "request", "task": "everyday", **change})
    assert events(host)[-2]["event"] == "error"
    assert events(host)[-2]["request_id"] == "request"
    assert events(host)[-1]["state"] == "idle"


def test_installed_exact_recipe_is_reused_without_download(host, monkeypatch):
    recipe = guided.RECIPES[0]
    model = add_model(host, recipe)
    report = offer(host)
    assert report["items"][0]["installed_ids"] == [model["id"]]
    monkeypatch.setattr(host, "start", lambda *_args, **_kwargs: pytest.fail("redownloaded an exact installed copy"))
    host.dispatch(install_command(recipe))
    installed = next(event for event in events(host) if event["event"] == "installed")
    assert installed["model"]["id"] == model["id"]
    assert installed["support_request_id"] == installed["request_id"] == "install"


@pytest.mark.parametrize("recipe", guided.RECIPES)
def test_guided_download_pins_own_metadata_and_checks_exact_receipt(host, monkeypatch, recipe):
    worker = host.worker = ExistingWorker()
    retained = host.messages = [{"role": "user", "content": "synthetic retained draft"}]
    offer(host)
    commands = []

    def preflight(command, _environment, _epoch):
        commands.append(command)
        return {"mode": "text", "status": "conditional", "source": {"repo": recipe.repo, "revision": recipe.revision}}

    def convert(command, _environment, _epoch):
        commands.append(command)
        artifact(Path(command[command.index("--output") + 1]), recipe)
        return 0

    monkeypatch.setattr(host, "run_support_check", preflight)
    monkeypatch.setattr(host, "run_converter", convert)
    host.dispatch(install_command(recipe))
    finish(host)
    assert len(commands) == 2
    assert all(command[command.index("--revision") + 1] == recipe.revision for command in commands)
    if recipe.metadata_profile is not None:
        assert all(command[command.index("--metadata-profile") + 1] == recipe.metadata_profile for command in commands)
    else:
        assert all("--metadata-profile" not in command for command in commands)
    assert host.worker is worker and not worker.closed and host.messages is retained
    assert len(host.models) == 1 and guided.matches_recipe(host.models[0]["path"], recipe)
    assert all(event.get("request_id") == "install" for event in events(host) if event["event"] == "progress")
    assert events(host)[-1]["state"] == "ready"


def test_stale_offer_and_changed_resources_do_not_stop_running_model(host, monkeypatch):
    recipe = guided.RECIPES[0]
    worker = host.worker = ExistingWorker()
    host.dispatch(install_command(recipe, offer_id="stale"))
    assert events(host)[-2]["event"] == "error"
    offer(host)
    monkeypatch.setattr(module, "host_resources", lambda _path: resources(free_disk_bytes=1))
    host.dispatch(install_command(recipe))
    assert events(host)[-2]["event"] == "error" and events(host)[-2]["request_id"] == "install"
    assert host.worker is worker and not worker.closed and host.job is None


def test_report_refresh_cannot_evict_the_current_install_binding(host):
    recipe = guided.RECIPES[0]
    add_model(host, recipe)
    offer(host)
    host.dispatch(install_command(recipe))
    for index in range(20):
        offer(host, request_id=f"refresh-{index}")
    assert recipe.id in host.guided_reports["offer"]
    assert len(host.guided_reports) == 8


def test_scoped_cancel_reaps_child_and_preserves_current_model_and_conversation(host):
    worker = host.worker = ExistingWorker()
    retained = host.messages = [{"role": "user", "content": "synthetic retained draft"}]
    host.active_model, host.active_mode, host.api_context = "old-model", "chat", 8192
    script = "import time;time.sleep(30)"

    def install(epoch, _kind):
        host.run_converter([sys.executable, "-c", script], {}, epoch)

    host.start(install, "installing", "Installing", "huggingface", request_id="install", guided_install=True)
    deadline = time.monotonic() + 2
    while host.converter is None and time.monotonic() < deadline:
        time.sleep(0.005)
    child = host.converter
    assert child is not None
    host.dispatch({"command": "cancel_install", "request_id": "wrong"})
    assert events(host)[-1]["event"] == "install_cancel_error" and child.poll() is None
    host.dispatch({"command": "cancel_install", "request_id": "install"})
    finish(host)
    assert child.poll() is not None and host.converter is None
    assert host.worker is worker and not worker.closed and host.messages is retained
    assert host.active_model == "old-model" and host.api_context == 8192 and host.epoch == 0
    assert not any(event["event"] in {"error", "installed"} for event in events(host))
    assert any(event["event"] == "install_cancelled" and event["request_id"] == "install" for event in events(host))
    assert events(host)[-1]["state"] == "ready"
    host.dispatch({"command": "cancel_install", "request_id": "install"})
    assert events(host)[-1]["event"] == "install_cancel_error"


@pytest.mark.parametrize("exception", [RuntimeError("synthetic failure"), module.DesktopError("Setup failed.")])
def test_candidate_failure_preserves_loaded_worker(host, exception):
    worker = host.worker = ExistingWorker()
    retained = host.messages = [{"role": "user", "content": "synthetic retained draft"}]

    def fail(_epoch, _kind):
        raise exception

    host.start(fail, "installing", "Installing", "huggingface", request_id="install", guided_install=True)
    finish(host)
    assert host.worker is worker and not worker.closed and host.messages is retained
    assert events(host)[-2]["event"] == "error" and events(host)[-2]["request_id"] == "install"
    assert events(host)[-1]["state"] == "ready"


@pytest.mark.parametrize("maximum,known", [(40960, True), (0xFFFFFFFF, False)])
def test_guided_context_applies_to_playground_and_api_without_leaking_to_advanced(host, monkeypatch, maximum, known):
    recipe = guided.RECIPES[0]
    model = add_model(host, recipe)
    offer(host)
    worker = host.worker = ExistingWorker()
    host.context_override = 8192

    class ReadyWorker:
        images = False

        def __init__(self, *_args, **_kwargs):
            self.closed = False

        def read(self):
            return {"ready": True, "mode": "text-generation-v1", "max_context_tokens": maximum, "max_new_tokens": 8192}

        def close(self):
            self.closed = True

    monkeypatch.setattr(module, "Worker", ReadyWorker)
    monkeypatch.setattr(module, "load_template", lambda _path: ({}, object()))
    host.dispatch(
        {
            "command": "run_guided",
            "request_id": "start",
            "recommendation_request_id": "offer",
            "recipe_id": recipe.id,
            "model_id": model["id"],
        }
    )
    finish(host)
    assert worker.closed
    assert host.context == host.api_context == recipe.context_tokens
    assert host.context_limit == maximum
    assert host.api_models()[0]["context_length"] == (recipe.context_tokens if known else None)
    assert host.context_capacity_known is known
    ready = [event for event in events(host) if event["event"] == "ready"][-1]
    assert ready["request_id"] == "start" and ready["model_id"] == model["id"]
    assert ready["api_context_tokens"] == recipe.context_tokens and ready["max_context_tokens"] == maximum
    if known:
        host.dispatch(
            {
                "command": "configure",
                "context_tokens": 40000,
                "max_new_tokens": 32,
                "system_prompt": None,
                "thinking": False,
            }
        )
        assert host.context == 40000 and host.api_context == recipe.context_tokens
    host.dispatch({"command": "run", "model_id": model["id"]})
    finish(host)
    assert host.context_limit == host.api_context == maximum and host.context == (40000 if known else 8192)


def test_wrong_installed_model_cannot_use_a_guided_offer_or_stop_current_worker(host):
    recipe = guided.RECIPES[0]
    other = add_model(host, guided.RECIPES[1])
    offer(host)
    worker = host.worker = ExistingWorker()
    host.dispatch(
        {
            "command": "run_guided",
            "request_id": "start",
            "recommendation_request_id": "offer",
            "recipe_id": recipe.id,
            "model_id": other["id"],
        }
    )
    assert events(host)[-2]["event"] == "error" and events(host)[-2]["request_id"] == "start"
    assert host.worker is worker and not worker.closed and host.epoch == 0


@pytest.mark.parametrize("running", [False, True])
def test_rejected_guided_operations_echo_authoritative_preserved_state(host, running):
    if running:
        host.worker = ExistingWorker()
        host.active_model = "existing"
        host.status("ready", "Ready")
    host.dispatch(install_command(guided.RECIPES[0], offer_id="stale"))
    error, status = events(host)[-2:]
    assert error["event"] == "error" and error["request_id"] == "install"
    assert error["operation_started"] is False
    assert error["preserved_model_id"] == ("existing" if running else None)
    assert status["state"] == ("ready" if running else "idle")


def test_rejected_candidate_echoes_the_actual_inflight_operation(host):
    offer(host)
    entered, release = threading.Event(), threading.Event()

    def operation(epoch):
        entered.set()
        assert release.wait(2)
        host.status("idle", "Done", epoch)

    host.start(operation, "generating", "Generating", request_id="other")
    assert entered.wait(2)
    try:
        host.dispatch(install_command(guided.RECIPES[0]))
        assert events(host)[-2]["operation_started"] is False
        assert events(host)[-1]["state"] == "generating"
    finally:
        release.set()
        finish(host)


def test_source_manifest_mismatch_fails_before_publication_and_keeps_worker(host, monkeypatch):
    recipe = guided.RECIPES[0]
    worker = host.worker = ExistingWorker()
    offer(host)
    monkeypatch.setattr(host, "check_install_support", lambda *_args, **_kwargs: recipe.revision)

    def convert(command, _environment, _epoch):
        path = artifact(Path(command[command.index("--output") + 1]), recipe)
        (path / "conversion.json").write_text("{}")
        return 0

    monkeypatch.setattr(host, "run_converter", convert)
    host.dispatch(install_command(recipe))
    finish(host)
    assert host.models == [] and not host.registry.exists() and list(host.model_dir.iterdir()) == []
    assert host.worker is worker and not worker.closed
    assert events(host)[-2]["event"] == "error" and events(host)[-1]["state"] == "ready"


def test_cancel_acknowledgement_is_visible_only_after_the_operation_slot_is_released(host):
    acknowledgements = []

    class ObservedOutput(io.StringIO):
        def write(self, text):
            if json.loads(text).get("event") == "install_cancelled":
                acknowledgements.append(host.job is None)
            return super().write(text)

    host.output = ObservedOutput()
    entered = threading.Event()

    def operation(epoch, _kind):
        entered.set()
        assert host.install_cancelled.wait(2)
        host.check_install_cancelled(epoch)

    host.start(operation, "installing", "Installing", "huggingface", request_id="install", guided_install=True)
    assert entered.wait(2)
    host.dispatch({"command": "cancel_install", "request_id": "install"})
    finish(host)
    assert acknowledgements == [True]


def test_recorded_starter_and_catalog_geometry_use_the_same_prefill_memory_plan():
    for starter in guided.RECIPES:
        discovered = replace(
            starter,
            catalog_item={"planning": {"query_heads": starter.query_heads, "download_bytes": starter.download_bytes}},
        )
        assert starter.runtime_memory == discovered.runtime_memory
    assert guided.RECIPES[0].runtime_memory == 11 * guided.GIB

    # A source's recorded native history cannot bypass the same GPU reserve
    # applied to that artifact when it is discovered through the catalog.
    host = gpu_resources()
    host["gpu_memory_budget_bytes"] = 12 * guided.GIB
    assert not guided.installation_allowed(guided.RECIPES[0], host)


@pytest.mark.parametrize("dtype", ["F16", "F32"])
@pytest.mark.parametrize("catalog", [False, True])
def test_reference_precision_copy_cannot_reuse_a_bf16_recipe(tmp_path, dtype, catalog):
    recipe = (
        guided.recipe_from_catalog({**catalog_candidate(), "quantization": "BF16"}) if catalog else guided.RECIPES[1]
    )
    bf16 = artifact(tmp_path / "bf16", recipe)
    reference = artifact(tmp_path / "reference", recipe, dtype=dtype)
    # Both artifacts have identical pinned sources and config bindings. Only
    # the actual output tensor storage distinguishes this incompatible copy.
    first = json.loads((bf16 / "conversion.json").read_text())
    second = json.loads((reference / "conversion.json").read_text())
    assert first["source_manifest_sha256"] == second["source_manifest_sha256"]
    assert first["config_sha256"] == second["config_sha256"]
    assert guided.matches_recipe(bf16, recipe)
    assert not guided.matches_recipe(reference, recipe)


@pytest.mark.parametrize("parameter_dtype", ["F16", "F32"])
def test_quantized_recipe_requires_bf16_parameters_and_allows_f32_normalization(tmp_path, parameter_dtype):
    recipe = guided.RECIPES[0]
    native = artifact(tmp_path / "native", recipe)
    incompatible = artifact(tmp_path / "incompatible", recipe, parameter_dtype=parameter_dtype)
    assert guided.matches_recipe(native, recipe)
    assert not guided.matches_recipe(incompatible, recipe)


def test_guided_precision_accepts_row_partitioned_integer_weights(tmp_path):
    recipe = replace(guided.RECIPES[0], quantization="4-bit / 6-bit")
    path = partitioned_artifact(artifact(tmp_path / "partitioned", guided.RECIPES[0]))
    assert guided.matches_recipe(path, recipe)


@pytest.mark.parametrize(
    "field,value",
    [("native_loading", "runtime_blocked"), ("source_validation", "not_performed"), ("conversion", "unsupported")],
)
def test_runtime_blocked_or_unvalidated_receipt_cannot_be_reused(tmp_path, field, value):
    recipe = guided.RECIPES[1]
    path = artifact(tmp_path / "model", recipe)
    receipt_path = path / "conversion.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["capability"][field] = value
    receipt_path.write_text(json.dumps(receipt))
    assert not guided.matches_recipe(path, recipe)


def test_installed_weights_must_remain_within_the_planned_size_allowance(tmp_path):
    recipe = replace(guided.RECIPES[1], estimated_weight_bytes=1)
    path = artifact(tmp_path / "model", recipe)
    assert guided.matches_recipe(path, recipe)
    with (path / "model.safetensors").open("r+b") as stream:
        # A sparse file exercises the size gate without allocating a model.
        stream.truncate(recipe.installed_bytes + 1)
    receipt_path = path / "conversion.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["output_bytes"] = recipe.installed_bytes + 1
    receipt_path.write_text(json.dumps(receipt))
    assert not guided.matches_recipe(path, recipe)


def test_weight_precision_check_never_reads_tensor_payloads(tmp_path, monkeypatch):
    import contextlib

    recipe = guided.RECIPES[0]
    path = artifact(tmp_path / "model", recipe)
    original = guided.open_regular

    class HeaderOnly:
        def __init__(self, stream):
            self.stream = stream
            self.allowed = 8

        def fileno(self):
            return self.stream.fileno()

        def read(self, count=-1):
            assert 0 <= count <= self.allowed - self.stream.tell()
            data = self.stream.read(count)
            if self.stream.tell() == 8:
                self.allowed = 8 + struct.unpack("<Q", data)[0]
            return data

    @contextlib.contextmanager
    def header_only(file):
        with original(file) as stream:
            if file.name == "model.safetensors":
                yield HeaderOnly(stream)
            else:
                yield stream

    # Every header reader obtains the same guarded file API. Any payload read
    # would exceed its header boundary and fail the check.
    monkeypatch.setattr(guided, "open_regular", header_only)
    monkeypatch.setattr("stark_convert.model_inventory.open_regular", header_only)
    monkeypatch.setattr("stark_convert.streaming.open_regular", header_only)
    assert guided.matches_recipe(path, recipe)
