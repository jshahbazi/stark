"""Metadata-only family browsing never downloads model weight payloads."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError
from test_dense_adapters import dense_fixture
from test_streaming import qwen_fixture
from test_vision_family_geometry import geometry

from stark_convert import model_catalog as catalog

PIN = "a" * 40
CATALOG_SEEDS = catalog._catalog_seeds


def test_gemma3_image_wrapper_is_discoverable_with_bounded_large_tokenizer_metadata(tmp_path, monkeypatch):
    from test_gemma3_vision import config_for_size

    from stark_convert.gemma3_contract import PROCESSOR

    hub = Hub()
    repo = "google/gemma-3-4b-it"
    config = {**config_for_size(2560), "torch_dtype": "bfloat16"}
    template = "{{ messages }}"
    # Metadata routing uses a synthetic reviewed program identity. Native
    # execution and real template qualification remain separate model checks.
    monkeypatch.setattr(catalog, "_GEMMA3_IMAGE_TEMPLATE", hashlib.sha256(template.encode()).hexdigest())
    monkeypatch.setattr(catalog, "get_model", lambda _repo: {"adapter": None})
    tokenizer_config = {"chat_template": template, "synthetic_padding": "x" * (1024 * 1024 + 128)}
    hub.add(
        tmp_path,
        repo,
        config,
        names={
            "config.json": 100,
            "model.safetensors": 8_600_158_944,
            "tokenizer_config.json": len(json.dumps(tokenizer_config)),
            "preprocessor_config.json": 1024,
        },
        sidecars={"tokenizer_config.json": tokenizer_config, "preprocessor_config.json": PROCESSOR},
    )
    hub.discovered["gemma3"] = [repo]
    report = catalog.family_catalog("gemma3", hub=hub)
    assert report["status"] == "ready"
    assert len(report["items"]) == 1
    item = report["items"][0]
    assert item["repo"] == repo and item["revision"] == PIN
    assert item["artifact_modes"] == ["vision-language"]
    assert item["default_artifact_mode"] == "vision-language"
    assert not any(event[0] in {"head", "download"} and event[-1].endswith(".safetensors") for event in hub.events)


@pytest.fixture(autouse=True)
def clean_offline_environment(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(catalog, "_catalog_seeds", lambda _family: [])
    monkeypatch.setattr(catalog, "get_model", lambda _repo: None)
    # These tests isolate discovery/metadata routing. Header admission has its
    # own tests with complete converter fixtures and bounded HTTP responses.
    monkeypatch.setattr(catalog, "_verify_hub_layout", lambda *args: dict.fromkeys(args[-1]))


class Hub:
    def __init__(self):
        self.repos = {}
        self.discovered = {}
        self.searched = {}
        self.cached = {}
        self.events = []

    def add(
        self,
        tmp_path,
        repo,
        config,
        *,
        names=None,
        parameters=None,
        revision=PIN,
        sidecars=None,
        pipeline_tag=None,
        tags=(),
        card_data=None,
    ):
        root = tmp_path / repo.replace("/", "--")
        root.mkdir(parents=True)
        (root / "config.json").write_text(json.dumps(config))
        for name, data in (sidecars or {}).items():
            (root / name).write_bytes(data if isinstance(data, bytes) else json.dumps(data).encode())
        names = names or {"model.safetensors": 1234567890, "config.json": (root / "config.json").stat().st_size}
        summary = parameters or {"BF16": 1000000000}
        info = SimpleNamespace(
            id=repo,
            sha=revision,
            siblings=[SimpleNamespace(rfilename=name, size=size) for name, size in names.items()],
            safetensors={"parameters": summary, "total": sum(summary.values())},
            pipeline_tag=pipeline_tag,
            tags=list(tags),
            cardData=card_data or {},
        )
        self.repos[repo] = root, info
        return root

    def HfApi(self):
        return self

    def list_models(self, *, filter=None, search=None, **kwargs):
        self.events.append(("list", filter or search, kwargs))
        results = self.searched.get(search, []) if search is not None else self.discovered.get(filter, [])
        return [SimpleNamespace(id=repo) for repo in results]

    def model_info(self, *, repo_id, **kwargs):
        self.events.append(("info", repo_id, kwargs))
        return self.repos[repo_id][1]

    def hf_hub_url(self, *, repo_id, filename, revision):
        return repo_id, filename, revision

    def try_to_load_from_cache(self, repo_id, filename, *, revision):
        self.events.append(("cache", repo_id, filename, revision))
        return self.cached.get((repo_id, filename, revision))

    def get_hf_file_metadata(self, url, **kwargs):
        repo, filename, revision = url
        assert not filename.endswith((".safetensors", ".gguf", ".bin"))
        self.events.append(("head", repo, filename))
        return SimpleNamespace(size=(self.repos[repo][0] / filename).stat().st_size, commit_hash=revision)

    def hf_hub_download(self, *, repo_id, filename, revision, **kwargs):
        assert not filename.endswith((".safetensors", ".gguf", ".bin"))
        self.events.append(("download", repo_id, filename))
        return str(self.repos[repo_id][0] / filename)


def test_live_compatible_fine_tune_has_pinned_download_identity_and_real_sizes(tmp_path):
    hub = Hub()
    hub.add(tmp_path, "someone/My-FineTune", {"model_type": "qwen3", "torch_dtype": "float16"})
    hub.discovered["qwen3"] = ["someone/My-FineTune"]
    report = catalog.family_catalog("qwen3", hub=hub)
    assert report["status"] == "ready" and len(report["items"]) == 1
    item = report["items"][0]
    assert item == {
        "id": item["id"],
        "repo": "someone/My-FineTune",
        "revision": PIN,
        "name": "My-FineTune",
        "size_label": "1B",
        "parameter_count": 1000000000,
        "weight_bytes": 1234567890,
        "quantization": "BF16",
        "artifact_modes": ["text"],
        "default_artifact_mode": "text",
        "downloadable": True,
        "model_group": item["model_group"],
    }
    assert item["model_group"]["name"] == "My-FineTune"
    assert item["model_group"]["publisher"] == "someone"
    assert len(item["model_group"]["id"]) == 32
    assert len(item["id"]) == 32
    assert [event[2] for event in hub.events if event[0] == "download"] == ["config.json"]


def test_catalog_reuses_pinned_huggingface_disk_cache_across_requests(tmp_path):
    hub = Hub()
    hub.add(tmp_path, "someone/Stable", {"model_type": "qwen3", "torch_dtype": "float16"})
    hub.discovered["qwen3"] = ["someone/Stable"]

    first = catalog.family_catalog("qwen3", hub=hub)
    cached_root = tmp_path / "hf" / "models--someone--Stable" / "snapshots" / PIN
    cached_root.mkdir(parents=True)
    cached_config = cached_root / "config.json"
    cached_config.write_bytes((hub.repos["someone/Stable"][0] / "config.json").read_bytes())
    hub.cached[("someone/Stable", "config.json", PIN)] = str(cached_config)
    second = catalog.family_catalog("qwen3", hub=hub)

    assert first["items"] == second["items"]
    assert [event for event in hub.events if event[0] == "head"] == [("head", "someone/Stable", "config.json")]
    assert [event for event in hub.events if event[0] == "download"] == [("download", "someone/Stable", "config.json")]
    assert len([event for event in hub.events if event[0] == "cache"]) == 2

    # A changed immutable revision cannot select the previous snapshot entry.
    hub.repos["someone/Stable"][1].sha = "b" * 40
    third = catalog.family_catalog("qwen3", hub=hub)
    assert third["items"][0]["revision"] == "b" * 40
    assert len([event for event in hub.events if event[0] == "head"]) == 2
    assert len([event for event in hub.events if event[0] == "download"]) == 2


def test_catalog_rejects_cached_path_outside_huggingface_snapshot(tmp_path):
    hub = Hub()
    root = hub.add(tmp_path, "someone/Outside", {"model_type": "qwen3", "torch_dtype": "float16"})
    hub.discovered["qwen3"] = ["someone/Outside"]
    other_snapshot = tmp_path / "hf" / "models--someone--Other" / "snapshots" / PIN
    other_snapshot.mkdir(parents=True)
    cached_config = other_snapshot / "config.json"
    cached_config.write_bytes((root / "config.json").read_bytes())
    hub.cached[("someone/Outside", "config.json", PIN)] = str(cached_config)

    report = catalog.family_catalog("qwen3", hub=hub)

    assert report["status"] == "ready" and len(report["items"]) == 1
    assert [event for event in hub.events if event[0] == "head"] == [("head", "someone/Outside", "config.json")]


@pytest.mark.parametrize(
    "cache_content,expected_status",
    [
        (b"{", "ready"),
        (b"x" * (catalog.MAX_CONFIG_BYTES + 1), "error"),
    ],
)
def test_catalog_does_not_trust_malformed_or_oversized_cached_metadata(tmp_path, cache_content, expected_status):
    hub = Hub()
    hub.add(tmp_path, "someone/CachedBad", {"model_type": "qwen3", "torch_dtype": "float16"})
    hub.discovered["qwen3"] = ["someone/CachedBad"]
    cached_root = tmp_path / "hf" / "models--someone--CachedBad" / "snapshots" / PIN
    cached_root.mkdir(parents=True)
    cached_config = cached_root / "config.json"
    cached_config.write_bytes(cache_content)
    hub.cached[("someone/CachedBad", "config.json", PIN)] = str(cached_config)

    report = catalog.family_catalog("qwen3", hub=hub)

    assert report["status"] == expected_status
    assert report["items"] == []
    assert not [event for event in hub.events if event[0] in {"head", "download"}]


def test_catalog_rejects_cached_snapshot_file_escaping_to_another_directory(tmp_path):
    hub = Hub()
    hub.add(tmp_path, "someone/Escaping", {"model_type": "qwen3", "torch_dtype": "float16"})
    hub.discovered["qwen3"] = ["someone/Escaping"]
    cached_root = tmp_path / "hf" / "models--someone--Escaping" / "snapshots" / PIN
    cached_root.mkdir(parents=True)
    outside = tmp_path / "outside-config.json"
    outside.write_text(json.dumps({"model_type": "qwen3", "torch_dtype": "float16"}))
    cached_config = cached_root / "config.json"
    cached_config.symlink_to(outside)
    hub.cached[("someone/Escaping", "config.json", PIN)] = str(cached_config)

    report = catalog.family_catalog("qwen3", hub=hub)

    assert report["status"] == "error"
    assert report["items"] == []
    assert not [event for event in hub.events if event[0] in {"head", "download"}]


def test_planning_is_opt_in_and_exposes_sanitized_dense_geometry(tmp_path):
    hub = Hub()
    template = (Path(__file__).parent / "fixtures" / "qwen3_4b_chat_template.jinja").read_text()
    hub.add(
        tmp_path,
        "someone/Coder",
        {
            "model_type": "qwen3",
            "torch_dtype": "bfloat16",
            "max_position_embeddings": 65536,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "hidden_size": 2048,
        },
        names={
            "model.safetensors": 123456,
            "config.json": 100,
            "README.md": 25,
            "tokenizer_config.json": 100,
        },
        sidecars={"tokenizer_config.json": {"chat_template": template}},
        tags=["coding", "text-generation", "unknown-tag"],
        card_data={"license": "apache-2.0", "extra": "ignored"},
    )
    hub.discovered["qwen3"] = ["someone/Coder"]

    default = catalog.family_catalog("qwen3", hub=hub)
    planned = catalog.family_catalog("qwen3", hub=hub, planning=True, deadline_seconds=10)

    assert "planning" not in default["items"][0]
    item = planned["items"][0]
    assert item["task_tags"] == ["coding"]
    assert item["license"] == "apache-2.0"
    assert item["planning"] == {
        "status": "available",
        "architecture": "qwen3",
        "context_limit": 65536,
        "attention_layers": 28,
        "query_heads": 16,
        "kv_heads": 4,
        "head_dim": 128,
        "hidden_size": 2048,
        "download_bytes": 123681,
        "task_tags": ["coding"],
        "license": "apache-2.0",
        "chat_template_available": True,
        "supports_tools": True,
        "tool_parser": "qwen3-hermes-v1",
        "blockers": [],
    }


def test_planning_requires_pinned_chat_template_and_tool_parser(tmp_path):
    hub = Hub()
    hub.add(
        tmp_path,
        "someone/BaseCheckpoint",
        {
            "model_type": "qwen3",
            "max_position_embeddings": 4096,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "hidden_size": 128,
        },
    )
    hub.discovered["qwen3"] = ["someone/BaseCheckpoint"]
    item = catalog.family_catalog("qwen3", hub=hub, planning=True)["items"][0]
    assert item["planning"]["chat_template_available"] is False
    assert item["planning"]["supports_tools"] is False
    assert "tokenizer_config_missing" in item["planning"]["blockers"]
    assert "tool_parser_unavailable" in item["planning"]["blockers"]
    assert item["planning"]["status"] == "unavailable"


@pytest.mark.parametrize(
    "family,config,blocker",
    [
        (
            "qwen3",
            {"model_type": "qwen3", "num_hidden_layers": 28, "num_attention_heads": 16},
            "context_limit_unknown",
        ),
        (
            "qwen3_5",
            {
                "model_type": "qwen3_5",
                "max_position_embeddings": 65536,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
                "hidden_size": 2048,
            },
            "runtime_layout_not_covered",
        ),
    ],
)
def test_planning_marks_unknown_context_and_hybrid_layout_unavailable(family, config, blocker):
    info = SimpleNamespace(tags=[], cardData={})
    planning = catalog._planning_metadata(family, config, info, ["model.safetensors"], {"model.safetensors": 100})
    assert planning["status"] == "unavailable"
    assert blocker in planning["blockers"]


@pytest.mark.parametrize("deadline", [0, -1, 601, True, "10"])
def test_planning_deadline_must_be_bounded_numeric_seconds(deadline):
    assert catalog.family_catalog("qwen3", hub=object(), planning=True, deadline_seconds=deadline)["status"] == "error"


@pytest.mark.parametrize(
    "config,names,parameters",
    [
        ({"model_type": "qwen3_moe"}, None, None),
        ({"model_type": "qwen3", "num_experts": 8}, None, None),
        ({"model_type": "llama"}, None, None),
        ({"model_type": "qwen3", "quantization_config": {"quant_method": "gptq", "bits": 4}}, None, None),
        ({"model_type": "qwen3", "auto_map": {"AutoModel": "private.Custom"}}, None, None),
        ({"model_type": "qwen3"}, {"model.gguf": 100}, None),
        ({"model_type": "qwen3"}, {"adapter_model.safetensors": 100, "adapter_config.json": 100}, None),
        ({"model_type": "qwen3"}, None, {"U32": 100}),
    ],
)
def test_metadata_proven_incompatible_candidates_are_not_downloadable(tmp_path, config, names, parameters):
    hub = Hub()
    hub.add(tmp_path, "someone/Wrong", config, names=names, parameters=parameters)
    hub.discovered["qwen3"] = ["someone/Wrong"]
    report = catalog.family_catalog("qwen3", hub=hub)
    assert report["status"] == "ready"
    if config["model_type"] in {"qwen3_moe", "llama"} or names and "adapter_model.safetensors" in names:
        assert report["items"] == []
    else:
        assert len(report["items"]) == 1
        assert report["items"][0]["downloadable"] is False
        assert report["items"][0]["unavailable_reason"]


@pytest.mark.parametrize("architecture", ["Qwen3Model", "Qwen3ForSequenceClassification", "Qwen3ForTextEmbedding"])
def test_non_generative_architecture_is_not_offered_as_a_chat_model(tmp_path, architecture):
    hub = Hub()
    hub.add(tmp_path, "someone/Model", {"model_type": "qwen3", "architectures": [architecture]})
    hub.discovered["qwen3"] = list(hub.repos)
    assert catalog.family_catalog("qwen3", hub=hub)["items"] == []


@pytest.mark.parametrize("task", ["feature-extraction", "sentence-similarity", "text-ranking", "text-classification"])
def test_hub_non_generative_task_excludes_even_a_causal_language_architecture(tmp_path, task):
    hub = Hub()
    hub.add(
        tmp_path,
        "someone/Model",
        {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]},
        pipeline_tag=task,
    )
    hub.discovered["qwen3"] = list(hub.repos)
    assert catalog.family_catalog("qwen3", hub=hub)["items"] == []
    assert not [event for event in hub.events if event[0] == "download"]


@pytest.mark.parametrize(
    "repo,tags",
    [
        ("someone/Model", ["reranker"]),
        ("someone/Model", ["tiny-random"]),
        ("someone/tiny-random-Qwen3", []),
        ("trl-internal-testing/tiny-Qwen3ForCausalLM", []),
    ],
)
def test_non_chat_and_internal_test_sources_do_not_displace_application_models(tmp_path, repo, tags):
    hub = Hub()
    hub.add(tmp_path, repo, {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}, tags=tags)
    hub.discovered["qwen3"] = [repo]
    assert catalog.family_catalog("qwen3", hub=hub)["items"] == []


@pytest.mark.parametrize(
    "family,architecture",
    [
        ("qwen3", "Qwen3ForCausalLM"),
        ("qwen2", "Qwen2ForCausalLM"),
        ("llama", "LlamaForCausalLM"),
        ("mistral", "MistralForCausalLM"),
        ("phi3", "Phi3ForCausalLM"),
        ("granite", "GraniteForCausalLM"),
        ("gemma2", "Gemma2ForCausalLM"),
        ("gemma3_text", "Gemma3ForCausalLM"),
        ("smollm3", "SmolLM3ForCausalLM"),
        ("lfm2", "Lfm2ForCausalLM"),
        ("qwen3_5", "Qwen3_5ForCausalLM"),
        ("qwen3_5", "Qwen3_5ForConditionalGeneration"),
        ("qwen4_exp", "Qwen4ExpForConditionalGeneration"),
    ],
)
def test_supported_generation_class_hints_remain_available(tmp_path, family, architecture):
    hub = Hub()
    hub.add(
        tmp_path,
        "someone/Model",
        {"model_type": family, "architectures": [architecture]},
        pipeline_tag="text-generation",
    )
    hub.discovered[family] = ["someone/Model"]
    assert len(catalog.family_catalog(family, hub=hub)["items"]) == 1


@pytest.mark.parametrize("bits", [4, 8])
def test_mlx_quantization_uses_metadata_and_does_not_call_packed_slots_parameters(tmp_path, bits):
    hub = Hub()
    hub.add(
        tmp_path,
        "someone/No-size-in-name",
        {"model_type": "qwen3", "quantization": {"bits": bits, "group_size": 64}},
        parameters={"U32": 100000000},
    )
    hub.discovered["qwen3"] = ["someone/No-size-in-name"]
    item = catalog.family_catalog("qwen3", hub=hub)["items"][0]
    assert item["quantization"] == f"{bits}-bit"
    assert item["parameter_count"] is None and item["size_label"] is None


@pytest.mark.parametrize("family", ["qwen3", "qwen2", "llama", "mistral"])
@pytest.mark.parametrize("tied", [False, True])
def test_quantized_capacity_matches_independent_float_tensor_inventory(tmp_path, family, tied):
    if family == "qwen3":
        config, tensors = qwen_fixture(tmp_path / "reference", tied=tied, attention_bias=True)
    else:
        config, tensors = dense_fixture(tmp_path / "reference", family, tied=tied)
    # Count the reference float checkpoint's independent tensor inventory,
    # counting the tied embedding once even when its alias is serialized twice.
    expected = sum(
        values.size for name, (_dtype, values) in tensors.items() if not (tied and name == "lm_head.weight")
    )
    assert catalog._dense_config_parameters(config, family) == expected
    if family in {"qwen2", "mistral"}:
        # Their config shapes are covered, but catalog preflight still excludes
        # these quantized routes. Capacity inference must not expand admission.
        return
    hub = Hub()
    hub.discovered[family] = []
    for bits in (4, 8):
        repo = f"someone/Anonymous-export-{bits}"
        hub.add(
            tmp_path,
            repo,
            {**config, "quantization": {"bits": bits, "group_size": 64}},
            parameters={"U32": 1234567, "BF16": 89},
        )
        hub.discovered[family].append(repo)
    items = catalog.family_catalog(family, hub=hub)["items"]
    assert len(items) == 2
    assert {item["parameter_count"] for item in items} == {expected}
    assert all(item["size_label"] for item in items)
    assert not any(event[0] == "download" and event[2].endswith(".safetensors") for event in hub.events)


@pytest.mark.parametrize(
    "changes",
    [
        {"intermediate_size": None},
        {"hidden_size": True},
        {"num_hidden_layers": 1000000},
        {"num_key_value_heads": 3},
        {"head_dim": 3},
        {"tie_word_embeddings": "false"},
        {"attention_bias": "false"},
        {"qkv_bias": True},
        {"use_qk_norm": False},
        {"mlp_bias": True},
        {"num_experts": 8},
        {"add_cross_attention": True},
        {"pruned_heads": {"0": [1]}},
    ],
)
def test_incomplete_or_unsupported_capacity_geometry_remains_unknown(tmp_path, changes):
    config, _ = qwen_fixture(tmp_path / "reference")
    assert catalog._dense_config_parameters({**config, **changes}, "qwen3") is None
    assert catalog._dense_config_parameters(config, "qwen3_moe") is None


def test_more_capacity_can_choose_unregistered_quantized_model_from_catalog_metadata(tmp_path):
    from stark_convert import guided_models as guided

    hub = Hub()
    template = (Path(__file__).parent / "fixtures" / "qwen3_4b_chat_template.jinja").read_text()
    common = {
        "model_type": "qwen3",
        "max_position_embeddings": 32768,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "vocab_size": 151936,
    }
    names = {"model.safetensors": 2 * guided.GIB, "config.json": 100, "tokenizer_config.json": 100}
    sidecars = {"tokenizer_config.json": {"chat_template": template}}
    hub.add(
        tmp_path,
        "someone/Anonymous-quantized",
        {**common, "quantization": {"bits": 4, "group_size": 64}},
        names=names,
        parameters={"U32": 100000000},
        sidecars=sidecars,
    )
    hub.add(
        tmp_path,
        "someone/Small-floating",
        common,
        names=names,
        parameters={"BF16": 600000000},
        sidecars=sidecars,
    )
    hub.discovered["qwen3"] = ["someone/Small-floating", "someone/Anonymous-quantized"]
    items = catalog.family_catalog("qwen3", hub=hub, planning=True)["items"]
    candidates = [guided.recipe_from_catalog(item) for item in items]
    assert all(candidate is not None for candidate in candidates)
    host = {
        "supported": True,
        "physical_memory_bytes": 64 * guided.GIB,
        "available_memory_bytes": 48 * guided.GIB,
        "free_disk_bytes": 100 * guided.GIB,
        "download_free_disk_bytes": 100 * guided.GIB,
        "gpu_available": True,
        "unified_memory": True,
        "gpu_memory_budget_bytes": 48 * guided.GIB,
    }
    report = guided.recommendation_report(host, {}, preference="answers", candidates=candidates)
    assert report["items"][0]["repo"] == "someone/Anonymous-quantized"


@pytest.mark.parametrize(
    "tag", ["code", "coding", "programming", "coder", "code-generation", "code_generation", "code-completion"]
)
def test_catalog_coding_aliases_reach_guided_ranking(tmp_path, tag):
    from stark_convert.guided_models import recipe_from_catalog

    hub = Hub()
    template = (Path(__file__).parent / "fixtures" / "qwen3_4b_chat_template.jinja").read_text()
    hub.add(
        tmp_path,
        "someone/Coder",
        {
            "model_type": "qwen3",
            "max_position_embeddings": 32768,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "hidden_size": 2048,
        },
        names={"model.safetensors": 123456, "config.json": 100, "tokenizer_config.json": 100},
        sidecars={"tokenizer_config.json": {"chat_template": template}},
        tags=[tag.upper()],
    )
    hub.discovered["qwen3"] = ["someone/Coder"]
    item = catalog.family_catalog("qwen3", hub=hub, planning=True)["items"][0]
    assert item["planning"]["task_tags"] == [tag]
    assert recipe_from_catalog(item).task_tags == ("code",)


def test_qwen35_mixed_mlx_quantization_badge_lists_declared_widths():
    config = {
        "model_type": "qwen3_5",
        "quantization": {
            "bits": 4,
            "group_size": 64,
            "mode": "affine",
            "model.layers.0.mlp.up_proj": {"bits": 5, "group_size": 64, "mode": "affine"},
            "model.layers.1.mlp.up_proj": {"bits": 6, "group_size": 64, "mode": "affine"},
            "model.layers.2.mlp.up_proj": {"bits": 8, "group_size": 64, "mode": "affine"},
            "model.layers.3.mlp.up_proj": False,
        },
    }
    entry = {"compressed_format": {"scheme": "mlx_affine", "bits": 4, "mixed_bits": [4, 5, 6, 8]}}

    assert catalog._quantization(config, {}, entry, family="qwen3_5") == "4/5/6/8-bit"


@pytest.mark.parametrize("bits", [4, 8])
def test_qwen35_uniform_mlx_quantization_badge_keeps_existing_label(bits):
    config = {
        "model_type": "qwen3_5",
        "quantization": {"bits": bits, "group_size": 64, "mode": "affine"},
    }
    entry = {"compressed_format": {"scheme": "mlx_affine", "bits": bits}}

    assert catalog._quantization(config, {}, entry, family="qwen3_5") == f"{bits}-bit"


def test_qwen35_uniform_catalog_does_not_offer_mixed_width_source():
    config = {
        "quantization": {"bits": 4, "group_size": 64, "model.layers.0.mlp.up_proj": {"bits": 8, "group_size": 64}}
    }
    entry = {"compressed_format": {"scheme": "mlx_affine", "bits": 4}}
    assert catalog._quantization(config, {}, entry, family="qwen3_5") is None


@pytest.mark.parametrize("groups", [[], ["attn", "shared", "head", "embed"]])
def test_lily_qwen38_badge_includes_fixed_q8_projections_beyond_default_q4(groups):
    config = {
        "model_type": "qwen4_exp",
        "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
        "lily": {
            "format": "qwen4_exp-affine-v1",
            "quantization": {
                "default": {"bits": 4, "group_size": 64, "mode": "affine"},
                "ngram_embedding": {"bits": 4, "group_size": 32, "mode": "affine"},
                "q8_groups": groups,
            },
        },
    }
    assert catalog._quantization(config, {}, None) == "4-bit / 8-bit"


@pytest.mark.parametrize("discovery_fails", [False, True])
def test_known_lily_source_remains_available_without_hub_search_results(tmp_path, monkeypatch, discovery_fails):
    from test_qwen38 import fixture

    from stark_convert.hybrid_prepare import SOURCES

    source = SOURCES["qwen38-lily-q4-xl"]
    repo, revision = source["weights_repo"], source["weights_revision"]
    hub = Hub()
    config, _tensors = fixture(tmp_path / "synthetic-lily")
    config["architectures"] = ["Qwen4ExpForConditionalGeneration"]
    hub.add(
        tmp_path,
        repo,
        config,
        revision=revision,
    )
    monkeypatch.setattr(catalog, "_catalog_seeds", CATALOG_SEEDS)
    if discovery_fails:
        monkeypatch.setattr(hub, "list_models", lambda **_kwargs: (_ for _ in ()).throw(OSError()))
    report = catalog.family_catalog("qwen4_exp", hub=hub)
    assert report["status"] == "ready"
    identities = {(item["repo"], item["revision"]) for item in report["items"]}
    assert (repo, revision) in identities
    mixed = SOURCES["qwen38-mixed"]
    serve = SOURCES["qwen38-mlx-serve-mixed"]
    assert identities == {
        (repo, revision),
        (mixed["weights_repo"], mixed["weights_revision"]),
        (serve["weights_repo"], serve["weights_revision"]),
    }
    item = next(item for item in report["items"] if item["repo"] == repo)
    assert item["quantization"] == source["quantization"]
    from stark_convert.native_catalog import source_choice

    assert source_choice("qwen4_exp", item) == "qwen38-lily-q4-xl"
    assert ("info", repo, {"revision": revision, "files_metadata": True, "timeout": 5}) in hub.events
    assert [event[2] for event in hub.events if event[0] == "download"] == ["config.json"]


def test_known_lily_source_cannot_silently_move_to_another_revision(tmp_path, monkeypatch):
    from stark_convert.hybrid_prepare import SOURCES

    source = SOURCES["qwen38-lily-q4-xl"]
    hub = Hub()
    hub.add(tmp_path, source["weights_repo"], {"model_type": "qwen4_exp"}, revision=PIN)
    monkeypatch.setattr(catalog, "_catalog_seeds", CATALOG_SEEDS)
    report = catalog.family_catalog("qwen4_exp", hub=hub)
    assert not any(item["repo"] == source["weights_repo"] for item in report["items"])
    assert not [event for event in hub.events if event[0] == "download"]


def test_native_mlx_serve_choice_keeps_its_immutable_revision_when_hub_head_moves(tmp_path, monkeypatch):
    from stark_convert.hybrid_prepare import SOURCES

    source = SOURCES["qwen38-mlx-serve-mixed"]
    hub = Hub()
    hub.add(tmp_path, source["weights_repo"], {"model_type": "qwen4_exp"}, revision=PIN)
    # This fixed native choice does not depend on a Hub search or the main ref.
    monkeypatch.setattr(catalog, "_catalog_seeds", lambda _family: [source["weights_repo"]])
    report = catalog.family_catalog("qwen4_exp", hub=hub)
    assert report["status"] == "ready"
    assert [(item["repo"], item["revision"]) for item in report["items"]] == [
        (source["weights_repo"], source["weights_revision"])
    ]
    assert not hub.events or all(event[0] == "list" for event in hub.events)


def test_seed_declared_size_is_preserved_without_parsing_the_repository_name(tmp_path, monkeypatch):
    hub = Hub()
    hub.add(tmp_path, "known/No-size-in-name", {"model_type": "qwen3"})
    entry = {
        "adapter": "qwen3",
        "size": "0.6B",
        "files": {"model_config": {"repo": "known/No-size-in-name", "filename": "config.json"}},
    }
    monkeypatch.setattr(catalog, "_catalog_seeds", lambda _family: ["known/No-size-in-name"])
    monkeypatch.setattr(catalog, "get_model", lambda _repo: entry)
    item = catalog.family_catalog("qwen3", hub=hub)["items"][0]
    assert item["size_label"] == "0.6B"


def test_model_sizes_sort_numerically_instead_of_lexical_repository_order(tmp_path):
    hub = Hub()
    for name, size in (("Model-10B", 10**10), ("Model-2B", 2 * 10**9), ("Model-600M", 6 * 10**8)):
        hub.add(tmp_path, "someone/" + name, {"model_type": "qwen3"}, parameters={"BF16": size})
    hub.discovered["qwen3"] = list(hub.repos)
    assert [item["size_label"] for item in catalog.family_catalog("qwen3", hub=hub)["items"]] == ["600M", "2B", "10B"]


@pytest.mark.parametrize(
    "name,quantization,expected",
    [
        ("Qwen3-0.6B", "BF16", "qwen3"),
        ("Qwen3-1.7B-MLX-8bit", "8-bit", "qwen3"),
        ("Qwen3-4B-Instruct-2507-4bit", "4-bit", "qwen3-instruct-2507"),
        ("Qwen3-4B-Thinking-2507", "BF16", "qwen3-thinking-2507"),
        ("Qwen3-4B-v2-MLX-4bit", "4-bit", "qwen3-v2"),
        ("Qwen3-4B-MLX-4bit", "BF16", "qwen3-mlx-4bit"),
        ("Model-4B-Instruct-128K", "BF16", "model-instruct-128k"),
        ("Model-4B-Instruct-32K", "BF16", "model-instruct-32k"),
    ],
)
def test_model_group_stem_strips_only_matching_format_and_size(name, quantization, expected):
    assert catalog._variant_stem(name, quantization) == expected


def test_model_groups_span_sizes_but_keep_publishers_and_finetunes_separate(monkeypatch):
    monkeypatch.setattr(catalog, "models", lambda: ())
    low = catalog._model_group("qwen3", "Qwen/Qwen3-0.6B", "Qwen3-0.6B", "BF16", None)
    high = catalog._model_group("qwen3", "Qwen/Qwen3-1.7B-MLX-8bit", "Qwen3-1.7B-MLX-8bit", "8-bit", None)
    replica = catalog._model_group("qwen3", "PrimeIntellect/Qwen3-0.6B", "Qwen3-0.6B", "BF16", None)
    instruct = catalog._model_group("qwen3", "Qwen/Qwen3-4B-Instruct-2507", "Qwen3-4B-Instruct-2507", "BF16", None)
    assert low["id"] == high["id"] and low["name"] == high["name"] == "Qwen3"
    assert low["id"] != replica["id"] and replica["publisher"] == "PrimeIntellect"
    assert low["id"] != instruct["id"] and instruct["name"] == "Qwen3-Instruct-2507"


def test_registered_cross_publisher_quantization_joins_only_its_registered_base(monkeypatch):
    base = {"repo": "Qwen/Qwen3.5-2B", "adapter": "qwen3_5"}
    quant = {
        "repo": "mlx-community/Qwen3.5-2B-MLX-4bit",
        "adapter": "qwen3_5",
        "compressed_format": {"scheme": "mlx_affine", "bits": 4},
        "files": {"tokenizer_config": {"repo": "Qwen/Qwen3.5-2B"}},
    }
    stranger = {"repo": "other/Qwen3.5-2B", "adapter": "qwen3_5"}
    monkeypatch.setattr(catalog, "models", lambda: (base, quant))
    quant_group = catalog._model_group("qwen3_5", quant["repo"], "Qwen3.5-2B-MLX-4bit", "4-bit", quant)
    base_group = catalog._model_group("qwen3_5", base["repo"], "Qwen3.5-2B", "BF16", base)
    assert quant_group["id"] == base_group["id"]
    assert (
        catalog._model_group("qwen3_5", stranger["repo"], "Qwen3.5-2B", "BF16", None)["id"]
        != catalog._model_group("qwen3_5", base["repo"], "Qwen3.5-2B", "BF16", base)["id"]
    )


def test_bf16_entry_cannot_bridge_publishers_with_a_borrowed_tokenizer():
    base = {"repo": "Qwen/Qwen3.5-2B", "adapter": "qwen3_5"}
    borrowed = {
        "repo": "other/Qwen3.5-2B",
        "adapter": "qwen3_5",
        "files": {"tokenizer_config": {"repo": "Qwen/Qwen3.5-2B"}},
    }
    borrowed_group = catalog._model_group("qwen3_5", borrowed["repo"], "Qwen3.5-2B", "BF16", borrowed)
    base_group = catalog._model_group("qwen3_5", base["repo"], "Qwen3.5-2B", "BF16", base)
    assert borrowed_group["id"] != base_group["id"]


def test_all_numeric_model_name_has_nonempty_group_descriptor():
    group = catalog._model_group("qwen3", "owner/4B", "4B", "BF16", None)
    assert group["name"] == "4B"
    assert group["publisher"] == "owner"
    assert len(group["id"]) == 32


def test_group_normalization_strips_only_first_parameter_size_token(monkeypatch):
    monkeypatch.setattr(catalog, "models", lambda: ())
    four_b = catalog._model_group("llama", "owner/Llama-4B-Instruct-1M", "Llama-4B-Instruct-1M", "BF16", None)
    eight_b = catalog._model_group("llama", "owner/Llama-8B-Instruct-1M", "Llama-8B-Instruct-1M", "BF16", None)
    two_m = catalog._model_group("llama", "owner/Llama-8B-Instruct-2M", "Llama-8B-Instruct-2M", "BF16", None)
    assert four_b["id"] == eight_b["id"]
    assert four_b["name"] == "Llama-Instruct-1M"
    assert two_m["id"] != eight_b["id"]


def test_every_size_remains_reachable_across_pages(tmp_path):
    hub = Hub()
    for name, size in (("A-small", 10**9), ("B-small", 10**9), ("C-large", 10**10)):
        hub.add(tmp_path, "someone/" + name, {"model_type": "qwen3"}, parameters={"BF16": size})
    hub.discovered["qwen3"] = list(hub.repos)
    report = catalog.family_catalog("qwen3", hub=hub, limit=2)
    assert [item["size_label"] for item in report["items"]] == ["1B", "1B"]
    assert report["has_more"] is True and report["next_offset"] == 2
    final = catalog.family_catalog("qwen3", hub=hub, limit=2, offset=report["next_offset"])
    assert [item["size_label"] for item in final["items"]] == ["10B"]
    assert final["has_more"] is False and final["next_offset"] is None


def test_full_bf16_qwen35_defaults_to_images_after_own_processing_metadata_check(tmp_path, monkeypatch):
    from stark_convert import vision_family

    template = b"synthetic reviewed test template"
    import hashlib

    monkeypatch.setattr(vision_family, "_TEMPLATE_SHA256", hashlib.sha256(template).hexdigest())
    sidecars = {
        "preprocessor_config.json": vision_family._PROCESSOR,
        "chat_template.jinja": template,
        "tokenizer_config.json": {},
    }
    hub = Hub()
    hub.add(
        tmp_path,
        "someone/Dense-FineTune",
        geometry(),
        names={"model.safetensors": 10, "config.json": 100, **{key: 100 for key in sidecars}},
        sidecars=sidecars,
    )
    hub.discovered["qwen3_5"] = ["someone/Dense-FineTune"]
    item = catalog.family_catalog("qwen3_5", hub=hub)["items"][0]
    assert item["artifact_modes"] == ["text", "vision-language"]
    assert item["default_artifact_mode"] == "vision-language"
    assert {event[2] for event in hub.events if event[0] == "download"} == {
        "config.json",
        "preprocessor_config.json",
        "chat_template.jinja",
        "tokenizer_config.json",
    }


def test_qwen35_quantized_and_unknown_processing_sources_remain_text_only(tmp_path):
    hub = Hub()
    config = geometry()
    config["quantization"] = {"bits": 4, "group_size": 64}
    hub.add(tmp_path, "someone/Quantized", config, parameters={"U32": 100})
    hub.add(tmp_path, "someone/Unknown-processor", geometry())
    hub.discovered["qwen3_5"] = ["someone/Quantized", "someone/Unknown-processor"]
    items = catalog.family_catalog("qwen3_5", hub=hub)["items"]
    assert len(items) == 2 and all(item["artifact_modes"] == ["text"] for item in items)


@pytest.mark.parametrize("complete_processing", [False, True])
def test_bf16_qwen_moe_catalog_requires_image_metadata_and_never_offers_text(
    tmp_path, monkeypatch, complete_processing
):
    from test_vision_bf16_moe import moe_geometry

    from stark_convert import vision_family

    template = b"synthetic reviewed BF16 MoE template"
    monkeypatch.setattr(vision_family, "_TEMPLATE_SHA256", hashlib.sha256(template).hexdigest())
    sidecars = (
        {
            "preprocessor_config.json": vision_family._PROCESSOR,
            "chat_template.jinja": template,
            "tokenizer_config.json": {},
        }
        if complete_processing
        else {}
    )
    hub = Hub()
    repo = "someone/BF16-MoE"
    hub.add(
        tmp_path,
        repo,
        moe_geometry(),
        names={"model.safetensors": 70_000_000_000, "config.json": 100, **{key: 100 for key in sidecars}},
        sidecars=sidecars,
    )
    hub.discovered["qwen3_5_moe"] = [repo]
    items = catalog.family_catalog("qwen3_5_moe", hub=hub)["items"]
    if complete_processing:
        assert len(items) == 1 and items[0]["artifact_modes"] == ["vision-language"]
        assert items[0]["default_artifact_mode"] == "vision-language"
    else:
        assert len(items) == 1 and items[0]["downloadable"] is False
        assert "processing metadata" in items[0]["unavailable_reason"]
    assert not any(event[0] in {"head", "download"} and event[-1].endswith(".safetensors") for event in hub.events)


def test_oversized_config_stops_at_head_and_never_reaches_download(tmp_path):
    hub = Hub()
    root = hub.add(tmp_path, "someone/Large", {"model_type": "qwen3"})
    with (root / "config.json").open("wb") as stream:
        stream.truncate(catalog.MAX_CONFIG_BYTES + 1)
    hub.discovered["qwen3"] = ["someone/Large"]
    report = catalog.family_catalog("qwen3", hub=hub)
    assert report["status"] == "ready" and report["items"] == []
    assert not [event for event in hub.events if event[0] == "download"]


def test_missing_sizes_are_unknown_not_partial_sums(tmp_path):
    hub = Hub()
    hub.add(
        tmp_path,
        "someone/Sharded",
        {"model_type": "qwen3"},
        names={"one.safetensors": 100, "two.safetensors": None, "model.safetensors.index.json": 100},
        sidecars={"model.safetensors.index.json": {"weight_map": {"a": "one.safetensors", "b": "two.safetensors"}}},
    )
    hub.discovered["qwen3"] = ["someone/Sharded"]
    assert catalog.family_catalog("qwen3", hub=hub)["items"][0]["weight_bytes"] is None


@pytest.mark.parametrize("indexed", [False, True])
def test_download_bytes_follow_converter_selection_not_alternate_or_optimizer_files(tmp_path, indexed):
    hub = Hub()
    names = {"model.safetensors": 100, "optimizer.safetensors": 900}
    sidecars = {}
    if indexed:
        names = {
            "one.safetensors": 50,
            "two.safetensors": 75,
            "optimizer.safetensors": 900,
            "model.safetensors.index.json": 100,
        }
        sidecars = {"model.safetensors.index.json": {"weight_map": {"a": "one.safetensors", "b": "two.safetensors"}}}
    hub.add(tmp_path, "someone/Alternatives", {"model_type": "qwen3"}, names=names, sidecars=sidecars)
    hub.discovered["qwen3"] = ["someone/Alternatives"]
    item = catalog.family_catalog("qwen3", hub=hub)["items"][0]
    assert item["weight_bytes"] == (125 if indexed else 100)
    assert item["parameter_count"] is None


def test_multiple_unindexed_checkpoints_explain_why_download_is_unavailable(tmp_path):
    hub = Hub()
    hub.add(
        tmp_path, "someone/Ambiguous", {"model_type": "qwen3"}, names={"one.safetensors": 100, "two.safetensors": 200}
    )
    hub.discovered["qwen3"] = ["someone/Ambiguous"]
    items = catalog.family_catalog("qwen3", hub=hub)["items"]
    assert len(items) == 1 and items[0]["downloadable"] is False
    assert "shard index" in items[0]["unavailable_reason"]


def test_source_revision_is_required_and_stable_id_changes_with_it(tmp_path):
    hub = Hub()
    hub.add(tmp_path, "someone/Model", {"model_type": "qwen3"})
    hub.discovered["qwen3"] = ["someone/Model"]
    first = catalog.family_catalog("qwen3", hub=hub)["items"][0]
    hub.repos["someone/Model"][1].sha = "b" * 40
    second = catalog.family_catalog("qwen3", hub=hub)["items"][0]
    assert first["id"] != second["id"]
    hub.repos["someone/Model"][1].sha = "main"
    assert catalog.family_catalog("qwen3", hub=hub)["status"] == "error"


def test_exact_official_image_source_keeps_its_recorded_revision(tmp_path):
    hub = Hub()
    hub.add(tmp_path, catalog.EXACT_VISION_REPO, {"model_type": "qwen3_5"}, revision=catalog.EXACT_VISION_REVISION)
    hub.discovered["qwen3_5"] = [catalog.EXACT_VISION_REPO]
    item = catalog.family_catalog("qwen3_5", hub=hub)["items"][0]
    assert item["revision"] == catalog.EXACT_VISION_REVISION
    assert next(event[2] for event in hub.events if event[0] == "info")["revision"] == catalog.EXACT_VISION_REVISION


def test_result_and_candidate_counts_are_bounded_even_if_api_ignores_limit(tmp_path):
    hub = Hub()
    for index in range(60):
        repo = f"someone/Model-{index}"
        hub.add(tmp_path, repo, {"model_type": "qwen3"})
    hub.discovered["qwen3"] = list(hub.repos)
    report = catalog.family_catalog("qwen3", hub=hub, limit=3)
    assert len(report["items"]) == 3 and report["has_more"] is True
    assert len([event for event in hub.events if event[0] == "info"]) <= catalog.MAX_CANDIDATES
    assert len(json.dumps(report).encode()) < 256 * 1024


def test_pagination_reaches_compatible_models_beyond_the_old_search_ceiling(tmp_path):
    hub = Hub()
    for index in range(100):
        hub.add(tmp_path, f"someone/Model-{index}", {"model_type": "llama" if index < 60 else "qwen3"})
    hub.discovered["qwen3"] = list(hub.repos)
    offset, reports, items = 0, [], []
    while True:
        report = catalog.family_catalog("qwen3", hub=hub, offset=offset)
        reports.append(report)
        assert report["status"] == "ready" and report["offset"] == offset
        assert len(report["items"]) <= catalog.MAX_ITEMS
        items.extend(report["items"])
        if report["next_offset"] is None:
            break
        assert report["next_offset"] > offset
        offset = report["next_offset"]
    assert len(reports) == 4 and not reports[0]["items"] and not reports[1]["items"]
    assert {item["repo"] for item in items} == {f"someone/Model-{index}" for index in range(60, 100)}
    assert len(items) == 40
    assert len([event for event in hub.events if event[0] == "info"]) == 100
    assert not reports[-1]["has_more"]


def test_pagination_deduplicates_seeds_and_architecture_aliases_without_skipping_models(tmp_path, monkeypatch):
    hub = Hub()
    for name in ("A", "B", "C", "D"):
        hub.add(tmp_path, f"someone/{name}", {"model_type": "qwen3_5"})
    monkeypatch.setattr(catalog, "_catalog_seeds", lambda _family: ["someone/A"])
    hub.discovered["qwen3_5"] = ["someone/A", "someone/B", "someone/B"]
    hub.discovered["qwen3_5_text"] = ["someone/B", "someone/C", "someone/D"]
    first = catalog.family_catalog("qwen3_5", hub=hub, limit=2)
    second = catalog.family_catalog("qwen3_5", hub=hub, limit=2, offset=first["next_offset"])
    assert {item["repo"] for item in first["items"] + second["items"]} == set(hub.repos)
    assert len(first["items"] + second["items"]) == 4
    assert second["next_offset"] is None


@pytest.mark.parametrize("offset", [True, False, -1, 1.5, "1", 10_000_001])
def test_invalid_pagination_offset_never_uses_the_hub(offset):
    assert catalog.family_catalog("qwen3", hub=object(), offset=offset)["status"] == "error"


def _repository_not_found():
    response = SimpleNamespace(headers={}, request=SimpleNamespace())
    return RepositoryNotFoundError("repository unavailable", response=response)


def test_inaccessible_full_page_advances_to_later_compatible_models(tmp_path, monkeypatch):
    hub = Hub()
    for index in range(catalog.MAX_ITEMS):
        hub.add(tmp_path, f"someone/Hidden-{index}", {"model_type": "qwen3"})
    later = "someone/Visible-Later"
    hub.add(tmp_path, later, {"model_type": "qwen3"})
    hub.discovered["qwen3"] = list(hub.repos)
    model_info = hub.model_info

    def hidden_repo(repo_id, **kwargs):
        if repo_id != later:
            raise _repository_not_found()
        return model_info(repo_id=repo_id, **kwargs)

    monkeypatch.setattr(hub, "model_info", hidden_repo)
    first = catalog.family_catalog("qwen3", hub=hub)
    assert first["status"] == "ready" and first["items"] == []
    assert first["next_offset"] == catalog.MAX_ITEMS
    second = catalog.family_catalog("qwen3", hub=hub, offset=first["next_offset"])
    assert second["status"] == "ready"
    assert [item["repo"] for item in second["items"]] == [later]
    assert second["next_offset"] is None


@pytest.mark.parametrize("planning", [False, True])
def test_partial_transient_failure_retries_whole_page_without_advancing(tmp_path, monkeypatch, planning):
    hub = Hub()
    flaky, stable = "someone/Flaky", "someone/Stable"
    hub.add(tmp_path, flaky, {"model_type": "qwen3"})
    hub.add(tmp_path, stable, {"model_type": "qwen3"})
    hub.discovered["qwen3"] = [flaky, stable]
    model_info = hub.model_info
    failed = False

    def transient_once(repo_id, **kwargs):
        nonlocal failed
        if repo_id == flaky and not failed:
            failed = True
            raise RuntimeError("temporary Hub failure")
        return model_info(repo_id=repo_id, **kwargs)

    monkeypatch.setattr(hub, "model_info", transient_once)
    first = catalog.family_catalog("qwen3", hub=hub, planning=planning)
    if planning:
        assert first["status"] == "ready"
        assert [item["repo"] for item in first["items"]] == [stable]
        assert first["incomplete"] is True and first["partial_failure"] is True
        assert first["offset"] == 0 and first["next_offset"] == 0
        assert "temporary Hub failure" not in json.dumps(first)
    else:
        assert first["status"] == "error" and first["items"] == []
        assert first["offset"] == 0 and first["next_offset"] is None
    retried = catalog.family_catalog("qwen3", hub=hub, offset=first["offset"], planning=planning)
    assert retried["status"] == "ready"
    assert {item["repo"] for item in retried["items"]} == {flaky, stable}
    if planning:
        assert retried["incomplete"] is False and retried["partial_failure"] is False


def test_guided_planning_preserves_seed_when_source_discovery_fails(tmp_path, monkeypatch):
    hub = Hub()
    repo = "someone/Stable-seed"
    hub.add(tmp_path, repo, {"model_type": "qwen3"})
    monkeypatch.setattr(catalog, "_catalog_seeds", lambda _family: [repo])

    def unavailable(**_kwargs):
        raise RuntimeError("private Hub error")

    monkeypatch.setattr(hub, "list_models", unavailable)
    report = catalog.family_catalog("qwen3", hub=hub, planning=True)
    assert report["status"] == "ready"
    assert [item["repo"] for item in report["items"]] == [repo]
    assert report["incomplete"] is True and report["partial_failure"] is True
    assert report["next_offset"] == report["offset"] == 0
    assert "private Hub error" not in json.dumps(report)


@pytest.mark.parametrize("method", ["get_hf_file_metadata", "hf_hub_download"])
def test_missing_required_hub_metadata_file_is_an_incompatible_candidate(tmp_path, monkeypatch, method):
    hub = Hub()
    missing, available = "someone/Missing", "someone/Available"
    hub.add(tmp_path, missing, {"model_type": "qwen3"})
    hub.add(tmp_path, available, {"model_type": "qwen3"})
    hub.discovered["qwen3"] = [missing, available]
    original = getattr(hub, method)

    def missing_file(*args, **kwargs):
        if method == "get_hf_file_metadata":
            repo, filename, _revision = args[0]
        else:
            repo, filename = kwargs["repo_id"], kwargs["filename"]
        if repo == missing and filename == "config.json":
            raise EntryNotFoundError("required metadata file is missing")
        return original(*args, **kwargs)

    monkeypatch.setattr(hub, method, missing_file)
    report = catalog.family_catalog("qwen3", hub=hub)
    assert report["status"] == "ready"
    assert [item["repo"] for item in report["items"]] == [available]


def test_missing_config_repository_is_an_incompatible_candidate(tmp_path, monkeypatch):
    hub = Hub()
    repo = "someone/Model"
    hub.add(tmp_path, repo, {"model_type": "qwen3"})
    hub.discovered["qwen3"] = [repo]
    entry = {
        "adapter": "qwen3",
        "files": {"model_config": {"repo": "someone/Missing-Config", "filename": "config.json"}},
    }
    model_info = hub.model_info

    def missing_config(repo_id, **kwargs):
        if repo_id == "someone/Missing-Config":
            raise _repository_not_found()
        return model_info(repo_id=repo_id, **kwargs)

    monkeypatch.setattr(hub, "model_info", missing_config)
    monkeypatch.setattr(catalog, "get_model", lambda candidate: entry if candidate == repo else None)
    report = catalog.family_catalog("qwen3", hub=hub)
    assert report["status"] == "ready" and report["items"] == []


@pytest.mark.parametrize(
    ("filename", "invalid_payload"),
    [
        ("config.json", b"{"),
        ("config.json", b'{"model_type":"qwen3","model_type":"qwen3"}'),
        ("config.json", b'{"model_type":NaN}'),
        ("config.json", b"\xff"),
        ("config.json", b"[" * 1100 + b"0" + b"]" * 1100),
        ("model.safetensors.index.json", b"{"),
    ],
)
def test_malformed_candidate_metadata_does_not_block_compatible_rows(tmp_path, filename, invalid_payload):
    hub = Hub()
    malformed_repo, compatible_repo = "someone/Malformed", "someone/Compatible"
    config = {"model_type": "qwen3"}
    names = {"config.json": 128, "model.safetensors": 1234567890}
    sidecars = {}
    if filename == "config.json":
        sidecars[filename] = invalid_payload
    else:
        names[filename] = len(invalid_payload)
        sidecars[filename] = invalid_payload
    hub.add(tmp_path, malformed_repo, config, names=names, sidecars=sidecars)
    hub.add(tmp_path, compatible_repo, config)
    hub.discovered["qwen3"] = [malformed_repo, compatible_repo]

    report = catalog.family_catalog("qwen3", hub=hub)

    assert report["status"] == "ready"
    assert [item["repo"] for item in report["items"]] == [compatible_repo]


def test_full_malformed_page_can_continue_to_later_compatible_candidate(tmp_path):
    hub = Hub()
    malformed_repos = [f"someone/Malformed-{index}" for index in range(catalog.MAX_ITEMS)]
    for repo in malformed_repos:
        hub.add(tmp_path, repo, {"model_type": "qwen3"}, sidecars={"config.json": b"{"})
    compatible_repo = "someone/Compatible"
    hub.add(tmp_path, compatible_repo, {"model_type": "qwen3"})
    hub.discovered["qwen3"] = [*malformed_repos, compatible_repo]

    first = catalog.family_catalog("qwen3", hub=hub)
    second = catalog.family_catalog("qwen3", hub=hub, offset=first["next_offset"])

    assert first["status"] == "ready" and first["items"] == []
    assert first["next_offset"] == catalog.MAX_ITEMS
    assert second["status"] == "ready"
    assert [item["repo"] for item in second["items"]] == [compatible_repo]


@pytest.mark.parametrize("oversized_file", ["config.json", "model.safetensors.index.json"])
def test_oversized_candidate_metadata_does_not_block_compatible_rows(tmp_path, oversized_file):
    hub = Hub()
    oversized_repo, compatible_repo = "someone/Oversized", "someone/Compatible"
    config = {"model_type": "qwen3"}
    names = {"config.json": 128, "model.safetensors": 1234567890}
    sidecars = {}
    if oversized_file == "config.json":
        names["config.json"] = catalog.MAX_CONFIG_BYTES + 1
        sidecars["config.json"] = json.dumps(config).encode() + b" " * (catalog.MAX_CONFIG_BYTES + 1)
    else:
        names["model.safetensors.index.json"] = catalog.MAX_CONFIG_BYTES + 1
        sidecars["model.safetensors.index.json"] = b" " * (catalog.MAX_CONFIG_BYTES + 1)
    hub.add(tmp_path, oversized_repo, config, names=names, sidecars=sidecars)
    hub.add(tmp_path, compatible_repo, config)
    hub.discovered["qwen3"] = [oversized_repo, compatible_repo]

    report = catalog.family_catalog("qwen3", hub=hub)

    assert report["status"] == "ready"
    assert [item["repo"] for item in report["items"]] == [compatible_repo]
    assert not [
        event
        for event in hub.events
        if event[0] == "download" and event[1] == oversized_repo and event[2] == oversized_file
    ]


def test_offline_or_invalid_family_never_uses_the_hub():
    assert catalog.family_catalog("qwen3", offline=True, hub=object())["status"] == "offline"
    assert catalog.family_catalog("modernbert", hub=object())["status"] == "error"
    assert catalog.family_catalog("qwen3_moe", hub=object())["status"] == "error"
    assert catalog.family_catalog("qwen3", limit=31, hub=object())["status"] == "error"


def test_native_glm_catalog_preserves_fixed_pinned_item_alongside_discovery():
    from stark_convert import hybrid_prepare, native_catalog

    item = native_catalog.catalog_item()
    source = hybrid_prepare.SOURCES[native_catalog.CHOICE]
    assert item["repo"] == source["weights_repo"]
    assert item["revision"] == source["weights_revision"]
    assert item["quantization"] == source["quantization"]
    assert item["weight_bytes"] == source["weight_bytes"]
    assert item["artifact_modes"] == ["text"]
    assert item["size_label"] is None and item["parameter_count"] is None
    assert native_catalog.source_choice(native_catalog.FAMILY_ID, item) == native_catalog.CHOICE

    hub = Hub()
    report = catalog.family_catalog(native_catalog.FAMILY_ID, hub=hub)
    assert report["status"] == "ready"
    assert report["items"] == [item]
    assert report["items"][0]["artifact_modes"] == ["text"]
    assert hub.events and all(event[0] == "list" for event in hub.events)

    mutated = native_catalog.family_definition()
    mutated["modes"]["text"]["status"] = "qualified"
    assert native_catalog.family_definition()["modes"]["text"]["status"] == "conditional"
    returned = native_catalog.catalog_item()
    returned["repo"] = "attacker/other"
    assert native_catalog.catalog_item()["repo"] == source["weights_repo"]


def test_native_glm_catalog_obeys_offline_and_limit_without_hub_access():
    from stark_convert.native_catalog import FAMILY_ID

    class NoHub:
        def __getattr__(self, name):
            pytest.fail(f"native GLM catalog unexpectedly accessed hub.{name}")

    assert catalog.family_catalog(FAMILY_ID, offline=True, hub=NoHub())["status"] == "offline"
    assert catalog.family_catalog(FAMILY_ID, limit=0, hub=NoHub())["status"] == "error"
    assert catalog.family_catalog(FAMILY_ID, limit=catalog.MAX_ITEMS + 1, hub=NoHub())["status"] == "error"


def test_native_glm_source_choice_rejects_modified_entry():
    from stark_convert.native_catalog import CHOICE, FAMILY_ID, catalog_item, source_choice

    item = catalog_item()
    assert source_choice("qwen3", item) is None
    assert source_choice(FAMILY_ID, {**item, "revision": "0" * 40}) is None
    assert source_choice(FAMILY_ID, {**item, "artifact_modes": ["text", "vision-language"]}) is None
    assert source_choice(FAMILY_ID, item) == CHOICE


def test_network_errors_have_fixed_content_without_exception_details(monkeypatch):
    class Broken:
        def HfApi(self):
            raise RuntimeError("private credential exception sentinel")

    report = catalog.family_catalog("qwen3", hub=Broken())
    assert report["status"] == "error"
    assert "sentinel" not in json.dumps(report) and "credential" not in json.dumps(report)
