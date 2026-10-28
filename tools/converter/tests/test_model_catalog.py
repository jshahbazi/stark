"""Metadata-only family browsing never downloads model weight payloads."""

import json
from types import SimpleNamespace

import pytest
from test_vision_family_geometry import geometry

from stark_convert import model_catalog as catalog

PIN = "a" * 40


@pytest.fixture(autouse=True)
def clean_offline_environment(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(catalog, "_catalog_seeds", lambda _family: [])
    monkeypatch.setattr(catalog, "get_model", lambda _repo: None)


class Hub:
    def __init__(self):
        self.repos = {}
        self.discovered = {}
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
        )
        self.repos[repo] = root, info
        return root

    def HfApi(self):
        return self

    def list_models(self, *, filter, **kwargs):
        self.events.append(("list", filter, kwargs))
        return [SimpleNamespace(id=repo) for repo in self.discovered.get(filter, [])]

    def model_info(self, *, repo_id, **kwargs):
        self.events.append(("info", repo_id, kwargs))
        return self.repos[repo_id][1]

    def hf_hub_url(self, *, repo_id, filename, revision):
        return repo_id, filename, revision

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
    assert report["status"] == "ready" and report["items"] == []


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


def test_bounded_page_keeps_larger_sizes_instead_of_only_the_smallest_models(tmp_path):
    hub = Hub()
    for name, size in (("A-small", 10**9), ("B-small", 10**9), ("C-large", 10**10)):
        hub.add(tmp_path, "someone/" + name, {"model_type": "qwen3"}, parameters={"BF16": size})
    hub.discovered["qwen3"] = list(hub.repos)
    report = catalog.family_catalog("qwen3", hub=hub, limit=2)
    assert [item["size_label"] for item in report["items"]] == ["1B", "10B"]
    assert report["has_more"] is True


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


def test_oversized_config_stops_at_head_and_never_reaches_download(tmp_path):
    hub = Hub()
    root = hub.add(tmp_path, "someone/Large", {"model_type": "qwen3"})
    with (root / "config.json").open("wb") as stream:
        stream.truncate(catalog.MAX_CONFIG_BYTES + 1)
    hub.discovered["qwen3"] = ["someone/Large"]
    report = catalog.family_catalog("qwen3", hub=hub)
    assert report["status"] == "error" and report["items"] == []
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


def test_multiple_unindexed_checkpoints_are_not_offered(tmp_path):
    hub = Hub()
    hub.add(
        tmp_path, "someone/Ambiguous", {"model_type": "qwen3"}, names={"one.safetensors": 100, "two.safetensors": 200}
    )
    hub.discovered["qwen3"] = ["someone/Ambiguous"]
    assert catalog.family_catalog("qwen3", hub=hub)["items"] == []


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


def test_offline_or_invalid_family_never_uses_the_hub():
    assert catalog.family_catalog("qwen3", offline=True, hub=object())["status"] == "offline"
    assert catalog.family_catalog("modernbert", hub=object())["status"] == "error"
    assert catalog.family_catalog("qwen3_moe", hub=object())["status"] == "error"
    assert catalog.family_catalog("qwen3", limit=31, hub=object())["status"] == "error"


def test_network_errors_have_fixed_content_without_exception_details(monkeypatch):
    class Broken:
        def HfApi(self):
            raise RuntimeError("private credential exception sentinel")

    report = catalog.family_catalog("qwen3", hub=Broken())
    assert report["status"] == "error"
    assert "sentinel" not in json.dumps(report) and "credential" not in json.dumps(report)
