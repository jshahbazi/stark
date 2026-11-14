"""A foreign-format export names the repository it says it was made from; nothing else changes."""

import io

import pytest
from test_desktop_model_browser import report as page_report
from test_model_catalog import Hub

from stark_convert import catalog_reasons, hub_headers, planning
from stark_convert import model_catalog as catalog
from stark_convert.catalog_cache import VerdictCache
from stark_convert.desktop import Desktop

AWQ = {"model_type": "qwen3", "quantization_config": {"quant_method": "awq", "bits": 4, "group_size": 128}}
BF16 = {"model_type": "qwen3", "torch_dtype": "bfloat16"}
MLX = {"model_type": "qwen3", "quantization": {"bits": 4, "group_size": 64}}
DECLARED = "base_model:quantized:"


@pytest.fixture(autouse=True)
def isolated_catalog(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(catalog, "_catalog_seeds", lambda _family: [])
    monkeypatch.setattr(catalog, "get_model", lambda _repo: None)
    monkeypatch.setattr(hub_headers, "checkpoint_headers", lambda *args: {})
    monkeypatch.setattr(planning, "plan_from_headers", lambda *args, **kwargs: None)


def rows(hub, tmp_path, sources, **options):
    for repo, (config, tags) in sources.items():
        hub.add(tmp_path, repo, config, tags=tags)
    hub.discovered["qwen3"] = list(sources)
    report = catalog.family_catalog("qwen3", hub=hub, **options)
    assert report["status"] == "ready", report["message"]
    return {item["repo"]: item for item in report["items"]}


@pytest.mark.parametrize(
    "tags,expected",
    [
        ([DECLARED + "maker/Model-7B"], "maker/Model-7B"),
        (["base_model:maker/Model-7B", DECLARED + "maker/Model-7B", "text-generation"], "maker/Model-7B"),
        # A repeated tag is still one declaration.
        ([DECLARED + "maker/Model-7B", DECLARED + "maker/Model-7B"], "maker/Model-7B"),
        ([], None),
        (["base_model:maker/Model-7B", "base_model:finetune:maker/Model-7B"], None),
        # Two different originals cannot both be the one to show instead.
        ([DECLARED + "maker/Model-7B", DECLARED + "other/Model-7B"], None),
        ([DECLARED + "someone/model-awq"], None),
        ([DECLARED + "Model-7B"], None),
        ([DECLARED + "maker/Model 7B"], None),
        ([DECLARED + "maker/Model/extra"], None),
        ([DECLARED + "https://example.com/maker/Model"], None),
        ([DECLARED + "maker/Model\n7B"], None),
        ([DECLARED + "maker/" + "x" * 97], None),
        ([DECLARED], None),
        ([7, None, {"tag": DECLARED + "maker/Model-7B"}], None),
    ],
)
def test_declared_original_needs_exactly_one_other_valid_repository(tags, expected):
    assert catalog_reasons.declared_original("someone/Model-AWQ", tags) == expected


@pytest.mark.parametrize("tags", [None, "base_model:quantized:maker/Model-7B", 7, {"a": 1}])
def test_declared_original_ignores_malformed_tag_lists(tags):
    assert catalog_reasons.declared_original("someone/Model-AWQ", tags) is None
    assert catalog_reasons.declared_original(None, [DECLARED + "maker/Model-7B"]) is None


def test_foreign_row_carries_its_declared_original(tmp_path):
    items = rows(
        Hub(),
        tmp_path,
        {
            "maker/Model-7B": (BF16, []),
            "someone/Model-7B-AWQ": (AWQ, [DECLARED + "maker/Model-7B"]),
            "someone/Untagged-AWQ": (AWQ, ["text-generation"]),
        },
    )
    export = items["someone/Model-7B-AWQ"]
    assert export["downloadable"] is False and export["quantization"] == "AWQ"
    assert export["original_repo"] == "maker/Model-7B"
    # The pointer changes neither the reason nor what any row can do.
    assert export["unavailable_reason"] == catalog_reasons.foreign_format(AWQ)[1]
    assert "original_repo" not in items["someone/Untagged-AWQ"]
    assert items["maker/Model-7B"]["downloadable"] is True and "original_repo" not in items["maker/Model-7B"]


def test_only_foreign_rows_point_at_an_original(tmp_path):
    # A supported MLX export declares the same relation and stays an ordinary row.
    items = rows(
        Hub(),
        tmp_path,
        {
            "maker/Model-7B": (BF16, []),
            "someone/Model-7B-4bit": (MLX, [DECLARED + "maker/Model-7B"]),
        },
    )
    assert items["someone/Model-7B-4bit"]["downloadable"] is True
    assert all("original_repo" not in item for item in items.values())


def test_pointer_survives_the_verdict_cache(tmp_path):
    sources = {"someone/Model-7B-AWQ": (AWQ, [DECLARED + "maker/Model-7B"])}
    cache_dir = tmp_path / "cache"
    first = rows(Hub(), tmp_path / "first", sources, cache_dir=cache_dir)
    second = rows(Hub(), tmp_path / "second", sources, cache_dir=cache_dir)
    assert first == second
    assert second["someone/Model-7B-AWQ"]["original_repo"] == "maker/Model-7B"
    found, stored = VerdictCache(cache_dir, "qwen3").load_recent("someone/Model-7B-AWQ")
    assert found and stored["original_repo"] == "maker/Model-7B"


@pytest.fixture
def host(tmp_path):
    value = Desktop(tmp_path / "unused-worker", tmp_path / "library", io.StringIO())
    yield value
    value.stop(quiet=True)


def unavailable_page(**changes):
    page = page_report()
    page["items"][0].update(downloadable=False, unavailable_reason="Fixture reason", **changes)
    return page


def test_backend_passes_a_well_formed_pointer(host):
    page = unavailable_page(original_repo="maker/Model-7B")
    assert host.catalog_report(page, "qwen3")["items"][0]["original_repo"] == "maker/Model-7B"
    assert "original_repo" not in host.catalog_report(unavailable_page(), "qwen3")["items"][0]


@pytest.mark.parametrize(
    "value",
    [
        "",
        "Model-7B",
        "maker/Model 7B",
        "maker/Model/extra",
        "https://huggingface.co/maker/Model-7B",
        "maker/Model-7B\n",
        "maker/" + "x" * 97,
        7,
        ["maker/Model-7B"],
        {"repo": "maker/Model-7B"},
    ],
)
def test_backend_rejects_a_malformed_pointer(host, value):
    with pytest.raises(ValueError):
        host.catalog_report(unavailable_page(original_repo=value), "qwen3")


def test_backend_rejects_a_pointer_on_a_downloadable_row_or_at_itself(host):
    page = page_report()
    page["items"][0]["original_repo"] = "maker/Model-7B"
    with pytest.raises(ValueError):
        host.catalog_report(page, "qwen3")
    own = unavailable_page()
    own["items"][0]["original_repo"] = own["items"][0]["repo"].upper()
    with pytest.raises(ValueError):
        host.catalog_report(own, "qwen3")


def test_declared_original_is_never_requested(tmp_path):
    # The declaration is display metadata: no request is made for the named repository.
    hub = Hub()
    items = rows(hub, tmp_path, {"someone/Model-7B-AWQ": (AWQ, [DECLARED + "absent/Model-7B"])})
    assert items["someone/Model-7B-AWQ"]["original_repo"] == "absent/Model-7B"
    assert items["someone/Model-7B-AWQ"]["downloadable"] is False
    assert all("absent/Model-7B" not in map(str, event) for event in hub.events)
    assert [event[1] for event in hub.events if event[0] == "info"] == ["someone/Model-7B-AWQ"]
