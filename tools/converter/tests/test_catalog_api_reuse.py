"""Catalogue verdict reuse, Hub API screening and header reads, and streamed rows."""

import io
import json
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from test_catalog_tensor_headers import add_checkpoint
from test_desktop_model_browser import report as page_report
from test_model_catalog import PIN, Hub
from test_qwen38_external import RangeHub
from test_streaming import qwen_fixture

from stark_convert import catalog_cache, cli, hub_headers, qwen38_external
from stark_convert import model_catalog as catalog
from stark_convert.desktop import Desktop
from stark_convert.streaming import ConversionError

ENDPOINT = "https://hub.test"


@pytest.fixture(autouse=True)
def isolated_catalog(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(catalog, "_catalog_seeds", lambda _family: [])
    monkeypatch.setattr(catalog, "get_model", lambda _repo: None)


@pytest.fixture
def skip_headers(monkeypatch):
    monkeypatch.setattr(catalog, "_verify_hub_layout", lambda *args: dict.fromkeys(args[-1]))


def requests(hub, *kinds):
    return [event for event in hub.events if event[0] in kinds]


class ListingHub(Hub):
    """A Hub whose listing carries the fields requested with ``expand``."""

    def __init__(self):
        super().__init__()
        self.listing = {}

    def list_models(self, *, filter=None, search=None, **kwargs):
        self.events.append(("list", filter or search, kwargs))
        results = self.searched.get(search, []) if search is not None else self.discovered.get(filter, [])
        return [SimpleNamespace(id=repo, **self.listing.get(repo, {})) for repo in results]


def listed(names, *, tags=(), pipeline_tag=None, sha=PIN):
    return {
        "sha": sha,
        "tags": list(tags),
        "pipeline_tag": pipeline_tag,
        "siblings": [SimpleNamespace(rfilename=name) for name in names],
    }


def test_cached_verdict_for_a_listed_revision_needs_no_repository_request(tmp_path, skip_headers):
    hub = ListingHub()
    hub.add(tmp_path / "hub", "someone/Stable", {"model_type": "qwen3", "torch_dtype": "float16"})
    hub.discovered["qwen3"] = ["someone/Stable"]
    hub.listing["someone/Stable"] = listed(["config.json", "model.safetensors"])
    cache = tmp_path / "cache"

    first = catalog.family_catalog("qwen3", hub=hub, cache_dir=cache)
    assert len(first["items"]) == 1 and len(requests(hub, "info", "download")) == 2
    hub.events.clear()
    second = catalog.family_catalog("qwen3", hub=hub, cache_dir=cache)

    assert second == first
    assert not requests(hub, "info", "head", "download", "cache")
    # Without a cache directory nothing is reused.
    catalog.family_catalog("qwen3", hub=hub)
    assert len(requests(hub, "info")) == 1


def age_head_confirmations(cache):
    """Make every stored head revision look last confirmed too long ago to trust."""
    for path in cache.rglob("*.json"):
        entry = json.loads(path.read_bytes())
        entry["confirmed"] -= catalog_cache.HEAD_SECONDS
        path.write_text(json.dumps(entry))


def test_unlisted_revision_is_trusted_briefly_then_confirmed_with_one_request(tmp_path, skip_headers):
    hub = Hub()
    hub.add(tmp_path / "hub", "someone/Seed", {"model_type": "qwen3", "torch_dtype": "float16"})
    hub.discovered["qwen3"] = ["someone/Seed"]
    cache = tmp_path / "cache"

    first = catalog.family_catalog("qwen3", hub=hub, cache_dir=cache)
    hub.events.clear()
    assert catalog.family_catalog("qwen3", hub=hub, cache_dir=cache) == first and len(first["items"]) == 1
    assert not requests(hub, "info", "head", "download", "cache")

    age_head_confirmations(cache)
    assert catalog.family_catalog("qwen3", hub=hub, cache_dir=cache) == first
    assert [event[:2] for event in requests(hub, "info")] == [("info", "someone/Seed")]
    assert not requests(hub, "head", "download", "cache")
    # Confirming the head renews the short trust.
    hub.events.clear()
    assert catalog.family_catalog("qwen3", hub=hub, cache_dir=cache) == first and not requests(hub, "info")


def test_changed_revision_or_converter_build_rechecks_the_source(tmp_path, skip_headers, monkeypatch):
    hub = Hub()
    hub.add(tmp_path / "hub", "someone/Moves", {"model_type": "qwen3", "torch_dtype": "float16"})
    hub.discovered["qwen3"] = ["someone/Moves"]
    cache = tmp_path / "cache"
    catalog.family_catalog("qwen3", hub=hub, cache_dir=cache)

    hub.events.clear()
    age_head_confirmations(cache)
    hub.repos["someone/Moves"][1].sha = "b" * 40
    moved = catalog.family_catalog("qwen3", hub=hub, cache_dir=cache)
    assert moved["items"][0]["revision"] == "b" * 40 and len(requests(hub, "download")) == 1

    hub.events.clear()
    monkeypatch.setattr(catalog_cache, "converter_fingerprint", lambda: "another build")
    rebuilt = catalog.family_catalog("qwen3", hub=hub, cache_dir=cache)
    assert rebuilt["items"] == moved["items"] and len(requests(hub, "download")) == 1


def test_rejections_are_remembered_and_access_failures_expire_quickly(tmp_path, skip_headers):
    hub = Hub()
    hub.add(tmp_path / "hub", "someone/Other", {"model_type": "llama"})
    hub.discovered["qwen3"] = ["someone/Other"]
    cache = tmp_path / "cache"
    assert catalog.family_catalog("qwen3", hub=hub, cache_dir=cache)["items"] == []
    hub.events.clear()
    assert catalog.family_catalog("qwen3", hub=hub, cache_dir=cache)["items"] == []
    assert not requests(hub, "download", "head")

    now = [1_000_000.0]
    store = catalog_cache.VerdictCache(cache, "qwen3", fingerprint="build", clock=lambda: now[0])
    store.store("someone/Gated", PIN, None, settled=False)
    store.store("someone/Settled", PIN, None)
    assert store.load("someone/Gated", PIN) == (True, None)
    assert store.load("someone/Gated", "c" * 40) == (False, None)
    now[0] += catalog_cache.VOLATILE_SECONDS
    assert store.load("someone/Gated", PIN) == (False, None)
    assert store.load("someone/Settled", PIN) == (True, None)
    now[0] += catalog_cache.SETTLED_SECONDS
    assert store.load("someone/Settled", PIN) == (False, None)


def test_unverified_header_verdict_is_checked_again(tmp_path, monkeypatch):
    hub = Hub()
    hub.add(tmp_path / "hub", "someone/Flaky", {"model_type": "qwen3", "torch_dtype": "float16"})
    hub.discovered["qwen3"] = ["someone/Flaky"]
    checks = []

    def unverified(*args):
        checks.append(args[2])
        return dict.fromkeys(args[-1], qwen38_external.UNVERIFIED)

    monkeypatch.setattr(catalog, "_verify_hub_layout", unverified)
    for _ in range(2):
        (item,) = catalog.family_catalog("qwen3", hub=hub, cache_dir=tmp_path / "cache")["items"]
        assert item["unavailable_reason"] == qwen38_external.UNVERIFIED
    assert checks == ["someone/Flaky", "someone/Flaky"]


def test_damaged_cache_entries_are_ignored(tmp_path):
    store = catalog_cache.VerdictCache(tmp_path, "qwen3", fingerprint="build")
    store.store("someone/Model", PIN, {"repo": "someone/Model"})
    assert store.load("someone/Model", PIN) == (True, {"repo": "someone/Model"})
    store._path("someone/Model").write_bytes(b"{not json")
    assert store.load("someone/Model", PIN) == (False, None)
    store._path("someone/Model").write_bytes(b"[]")
    assert store.load("someone/Model", PIN) == (False, None)
    assert store.load("someone/Never", PIN) == (False, None)


def test_listing_fields_reject_sources_without_any_repository_request(tmp_path, skip_headers):
    hub = ListingHub()
    config = {"model_type": "qwen3", "torch_dtype": "float16"}
    rejected = {
        "someone/Embedder": listed(["config.json", "model.safetensors"], pipeline_tag="feature-extraction"),
        "someone/Reranker": listed(["config.json", "model.safetensors"], tags=["text-ranking"]),
        "someone/NoConfig": listed(["model.gguf", "README.md"]),
        "someone/NoWeights": listed(["config.json", "adapter_model.safetensors"]),
    }
    for repo, listing in rejected.items():
        hub.add(tmp_path / "hub", repo, config)
        hub.listing[repo] = listing
    hub.add(tmp_path / "hub", "someone/Kept", config)
    hub.listing["someone/Kept"] = listed(["config.json", "model.safetensors"], pipeline_tag="text-generation")
    hub.add(tmp_path / "hub", "someone/Unlisted", config)
    hub.discovered["qwen3"] = [*rejected, "someone/Kept", "someone/Unlisted"]

    result = catalog.family_catalog("qwen3", hub=hub)

    assert sorted(item["repo"] for item in result["items"]) == ["someone/Kept", "someone/Unlisted"]
    assert sorted(event[1] for event in requests(hub, "info")) == ["someone/Kept", "someone/Unlisted"]
    assert result["next_offset"] is None
    listings = requests(hub, "list")
    assert listings and all(event[2]["expand"] == ["sha", "pipeline_tag", "tags", "siblings"] for event in listings)


def test_screened_sources_use_positions_so_a_page_reaches_further(tmp_path, skip_headers, monkeypatch):
    hub = ListingHub()
    config = {"model_type": "qwen3", "torch_dtype": "float16"}
    order = []
    for index in range(6):
        skipped, kept = f"someone/Skip{index}", f"someone/Keep{index}"
        hub.add(tmp_path / "hub", kept, config)
        hub.listing[skipped] = listed(["config.json", "model.safetensors"], pipeline_tag="feature-extraction")
        hub.listing[kept] = listed(["config.json", "model.safetensors"])
        order.extend([skipped, kept])
    hub.discovered["qwen3"] = order

    first = catalog.family_catalog("qwen3", hub=hub, limit=2)
    assert sorted(item["repo"] for item in first["items"]) == ["someone/Keep0", "someone/Keep1"]
    assert first["next_offset"] == 4 and first["has_more"] is True
    second = catalog.family_catalog("qwen3", hub=hub, limit=2, offset=first["next_offset"])
    assert sorted(item["repo"] for item in second["items"]) == ["someone/Keep2", "someone/Keep3"]
    assert second["next_offset"] == 8

    # A long run of screened sources ends the page with a continuation instead of walking on.
    monkeypatch.setattr(catalog, "MAX_SCREENED", 1)
    hub.events.clear()
    hub.discovered["qwen3"] = [f"someone/Skip{index}" for index in range(6)] + ["someone/Keep0"]
    empty = catalog.family_catalog("qwen3", hub=hub, limit=2)
    assert empty["status"] == "ready" and empty["items"] == [] and empty["next_offset"] == 3
    assert not requests(hub, "info")


def test_hub_parsed_architecture_rejects_before_reading_the_config(tmp_path, skip_headers):
    hub = Hub()
    hub.add(tmp_path / "hub", "someone/Classifier", {"model_type": "qwen3"})
    hub.add(tmp_path / "hub", "someone/Missing", {"model_type": "qwen3"}, names={"model.safetensors": 10})
    hub.repos["someone/Classifier"][1].config = {"architectures": ["Qwen3ForSequenceClassification"]}
    # An inventory really without the configuration needs no request to say so.
    hub.repos["someone/Missing"][1].siblings = [SimpleNamespace(rfilename="model.safetensors", size=10)]
    hub.discovered["qwen3"] = ["someone/Classifier", "someone/Missing"]

    assert catalog.family_catalog("qwen3", hub=hub)["items"] == []
    assert not requests(hub, "head", "download", "cache")


class ApiHub(RangeHub):
    """A range Hub that also serves the parsed shard headers of indexed checkpoints."""

    def __init__(self, source):
        super().__init__()
        self.constants = SimpleNamespace(ENDPOINT=ENDPOINT)
        self.source = source
        self.described = PIN
        self.status = 200
        self.rewrite = lambda headers: headers

    @contextmanager
    def stream(self, method, url, *, headers, timeout, follow_redirects):
        if not isinstance(url, str):
            with super().stream(
                method, url, headers=headers, timeout=timeout, follow_redirects=follow_redirects
            ) as response:
                yield response
            return
        repo = url.removeprefix(ENDPOINT + "/api/models/").removesuffix("/safetensors-metadata/" + hub_headers.INDEX)
        assert method == "GET" and repo in self.repos
        self.events.append(("api", repo))
        parsed = {}
        for path in self.source.glob("*.safetensors"):
            data = path.read_bytes()
            parsed[path.name] = json.loads(data[8 : 8 + int.from_bytes(data[:8], "little")])
        body = json.dumps({"sharded": True, "headers": self.rewrite(parsed)}).encode()
        yield SimpleNamespace(
            status_code=self.status, headers={"etag": f'"{self.described}"'}, iter_bytes=lambda: iter([body])
        )


def sharded_checkpoint(tmp_path):
    source = tmp_path / "source"
    qwen_fixture(source, tied=True, sharded=True)
    hub = ApiHub(source)
    files = {path.name: path.read_bytes() for path in source.iterdir() if path.is_file()}
    hub.add(
        tmp_path / "hub",
        "example/Model",
        json.loads(files["config.json"]),
        names={name: len(data) for name, data in files.items()},
        sidecars={name: data for name, data in files.items() if name != "config.json"},
    )
    hub.discovered["qwen3"] = ["example/Model"]
    return hub


def test_indexed_checkpoint_headers_come_from_one_hub_api_call(tmp_path):
    hub = sharded_checkpoint(tmp_path)

    result = catalog.family_catalog("qwen3", hub=hub)

    assert len(result["items"]) == 1 and result["items"][0]["downloadable"] is True
    assert requests(hub, "api") == [("api", "example/Model")] and not requests(hub, "range")


def test_api_headers_match_the_exact_ranged_headers(tmp_path):
    hub = sharded_checkpoint(tmp_path)
    shards = sorted(path.name for path in hub.source.glob("*.safetensors"))
    sizes = {name: (hub.source / name).stat().st_size for name in shards}

    parsed = hub_headers.indexed_headers(hub, "example/Model", PIN, shards, sizes, lambda: None)
    ranged = {
        name: hub_headers.read_header(hub, "example/Model", PIN, name, lambda: None, sizes[name]) for name in shards
    }

    assert parsed == ranged


@pytest.mark.parametrize("case", ["other_revision", "error_status", "missing_shard", "bad_offsets", "unknown_size"])
def test_unusable_api_reply_falls_back_to_exact_header_ranges(tmp_path, case):
    hub = sharded_checkpoint(tmp_path)
    if case == "other_revision":
        hub.described = "b" * 40
    elif case == "error_status":
        hub.status = 422
    elif case == "missing_shard":
        hub.rewrite = lambda headers: dict(list(headers.items())[:1])
    elif case == "bad_offsets":

        def shifted(headers):
            for header in headers.values():
                for name, info in header.items():
                    if name != "__metadata__":
                        info["data_offsets"] = [info["data_offsets"][0] + 1, info["data_offsets"][1] + 1]
            return headers

        hub.rewrite = shifted
    else:
        for sibling in hub.repos["example/Model"][1].siblings:
            if sibling.rfilename.endswith(".safetensors"):
                sibling.size = None

    result = catalog.family_catalog("qwen3", hub=hub)

    assert len(result["items"]) == 1 and result["items"][0]["downloadable"] is True
    assert len(requests(hub, "range")) == 4
    assert len(requests(hub, "api")) == (0 if case == "unknown_size" else 1)


def test_single_file_checkpoint_does_not_ask_the_index_api(tmp_path):
    source = tmp_path / "source"
    qwen_fixture(source, tied=True)
    hub, repo, _ = add_checkpoint(tmp_path, source)
    hub.constants = SimpleNamespace(ENDPOINT=ENDPOINT)

    result = catalog.family_catalog("qwen3", hub=hub)

    assert result["items"][0]["downloadable"] is True
    assert len(requests(hub, "range")) == 2


class RedirectingHub:
    """One pinned file whose Hub URL redirects to signed storage."""

    def __init__(self, data, *, storage_fails=False):
        self.data, self.storage_fails, self.calls = data, storage_fails, []
        self.utils = SimpleNamespace(
            build_hf_headers=lambda: {"authorization": "Bearer hub-only"}, get_session=lambda: self
        )

    def hf_hub_url(self, *, repo_id, filename, revision):
        return f"https://hub.test/{repo_id}/resolve/{revision}/{filename}"

    @contextmanager
    def stream(self, method, url, *, headers, timeout, follow_redirects):
        storage = url.startswith("https://storage.test/")
        self.calls.append(("storage" if storage else "hub", headers["Range"], "authorization" in headers))
        assert follow_redirects is not storage
        start, end = map(int, headers["Range"].removeprefix("bytes=").split("-"))
        yield SimpleNamespace(
            status_code=403 if storage and self.storage_fails else 206,
            headers={"content-range": f"bytes {start}-{end}/{len(self.data)}"},
            iter_raw=lambda chunk_size: iter([self.data[start : end + 1]]),
            history=[] if storage else [object()],
            url="https://storage.test/blob?signature=1",
        )


def safetensors_bytes():
    header = json.dumps({"weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}).encode()
    return len(header).to_bytes(8, "little") + header + bytes(8)


@pytest.mark.parametrize("storage_fails", [False, True])
def test_header_body_reuses_the_resolved_storage_location_without_hub_credentials(storage_fails):
    data = safetensors_bytes()
    hub = RedirectingHub(data, storage_fails=storage_fails)
    body = f"bytes=8-{len(data) - 9}"

    tensors = hub_headers.read_header(hub, "owner/Model", PIN, "model.safetensors", lambda: None, len(data))

    assert list(tensors) == ["weight"] and tensors["weight"].shape == (2,)
    expected = [("hub", "bytes=0-7", True), ("storage", body, False)]
    assert hub.calls == (expected + [("hub", body, True)] if storage_fails else expected)


def test_header_from_storage_is_still_bound_to_the_pinned_size():
    data = safetensors_bytes()
    with pytest.raises(hub_headers.HeaderAccessError):
        hub_headers.read_header(RedirectingHub(data), "owner/Model", PIN, "model.safetensors", lambda: None, 1)
    with pytest.raises(ConversionError):
        truncated = (len(data)).to_bytes(8, "little") + data[8:]
        hub_headers.read_header(RedirectingHub(truncated), "owner/Model", PIN, "model.safetensors", lambda: None)


def test_rows_are_reported_as_each_source_is_checked(tmp_path, skip_headers):
    hub = Hub()
    for name in ("One", "Two", "Other"):
        config = {"model_type": "llama" if name == "Other" else "qwen3", "torch_dtype": "float16"}
        hub.add(tmp_path / "hub", "someone/" + name, config)
    hub.discovered["qwen3"] = ["someone/One", "someone/Other", "someone/Two"]
    seen = []

    result = catalog.family_catalog("qwen3", hub=hub, on_item=seen.append)

    assert sorted(item["repo"] for item in seen) == ["someone/One", "someone/Two"]
    assert sorted(seen, key=lambda item: item["repo"]) == sorted(result["items"], key=lambda item: item["repo"])


def test_cli_streams_rows_before_the_report_and_passes_the_cache_directory(tmp_path, monkeypatch, capsys):
    called = {}

    def family_catalog(family, **options):
        called.update(options, family=family)
        for item in page_report()["items"]:
            options["on_item"](item)
        return page_report()

    monkeypatch.setattr(catalog, "family_catalog", family_catalog)
    arguments = ["stark-convert", "--family-models", "qwen3", "--catalog-stream", "--catalog-cache-dir", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", arguments)
    assert cli.main() == 0

    partial, final = map(json.loads, capsys.readouterr().out.splitlines())
    assert partial == {"partial_items": page_report()["items"]} and final == page_report()
    assert called["family"] == "qwen3" and called["cache_dir"] == tmp_path and called["offset"] == 0

    monkeypatch.setattr(sys, "argv", ["stark-convert", "--catalog-stream", "--list-families"])
    assert cli.main() != 0


@pytest.fixture
def host(tmp_path):
    value = Desktop(tmp_path / "unused-worker", tmp_path / "library", io.StringIO())
    yield value
    value.stop(quiet=True)


def emitted(host):
    return [json.loads(line) for line in host.output.getvalue().splitlines()]


def test_desktop_publishes_checked_rows_before_the_page_and_fences_stale_ones(host, monkeypatch):
    host.catalog_request_id, host.catalog_family_id = "request", "qwen3"
    final = page_report()
    commands = []

    def check(command, _environment, _generation, on_partial):
        commands.append(command)
        on_partial(final["items"])
        partial = emitted(host)[-1]
        assert partial["event"] == "model_catalog" and partial["partial"] is True
        assert partial["report"]["items"][0]["installed_ids"] == [] and partial["report"]["next_offset"] is None
        # A checked row can be installed before its page completes.
        assert set(host.catalog_entries) == {"b" * 32}
        return final

    monkeypatch.setattr(host, "run_catalog_check", check)
    host.load_catalog("qwen3", "request", host.catalog_generation)

    command = commands[0]
    assert "--catalog-stream" in command
    assert command[command.index("--catalog-cache-dir") + 1] == str(host.data_dir / "catalog-cache")
    page = emitted(host)[-1]
    assert page["event"] == "model_catalog" and "partial" not in page and page["report"]["status"] == "ready"

    before = len(emitted(host))
    host.cancel_catalog("request")
    host.publish_catalog(
        host.catalog_report(final, "qwen3"), "qwen3", "request", host.catalog_generation - 1, 0, partial=True
    )
    assert len(emitted(host)) == before and host.catalog_entries == {}


def test_malformed_streamed_row_fails_the_page_closed(host, monkeypatch):
    host.catalog_request_id, host.catalog_family_id = "request", "qwen3"

    def check(_command, _environment, _generation, on_partial):
        on_partial([{"id": "not an entry"}])
        return page_report()

    monkeypatch.setattr(host, "run_catalog_check", check)
    host.load_catalog("qwen3", "request", host.catalog_generation)

    (event,) = [event for event in emitted(host) if event["event"] == "model_catalog"]
    assert event["report"]["status"] == "error" and event["report"]["items"] == [] and host.catalog_entries == {}


def test_catalogue_child_lines_are_read_as_they_arrive(host):
    final = page_report()
    lines = [{"partial_items": final["items"]}, {"partial_items": []}, final]
    script = "import json,sys\nfor line in json.loads(sys.argv[1]): print(json.dumps(line), flush=True)"
    command = [sys.executable, "-c", script, json.dumps(lines)]
    partials = []

    assert host.run_catalog_check(command, {}, host.catalog_generation, partials.append) == final
    assert partials == [final["items"], []]
    # Without a row callback the child's whole output is the report.
    single = [sys.executable, "-c", script, json.dumps([final])]
    assert host.run_catalog_check(single, {}, host.catalog_generation) == final
    with pytest.raises(ValueError):
        host.run_catalog_check(
            [sys.executable, "-c", "print('not json')"], {}, host.catalog_generation, partials.append
        )
