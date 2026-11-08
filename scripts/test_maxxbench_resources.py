"""Model-free tests: resource ownership boundaries, races and fixed receipts."""

import json
from types import SimpleNamespace

import maxxbench_resources as resources
import pytest

BASE_IMAGE = "sha256:" + "a" * 64
NEW_IMAGE = "sha256:" + "b" * 64
BASE_CACHE = "a" * 25
NEW_CACHE = "b" * 25


def record(*, shared=False, mutable=False, reclaimable=True, parents=()):
    return {"Shared": shared, "Mutable": mutable, "Reclaimable": reclaimable,
            "Parents": list(parents), "Type": "regular"}


class FakeDocker:
    def __init__(self):
        self.selected = {"name": "desktop-linux", "workers": ["worker-one"]}
        self.image_ids = {BASE_IMAGE}
        self.cache_records = {BASE_CACHE: record()}
        self.volume_ids = {f"protected-volume-{value}" for value in range(61)}
        self.removed_images = []
        self.pruned_cache = []
        self.in_use_images = set()
        self.race_cache = set()

    def builder(self):
        return self.selected

    def images(self):
        return set(self.image_ids)

    def volumes(self):
        return set(self.volume_ids)

    def cache(self, _builder, filters=()):
        result = {key: dict(value) for key, value in self.cache_records.items()}
        for selector in filters:
            if selector.startswith("id="):
                identifier = selector[3:]
                if identifier in self.race_cache and identifier in result:
                    result[identifier]["Shared"] = True
                result = {key: value for key, value in result.items() if key == identifier}
            elif selector == "private~=^$":
                result = {key: value for key, value in result.items() if value["Shared"] is False}
            elif selector == "immutable~=^$":
                result = {key: value for key, value in result.items() if value["Mutable"] is False}
            else:
                pytest.fail("unexpected cache selector")
        return result

    def image_in_use(self, identifier):
        return identifier in self.in_use_images

    def remove_image(self, identifier):
        self.removed_images.append(identifier)
        self.image_ids.remove(identifier)

    def prune_cache(self, _builder, identifier):
        self.pruned_cache.append(identifier)
        del self.cache_records[identifier]


@pytest.fixture(autouse=True)
def free_space(monkeypatch):
    monkeypatch.setattr(resources.shutil, "disk_usage", lambda _path: SimpleNamespace(free=30 * 1024**3))


def test_only_new_unshared_immutable_unused_cache_and_new_images_are_removed(tmp_path):
    client = FakeDocker()
    volume_ids = set(client.volume_ids)
    path = tmp_path / "cleanup.json"
    with resources.task_resources(path, client=client):
        client.image_ids.add(NEW_IMAGE)
        client.cache_records.update({NEW_CACHE: record(), "c" * 25: record(shared=True),
                                     "d" * 25: record(mutable=True), "e" * 25: record(reclaimable=False)})
    assert client.removed_images == [NEW_IMAGE]
    assert client.pruned_cache == [NEW_CACHE]
    assert BASE_IMAGE in client.image_ids and BASE_CACHE in client.cache_records
    assert client.volume_ids == volume_ids and len(volume_ids) == 61
    receipt = json.loads(path.read_text())
    assert receipt["state"] == "finished" and receipt["cleanup_failed"] is False
    assert receipt["protected_image_ids"] == [BASE_IMAGE]
    assert receipt["protected_cache_ids"] == [BASE_CACHE]
    assert receipt["protected_volume_count"] == 61
    assert "protected-volume" not in path.read_text()
    assert path.stat().st_mode & 0o777 == 0o600


def test_currently_shared_record_is_not_pruned_after_recheck_race(tmp_path):
    client = FakeDocker()
    with resources.task_resources(tmp_path / "cleanup.json", client=client):
        client.cache_records[NEW_CACHE] = record()
        client.race_cache.add(NEW_CACHE)
    assert not client.pruned_cache


def test_cache_children_are_pruned_before_owned_parents(tmp_path):
    client = FakeDocker()
    parent = "z" * 25
    with resources.task_resources(tmp_path / "cleanup.json", client=client):
        client.cache_records[parent] = record(parents=[BASE_CACHE])
        client.cache_records[NEW_CACHE] = record(parents=[parent])
    assert client.pruned_cache == [NEW_CACHE, parent]
    assert BASE_CACHE in client.cache_records


def test_in_use_image_is_retained_and_cleanup_failure_surfaces(tmp_path):
    client = FakeDocker()
    path = tmp_path / "cleanup.json"
    with pytest.raises(resources.ResourceError, match="resource_cleanup_failed"), resources.task_resources(path, client=client):
        client.image_ids.add(NEW_IMAGE)
        client.in_use_images.add(NEW_IMAGE)
    assert not client.removed_images
    assert NEW_IMAGE in client.image_ids
    assert json.loads(path.read_text())["state"] == "cleanup_failed"


def test_worker_identity_change_aborts_before_any_deletion(tmp_path):
    client = FakeDocker()
    with (
        pytest.raises(resources.ResourceError, match="resource_identity_changed"),
        resources.task_resources(tmp_path / "cleanup.json", client=client),
    ):
        client.image_ids.add(NEW_IMAGE)
        client.cache_records[NEW_CACHE] = record()
        client.selected = {"name": "desktop-linux", "workers": ["different-worker"]}
    assert not client.removed_images and not client.pruned_cache


def test_resource_cleanup_runs_after_body_failure_without_logging_exception(tmp_path):
    client = FakeDocker()
    path = tmp_path / "cleanup.json"
    with pytest.raises(RuntimeError, match="private-marker"), resources.task_resources(path, client=client):
        client.image_ids.add(NEW_IMAGE)
        client.cache_records[NEW_CACHE] = record()
        raise RuntimeError("private-marker")
    assert client.removed_images == [NEW_IMAGE] and client.pruned_cache == [NEW_CACHE]
    assert json.loads(path.read_text())["body_failed"] is True
    assert "private-marker" not in path.read_text()


def test_low_disk_guard_applies_before_oracle_or_model_task_resources(tmp_path, monkeypatch):
    monkeypatch.setattr(resources.shutil, "disk_usage", lambda _path: SimpleNamespace(free=resources.MIN_FREE_BYTES - 1))
    client = FakeDocker()
    client.builder = lambda: pytest.fail("Docker queried despite insufficient disk")
    with (
        pytest.raises(resources.ResourceError, match="insufficient_disk_space"),
        resources.task_resources(tmp_path / "cleanup.json", client=client),
    ):
        pytest.fail("task started with insufficient disk")


def test_cli_removal_commands_are_exact_id_and_have_no_volume_or_force_image_path(monkeypatch):
    commands = []

    def run(command, **kwargs):
        commands.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(resources.subprocess, "run", run)
    client = resources.DockerResources()
    client.remove_image(NEW_IMAGE)
    client.prune_cache({"name": "desktop-linux"}, NEW_CACHE)
    assert commands[0][0] == ["docker", "image", "rm", "--no-prune", NEW_IMAGE]
    assert commands[1][0] == ["docker", "buildx", "prune", "--builder", "desktop-linux", "--force",
                              "--filter", "id=" + NEW_CACHE, "--filter", "private~=^$",
                              "--filter", "immutable~=^$"]
    assert all(options["stderr"] == resources.subprocess.DEVNULL for _command, options in commands)
    assert all("--all" not in command and "--volumes" not in command for command, _options in commands)


def test_cli_diagnostics_and_invalid_id_never_escape_fixed_error_boundary(monkeypatch):
    monkeypatch.setattr(resources.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout="private-marker"))
    client = resources.DockerResources()
    with pytest.raises(resources.ResourceError, match="resource_cleanup_failed") as error:
        client.remove_image(NEW_IMAGE)
    assert "private-marker" not in str(error.value)
    with pytest.raises(resources.ResourceError, match="resource_cleanup_failed"):
        client.remove_image("--all")


def test_missing_protected_volume_aborts_cleanup_without_mutating_other_resources(tmp_path):
    client = FakeDocker()
    with (
        pytest.raises(resources.ResourceError, match="resource_identity_changed"),
        resources.task_resources(tmp_path / "cleanup.json", client=client),
    ):
        client.volume_ids.remove("protected-volume-0")
        client.image_ids.add(NEW_IMAGE)
    assert not client.removed_images and not client.pruned_cache


def test_partial_cleanup_preserves_deletion_provenance_when_later_snapshot_fails(tmp_path):
    client = FakeDocker()
    original_cache = client.cache
    path = tmp_path / "cleanup.json"
    with (
        pytest.raises(resources.ResourceError, match="resource_snapshot_failed"),
        resources.task_resources(path, client=client),
    ):
        client.image_ids.add(NEW_IMAGE)

        def failed_snapshot(*args, **kwargs):
            if client.removed_images:
                raise resources.ResourceError("resource_snapshot_failed")
            return original_cache(*args, **kwargs)

        client.cache = failed_snapshot
    receipt = json.loads(path.read_text())
    assert receipt["image_removal_attempted_ids"] == [NEW_IMAGE]
    assert receipt["removed_image_ids"] == [NEW_IMAGE]
    assert receipt["failure"] == "resource_snapshot_failed"
