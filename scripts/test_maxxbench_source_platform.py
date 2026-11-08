"""Model-free checks for the exact Cliffy source recipe's platform policy."""

import asyncio
import hashlib
import json
from types import SimpleNamespace

import maxxbench_verifier_compat as compat
import pytest


@pytest.fixture
def source_case(tmp_path, monkeypatch):
    monkeypatch.delenv("DOCKER_DEFAULT_PLATFORM", raising=False)
    path = tmp_path / "cliffy-config-file-parsing"
    (path / "environment").mkdir(parents=True)
    recipe = path / "environment/Dockerfile"
    recipe.write_text("FROM synthetic-source\nRUN synthetic-x86-program\n")
    checksum = hashlib.sha256(recipe.read_bytes()).hexdigest()
    monkeypatch.setattr(compat, "_SOURCE_DOCKERFILE_SHA256", checksum)
    (path / "task.toml").write_text('[metadata]\ntask_id = "cliffy-config-file-parsing"\n[environment.env]\n')
    config = SimpleNamespace(task=SimpleNamespace(path=path), environment=SimpleNamespace(
        env={"FIXED": "unchanged", "PRIVATE": "not-published"}, kwargs={}, force_build=True,
    ))
    calls = []

    class Trial:
        def __init__(self, supplied):
            self.config = supplied
            self._environment = SimpleNamespace(session_id=supplied.task.path.name)

        @classmethod
        async def create(cls, supplied):
            calls.append(supplied)
            return cls(supplied)

    compat.patch_source_image_platform(Trial)
    return path, config, Trial, calls, checksum


def test_exact_recipe_uses_copied_config_and_fixed_aggregate_metadata(source_case):
    _path, config, Trial, calls, checksum = source_case
    original_env, original_kwargs = config.environment.env, config.environment.kwargs
    trial = asyncio.run(Trial.create(config))
    assert calls[0] is not config and calls[0].environment is not config.environment
    assert calls[0].environment.env["DOCKER_DEFAULT_PLATFORM"] == "linux/amd64"
    assert calls[0].environment.force_build is True
    assert config.environment.env is original_env and "DOCKER_DEFAULT_PLATFORM" not in original_env
    assert config.environment.kwargs is original_kwargs and original_kwargs == {}
    metadata = compat.source_platform_metadata(trial)
    assert metadata == {"selected": "linux/amd64", "observed": None, "dockerfile_sha256": checksum}
    assert "PRIVATE" not in json.dumps(metadata) and "not-published" not in json.dumps(metadata)
    assert "DOCKER_DEFAULT_PLATFORM" not in compat.os.environ


def test_other_tasks_keep_identical_config_and_require_no_source_files(source_case, monkeypatch):
    path, config, Trial, calls, _checksum = source_case
    config.task.path = path.parent / "another-task-without-files"
    trial = asyncio.run(Trial.create(config))
    assert calls == [config] and trial.config is config
    assert isinstance(trial, Trial)
    assert compat.source_platform_metadata(trial) == {"selected": None, "observed": None, "dockerfile_sha256": None}
    image = {"id": "sha256:" + "a" * 64, "reference": "sample-task-main:latest"}
    monkeypatch.setattr(compat.subprocess, "run", lambda command, **_options: SimpleNamespace(
        returncode=0, stdout="b" * 64 if command[1] == "ps" else json.dumps(image)))
    assert compat.capture_environment_image(trial) == image


@pytest.mark.parametrize("mutation,category", [
    ("recipe", "source_platform_recipe_changed"), ("missing_recipe", "source_platform_recipe_changed"),
    ("task", "source_platform_task_changed"), ("invalid_task", "source_platform_recipe_changed"),
])
def test_changed_or_missing_pinned_source_fails_before_trial_creation(source_case, mutation, category):
    path, config, Trial, calls, _checksum = source_case
    if mutation == "recipe":
        (path / "environment/Dockerfile").write_text("FROM changed\n")
    elif mutation == "missing_recipe":
        (path / "environment/Dockerfile").unlink()
    elif mutation == "task":
        (path / "task.toml").write_text('[metadata]\ntask_id = "another-task"\n')
    else:
        (path / "task.toml").write_text("PRIVATE_INVALID_TOML = [\n")
    with pytest.raises(compat.SourcePlatformError) as error:
        asyncio.run(Trial.create(config))
    assert str(error.value) == category and not calls


@pytest.mark.parametrize("scope", ["process", "trial", "task", "persistent_kwargs"])
def test_explicit_conflicting_source_platform_is_not_overwritten(source_case, monkeypatch, scope):
    path, config, Trial, calls, _checksum = source_case
    if scope == "process":
        monkeypatch.setenv("DOCKER_DEFAULT_PLATFORM", "linux/arm64")
    elif scope == "trial":
        config.environment.env["DOCKER_DEFAULT_PLATFORM"] = "linux/arm64"
    elif scope == "task":
        (path / "task.toml").write_text('[metadata]\ntask_id="cliffy-config-file-parsing"\n'
                                       '[environment.env]\nDOCKER_DEFAULT_PLATFORM="linux/arm64"\n')
    else:
        config.environment.kwargs["persistent_env"] = {"DOCKER_DEFAULT_PLATFORM": "linux/arm64"}
    before = repr(config)
    with pytest.raises(compat.SourcePlatformError, match="^source_platform_conflict$"):
        asyncio.run(Trial.create(config))
    assert repr(config) == before and not calls


def test_source_platform_respects_pier_persistent_kwargs_precedence(source_case):
    _path, config, Trial, _calls, _checksum = source_case
    config.environment.kwargs["persistent_env"] = None
    trial = asyncio.run(Trial.create(config))
    assert trial.config.environment.kwargs["persistent_env"] == {"DOCKER_DEFAULT_PLATFORM": "linux/amd64"}
    assert config.environment.kwargs["persistent_env"] is None


def test_hook_is_idempotent_and_preserves_subclass_dispatch(source_case):
    _path, config, Trial, calls, _checksum = source_case
    first = Trial.create.__func__
    compat.patch_source_image_platform(Trial)
    assert Trial.create.__func__ is first

    class Subclass(Trial):
        pass

    asyncio.run(Subclass.create(config))
    assert len(calls) == 1


@pytest.mark.parametrize("architecture,category", [("amd64", None), ("arm64", "source_platform_changed"),
                                                  (None, "source_platform_unavailable")])
def test_selected_source_captures_actual_platform_within_existing_five_second_budget(
    source_case, monkeypatch, architecture, category,
):
    _path, config, Trial, _calls, _checksum = source_case
    trial = asyncio.run(Trial.create(config))
    trial._environment = SimpleNamespace(session_id="cliffy-config-file-parsing")
    image = {"id": "sha256:" + "a" * 64, "reference": "cliffy-config-file-parsing-main:latest"}
    clock = iter((100, 100, 101, 102))
    monkeypatch.setattr(compat.time, "monotonic", lambda: next(clock))
    commands = []

    def run(command, **options):
        commands.append((command, options))
        if command[1] == "ps":
            payload = "b" * 64
        elif command[1] == "container":
            payload = json.dumps(image)
        else:
            payload = json.dumps([{"Id": image["id"], "Os": "linux", "Architecture": architecture,
                                   "Config": {"Env": ["PRIVATE_NATIVE_METADATA"]}}])
        return SimpleNamespace(returncode=0, stdout=payload)

    monkeypatch.setattr(compat.subprocess, "run", run)
    if category is None:
        assert compat.capture_environment_image(trial) == image
        metadata = compat.source_platform_metadata(trial)
        assert metadata["observed"] == "linux/amd64"
        assert "PRIVATE_NATIVE_METADATA" not in json.dumps(metadata)
    else:
        with pytest.raises(compat.SourcePlatformError, match="^" + category + "$"):
            compat.capture_environment_image(trial)
    assert [options["timeout"] for _command, options in commands] == [5, 4, 3]
    assert all(options["stderr"] == compat.subprocess.DEVNULL for _command, options in commands)
