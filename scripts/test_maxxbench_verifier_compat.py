"""Exercise scoped compatibility through the exact pinned upstream hook."""

import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import maxxbench_verifier_compat as compat
import pytest

IMAGE = {"id": "sha256:" + "a" * 64, "reference": "sample-task-main:latest"}


def tree_hashes(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


def context(tmp_path, task="sample-task", *, compose=None):
    path = tmp_path / task / "tests"
    path.mkdir(parents=True)
    (path / "Dockerfile").write_text("ARG BASE_IMAGE\nFROM ${BASE_IMAGE}\n")
    (path / "test.patch").write_text("synthetic verifier fixture\n")
    if compose is not None:
        (path / "docker-compose.yaml").write_text(compose)
    return path


def known_compose(task):
    return ("services:\n  main:\n    build:\n      args:\n"
            f"        BASE_IMAGE: hb__datacurve-{task}__agent-9db111e0f51ec78c\n")


@pytest.fixture
def make_trial(monkeypatch):
    fixture = Path(__file__).parent / "fixtures/maxxbench/run.py"
    spec = importlib.util.spec_from_file_location("stock_maxxbench_handoff_fixture", fixture)
    stock = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stock)
    calls = []
    monkeypatch.setattr(compat.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=IMAGE["id"]))

    def make(path, behavior=None):
        class Environment:
            def __init__(self):
                self._env_vars = SimpleNamespace(main_image_name="original-compose-name", other="unchanged")

            def bound_compose_vars(self):
                return self._env_vars

        class Trial:
            def __init__(self):
                self._environment = Environment()
                self.record = []
                setattr(self, compat.TRIAL_IMAGE_ATTRIBUTE, dict(IMAGE))

            def _verifier_env_build_context(self, _step):
                return path

            async def _verify_with_separate_environment(self, env_config, **kwargs):
                generated = self._verifier_env_build_context(kwargs["step_cfg"])
                compose = generated / "docker-compose.yaml"
                item = {
                    "view": self, "context": generated,
                    "compose": json.loads(compose.read_text()) if compose.exists() else None,
                    "getter_reference": self._environment._env_vars.main_image_name,
                    "bound_vars": self._environment.bound_compose_vars(),
                    "env_config": env_config, "arguments": kwargs,
                }
                self.record.append(item)
                calls.append(item)
                if behavior is not None:
                    return await behavior(self)
                return "verified"

        stock.patch_verifier_image_handoff(Trial)
        compat.patch_verifier_image_compat(Trial)
        return Trial()

    return make, calls


async def invoke(trial):
    return await trial._verify_with_separate_environment(
        "config", key="trial", step_cfg=None, verifier_env={"fixed": "value"},
        artifacts_dir=Path("artifacts"), artifacts=None,
    )


def test_stock_hook_uses_verified_reference_without_mutating_environment_or_source(tmp_path, make_trial):
    make, calls = make_trial
    source = context(tmp_path)
    trial = make(source)
    environment = trial._environment
    variables = environment._env_vars
    before = tree_hashes(source)
    assert asyncio.run(invoke(trial)) == "verified"
    assert calls[0]["compose"] == {"services": {"main": {"build": {"args": {"BASE_IMAGE": IMAGE["reference"]}}}}}
    assert calls[0]["getter_reference"] == IMAGE["reference"]
    assert calls[0]["bound_vars"] is variables
    assert calls[0]["bound_vars"].main_image_name == "original-compose-name"
    assert trial._environment is environment and environment._env_vars is variables
    assert "_verifier_env_build_context" not in trial.__dict__
    assert calls[0]["view"].__dict__ == {}
    assert not calls[0]["context"].exists()
    assert tree_hashes(source) == before


@pytest.mark.parametrize("task", ["abs-stepped-slices", "adaptix-name-mapping-aliases", "anko-typed-variable-bindings"])
def test_known_single_override_is_pruned_only_in_private_copy(tmp_path, make_trial, task):
    make, calls = make_trial
    source = context(tmp_path, task, compose=known_compose(task))
    before = tree_hashes(source)
    trial = make(source)
    assert asyncio.run(invoke(trial)) == "verified"
    assert calls[0]["compose"]["services"]["main"]["build"]["args"] == {"BASE_IMAGE": IMAGE["reference"]}
    assert calls[0]["context"] != source and not calls[0]["context"].exists()
    assert tree_hashes(source) == before


@pytest.mark.parametrize("task,compose", [
    ("unknown-task", known_compose("abs-stepped-slices")),
    ("abs-stepped-slices", known_compose("abs-stepped-slices") + "    image: unexpected\n"),
    ("abs-stepped-slices", "services:\n  sidecar:\n    image: unexpected\n"),
])
def test_unknown_compose_shape_is_rejected_without_source_changes(tmp_path, make_trial, task, compose):
    make, calls = make_trial
    source = context(tmp_path, task, compose=compose)
    before = tree_hashes(source)
    with pytest.raises(compat.VerifierHandoffError, match="verifier_compose_unsupported"):
        asyncio.run(invoke(make(source)))
    assert not calls and tree_hashes(source) == before


def test_tag_id_drift_is_rejected_before_verifier_execution(tmp_path, make_trial, monkeypatch):
    make, calls = make_trial
    source = context(tmp_path)
    monkeypatch.setattr(compat.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="sha256:" + "b" * 64))
    with pytest.raises(compat.VerifierHandoffError, match="verifier_base_image_changed"):
        asyncio.run(invoke(make(source)))
    assert not calls


def test_missing_capture_cannot_fall_back_to_calculated_tag(tmp_path, make_trial):
    make, calls = make_trial
    trial = make(context(tmp_path))
    delattr(trial, compat.TRIAL_IMAGE_ATTRIBUTE)
    with pytest.raises(compat.VerifierHandoffError, match="verifier_base_image_unavailable"):
        asyncio.run(invoke(trial))
    assert not calls


def test_cancellation_restores_private_context_and_preserves_original_pointers(tmp_path, make_trial):
    make, calls = make_trial
    source = context(tmp_path, "abs-stepped-slices", compose=known_compose("abs-stepped-slices"))
    before = tree_hashes(source)

    async def scenario():
        entered = asyncio.Event()

        async def wait(_view):
            entered.set()
            await asyncio.Event().wait()

        trial = make(source, wait)
        original_environment = trial._environment
        original_variables = original_environment._env_vars
        override = lambda _step: source
        trial._verifier_env_build_context = override
        running = asyncio.create_task(invoke(trial))
        await entered.wait()
        temporary = calls[0]["context"]
        assert temporary.exists()
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert trial._verifier_env_build_context is override
        assert trial._environment is original_environment
        assert original_environment._env_vars is original_variables
        assert calls[0]["view"].__dict__ == {} and not temporary.exists()
        assert asyncio.all_tasks() == {asyncio.current_task()}

    asyncio.run(scenario())
    assert tree_hashes(source) == before


def test_non_base_verifier_uses_stock_hook_with_original_trial(tmp_path, make_trial, monkeypatch):
    make, calls = make_trial
    source = context(tmp_path)
    (source / "Dockerfile").write_text("FROM python:3.12\n")
    monkeypatch.setattr(compat.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("unneeded Docker metadata query"))
    trial = make(source)
    assert asyncio.run(invoke(trial)) == "verified"
    assert calls[0]["view"] is trial and calls[0]["context"] == source


def test_capture_reads_only_id_and_buildable_reference(monkeypatch):
    trial = SimpleNamespace(_environment=SimpleNamespace(session_id="sample-task"))
    commands = []

    def run(command, **kwargs):
        commands.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="b" * 64 if command[1] == "ps" else json.dumps(IMAGE))

    monkeypatch.setattr(compat.subprocess, "run", run)
    assert compat.capture_environment_image(trial) == IMAGE
    assert commands[1][0][-1] == compat.CONTAINER_IMAGE_FORMAT
    assert all(options["stderr"] == compat.subprocess.DEVNULL and 0 < options["timeout"] <= 5
               for _command, options in commands)


@pytest.mark.parametrize("image", [{"id": IMAGE["id"], "reference": IMAGE["id"]},
                                  {**IMAGE, "private": "unwanted"},
                                  {"id": IMAGE["id"], "reference": "https://user:secret@registry/image"}])
def test_invalid_image_record_never_reaches_docker(image, monkeypatch):
    monkeypatch.setattr(compat.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("invalid image was inspected"))
    with pytest.raises(compat.VerifierHandoffError, match="verifier_base_image_unavailable"):
        compat.verify_image_reference(image)
