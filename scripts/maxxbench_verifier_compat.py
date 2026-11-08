"""Scoped source and verifier compatibility for pinned MaxxBench/Pier images.

The upstream hook remains installed and receives a private delegation view.
Actual environment methods stay bound to their original environment and vars.
Task source, tests and global Compose/environment settings are never changed.
"""

from __future__ import annotations

import contextlib
import copy
import functools
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import tomllib
from pathlib import Path

TRIAL_IMAGE_ATTRIBUTE = "_stark_maxxbench_environment_image"
TRIAL_SOURCE_PLATFORM_ATTRIBUTE = "_stark_maxxbench_source_platform"
TRIAL_OBSERVED_PLATFORM_ATTRIBUTE = "_stark_maxxbench_observed_source_platform"
_SOURCE_TASK = "cliffy-config-file-parsing"
_SOURCE_DOCKERFILE_SHA256 = "2575713dcc93a27bc1b3b052aa02129f13d2dba753f7ca7fe97a796397d9bef1"
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
_IMAGE_REFERENCE = re.compile(r"[a-z0-9][a-z0-9._:/-]{0,255}(?:@sha256:[0-9a-f]{64})?\Z")
_BASE_IMAGE = re.compile(r"(?m)^\s*ARG\s+BASE_IMAGE\s*$")
CONTAINER_IMAGE_FORMAT = '{"id":{{json .Image}},"reference":{{json .Config.Image}}}'
# Exact release bytes, each containing only services.main.build.args.BASE_IMAGE.
# This deliberately refuses edits, extra settings and arbitrary YAML features.
_STALE_COMPOSE_SHA256 = {
    "abs-stepped-slices": "8983f971fa5178381a28fa5074716f02a0cc55ac9a5d3640b0dff27e39fb37c5",
    "adaptix-name-mapping-aliases": "1be2e5055c1793ff21466f1c411ed4bfd200786eb5b58ac30321ee7d0c5fe560",
    "anko-typed-variable-bindings": "efea333e64b46b27a7e424f62a84e068b18967f5f244050def8daf42f029779e",
}


class VerifierHandoffError(ValueError):
    """A fixed infrastructure category with no native or Docker diagnostics."""


class SourcePlatformError(ValueError):
    """A fixed source-environment platform policy failure."""


def source_platform_metadata(trial):
    selection = getattr(trial, TRIAL_SOURCE_PLATFORM_ATTRIBUTE, None)
    observed = getattr(trial, TRIAL_OBSERVED_PLATFORM_ATTRIBUTE, None)
    expected = {"platform": "linux/amd64", "dockerfile_sha256": _SOURCE_DOCKERFILE_SHA256}
    if selection is None:
        return {"selected": None, "observed": None, "dockerfile_sha256": None}
    if selection != expected:
        raise SourcePlatformError("source_platform_policy_invalid")
    if observed is not None and observed != "linux/amd64":
        raise SourcePlatformError("source_platform_changed")
    return {"selected": selection["platform"], "observed": observed,
            "dockerfile_sha256": selection["dockerfile_sha256"]}


def valid_image(value):
    return (
        isinstance(value, dict) and set(value) == {"id", "reference"}
        and isinstance(value["id"], str) and _IMAGE_ID.fullmatch(value["id"]) is not None
        and isinstance(value["reference"], str) and _IMAGE_REFERENCE.fullmatch(value["reference"]) is not None
        # A local config/image ID is not a portable Dockerfile FROM reference.
        and _IMAGE_ID.fullmatch(value["reference"]) is None
    )


def capture_environment_image(trial):
    """Capture only the exact main container's immutable ID and image reference."""
    environment = getattr(trial, "_environment", None)
    project = getattr(environment, "session_id", None)
    if not isinstance(project, str) or re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,255}", project) is None:
        return None
    deadline = time.monotonic() + 5

    def inspect(command):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        return subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                              timeout=remaining, check=False)

    try:
        container = inspect([
            "docker", "ps", "--all", "--no-trunc",
            "--filter", "label=com.docker.compose.project=" + project,
            "--filter", "label=com.docker.compose.service=main", "--format", "{{.ID}}",
        ])
        if container is None or container.returncode != 0:
            return None
        container_id = container.stdout.strip()
        if re.fullmatch(r"[0-9a-f]{64}", container_id) is None:
            return None
        result = inspect(["docker", "container", "inspect", container_id, "--format", CONTAINER_IMAGE_FORMAT])
        if result is None or result.returncode != 0 or len(result.stdout) > 2048:
            return None
        image = json.loads(result.stdout)
    except (OSError, subprocess.TimeoutExpired, ValueError, TypeError):
        return None
    if not valid_image(image):
        return None
    source = source_platform_metadata(trial)
    if source["selected"] is not None:
        # Keep the complete container and image inspection within the original
        # five-second budget; no extra blocking phase is added to progress.
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SourcePlatformError("source_platform_unavailable")
        try:
            platform = verify_image_reference(image, timeout=remaining)
        except VerifierHandoffError:
            raise SourcePlatformError("source_platform_unavailable") from None
        if platform != source["selected"]:
            raise SourcePlatformError("source_platform_changed")
        setattr(trial, TRIAL_OBSERVED_PLATFORM_ATTRIBUTE, platform)
    return image


def verify_image_reference(image, *, timeout=5):
    """Verify the captured ID and return its allowlisted Linux build platform."""
    if not valid_image(image):
        raise VerifierHandoffError("verifier_base_image_unavailable")
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image["reference"]],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=timeout, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise VerifierHandoffError("verifier_base_image_unavailable") from None
    if result.returncode != 0 or len(result.stdout) > 1024 * 1024:
        raise VerifierHandoffError("verifier_base_image_unavailable")
    try:
        entries = json.loads(result.stdout)
    except (ValueError, TypeError):
        raise VerifierHandoffError("verifier_base_image_unavailable") from None
    if not isinstance(entries, list) or len(entries) != 1 or not isinstance(entries[0], dict):
        raise VerifierHandoffError("verifier_base_image_unavailable")
    metadata = entries[0]
    if metadata.get("Id") != image["id"]:
        raise VerifierHandoffError("verifier_base_image_changed")
    # Inspect JSON only in memory: Variant is absent from some Docker versions'
    # template contexts. Never retain config, labels or other image metadata.
    system = metadata.get("Os")
    architecture = metadata.get("Architecture")
    variant = metadata.get("Variant")
    if (system != "linux" or not isinstance(architecture, str)
            or (variant is not None and not isinstance(variant, str))
            or architecture not in {"amd64", "arm64", "arm"}
            or variant not in {None, "", "v5", "v6", "v7", "v8"}
            or (architecture == "amd64" and variant not in {None, ""})
            or (architecture == "arm64" and variant not in {None, "", "v8"})
            or (architecture == "arm" and variant not in {"v5", "v6", "v7"})):
        raise VerifierHandoffError("verifier_base_image_platform_unavailable")
    return f"{system}/{architecture}" + (f"/{variant}" if variant else "")


def _platform_config(config, task_env, platform, error_class, category):
    """Copy a config; refuse conflicting explicit selectors at Pier boundaries."""
    environment = config.environment
    persistent = environment.kwargs.get("persistent_env")
    if persistent is not None and not isinstance(persistent, dict):
        raise error_class(category)
    settings = (os.environ, environment.env, task_env, persistent or {})
    for setting in settings:
        if "DOCKER_DEFAULT_PLATFORM" in setting and setting["DOCKER_DEFAULT_PLATFORM"] != platform:
            raise error_class(category)
    copied = copy.deepcopy(config)
    copied.environment.env["DOCKER_DEFAULT_PLATFORM"] = platform
    # Pier kwargs take precedence over config.env at its factory boundary.
    if "persistent_env" in environment.kwargs:
        copied.environment.kwargs["persistent_env"] = {
            **(copied.environment.kwargs["persistent_env"] or {}), "DOCKER_DEFAULT_PLATFORM": platform,
        }
    return copied


def _verifier_config(config, env_config, platform):
    return _platform_config(config, getattr(env_config, "env", {}) or {}, platform,
                            VerifierHandoffError, "verifier_base_image_platform_conflict")


def patch_source_image_platform(trial_class):
    """Select AMD64 only for the pinned Cliffy recipe's x86-only Deno binary."""
    if getattr(trial_class, "_stark_maxxbench_source_platform_patched", False):
        return
    original = trial_class.create.__func__

    @classmethod
    @functools.wraps(original)
    async def create(cls, config):
        path = config.task.path
        selection = None
        if path is not None and path.name == _SOURCE_TASK:
            try:
                recipe = path / "environment/Dockerfile"
                checksum = hashlib.sha256(recipe.read_bytes()).hexdigest()
                task = tomllib.loads((path / "task.toml").read_text())
            except (OSError, ValueError):
                raise SourcePlatformError("source_platform_recipe_changed") from None
            if checksum != _SOURCE_DOCKERFILE_SHA256:
                raise SourcePlatformError("source_platform_recipe_changed")
            metadata = task.get("metadata")
            environment = task.get("environment", {})
            if not isinstance(metadata, dict) or metadata.get("task_id") != _SOURCE_TASK:
                raise SourcePlatformError("source_platform_task_changed")
            if not isinstance(environment, dict):
                raise SourcePlatformError("source_platform_policy_invalid")
            task_env = environment.get("env", {})
            if not isinstance(task_env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in task_env.items()):
                raise SourcePlatformError("source_platform_policy_invalid")
            config = _platform_config(config, task_env, "linux/amd64", SourcePlatformError, "source_platform_conflict")
            selection = {"platform": "linux/amd64", "dockerfile_sha256": checksum}
        trial = await original(cls, config)
        if selection is not None:
            setattr(trial, TRIAL_SOURCE_PLATFORM_ATTRIBUTE, selection)
        return trial

    trial_class.create = create
    trial_class._stark_maxxbench_source_platform_patched = True


class _VariablesView:
    def __init__(self, original, reference):
        self._original = original
        self.main_image_name = reference

    def __getattr__(self, name):
        return getattr(self._original, name)


class _EnvironmentView:
    def __init__(self, original, reference):
        self._original = original
        self._env_vars = _VariablesView(original._env_vars, reference)

    def __getattr__(self, name):
        # In particular, delegated methods remain bound to original, so their
        # Compose configuration reads original._env_vars, never the view's.
        return getattr(self._original, name)


class _TrialView:
    def __init__(self, original, reference, context, config):
        self._original = original
        self._environment = _EnvironmentView(original._environment, reference)
        self._verifier_env_build_context = lambda _step: context
        self.config = config

    def __getattr__(self, name):
        return getattr(self._original, name)


@contextlib.contextmanager
def _handoff_context(context):
    compose = context / "docker-compose.yaml"
    if not compose.exists():
        yield context
        return
    expected = _STALE_COMPOSE_SHA256.get(context.parent.name)
    if context.name != "tests" or expected is None or hashlib.sha256(compose.read_bytes()).hexdigest() != expected:
        raise VerifierHandoffError("verifier_compose_unsupported")
    with tempfile.TemporaryDirectory(prefix="stark-maxxbench-handoff-") as directory:
        private = Path(directory) / "tests"
        shutil.copytree(context, private)
        (private / "docker-compose.yaml").unlink()
        yield private


def patch_verifier_image_compat(trial_class):
    """Wrap the installed stock hook without changing its source or real state."""
    if getattr(trial_class, "_stark_maxxbench_image_compat", False):
        return
    if not getattr(trial_class, "_maxxbench_image_handoff", False):
        raise VerifierHandoffError("verifier_upstream_hook_missing")
    upstream = trial_class._verify_with_separate_environment

    @functools.wraps(upstream)
    async def verify(self, env_config, *, key, step_cfg, verifier_env, artifacts_dir, artifacts=None):
        ownership = getattr(self, "_stark_maxxbench_resource_ownership", None)
        if ownership is not None:
            ownership.register_project(self._separate_verifier_session_id(key))
        context = self._verifier_env_build_context(step_cfg)
        dockerfile = context / "Dockerfile"
        arguments = {"key": key, "step_cfg": step_cfg, "verifier_env": verifier_env,
                     "artifacts_dir": artifacts_dir, "artifacts": artifacts}
        if not dockerfile.is_file() or _BASE_IMAGE.search(dockerfile.read_text()) is None:
            return await upstream(self, env_config, **arguments)
        image = getattr(self, TRIAL_IMAGE_ATTRIBUTE, None)
        platform = verify_image_reference(image)
        config = _verifier_config(self.config, env_config, platform)
        with _handoff_context(context) as private:
            view = _TrialView(self, image["reference"], private, config)
            try:
                return await upstream(view, env_config, **arguments)
            finally:
                # Only disposable view pointers were changed by either hook.
                # Release them on cancellation and failures as well as success.
                view.__dict__.clear()

    trial_class._verify_with_separate_environment = verify
    trial_class._stark_maxxbench_image_compat = True
