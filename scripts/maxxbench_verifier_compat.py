"""Scoped compatibility for the pinned MaxxBench/Pier verifier image handoff.

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
from pathlib import Path

TRIAL_IMAGE_ATTRIBUTE = "_stark_maxxbench_environment_image"
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
    return image if valid_image(image) else None


def verify_image_reference(image):
    """Verify the captured ID and return its allowlisted Linux build platform."""
    if not valid_image(image):
        raise VerifierHandoffError("verifier_base_image_unavailable")
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image["reference"]],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=5, check=False,
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


def _verifier_config(config, env_config, platform):
    """Copy only the verifier config; refuse conflicting explicit selectors."""
    environment = config.environment
    persistent = environment.kwargs.get("persistent_env")
    if persistent is not None and not isinstance(persistent, dict):
        raise VerifierHandoffError("verifier_base_image_platform_conflict")
    settings = (os.environ, environment.env, getattr(env_config, "env", {}) or {}, persistent or {})
    for setting in settings:
        if "DOCKER_DEFAULT_PLATFORM" in setting and setting["DOCKER_DEFAULT_PLATFORM"] != platform:
            raise VerifierHandoffError("verifier_base_image_platform_conflict")
    copied = copy.deepcopy(config)
    copied.environment.env["DOCKER_DEFAULT_PLATFORM"] = platform
    # Pier kwargs take precedence over config.env at its factory boundary.
    if "persistent_env" in environment.kwargs:
        copied.environment.kwargs["persistent_env"] = {
            **(copied.environment.kwargs["persistent_env"] or {}), "DOCKER_DEFAULT_PLATFORM": platform,
        }
    return copied


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
