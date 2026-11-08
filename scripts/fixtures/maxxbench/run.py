# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["datacurve-pier==0.3.1"]
# ///
"""Run Pier with the task-image handoff required by MaxxBench's verifiers."""

import functools
import json
import re
import shutil
import tempfile
from importlib.metadata import version
from pathlib import Path


def patch_verifier_image_handoff(trial_class):
    """Supply BASE_IMAGE without changing the checked-in task or grading logic."""
    if getattr(trial_class, "_maxxbench_image_handoff", False):
        return
    original = trial_class._verify_with_separate_environment

    @functools.wraps(original)
    async def verify(
        self, env_config, *, key, step_cfg, verifier_env, artifacts_dir, artifacts=None
    ):
        context = self._verifier_env_build_context(step_cfg)
        dockerfile = context / "Dockerfile"
        needs_image = dockerfile.is_file() and re.search(
            r"(?m)^\s*ARG\s+BASE_IMAGE\s*$", dockerfile.read_text()
        )
        arguments = dict(
            key=key, step_cfg=step_cfg, verifier_env=verifier_env,
            artifacts_dir=artifacts_dir, artifacts=artifacts,
        )
        if not needs_image:
            return await original(self, env_config, **arguments)

        image = getattr(
            getattr(self._environment, "_env_vars", None), "main_image_name", None
        )
        if not image:
            raise RuntimeError("This verifier requires the rebuilt Docker task image.")
        if (context / "docker-compose.yaml").exists():
            raise RuntimeError("Refusing to replace an existing verifier Compose file.")

        # Keep generated Compose configuration outside the task tree. Different
        # trials can grade concurrently without altering shared release files.
        with tempfile.TemporaryDirectory(prefix="maxxbench-verifier-") as directory:
            temporary_context = Path(directory) / "tests"
            shutil.copytree(context, temporary_context)
            compose = {"services": {"main": {"build": {"args": {"BASE_IMAGE": image}}}}}
            (temporary_context / "docker-compose.yaml").write_text(
                json.dumps(compose, indent=2) + "\n"
            )
            attribute = "_verifier_env_build_context"
            sentinel = object()
            previous = self.__dict__.get(attribute, sentinel)
            setattr(self, attribute, lambda _step: temporary_context)
            try:
                return await original(self, env_config, **arguments)
            finally:
                if previous is sentinel:
                    delattr(self, attribute)
                else:
                    setattr(self, attribute, previous)

    trial_class._verify_with_separate_environment = verify
    trial_class._maxxbench_image_handoff = True


def main():
    if version("datacurve-pier") != "0.3.1":
        raise SystemExit("Use `uv run run.py` to run the pinned Pier version.")
    from pier.cli.main import app
    from pier.trial.trial import Trial

    patch_verifier_image_handoff(Trial)
    app()


if __name__ == "__main__":
    main()
