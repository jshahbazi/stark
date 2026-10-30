"""Credential-scope and publication checks for the Stark signing workflow."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

SPEC = importlib.util.spec_from_file_location(
    "check_repository", Path(__file__).with_name("check_repository.py")
)
assert SPEC is not None and SPEC.loader is not None
policy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(policy)


def signing_workflow() -> dict:
    return {
        "on": {"workflow_dispatch": {"inputs": {"version": {"required": True}}}},
        "permissions": {"contents": "read"},
        "jobs": {
            "sign": {
                "if": policy.SIGNING_CONDITION,
                "runs-on": "macos-26",
                "environment": "macos-release",
                "steps": [
                    {
                        "name": "Import temporary Apple credentials",
                        "env": {
                            name: f"${{{{ secrets.{name} }}}}"
                            for name in policy.SIGNING_SECRETS
                        },
                        "run": "python .github/scripts/apple_credentials.py setup",
                    },
                    {
                        "name": "Sign and notarize the standalone bundle",
                        "env": {
                            "MACOS_SIGNING_IDENTITY": "${{ secrets.MACOS_SIGNING_IDENTITY }}"
                        },
                        "run": "uv run --python 3.13.13 python scripts/sign_distribution.py",
                    },
                    {
                        "if": "always()",
                        "run": "python .github/scripts/apple_credentials.py cleanup",
                    },
                ],
            },
            "publish": {
                "if": policy.SIGNING_CONDITION,
                "needs": "sign",
                "runs-on": "ubuntu-24.04",
                "permissions": {"contents": "write"},
                "steps": [
                    {
                        "env": {"GH_TOKEN": "${{ github.token }}"},
                        "run": 'gh release create "$STARK_RELEASE_VERSION" --latest',
                    }
                ],
            },
        },
    }


class SigningWorkflowPolicyTests(unittest.TestCase):
    def check(self, document: dict, name: str = "signed-release.yml") -> list[str]:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / ".github/workflows" / name
            path.parent.mkdir(parents=True)
            path.write_text(yaml.safe_dump(document, sort_keys=False))
            with patch.object(policy, "ROOT", root):
                return policy.check_workflows([path])

    def test_protected_manual_signing_and_separate_publisher_are_allowed(self):
        self.assertEqual(self.check(signing_workflow()), [])

    def test_other_workflows_cannot_reuse_apple_secrets(self):
        errors = self.check(signing_workflow(), "other.yml")
        self.assertTrue(
            any("secrets reference is forbidden" in error for error in errors)
        )

    def test_automatic_trigger_or_unprotected_scope_is_rejected(self):
        mutations = (
            lambda document: document["on"].update({"pull_request": {}}),
            lambda document: document["jobs"]["sign"].update({"if": "true"}),
            lambda document: document["jobs"]["sign"].update({"environment": "other"}),
            lambda document: document["jobs"]["sign"].update(
                {"permissions": {"contents": "write"}}
            ),
            lambda document: document["jobs"]["publish"].update(
                {"environment": "macos-release"}
            ),
            lambda document: document["jobs"]["publish"].update({"needs": []}),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                document = signing_workflow()
                mutate(document)
                self.assertTrue(self.check(document))

    def test_missing_or_conditional_cleanup_is_rejected(self):
        for cleanup in (None, {"if": "success()"}):
            with self.subTest(cleanup=cleanup):
                document = signing_workflow()
                steps = document["jobs"]["sign"]["steps"]
                if cleanup is None:
                    steps.pop()
                else:
                    steps[-1].update(cleanup)
                self.assertTrue(self.check(document))

    def test_new_secret_or_secret_use_outside_import_is_rejected(self):
        for target in ("import", "publish", "job", "dynamic", "aggregate"):
            with self.subTest(target=target):
                document = signing_workflow()
                if target == "import":
                    document["jobs"]["sign"]["steps"][0]["env"]["EXTRA"] = (
                        "${{ secrets.EXTRA }}"
                    )
                elif target == "publish":
                    document["jobs"]["publish"]["steps"][0]["env"][
                        "APPLE_API_KEY_ID"
                    ] = "${{ secrets.APPLE_API_KEY_ID }}"
                elif target == "job":
                    document["jobs"]["sign"]["env"] = {
                        "MACOS_SIGNING_IDENTITY": "${{ secrets.MACOS_SIGNING_IDENTITY }}"
                    }
                elif target == "dynamic":
                    document["jobs"]["sign"]["steps"][1]["env"][
                        "MACOS_SIGNING_IDENTITY"
                    ] = "${{ secrets['MACOS_SIGNING_IDENTITY'] }}"
                else:
                    document["jobs"]["publish"]["steps"][0]["env"]["ALL"] = (
                        "${{ toJSON(secrets) }}"
                    )
                self.assertTrue(self.check(document))

    def test_yaml_alias_does_not_hide_credential_use_in_publisher(self):
        document = signing_workflow()
        document["jobs"]["publish"]["steps"][0]["env"] = document["jobs"]["sign"][
            "steps"
        ][0]["env"]
        errors = self.check(document)
        self.assertTrue(any("approved signing steps" in error for error in errors))

    def test_missing_setup_secret_and_reusable_secrets_mapping_are_rejected(self):
        for inherited in (False, True):
            with self.subTest(inherited=inherited):
                document = copy.deepcopy(signing_workflow())
                if inherited:
                    document["jobs"]["publish"]["secrets"] = "inherit"
                else:
                    del document["jobs"]["sign"]["steps"][0]["env"]["APPLE_API_KEY_ID"]
                self.assertTrue(self.check(document))

    def test_inherited_release_workflow_still_cannot_be_enabled(self):
        errors = self.check(signing_workflow(), "release.yml")
        self.assertTrue(
            any("inherited workflow job is not disabled" in error for error in errors)
        )


class LockfileCoveragePolicyTests(unittest.TestCase):
    def test_removed_lily_runtime_lockfile_is_not_required_or_scanned(self):
        files = [
            "Cargo.lock",
            "scripts/ci/check_repository.py.lock",
            "tools/uv.lock",
            "tools/converter/uv.lock",
            "tools/stock_benchmark/mlx/uv.lock",
            "tools/stock_benchmark/mlx_current/uv.lock",
            "tools/stock_benchmark/vllm/uv.lock",
        ]
        self.assertNotIn("vendor/lily-core/Cargo.lock", policy.SCANNED_LOCKFILES)
        self.assertEqual(policy.check_lockfile_coverage(files), [])

        workflow = yaml.safe_load(
            (Path(__file__).resolve().parents[2] / ".github/workflows/security.yml").read_text()
        )
        security_script = "\n".join(
            str(step.get("run", ""))
            for job in workflow["jobs"].values()
            for step in job.get("steps", [])
        )
        self.assertNotIn("vendor/lily-core/Cargo.lock", security_script)


class SignedReleasePublicationTests(unittest.TestCase):
    def run_publish(
        self, *, corrupt_checksum: bool = False, existing_version: str = ""
    ) -> tuple[subprocess.CompletedProcess[str], list[list[str]]]:
        workflow_path = (
            Path(__file__).resolve().parents[2] / ".github/workflows/signed-release.yml"
        )
        workflow = yaml.safe_load(workflow_path.read_text())
        publish = workflow["jobs"]["publish"]["steps"][-1]["run"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            assets = root / "artifacts/signed-release"
            assets.mkdir(parents=True)
            stem = "stark-1.2.3-aarch64-apple-darwin"
            for suffix in ("tar.gz", "zip", "dmg"):
                name = f"{stem}.{suffix}"
                content = f"synthetic {suffix}\n".encode()
                (assets / name).write_bytes(content)
                digest = hashlib.sha256(content).hexdigest()
                (assets / f"{name}.sha256").write_text(f"{digest}  {name}\n")
            dmg = (assets / f"{stem}.dmg").read_bytes()
            (assets / "Stark.dmg").write_bytes(dmg)
            (assets / "Stark.dmg.sha256").write_text(
                f"{hashlib.sha256(dmg).hexdigest()}  Stark.dmg\n"
            )
            for name in (
                f"{stem}.dmg.notarization.json",
                f"{stem}.dmg.verification.json",
                f"{stem}.notarization.json",
                "verification.json",
                "install.sh",
            ):
                (assets / name).write_text("synthetic receipt or installer\n")
            if corrupt_checksum:
                (assets / f"{stem}.tar.gz").write_bytes(b"modified after signing\n")

            bin_dir = root / "bin"
            bin_dir.mkdir()
            gh_log = root / "gh.jsonl"
            gh = bin_dir / "gh"
            gh.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "with open(os.environ['GH_TEST_LOG'], 'a') as log:\n"
                "    log.write(json.dumps(sys.argv[1:]) + '\\n')\n"
                "sys.exit(1 if sys.argv[1:3] == ['release', 'view'] else 0)\n"
            )
            gh.chmod(0o755)
            result = subprocess.run(
                ["bash", "-c", publish],
                cwd=root,
                env={
                    **os.environ,
                    "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
                    "GH_TEST_LOG": str(gh_log),
                    "GH_REPO": "test/stark",
                    "GITHUB_SHA": "a" * 40,
                    "STARK_RELEASE_VERSION": "1.2.3",
                    "EXISTING_VERSION": existing_version,
                    "RUNNER_TEMP": str(root),
                },
                text=True,
                capture_output=True,
                check=False,
            )
            calls = (
                [json.loads(line) for line in gh_log.read_text().splitlines()]
                if gh_log.exists()
                else []
            )
            return result, calls

    def test_verified_release_is_published_as_latest_after_creating_immutable_tag(self):
        result, calls = self.run_publish()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            [call[:2] for call in calls],
            [["release", "view"], ["api", "--method"], ["release", "create"]],
        )
        tag, release = calls[1:]
        self.assertIn("ref=refs/tags/v1.2.3", tag)
        self.assertIn("sha=" + "a" * 40, tag)
        self.assertIn("POST", tag)
        self.assertEqual(release[2], "v1.2.3")
        self.assertIn("--verify-tag", release)
        self.assertIn("--latest", release)
        self.assertNotIn("--draft", release)
        self.assertNotIn("--prerelease", release)
        self.assertIn("Stark.dmg", release)
        self.assertIn("Stark.dmg.sha256", release)

    def test_checksum_failure_cannot_create_a_tag_or_release(self):
        result, calls = self.run_publish(corrupt_checksum=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual([call[:2] for call in calls], [["release", "view"]])

    def test_existing_version_remains_artifact_only(self):
        result, calls = self.run_publish(existing_version="1.2.3")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
