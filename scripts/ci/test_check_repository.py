"""Credential-scope checks for the dedicated Stark signing workflow."""

from __future__ import annotations

import copy
import importlib.util
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
                        "run": 'gh release create "$STARK_RELEASE_VERSION" --draft',
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


if __name__ == "__main__":
    unittest.main()
