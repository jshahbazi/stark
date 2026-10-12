"""Build and exercise the installed wheel outside the checkout using tiny local files."""

import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from pathlib import Path

from stark_convert.provenance import SOURCE_INPUTS, toolchain_provenance

PROJECT = Path(__file__).resolve().parents[1]


def run(command, *, cwd, environment, expected=0):
    result = subprocess.run(command, cwd=cwd, env=environment, capture_output=True, check=False)
    if result.returncode != expected:
        raise RuntimeError("installed_package_command_failed")
    return result


def check():
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("missing_uv")
    # Fixture helpers are used only by this checkout-side verification script.
    # The child environment has no checkout path, Hub credentials or Hub cache.
    sys.path.insert(0, str(PROJECT / "tests"))
    from test_cli import source_fixture

    expected_provenance = toolchain_provenance()
    with tempfile.TemporaryDirectory(prefix="stark installed package ") as temporary:
        root = Path(temporary)
        environment = os.environ.copy()
        for name in (
            "PYTHONPATH",
            "PYTHONHOME",
            "VIRTUAL_ENV",
            "HF_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
            "HF_TOKEN_PATH",
            "HF_HUB_CACHE",
            "HUGGINGFACE_HUB_CACHE",
        ):
            environment.pop(name, None)
        environment.update(
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            HF_HUB_DISABLE_TELEMETRY="1",
            HF_HOME=str(root / "empty-hub"),
            HF_HUB_CACHE=str(root / "empty-hub" / "hub"),
            HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
            TOKENIZERS_PARALLELISM="false",
        )
        build = root / "dist"
        run(
            [uv, "build", str(PROJECT), "--python", sys.executable, "--out-dir", str(build), "--no-build-logs"],
            cwd=root,
            environment=environment,
        )
        wheel = next(build.glob("*.whl"))
        sdist = next(build.glob("*.tar.gz"))
        with tarfile.open(sdist) as archive:
            names = {"/".join(name.split("/")[1:]) for name in archive.getnames()}
            assert {*SOURCE_INPUTS, "uv.lock"} <= names
            assert not any(name.startswith(("tests/", "build/", "dist/")) for name in names)
        with zipfile.ZipFile(wheel) as archive:
            assert any(name.endswith(".dist-info/licenses/LICENSE.lalamo") for name in archive.namelist())
            for name in (*SOURCE_INPUTS, "uv.lock"):
                assert archive.read("stark_convert/_build_inputs/" + name) == (PROJECT / name).read_bytes()
        requirements = root / "requirements.txt"
        run(
            [
                uv,
                "export",
                "--project",
                str(PROJECT),
                "--locked",
                "--no-dev",
                "--no-emit-project",
                "--output-file",
                str(requirements),
            ],
            cwd=root,
            environment=environment,
        )
        venv = root / "venv"
        run([uv, "venv", "--python", sys.executable, str(venv)], cwd=root, environment=environment)
        python = venv / "bin" / "python"
        run(
            [uv, "pip", "install", "--python", str(python), "--require-hashes", "-r", str(requirements)],
            cwd=root,
            environment=environment,
        )
        run(
            [uv, "pip", "install", "--python", str(python), "--no-deps", "--compile-bytecode", str(wheel)],
            cwd=root,
            environment=environment,
        )
        environment["PATH"] = str(venv / "bin") + os.pathsep + os.defpath
        inspect = (
            "import json,sys; from pathlib import Path; import stark_convert; "
            "from stark_convert.provenance import toolchain_provenance; "
            "assert Path(stark_convert.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()); "
            "assert list((Path(stark_convert.__file__).parent / '_build_inputs').rglob('*.pyc')); "
            "print(json.dumps(toolchain_provenance())); "
            "assert not any(n.split('.')[0] in {'lalamo','jax','torch'} for n in sys.modules)"
        )
        result = run([str(python), "-I", "-c", inspect], cwd=root, environment=environment)
        assert json.loads(result.stdout) == expected_provenance
        converter = str(venv / "bin" / "stark-convert")
        run([converter, "--help"], cwd=root, environment=environment)
        result = run([converter, "--version"], cwd=root, environment=environment)
        expected_version = tomllib.loads((PROJECT / "pyproject.toml").read_text())["project"]["version"]
        assert result.stdout.decode().strip() == "stark-convert " + expected_version
        probe = run([converter, "Qwen/Qwen3-0.6B", "--capabilities"], cwd=root, environment=environment)
        assert json.loads(probe.stdout)["status"] == "ok"
        source = source_fixture(root / "downloaded model", sharded=True)
        outputs = {}
        for mode, options in {
            "catalog": ["--repo", "Qwen/Qwen3-0.6B"],
            "generic": ["--metadata-profile", "hf-generic"],
        }.items():
            output = root / mode
            result = run(
                [converter, str(source), "--output", str(output), "--offline", *options],
                cwd=root,
                environment=environment,
            )
            receipt = json.loads(result.stdout)
            assert receipt["status"] == "ok"
            assert receipt["converter_sources_sha256"] == expected_provenance["converter_sources_sha256"]
            assert receipt["catalogued"] == (mode == "catalog")
            outputs[mode] = output
        benchmark = str(venv / "bin" / "stark-convert-benchmark")
        result = run(
            [
                benchmark,
                "--source",
                str(source),
                "--metadata-profile",
                "hf-generic",
                "--output",
                str(root / "benchmark"),
                "--repeats",
                "2",
                "--reference-output",
                str(outputs["generic"]),
            ],
            cwd=root,
            environment=environment,
        )
        report = json.loads(result.stdout)
        assert report["status"] == "passed" and len(report["runs"]) == 2
        assert all(parity["passed"] for parity in report["parity"])
        assert report == json.loads((root / "benchmark" / "report.json").read_text())
        result = run(
            [converter, str(source), "--output", str(root / "missing-policy"), "--offline"],
            cwd=root,
            environment=environment,
            expected=1,
        )
        assert json.loads(result.stderr)["error_category"] == "local_profile_required"
        (source / "tokenizer.json").unlink()
        result = run(
            [
                converter,
                str(source),
                "--metadata-profile",
                "hf-generic",
                "--offline",
                "--output",
                str(root / "incomplete"),
            ],
            cwd=root,
            environment=environment,
            expected=1,
        )
        assert json.loads(result.stderr)["status"] == "error"
        assert not (root / "incomplete").exists()
        assert not list(root.glob(".stark-convert-*"))
    return {
        "status": "passed",
        "checks": [
            "sdist",
            "wheel",
            "license",
            "provenance",
            "local_catalog",
            "local_generic",
            "benchmark_parity",
            "incomplete_download",
        ],
        "temporary_artifacts_removed": True,
    }


def main():
    try:
        report = check()
    except Exception:
        print('{"status":"failed","error_category":"installed_package_check_failed"}')
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
