#!/usr/bin/env python3
"""Content-free process checks for the generation worker's startup boundary."""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path


def run_checks(binary: Path) -> int:
    with tempfile.TemporaryDirectory(prefix="stark-generate-process-") as directory:
        missing_model = str(Path(directory) / "private-marker-no-model")
        cases = [
            ([], "invalid_arguments"),
            (["--model-path"], "invalid_arguments"),
            (["--model-path", ""], "invalid_arguments"),
            (["--unknown-private-marker", missing_model], "invalid_arguments"),
            (["--model-path", missing_model, "--unknown-private-marker"], "invalid_arguments"),
            (["--model-path", missing_model], "model_config_unavailable"),
        ]
        environment = os.environ.copy()
        environment["STARK_CAPTURE_TEST_SECRET"] = "private-marker"
        environment["UZU_CAPTURE_TEST_SECRET"] = "private-marker"
        for arguments, error in cases:
            result = subprocess.run(
                [str(binary), *arguments],
                input=b"",
                capture_output=True,
                env=environment,
                timeout=15,
                check=False,
            )
            # Check bytes as well as JSON to exclude extra native diagnostics.
            if result.returncode != 1 or result.stderr or len(result.stdout.splitlines()) != 1:
                raise RuntimeError("worker_process_boundary_failed")
            try:
                reply = json.loads(result.stdout)
            except (ValueError, UnicodeError):
                raise RuntimeError("worker_process_boundary_failed") from None
            if reply != {"ready": False, "error": error} or b"private-marker" in result.stdout:
                raise RuntimeError("worker_process_boundary_failed")
        return len(cases)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    arguments = parser.parse_args()
    cases = run_checks(arguments.binary.resolve(strict=True))
    print(json.dumps({"passed": True, "cases": cases}, sort_keys=True))


if __name__ == "__main__":
    main()
