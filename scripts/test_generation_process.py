#!/usr/bin/env python3
"""Content-free process checks for the generation worker's startup boundary."""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path


def run_checks(binary: Path, *, glm_exl3_enabled: bool = False) -> int:
    with tempfile.TemporaryDirectory(prefix="stark-generate-process-") as directory:
        missing_model = str(Path(directory) / "private-marker-no-model")
        exl3_fixture = Path(directory) / "private-marker-exl3"
        exl3_fixture.mkdir()
        (exl3_fixture / "config.json").write_text(json.dumps({
            "model_type": "glm5_next",
            "vocab_size": 2,
            "max_position_embeddings": 262_144,
            "eos_token_id": 1,
            "expert_streaming": {
                "family": "mlx-stream",
                "format": "mtplx-expert-manifest-v2",
                "manifest": "expert-manifest-v2.json",
                "runtime_manifest": "expert-manifest.json",
                "quant": "exl3-k3-mul1",
            },
        }))
        (exl3_fixture / "tokenizer.json").write_text(json.dumps({
            "version": "1.0",
            "model": {"type": "WordLevel", "vocab": {"unk": 0, "eos": 1}, "unk_token": "unk"},
        }))
        invalid_context = Path(directory) / "private-marker-invalid-context"
        invalid_context.mkdir()
        (invalid_context / "config.json").write_text(json.dumps({
            "model_type": "flint", "vocab_size": 2, "max_position_embeddings": 1,
        }))
        cases = [
            ([], "invalid_arguments"),
            (["--model-path"], "invalid_arguments"),
            (["--model-path", ""], "invalid_arguments"),
            (["--unknown-private-marker", missing_model], "invalid_arguments"),
            (["--model-path", missing_model, "--unknown-private-marker"], "invalid_arguments"),
            (["--model-path", missing_model], "model_config_unavailable"),
            (["--model-path", str(invalid_context)], "invalid_finite_context_limit"),
            (
                ["--model-path", str(exl3_fixture)],
                "hybrid_model_load_failed" if glm_exl3_enabled else "glm_exl3_backend_unavailable",
            ),
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
    parser.add_argument("--glm-exl3-enabled", action="store_true", help="Expect an optional bridge-enabled worker")
    arguments = parser.parse_args()
    cases = run_checks(arguments.binary.resolve(strict=True), glm_exl3_enabled=arguments.glm_exl3_enabled)
    print(json.dumps({"passed": True, "cases": cases}, sort_keys=True))


if __name__ == "__main__":
    main()
