"""Create private model-free canonical larger-Gemma vision RMS fixtures."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
from pathlib import Path

SOURCE = "4f874549f79deda4bb9ce7fb7d7b8e7349122be1711f733f7a4157e46ed10cc3"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(output):
    import torch
    import transformers
    from transformers.models.gemma4 import modeling_gemma4

    if (
        torch.__version__.split("+")[0] != "2.14.1"
        or transformers.__version__ != "5.15.1"
        or digest(modeling_gemma4.__file__) != SOURCE
    ):
        raise ValueError("reference_identity_changed")
    output = output.resolve()
    if (
        output.exists()
        or subprocess.run(
            ["git", "check-ignore", "--quiet", "--", str(output)],
            cwd=Path(__file__).resolve().parents[1],
            check=False,
        ).returncode
    ):
        raise ValueError("output_must_be_new_and_ignored")
    output.mkdir(mode=0o700, parents=True)
    torch.set_num_threads(2)
    torch.manual_seed(20261107)
    cases = []
    for width in (72, 1152):
        for weighted in (False, True):
            torch.manual_seed(20261107)
            rows = 4096 if width == 72 else 512
            inputs = torch.randn(rows, width).bfloat16()
            gains = torch.linspace(-1.25, 1.5, width).bfloat16()
            norm = modeling_gemma4.Gemma4RMSNorm(width, eps=1e-6, with_scale=weighted)
            if weighted:
                norm.weight = torch.nn.Parameter(gains, requires_grad=False)
            with torch.inference_mode():
                expected = norm(inputs)
            # Negative control only: the independent oracle above is the pinned
            # upstream module. Require real BF16 ties that reject legacy sums.
            total = torch.zeros(rows)
            values = inputs.float()
            for channel in range(width):
                total += values[:, channel] * values[:, channel]
            inverse = 1.0 / torch.sqrt(total[:, None] / width + 1e-6)
            legacy = values * inverse
            if weighted:
                legacy *= gains.float()
            witnesses = int((legacy.bfloat16() != expected).sum())
            if witnesses == 0 or not torch.isfinite(expected).all():
                raise ValueError("fixture_missing_rounding_witness")
            records = {}
            for name, value in (
                ("input", inputs),
                ("gains", gains),
                ("expected", expected),
            ):
                path = output / f"case-{len(cases)}-{name}.bf16"
                path.write_bytes(
                    value.view(torch.uint16).numpy().astype("<u2").tobytes()
                )
                records[name] = {"file": path.name, "sha256": digest(path)}
            cases.append(
                {
                    "width": width,
                    "rows": rows,
                    "has_scale": weighted,
                    "epsilon": 1e-6,
                    "legacy_rounding_witnesses": witnesses,
                    **records,
                }
            )
    manifest = {
        "scope": "model-free-larger-vision-rms",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "source_sha256": SOURCE,
        "producer_sha256": digest(__file__),
        "seed": 20261107,
        "cases": cases,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
    return {
        "passed": True,
        "cases": len(cases),
        "manifest_sha256": digest(output / "manifest.json"),
    }


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        with (
            open(os.devnull, "w") as quiet,
            contextlib.redirect_stdout(quiet),
            contextlib.redirect_stderr(quiet),
        ):
            report = run(args.output)
    except Exception as error:  # noqa: BLE001 - content-free private fixture errors
        print(json.dumps({"passed": False, "error_type": type(error).__name__}))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
