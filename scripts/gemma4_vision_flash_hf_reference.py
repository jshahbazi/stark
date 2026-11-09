"""Create private model-free pinned Torch CPU Flash attention fixtures."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
from pathlib import Path

REVISION = "5c4886908584029761b579af026dcfb627c84070"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(output):
    import torch

    if (
        torch.__version__.split("+")[0] != "2.14.1"
        or torch.version.git_version != REVISION
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
    witnesses = 0
    query_rows, heads, width, padded = 8, 2, 72, 2520
    with torch.inference_mode():
        for valid in (31, 511, 513, 1025, 2394, 2520):
            queries = torch.randn(1, heads, query_rows, width).bfloat16()
            keys = torch.randn(1, heads, padded, width).bfloat16()
            values = torch.randn(1, heads, padded, width).bfloat16()
            valid_mask = torch.arange(padded)[None, :] < valid
            mask = torch.zeros(1, padded, dtype=torch.bfloat16).masked_fill(
                ~valid_mask, -float("inf")
            )
            expected, _ = (
                torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(
                    queries,
                    keys,
                    values,
                    attn_mask=mask,
                    scale=1.0,
                )
            )
            ordinary = torch.nn.functional.scaled_dot_product_attention(
                queries,
                keys,
                values,
                attn_mask=mask,
                scale=1.0,
            )
            if not torch.equal(expected, ordinary):
                raise ValueError("default_attention_backend_changed")
            scores = queries.float() @ keys.float().transpose(-1, -2)
            precise = (
                scores.masked_fill(~valid_mask, -float("inf")).softmax(-1)
                @ values.float()
            ).bfloat16()
            rounding_witnesses = int((precise != expected).sum())
            witnesses += rounding_witnesses
            q = torch.ones(heads, query_rows, 128, dtype=torch.bfloat16)
            k = torch.ones(valid, heads, 128, dtype=torch.bfloat16)
            v = torch.ones_like(k)
            # Nonzero padding canaries verify that only logical72 channels
            # participate; production prepares zero tails separately.
            q[:, :, :width] = queries[0]
            k[:, :, :width] = keys[0, :, :valid].transpose(0, 1)
            v[:, :, :width] = values[0, :, :valid].transpose(0, 1)
            records = {}
            for name, value in (
                ("queries", q),
                ("keys", k),
                ("values", v),
                ("expected", expected[0].transpose(0, 1).contiguous()),
            ):
                path = output / f"case-{len(cases)}-{name}.bf16"
                path.write_bytes(
                    value.contiguous()
                    .view(torch.uint16)
                    .numpy()
                    .astype("<u2")
                    .tobytes()
                )
                records[name] = {"file": path.name, "sha256": digest(path)}
            cases.append(
                {
                    "query_rows": query_rows,
                    "valid_key_rows": valid,
                    "source_padded_key_rows": padded,
                    "heads": heads,
                    "logical_head_dim": width,
                    "physical_head_dim": 128,
                    "ordinary_f32_rounding_witnesses": rounding_witnesses,
                    **records,
                }
            )
    if witnesses == 0:
        raise ValueError("missing_probability_rounding_witnesses")
    # A single unmasked key makes the pinned operator's F32 logsumexp equal
    # its F32 QK score exactly. This independently exposes CPUBlas' no-downcast
    # accumulation, before exponential/PV rounding can hide score differences.
    torch.manual_seed(20261108)
    witness_q = torch.randn(1, 1, 256, 72).bfloat16()
    witness_k = torch.randn(1, 1, 2520, 72).bfloat16()
    mask = torch.full((1, 2520), -float("inf"), dtype=torch.bfloat16)
    mask[:, 0] = 0
    _, lse = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(
        witness_q, witness_k, torch.zeros_like(witness_k), attn_mask=mask, scale=1.0
    )
    source_scores = lse[0, 0].float().contiguous()
    x, y = witness_q[0, 0].float(), witness_k[0, 0, 0].float()
    sums = torch.zeros(256, 8, 4)
    for start in (0, 32):
        sums += (x[:, start : start + 32] * y[start : start + 32]).reshape(256, 8, 4)
    for offset in (4, 2, 1):
        sums[:, :offset] = sums[:, :offset] + sums[:, offset : offset * 2]
    dot = (sums[:, 0, 0] + sums[:, 0, 1]) + (sums[:, 0, 2] + sums[:, 0, 3])
    tail = x[:, 64:68] * y[64:68] + x[:, 68:72] * y[68:72]
    legacy_scores = dot + ((tail[:, 0] + tail[:, 1]) + (tail[:, 2] + tail[:, 3]))
    qk_witnesses = int((legacy_scores != source_scores).sum())
    if qk_witnesses == 0:
        raise ValueError("missing_qk_accumulation_witness")
    q = torch.ones(256, 128, dtype=torch.bfloat16)
    k = torch.ones(1, 128, dtype=torch.bfloat16)
    q[:, :72], k[:, :72] = witness_q[0, 0], witness_k[0, 0, :1]
    qk_records = {}
    for name, value in (("queries", q), ("keys", k)):
        path = output / f"qk-witness-{name}.bf16"
        path.write_bytes(
            value.contiguous().view(torch.uint16).numpy().astype("<u2").tobytes()
        )
        qk_records[name] = {"file": path.name, "sha256": digest(path)}
    path = output / "qk-witness-expected.f32"
    path.write_bytes(source_scores.numpy().astype("<f4").tobytes())
    qk_records["expected"] = {"file": path.name, "sha256": digest(path)}
    manifest = {
        "scope": "model-free-larger-vision-flash",
        "torch_version": torch.__version__,
        "torch_source_revision": REVISION,
        "producer_sha256": digest(__file__),
        "oracle": "aten._scaled_dot_product_flash_attention_for_cpu",
        "seed": 20261107,
        "cases": cases,
        "qk_witness": {
            "query_rows": 256,
            "key_rows": 1,
            "heads": 1,
            "source_padded_key_rows": 2520,
            "legacy32_f32_differences": qk_witnesses,
            "oracle": "single-unmasked-key-pinned-flash-logsumexp",
            **qk_records,
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
    return {
        "passed": True,
        "cases": len(cases),
        "rounding_witnesses": witnesses,
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
            result = run(args.output)
    except Exception as error:  # noqa: BLE001 - keep private fixture errors content-free
        print(json.dumps({"passed": False, "error_type": type(error).__name__}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
