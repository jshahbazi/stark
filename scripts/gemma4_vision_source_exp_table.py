"""Generate mathematical constants for the larger-vision scalar exp policy."""

from __future__ import annotations

import argparse
import decimal
import struct
from pathlib import Path


def split(value):
    # The numerical formula uses F64-rounded constants, represented with F32
    # pairs on Metal. No values are extracted from a host library or checkpoint.
    rounded = decimal.Decimal(float(value))
    high = struct.unpack("<f", struct.pack("<f", float(rounded)))[0]
    low = struct.unpack(
        "<f", struct.pack("<f", float(rounded - decimal.Decimal(high)))
    )[0]

    def literal(number):
        text = format(number, ".9g")
        if "." not in text and "e" not in text:
            text += ".0"
        return text + "f"

    return f"float2({literal(high)}, {literal(low)})"


def generate():
    with decimal.localcontext() as context:
        context.prec = 100
        ln2 = decimal.Decimal(2).ln()
        a = ln2 / 128
        # Leading cubic Chebyshev correction: r^3 = T3(2r)/32 + 3r/16.
        # The remaining quadratic approximation is independently source-tested;
        # it is not a claim of universal bit-exactness to any host libm.
        constants = {
            "INV": decimal.Decimal(128) / ln2,
            "C0": a * a / 2,
            "C1": a + a * a * a / 32,
        }
        lines = [
            "// Generated mathematical constants; no host code/data copied.",
            "// Regenerate with scripts/gemma4_vision_source_exp_table.py.",
        ]
        for name, value in constants.items():
            lines.append(
                f"constant float2 GEMMA4_FLASH_SOURCE_{name} = {split(value)};"
            )
        lines.append("constant float2 GEMMA4_FLASH_SOURCE_POW2[128] = {")
        for index in range(128):
            lines.append(f"  {split((ln2 * index / 128).exp())},")
        lines.append("};")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(generate())


if __name__ == "__main__":
    main()
