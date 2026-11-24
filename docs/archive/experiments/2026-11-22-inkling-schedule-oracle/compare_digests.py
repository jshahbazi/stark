"""Compare two aggregate `inkling_slice` reports while ignoring timings.

Use it for Stark-versus-Stark controls, such as one command run with and
without an experiment selector. Every leaf is compared except keys that only
carry wall time, GPU time or throughput. Exit status 1 means at least one
compared leaf differs; only key paths are printed.
"""

import argparse
import json
import re
import sys
from pathlib import Path

TIMING = re.compile(r"(^|_)(ms|milliseconds|seconds|tok_s|ratio|ratios)$|_ms$|^wall|^paired_wall")


def leaves(value, path=""):
    if isinstance(value, dict):
        for key, item in value.items():
            if TIMING.search(key):
                continue
            yield from leaves(item, f"{path}.{key}" if path else key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from leaves(item, f"{path}[{index}]")
    else:
        yield path, value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--ignore", action="append", default=[], help="Extra key-path regular expression to skip")
    args = parser.parse_args()
    ignored = [re.compile(pattern) for pattern in args.ignore]
    left = dict(leaves(json.loads(args.left.read_text())))
    right = dict(leaves(json.loads(args.right.read_text())))
    keys = sorted(set(left) | set(right))
    differing = [
        key
        for key in keys
        if not any(pattern.search(key) for pattern in ignored) and left.get(key, None) != right.get(key, None)
    ]
    compared = len(keys) - sum(1 for key in keys if any(pattern.search(key) for pattern in ignored))
    for key in differing:
        print(f"differs: {key}")
    print(f"compared={compared} differing={len(differing)}")
    sys.exit(1 if differing else 0)


if __name__ == "__main__":
    main()
