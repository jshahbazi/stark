"""Build and exercise the model-free N-API boundary fixture outside the checkout."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", required=True, type=Path)
    args = parser.parse_args()
    fixture = Path(__file__).resolve().parent
    cargo = shutil.which("cargo") or str(Path.home() / ".cargo/bin/cargo")
    node = shutil.which("node")
    if node is None:
        raise SystemExit("Node.js with Node-API 10 support is required")
    target = args.target_dir.resolve()
    env = os.environ | {"CARGO_PROFILE_DEV_DEBUG": "0", "CARGO_INCREMENTAL": "0"}
    subprocess.run(
        [cargo, "+1.98.1", "build", "--manifest-path", str(fixture / "Cargo.toml"),
         "--locked", "--offline", "--target-dir", str(target), "-j", "1"],
        check=True, env=env,
    )
    library = {
        "darwin": "libstark_napi_identity_test.dylib",
        "linux": "libstark_napi_identity_test.so",
        "win32": "stark_napi_identity_test.dll",
    }[sys.platform]
    addon = target / "debug/stark_napi_identity_test.node"
    shutil.copy2(target / "debug" / library, addon)
    for script, extra in [("probe.cjs", ["fixed"]), ("test.cjs", []), ("async.cjs", []), ("external.cjs", [])]:
        subprocess.run([node, "--expose-gc", str(fixture / script), str(addon), *extra], check=True)


if __name__ == "__main__":
    main()
