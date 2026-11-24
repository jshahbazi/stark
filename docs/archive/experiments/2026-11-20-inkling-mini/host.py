"""Record only aggregate macOS host conditions across serial inference slots."""

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
import hybrid_host_metrics as metrics  # noqa: E402


def group(name):
    name = name.lower()
    if "stark-generate" in name or "stark-score" in name:
        return "other_stark"
    if "llama-server" in name:
        return "llama"
    if "ollama" in name:
        return "ollama"
    if any(label in name for label in ("python", "nvfp4", "probe", "mlxrunner")):
        return "probe_or_python"
    return "other"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--interval", type=float, default=2)
    args = parser.parse_args()
    previous = metrics.gpu_clients()
    with args.output.open("x", buffering=1) as out:
        while True:
            time.sleep(args.interval)
            now = time.time_ns()
            clients = metrics.gpu_clients()
            changes = Counter()
            for key in clients.keys() & previous.keys():
                current = clients[key]
                changes[group(current["name"])] += max(0, current["counter"] - previous[key]["counter"])
            previous = clients
            pressure = subprocess.check_output(
                ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True, timeout=5
            )
            record = {
                "unix_ns": now,
                "thermal": metrics.thermal_state(),
                "pressure": int(pressure),
                "gpu_percent": metrics.gpu_device_utilization(),
                "metal_counter_delta": dict(changes),
                **metrics.vm_counters(),
            }
            print(json.dumps(record), file=out)


if __name__ == "__main__":
    main()
