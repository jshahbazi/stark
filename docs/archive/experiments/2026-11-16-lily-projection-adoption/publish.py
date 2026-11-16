"""Publish source-bound aggregate evidence, never model payloads or values."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ARCHIVE = ROOT / "docs/archive/experiments/2026-11-16-lily-projection-adoption"
OUTPUT = ROOT / "docs/archive/runs/data/2026-11-16-lily-projection-reuse.json"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def inspect(value):
    if isinstance(value, dict):
        assert not set(value).intersection({
            "prompt", "rendered_prompt", "text", "input_token_ids",
            "generated_token_ids", "probabilities", "logits", "traceback",
            "exception", "credentials", "content",
        })
        for child in value.values():
            inspect(child)
    elif isinstance(value, list):
        for child in value:
            inspect(child)


sources = {name: (HERE / name).read_bytes() for name in (
    "qualify.py", "qualify-v2.py", "qualify-v3.py", "publish.py",
)}
names = (
    "source-identity.json", "lifecycle.json", "lifecycle-details.json",
    "host-preflight-probe.json", "native-lifecycle-v2.json", "arms-v2.json",
    "arms-v3.json", "book-v3.json", "prompts-v3.json",
)
receipts = {}
for name in names:
    raw = (HERE / name).read_bytes()
    value = json.loads(raw)
    inspect(value)
    receipts[name] = {"private_receipt_sha256": digest(raw), "aggregate": value}
for name, count in (
    ("native-lifecycle-v2.json", 18), ("arms-v3.json", 25),
    ("book-v3.json", 5), ("prompts-v3.json", 8),
):
    value = receipts[name]["aggregate"]
    assert value["status"] == "passed"
    assert len(value["checks"]) == count and all(value["checks"].values())
for receipt, source in (
    ("lifecycle.json", "qualify.py"),
    ("native-lifecycle-v2.json", "qualify-v2.py"),
    ("arms-v2.json", "qualify-v2.py"),
    ("arms-v3.json", "qualify-v3.py"),
    ("book-v3.json", "qualify-v3.py"),
    ("prompts-v3.json", "qualify-v3.py"),
):
    assert receipts[receipt]["aggregate"]["harness_sha256"] == digest(sources[source])
identity = receipts["source-identity.json"]["aggregate"]
for name, expected in identity["source_sha256"].items():
    assert digest((ROOT / name).read_bytes()) == expected
assert digest((HERE / "target/release/stark-generate").read_bytes()) == identity["generator_sha256"]
patch = subprocess.check_output([
    "git", "diff", "HEAD", "--no-ext-diff", "--binary", "--",
    "vendor/lily-core/src/kernels/quant.rs",
    "vendor/lily-core/src/kernels/metal/stark_decode_rows.metal",
    "vendor/lily-core/src/kernels/stark_projection_reuse_tests.rs",
    "vendor/lily-core/src/kernels/official_head_q8_rows_tests.rs",
], cwd=ROOT)
assert patch and b"new file mode" in patch
sources["integration.patch"] = patch
result = {
    "schema": "stark-lily-projection-adoption-public-v1",
    "attempt": "20261116-LILY-REUSE-01",
    "base": identity["base"],
    "scope": "final native worker functional controls and source provenance",
    "model": {
        "repository": "fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl",
        "revision": "0f94842e3682b49acace4daeb4cf2e65a6f94f83",
        "existing_payload_manifest_sha256": "97c11dd6f7ee5a69e23759b9774acb21a786c48552ca2ede9dac8b6be4b5bc98",
    },
    "sources_sha256": {name: digest(raw) for name, raw in sources.items()},
    "tests": {
        "core": {"passed": 97, "ignored": 55},
        "projection_reuse": {"passed": 1, "cases": 240, "compared_bytes": 10436400, "unequal_bytes": 0},
        "decode_rows": {"passed": 1},
        "gdn_projection_rollback": {"passed": 1},
        "generator": {"passed": 145, "ignored": 5},
        "physical_engine": {"passed": 1076, "ignored": 163, "integration_passed": 9, "documentation_passed": 3},
        "native_checks": 56,
        "test_executable_sha256": {
            "core": "9f5398bc7b26931c2f0f2545fc237b40735b737cc3d6c53fdeda846261977a3c",
            "generator": "f51ffe63cf473b1a035a6963c42554accfba5b7da852f540a08d254aee0ff4b9",
            "engine": "e88017be502a22db063721e94b805ff3d705f3c9138deb5a08c0df78623d6cab",
        },
    },
    "receipts": receipts,
    "performance": "No new controlled timing comparison; prior paired studies remain source-bound.",
    "host_admission": "Resource-deferred wrapper; native continuation retains loader/request admission.",
    "installed_or_released": False,
}
inspect(result)
assert not OUTPUT.exists()
ARCHIVE.mkdir(parents=True, exist_ok=True)
for name, raw in sources.items():
    target = ARCHIVE / name
    assert not target.exists()
    target.write_bytes(raw)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps({"public_data_sha256": digest(OUTPUT.read_bytes()), "source_files": len(sources), "private_receipts": len(receipts), "native_checks": 56}))
