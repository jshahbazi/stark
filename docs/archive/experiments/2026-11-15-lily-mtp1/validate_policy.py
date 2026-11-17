"""Artifact-bound selector and image lifecycle check; hashes/counters only."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_generation_prefix_cache import Worker
from benchmark_qwen38_mtp import exact_prefix, ids_sha256, sha256_file

MODEL = "/Users/johnshahbazian/Library/Application Support/Stark/models/qwen38-flash-next-lily-q4-xl-0f94842e"
BASE = "/Users/johnshahbazian/development/stark-llama-v1-study/runtime/llama-v1-baseline-target/release/stark-generate"
CANDIDATE = str(ROOT/"runtime/build-candidate/release/stark-generate")
OUT = ROOT/"runtime/lily-one-draft/policy-validation.json"


def digest(reply):
    assert "error" not in reply
    ack = reply["mtp"]
    return {"output_sha256": ids_sha256(reply["generated_token_ids"]),
            "output_tokens": len(reply["generated_token_ids"]), "sampled_tokens": reply["sampled_tokens"],
            "enabled": ack["enabled"], "drafts": ack["drafts"],
            "counts": {k: ack["stats"][k] for k in ("rounds", "proposed", "accepted", "verified_rows")},
            "reused_input_tokens": reply.get("reused_input_tokens", 0)}


def main():
    os.umask(0o077)
    assert not OUT.exists()
    for name in tuple(os.environ):
        if name.startswith(("STARK_", "UZU_", "LILY_", "GGML_")):
            os.environ.pop(name)
    corpus = (ROOT/"runtime/lily-one-draft/results/model-corpus.rs").read_text()
    selectors = {"enabled": {"mtp": True}, "one": {"mtp_drafts": 1}, "two": {"mtp_drafts": 2},
                 "off": {"mtp": False}, "zero": {"mtp_drafts": 0}, "omitted": {}}
    report = {"attempt": "20261115-LILY-MTP1-02", "worker_sha256": {"baseline": sha256_file(Path(BASE)),
              "candidate": sha256_file(Path(CANDIDATE))}, "harness_sha256": sha256_file(Path(__file__)),
              "checks": [], "image_qualification": "lifecycle only; not answer quality"}
    references = {}
    try:
        for label, binary, vision in [("baseline_text", BASE, False), ("baseline_vision", BASE, True),
                                      ("candidate_text", CANDIDATE, False), ("candidate_vision", CANDIDATE, True)]:
            print(json.dumps({"stage": "checking_policy", "worker": label}), flush=True)
            with Worker(binary, MODEL, prefix_cache=True, vision=vision, timeout=900) as worker:
                assert worker.ready["mtp"]["recommended_drafts"] == (1 if label.startswith("candidate") else 2)
                assert worker.ready["mtp"]["default_drafts"] == 0 and worker.ready["mtp"]["max_drafts"] == 3
                assert ("image_input" in worker.ready) is vision
                ids = exact_prefix(worker.tokenize(corpus), 512)
                for name, choice in selectors.items():
                    result = digest(worker.request({"operation": "generate", "input_token_ids": ids,
                        "context_tokens": 1024, "max_new_tokens": 32, "seed": 42,
                        "sampling": {"mode": "stochastic", "temperature": 1, "top_k": 20, "top_p": .95}, **choice}))
                    expected = 1 if name == "one" or (name == "enabled" and label.startswith("candidate")) else 2 if name in ("two", "enabled") else 0
                    assert result["drafts"] == expected and result["enabled"] is (expected > 0)
                    assert result["reused_input_tokens"] == 0
                    if expected > 0:
                        assert result["counts"]["proposed"] > 0
                    else:
                        assert all(v == 0 for v in result["counts"].values())
                    if label == "baseline_text":
                        references[expected] = result
                    else:
                        assert result == references[expected]
                    report["checks"].append({"worker": label, "selector": name, "result": result, "passed": True})
                if label == "candidate_vision":
                    image = digest(worker.image_transaction(8192))
                    assert image["drafts"] == 0 and not image["enabled"]
                    assert image["counts"]["proposed"] == 0 and image["reused_input_tokens"] == 0
                    report["checks"].append({"worker": label, "selector": "image", "result": image, "passed": True})
                    result = digest(worker.request({"operation": "generate", "input_token_ids": ids,
                        "context_tokens": 1024, "max_new_tokens": 32, "seed": 42,
                        "sampling": {"mode": "stochastic", "temperature": 1, "top_k": 20, "top_p": .95}, "mtp": True}))
                    assert result == references[1]
                    report["checks"].append({"worker": label, "selector": "text_after_image", "result": result, "passed": True})
        report["passed"] = True
    except BaseException as error:
        report["passed"] = False
        report["failure_type"] = type(error).__name__
    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"passed": report["passed"], "checks": len(report["checks"]), "failure_type": report.get("failure_type")}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
