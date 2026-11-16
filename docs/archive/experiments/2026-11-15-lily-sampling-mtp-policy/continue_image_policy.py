"""Continue the retained selector run with an available native template helper."""
import json
import os
from pathlib import Path
import time

import validate_policy_initial as prior


def main():
    os.umask(0o077)
    out = prior.ROOT / "runtime/lily-topk-policy/image-policy-continuation.json"
    assert not out.exists()
    earlier = prior.ROOT / "runtime/lily-topk-policy/policy-validation.json"
    initial = json.loads(earlier.read_text())
    assert len(initial["checks"]) == 44 and initial["failure_type"] == "ChatError"
    before = next(r["result"] for r in initial["checks"] if r["worker"] == "before_text" and r["case"] == "explicit_one")
    for name in tuple(os.environ):
        if name.startswith(("STARK_", "UZU_", "LILY_", "GGML_")):
            os.environ.pop(name)
    os.environ["STARK_REQUEST_BUDGET_MIB"] = "16"
    helper = prior.ROOT / "runtime/build-candidate/release/stark-template"
    assert helper.is_file()
    os.environ["PATH"] = str(helper.parent) + os.pathsep + os.environ.get("PATH", "")
    report = {"attempt": "20261115-LILY-MTP1-04b", "harness_sha256": prior.sha256_file(Path(__file__)),
        "predecessor_sha256": prior.sha256_file(earlier), "template_sha256": prior.sha256_file(helper),
        "candidate_sha256": prior.sha256_file(prior.CAND), "checks": []}
    assert report["candidate_sha256"] == initial["worker_sha256"]["candidate"]
    identity = json.loads(prior.PRIOR.read_text())
    def unchanged():
        return all([s.st_size, s.st_mtime_ns, s.st_ino] == expected
            for name, expected in identity["model_file_identities"].items() for s in [(prior.MODEL / name).stat()])
    try:
        assert unchanged()
        with prior.Worker(str(prior.CAND), str(prior.MODEL), prefix_cache=True, vision=True, timeout=900) as worker:
            assert worker.ready["mtp"]["sampling_recommendation_policy"] == prior.POLICY
            corpus = (prior.ROOT / "runtime/lily-one-draft/results/model-corpus.rs").read_text()
            ids = prior.exact_prefix(worker.tokenize(corpus), 512)
            def text():
                return prior.digest(worker.request({"operation": "generate", "input_token_ids": ids,
                    "context_tokens": 1024, "max_new_tokens": 32, "seed": 42, "sampling": prior.TOPK, "mtp": True}))
            first = text()
            assert first == before
            report["checks"].append({"case": "text_before_image", "result": first, "passed": True})
            image = prior.digest(worker.image_transaction(8192))
            assert image["drafts"] == 0 and image["enabled"] is False
            assert all(value == 0 for value in image["counts"].values()) and image["reused_input_tokens"] == 0
            report["checks"].append({"case": "image", "result": image, "passed": True})
            final = text()
            assert final == before
            report["checks"].append({"case": "text_after_image", "result": final, "passed": True})
        report["passed"] = unchanged() and prior.sha256_file(prior.CAND) == report["candidate_sha256"]
    except BaseException as error:
        report["passed"] = False
        report["failure_type"] = type(error).__name__
    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": report["passed"], "checks": len(report["checks"]), "failure_type": report.get("failure_type")}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
