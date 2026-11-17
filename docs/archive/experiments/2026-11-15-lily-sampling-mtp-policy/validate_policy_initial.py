"""Pinned-artifact sampler policy controls; save hashes/counts only."""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_generation_prefix_cache import Worker
from benchmark_qwen38_mtp import exact_prefix, ids_sha256, sha256_file
from stark_convert.mtp import mtp_capabilities, resolve_mtp

MODEL = Path("/Users/johnshahbazian/Library/Application Support/Stark/models/qwen38-flash-next-lily-q4-xl-0f94842e")
BASE = Path("/Users/johnshahbazian/development/stark-llama-v1-study/runtime/llama-v1-baseline-target/release/stark-generate")
CAND = ROOT / "runtime/build-topk-candidate/release/stark-generate"
OUT = ROOT / "runtime/lily-topk-policy/policy-validation.json"
PRIOR = Path("/Users/johnshahbazian/development/stark-llama-v1-study/runtime/llama-v1-lily-baseline-20261115-v2/identity.json")
TOPK = {"mode": "stochastic", "temperature": 1, "top_k": 20, "top_p": .95}
POLICY = "lily-topk20-one-draft-v1"


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
    os.environ["STARK_REQUEST_BUDGET_MIB"] = "16"
    identity = json.loads(PRIOR.read_text())
    def unchanged():
        return all([s.st_size, s.st_mtime_ns, s.st_ino] == expected
            for name, expected in identity["model_file_identities"].items() for s in [(MODEL / name).stat()])
    assert unchanged()
    corpus = ROOT / "runtime/lily-one-draft/results/model-corpus.rs"
    report = {"attempt": "20261115-LILY-MTP1-04", "harness_sha256": sha256_file(Path(__file__)),
              "worker_sha256": {"before": sha256_file(BASE), "candidate": sha256_file(CAND)},
              "corpus_sha256": sha256_file(corpus), "model_manifest_sha256": identity["model_content_manifest_sha256"],
              "checks": [], "image_scope": "lifecycle only; not image answer quality"}
    build = json.loads((ROOT / "runtime/lily-topk-policy/build-identity.json").read_text())
    assert report["worker_sha256"]["before"] == "61b9c8cc724407573169eb2abad7f908e4943eead7ac3122eeec3ef841a48640"
    assert report["worker_sha256"]["candidate"] == build["candidate_sha256"]
    report["native_patch_sha256"] = build["native_patch_sha256"]
    report["python_patch_sha256"] = build["python_patch_sha256"]
    references = {}
    try:
        for label, binary, vision in [("before_text", BASE, False), ("before_vision", BASE, True),
                                      ("candidate_text", CAND, False), ("candidate_vision", CAND, True)]:
            candidate = label.startswith("candidate")
            print(json.dumps({"stage": "policy_controls", "worker": label}), flush=True)
            with Worker(str(binary), str(MODEL), prefix_cache=True, vision=vision, timeout=900) as worker:
                capability = mtp_capabilities(worker.ready)
                assert capability["recommended_drafts"] == 2 and capability["default_drafts"] == 0
                assert capability["max_drafts"] == 3
                assert capability.get("sampling_recommendation_policy") == (POLICY if candidate else None)
                assert ("image_input" in worker.ready) is vision
                ids = exact_prefix(worker.tokenize(corpus.read_text()), 512)
                for name, selector, depth in [
                    ("explicit_zero", {"mtp_drafts": 0}, 0), ("explicit_one", {"mtp_drafts": 1}, 1),
                    ("explicit_two", {"mtp_drafts": 2}, 2), ("explicit_three", {"mtp_drafts": 3}, 3),
                    ("auto", {"mtp": True}, 1 if candidate else 2),
                    ("off", {"mtp": False}, 0), ("omitted", {}, 0),
                ]:
                    result = digest(worker.request({"operation": "generate", "input_token_ids": ids,
                        "context_tokens": 1024, "max_new_tokens": 32, "seed": 42,
                        "sampling": TOPK, **selector}))
                    assert result["drafts"] == depth and result["enabled"] is (depth > 0)
                    assert result["reused_input_tokens"] == 0
                    assert (result["counts"]["proposed"] > 0) is (depth > 0)
                    if depth == 0:
                        assert all(value == 0 for value in result["counts"].values())
                    if label == "before_text" and name.startswith("explicit_"):
                        references[depth] = result
                    else:
                        assert result == references[depth]
                    report["checks"].append({"worker": label, "case": name, "result": result, "passed": True})
                if candidate:
                    for name, sampling, depth in [
                        ("greedy", {"mode": "greedy"}, 2),
                        ("nucleus", {"mode": "stochastic", "temperature": 1, "top_p": .95}, 2),
                        ("near_top_k", {**TOPK, "top_k": 21}, 2),
                        ("near_top_p", {**TOPK, "top_p": .94}, 2),
                        ("near_temperature", {**TOPK, "temperature": .9}, 2),
                        ("min_p_zero", {**TOPK, "min_p": 0}, 2),
                        ("same_native_f32", {**TOPK, "temperature": 1.0000000001, "top_p": .9500000001}, 1),
                    ]:
                        assert resolve_mtp(True, capability, sampling) == depth
                        results = []
                        for selector in ({"mtp": True}, {"mtp_drafts": depth}):
                            results.append(digest(worker.request({"operation": "generate", "input_token_ids": ids,
                                "context_tokens": 1024, "max_new_tokens": 16, "seed": 42,
                                "sampling": sampling, **selector})))
                        assert results[0] == results[1] and results[0]["drafts"] == depth
                        report["checks"].append({"worker": label, "case": name, "auto_matches_explicit": True,
                            "result": results[0], "passed": True})
                    plan = worker.request({"operation": "plan_generation", "input_tokens": 512,
                        "input_token_ids": ids, "context_tokens": 1024, "sampling": TOPK, "mtp": True})
                    assert "error" not in plan
                    report["checks"].append({"worker": label, "case": "automatic_plan", "passed": True,
                        "plan": {k: v for k, v in plan.items() if k in ("context_tokens", "max_new_tokens", "mtp_drafts")}})
                if label == "candidate_vision":
                    image = digest(worker.image_transaction(8192))
                    assert image["drafts"] == 0 and not image["enabled"]
                    assert image["counts"]["proposed"] == 0 and image["reused_input_tokens"] == 0
                    report["checks"].append({"worker": label, "case": "image", "result": image, "passed": True})
                    result = digest(worker.request({"operation": "generate", "input_token_ids": ids,
                        "context_tokens": 1024, "max_new_tokens": 32, "seed": 42, "sampling": TOPK, "mtp": True}))
                    assert result == references[1]
                    report["checks"].append({"worker": label, "case": "text_after_image", "result": result, "passed": True})
        report["passed"] = unchanged() and sha256_file(CAND) == report["worker_sha256"]["candidate"]
    except BaseException as error:
        report["passed"] = False
        report["failure_type"] = type(error).__name__
    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": report["passed"], "checks": len(report["checks"]), "failure_type": report.get("failure_type")}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
