"""Combine source-bound content-free qualification receipts."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRIVATE = ROOT / "runtime/lily-topk-policy"
ARCHIVE = ROOT / "docs/archive/experiments/2026-11-15-lily-sampling-mtp-policy"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def check_content_boundary(value):
    if isinstance(value, dict):
        assert not (set(value) & {"text", "prompt", "rendered_prompt", "input_token_ids",
                                 "generated_token_ids", "token_ids", "logits", "probabilities"})
        for child in value.values():
            check_content_boundary(child)
    elif isinstance(value, list):
        for child in value:
            check_content_boundary(child)


def main():
    build = read(PRIVATE / "build-identity.json")
    initial_path = PRIVATE / "policy-validation.json"
    image_path = PRIVATE / "image-policy-continuation.json"
    prefix_path = ROOT / "runtime/lily-mtp-policy-qualification/results.json"
    performance_path = PRIVATE / "performance-private/aggregate.json"
    initial, image, prefix, performance = map(read, [initial_path, image_path, prefix_path, performance_path])
    assert initial["passed"] is False and initial["failure_type"] == "ChatError"
    assert len(initial["checks"]) == 44 and all(row["passed"] for row in initial["checks"])
    assert image["passed"] is True and len(image["checks"]) == 3 and all(row["passed"] for row in image["checks"])
    assert image["predecessor_sha256"] == sha(initial_path)
    assert prefix["passed"] and len(prefix["transition_rows"]) == 7
    assert all(row["output_sha256"] and row["identical_to_fresh"] and row["identical_to_baseline"]
               for row in prefix["transition_rows"])
    assert prefix["cached_matches_baseline"] is True
    assert performance["request_counts"] == {"measured": 64, "cell_warmups": 16, "warmmax4352_setups": 4}
    candidate = build["candidate_sha256"]
    before = build["before_sha256"]
    assert initial["worker_sha256"] == performance["worker_sha256"] == {"before": before, "candidate": candidate}
    assert image["candidate_sha256"] == prefix["candidate_sha256"] == candidate
    assert prefix["baseline_sha256"] == before
    assert initial["harness_sha256"] == sha(ARCHIVE / "validate_policy_initial.py")
    assert image["harness_sha256"] == sha(ARCHIVE / "continue_image_policy.py")
    assert prefix["source_sha256"] == sha(ARCHIVE / "qualify_sampler_prefix_transition.py")
    assert performance["harness_sha256"] == sha(ARCHIVE / "run_performance.py")
    assert performance["publication_harness_sha256"] == sha(ARCHIVE / "publish_performance.py")
    assert initial["native_patch_sha256"] == build["native_patch_sha256"] == performance["native_patch_sha256"]
    assert initial["python_patch_sha256"] == build["python_patch_sha256"] == performance["python_patch_sha256"]
    prefix["prior_benchmark_corpus_sha256"] = prefix.pop("corpus_sha256")
    prefix["synthetic_input_identity"] = {
        "generation_helper_sha256": sha(ROOT / "scripts/validate_generation_prefix_cache.py"),
        "session_kind": "user-assistant", "first_tokens_target": 5000, "seed": 17,
        "tools": 0, "instruction_paragraphs": 24, "follow_up_sentences": [1, 1, 1],
        "actual_input_tokens": 4615, "input_token_hash_recorded": False,
    }
    report = {
        "schema": "stark-lily-sampling-mtp-qualification-v1",
        "attempts": ["20261115-LILY-MTP1-04", "20261115-LILY-MTP1-04b", "20261115-LILY-MTP1-05",
                     performance["attempt"], "20261115-LILY-MTP1-06"],
        "implementation": {key: build[key] for key in ("source_base", "native_patch_sha256", "python_patch_sha256",
                                                     "candidate_sha256", "before_sha256", "rust_toolchain", "build")},
        "source_snapshots": {p.name: sha(p) for p in ARCHIVE.glob("*.py")},
        "receipt_sha256": {"functional_initial": sha(initial_path), "image_continuation": sha(image_path),
                           "prefix": sha(prefix_path), "performance": sha(performance_path)},
        "functional_initial": initial,
        "functional_image_continuation": image,
        "functional_passed_check_count": 47,
        "functional_complete": True,
        "initial_image_helper_failure_preserved": True,
        "original_functional_checks_rerun": 0,
        "prefix": prefix,
        "performance": performance,
        "publication_source_sha256": sha(Path(__file__)),
        "publication_metadata_revision": {
            "prior_publication_sha256": sha(PRIVATE / "qualification-publication-initial.json"),
            "prior_source_sha256": sha(PRIVATE / "publish_qualification_as_initial.py"),
            "reason": "Name the completed performance continuation separately from its zero-worker metadata preflight",
            "native_requests_repeated": 0,
        },
        "adoption": "Narrow automatic sampling policy implemented on branch; no installation or release",
    }
    check_content_boundary(report)
    target = ROOT / "docs/benchmarks/data/lily-sampling-mtp-policy-2026-11-15.json"
    assert not target.exists()
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"functional_checks": 47, "prefix_comparisons": 7,
                      "performance_requests": 64, "performance_endpoint_met": performance["endpoint_met"]}))


if __name__ == "__main__":
    main()
