"""Reproduce a source-bound greedy divergence; publish numerical aggregates only."""

import argparse
import ast
import contextlib
import gc
import hashlib
import importlib.util
import json
import logging
import os
import selectors
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest_file(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def load_validator():
    spec = importlib.util.spec_from_file_location("validation", ROOT / "scripts/validate_colgrep_generation.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class Protocol:
    def __init__(self, executable, model, model_flag):
        self.process = subprocess.Popen(
            [str(executable), model_flag, str(model)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        try:
            self.ready = self.read()
            if self.ready.get("ready") is not True:
                raise RuntimeError("native_not_ready")
        except Exception:
            with contextlib.suppress(Exception):
                self.close()
            raise

    def read(self):
        if not self.selector.select(300):
            raise RuntimeError("native_timeout")
        data = self.process.stdout.readline(1_048_577)
        if not data or not data.endswith(b"\n") or len(data) > 1_048_576:
            raise RuntimeError("native_protocol_invalid")
        return json.loads(data)

    def request(self, request):
        self.process.stdin.write(json.dumps(request).encode() + b"\n")
        self.process.stdin.flush()
        result = self.read()
        if "error" in result:
            raise RuntimeError("native_request_failed")
        return result

    def close(self):
        with contextlib.suppress(BrokenPipeError):
            self.process.stdin.close()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)
        diagnostics = self.process.stderr.read()
        self.selector.close()
        self.process.stdout.close()
        self.process.stderr.close()
        if diagnostics or self.process.returncode != 0:
            raise RuntimeError("native_close_failed")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def locations(validator, text):
    parsed = validator.calls(text)
    values = [parameters.get("locations", "") for name, parameters in parsed if name == "finish"]
    if len(values) != 1:
        raise RuntimeError("finish_contract_invalid")
    entries = ast.literal_eval(values[0])
    location_set = set(entries)
    covered = set()
    for entry in entries:
        filename, lines = entry.rsplit(":", 1)
        start, end = map(int, lines.split("-", 1))
        covered.update((filename, line) for line in range(start, end + 1))
    return location_set, covered


def margin_and_ranks(torch, logits, native_token, reference_token):
    top = torch.topk(logits, 2)
    maximum = float(top.values[0])
    return {
        "top1_top2_margin": maximum - float(top.values[1]),
        "native_choice_gap_from_top1": maximum - float(logits[native_token]),
        "reference_choice_gap_from_top1": maximum - float(logits[reference_token]),
        "native_choice_rank_strict": int((logits > logits[native_token]).sum()) + 1,
        "reference_choice_rank_strict": int((logits > logits[reference_token]).sum()) + 1,
        "top_tie_count": int((logits == top.values[0]).sum()),
        "native_choice_is_argmax": int(torch.argmax(logits)) == native_token,
        "reference_choice_is_argmax": int(torch.argmax(logits)) == reference_token,
    }


def difference(torch, left, right):
    delta = (left.float() - right.float()).double()
    return {
        "max_abs_error": float(delta.abs().max()),
        "mean_abs_error": float(delta.abs().mean()),
        "rmse": float(torch.sqrt((delta * delta).mean())),
        "finite": bool(torch.isfinite(delta).all()),
    }


def seal_model(model):
    return {
        path.name: digest_file(path)
        for path in sorted(model.iterdir())
        if path.is_file() and path.suffix in {".json", ".safetensors"}
    }


def seal_harness(validator, upstream):
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=upstream).decode().strip()
    if revision != validator.UPSTREAM:
        raise RuntimeError("upstream_revision_mismatch")
    assets = {path.name: digest_file(path) for path in sorted((upstream / "agent/assets").iterdir()) if path.is_file()}
    if any(assets.get(name) != expected for name, expected in validator.ASSET_HASHES.items()):
        raise RuntimeError("upstream_content_mismatch")
    return assets


def compiler_version():
    executable = shutil.which("rustc") or str(Path.home() / ".cargo/bin/rustc")
    return subprocess.check_output([executable, "--version"]).decode().strip()


def run(args, progress):
    if args.report.resolve() == args.reference_report.resolve():
        raise RuntimeError("report_destination_invalid")
    validator = load_validator()
    validator_hash = digest_file(ROOT / "scripts/validate_colgrep_generation.py")
    script_hash = digest_file(Path(__file__))
    tools_lock_hash = digest_file(ROOT / "tools/uv.lock")
    cargo_lock_hash = digest_file(ROOT / "Cargo.lock")
    original = json.loads(args.reference_report.read_text())
    original_hash = digest_file(args.reference_report)
    original_case = next(case for case in original["cases"] if case["case"] == "finish")
    source_seal = validator.seal_source(args.source)
    model_seal = seal_model(args.model)
    harness_seal = seal_harness(validator, args.upstream)
    generation_binary = args.worker
    score_binary = args.score_worker
    generation_hash = digest_file(generation_binary)
    score_hash = digest_file(score_binary)
    if (
        original["revision"] != validator.REVISION
        or original["upstream_harness_revision"] != validator.UPSTREAM
        or original["worker_sha256"] != generation_hash
        or original["source_files"] != source_seal
        or original["model_files"] != model_seal
        or original["harness_files"] != harness_seal
        or original["tools_lock_sha256"] != tools_lock_hash
        or original["cargo_lock_sha256"] != cargo_lock_hash
        or digest_file(args.source / "chat_template.jinja") != harness_seal["chat_template.jinja"]
    ):
        raise RuntimeError("reference_report_binding_failed")
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logging.disable(logging.CRITICAL)
    warnings.filterwarnings("ignore")
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(8)
    expected_environment = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "dtype": "bfloat16",
        "device": "mps",
        "attention": "eager",
    }
    if any(original["reference"].get(name) != value for name, value in expected_environment.items()):
        raise RuntimeError("reference_environment_mismatch")
    tokenizer = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
    case = next(case for case in validator.fixtures(tokenizer, args.upstream)[0] if case["name"] == "finish")
    started = time.monotonic()
    with validator.Worker(generation_binary, args.model) as worker:
        native = worker.request(
            {"operation": "generate", "input_token_ids": case["ids"], "context_tokens": 4096, "max_new_tokens": 256}
        )
    progress.write(json.dumps({"phase": "native_finish_completed"}) + "\n")
    progress.flush()
    native_ids = native["generated_token_ids"]
    native_locations, native_coverage = locations(validator, native["text"])

    model = (
        AutoModelForCausalLM.from_pretrained(
            args.source, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager"
        )
        .to("mps")
        .eval()
    )
    inputs = torch.tensor([case["ids"]], device="mps")
    with torch.inference_mode():
        output = model.generate(
            inputs,
            attention_mask=torch.ones_like(inputs),
            max_new_tokens=256,
            do_sample=False,
            use_cache=True,
            eos_token_id=sorted(validator.EOS),
            pad_token_id=1,
            return_dict_in_generate=True,
            output_logits=True,
        )
    reference_full = output.sequences[0, len(case["ids"]) :].tolist()
    reference_eos = reference_full[-1] if reference_full[-1] in validator.EOS else None
    reference_ids = reference_full[:-1] if reference_eos is not None else reference_full
    native_ids_hash = validator.digest_ids(native_ids)
    reference_ids_hash = validator.digest_ids(reference_ids)
    if (
        native_ids_hash != original_case["generated_ids_sha256"]
        or reference_ids_hash != original_case["reference_ids_sha256"]
        or len(native_ids) != original_case["generated_tokens"]
        or len(reference_ids) != original_case["reference_tokens"]
        or len(case["ids"]) != original_case["input_tokens"]
    ):
        raise RuntimeError("original_finish_outputs_not_reproduced")
    common = 0
    for native_id, reference_id in zip(native_ids, reference_ids, strict=False):
        if native_id != reference_id:
            break
        common += 1
    if common >= min(len(native_ids), len(reference_ids)):
        raise RuntimeError("first_divergence_missing")
    native_token, reference_token = native_ids[common], reference_ids[common]
    cached_reference_logits = output.logits[common][0].float().cpu()
    reference_text = tokenizer.decode(reference_ids, skip_special_tokens=False)
    reference_locations, reference_coverage = locations(validator, reference_text)
    shared_prefix = case["ids"] + reference_ids[:common]
    forced_inputs = torch.tensor([shared_prefix], device="mps")
    with torch.inference_mode():
        forced = model(
            input_ids=forced_inputs,
            attention_mask=torch.ones_like(forced_inputs),
            use_cache=False,
            logits_to_keep=1,
        )
    forced_reference_logits = forced.logits[0, -1].float().cpu()
    candidates = set(torch.topk(cached_reference_logits, 2).indices.tolist())
    candidates.update(torch.topk(forced_reference_logits, 2).indices.tolist())
    candidates.update([native_token, reference_token])
    candidates = sorted(candidates)
    base = {
        "schema": 1,
        "scope": "finish_divergence_diagnosis",
        "reference_report_sha256": original_hash,
        "original_reference_passed": original["passed"],
        "native_generated_ids_sha256": native_ids_hash,
        "reference_generated_ids_sha256": reference_ids_hash,
        "original_finish_output_binding": True,
        "model_revision": validator.REVISION,
        "upstream_harness_revision": validator.UPSTREAM,
        "reference": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "dtype": "bfloat16",
            "device": "mps",
            "attention": "eager",
        },
        "native_tokens": len(native_ids),
        "reference_tokens": len(reference_ids),
        "common_prefix_tokens": common,
        "identical_teacher_forced_prefix_tokens": len(shared_prefix),
        "finish_location_set_exact": native_locations == reference_locations,
        "finish_line_coverage_exact": native_coverage == reference_coverage,
        "native_locations_valid": validator.finish_valid(validator.calls(native["text"])),
        "reference_locations_valid": validator.finish_valid(validator.calls(reference_text)),
        "eos_exact": native["stop_token_id"] == reference_eos,
        "cached_bf16_reference": margin_and_ranks(torch, cached_reference_logits, native_token, reference_token),
        "teacher_forced_bf16_reference": margin_and_ranks(
            torch, forced_reference_logits, native_token, reference_token
        ),
        "cached_vs_teacher_forced_bf16_full_vocab": difference(
            torch, cached_reference_logits, forced_reference_logits
        ),
        "generation_worker_sha256": generation_hash,
        "score_worker_sha256": score_hash,
        "diagnosis_script_sha256": script_hash,
        "validator_sha256": validator_hash,
        "tools_lock_sha256": tools_lock_hash,
        "cargo_lock_sha256": cargo_lock_hash,
        "source_files": source_seal,
        "model_files": model_seal,
        "harness_files": harness_seal,
        "generation_named_build": {
            key: original["named_build"][key]
            for key in ["source_commit", "profile", "package", "features", "rust_toolchain"]
        },
        "score_build_declaration": {
            "profile": "release",
            "rustc": compiler_version(),
            "package_features": ["probability-scoring"]
            + (["optimized"] if args.score_feature_profile == "optimized" else []),
            "engine_features": ["metal"]
            + (
                [
                    "deferred-state-init",
                    "scoring-last-layer-pruning",
                    "direct-delta-conv",
                    "rope-cache",
                    "compact-delta-conv",
                    "attention-kv-only",
                ]
                if args.score_feature_profile == "optimized"
                else []
            ),
            "feature_source": "caller_declared_build_profile_not_inferred_from_binary",
        },
    }
    del output, inputs, forced, forced_inputs, model
    gc.collect()
    torch.mps.empty_cache()
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.source, local_files_only=True, dtype=torch.float32, attn_implementation="eager"
        )
        .to("mps")
        .eval()
    )
    forced_inputs = torch.tensor([shared_prefix], device="mps")
    with torch.inference_mode():
        forced = model(
            input_ids=forced_inputs, attention_mask=torch.ones_like(forced_inputs), use_cache=False, logits_to_keep=1
        )
    f32_reference_logits = forced.logits[0, -1].float().cpu()
    candidates = sorted(set(candidates) | set(torch.topk(f32_reference_logits, 2).indices.tolist()))
    base["teacher_forced_f32_reference"] = margin_and_ranks(torch, f32_reference_logits, native_token, reference_token)
    base["teacher_forced_bf16_vs_f32_full_vocab"] = difference(torch, forced_reference_logits, f32_reference_logits)
    del forced, forced_inputs, model
    gc.collect()
    torch.mps.empty_cache()
    progress.write(json.dumps({"phase": "reference_finish_and_shared_prefix_completed"}) + "\n")
    progress.flush()

    with validator.Worker(generation_binary, args.model) as worker:
        fresh = worker.request(
            {"operation": "generate", "input_token_ids": shared_prefix, "context_tokens": 4096, "max_new_tokens": 1}
        )
    fresh_token = fresh["generated_token_ids"][0]
    if fresh_token not in candidates:
        candidates.append(fresh_token)
        candidates.sort()
    with Protocol(score_binary, args.model, "--score-model-path") as scorer:
        scored = scorer.request(
            {
                "operation": "score",
                "input_token_ids": shared_prefix,
                "candidate_token_ids": candidates,
                "context_tokens": 4096,
                "temperature": 1.0,
            }
        )
    native_logits = torch.tensor(scored["logits"], dtype=torch.float32)
    cached_candidates = cached_reference_logits[candidates]
    forced_candidates = forced_reference_logits[candidates]
    candidate_native_token_index = candidates.index(native_token)
    candidate_reference_token_index = candidates.index(reference_token)
    native_gap = float(native_logits[candidate_reference_token_index] - native_logits[candidate_native_token_index])
    native_subset_maximum = float(native_logits.max())
    base.update(
        {
            "candidate_subset_count": len(candidates),
            "native_logit_scope": "candidate_subset_only_not_full_vocabulary_or_original_incremental_stream",
            "native_cached_vs_fresh_argmax_exact": fresh_token == native_token,
            "fresh_native_matches_cached_reference_argmax": fresh_token == reference_token,
            "fresh_native_matches_teacher_forced_reference_argmax": fresh_token
            == int(torch.argmax(forced_reference_logits)),
            "native_score_subset_argmax_matches_fresh_stream": candidates[int(torch.argmax(native_logits))]
            == fresh_token,
            "native_score_reference_minus_native_choice_gap": native_gap,
            "native_score_native_choice_gap_from_subset_top1": native_subset_maximum
            - float(native_logits[candidate_native_token_index]),
            "native_score_subset_top_tie_count": int((native_logits == native_logits.max()).sum()),
            "native_score_vs_cached_reference_candidate_subset": difference(torch, native_logits, cached_candidates),
            "native_score_vs_teacher_forced_reference_candidate_subset": difference(
                torch, native_logits, forced_candidates
            ),
            "native_score_vs_f32_reference_candidate_subset": difference(
                torch, native_logits, f32_reference_logits[candidates]
            ),
            "cached_reference_margin_within_observed_candidate_error": (
                base["cached_bf16_reference"]["top1_top2_margin"]
                <= float((native_logits - cached_candidates).abs().max())
            ),
            "elapsed_seconds": time.monotonic() - started,
        }
    )
    controls = {
        "source_unchanged": source_seal == validator.seal_source(args.source),
        "model_unchanged": model_seal == seal_model(args.model),
        "harness_unchanged": harness_seal == seal_harness(validator, args.upstream),
        "reference_report_unchanged": original_hash == digest_file(args.reference_report),
        "generation_worker_unchanged": generation_hash == digest_file(generation_binary),
        "score_worker_unchanged": score_hash == digest_file(score_binary),
        "validator_unchanged": validator_hash == digest_file(ROOT / "scripts/validate_colgrep_generation.py"),
        "diagnosis_script_unchanged": script_hash == digest_file(Path(__file__)),
        "tools_lock_unchanged": tools_lock_hash == digest_file(ROOT / "tools/uv.lock"),
        "cargo_lock_unchanged": cargo_lock_hash == digest_file(ROOT / "Cargo.lock"),
    }
    base["controls"] = controls
    base["diagnosis_reproduced"] = all(controls.values())
    fresh_reference_agreement = (
        base["fresh_native_matches_teacher_forced_reference_argmax"]
        and base["native_score_subset_argmax_matches_fresh_stream"]
        and base["native_score_vs_teacher_forced_reference_candidate_subset"]["max_abs_error"] == 0.0
    )
    base["interpretation"] = (
        "Fresh native generation and score agree with BF16 teacher forcing on the selected candidates. "
        "The original incremental native choice differs at a small reference margin. "
        "Generation execution/history numerical drift is the leading explanation; "
        "the original incremental logits and a specific kernel root cause were not observed."
        if fresh_reference_agreement
        else "The observed differences require further diagnosis. The original incremental native logits "
        "and a specific kernel root cause were not observed."
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(base, indent=2) + "\n")
    progress.write(
        json.dumps(
            {
                "phase": "diagnosis_completed",
                "common_prefix_tokens": common,
                "finish_location_set_exact": base["finish_location_set_exact"],
                "cached_vs_fresh_argmax_exact": base["native_cached_vs_fresh_argmax_exact"],
            }
        )
        + "\n"
    )
    progress.flush()
    return 0 if base["diagnosis_reproduced"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["source", "model", "worker", "score-worker", "upstream", "reference-report", "report"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--score-feature-profile", choices=["default", "optimized"], default="default")
    args = parser.parse_args()
    # Native imports cannot emit diagnostics or exception dumps into the transcript.
    saved_stdout = os.dup(sys.stdout.fileno())
    with os.fdopen(saved_stdout, "w") as progress, open(os.devnull, "w") as null:
        os.dup2(null.fileno(), sys.stdout.fileno())
        os.dup2(null.fileno(), sys.stderr.fileno())
        try:
            raise SystemExit(run(args, progress))
        except Exception:
            progress.write("numerical_diagnosis_failed\n")
            progress.flush()
            raise SystemExit(1) from None
