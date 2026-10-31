"""Private, source-bound GDN trial; hashes and timing aggregates only.

Run through the locked tools/converter uv environment after all three isolated
workers and their build receipts exist. This controller does not build, change
source, download, rehash model payloads, or enable an experimental runtime flag.
It deliberately refuses a stale payload-hash receipt instead of silently
reading 107 GB while another session may own the machine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[4]
CONTROL = Path("/Users/johnshahbazian/development/stark/runtime/qwen38-mlx-comparison-20261030")
DEFAULT_MODEL = CONTROL / "mlx-model-mixed-4-8bit-788c07a"
DEFAULT_CORPUS = Path("/tmp/stark-lily-research-20261028/src/qwen4exp/model.rs")
MANIFEST_SHA = "11246ebbea8a24a988517ebeb59623b857df89120e5ae3becd48807194542ef6"
PAYLOAD_RECEIPT_SHA = "e112599b932abf197011afe5eb91bc1f8a2bb55716eda470b6ee4ef0fcb275e8"
CORPUS_SHA = "daecdf178c6429a721f9e1b5ae379c664adb8f8af55b381535355636ec469955"
PREFIX_SHA = "fd1222c74a823b3a6cf5ed3ffd3dd22cb627afde17c21cef8b2150ac9e14b322"
OUTPUT_SHA = "cf67e22bcf0d71a09bd998471f860b6da8fd4383a75c82b0bc5a3a00a2b7dafb"
REPOSITORY = "ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit"
REVISION = "788c07a8725642bccf13a0a3b8a21954ab84b4f2"
BASE_COMMIT = "645ad5562f44926341b3a74001ed4243e40db3e9"
ARMS = ("baseline", "staged", "direct")
BLOCKS = ("baseline", "staged", "direct", "direct", "staged", "baseline")
POLICY_REL = "crates/stark-engine/src/engine/hybrid/lily.rs"
POLICY_KEYS = (
    "precision_policy", "head_value_precision_policy", "kv_format", "kv_cache_format",
    "speculation", "mtp", "drafter", "pld",
)


class TrialFailure(RuntimeError):
    """Only fixed, controller-owned categories may be emitted."""


class PrivateParser(argparse.ArgumentParser):
    def error(self, message):
        raise TrialFailure("invalid_arguments")


def check(condition, category):
    if not condition:
        raise TrialFailure(category)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def id_sha(values):
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def safe_failure(exc):
    return {
        "failure_type": type(exc).__name__,
        "failure_category": exc.args[0] if isinstance(exc, TrialFailure)
        else "controller_interrupted" if isinstance(exc, KeyboardInterrupt) else "controller_operation_failed",
    }


def persist(path, report, *, initial=False):
    payload = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    target = path if initial else path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if not initial:
            target.replace(path)
    finally:
        if not initial:
            target.unlink(missing_ok=True)


def checkpoint_snapshot(model):
    snapshot = {}
    for path in sorted(model.rglob("*")):
        relative = path.relative_to(model)
        if ".cache" in relative.parts or not path.is_file():
            continue
        check(not path.is_symlink(), "checkpoint_symlink")
        metadata = path.stat()
        snapshot[relative.as_posix()] = [metadata.st_size, metadata.st_mtime_ns, metadata.st_ino]
    return snapshot


def validate_checkpoint(model, manifest_path, receipt_path):
    check(sha(manifest_path) == MANIFEST_SHA, "checkpoint_manifest_identity")
    check(sha(receipt_path) == PAYLOAD_RECEIPT_SHA, "checkpoint_hash_receipt_identity")
    manifest = json.loads(manifest_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    check(manifest.get("repository") == REPOSITORY and manifest.get("revision") == REVISION,
          "checkpoint_manifest_revision")
    check(receipt.get("schema") == "stark-shared-checkpoint-payload-hashes-v1", "checkpoint_receipt_schema")
    check(receipt.get("repository") == REPOSITORY and receipt.get("revision") == REVISION,
          "checkpoint_receipt_revision")
    check(receipt.get("manifest_sha256") == MANIFEST_SHA and receipt.get("completed") is True,
          "checkpoint_receipt_completion")
    check(receipt.get("all_manifest_lfs_payloads_verified") is True, "checkpoint_payload_qualification")
    snapshot = checkpoint_snapshot(model)
    check(snapshot == receipt.get("checkpoint_snapshot"), "checkpoint_stat_identity_changed")
    files = manifest.get("files")
    check(isinstance(files, list) and len(files) == 113, "checkpoint_manifest_file_count")
    check({item["name"] for item in files} == set(snapshot), "checkpoint_file_set")
    check(sum(item[0] for item in snapshot.values()) == 107326639344, "checkpoint_total_bytes")
    hashes = receipt.get("payload_sha256", {})
    payload_names = {item["name"] for item in files if item.get("lfs_sha256")}
    check(set(hashes) == payload_names, "checkpoint_payload_receipt_file_set")
    for item in files:
        check(snapshot[item["name"]][0] == item["bytes"], "checkpoint_file_size")
        if item.get("lfs_sha256"):
            check(hashes[item["name"]] == item["lfs_sha256"], "checkpoint_payload_hash_binding")
    check(sha(model / "config.json") == manifest.get("config_sha256"), "checkpoint_config_identity")
    # Bind the actual tokenizer read, independently of the stat-based weight gate.
    tokenizer = next(item for item in files if item["name"] == "tokenizer.json")
    check(sha(model / "tokenizer.json") == tokenizer["lfs_sha256"], "checkpoint_tokenizer_identity")
    check(checkpoint_snapshot(model) == snapshot, "checkpoint_changed_during_validation")
    return snapshot


def relative_file(root, name):
    relative = PurePosixPath(name)
    check(not relative.is_absolute() and bool(relative.parts)
          and all(part not in (".", "..") for part in relative.parts), "source_relative_path")
    path = root.joinpath(*relative.parts)
    check(path.resolve().is_relative_to(root.resolve()), "source_path_escape")
    return path


def source_map(root):
    result = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            check(path.resolve().is_relative_to(root.resolve()), "source_symlink_escape")
            result[relative] = hashlib.sha256(os.readlink(path).encode()).hexdigest()
        elif path.is_file():
            result[relative] = sha(path)
    return result


def tree_sha(files):
    digest = hashlib.sha256()
    for relative, hashed in sorted(files.items()):
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(hashed))
    return digest.hexdigest()


def validate_source_policy(root):
    source = "".join((root / POLICY_REL).read_text().split())
    check("Qwen4ExpModel::load_with(&context,path,NgramStorage::Paged,false,VisionMode::Off," in source,
          "source_no_mtp_policy")
    check("model.set_kv_format(ifmodel.config.official_mlx_rounding(){KvFormat::Q8Affine64" in source,
          "source_official_kv_policy")


def seal_preparation(trial_root):
    preparation_path = trial_root / "preparation.json"
    preparation = json.loads(preparation_path.read_text())
    preparation_sha = sha(preparation_path)
    check(preparation.get("schema") == "qwen38-gdn-prework-model-trial-preparation-v1", "preparation_schema")
    check(preparation.get("base_commit") == BASE_COMMIT, "preparation_base_commit")
    check(set(preparation.get("source_manifests", {})) == set(ARMS), "preparation_arm_set")
    sealed = {"preparation_sha256": preparation_sha, "arms": {}}
    for arm in ARMS:
        item = preparation["source_manifests"][arm]
        root = relative_file(trial_root, item["source_root"])
        manifest_path = relative_file(trial_root, item["manifest_path"])
        check(sha(manifest_path) == item["source_manifest_sha256"], "source_manifest_identity")
        manifest = json.loads(manifest_path.read_text())
        check(manifest.get("variant") == arm and manifest.get("base_commit") == BASE_COMMIT,
              "source_manifest_arm")
        files = source_map(root)
        check(files == item["files"] == manifest.get("files"), "source_file_identity")
        check(len(files) == item["file_count"] == manifest.get("file_count"), "source_file_count")
        check(tree_sha(files) == item["source_tree_manifest_sha256"]
              == manifest.get("source_tree_manifest_sha256"), "source_tree_identity")
        validate_source_policy(root)
        binary = trial_root / "builds" / arm / "release/stark-generate"
        build_receipt_path = trial_root / "builds" / arm / "build-receipt.json"
        receipt = json.loads(build_receipt_path.read_text())
        check(receipt.get("schema") == "qwen38-gdn-prework-build-receipt-v1"
              and receipt.get("completed") is True, "build_receipt_completion")
        check(receipt.get("locked_toolchain") == "1.98.1", "build_toolchain")
        command = receipt.get("command")
        check(isinstance(command, str | list) and bool(command), "build_command_missing")
        command_text = command if isinstance(command, str) else " ".join(command)
        check(all(token in command_text for token in
                  ("+1.98.1", "build", "--locked", "--release", "stark-generate")), "build_command_policy")
        check(receipt.get("preparation_sha256") == preparation_sha, "build_preparation_binding")
        check(receipt.get("source_manifest_sha256") == item["source_manifest_sha256"]
              and receipt.get("source_tree_manifest_sha256") == item["source_tree_manifest_sha256"],
              "build_source_binding")
        worker_sha = sha(binary)
        check(worker_sha == receipt.get("worker_sha256"), "build_worker_binding")
        sealed["arms"][arm] = {
            "worker_sha256": worker_sha,
            "build_receipt_sha256": sha(build_receipt_path),
            "source_manifest_sha256": item["source_manifest_sha256"],
            "source_tree_manifest_sha256": item["source_tree_manifest_sha256"],
            "source_file_count": len(files),
        }
    return sealed


def process_snapshot():
    output = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,comm="], check=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=10,
    ).stdout
    rows = []
    for line in output.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) == 3:
            rows.append((int(parts[0]), int(parts[1]), Path(parts[2]).name))
    return rows


def competition_category(command):
    name = command.lower()
    if re.fullmatch(r"python(?:\d+(?:\.\d+)*)?(?:\.exe)?", name) or "pytest" in name:
        return "possible_python_model_or_benchmark"
    if name in {"cargo", "rustc", "rustfmt", "metal", "metallib"}:
        return "compiler_or_build"
    if any(word in name for word in (
        "stark-generate", "stark-score", "mlx-serve", "llama-server", "llama-cli", "ollama",
        "gdn_prework_research", "hybrid_profile", "benchmark", "lmstudio", "lm studio",
    )):
        return "other_model_or_benchmark"
    return None


GPU_MODEL_MAPPING_MARKERS = (
    "mlx", "torch", "metal.framework", "metalperformance", "tensorflow", "coreml.framework",
    "onnxruntime", "libcuda", "ggml",
)


def small_python_metadata(pid):
    """A Python name alone does not establish competing model or GPU work."""
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "pid=,ppid=,rss=,lstart=,comm="],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=10,
    )
    if result.returncode == 1 and not result.stdout.strip():
        return {"pid": pid, "inspection_status": "vanished_before_inspection"}
    check(result.returncode == 0, "python_metadata_inspection_failed")
    output = result.stdout.strip().split(None, 8)
    check(len(output) == 9, "python_metadata_inspection_failed")
    actual_pid, parent, rss = map(int, output[:3])
    check(actual_pid == pid and rss >= 0, "python_metadata_inspection_failed")
    metadata = {
        "pid": pid, "ppid": parent, "rss_kib": rss, "comm": Path(output[8]).name,
        "start_time": " ".join(output[3:8]),
    }
    mapped = subprocess.run(
        ["lsof", "-a", "-p", str(pid), "-d", "txt", "-Ffn"],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=10,
    )
    if mapped.returncode != 0 or not any(line.startswith("n") for line in mapped.stdout.splitlines()):
        live = subprocess.run(
            ["ps", "-p", str(pid), "-o", "pid="],
            check=False, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=10,
        )
        if live.returncode == 1 and not live.stdout.strip():
            return {**metadata, "inspection_status": "vanished_before_mapping_inspection"}
        check(False, "python_mapping_inspection_failed")
    paths = sorted({line[1:] for line in mapped.stdout.splitlines() if line.startswith("n")})
    check(bool(paths), "python_mapping_inspection_failed")
    markers = sorted({marker for path in paths for marker in GPU_MODEL_MAPPING_MARKERS if marker in path.lower()})
    return {
        **metadata, "inspection_status": "inspected", "txt_mapping_sha256": identity(paths),
        "txt_mapping_count": len(paths), "gpu_or_model_mapping_markers": markers,
    }


def reviewed_background_metadata(pid):
    """Bind one explicitly reviewed small Python daemon using metadata only."""
    output = subprocess.run(
        ["ps", "-p", str(pid), "-o", "pid=,ppid=,rss=,lstart=,comm="], check=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=10,
    ).stdout.strip().split(None, 8)
    check(len(output) == 9, "reviewed_background_metadata_missing")
    actual_pid, parent, rss = map(int, output[:3])
    command = output[8]
    check(actual_pid == pid and parent == 1, "reviewed_background_parent_changed")
    check(0 < rss < 65536, "reviewed_background_memory_bound")
    check(competition_category(Path(command).name) == "possible_python_model_or_benchmark",
          "reviewed_background_executable_type")
    mapped = subprocess.run(
        ["lsof", "-a", "-p", str(pid), "-d", "txt", "-Ffn"], check=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=10,
    ).stdout.splitlines()
    check(f"p{pid}" in mapped and "ftxt" in mapped, "reviewed_background_mappings_missing")
    paths = sorted({line[1:] for line in mapped if line.startswith("n")})
    check(bool(paths), "reviewed_background_mappings_missing")
    check(not any(marker in path.lower() for path in paths for marker in GPU_MODEL_MAPPING_MARKERS),
          "reviewed_background_gpu_or_model_mapping")
    return {
        "pid": pid, "ppid": parent, "rss_kib": rss, "comm": command,
        "start_time": " ".join(output[3:8]), "txt_mapping_sha256": identity(paths),
        "txt_mapping_count": len(paths), "gpu_or_model_mapping_markers_absent": True,
    }


class QuietWindow:
    def __init__(self, reviewed_background_pid=None):
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.failed = threading.Event()
        self.worker = None
        self.polls = 0
        self.observations = []
        self.observed_small_python = {}
        self.failure_category = None
        self.ancestors = {os.getpid()}
        rows = process_snapshot()
        parents = {pid: parent for pid, parent, _ in rows}
        current = os.getpid()
        while current in parents and parents[current] not in self.ancestors:
            current = parents[current]
            self.ancestors.add(current)
        self.reviewed_background = (
            reviewed_background_metadata(reviewed_background_pid) if reviewed_background_pid is not None else None
        )
        self.reviewed_background_initial_rss = (
            self.reviewed_background["rss_kib"] if self.reviewed_background else None
        )
        self.thread = threading.Thread(target=self._run, name="gdn-trial-quiet-window", daemon=True)

    def _stop_own_worker(self):
        worker = self.worker
        if worker is None or worker.process.poll() is not None:
            return
        try:
            if os.getpgid(worker.process.pid) == worker.process.pid:
                os.killpg(worker.process.pid, signal.SIGTERM)
            else:
                worker.process.terminate()
        except ProcessLookupError:
            pass

    def poll(self):
        with self.lock:
            if self.failed.is_set():
                return
            try:
                rows = process_snapshot()
                self.polls += 1
                allowed = self.ancestors | ({self.worker.process.pid} if self.worker else set())
                if self.reviewed_background is not None:
                    current = reviewed_background_metadata(self.reviewed_background["pid"])
                    check(all(current[key] == self.reviewed_background[key] for key in (
                        "pid", "ppid", "comm", "start_time", "txt_mapping_sha256",
                    )), "reviewed_background_identity_changed")
                    self.reviewed_background = current
                    allowed.add(current["pid"])
                found = []
                for pid, parent, command in rows:
                    if pid in allowed:
                        continue
                    category = competition_category(command)
                    if category == "possible_python_model_or_benchmark":
                        metadata = small_python_metadata(pid)
                        if metadata.get("rss_kib", 0) >= 65536:
                            found.append({**metadata, "category": "python_memory_threshold"})
                            continue
                        if metadata["inspection_status"] != "inspected":
                            self.observed_small_python[pid] = {
                                "ppid": parent, "comm": command, **metadata,
                                "gpu_competition_established": False,
                            }
                            continue
                        markers = metadata["gpu_or_model_mapping_markers"]
                        if markers:
                            found.append({**metadata, "category": "python_gpu_or_model_mapping"})
                        else:
                            self.observed_small_python[pid] = {
                                **metadata, "gpu_competition_established": False,
                            }
                    elif category:
                        found.append({"pid": pid, "ppid": parent, "comm": command, "category": category})
                if found:
                    self.observations = found
                    self.failure_category = "competing_process_detected"
                    self.failed.set()
                    self._stop_own_worker()
            except TrialFailure as exc:
                self.failure_category = exc.args[0]
                self.failed.set()
                self._stop_own_worker()
            except Exception:  # noqa: BLE001 - any watcher failure invalidates the timing window.
                self.failure_category = "process_watcher_failed"
                self.failed.set()
                self._stop_own_worker()

    def _run(self):
        while not self.stop_event.wait(1.0):
            self.poll()

    def start(self):
        self.poll()
        self.check()
        self.thread.start()

    def check(self):
        if self.failed.is_set():
            raise TrialFailure(self.failure_category)

    def snapshot(self):
        with self.lock:
            return {
                "poll_seconds": 1.0, "polls": self.polls,
                "comm_only_no_arguments": True, "foreign_processes_signalled": False,
                "monitor_semantics": "python-metadata-threshold-v2",
                "python_classification": "name triggers metadata inspection; stop at RSS>=65536KiB or GPU/model mappings",
                "observed_small_or_vanished_python": list(self.observed_small_python.values()),
                "limitations": "Sampling can miss short GPU activity; vanished Python does not establish GPU inactivity.",
                "reviewed_background_allowance": None if self.reviewed_background is None else {
                    **self.reviewed_background,
                    "initial_rss_kib": self.reviewed_background_initial_rss,
                    "rss_upper_bound_kib_exclusive": 65536,
                    "basis": "explicit PID review; parent1; stable comm/start time/txt mappings; bounded RSS",
                    "zero_cpu_activity_claimed": False,
                    "children_exempted": False,
                },
                "failed": self.failed.is_set(), "failure_category": self.failure_category,
                "observations": self.observations,
            }

    def close(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=12)


def validate_environment():
    forbidden = [name for name in os.environ if name.startswith("STARK_")]
    forbidden.extend(name for name in (
        "DYLD_INSERT_LIBRARIES", "MTL_CAPTURE_ENABLED", "MTL_DEBUG_LAYER", "MTL_SHADER_VALIDATION",
        "METAL_DEVICE_WRAPPER_TYPE",
    ) if os.environ.get(name))
    check(not forbidden, "profiling_or_runtime_override_environment")


def aggregate(rows):
    rates = [row["decode_tokens_per_second"] for row in rows]
    return {
        "measured_requests": len(rows),
        "pooled_decode_tokens_per_second": sum(row["useful_decode_tokens"] for row in rows)
        / sum(row["decode_seconds"] for row in rows),
        "median_decode_tokens_per_second": statistics.median(rates),
        "minimum_decode_tokens_per_second": min(rates),
        "maximum_decode_tokens_per_second": max(rates),
        "request_decode_tokens_per_second": rates,
        "pooled_prefill_tokens_per_second": sum(row["input_tokens"] for row in rows)
        / sum(row["prefill_seconds"] for row in rows),
        "median_wall_seconds": statistics.median(row["wall_seconds"] for row in rows),
    }


def summarize(report):
    summaries = {}
    globally_clean = report.get("completed") is True and not report["quiet_window"]["failed"]
    for arm in ARMS:
        calls = [row for row in report["rows"] if row["arm"] == arm]
        rows = [row for row in calls if not row["warmup"] and row["status"] in ("complete", "parity_failed")]
        qualified = globally_clean and len(calls) == 8 and len(rows) == 6 \
            and all(row["status"] == "complete" and row.get("historical_output_exact") for row in calls)
        summary = {"qualified": qualified, "completed_calls": sum(row["status"] != "started" for row in calls)}
        if rows:
            summary.update(aggregate(rows))
        blocks = []
        for number in sorted({row["block"] for row in rows}):
            block_rows = [row for row in rows if row["block"] == number]
            blocks.append({"block": number, **aggregate(block_rows)})
        summary["blocks"] = blocks
        if len(blocks) == 2:
            summary["later_vs_earlier_block_pooled_percent"] = (
                blocks[1]["pooled_decode_tokens_per_second"] / blocks[0]["pooled_decode_tokens_per_second"] - 1
            ) * 100
        if not qualified:
            summary["timing_scope"] = "descriptive only; no optimization gain qualification"
        summaries[arm] = summary
    report["summary"] = summaries
    report["comparisons"] = {}
    if summaries["baseline"]["qualified"]:
        base = summaries["baseline"]["pooled_decode_tokens_per_second"]
        for arm in ("staged", "direct"):
            if summaries[arm]["qualified"]:
                report["comparisons"][arm] = {
                    "pooled_decode_change_percent": (summaries[arm]["pooled_decode_tokens_per_second"] / base - 1) * 100,
                    "scope": "six measured requests per arm; order/drift remain; no confidence or sustained-gain claim",
                }
    report["all_arms_qualified"] = all(item["qualified"] for item in summaries.values())


def support_seal(chat_module, tokenizer_module):
    package_root = Path(chat_module.__file__).resolve().parent
    files = {str(path): sha(path) for path in sorted(package_root.rglob("*.py"))}
    files[str(Path(tokenizer_module.__file__).resolve())] = sha(tokenizer_module.__file__)
    extension = tokenizer_module.tokenizers.__file__
    files[str(Path(extension).resolve())] = sha(extension)
    files[str(Path(sys.executable).resolve())] = sha(Path(sys.executable).resolve())
    return identity(files)


def main(argv=None):
    parser = PrivateParser(description=__doc__)
    parser.add_argument("--trial-root", type=Path, default=ROOT / "runtime/gdn-prework-gpu-20261031")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--checkpoint-manifest", type=Path, default=CONTROL / "equivalent-mlx-pack-manifest-v2.json")
    parser.add_argument("--checkpoint-hash-receipt", type=Path,
                        default=CONTROL / "matched-checkpoint-payload-hashes-v1.json")
    parser.add_argument("--reviewed-background-pid", type=int,
                        help="Explicitly reviewed small parent1 Python daemon; stable metadata rechecked every poll")
    args = parser.parse_args(argv)
    check(args.output.parent.is_dir(), "output_parent_missing")
    report = {
        "schema": "qwen38-gdn-prework-model-trial-v2", "started_utc": datetime.now(UTC).isoformat(),
        "completed": False, "all_arms_qualified": False, "rows": [], "blocks": [],
        "primary_statistic": "pooled_decode_tokens_per_second", "block_order": list(BLOCKS),
        "calls_per_block": 4, "warmups_per_block": 1, "measured_requests_per_arm": 6,
        "input_tokens": 4096, "output_tokens": 256, "useful_decode_numerator": 255,
        "sampling": {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "seed": 42},
        "speculation": False, "prefix_reuse": False, "stored_weights_or_quantization_changed": False,
        "historical_output_token_sha256": OUTPUT_SHA, "checkpoint_manifest_sha256": MANIFEST_SHA,
        "checkpoint_payload_receipt_sha256": PAYLOAD_RECEIPT_SHA,
        "checkpoint_payloads_rehashed_this_run": False, "corpus_sha256": CORPUS_SHA,
        "input_token_sha256": PREFIX_SHA, "driver_sha256": sha(__file__),
        "source_base_commit": BASE_COMMIT, "qualification_scope": "experimental isolated workers only",
        "timing_note": "255 useful post-prefill outputs / native decode_seconds; prefill and transport excluded",
        "uncertainty": "Six measured requests per arm; mirrored blocks expose drift but do not establish confidence.",
        "quiet_window": {"failed": False, "polls": 0},
    }
    persist(args.output, report, initial=True)
    watcher = None
    try:
        validate_environment()
        watcher = QuietWindow(args.reviewed_background_pid)
        watcher.start()
        import tokenizers
        from stark_convert import chat as chat_module
        from tokenizers import Tokenizer

        support = support_seal(chat_module, tokenizers)
        seals = seal_preparation(args.trial_root)
        snapshot = validate_checkpoint(args.model, args.checkpoint_manifest, args.checkpoint_hash_receipt)
        check(sha(args.corpus) == CORPUS_SHA, "corpus_identity")
        tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
        ids = tokenizer.encode(args.corpus.read_text(), add_special_tokens=False).ids[:4096]
        check(len(ids) == 4096 and id_sha(ids) == PREFIX_SHA, "input_prefix_identity")
        report["source_and_worker_seals"] = seals
        report["support_code_sha256"] = support
        report["checkpoint_stat_snapshot_sha256"] = identity(snapshot)
        baseline_ready_policy = None

        def verify_seals():
            watcher.check()
            validate_environment()
            check(sha(__file__) == report["driver_sha256"], "driver_changed")
            check(support_seal(chat_module, tokenizers) == support, "support_code_changed")
            check(seal_preparation(args.trial_root) == seals, "source_or_worker_changed")
            check(checkpoint_snapshot(args.model) == snapshot, "checkpoint_changed")
            check(sha(args.checkpoint_manifest) == MANIFEST_SHA
                  and sha(args.checkpoint_hash_receipt) == PAYLOAD_RECEIPT_SHA, "checkpoint_evidence_changed")
            check(sha(args.corpus) == CORPUS_SHA, "corpus_changed")

        for block, arm in enumerate(BLOCKS):
            verify_seals()
            block_row = {"block": block, "arm": arm, "status": "loading", "before_seals_verified": True}
            report["blocks"].append(block_row)
            report["quiet_window"] = watcher.snapshot()
            persist(args.output, report)
            worker = None
            try:
                binary = args.trial_root / "builds" / arm / "release/stark-generate"
                # The lock prevents the watcher classifying this newly created
                # child during the brief launch-to-registration interval.
                with watcher.lock:
                    watcher.check()
                    worker = chat_module.Worker(binary, args.model, 1800)
                    watcher.worker = worker
                ready = worker.read()
                watcher.check()
                check(ready.get("ready") is True and ready.get("mode") == "text-generation-v1", "worker_ready")
                check(ready.get("state_policy") == "fresh-per-request", "worker_state_policy")
                check(ready.get("max_context_tokens", 0) >= 4352 and ready.get("max_new_tokens", 0) >= 256,
                      "worker_token_limits")
                ready_policy = {key: ready[key] for key in POLICY_KEYS if key in ready}
                for key in ("speculation", "mtp", "drafter", "pld"):
                    check(key not in ready_policy or ready_policy[key] in (False, None, "off", "disabled"),
                          "worker_speculation_policy")
                if baseline_ready_policy is None:
                    check(arm == "baseline", "baseline_ready_order")
                    baseline_ready_policy = ready_policy
                check(ready_policy == baseline_ready_policy, "worker_precision_policy_changed")
                block_row["ready_policy_sha256"] = identity(ready_policy)
                block_row["precision_metadata_exposed"] = bool(ready_policy)
                block_row["status"] = "running"
                for replay in range(4):
                    watcher.check()
                    row = {"block": block, "arm": arm, "replay": replay, "warmup": replay == 0, "status": "started"}
                    report["rows"].append(row)
                    persist(args.output, report)
                    started = time.monotonic()
                    try:
                        result = worker.request({
                            "operation": "generate", "input_token_ids": ids, "context_tokens": 4352,
                            "max_new_tokens": 256, "seed": 42,
                            "sampling": {"mode": "stochastic", "temperature": 1.0, "top_k": 20, "top_p": 0.95},
                        })
                        elapsed = time.monotonic() - started
                        watcher.check()
                        generated = result.get("generated_token_ids")
                        check(isinstance(generated, list) and len(generated) == 256
                              and all(type(value) is int and 0 <= value < ready["vocab_size"] for value in generated),
                              "output_token_ids")
                        check(result.get("sampled_tokens") == 256 and result.get("finish_reason") == "length",
                              "output_count_or_finish")
                        prefill, decode = result.get("prefill_seconds"), result.get("decode_seconds")
                        check(all(type(value) in (int, float) and math.isfinite(value) and value > 0
                                  for value in (prefill, decode)), "request_duration")
                        digest = id_sha(generated)
                        row.update({
                            "status": "complete" if digest == OUTPUT_SHA else "parity_failed",
                            "historical_output_exact": digest == OUTPUT_SHA, "output_token_sha256": digest,
                            "input_tokens": 4096, "output_tokens": 256, "useful_decode_tokens": 255,
                            "prefill_seconds": prefill, "decode_seconds": decode, "wall_seconds": elapsed,
                            "decode_tokens_per_second": 255 / decode,
                        })
                        # A different first baseline is not the historical
                        # reference and cannot qualify either candidate.
                        if block == 0 and replay == 0:
                            check(digest == OUTPUT_SHA, "first_baseline_historical_output_changed")
                    except BaseException as exc:
                        row.update({"status": "interrupted" if watcher.failed.is_set() else "failed", **safe_failure(exc)})
                        if watcher.failed.is_set():
                            row["failure_category"] = watcher.failure_category
                        raise
                    finally:
                        report["quiet_window"] = watcher.snapshot()
                        persist(args.output, report)
                    print(json.dumps({"phase": "request_complete", "block": block, "arm": arm,
                                      "replay": replay, "historical_output_exact": row["historical_output_exact"]}), flush=True)
                block_row["status"] = "complete"
            finally:
                if worker is not None:
                    try:
                        worker.close()
                    finally:
                        with watcher.lock:
                            watcher.worker = None
            verify_seals()
            block_row["after_seals_verified"] = True
            report["quiet_window"] = watcher.snapshot()
            persist(args.output, report)
        verify_seals()
        watcher.poll()
        watcher.check()
        report["completed"] = True
    except BaseException as exc:  # noqa: BLE001 - persist interrupted runs without private exception text.
        report.update(safe_failure(exc))
        if watcher and watcher.failed.is_set():
            report["failure_category"] = watcher.failure_category
        for block_row in report["blocks"]:
            if block_row["status"] in ("loading", "running"):
                block_row["status"] = "interrupted"
    finally:
        if watcher:
            watcher.close()
            report["quiet_window"] = watcher.snapshot()
        summarize(report)
        report["finished_utc"] = datetime.now(UTC).isoformat()
        persist(args.output, report)
    print(json.dumps({"completed": report["completed"], "all_arms_qualified": report["all_arms_qualified"],
                      "failure_category": report.get("failure_category"), "summary": report["summary"],
                      "comparisons": report["comparisons"]}), flush=True)
    return 0 if report["all_arms_qualified"] else 1


if __name__ == "__main__":
    try:
        code = main()
    except SystemExit as exc:
        if exc.code == 0:
            raise
        print(json.dumps({"completed": False, "phase": "preflight", "failure_category": "invalid_arguments"}), flush=True)
        code = 1
    except BaseException as exc:  # noqa: BLE001 - never expose private exception text or tracebacks.
        print(json.dumps({"completed": False, "phase": "preflight", **safe_failure(exc)}), flush=True)
        code = 1
    raise SystemExit(code)
