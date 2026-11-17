"""Before/candidate ABBA using mtp:true; hashes/counts/timings only."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"scripts"))
from validate_generation_prefix_cache import Worker
from benchmark_qwen38_mtp import _validate_result, exact_prefix, ids_sha256, sha256_file
from benchmark_clean_qwen import Monitor, paging
from benchmark_glm_mlx_serve import screen_attempt


def main():
    os.umask(0o077)
    out = ROOT/"runtime/lily-one-draft/default-comparison.json"
    assert not out.exists()
    for name in tuple(os.environ):
        if name.startswith(("STARK_", "UZU_", "LILY_", "GGML_")):
            os.environ.pop(name)
    baseline = Path("/Users/johnshahbazian/development/stark-llama-v1-study/runtime/llama-v1-baseline-target/release/stark-generate")
    candidate = ROOT/"runtime/build-candidate/release/stark-generate"
    model = Path("/Users/johnshahbazian/Library/Application Support/Stark/models/qwen38-flash-next-lily-q4-xl-0f94842e")
    corpus = ROOT/"runtime/lily-one-draft/results/model-corpus.rs"
    previous = json.loads((ROOT/"runtime/lily-one-draft/results/input512-top20-output256-seed42.json").read_text())
    golden = {depth: next(r for r in previous["requests"] if r["depth"] == depth) for depth in (1,2)}
    report = {"attempt": "20261115-LILY-MTP1-02", "order": ["baseline","candidate","candidate","baseline"],
              "worker_sha256": {"baseline": sha256_file(baseline), "candidate": sha256_file(candidate)},
              "harness_sha256": sha256_file(Path(__file__)), "corpus_sha256": sha256_file(corpus),
              "input_tokens": 512, "output_tokens": 256, "seed": 42,
              "sampling": {"mode":"stochastic","temperature":1,"top_k":20,"top_p":.95},
              "host_policy":"all observations retained; no quiet admission", "sessions": []}
    for index,label in enumerate(report["order"]):
        depth = 2 if label == "baseline" else 1
        session = {"label":label,"recommended_drafts":depth,"requests":[]}
        print(json.dumps({"stage":"starting_session","index":index,"worker":label}),flush=True)
        try:
            with Worker(str(baseline if label=="baseline" else candidate), str(model), prefix_cache=True, vision=True, timeout=900) as worker:
                assert worker.ready["mtp"]["recommended_drafts"] == depth
                assert worker.ready["image_input"]["family_profile"] == "qwen38-lily-retained-bf16-vision-v1"
                ids = exact_prefix(worker.tokenize(corpus.read_text()),512)
                assert ids_sha256(ids) == previous["input_sha256"]
                with Monitor() as monitor:
                    for number in range(9):
                        phase = f"session{index}-request{number}"
                        monitor.set_phase(phase)
                        before = paging()
                        start = time.monotonic()
                        result = worker.request({"operation":"generate","input_token_ids":ids,"context_tokens":768,
                            "max_new_tokens":256,"seed":42,"sampling":report["sampling"],"mtp":True})
                        elapsed = time.monotonic()-start
                        item = _validate_result(result,worker.ready,argparse.Namespace(tokens=256),"on",depth,worker.ready["vocab_size"])
                        assert result.get("reused_input_tokens",0) == 0
                        assert item["output_sha256"] == golden[depth]["output_sha256"]
                        for key in ("rounds","proposed","accepted","verified_rows"):
                            assert item["mtp"][key] == golden[depth]["mtp"][key]
                        item.update(wall_seconds=elapsed, warmup=number==0, reused_input_tokens=result.get("reused_input_tokens",0))
                        try:
                            item["host_screen"] = screen_attempt(monitor,phase,before,paging())
                        except ValueError as error:
                            if str(error)!="host_observations_missing":raise
                            item["host_screen"]={"eligible":False,"observation_status":"missing"}
                        session["requests"].append(item)
            session["passed"]=True
        except BaseException as error:
            session["passed"]=False
            session["failure_type"]=type(error).__name__
        report["sessions"].append(session)
        out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
        print(json.dumps({"stage":"completed_session","index":index,"passed":session["passed"]}),flush=True)
        if not session["passed"]:break
    report["passed"] = len(report["sessions"])==4 and all(s["passed"] for s in report["sessions"])
    if report["passed"]:
        runs=[[r for r in s["requests"] if not r["warmup"]] for s in report["sessions"]]
        pairs=list(zip(runs[0],runs[1]))+list(zip(runs[3],runs[2]))
        report["paired_wall_change_candidate_vs_before"]=statistics.median(c["wall_seconds"]/b["wall_seconds"]-1 for b,c in pairs)
        report["candidate_wall_wins"]=sum(c["wall_seconds"]<b["wall_seconds"] for b,c in pairs)
        report["paired_useful_decode_change_candidate_vs_before"]=statistics.median(
            ((c["output_tokens"]-1)/c["decode_seconds"])/((b["output_tokens"]-1)/b["decode_seconds"])-1 for b,c in pairs)
        report["summary"]={}
        for label in ("baseline","candidate"):
            rows=[r for s in report["sessions"] if s["label"]==label for r in s["requests"] if not r["warmup"]]
            report["summary"][label]={"requests":len(rows),"median_wall_seconds":statistics.median(r["wall_seconds"] for r in rows),
                "median_useful_decode_tok_s":statistics.median((r["output_tokens"]-1)/r["decode_seconds"] for r in rows),
                "min_wall_seconds":min(r["wall_seconds"] for r in rows),"max_wall_seconds":max(r["wall_seconds"] for r in rows)}
    report["finished_utc"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
    out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:report.get(k) for k in ("passed","paired_wall_change_candidate_vs_before","candidate_wall_wins","paired_useful_decode_change_candidate_vs_before")}),flush=True)
    return 0 if report["passed"] else 1


if __name__=="__main__":raise SystemExit(main())
