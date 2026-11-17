"""Balanced 0/1/2 crossover with rootless clocks; hashes/counts/metrics only."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"scripts"))
from validate_generation_prefix_cache import Worker
from benchmark_qwen38_mtp import _validate_result, exact_prefix, ids_sha256, sha256_file
from benchmark_clean_qwen import Monitor, paging
from benchmark_glm_mlx_serve import screen_attempt
from qwen38_clock_metrics import ClockSampler, summarize as clock_summary

OUT = ROOT/"runtime/lily-mtp-reconcile-20261115/results-v2"
MODEL = Path("/Users/johnshahbazian/Library/Application Support/Stark/models/qwen38-flash-next-lily-q4-xl-0f94842e")
BASE = Path("/Users/johnshahbazian/development/stark-llama-v1-study/runtime/llama-v1-baseline-target/release/stark-generate")
CAND = ROOT/"runtime/build-candidate/release/stark-generate"
CLOCK = Path("/Users/johnshahbazian/development/stark/runtime/flashnext-moe-2026-11-07/tools/macmon")
PRIOR = Path("/Users/johnshahbazian/development/stark-llama-v1-study/runtime/llama-v1-lily-baseline-20261115-v2/identity.json")


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True)+"\n")


def describe(values):
    if not values:
        return None
    median = statistics.median(values)
    return {"n":len(values),"min":min(values),"median":median,"max":max(values),
            "mad":statistics.median(abs(v-median) for v in values),
            "relative_range":(max(values)-min(values))/median}


def order_plan(repeats, seed):
    choices = list(itertools.permutations((0,1,2)))*repeats
    rng = random.Random(seed)
    for _ in range(10000):
        rng.shuffle(choices)
        if all(a[-1]!=b[0] for a,b in zip(choices,choices[1:])):
            return [list(p) for p in choices]
    raise ValueError("balanced_order_unavailable")


def interval(values, seed):
    rng = random.Random(seed)
    n = len(values)
    samples=[]
    for _ in range(10000):
        draw=[]
        while len(draw)<n:
            start=rng.randrange(n)
            draw.extend((values[start],values[(start+1)%n]))
        samples.append(statistics.median(draw[:n]))
    samples.sort()
    return [samples[249], samples[9749]]


def analyze(cell,capture,index):
    items=cell["warmups"]+[cell["selector_probe"]]+[r for b in cell["blocks"] for r in b["requests"]]
    for item in items:
        item["clock_full_request"]=clock_summary(capture["samples"],item["started_unix_seconds"],item["finished_unix_seconds"])
        item["clock_approximate_decode"]=clock_summary(capture["samples"],item["started_unix_seconds"]+item["prefill_seconds"],item["finished_unix_seconds"])
    cell["arm_summary"]={}
    for depth in (0,1,2):
        rows=[r for b in cell["blocks"] for r in b["requests"] if r["depth"]==depth]
        same=[r for r in items if r["depth"]==depth]
        assert len({r["output_sha256"] for r in same})==1
        assert len({tuple(r["mtp"][k] for k in ("rounds","proposed","accepted","verified_rows")) for r in same})==1
        cell["arm_summary"][str(depth)]={"useful_decode_tok_s":describe([(r["output_tokens"]-1)/r["decode_seconds"] for r in rows]),
            "wall_seconds":describe([r["wall_seconds"] for r in rows]),"prefill_seconds":describe([r["prefill_seconds"] for r in rows]),
            "output_sha256":rows[0]["output_sha256"],"mtp_counts":{k:rows[0]["mtp"][k] for k in ("rounds","proposed","accepted","verified_rows")},
            "clock_mean_active_gpu_mhz":describe([r["clock_approximate_decode"]["mean_active_gpu_mhz"] for r in rows if r["clock_approximate_decode"]["mean_active_gpu_mhz"] is not None])}
    cell["contrasts"]={}
    for a,b in ((1,0),(2,0),(1,2)):
        blocks=[{r["depth"]:r for r in block["requests"]} for block in cell["blocks"]]
        wall=[p[a]["wall_seconds"]/p[b]["wall_seconds"]-1 for p in blocks]
        rate=[((p[a]["output_tokens"]-1)/p[a]["decode_seconds"])/((p[b]["output_tokens"]-1)/p[b]["decode_seconds"])-1 for p in blocks]
        cell["contrasts"][f"{a}_vs_{b}"]={"paired_wall_change":statistics.median(wall),"wall_wins":sum(v<0 for v in wall),
            "paired_decode_change":statistics.median(rate),"decode_wins":sum(v>0 for v in rate),
            "wall_change_moving_block_interval":interval(wall,12900+index+a*10+b),"block_count":len(blocks),
            "wall_changes":wall,"decode_changes":rate}


def main():
    os.umask(0o077)
    OUT.mkdir(parents=True,exist_ok=False)
    for name in tuple(os.environ):
        if name.startswith(("STARK_","UZU_","LILY_","GGML_")):
            os.environ.pop(name)
    identity=json.loads(PRIOR.read_text())
    def unchanged():
        return all([s.st_size,s.st_mtime_ns,s.st_ino]==expected
            for name,expected in identity["model_file_identities"].items() for s in [(MODEL/name).stat()])
    assert unchanged()
    assert sha256_file(BASE)=="61b9c8cc724407573169eb2abad7f908e4943eead7ac3122eeec3ef841a48640"
    assert sha256_file(CAND)=="5618180b1821f1883778cc8099b4fa408699ff6a5cb0d71b33907f9946d441a4"
    assert sha256_file(CLOCK)=="57259ce6845d0904be247c8e860fcc923300bab52e28a01f476c22c852051286"
    corpus=ROOT/"runtime/lily-one-draft/results/model-corpus.rs"
    held=ROOT/"runtime/lily-one-draft/results/held-corpus.rs"
    matrix=[("before",512,"top20",256,42,corpus,1),
            ("candidate",512,"top20",256,42,corpus,2),
            ("candidate",512,"greedy",256,42,corpus,2),
            ("candidate",4096,"top20",256,42,corpus,2),
            ("candidate",4096,"nucleus",256,42,corpus,2),
            ("candidate",512,"top20",128,7,held,2)]
    plans=[order_plan(repeats,20261115+i) for i,(*_,repeats) in enumerate(matrix)]
    report={"attempt":"20261115-LILY-MTP1-03","source_commit":"86cb55ca5cad30d66ea613709587cfffb7fdd188",
        "worker_sha256":{"before":sha256_file(BASE),"candidate":sha256_file(CAND)},
        "model_content_manifest_sha256":identity["model_content_manifest_sha256"],
        "harness_sha256":sha256_file(Path(__file__)),"clock_helper_sha256":sha256_file(ROOT/"scripts/qwen38_clock_metrics.py"),
        "paired_validation_helper_sha256":sha256_file(ROOT/"scripts/benchmark_qwen38_mtp.py"),
        "corpus_sha256":sha256_file(corpus),"held_corpus_sha256":sha256_file(held),
        "started_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
        "clock_policy":"unavailable: sealed macmon binary omits source-bracket schema; no synthetic clock values",
        "continuation":{"prior_harness_sha256":'94b1f671c0a5c75ad2972590a53476edc83b5affd12d497df5455ba5508583bd',"initial_results_summary_sha256":sha256_file(ROOT/"runtime/lily-mtp-reconcile-20261115/results/summary.json"),"reason":"clock-schema mismatch caused empty clock summary StatisticsError after first cell; recover timings and continue without rerunning"},
        "host_policy":"all observed activity retained; no quiet admission, cooling or retries",
        "bootstrap":{"method":"circular moving blocks","block_length":2,"draws":10000,"interval":[.025,.975]},
        "plans":plans,"cells":[]}
    save(OUT/"summary.json",report)
    for index,(build,length,mode,budget,seed,source,repeats) in enumerate(matrix):
        name=f"{build}-input{length}-{mode}-out{budget}-seed{seed}"
        cell={"name":name,"build":build,"input_tokens":length,"context_tokens":length+budget,
            "output_budget":budget,"seed":seed,"order_plan":plans[index],"warmups":[],"blocks":[],
            "sampling":{"mode":"greedy"} if mode=="greedy" else
                {"mode":"stochastic","temperature":1,"top_p":.95,**({"top_k":20} if mode=="top20" else {})}}
        print(json.dumps({"stage":"starting_cell","cell":name,"blocks":len(plans[index])}),flush=True)
        if index==0:
            original_path=ROOT/"runtime/lily-mtp-reconcile-20261115/results"/(name+".json")
            cell=json.loads(original_path.read_text())
            cell["recovered_from_sha256"]=sha256_file(original_path)
            cell["initial_auxiliary_failure_type"]=cell.pop("failure_type")
            capture=json.loads((ROOT/"runtime/lily-mtp-reconcile-20261115/results"/(name+"-clock-private.json")).read_text())
            analyze(cell,capture,index)
            cell["passed"]=unchanged() and sha256_file(BASE)==report["worker_sha256"]["before"] and sha256_file(CAND)==report["worker_sha256"]["candidate"]
            assert cell["passed"]
            save(OUT/(name+".json"),cell)
            report["cells"].append({k:v for k,v in cell.items() if k not in ("warmups","blocks","ready","selector_probe")})
            save(OUT/"summary.json",report)
            print(json.dumps({"stage":"recovered_cell","cell":name,"passed":cell["passed"],"one_vs_two":cell["contrasts"]["1_vs_2"]}),flush=True)
            continue
        worker=monitor=clocks=None
        try:
            worker=Worker(str(BASE if build=="before" else CAND),str(MODEL),prefix_cache=True,vision=True,timeout=900)
            ready=worker.ready
            assert ready["mtp"]["recommended_drafts"]==(2 if build=="before" else 1)
            assert ready["image_input"]["family_profile"]=="qwen38-lily-retained-bf16-vision-v1"
            cell["ready"]={k:ready[k] for k in ("mtp","image_input","prefix_cache")}
            ids=exact_prefix(worker.tokenize(source.read_text()),length)
            cell["input_sha256"]=ids_sha256(ids)
            # Clock source schema failed preflight; retain timing study without clocks.
            monitor=Monitor().__enter__()
            serial=0
            def request(depth,selector=False):
                nonlocal serial
                serial+=1
                phase=f"cell{index}-request{serial}"
                monitor.set_phase(phase)
                before=paging()
                start_unix,start=time.time(),time.monotonic()
                result=worker.request({"operation":"generate","input_token_ids":ids,"context_tokens":length+budget,
                    "max_new_tokens":budget,"seed":seed,"sampling":cell["sampling"],
                    **({"mtp":True} if selector else {"mtp_drafts":depth})})
                elapsed,end_unix=time.monotonic()-start,time.time()
                assert result.get("input_tokens")==length and result.get("reused_input_tokens",0)==0
                item=_validate_result(result,ready,argparse.Namespace(tokens=budget),"off" if depth==0 else "on",depth,ready["vocab_size"])
                item.update(depth=depth,selector="enabled" if selector else "explicit",wall_seconds=elapsed,
                    started_unix_seconds=start_unix,finished_unix_seconds=end_unix,request_serial=serial,reused_input_tokens=0)
                try:
                    item["host_screen"]=screen_attempt(monitor,phase,before,paging())
                except ValueError as error:
                    if str(error)!="host_observations_missing":raise
                    item["host_screen"]={"eligible":False,"observation_status":"missing"}
                return item
            recommended=ready["mtp"]["recommended_drafts"]
            cell["selector_probe"]=request(recommended,True)
            for depth in (0,1,2):cell["warmups"].append(request(depth))
            for number,order in enumerate(plans[index]):
                block={"number":number,"order":order,"requests":[]}
                for depth in order:
                    block["requests"].append(request(depth))
                cell["blocks"].append(block)
                save(OUT/(name+".json"),cell)
            capture={"samples":[],"sample_count":0,"status":"unavailable_schema","binary_sha256":sha256_file(CLOCK)}
            save(OUT/(name+"-clock-private.json"),capture)
            cell["clock_capture_summary"]={k:v for k,v in capture.items() if k!="samples"}
            cell["clock_capture_sha256"]=sha256_file(OUT/(name+"-clock-private.json"))
            analyze(cell,capture,index)
            cell["passed"]=unchanged() and sha256_file(BASE)==report["worker_sha256"]["before"] and sha256_file(CAND)==report["worker_sha256"]["candidate"]
        except BaseException as error:
            cell["passed"]=False
            cell["failure_type"]=type(error).__name__
        finally:
            if clocks is not None:clocks.close()
            if monitor is not None:monitor.__exit__(None,None,None)
            if worker is not None:worker.close()
        save(OUT/(name+".json"),cell)
        report["cells"].append({k:v for k,v in cell.items() if k not in ("warmups","blocks","ready","selector_probe")})
        save(OUT/"summary.json",report)
        print(json.dumps({"stage":"completed_cell","cell":name,"passed":cell["passed"],"one_vs_two":cell.get("contrasts",{}).get("1_vs_2")}),flush=True)
        if not cell["passed"]:break
    report["completed"]=len(report["cells"])==len(matrix) and all(c["passed"] for c in report["cells"])
    report["finished_utc"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
    save(OUT/"summary.json",report)
    return 0 if report["completed"] else 1


if __name__=="__main__":raise SystemExit(main())
