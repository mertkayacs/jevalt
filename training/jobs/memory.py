# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "huggingface_hub>=1.0",
#   "llama-cpp-python==0.3.35",
#   "numpy",
# ]
# ///
"""Measure peak RSS and latency for each GGUF file and load configuration.

For each file x config, run a fresh subprocess that loads the model with those
settings, runs probe requests through DecisionEngine, and prints JSON:
peak_rss_gb from VmHWM, rss_end_gb, sec_per_request, and the probabilities of
the first request. Also record per config whether the answers match the
mmap+repack run (argmax agreement).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
import time

from huggingface_hub import HfApi, hf_hub_download

p = argparse.ArgumentParser()
p.add_argument("--gguf-repo", required=True)
p.add_argument("--files", nargs="+", required=True)
p.add_argument("--configs", nargs="+", default=["mmap+repack", "nommap+repack", "mmap+norepack", "nommap+norepack"])
p.add_argument("--ctx", type=int, default=4096)
p.add_argument("--threads", type=int, default=4)
p.add_argument("--probe", default="data/smoke/valid.jsonl", help="work-repo path of a JSONL of requests")
p.add_argument("--probe-n", type=int, default=8)
p.add_argument("--work", default="mertkayacs/jev-work")
p.add_argument("--tag", required=True)
args = p.parse_args()
api = HfApi()
T0 = time.time()

CONFIG_MAP = {
    "mmap+repack": {"mmap": True, "repack": True},
    "nommap+repack": {"mmap": False, "repack": True},
    "mmap+norepack": {"mmap": True, "repack": False},
    "nommap+norepack": {"mmap": False, "repack": False},
}


def log(*a):
    print(time.strftime("%H:%M:%S"), f"[{(time.time() - T0) / 60:5.1f}m]", *a, flush=True)


with tarfile.open(hf_hub_download(args.work, "code/jev-code.tar.gz", repo_type="dataset")) as t:
    t.extractall("/tmp/code", filter="data")
sys.path.insert(0, "/tmp/code/jevalt/src")

probe_path_local = hf_hub_download(args.work, args.probe, repo_type="dataset")
with open(probe_path_local) as fh:
    probes = [json.loads(line) for line in fh][: args.probe_n]
reqs = [{"state": r["state"], "questions": r["questions"]} for r in probes]

probe_dump = "/tmp/probe.json"
json.dump(reqs, open(probe_dump, "w"))

results = {"tag": args.tag, "ctx": args.ctx, "threads": args.threads, "probe_n": len(reqs), "runs": {}}

baseline_argmax = None

for fname in args.files:
    results["runs"][fname] = {}
    gguf_path = hf_hub_download(args.gguf_repo, fname)

    for config_name in args.configs:
        cfg = CONFIG_MAP[config_name]
        mmap_arg = "True" if cfg["mmap"] else "False"
        repack_arg = "True" if cfg["repack"] else "False"

        worker = f"""
import json, sys, time
sys.path.insert(0, "/tmp/code/jevalt/src")
from jevalt.backends.llamacpp import LlamaCppBackend
from jevalt.engine import DecisionEngine

backend = LlamaCppBackend(
    sys.argv[1],
    n_ctx={args.ctx},
    n_threads={args.threads},
    mmap={mmap_arg},
    repack={repack_arg},
)
eng = DecisionEngine(backend, model="probe")
reqs = json.load(open("{probe_dump}"))
outputs = []
t0 = time.perf_counter()
for r in reqs:
    out = eng.predict(r)
    outputs.append(out)
elapsed = time.perf_counter() - t0

# argmax of every field of every request, keyed "request_index:field"
def top(v):
    probs = v.get("probabilities") or {{"yes": v["noul"], "no": 1 - v["noul"]}}
    return max(probs, key=probs.get)
argmax = {{f"{{i}}:{{k}}": top(v) for i, out in enumerate(outputs) for k, v in out["answers"].items()}}
first = outputs[0]["answers"]

# first request probabilities
first_probs = {{k: v.get("probabilities") or {{"yes": v["noul"], "no": 1 - v["noul"]}} for k, v in first.items()}}

status = dict(line.split(":", 1) for line in open("/proc/self/status") if ":" in line)
print(json.dumps({{
    "peak_rss_gb": int(status["VmHWM"].split()[0]) / 1e6,
    "rss_end_gb": int(status["VmRSS"].split()[0]) / 1e6,
    "sec_per_request": round(elapsed / len(reqs), 2),
    "first_probs": first_probs,
    "argmax": argmax,
}}))
"""
        open("/tmp/worker.py", "w").write(worker)

        log(fname, config_name)
        out = subprocess.run([sys.executable, "/tmp/worker.py", gguf_path], capture_output=True, text=True, timeout=600)
        last = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else out.stderr[-500:]
        if last.startswith("{"):
            run_result = json.loads(last)
        else:
            run_result = {"error": last}

        if config_name == "mmap+repack":
            baseline_argmax = run_result.get("argmax")
        elif baseline_argmax is not None and "argmax" in run_result:
            agree = sum(
                1 for k in baseline_argmax
                if k in run_result["argmax"] and baseline_argmax[k] == run_result["argmax"][k]
            ) / len(baseline_argmax) if baseline_argmax else 0
            run_result["argmax_agreement"] = round(agree, 3)

        results["runs"][fname][config_name] = run_result
        log(fname, config_name, {k: v for k, v in run_result.items() if k != "first_probs"})

out_path = "/tmp/memory-results.json"
json.dump(results, open(out_path, "w"), indent=2)
dest = f"results/memory-{args.tag}-{int(time.time())}/"
api.upload_file(path_or_fileobj=out_path, path_in_repo=dest + "results.json", repo_id=args.work, repo_type="dataset")
log("uploaded", dest)
