# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch==2.10.*",
#   "causal-conv1d @ https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
#   "flash-linear-attention>=0.5",
#   "transformers>=5.17",
#   "accelerate>=1.10",
#   "peft>=0.21",
#   "datasets>=3.0",
#   "huggingface_hub>=1.0",
#   "numpy",
#   "pillow",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# ///
"""Bake-off: score candidate start checkpoints on the same suites.

Runs on a Hugging Face Job (GPU). Code comes from a private work repo, results go back to it.
Intern-Decision-4B runs through the jevalt engine (and is checked against InternLM's own
inference code); Kev-4B runs through its own TypeSafe-compatible server in a separate env.
"""

import json
import os
import subprocess
import sys
import tarfile
import time
import urllib.request

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

WORK = os.environ.get("JEV_WORK_REPO", "mertkayacs/jev-work")
STAMP = time.strftime("%Y%m%d-%H%M%S")
api = HfApi()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


tar = hf_hub_download(WORK, "code/jev-code.tar.gz", repo_type="dataset")
with tarfile.open(tar) as t:
    t.extractall("/tmp/code", filter="data")
sys.path[:0] = ["/tmp/code/jevalt/src", "/tmp/code/jevoss/src"]

from jevalt.backends.hf import HFBackend  # noqa: E402
from jevalt.engine import DecisionEngine  # noqa: E402
from jevoss import report, suites  # noqa: E402
from jevoss.runner import http_predictor, run  # noqa: E402

SUITES = {
    "typed-decisions": {},
    "jevbench-easy": {},
    "jevbench-original": {},
    "jevbench-hard": {},
    "massive-en": {"limit": 300},
    "massive-tr": {"limit": 300},
    "massive-de": {"limit": 300},
    "gmmlu-en": {"limit": 300},
    "gmmlu-tr": {"limit": 300},
    "gmmlu-de": {"limit": 300},
}
ITEMS = {name: suites.load(name, **kw) for name, kw in SUITES.items()}
log("suites", {k: len(v) for k, v in ITEMS.items()})
results, audit = [], {}


def evaluate(label, predict, workers=1):
    for name, items in ITEMS.items():
        t0 = time.time()
        decisions, raw = run(items, predict, workers=workers, on_error="record")
        errors = sum(1 for r in raw if r["error"])
        res = report.result(decisions, suite=name, model=label, extra={"errors": errors, "seconds": round(time.time() - t0, 1)})
        results.append(res)
        audit[f"{label}/{name}"] = raw[:50]
        o = res["overall"]
        log(f"{label:28s} {name:18s} n={o['n']:5d} acc={o['accuracy']:.3f} brier={o['brier']:.3f} ece={o['ece']:.3f} err={errors} {res['seconds']}s")


# 1. Intern-Decision-4B through the jevalt engine, with InternLM's fitted temperature.
local = snapshot_download("internlm/Intern-Decision-4B", allow_patterns=["*.json", "*.jinja", "*.py", "*.txt", "*.safetensors", "merges.txt", "vocab.json"])
backend = HFBackend(local)
engine = DecisionEngine(backend, model="Intern-Decision-4B", calibration={"temperatures": {"default": 1.99241824}})
evaluate("Intern-Decision-4B", engine.predict)

# Parity: InternLM's own DecisionEngine on 30 typed-decisions rows must match ours.
sys.path.insert(0, local)
try:
    from inference import DecisionEngine as InternEngine

    theirs = InternEngine(local)
    worst = 0.0
    for item in ITEMS["typed-decisions"][:30]:
        req = {"state": item["state"], "questions": item["questions"]}
        a, b = engine.predict(req)["answers"], theirs.predict(req)["answers"]
        for f in a:
            pa = a[f].get("probabilities") or {"yes": a[f]["noul"]}
            pb = b[f].get("probabilities") or {"yes": b[f]["noul"]}
            worst = max(worst, max(abs(pa[k] - pb[k]) for k in pa if k in pb))
    log("parity max abs diff vs InternLM engine:", round(worst, 6))
    results.append({"suite": "parity", "model": "Intern-Decision-4B", "max_abs_diff": worst})
    del theirs
except Exception as err:  # noqa: BLE001
    log("parity check failed:", repr(err))
    results.append({"suite": "parity", "model": "Intern-Decision-4B", "error": repr(err)})
del engine, backend
import torch  # noqa: E402

torch.cuda.empty_cache()

# 2. Kev-4B through its own server (separate env: it pins torch<2.9).
try:
    subprocess.run("uv venv -q /tmp/kev && uv pip install -q --python /tmp/kev/bin/python 'kev[serve] @ git+https://github.com/jaredpalmer/kev' flash-linear-attention", shell=True, check=True)
    server = subprocess.Popen(["/tmp/kev/bin/python", "-m", "kev.serve", "--run", "jaredpalmer/kev-4b", "--port", "8008"], stdout=open("/tmp/kev.log", "w"), stderr=subprocess.STDOUT)
    for _ in range(180):
        try:
            urllib.request.urlopen("http://127.0.0.1:8008/v1/models", timeout=5)
            break
        except Exception:  # noqa: BLE001
            time.sleep(5)
    evaluate("Kev-4B", http_predictor("http://127.0.0.1:8008", model="kev-latest"), workers=4)
    server.terminate()
except Exception as err:  # noqa: BLE001
    log("kev failed:", repr(err))
    log(open("/tmp/kev.log").read()[-3000:] if os.path.exists("/tmp/kev.log") else "")

os.makedirs("/tmp/out", exist_ok=True)
report.dump("/tmp/out/results.json", results)
report.dump("/tmp/out/audit.json", audit)
with open("/tmp/out/table.md", "w") as fh:
    fh.write(report.table([r for r in results if "overall" in r]))
api.upload_folder(repo_id=WORK, repo_type="dataset", folder_path="/tmp/out", path_in_repo=f"results/bakeoff-{STAMP}")
log("uploaded", f"results/bakeoff-{STAMP}")
print(open("/tmp/out/table.md").read(), flush=True)
