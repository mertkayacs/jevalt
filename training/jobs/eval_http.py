# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "datasets>=3.0",
#   "huggingface_hub>=1.0",
# ]
# ///
"""Score any third-party decision server that speaks /v1/systemone.

Installs the server in its own environment, starts it, runs the suites through the
jevoss client, and uploads results.json to the work repo.
"""

import argparse
import json
import os
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request

from huggingface_hub import HfApi, hf_hub_download

p = argparse.ArgumentParser()
p.add_argument("--install", required=True, help="pip spec for the server package")
p.add_argument("--serve", required=True, help="command that starts the server")
p.add_argument("--port", type=int, default=8000)
p.add_argument("--model", default="jev-latest")
p.add_argument("--tag", required=True, help="name for this model in results (--label is taken by hf jobs)")
p.add_argument("--suites", nargs="*", default=[])
p.add_argument("--probe-suite", default="", help="run the robustness probes on this suite")
p.add_argument("--probe-limit", type=int, default=100)
p.add_argument("--limit", type=int, default=300)
p.add_argument("--workers", type=int, default=2)
p.add_argument("--server-env", nargs="*", default=[], help="KEY=VALUE for the server process (-e/--env is taken by hf jobs)")
p.add_argument("--work", default="mertkayacs/jev-work")
p.add_argument("--official-shapes", action="store_true", help="send the shapes the TypeSafe docs use (Noul criteria true/false, Score levels as a list)")
a = p.parse_args()


def official(item):
    """Same content in the documented shapes: Noul criteria keyed true/false, Score levels as an ordered list
    (gold labels move to the level index)."""
    item = json.loads(json.dumps(item))
    for qid, q in item["questions"].items():
        c = q.get("criteria")
        if q["type"] == "noul" and isinstance(c, dict):
            names = {"yes": "true", "true": "true", "1": "true", "no": "false", "false": "false", "0": "false"}
            q["criteria"] = {names.get(str(k).lower(), str(k)): v for k, v in c.items()}
        elif q["type"] == "score" and isinstance(c, dict):
            index = {str(k): str(i) for i, k in enumerate(c)}
            q["criteria"] = list(c.values())
            t = item["targets"][qid]
            t["label"] = index.get(str(t["label"]), str(t["label"]))
            if t.get("dist"):
                t["dist"] = {index.get(str(k), str(k)): v for k, v in t["dist"].items()}
    return item


def log(*x):
    print(time.strftime("%H:%M:%S"), *x, flush=True)


with tarfile.open(hf_hub_download(a.work, "code/jev-code.tar.gz", repo_type="dataset")) as t:
    t.extractall("/tmp/code", filter="data")
sys.path.insert(0, "/tmp/code/jevoss/src")
from jevoss import probes, report, suites  # noqa: E402
from jevoss.compare import dump as dump_decisions  # noqa: E402
from jevoss.runner import http_predictor, run  # noqa: E402

subprocess.run(f"uv venv -q /tmp/srv && uv pip install -q --python /tmp/srv/bin/python '{a.install}'", shell=True, check=True)
env = {**os.environ, "PATH": f"/tmp/srv/bin:{os.environ['PATH']}", **dict(kv.split("=", 1) for kv in a.server_env)}
server = subprocess.Popen(a.serve, shell=True, env=env, stdout=open("/tmp/server.log", "w"), stderr=subprocess.STDOUT)
for _ in range(240):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{a.port}/v1/models", timeout=5)
        break
    except urllib.error.HTTPError:
        break  # any HTTP answer means the server is up; not every server implements /v1/models
    except Exception:  # noqa: BLE001
        time.sleep(5)
else:
    print(open("/tmp/server.log").read()[-3000:])
    raise SystemExit("server did not start")
log("server up")

predict = http_predictor(f"http://127.0.0.1:{a.port}", model=a.model)
results = []
for name in a.suites:
    if name.endswith(".jsonl"):  # a frozen split in the work repo, named like evaluate.py names it
        items = suites.load(hf_hub_download(a.work, name, repo_type="dataset"))
        name = os.path.basename(name).removesuffix(".jsonl")
    else:
        kw = {"limit": a.limit} if name.startswith(("massive", "gmmlu")) else {}
        items = suites.load(name, **kw)
    if a.official_shapes:
        items = [official(i) for i in items]
    t = time.time()
    decisions, raw = run(items, predict, workers=a.workers, on_error="record")
    res = report.result(decisions, suite=name, model=a.tag, extra={"errors": sum(1 for r in raw if r["error"]), "seconds": round(time.time() - t, 1)})
    results.append(res)
    os.makedirs("/tmp/out/decisions", exist_ok=True)
    dump_decisions(f"/tmp/out/decisions/{name}.jsonl", decisions)
    o = res["overall"]
    log(f"{a.tag} {name:18s} n={o['n']} acc={o['accuracy']:.3f} brier={o['brier']:.3f} ece={o['ece']:.3f} errors={res['errors']}")
probe_results = []
if a.probe_suite:
    items = suites.load(a.probe_suite)[: a.probe_limit]
    for fn in (probes.probe_permutations, probes.probe_injections, probes.probe_distractors, probes.probe_noul_twins, probes.probe_determinism):
        try:
            r = fn(items[:20], predict) if fn is probes.probe_determinism else fn(items, predict, workers=a.workers) if fn is not probes.probe_noul_twins else fn(items, predict)
            probe_results.append({**r, "model": a.tag, "suite": a.probe_suite})
            log("probe", r)
        except Exception as err:  # noqa: BLE001
            log("probe failed", fn.__name__, repr(err))
server.terminate()
stamp = time.strftime("%Y%m%d-%H%M%S")
os.makedirs("/tmp/out", exist_ok=True)
report.dump("/tmp/out/results.json", {"label": a.tag, "results": results, "probes": probe_results})
HfApi().upload_folder(repo_id=a.work, repo_type="dataset", folder_path="/tmp/out", path_in_repo=f"results/eval-{a.tag}-{stamp}")
log("uploaded", f"results/eval-{a.tag}-{stamp}")
