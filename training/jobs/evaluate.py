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
"""Evaluate one checkpoint (optionally with a LoRA adapter) on suites and probes.

Writes results.json (the same shape the sites read) to the work repo.
Suites can be registry names or canonical JSONL paths inside the work repo.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tarfile
import time

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

p = argparse.ArgumentParser()
p.add_argument("--base", default="internlm/Intern-Decision-4B")
p.add_argument("--adapter", default="")
p.add_argument("--tag", required=True, help="name for this model in results (--label is taken by hf jobs)")
p.add_argument("--calibration", default="", help="work-repo path of calibration.json, or 'fit:<suite>' to fit on that suite first")
p.add_argument("--suites", nargs="+", required=True)
p.add_argument("--limit", type=int, default=0)
p.add_argument("--reasoning", nargs="+", default=["off"])
p.add_argument("--auto-suites", nargs="*", default=[], help="suites that also run with reasoning auto (it needs a fitted auto threshold)")
p.add_argument("--probe-suite", default="")
p.add_argument("--probe-limit", type=int, default=120)
p.add_argument("--fit-auto", action="store_true", help="with fit:<suite>, also run the suite with reasoning on and fit the auto threshold")
p.add_argument("--auto-limit", type=int, default=0, help="the reasoning-on run for the auto fit uses this many fit items (0 = all)")
p.add_argument("--work", default="mertkayacs/jev-work")
args = p.parse_args()
T0 = time.time()
api = HfApi()


def log(*a):
    print(time.strftime("%H:%M:%S"), f"[{(time.time() - T0) / 60:5.1f}m]", *a, flush=True)


with tarfile.open(hf_hub_download(args.work, "code/jev-code.tar.gz", repo_type="dataset")) as t:
    t.extractall("/tmp/code", filter="data")
sys.path[:0] = ["/tmp/code/jevalt/src", "/tmp/code/jevoss/src"]

import torch  # noqa: E402
import transformers  # noqa: E402
from transformers import AutoConfig, AutoTokenizer  # noqa: E402

from jevalt.backends.hf import HFBackend  # noqa: E402
from jevalt.engine import DecisionEngine  # noqa: E402
from jevoss import calibrate, probes, report, suites  # noqa: E402
from jevoss.compare import dump as dump_decisions  # noqa: E402
from jevoss.runner import run  # noqa: E402

base = snapshot_download(args.base, allow_patterns=["*.json", "*.jinja", "*.safetensors", "merges.txt", "vocab.json", "*.txt"])
cfg = AutoConfig.from_pretrained(base)
model = getattr(transformers, cfg.architectures[0]).from_pretrained(base, dtype=torch.bfloat16).cuda().eval()
if args.adapter:
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload().eval()
    log("adapter merged", args.adapter)
backend = HFBackend(base, model=model, tokenizer=AutoTokenizer.from_pretrained(base))


def load(name):
    if name.endswith(".jsonl"):
        return suites.load(hf_hub_download(args.work, name, repo_type="dataset"))
    kw = {"limit": args.limit} if args.limit and name.startswith(("massive", "gmmlu")) else {}
    return suites.load(name, **kw)


calibration = {}
if args.calibration.startswith("fit:"):
    raw_engine = DecisionEngine(backend, model=args.tag)
    fit_items = load(args.calibration[4:])
    held, _ = run(fit_items, raw_engine.predict)
    calibration = calibrate.fit(held)
    log("fitted calibration", calibration["temperatures"])
    # Conformal thresholds apply to calibrated probabilities, so they are fitted after the temperatures.
    calibrated = calibrate.apply(held, calibration)
    calibration["conformal"] = calibrate.fit_conformal(calibrated)
    if args.fit_auto:
        # Reasoning is slow on one GPU; a seeded sample of the fit items is enough to place a per-type threshold.
        auto_items = fit_items
        if args.auto_limit:  # the same number per language, so every language gets a threshold and rated traces
            by_lang = {}
            for item in fit_items:
                by_lang.setdefault(item.get("lang", ""), []).append(item)
            k = max(1, args.auto_limit // len(by_lang))
            auto_items = [x for items in by_lang.values() for x in random.Random(0).sample(items, min(k, len(items)))]
        on, on_raw = run(auto_items, DecisionEngine(backend, model=args.tag, calibration=calibration).predict, extra={"reasoning": "on"}, on_error="record")
        os.makedirs("/tmp/out", exist_ok=True)
        with open("/tmp/out/reasoning-on.jsonl", "w") as fh:  # the traces themselves, for the native-feel rating
            for item, raw in zip(auto_items, on_raw):
                fh.write(json.dumps({"id": item["id"], "lang": item.get("lang"), "response": raw["response"], "error": raw["error"]}, ensure_ascii=False) + "\n")
        auto = calibrate.fit_auto_threshold(calibrated, on)
        calibration["auto_threshold"] = {kind: v["threshold"] for kind, v in auto.items()}
        calibration["auto_fit"] = auto
        log("auto thresholds", calibration["auto_threshold"])
    calibration["fit"] = {"suite": args.calibration[4:], "decisions": len(held), "date": time.strftime("%Y-%m-%d")}
    os.makedirs("/tmp/out", exist_ok=True)
    json.dump(calibration, open("/tmp/out/calibration.json", "w"), indent=1)
elif args.calibration:
    calibration = json.load(open(hf_hub_download(args.work, args.calibration, repo_type="dataset")))
engine = DecisionEngine(backend, model=args.tag, calibration=calibration)

results = []
for name in args.suites + [x for x in args.auto_suites if x not in args.suites]:
    items = load(name)
    modes = list(args.reasoning) if name in args.suites else ["off"]  # auto-only suites get their own off pass to pair with
    modes += ["auto"] if name in args.auto_suites and "auto" not in modes else []
    for mode in modes:
        t = time.time()
        extra = {"reasoning": mode} if mode != "off" else None
        decisions, raw = run(items, engine.predict, extra=extra, on_error="record")
        res = report.result(decisions, suite=name.split("/")[-1].removesuffix(".jsonl"), model=args.tag, extra={"reasoning": mode, "errors": sum(1 for r in raw if r["error"]), "seconds": round(time.time() - t, 1)})
        results.append(res)
        os.makedirs("/tmp/out/decisions", exist_ok=True)
        dump_decisions(f"/tmp/out/decisions/{res['suite']}-{mode}.jsonl", decisions)
        o = res["overall"]
        log(f"{name:40s} {mode:4s} n={o['n']:5d} acc={o['accuracy']:.3f} brier={o['brier']:.3f} ece={o['ece']:.3f} nll={o['nll']:.3f}")

probe_results = []
if args.probe_suite:
    items = load(args.probe_suite)[: args.probe_limit]
    for fn in (probes.probe_permutations, probes.probe_injections, probes.probe_distractors, probes.probe_noul_twins, probes.probe_determinism):
        try:
            r = fn(items[:20], engine.predict) if fn is probes.probe_determinism else fn(items, engine.predict)
            probe_results.append({**r, "model": args.tag, "suite": args.probe_suite})
            log("probe", json.dumps(r))
        except Exception as err:  # noqa: BLE001
            log("probe failed", fn.__name__, repr(err))

stamp = time.strftime("%Y%m%d-%H%M%S")
os.makedirs("/tmp/out", exist_ok=True)
report.dump("/tmp/out/results.json", {"label": args.tag, "base": args.base, "adapter": args.adapter, "calibration": calibration, "results": results, "probes": probe_results})
with open("/tmp/out/table.md", "w") as fh:
    fh.write(report.table(results))
api.upload_folder(repo_id=args.work, repo_type="dataset", folder_path="/tmp/out", path_in_repo=f"results/eval-{args.tag}-{stamp}")
log("uploaded", f"results/eval-{args.tag}-{stamp}")
print(open("/tmp/out/table.md").read(), flush=True)
