# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch>=2.8",
#   "transformers>=5.17",
#   "accelerate>=1.10",
#   "peft>=0.21",
#   "huggingface_hub>=1.0",
#   "llama-cpp-python>=0.3",
#   "numpy",
#   "psutil",
# ]
# ///
"""Merge an adapter, convert to GGUF, quantize, and prove the 4 GB claim.

Stages (CPU job is enough):
1. merge the LoRA adapter into the base, save bf16 weights (+ tokenizer, calibration)
2. convert with llama.cpp's convert_hf_to_gguf.py (pinned release), quantize
3. parity: GGUF decisions vs bf16 decisions on held-out requests (argmax agreement, max prob gap)
4. memory: peak RSS of a fresh process serving the Q4_K_M file at 4k context
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import time
import urllib.request

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

p = argparse.ArgumentParser()
p.add_argument("--base", default="internlm/Intern-Decision-4B")
p.add_argument("--adapter", default="", help="Hub repo with a LoRA adapter; empty = export the base as is")
p.add_argument("--merged-repo", required=True)
p.add_argument("--gguf-repo", required=True)
p.add_argument("--quants", nargs="+", default=["Q4_K_M", "Q5_K_M", "Q8_0"])
p.add_argument("--calibration", default="", help="path of calibration.json inside the work repo")
p.add_argument("--probe", default="data/smoke/valid.jsonl", help="requests for parity, inside the work repo")
p.add_argument("--probe-n", type=int, default=24)
p.add_argument("--work", default="mertkayacs/jev-work")
p.add_argument("--private", action="store_true")
p.add_argument("--imatrix-data", nargs="*", default=[], help="work-repo JSONL files whose rendered prompts calibrate the importance matrix")
p.add_argument("--parity-f16", action="store_true", help="also check the unquantized GGUF (isolates conversion from quantization)")
args = p.parse_args()
api = HfApi()
T0 = time.time()


def log(*a):
    print(time.strftime("%H:%M:%S"), f"[{(time.time() - T0) / 60:5.1f}m]", *a, flush=True)


def sh(cmd):
    log("$", cmd[:160])
    subprocess.run(cmd, shell=True, check=True)


def usable_cpus() -> int:
    """CPUs the container may use. os.cpu_count() reports the host (64 on HF cpu-upgrade, quota 8)."""
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            return max(1, int(quota) // int(period))
    except (OSError, ValueError):
        pass
    return len(os.sched_getaffinity(0))


CPUS = usable_cpus()


with tarfile.open(hf_hub_download(args.work, "code/jev-code.tar.gz", repo_type="dataset")) as t:
    t.extractall("/tmp/code", filter="data")
sys.path.insert(0, "/tmp/code/jevalt/src")

import torch  # noqa: E402
import transformers  # noqa: E402

torch.set_num_threads(CPUS)
from transformers import AutoConfig, AutoTokenizer  # noqa: E402

# 1. merge
base = snapshot_download(args.base, allow_patterns=["*.json", "*.jinja", "*.safetensors", "merges.txt", "vocab.json", "*.txt"])
cfg = AutoConfig.from_pretrained(base)
model = getattr(transformers, cfg.architectures[0]).from_pretrained(base, dtype=torch.bfloat16)
if args.adapter:
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    log("merged adapter", args.adapter)
merged = "/tmp/merged"
model.save_pretrained(merged, safe_serialization=True)
AutoTokenizer.from_pretrained(base).save_pretrained(merged)

for extra in ("chat_template.jinja", "preprocessor_config.json", "video_preprocessor_config.json", "LICENSE", "LICENSE-QWEN"):
    if os.path.exists(os.path.join(base, extra)):
        sh(f"cp {os.path.join(base, extra)} {merged}/")
if args.calibration:
    sh(f"cp {hf_hub_download(args.work, args.calibration, repo_type='dataset')} {merged}/calibration.json")
del model

# 2. convert + quantize with a pinned llama.cpp release
releases = json.load(urllib.request.urlopen("https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10"))
rel = next(r for r in releases if r["tag_name"].startswith("b") and any(a["name"].endswith("bin-ubuntu-x64.tar.gz") for a in r["assets"]))
tag = rel["tag_name"]
bin_url = next(a["browser_download_url"] for a in rel["assets"] if a["name"].endswith("bin-ubuntu-x64.tar.gz"))
log("llama.cpp", tag)
sh(f"git clone -q --depth 1 --branch {tag} https://github.com/ggml-org/llama.cpp /tmp/llama.cpp")
sh("pip install -q -r /tmp/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt 2>&1 | tail -2 || true")
sh(f"mkdir -p /tmp/llamabin && curl -sL {bin_url} | tar xz -C /tmp/llamabin")
quant_bin = subprocess.run("find /tmp/llamabin -name llama-quantize -type f | head -1", shell=True, capture_output=True, text=True).stdout.strip()
sh(f"chmod +x {quant_bin}")
name = args.gguf_repo.split("/")[-1].removesuffix("-GGUF")
os.makedirs("/tmp/gguf", exist_ok=True)
f16 = f"/tmp/gguf/{name}-F16.gguf"
sh(f"{sys.executable} /tmp/llama.cpp/convert_hf_to_gguf.py {merged} --outfile {f16} --outtype f16 --no-mtp")  # no MTP weights in the checkpoint; keeps older runtimes happy
imatrix_flag = ""
if args.imatrix_data:
    from jevalt.format import compile_request, render

    with open("/tmp/imatrix.txt", "w") as fh:
        for path in args.imatrix_data:
            for line in open(hf_hub_download(args.work, path, repo_type="dataset")):
                row = json.loads(line)
                fh.write(render(compile_request(row, reasoning=row.get("reasoning") or None)) + "\n")
    imat_bin = quant_bin.replace("llama-quantize", "llama-imatrix")
    sh(f"LD_LIBRARY_PATH={os.path.dirname(quant_bin)} {imat_bin} -m {f16} -f /tmp/imatrix.txt -o /tmp/imatrix.gguf --parse-special -c 2048 -t {CPUS} > /tmp/imatrix.log 2>&1 || (tail -20 /tmp/imatrix.log; false)")
    imatrix_flag = "--imatrix /tmp/imatrix.gguf"
files = {"F16": f16} if args.parity_f16 else {}
for q in args.quants:
    out = f"/tmp/gguf/{name}-{q}.gguf"
    sh(f"LD_LIBRARY_PATH={os.path.dirname(quant_bin)} {quant_bin} {imatrix_flag} {f16} {out} {q} {CPUS} > /tmp/quant-{q}.log 2>&1")
    files[q] = out
    log(q, round(os.path.getsize(out) / 1e9, 3), "GB")

# 3. parity on held-out requests
from jevalt.backends.hf import HFBackend  # noqa: E402
from jevalt.backends.llamacpp import LlamaCppBackend  # noqa: E402
from jevalt.engine import DecisionEngine  # noqa: E402

from jevalt.format import compile_request, render  # noqa: E402

hf_tok = AutoTokenizer.from_pretrained(merged)
with open(hf_hub_download(args.work, args.probe, repo_type="dataset")) as fh:
    rows = [json.loads(line) for line in fh]
# Parity and memory run at 4k context: keep requests whose prompt fits with room for the answer skeleton.
fits = [r for r in rows if len(hf_tok(render(compile_request(r)), add_special_tokens=False)["input_ids"]) <= 3500]
reqs = [{"state": r["state"], "questions": r["questions"]} for r in fits[: args.probe_n]]
log("parity requests", len(reqs), "of", len(rows), "fit in 3,500 tokens")
ref = DecisionEngine(HFBackend(merged, device="cpu", dtype="float32"), model=name)
ref_out = [ref.predict(r) for r in reqs]
del ref
report = {"llama_cpp": tag, "imatrix": bool(imatrix_flag), "sizes_gb": {q: round(os.path.getsize(f) / 1e9, 3) for q, f in files.items()}, "parity": {}}
token_checked = False
for q, path in files.items():
    backend = LlamaCppBackend(path, n_ctx=4096, n_threads=CPUS)
    if not token_checked:
        mismatched = 0
        for req in reqs:
            text = render(compile_request(req))
            if hf_tok(text, add_special_tokens=False)["input_ids"] != backend.tokenize(text):
                mismatched += 1
        report["token_mismatch_requests"] = mismatched
        log("token id mismatches (HF vs llama.cpp):", mismatched, "of", len(reqs))
        token_checked = True
    eng = DecisionEngine(backend, model=name)
    agree, gap, n = 0, 0.0, 0
    t = time.time()
    for req, a in zip(reqs, ref_out):
        b = eng.predict(req)
        for f in a["answers"]:
            pa = a["answers"][f].get("probabilities") or {"yes": a["answers"][f]["noul"], "no": 1 - a["answers"][f]["noul"]}
            pb = b["answers"][f].get("probabilities") or {"yes": b["answers"][f]["noul"], "no": 1 - b["answers"][f]["noul"]}
            agree += max(pa, key=pa.get) == max(pb, key=pb.get)
            gap = max(gap, max(abs(pa[k] - pb[k]) for k in pa))
            n += 1
    report["parity"][q] = {"argmax_agreement": agree / n, "max_prob_gap": round(gap, 4), "decisions": n, "sec_per_request": round((time.time() - t) / len(reqs), 2)}
    log(q, report["parity"][q])
    del eng

# 4. peak RSS of a fresh process at 4k context
probe_path = "/tmp/probe.json"
json.dump(reqs[:8], open(probe_path, "w"))
rss_script = f"""
import json, sys
sys.path.insert(0, "/tmp/code/jevalt/src")
from jevalt.backends.llamacpp import LlamaCppBackend
from jevalt.engine import DecisionEngine
eng = DecisionEngine(LlamaCppBackend(sys.argv[1], n_ctx=4096, n_threads=4, mmap=False), model="probe")
for r in json.load(open("{probe_path}")):
    eng.predict(r)
# VmHWM is the peak RSS of this process image; ru_maxrss would still carry the forked parent's peak.
status = dict(line.split(":", 1) for line in open("/proc/self/status") if ":" in line)
print(json.dumps({{"peak_rss_gb": int(status["VmHWM"].split()[0]) / 1e6, "rss_end_gb": int(status["VmRSS"].split()[0]) / 1e6}}))
"""
open("/tmp/rss.py", "w").write(rss_script)
report["memory"] = {}
for q, path in files.items():
    if q == "F16":
        continue
    out = subprocess.run([sys.executable, "/tmp/rss.py", path], capture_output=True, text=True)
    last = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else out.stderr[-500:]
    report["memory"][q] = json.loads(last) if last.startswith("{") else {"error": last}
    log("memory", q, report["memory"][q])

json.dump(report, open("/tmp/gguf/export-report.json", "w"), indent=2)
api.create_repo(args.merged_repo, private=args.private, exist_ok=True)
api.upload_folder(repo_id=args.merged_repo, folder_path=merged, commit_message="merged weights")
api.create_repo(args.gguf_repo, private=args.private, exist_ok=True)
for q, path in files.items():
    if q == "F16":
        continue
    api.upload_file(path_or_fileobj=path, path_in_repo=os.path.basename(path), repo_id=args.gguf_repo)
if imatrix_flag:
    api.upload_file(path_or_fileobj="/tmp/imatrix.gguf", path_in_repo="imatrix.gguf", repo_id=args.gguf_repo)
api.upload_file(path_or_fileobj="/tmp/gguf/export-report.json", path_in_repo="export-report.json", repo_id=args.gguf_repo)
log("done", json.dumps(report))
