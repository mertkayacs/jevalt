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
#   "fastapi>=0.115",
#   "uvicorn>=0.30",
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
"""Serve one released model on /v1/systemone, for live game replays and HTTP checks.

Run on a GPU flavor with `--expose 8000`; the jobs proxy wants an HF token in the
Authorization header. The job ends at its timeout or when cancelled.
"""

from __future__ import annotations

import argparse
import sys
import tarfile

from huggingface_hub import hf_hub_download

p = argparse.ArgumentParser()
p.add_argument("--model", required=True, help="merged model repo with calibration.json")
p.add_argument("--port", type=int, default=8000)
p.add_argument("--work", default="mertkayacs/jev-work")
args = p.parse_args()

with tarfile.open(hf_hub_download(args.work, "code/jev-code.tar.gz", repo_type="dataset")) as t:
    t.extractall("/tmp/code", filter="data")
sys.path.insert(0, "/tmp/code/jevalt/src")

from jevalt.cli import main  # noqa: E402

main(["serve", "--backend", "hf", "--model", args.model, "--device", "cuda", "--host", "0.0.0.0", "--port", str(args.port)])
