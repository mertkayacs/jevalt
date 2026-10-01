"""`jevalt serve` and `jevalt ask`."""

from __future__ import annotations

import argparse
import json
import os
import sys

DEFAULT_REPO = "mertkayacs/Deem-4B-GGUF"
DEFAULT_FILE = "Deem-4B-Q4_K_M.gguf"


def load_engine(args):
    from huggingface_hub import hf_hub_download

    from .engine import DecisionEngine

    calibration = {}
    if args.backend == "gguf":
        from .backends.llamacpp import LlamaCppBackend

        path = args.file if os.path.exists(args.file) else hf_hub_download(args.model, args.file)
        backend = LlamaCppBackend(path, n_ctx=args.ctx, n_threads=args.threads, mmap=True if args.mmap else False, repack=not args.no_repack)
        repo = args.model
    else:
        from .backends.hf import HFBackend

        backend = HFBackend(args.model, device=args.device)
        repo = args.model
    try:
        with open(hf_hub_download(repo, "calibration.json")) as fh:
            calibration = json.load(fh)
    except Exception:  # noqa: BLE001, a checkpoint without calibration still works (T=1)
        print("no calibration.json found; probabilities are uncalibrated", file=sys.stderr)
    name = repo.split("/")[-1].removesuffix("-GGUF")
    return DecisionEngine(backend, model=name, calibration=calibration)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="jevalt", description="Local decision models with the Jev contract.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("serve", "ask"):
        s = sub.add_parser(name)
        s.add_argument("--backend", choices=["gguf", "hf"], default="gguf")
        s.add_argument("--model", default=DEFAULT_REPO, help="Hub repo (GGUF repo for --backend gguf)")
        s.add_argument("--file", default=DEFAULT_FILE, help="GGUF file name in the repo, or a local path")
        s.add_argument("--ctx", type=int, default=8192)
        s.add_argument("--threads", type=int, default=None)
        s.add_argument("--device", default=None)
        s.add_argument("--mmap", action="store_true", help="memory-map the GGUF file (lower disk I/O, higher peak RAM on Q4_K_M)")
        s.add_argument("--no-repack", action="store_true", help="disable weight repacking (saves RAM)")
        if name == "serve":
            s.add_argument("--host", default="127.0.0.1")
            s.add_argument("--port", type=int, default=8000)
        else:
            s.add_argument("request", help="JSON request, a path to one, or - for stdin")
    args = parser.parse_args(argv)
    engine = load_engine(args)
    if args.cmd == "serve":
        import uvicorn

        from .server import create_app

        uvicorn.run(create_app(engine), host=args.host, port=args.port)
    else:
        raw = sys.stdin.read() if args.request == "-" else open(args.request).read() if os.path.exists(args.request) else args.request
        print(json.dumps(engine.predict(json.loads(raw)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
