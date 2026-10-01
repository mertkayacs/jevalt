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
"""Fine-tune a decision checkpoint with a soft-label decision loss.

For every field the model's logits at the position before its <decision> marker,
restricted to that field's answer symbols, are pushed towards the gold
distribution with soft cross-entropy (a proper scoring rule). Rows that carry a
reasoning trace also get next-token loss on the trace. Only the needed logits
are materialized: symbol logits for decisions, full vocabulary only on the
trace tokens.

Runs locally or as a Hugging Face Job. Data: JSONL files in the canonical record
format (see DESIGN.md), fetched from a Hub dataset repo.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import tarfile
import time

import torch
import torch.nn.functional as F
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

p = argparse.ArgumentParser()
p.add_argument("--base", default="internlm/Intern-Decision-4B")
p.add_argument("--work", default="mertkayacs/jev-work", help="private dataset repo with code bundle and data")
p.add_argument("--train", nargs="+", required=True, help="paths inside --work, JSONL")
p.add_argument("--valid", nargs="*", default=[])
p.add_argument("--out", required=True, help="Hub model repo for the adapter (private)")
p.add_argument("--epochs", type=float, default=1.0)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--rank", type=int, default=32)
p.add_argument("--alpha", type=int, default=32)
p.add_argument("--tokens-per-batch", type=int, default=12000)
p.add_argument("--grad-accum", type=int, default=2)
p.add_argument("--max-len", type=int, default=4096)
p.add_argument("--reasoning-weight", type=float, default=0.3)
p.add_argument("--eval-every", type=int, default=200)
p.add_argument("--time-budget-min", type=float, default=0, help="stop and save before this many minutes")
p.add_argument("--limit", type=int, default=0)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--init-adapter", default="", help="continue from an adapter repo")
p.add_argument("--grad-ckpt", action="store_true", help="trade speed for memory (needed on 24 GB cards)")
args = p.parse_args()

START = time.time()
random.seed(args.seed)
torch.manual_seed(args.seed)
api = HfApi()


def log(*a):
    print(time.strftime("%H:%M:%S"), f"[{(time.time() - START) / 60:5.1f}m]", *a, flush=True)


with tarfile.open(hf_hub_download(args.work, "code/jev-code.tar.gz", repo_type="dataset")) as t:
    t.extractall("/tmp/code", filter="data")
sys.path.insert(0, "/tmp/code/jevalt/src")
from jevalt.format import DECISION_TOKEN, UNKNOWN, compile_request, render  # noqa: E402


def read(paths):
    rows = []
    for path in paths:
        local = hf_hub_download(args.work, path, repo_type="dataset")
        with open(local, encoding="utf-8") as fh:
            rows.extend(json.loads(line) for line in fh if line.strip())
    return rows


from transformers import AutoConfig, AutoTokenizer  # noqa: E402

base_dir = snapshot_download(args.base, allow_patterns=["*.json", "*.jinja", "*.safetensors", "merges.txt", "vocab.json", "*.txt"])
tok = AutoTokenizer.from_pretrained(base_dir)
MARKER = tok.convert_tokens_to_ids(DECISION_TOKEN)
THINK_END = tok.convert_tokens_to_ids("</think>")
SYM = {}


def sym_id(s):
    if s not in SYM:
        ids = tok.encode(s, add_special_tokens=False)
        assert len(ids) == 1, s
        SYM[s] = ids[0]
    return SYM[s]


def encode(row):
    """Token ids, per-field (position, symbol ids, target probs), and trace token span."""
    # A target that puts mass on "unknown" needs the unknown option in the prompt, even when the row lacks the flag;
    # without it the unknown mass would be dropped and the rest renormalised.
    abstain = bool(row.get("abstain")) or any(
        UNKNOWN in (t.get("dist") or {}) or t.get("label") == UNKNOWN for t in row["targets"].values()
    )
    compiled = compile_request(row, reasoning=row.get("reasoning") or None, abstain=abstain, language=row.get("lang", "en"))
    text = render(compiled)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) > args.max_len:
        return None
    marks = [i - 1 for i, t in enumerate(ids) if t == MARKER]
    fields = []
    for pos, field in zip(marks, compiled.fields):
        target = row["targets"][field.name]
        dist = target.get("dist") or {target["label"]: 1.0}
        probs = [float(dist.get(v, 0.0)) for v in field.values]
        total = sum(probs)
        if total <= 0:
            return None
        fields.append((pos, [sym_id(s) for s in field.symbols], [x / total for x in probs]))
    trace = None
    if row.get("reasoning"):
        head = len(tok(render(compiled, close=False), add_special_tokens=False)["input_ids"])
        end = ids.index(THINK_END, head)
        trace = (head, end + 1)  # predict trace tokens and the closing </think>
    return {"ids": ids, "fields": fields, "trace": trace}


def encode_all(rows, name):
    out, skipped = [], 0
    for row in rows:
        e = encode(row)
        if e is None:
            skipped += 1
        else:
            out.append(e)
    log(f"{name}: {len(out)} rows encoded, {skipped} skipped, {sum(len(e['ids']) for e in out) / 1e6:.2f}M tokens")
    return out


train_rows = read(args.train)
random.shuffle(train_rows)
if args.limit:
    train_rows = train_rows[: args.limit]
train = encode_all(train_rows, "train")
valid = encode_all(read(args.valid), "valid") if args.valid else []


def batches(data, shuffle=True):
    """Length-bucketed batches under a token budget (right padding is safe: the model is causal)."""
    order = sorted(range(len(data)), key=lambda i: len(data[i]["ids"]))
    chunks, cur, longest = [], [], 0
    for i in order:
        n = len(data[i]["ids"])
        if cur and max(longest, n) * (len(cur) + 1) > args.tokens_per_batch:
            chunks.append(cur)
            cur, longest = [], 0
        cur.append(i)
        longest = max(longest, n)
    if cur:
        chunks.append(cur)
    if shuffle:
        random.shuffle(chunks)
    return chunks


config = AutoConfig.from_pretrained(base_dir)
import transformers  # noqa: E402

model_cls = getattr(transformers, config.architectures[0])
model = model_cls.from_pretrained(base_dir, dtype=torch.bfloat16).cuda()
model.config.use_cache = False
text_model = getattr(model.model, "language_model", model.model)
W = model.get_output_embeddings().weight  # tied to the input embeddings

from peft import LoraConfig, PeftModel, get_peft_model  # noqa: E402

targets = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|in_proj_qkvz|in_proj_ba|out_proj)$"
for param in model.parameters():
    param.requires_grad_(False)
if args.init_adapter:
    model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
else:
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0, target_modules=targets, task_type="CAUSAL_LM"))
model.print_trainable_parameters()
if args.grad_ckpt:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()


def step_loss(batch_ids, data):
    rows = [data[i] for i in batch_ids]
    width = max(len(r["ids"]) for r in rows)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids = torch.full((len(rows), width), pad, dtype=torch.long)
    for b, r in enumerate(rows):
        ids[b, : len(r["ids"])] = torch.tensor(r["ids"])
    ids = ids.cuda()
    # Right padding needs no attention mask: every layer is causal, so real tokens never
    # see the pads after them, and the linear-attention kernels stay on their fast path.
    hidden = text_model(input_ids=ids).last_hidden_state
    dec_loss, n_dec, lm_loss, n_lm, brier, hits = hidden.new_zeros(()), 0, hidden.new_zeros(()), 0, 0.0, 0
    for b, r in enumerate(rows):
        for pos, sym, target in r["fields"]:
            logits = (hidden[b, pos].float() @ W[sym].float().T)
            logp = F.log_softmax(logits, dim=-1)
            t = torch.tensor(target, device=logits.device)
            dec_loss = dec_loss - (t * logp).sum()
            brier += float(((logp.detach().exp() - t) ** 2).sum())
            hits += int(int(logp.argmax()) == int(t.argmax()))
            n_dec += 1
        if r["trace"]:
            a, z = r["trace"]
            h = hidden[b, a - 1 : z - 1]
            gold = ids[b, a:z]
            lm_loss = lm_loss + F.cross_entropy((h.to(W.dtype) @ W.T).float(), gold, reduction="sum")
            n_lm += z - a
    loss = dec_loss / max(n_dec, 1)
    if n_lm:
        loss = loss + args.reasoning_weight * lm_loss / n_lm
    return loss, n_dec, brier, hits


@torch.no_grad()
def evaluate():
    model.eval()
    total, n, brier, hits = 0.0, 0, 0.0, 0
    for chunk in batches(valid, shuffle=False):
        loss, k, b, h = step_loss(chunk, valid)
        total += float(loss) * k
        n += k
        brier += b
        hits += h
    model.train()
    return total / max(n, 1), brier / max(n, 1), hits / max(n, 1)


@torch.no_grad()
def parity(rows, k=3):
    """The training forward (text model + tied head) must match the full model's logits."""
    model.eval()
    worst = 0.0
    for r in rows[:k]:
        ids = torch.tensor([r["ids"]]).cuda()
        hidden = text_model(input_ids=ids).last_hidden_state[0]
        positions = torch.tensor([pos for pos, _, _ in r["fields"]]).cuda()
        full = model(input_ids=ids, use_cache=False, logits_to_keep=positions).logits[0].float()
        for i, (pos, sym, _) in enumerate(r["fields"]):
            a = F.softmax(hidden[pos].float() @ W[sym].float().T, -1)
            b = F.softmax(full[i, sym], -1)
            worst = max(worst, float((a - b).abs().max()))
    model.train()
    return worst


log("parity train-path vs model-path, max abs prob diff:", round(parity(valid or train), 5))
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.99))
plan = batches(train)
total_steps = max(1, int(len(plan) * args.epochs / args.grad_accum))
warmup = max(1, int(0.03 * total_steps))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))))
log(f"{len(plan)} batches/epoch, {total_steps} optimizer steps, warmup {warmup}")

best, step, seen_tokens, t0 = float("inf"), 0, 0, time.time()
out_dir = "/tmp/adapter"
if valid:
    v_loss, v_brier, v_acc = evaluate()
    log(f"valid before training: loss={v_loss:.4f} brier={v_brier:.4f} acc={v_acc:.4f}")
    best = v_brier
    model.save_pretrained(out_dir)
model.train()
micro, epoch_float, stop = 0, 0.0, False
while not stop:
    for chunk in plan:
        loss, k, _, _ = step_loss(chunk, train)
        (loss / args.grad_accum).backward()
        seen_tokens += sum(len(train[i]["ids"]) for i in chunk)
        micro += 1
        if micro % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 20 == 0:
                rate = seen_tokens / (time.time() - t0)
                log(f"step {step}/{total_steps} loss={float(loss):.4f} lr={sched.get_last_lr()[0]:.2e} tok/s={rate:.0f}")
            if valid and step % args.eval_every == 0:
                v_loss, v_brier, v_acc = evaluate()
                log(f"valid step {step}: loss={v_loss:.4f} brier={v_brier:.4f} acc={v_acc:.4f}")
                if v_brier < best:
                    best = v_brier
                    model.save_pretrained(out_dir)
                    log("new best adapter saved")
            if step >= total_steps or (args.time_budget_min and (time.time() - START) / 60 > args.time_budget_min):
                stop = True
                break
    epoch_float += 1

if valid:
    v_loss, v_brier, v_acc = evaluate()
    log(f"valid final: loss={v_loss:.4f} brier={v_brier:.4f} acc={v_acc:.4f}")
    if v_brier < best:
        model.save_pretrained(out_dir)
        best = v_brier
else:
    model.save_pretrained(out_dir)
tok.save_pretrained(out_dir)
cfg_path = f"{out_dir}/adapter_config.json"
adapter_cfg = json.load(open(cfg_path))
adapter_cfg["base_model_name_or_path"] = args.base  # PEFT writes the local cache path
json.dump(adapter_cfg, open(cfg_path, "w"), indent=2)
with open(f"{out_dir}/README.md", "w") as fh:
    fh.write(f"---\nbase_model: {args.base}\nlibrary_name: peft\nlicense: apache-2.0\n---\n\nWork-in-progress LoRA adapter for JevAlt. Not a release.\n")
with open(f"{out_dir}/training_args.json", "w") as fh:
    json.dump({**vars(args), "best_valid_brier": best, "steps": step, "minutes": round((time.time() - START) / 60, 1), "tokens_seen": seen_tokens}, fh, indent=2)
api.create_repo(args.out, private=True, exist_ok=True)
api.upload_folder(repo_id=args.out, folder_path=out_dir, commit_message=f"adapter, best valid brier {best:.4f}")
log("pushed adapter to", args.out, "best valid brier", round(best, 4))
