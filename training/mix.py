"""Compose a training mix from canonical JSONL files and upload it to the work repo.

Each source is `path:count` (count 0 = all rows). Choice rows with three or more
options get extra copies with shuffled option order, so the model learns that
position carries no information.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path


def read(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def needs_unknown(row: dict) -> bool:
    """Same rule as train.py: the unknown option is shown when the flag is set or a target uses it."""
    return bool(row.get("abstain")) or any(
        "unknown" in (t.get("dist") or {}) or t.get("label") == "unknown" for t in row.get("targets", {}).values()
    )


def permuted(row: dict, rng: random.Random) -> dict | None:
    out = copy.deepcopy(row)
    changed = False
    for q in out["questions"].values():
        if q["type"] == "choice" and len(q["criteria"]) >= 3:
            keys = list(q["criteria"])
            for _ in range(5):
                rng.shuffle(keys)
                if keys != list(q["criteria"]):
                    break
            q["criteria"] = {k: q["criteria"][k] for k in keys}
            changed = True
    if not changed:
        return None
    out["id"] = row["id"] + "-perm"
    out["tags"] = list(row.get("tags", [])) + ["aug:permutation"]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", nargs="+", required=True, help="path:count")
    p.add_argument("--valid", nargs="+", default=[], help="path:count")
    p.add_argument("--permute-frac", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="local output directory")
    p.add_argument("--upload", default="", help="work repo path prefix, e.g. data/mix/multi-v1")
    p.add_argument("--work", default="mertkayacs/jev-work")
    a = p.parse_args()
    rng = random.Random(a.seed)

    def take(specs):
        rows, report = [], {}
        for spec in specs:
            path, _, count = spec.rpartition(":")
            data = read(path)
            rng.shuffle(data)
            n = int(count) if count and int(count) > 0 else len(data)
            rows.extend(data[:n])
            report[path] = min(n, len(data))
        return rows, report

    train, train_report = take(a.train)
    extra = [r for r in (permuted(row, rng) for row in train if rng.random() < a.permute_frac) if r]
    train += extra
    rng.shuffle(train)
    valid, valid_report = take(a.valid)

    # Identical prompts: keep the first copy; drop every copy when their targets disagree (public sets such as
    # MASSIVE repeat utterances, sometimes with different labels). Valid wins over train so nothing leaks.
    def key(row):
        # F1 twins share state and questions but differ in whether the unknown option is shown.
        return hashlib.sha1(json.dumps([row["state"], row["questions"], needs_unknown(row)], sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    groups: dict[str, list[dict]] = {}
    for row in valid + [r for r in train if not r["id"].endswith("-perm")]:
        groups.setdefault(key(row), []).append(row)
    drop, conflicts = set(), 0
    for rows in groups.values():
        if len(rows) < 2:
            continue
        if len({json.dumps(r["targets"], sort_keys=True) for r in rows}) > 1:
            drop.update(r["id"] for r in rows)
            conflicts += 1
        else:
            drop.update(r["id"] for r in rows[1:])
    valid = [r for r in valid if r["id"] not in drop]
    train = [r for r in train if r["id"] not in drop and r["id"].removesuffix("-perm") not in drop]
    dropped = sorted(drop)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("valid", valid)):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    langs = {}
    for r in train:
        langs[r.get("lang", "en")] = langs.get(r.get("lang", "en"), 0) + 1
    manifest = {"train_sources": train_report, "valid_sources": valid_report, "permuted_copies": len(extra), "train_rows": len(train), "valid_rows": len(valid), "train_by_lang": langs, "seed": a.seed, "duplicates_dropped": len(dropped), "duplicate_label_conflicts": conflicts, "dropped_ids_sample": dropped[:20]}
    json.dump(manifest, open(out / "manifest.json", "w"), indent=2)
    print(json.dumps(manifest, indent=2))
    if a.upload:
        from huggingface_hub import HfApi

        HfApi().upload_folder(repo_id=a.work, repo_type="dataset", folder_path=str(out), path_in_repo=a.upload)
        print("uploaded to", a.upload)


if __name__ == "__main__":
    main()
