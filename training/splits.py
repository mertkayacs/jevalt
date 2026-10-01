"""Freeze train / calibration / validation / test splits per language.

Every row belongs to a group (its `group:` tag, else its id). Rows derived from the
same source row share the group, so twins and padded copies never straddle splits.
A group's split comes from a keyed hash of the group name, so it is stable when rows
are added later. Test files get a SHA-256 in SPLITS.sha256; train never sees them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

SPLITS = (("test", 0.10), ("calib", 0.05), ("valid", 0.03))  # the rest is train


def group_of(row: dict) -> str:
    return next((t for t in row.get("tags", []) if t.startswith("group:")), f"id:{row['id']}")


def split_of(group: str, seed: str) -> str:
    u = int(hashlib.sha256(f"{seed}|{group}".encode()).hexdigest()[:12], 16) / 16**12
    edge = 0.0
    for name, share in SPLITS:
        edge += share
        if u < edge:
            return name
    return "train"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lang", required=True)
    p.add_argument("--inputs", nargs="+", required=True, help="JSONL files for this language")
    p.add_argument("--out", required=True, help="output directory, one sub-folder per language")
    p.add_argument("--seed", default="jevalt-splits-v1")
    a = p.parse_args()

    rows, seen = [], set()
    for path in a.inputs:
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("lang", a.lang) != a.lang or row["id"] in seen:
                continue
            seen.add(row["id"])
            rows.append(row)

    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[split_of(group_of(row), a.seed)].append(row)

    out = Path(a.out) / a.lang
    out.mkdir(parents=True, exist_ok=True)
    report = {"lang": a.lang, "seed": a.seed, "inputs": a.inputs, "splits": {}}
    for name in ("train", "calib", "valid", "test"):
        part = sorted(buckets[name], key=lambda r: r["id"])
        path = out / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in part)
        report["splits"][name] = {"rows": len(part), "by_source": dict(Counter(r.get("source", "?").split("+")[-1] for r in part))}
    digest = hashlib.sha256((out / "test.jsonl").read_bytes()).hexdigest()
    with open(Path(a.out) / "SPLITS.sha256", "a", encoding="utf-8") as fh:
        fh.write(f"{digest}  {a.lang}/test.jsonl\n")
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({k: v["rows"] for k, v in report["splits"].items()}), a.lang, "test sha256", digest[:16])


if __name__ == "__main__":
    main()
