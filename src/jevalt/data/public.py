"""P: public datasets, license gate first (DATA.md section 7).

``license_table`` reads each dataset's Hugging Face card (metadata and text) and
writes ``reports/licenses.md`` with the URL, revision and the card's own license
lines. Converters turn MASSIVE and Open-Jev into canonical rows; natural
instructions and label descriptions for MASSIVE are written once per language by
a teacher and cached.
"""

from __future__ import annotations

import gzip
import json
import re
from collections.abc import Iterator, Mapping
from datetime import date
from typing import Any

from .common import LANG_NAMES, data_dir, one_hot, rng, sha, smooth, target
from .providers import TEACHERS, Client
from .validate import FIELD_RE, LABEL_RE, check_row

DATASETS = [
    # (hub id, planned use, note)
    ("AmazonScience/massive", "train (tr-TR, de-DE, en-US)", "CC BY 4.0 needs attribution. TR/DE utterances were localized by translators from English originals. JevBench and jev-bench-tr ship a `massive` config: decontaminate before training."),
    ("ZefanCai/Open-Jev", "train (en, tr)", "Generated records CC0-1.0. customer-control-v1 rows carry upstream question text with an unverified license: excluded. Wikispeedia rows are not in the public configs."),
    ("google-research-datasets/paws-x", "train (de, en)", "Card metadata says 'other'; the card's licensing section allows any use (see quote)."),
    ("LocalLLaMA/typed-decisions", "train split only", ""),
    ("tasksource/tasksource-jev-typed-decisions", "check before use", "License tag 'other'."),
    ("PolyAI/banking77", "train", "JevBench ships a `banking77` config: decontaminate before training."),
    ("nyu-mll/multi_nli", "check before use", "Mixed licenses by genre (tags list several)."),
    ("Team-ACE/ToolACE", "train", ""),
    ("CohereLabs/Global-MMLU", "evaluation only", ""),
    ("facebook/xnli", "evaluation only", "The card states no license (metadata empty, licensing section 'More Information Needed')."),
    ("fancyzhx/ag_news", "evaluation only", "License tag 'unknown'."),
    ("Praveenrajus/jev-bench", "evaluation only", ""),
    ("hayriyigit/jev-bench-tr", "evaluation only", ""),
    ("allenai/wildjailbreak", "evaluation only", "Gated (auto)."),
]
LICENSE_LINE = re.compile(r"licen[cs]e|creative commons|cc[- ]by|cc0|apache|mit license|public domain|non-?commercial", re.IGNORECASE)
# Lines of license legal text itself say nothing about which license applies.
BOILERPLATE = re.compile(
    r"lawyer|as-is|warrant|Creative Commons Corporation|Licensor|Licensed (Material|Rights)|Public License[,.]|^\s*\||^\s*\d+\.\s|^\s*[a-z]\.\s",
    re.IGNORECASE,
)
STRONG = re.compile(r"licensed under|released under|license:|licen[cs]ed? (is|as)|under the terms|dedicat|Attribution \d\.\d International\s*$|Apache License|MIT License", re.IGNORECASE)


def _license_section(body: str) -> list[str]:
    """First lines under a markdown heading about licensing, if the card has one."""
    lines = body.splitlines()
    for i, line in enumerate(lines):
        if line.lstrip().startswith("#") and re.search(r"licen[cs]", line, re.IGNORECASE):
            after = [ln.strip() for ln in lines[i + 1 : i + 12] if ln.strip() and not ln.lstrip().startswith(("#", "```"))]
            return after[:3]
    return []


def license_table() -> str:
    """Write reports/licenses.md from the live dataset cards and return its path."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    lines = [
        "# Public dataset licenses",
        "",
        f"Checked {date.today().isoformat()} against each dataset's Hugging Face card (metadata `license` and the card text). "
        "Only rows with a confirmed license go into training; evaluation-only sets never enter train.",
        "",
        "| Dataset | Card metadata license | Planned use | Revision | Note |",
        "|---|---|---|---|---|",
    ]
    quotes = []
    for repo, use, note in DATASETS:
        url = f"https://huggingface.co/datasets/{repo}"
        try:
            info = api.dataset_info(repo)
            meta = (info.card_data or {}).get("license") if info.card_data else None
            meta = ", ".join(meta) if isinstance(meta, list) else (meta or "none")
            revision = info.sha[:10]
            try:
                card = open(hf_hub_download(repo, "README.md", repo_type="dataset", revision=info.sha), encoding="utf-8").read()
            except Exception:  # gated cards cannot be read without accepting terms
                card = ""
            body = card.split("---", 2)[-1] if card.startswith("---") else card
            hits = _license_section(body)
            if not hits:
                found = [ln.strip() for ln in body.splitlines() if LICENSE_LINE.search(ln) and len(ln.strip()) > 8 and not BOILERPLATE.search(ln)]
                hits = ([ln for ln in found if STRONG.search(ln)] + [ln for ln in found if not STRONG.search(ln)])[:3]
        except Exception as exc:  # keep going: one missing card should not hide the rest
            meta, revision, hits = f"error: {type(exc).__name__}", "-", []
        lines.append(f"| [{repo}]({url}) | {meta} | {use} | {revision} | {note} |")
        quotes.append(f"### {repo}\n\n{url}\n\n" + ("\n".join(f"> {h[:300]}" for h in hits) if hits else "> (no license sentence in the card body)"))
    lines += ["", "## License sentences quoted from the cards", ""] + quotes
    path = data_dir() / "reports" / "licenses.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return str(path)


# ---------------------------------------------------------------- MASSIVE

MASSIVE_LOCALE = {"en": "en-US", "tr": "tr-TR", "de": "de-DE"}
MASSIVE_PROMPT = """Write {language} text for a classifier that sorts short requests people say to a voice assistant.
1. "scenario_instructions": 3 different short questions asking which area a request belongs to.
2. "intent_instructions": 3 different short questions asking which specific intent a request expresses.
3. "scenarios": a short {language} description (at most 12 words) for every scenario label below.
4. "intents": a short {language} description (at most 12 words) for every intent label below.
Scenario labels: {scenarios}
Intent labels: {intents}
Reply with JSON only: {{"scenario_instructions": [...], "intent_instructions": [...], "scenarios": {{"<label>": "..."}}, "intents": {{"<label>": "..."}}}}"""


def _massive_file(lang: str, split: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download("AmazonScience/massive", f"{MASSIVE_LOCALE[lang]}/{split}/0000.parquet", repo_type="dataset",
                           revision="refs/convert/parquet", local_dir=str(data_dir() / "public" / "raw" / "massive"))


def _massive_labels(path: str) -> tuple[list[str], list[str]]:
    import pyarrow.parquet as pq

    features = json.loads(pq.ParquetFile(path).schema_arrow.metadata[b"huggingface"])["info"]["features"]
    return features["scenario"]["names"], features["intent"]["names"]


async def massive_texts(client: Client, lang: str, scenarios: list[str], intents: list[str]) -> dict:
    path = data_dir() / "templates" / lang / "massive.json"
    if path.exists():
        return json.loads(path.read_text())
    writer = {"en": "ds", "tr": "glm", "de": "mistral"}[lang]

    def check(data: Any) -> dict:
        if not isinstance(data, Mapping):
            raise ValueError("expected an object")
        for key, labels in (("scenarios", scenarios), ("intents", intents)):
            missing = set(labels) - set(data.get(key) or {})
            if missing:
                raise ValueError(f"{key} misses {sorted(missing)[:10]}")
        for key in ("scenario_instructions", "intent_instructions"):
            if len(data.get(key) or []) < 2:
                raise ValueError(f"{key} needs 3 questions")
        return data

    msg = MASSIVE_PROMPT.format(language=LANG_NAMES[lang], scenarios=", ".join(scenarios), intents=", ".join(intents))
    data, reply = await client.ask_json(writer, [{"role": "user", "content": msg}], check=check, temperature=0.3, max_tokens=12000, purpose="massive")
    data["writer"] = f"{TEACHERS[writer].provider}/{reply.model}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    return data


def massive_rows(lang: str, texts: Mapping, *, split: str = "train", limit: int | None = None, seed: str = "p") -> Iterator[dict]:
    """Stream MASSIVE utterances as rows with a scenario and an intent question."""
    import pyarrow.parquet as pq

    path = _massive_file(lang, split)
    scenarios, intents = _massive_labels(path)
    by_scenario: dict[str, list[str]] = {}
    for intent in intents:
        by_scenario.setdefault(intent.split("_", 1)[0], []).append(intent)
    n = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=512, columns=["id", "utt", "scenario", "intent"]):
        for rec in batch.to_pylist():
            r = rng("massive", lang, rec["id"], seed)
            scen, intent = scenarios[rec["scenario"]], intents[rec["intent"]]
            if r.random() < 0.25:
                options = list(intents)
            else:
                others = r.sample([s for s in by_scenario if s != intent.split("_", 1)[0]], r.randint(1, 3))
                options = by_scenario[intent.split("_", 1)[0]] + [i for s in others for i in by_scenario[s]]
                r.shuffle(options)
            scen_opts = list(scenarios)
            r.shuffle(scen_opts)
            row = {
                "lang": lang, "source": "pub:massive", "license": "cc-by-4.0", "split": "train",
                "tags": ["pub:massive", "domain:assistant_requests", "type:choice", f"group:massive-{rec['id']}"],
                "state": rec["utt"],
                "questions": {
                    "scenario": {"type": "choice", "instructions": r.choice(texts["scenario_instructions"]), "criteria": {s: texts["scenarios"][s] for s in scen_opts}},
                    "intent": {"type": "choice", "instructions": r.choice(texts["intent_instructions"]), "criteria": {i: texts["intents"][i] for i in options}},
                },
            }
            row["targets"] = {"scenario": target(one_hot(scen, scen_opts)), "intent": target(one_hot(intent, options))}
            row = {"id": f"{lang}-p-massive-{rec['id']}", **row}
            if not check_row(row, min_q=1):
                yield row
                n += 1
                if limit and n >= limit:
                    return


# ---------------------------------------------------------------- Open-Jev


def _openjev_file(config: str, split: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download("ZefanCai/Open-Jev", f"raw/{config}/{split}.jsonl.gz", repo_type="dataset", local_dir=str(data_dir() / "public" / "raw" / "openjev"))


def _clean_license(record: Mapping) -> bool:
    lic = str(((record.get("metadata") or {}).get("provenance") or {}).get("license") or "")
    return lic.startswith("CC0-1.0") and "not verified" not in lic


def _head(record: Mapping) -> tuple[str, dict, dict] | None:
    """One Open-Jev decision head -> (field, question, target distribution)."""
    md = record.get("metadata") or {}
    field = str(md.get("question_id") or record["id"].rsplit(":", 1)[-1])
    field = re.sub(r"\W+", "_", field).strip("_") or "decision"
    if not FIELD_RE.match(field):
        field = "q_" + field
    kind, options, tgt = record["kind"], record.get("options") or [], record.get("target") or []
    if kind == "choice":
        criteria = {}
        for opt in options:
            label, _, desc = str(opt).partition(": ")
            criteria[label.strip()] = desc.strip() or None
        if len(criteria) != len(options) or any(not LABEL_RE.match(k) for k in criteria):
            return None
        dist = dict(zip(criteria, tgt))
    elif kind == "score":
        criteria = [str(o) for o in options]
        dist = {str(i): p for i, p in enumerate(tgt)}
    elif kind == "noul":
        criteria = None
        dist = dict(zip([str(o).lower() for o in options], tgt)) if options else {"no": 1 - tgt[-1], "yes": tgt[-1]}
    else:
        return None
    total = sum(dist.values())
    if total <= 0 or len(dist) < 2:
        return None
    question = {"type": kind, "instructions": record["question"]}
    if criteria is not None:
        question["criteria"] = criteria
    return field, question, smooth({k: v / total for k, v in dist.items()})


def openjev_rows(lang: str, config: str = "mailroom-control-v1", *, split: str = "train", limit: int | None = None) -> Iterator[dict]:
    """Stream Open-Jev heads grouped per state into rows of at most six questions.

    Heads of one case are contiguous in the raw files; a group is flushed when the
    state changes. Open-Jev questions are English, so TR rows are tagged qlang:en.
    """
    want = {"en": {"en", None}, "tr": {"tr"}}.get(lang)
    if want is None:
        return
    n = 0
    group_key, heads, first = None, [], None

    def flush():
        nonlocal n
        rows = []
        for i in range(0, len(heads), 6):
            chunk = heads[i : i + 6]
            questions, targets = {}, {}
            for field, q, dist in chunk:
                while field in questions:
                    field += "_2"
                questions[field] = q
                targets[field] = target(dist)
            row = {
                "lang": lang, "source": "pub:open-jev", "license": "cc0-1.0", "split": "train",
                "tags": ["pub:open-jev", f"config:{config}", f"osrc:{first['source']}", f"group:openjev-{first['group_id']}"]
                + (["qlang:en"] if lang != "en" else []),
                "state": first["state"], "questions": questions, "targets": targets,
            }
            row = {"id": f"{lang}-p-openjev-{sha([first['group_id'], first['state'], list(questions)])[:10]}", **row}
            if not check_row(row, min_q=1):
                rows.append(row)
        return rows

    with gzip.open(_openjev_file(config, split), "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            md = rec.get("metadata") or {}
            if md.get("language") not in want or not _clean_license(rec):
                continue
            key = (rec["group_id"], sha(rec["state"]))
            if key != group_key and heads:
                for row in flush():
                    yield row
                    n += 1
                    if limit and n >= limit:
                        return
                heads = []
            if key != group_key:
                group_key, first = key, rec
            head = _head(rec)
            if head:
                heads.append(head)
        if heads:
            for row in flush():
                yield row
                n += 1
                if limit and n >= limit:
                    return


# ---------------------------------------------------------------- typed-decisions (EN, Apache-2.0)


def typed_decisions_rows(*, split: str = "train", limit: int | None = None) -> Iterator[dict]:
    """LocalLLaMA/typed-decisions: five typed questions per state, gold = the dataset's teacher
    distributions. Only the train split may enter training; the test split is an evaluation suite."""
    if split != "train":
        raise ValueError("typed-decisions: only the train split may enter training data")
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "LocalLLaMA/typed-decisions", "all/train-00000-of-00001.parquet", repo_type="dataset",
        local_dir=str(data_dir() / "public" / "raw" / "typed-decisions"),
    )
    n = 0
    for rec in pq.read_table(path).to_pylist():
        questions = json.loads(rec["questions"])
        gold = json.loads(rec["gold"])
        targets = {}
        for name, q in questions.items():
            probs = gold[name]["probabilities"]
            if q["type"] == "noul":
                dist = {("yes" if str(k).lower() in {"yes", "true", "1"} else "no"): float(v) for k, v in probs.items()}
            else:
                dist = {str(k): float(v) for k, v in probs.items()}
            total = sum(dist.values())
            if total <= 0:
                break
            targets[name] = target({k: v / total for k, v in dist.items()})
        else:
            state = json.loads(rec["state"]) if str(rec["state"]).startswith("{") else rec["state"]
            row = {
                "id": f"en-p-typed-{rec['id']}", "lang": "en", "source": "pub:typed-decisions", "license": "apache-2.0",
                "split": "train", "tags": ["pub:typed-decisions", f"workflow:{rec.get('workflow', '')}", f"group:typed-{rec['id']}"],
                "state": state, "questions": questions, "targets": targets,
            }
            if not check_row(row, min_q=1):
                yield row
                n += 1
                if limit and n >= limit:
                    return


# ---------------------------------------------------------------- PAWS-X (de, en)

PAWSX_PROMPT = """Write {language} text for a yes/no question about two sentences shown to a model.
"instructions": 3 different short questions (at most 14 words each) asking whether the two sentences say the same thing, so that "yes" means they are paraphrases and "no" means the meaning differs.
"sentence_1_label" and "sentence_2_label": short natural {language} labels for the two sentences, for example the words a {language} writer would use for "Sentence 1" and "Sentence 2".
Reply with JSON only: {{"instructions": [...], "sentence_1_label": "...", "sentence_2_label": "..."}}"""


async def pawsx_texts(client: Client, lang: str) -> dict:
    """Question wording for PAWS-X rows, written once per language by a teacher and cached."""
    path = data_dir() / "templates" / lang / "pawsx.json"
    if path.exists():
        return json.loads(path.read_text())
    writer = "glm"

    def check(data: Any) -> dict:
        if not isinstance(data, Mapping) or len(data.get("instructions") or []) < 3:
            raise ValueError("expected 3 instructions")
        for key in ("sentence_1_label", "sentence_2_label"):
            if not isinstance(data.get(key), str) or not data[key].strip():
                raise ValueError(f"missing {key}")
        return dict(data)

    msg = PAWSX_PROMPT.format(language=LANG_NAMES[lang])
    data, reply = await client.ask_json(writer, [{"role": "user", "content": msg}], check=check, temperature=0.3, max_tokens=1500, purpose="pawsx")
    data["writer"] = f"{TEACHERS[writer].provider}/{reply.model}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    return data


def pawsx_rows(lang: str, texts: Mapping, *, split: str = "train", limit: int | None = None, seed: str = "p") -> Iterator[dict]:
    """PAWS-X paraphrase pairs as one Noul question (yes = same meaning), gold from the human label."""
    if lang not in {"de", "en"}:
        return
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "google-research-datasets/paws-x", f"{lang}/{split}-00000-of-00001.parquet", repo_type="dataset",
        local_dir=str(data_dir() / "public" / "raw" / "paws-x"),
    )
    n = 0
    for rec in pq.read_table(path).to_pylist():
        s1, s2 = (rec.get("sentence1") or "").strip(), (rec.get("sentence2") or "").strip()
        if not s1 or not s2 or s1 == s2:
            continue
        r = rng("pawsx", lang, rec["id"], seed)
        row = {
            "id": f"{lang}-p-pawsx-{rec['id']}", "lang": lang, "source": "pub:paws-x", "license": "other-permissive",
            "split": "train", "tags": ["pub:paws-x", "type:noul", f"group:pawsx-{lang}-{rec['id']}"],
            "state": f"{texts['sentence_1_label']}: {s1}\n{texts['sentence_2_label']}: {s2}",
            "questions": {"same_meaning": {"type": "noul", "instructions": r.choice(texts["instructions"])}},
            "targets": {"same_meaning": target(one_hot("yes" if int(rec["label"]) == 1 else "no", ["yes", "no"]))},
        }
        if not check_row(row, min_q=1):
            yield row
            n += 1
            if limit and n >= limit:
                return
