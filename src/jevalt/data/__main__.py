"""CLI: python -m jevalt.data <stage> --lang en,tr,de [options]

Stages: quota, licenses, public, gen, label, fix, reason, qa, pilot.
Outputs go to $JEVALT_DATA (default ~/projects/jev/data), never into the repo.
"""

from __future__ import annotations

import argparse
import asyncio
import json

from . import bakeoff, pipeline, public, qa
from .common import LANGS, data_dir, write_jsonl
from .providers import Client

PILOT_SETS = ["f1", "f2", "f3", "f5", "f6", "f8", "f10", "f12"]
FULL_SETS = ["f1", "f2", "f3", "f5", "f6", "f7", "f8", "f10", "f11", "f12"]  # F9 skipped: needs teacher calls
PILOT_TRACES = {"g": 4, "f5": 3, "f6": 3, "f10": 3, "f12": 3}
FULL_TRACES = {"g": 120, "f5": 40, "f6": 40, "f9": 20, "f10": 40, "f12": 40}  # about 280 per language
FULL_G = {"en": 1500, "tr": 3000, "de": 2500}  # G rows kept after labels in the full run


async def run(args: argparse.Namespace) -> None:
    langs = [x for x in args.lang.split(",") if x]
    if args.stage == "licenses":
        print(public.license_table())
        return
    if args.stage == "qa":
        for lang in langs:
            _qa(lang, args.run)
        return
    # The pilot never idles on a quota-limited plan: calls fail fast and stages switch teachers.
    wait = (args.stage != "pilot" or args.wait) and not args.no_wait
    async with Client(run=args.run, concurrency=args.concurrency, wait_on_quota=wait) as client:
        if args.stage == "quota":
            print(json.dumps(await client.kimi_usage(), indent=1))
        elif args.stage == "public":
            for lang in langs:
                await _public(client, lang, args.limit)
        elif args.stage == "gen":
            await asyncio.gather(*[pipeline.stage_gen(client, lang, run=args.run, n=args.n, seed=args.seed) for lang in langs])
        elif args.stage == "label":
            targets = {lang: FULL_G[lang] if args.run == "full" else pipeline.G_TARGET for lang in langs}
            await asyncio.gather(*[pipeline.stage_label(client, lang, run=args.run, target_rows=targets[lang]) for lang in langs])
        elif args.stage == "fix":
            sets = args.sets.split(",") if args.sets else PILOT_SETS
            await asyncio.gather(*[pipeline.stage_fix(client, lang, run=args.run, sets=sets, n=args.fix_n, seed=args.seed) for lang in langs])
        elif args.stage == "reason":
            # English and German keep the traces of their GLM-only pass (about 150 each); new traces (DeepSeek V4 Pro on
            # the router) go to Turkish, the priority language, to stay inside the spend cap.
            plans = {lang: PILOT_TRACES if args.run == "pilot" else FULL_TRACES if lang == "tr" else {} for lang in langs}
            await asyncio.gather(*[pipeline.stage_reason(client, lang, run=args.run, plan=plans[lang]) for lang in langs])
        elif args.stage == "bakeoff":
            out_dir = data_dir() / "reports"
            combined = {}
            for lang in langs:
                print(f"Bake-off: {lang}", flush=True)
                results = await bakeoff.bakeoff(lang, args.run, args.concurrency)
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / f"native-bakeoff-{lang}.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
                combined[lang] = results
            for lang in langs:
                bakeoff.write_report(lang, combined.get(lang, {}), out_dir)
            if combined:
                bakeoff.write_combined_report(combined, out_dir)
            return
        elif args.stage == "pilot":
            before = await client.kimi_usage()

            async def one(lang: str) -> None:
                prefetch = asyncio.ensure_future(pipeline.prefetch_templates(client, lang, PILOT_SETS))
                order = ["gen", "label", "fix", "reason"]
                todo = order[order.index(args.start):]
                sets = args.sets.split(",") if args.sets else PILOT_SETS
                try:
                    if "gen" in todo:
                        await pipeline.stage_gen(client, lang, run=args.run, n=args.n, seed=args.seed)
                        print(lang, "gen done", flush=True)
                    if "label" in todo:
                        await pipeline.stage_label(client, lang, run=args.run)
                        print(lang, "label done", flush=True)
                    if "fix" in todo:
                        print(lang, "fix", await pipeline.stage_fix(client, lang, run=args.run, sets=sets, n=args.fix_n, seed=args.seed), flush=True)
                    if "reason" in todo:
                        print(lang, "traces", await pipeline.stage_reason(client, lang, run=args.run, plan=PILOT_TRACES), flush=True)
                except Exception:  # keep the other languages running
                    import traceback

                    traceback.print_exc()
                await prefetch

            await asyncio.gather(*[one(lang) for lang in langs])
            after = await client.kimi_usage()
            (data_dir() / args.run).mkdir(parents=True, exist_ok=True)
            (data_dir() / args.run / "kimi_usage.json").write_text(json.dumps({"before": before, "after": after}, indent=1))
            for lang in langs:
                _qa(lang, args.run)
        print({k: dict(v) for k, v in client.stats.items()})


def _qa(lang: str, run_name: str) -> None:
    rows, facts = pipeline.assemble(lang, run=run_name)
    (pipeline.stage_dir(run_name, lang) / "dups.json").write_text(json.dumps(facts["dups"], indent=1))
    usage_path = data_dir() / run_name / "kimi_usage.json"
    kimi = json.loads(usage_path.read_text()) if usage_path.exists() else None
    print(lang, len(rows), "rows;", "schema fails:", len(facts["gates"]["schema_fail"]), "report:", pipeline.write_report(lang, run=run_name, kimi=kimi))


async def _public(client: Client, lang: str, limit: int) -> None:
    out = data_dir() / "public" / lang
    scen_path = public._massive_file(lang, "train")
    scenarios, intents = public._massive_labels(scen_path)
    texts = await public.massive_texts(client, lang, scenarios, intents)
    n1 = write_jsonl(out / "massive.sample.jsonl", public.massive_rows(lang, texts, limit=limit))
    n2 = write_jsonl(out / "openjev.sample.jsonl", public.openjev_rows(lang, limit=limit)) if lang in {"en", "tr"} else 0
    n3 = write_jsonl(out / "typed.sample.jsonl", public.typed_decisions_rows(limit=limit)) if lang == "en" else 0
    n4 = 0
    if lang in {"de", "en"}:
        paws = await public.pawsx_texts(client, lang)
        n4 = write_jsonl(out / "pawsx.sample.jsonl", public.pawsx_rows(lang, paws, limit=limit))
    names = [f for f in ("massive.sample.jsonl", "openjev.sample.jsonl", "typed.sample.jsonl", "pawsx.sample.jsonl") if (out / f).exists()]
    bad = [r["id"] for path in names for r in _read(out / path) if qa.schema_problems(r)]
    print(lang, "massive", n1, "open-jev", n2, "typed-decisions", n3, "paws-x", n4, "schema problems", bad[:5])


def _read(path):
    from .common import read_jsonl

    return read_jsonl(path)


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m jevalt.data", description=__doc__.splitlines()[0])
    p.add_argument("stage", choices=["quota", "licenses", "public", "gen", "label", "fix", "reason", "qa", "pilot", "bakeoff"])
    p.add_argument("--lang", default=",".join(LANGS))
    p.add_argument("--run", default="pilot", help="output folder under the data dir and tag in the call log")
    p.add_argument("--n", type=int, default=56, help="G cards to write per language")
    p.add_argument("--fix-n", type=int, default=16, help="rows per fix set")
    p.add_argument("--sets", default="", help="fix sets, for example f1,f5")
    p.add_argument("--limit", type=int, default=20, help="rows per public converter sample")
    p.add_argument("--seed", default="pilot")
    p.add_argument("--concurrency", type=int, default=4, help="requests in flight per provider")
    p.add_argument("--wait", action="store_true", help="pilot: wait out quota limits instead of switching teachers")
    p.add_argument("--no-wait", action="store_true", help="fail fast on a used-up plan so writers fall back instead of waiting (gen)")
    p.add_argument("--start", default="gen", choices=["gen", "label", "fix", "reason"], help="pilot: resume at this stage")
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
