"""Generation recovery: missing cards are re-requested in small chunks, from a fallback
writer when the original provider is quota-limited (no network, writer mocked)."""

import asyncio

from jevalt.data import generate


class FakeClient:
    def __init__(self, ollama_up):
        self.ollama_up = ollama_up

    async def available(self, provider):
        return provider != "ollama" or self.ollama_up


def run(writer, lang, cards, results, ollama_up):
    calls = []

    async def fake_batch(client, lang_, writer_, chunk, batch, cache_only=False):
        calls.append((writer_, len(chunk)))
        return results(writer_, chunk)

    generate.write_batch, original = fake_batch, generate.write_batch
    try:
        rows, drops = asyncio.run(generate.write_cards(FakeClient(ollama_up), lang, writer, cards, 0))
    finally:
        generate.write_batch = original
    return rows, drops, calls


def cards(n):
    return [{"card": i + 1, "domain": "logistics"} for i in range(n)]


def test_complete_batches_are_not_retried():
    rows, drops, calls = run("dst", "en", cards(8), lambda w, c: ([{"id": x["card"]} for x in c], []), True)
    assert calls == [("dst", 8)] and len(rows) == 8


def test_failed_batch_on_limited_ollama_moves_to_the_language_fallback():
    def results(writer, chunk):
        if writer == "k3ot":
            return [], [{"stage": "gen", "reason": "writer_failed"}]
        return [{"id": x["card"], "by": writer} for x in chunk], []

    rows, drops, calls = run("k3ot", "de", cards(8), results, ollama_up=True)  # same choice whether or not Ollama answers
    assert calls == [("k3ot", 8), ("glmt", 4), ("glmt", 4)]
    assert len(rows) == 8 and {r["by"] for r in rows} == {"glmt"}
    assert all(d.get("recovered_later") for d in drops)


def test_turkish_fallback_is_glmt_and_empty_chunks_try_without_thinking():
    # TR writers are mistral+ds on Ollama; when mistral fails the sibling ds
    # is tried first, then glmt (Z.ai), then glm (no-thinking).
    def results(writer, chunk):
        if writer in {"mistral", "ds", "glmt"}:
            return [], [{"stage": "gen", "reason": "writer_failed"}]
        return [{"id": x["card"], "by": writer} for x in chunk], []

    rows, _, calls = run("mistral", "tr", cards(4), results, ollama_up=False)
    assert calls == [("mistral", 4), ("ds", 4), ("glmt", 4), ("glm", 4)] and {r["by"] for r in rows} == {"glm"}


def test_only_missing_cards_are_requested_again():
    def results(writer, chunk):
        if len(chunk) == 8:  # the writer skipped cards 7 and 8
            return [{"id": x["card"]} for x in chunk[:6]], [{"stage": "gen", "reason": "writer_skipped_card", "card": 7},
                                                             {"stage": "gen", "reason": "writer_skipped_card", "card": 8}]
        return [{"id": x["card"]} for x in chunk], []

    rows, _, calls = run("glmt", "en", cards(8), results, ollama_up=True)
    assert calls == [("glmt", 8), ("glmt", 2)] and sorted(r["id"] for r in rows) == list(range(1, 9))
