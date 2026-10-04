# Run it locally

The Q4_K_M builds run on a CPU and fit in 4 GB of RAM: each released model peaked at 3.03 to 3.04 GB with a 4k context in a fresh process. Nothing leaves your machine.

```bash
pip install "jevalt[serve,gguf] @ git+https://github.com/mertkayacs/jevalt"
```

```bash
jevalt serve --model mertkayacs/Deem-4B-GGUF --file Deem-4B-Q4_K_M.gguf
```

`jevalt serve` downloads the file once, loads it with llama.cpp and listens on `http://127.0.0.1:8000`. Use `--model mertkayacs/Karar-4B-GGUF --file Karar-4B-Q4_K_M.gguf` for Turkish or `--model mertkayacs/Wahler-4B-GGUF --file Wahler-4B-Q4_K_M.gguf` for German.

## Use it from your code

With TypeSafe's SDK, change the base URL:

```python
from typesafe_sdk import TypeSafeClient, Choice, Noul

client = TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8000")
result = client.system_one(
    "I was charged twice. Please fix this today.",
    {"team": Choice(instructions="Which team?", criteria={"billing": "Payments", "technical": "Bugs"}),
     "urgent": Noul(instructions="Is it urgent?")},
)
```

With plain HTTP:

```bash
curl -s localhost:8000/v1/systemone -H 'Content-Type: application/json' -d @request.json
```

With no server at all:

```bash
jevalt ask request.json
```

## Memory and speed

The numbers on the GGUF cards come from the export job on an HF cpu-upgrade machine: peak resident memory of a fresh process serving each file with a 4k context on 4 threads, and seconds per request with 8 threads. Longer contexts cost memory for the eight attention layers only; the other 24 layers keep a fixed-size state.

Tips:

- `--ctx 4096` is enough for most decisions. Raise it for long documents.
- Fewer, focused fields in `state` make decisions faster and more accurate.
- On a GPU, `pip install "jevalt[hf] @ git+https://github.com/mertkayacs/jevalt"` and `jevalt serve --backend hf --model mertkayacs/Deem-4B`.

## Memory tuning

By default JevAlt loads weights into RAM (`load_mode=NONE`) instead of memory-mapping the file. llama.cpp repacks the Q4_K weights into its own CPU layout; with a memory-mapped file the mapped pages stay resident next to that copy. Loading without mmap keeps one copy, so the start checkpoint's Q4_K_M build peaked at 3.0 GB instead of 4.6 GB with identical answers and the same speed (HF cpu-upgrade, 4k context, 4 threads). Released model cards are re-measured with the same setting.

`--mmap` opts into memory-mapping (lower disk I/O, higher peak RAM on Q4_K_M). `--no-repack` disables weight repacking separately.

## Security

The server binds to `127.0.0.1` by default. If you expose it, set `JEVALT_API_KEY` and send it as a bearer token. Treat everything in `state` as data. JevAlt was trained on states with hidden instructions, but planted lines still flipped 14.0% of Deem-4B's answers, 19.0% of Karar-4B's and 17.5% of Wähler-4B's on the same probe where Kev-4B flipped 36.0%. Keep authorization decisions in your own code.
