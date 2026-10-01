"""llama.cpp backend for GGUF files (CPU friendly, fits in 4 GB with Q4_K_M).

Uses llama-cpp-python's low-level API so logits are produced only at the
positions we read (one per decision marker), instead of a logits buffer for
the whole batch. That keeps memory flat for long states and large vocabularies.
"""

from __future__ import annotations

import ctypes

import numpy as np

from ..format import DECISION_TOKEN, Compiled


class LlamaCppBackend:
    name = "llama.cpp"

    def __init__(self, path: str, *, n_ctx: int = 8192, n_batch: int = 512, n_threads: int | None = None, n_gpu_layers: int = 0, mmap: bool | None = False, repack: bool = True):
        import llama_cpp as L

        self.L = L
        L.llama_backend_init()
        mparams = L.llama_model_default_params()
        mparams.n_gpu_layers = n_gpu_layers
        if mmap is not None and hasattr(mparams, "load_mode"):
            if mmap:
                mparams.load_mode = getattr(L, "LLAMA_LOAD_MODE_MMAP", 1)
            else:
                mparams.load_mode = getattr(L, "LLAMA_LOAD_MODE_NONE", 0)
        if not repack and hasattr(mparams, "use_extra_bufts"):
            mparams.use_extra_bufts = False
        self.model = L.llama_model_load_from_file(path.encode(), mparams)
        if not self.model:
            raise RuntimeError(f"could not load {path}")
        cparams = L.llama_context_default_params()
        cparams.n_ctx = n_ctx
        cparams.n_batch = n_batch
        cparams.n_ubatch = n_batch
        if n_threads:
            cparams.n_threads = n_threads
            cparams.n_threads_batch = n_threads
        self.ctx = L.llama_init_from_model(self.model, cparams)
        self.vocab = L.llama_model_get_vocab(self.model)
        self.n_vocab = L.llama_vocab_n_tokens(self.vocab)
        self.n_ctx, self.n_batch = n_ctx, n_batch
        self.marker = self._single(DECISION_TOKEN)
        self.stop = {self._single("</think>"), self._single("<|im_end|>")}
        self._symbols: dict[str, int] = {}

    # -- tokenizer -------------------------------------------------------
    def tokenize(self, text: str) -> list[int]:
        data = text.encode("utf-8")
        cap = len(data) + 16
        buf = (self.L.llama_token * cap)()
        n = self.L.llama_tokenize(self.vocab, data, len(data), buf, cap, False, True)
        if n < 0:
            raise RuntimeError("tokenization buffer too small")
        return list(buf[:n])

    def _single(self, text: str) -> int:
        ids = self.tokenize(text)
        if len(ids) != 1:
            raise ValueError(f"{text!r} is not a single token in this GGUF")
        return ids[0]

    def _symbol_id(self, symbol: str) -> int:
        if symbol not in self._symbols:
            self._symbols[symbol] = self._single(symbol)
        return self._symbols[symbol]

    def piece(self, token: int) -> bytes:
        buf = ctypes.create_string_buffer(64)
        n = self.L.llama_token_to_piece(self.vocab, token, buf, len(buf), 0, True)
        return buf.raw[:n]

    def count_tokens(self, text: str) -> int:
        return len(self.tokenize(text))

    # -- evaluation ------------------------------------------------------
    def _reset(self):
        L = self.L
        if hasattr(L, "llama_memory_clear"):
            L.llama_memory_clear(L.llama_get_memory(self.ctx), True)
        elif hasattr(L, "llama_kv_self_clear"):
            L.llama_kv_self_clear(self.ctx)
        else:
            L.llama_kv_cache_clear(self.ctx)

    def _eval(self, tokens: list[int], start: int, want: set[int]) -> dict[int, np.ndarray]:
        """Decode tokens at positions start.. in chunks; return logits for positions in `want`."""
        L = self.L
        out: dict[int, np.ndarray] = {}
        for lo in range(0, len(tokens), self.n_batch):
            chunk = tokens[lo : lo + self.n_batch]
            batch = L.llama_batch_init(len(chunk), 0, 1)
            try:
                rows = []
                for j, tok in enumerate(chunk):
                    pos = start + lo + j
                    batch.token[j] = tok
                    batch.pos[j] = pos
                    batch.n_seq_id[j] = 1
                    batch.seq_id[j][0] = 0
                    batch.logits[j] = pos in want
                    if pos in want:
                        rows.append((j, pos))
                batch.n_tokens = len(chunk)
                if L.llama_decode(self.ctx, batch) != 0:
                    raise RuntimeError("llama_decode failed (context too small?)")
                for j, pos in rows:
                    ptr = L.llama_get_logits_ith(self.ctx, j)
                    out[pos] = np.ctypeslib.as_array(ptr, shape=(self.n_vocab,)).copy()
            finally:
                L.llama_batch_free(batch)
        return out

    def field_logits(self, text: str, compiled: Compiled) -> list[list[float]]:
        tokens = self.tokenize(text)
        if len(tokens) > self.n_ctx:
            raise ValueError(f"prompt has {len(tokens)} tokens, above n_ctx={self.n_ctx}")
        positions = [i - 1 for i, t in enumerate(tokens) if t == self.marker]
        if len(positions) != len(compiled.fields):
            raise ValueError("decision marker count does not match the question count")
        self._reset()
        logits = self._eval(tokens, 0, set(positions))
        rows = []
        for pos, field in zip(positions, compiled.fields):
            rows.append([float(logits[pos][self._symbol_id(s)]) for s in field.symbols])
        return rows

    def continue_reasoning(self, prefix: str, max_tokens: int) -> str:
        tokens = self.tokenize(prefix)
        self._reset()
        last = len(tokens) - 1
        logits = self._eval(tokens, 0, {last})[last]
        out = bytearray()
        pos = len(tokens)
        for _ in range(max_tokens):
            tok = int(np.argmax(logits))
            if tok in self.stop:
                break
            out += self.piece(tok)
            logits = self._eval([tok], pos, {pos})[pos]
            pos += 1
        return out.decode("utf-8", errors="replace")
