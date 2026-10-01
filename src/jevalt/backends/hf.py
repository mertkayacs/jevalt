"""Transformers backend (GPU or CPU)."""

from __future__ import annotations

from ..format import DECISION_TOKEN, Compiled


class HFBackend:
    name = "transformers"

    def __init__(self, model_id: str, *, device: str | None = None, dtype: str = "bfloat16", revision: str | None = None, model=None, tokenizer=None):
        import torch
        from transformers import AutoConfig, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.marker = self.tokenizer.convert_tokens_to_ids(DECISION_TOKEN)
        if self.marker is None or self.marker == self.tokenizer.unk_token_id:
            raise ValueError("tokenizer has no <decision> token; this is not a decision checkpoint")
        if model is None:
            arch = (AutoConfig.from_pretrained(model_id, revision=revision).architectures or [""])[0]
            if arch.endswith("ForConditionalGeneration"):
                import transformers

                cls = getattr(transformers, arch)
            else:
                from transformers import AutoModelForCausalLM as cls
            model = cls.from_pretrained(model_id, revision=revision, dtype=getattr(torch, dtype)).to(self.device).eval()
        self.model = model
        self.stop_ids = [self.tokenizer.convert_tokens_to_ids(t) for t in ("</think>", "<|im_end|>")]
        self._symbols: dict[str, int] = {}

    def _symbol_id(self, symbol: str) -> int:
        if symbol not in self._symbols:
            ids = self.tokenizer.encode(symbol, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"answer symbol {symbol!r} is not a single token")
            self._symbols[symbol] = ids[0]
        return self._symbols[symbol]

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def field_logits(self, text: str, compiled: Compiled) -> list[list[float]]:
        torch = self.torch
        ids = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.device)
        positions = (ids[0] == self.marker).nonzero().flatten() - 1
        if len(positions) != len(compiled.fields):
            raise ValueError("decision marker count does not match the question count")
        with torch.inference_mode():
            logits = self.model(input_ids=ids, use_cache=False, logits_to_keep=positions).logits[0].float()
        rows = []
        for i, field in enumerate(compiled.fields):
            sym = torch.tensor([self._symbol_id(s) for s in field.symbols], device=logits.device)
            rows.append(logits[i, sym].tolist())
        return rows

    def continue_reasoning(self, prefix: str, max_tokens: int) -> str:
        torch = self.torch
        ids = self.tokenizer(prefix, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.device)
        with torch.inference_mode():
            out = self.model.generate(input_ids=ids, max_new_tokens=max_tokens, do_sample=False, eos_token_id=self.stop_ids, pad_token_id=self.stop_ids[-1])
        new = out[0, ids.shape[1] :]
        text = self.tokenizer.decode(new, skip_special_tokens=False)
        return text.split("</think>")[0].split("<|im_end|>")[0]
