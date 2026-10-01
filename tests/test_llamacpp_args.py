"""Test LlamaCppBackend mmap and repack args with a fake llama_cpp module."""

import ctypes
import sys
import types

import numpy as np
import pytest


class FakeMparams:
    def __init__(self, *, with_load_mode=True, with_use_extra_bufts=True):
        self.n_gpu_layers = 0
        self.load_mode = -1  # LLAMA_LOAD_MODE_AUTO
        self.use_extra_bufts = True
        if not with_load_mode:
            del self.load_mode
        if not with_use_extra_bufts:
            del self.use_extra_bufts


class FakeCparams:
    n_ctx = 0
    n_batch = 0
    n_ubatch = 0
    n_threads = 0
    n_threads_batch = 0


def make_fake_llama_cpp(*, with_load_mode=True, with_use_extra_bufts=True):
    mod = types.ModuleType("llama_cpp")
    mod.LLAMA_LOAD_MODE_AUTO = -1
    mod.LLAMA_LOAD_MODE_NONE = 0
    mod.LLAMA_LOAD_MODE_MMAP = 1
    mod.LLAMA_LOAD_MODE_MLOCK = 2
    mod.LLAMA_LOAD_MODE_MMAP_MLOCK = 3
    mod.LLAMA_LOAD_MODE_DIRECT_IO = 4

    captured = {}

    def llama_model_default_params():
        m = FakeMparams(with_load_mode=with_load_mode, with_use_extra_bufts=with_use_extra_bufts)
        captured["mparams"] = m
        return m

    mod.llama_token = ctypes.c_int
    mod.llama_backend_init = lambda: None
    mod.llama_model_default_params = llama_model_default_params
    mod.llama_model_load_from_file = lambda path, mparams: 1
    mod.llama_context_default_params = lambda: FakeCparams()
    mod.llama_init_from_model = lambda model, cparams: 1
    mod.llama_model_get_vocab = lambda model: 1
    mod.llama_vocab_n_tokens = lambda vocab: 128
    mod.llama_tokenize = lambda vocab, data, n, buf, cap, add_special, parse_special: 1
    mod.llama_batch_init = lambda n_tokens, embd, seq: types.SimpleNamespace(
        n_tokens=0, token=(ctypes.c_int * 64)(), pos=(ctypes.c_int * 64)(),
        n_seq_id=(ctypes.c_int * 64)(), seq_id=(ctypes.POINTER(ctypes.c_int) * 64)(),
        logits=(ctypes.c_float * 64)(),
    )
    mod.llama_batch_free = lambda b: None
    mod.llama_decode = lambda ctx, batch: 0
    mod.llama_get_logits_ith = lambda ctx, i: (ctypes.c_float * 128)()
    mod.llama_token_to_piece = lambda vocab, token, buf, n, l, t: 1
    mod.llama_memory_clear = lambda mem, clear: None
    mod.llama_get_memory = lambda ctx: None

    return mod, captured


@pytest.fixture
def fake_llama(monkeypatch):
    mod, captured = make_fake_llama_cpp()
    monkeypatch.setitem(sys.modules, "llama_cpp", mod)
    return captured


@pytest.fixture
def fake_llama_no_attrs(monkeypatch):
    mod, captured = make_fake_llama_cpp(with_load_mode=False, with_use_extra_bufts=False)
    monkeypatch.setitem(sys.modules, "llama_cpp", mod)
    return captured


def test_mmap_false_sets_none(fake_llama):
    from jevalt.backends.llamacpp import LlamaCppBackend

    LlamaCppBackend("/fake.gguf", mmap=False)
    m = fake_llama["mparams"]
    assert m.load_mode == 0  # LLAMA_LOAD_MODE_NONE


def test_mmap_true_sets_mmap(fake_llama):
    from jevalt.backends.llamacpp import LlamaCppBackend

    LlamaCppBackend("/fake.gguf", mmap=True)
    m = fake_llama["mparams"]
    assert m.load_mode == 1  # LLAMA_LOAD_MODE_MMAP


def test_mmap_default_is_false(fake_llama):
    from jevalt.backends.llamacpp import LlamaCppBackend

    LlamaCppBackend("/fake.gguf")
    m = fake_llama["mparams"]
    assert m.load_mode == 0  # LLAMA_LOAD_MODE_NONE is the default now


def test_repack_false_disables(fake_llama):
    from jevalt.backends.llamacpp import LlamaCppBackend

    LlamaCppBackend("/fake.gguf", repack=False)
    m = fake_llama["mparams"]
    assert m.use_extra_bufts is False


def test_repack_true_keeps_default(fake_llama):
    from jevalt.backends.llamacpp import LlamaCppBackend

    LlamaCppBackend("/fake.gguf", repack=True)
    m = fake_llama["mparams"]
    assert m.use_extra_bufts is True


def test_missing_attrs_still_loads(fake_llama_no_attrs):
    from jevalt.backends.llamacpp import LlamaCppBackend

    backend = LlamaCppBackend("/fake.gguf", mmap=False, repack=False)
    m = fake_llama_no_attrs["mparams"]
    assert not hasattr(m, "load_mode")
    assert not hasattr(m, "use_extra_bufts")
    assert backend.model == 1
