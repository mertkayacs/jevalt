"""Tests for training/cards.py: card() and gguf_card() with fixture JSONs."""

import json

import pytest

from training.cards import card, gguf_card

# --- Fixtures ---

OURS = {
    "results": [
        {"suite": "heldout-en", "reasoning": "off", "overall": {"n": 100, "accuracy": 0.85, "brier": 0.12}},
        {"suite": "heldout-tr", "reasoning": "off", "overall": {"n": 100, "accuracy": 0.82, "brier": 0.14}},
    ]
}

BASE = {
    "results": [
        {"suite": "heldout-en", "reasoning": "off", "overall": {"n": 100, "accuracy": 0.78, "brier": 0.18}},
        {"suite": "heldout-tr", "reasoning": "off", "overall": {"n": 100, "accuracy": 0.75, "brier": 0.20}},
    ]
}

GGUF_REPORT = {
    "sizes_gb": {"Q4_K_M": 2.7, "Q5_K_M": 3.2, "Q8_0": 4.5},
    "parity": {
        "Q4_K_M": {"argmax_agreement": 0.98, "max_prob_gap": 0.02, "sec_per_request": 0.24},
        "Q5_K_M": {"argmax_agreement": 0.99, "max_prob_gap": 0.01, "sec_per_request": 0.28},
    },
    "memory": {"Q4_K_M": {"peak_rss_gb": 3.4}, "Q5_K_M": {"peak_rss_gb": 3.2}},
}

MEMORY_REPORT = {
    "runs": {
        "Deem-4B-Q4_K_M.gguf": {
            "mmap+repack": {"peak_rss_gb": 4.64, "sec_per_request": 0.24, "argmax_agreement": 1.0},
            "nommap+repack": {"peak_rss_gb": 3.39, "sec_per_request": 0.30, "argmax_agreement": 1.0},
        }
    }
}

TRAINING = {
    "base_model": "internlm/Intern-Decision-4B",
    "lora_rank": 32,
    "lora_alpha": 32,
    "epochs": 3,
    "teachers": ["Kimi K3", "GLM 5.x", "DeepSeek V4"],
    "sources": [
        {"source": "Generated scenarios", "en": 20000, "tr": 15000, "de": 12000},
        {"source": "Fix sets F1-F12", "en": 5000, "tr": 4000, "de": 3000},
    ],
}

PROBE_OURS = {
    "probes": {
        "permutation": {"value": 0.088},
        "injection": {"value": 0.36},
        "distractors": {"value": 0.054},
        "noul-vs-choice": {"value": 0.033},
        "determinism": {"value": 0.0},
    }
}

PROBE_BASE = {
    "probes": {
        "permutation": {"value": 0.0875},
        "injection": {"value": 0.415},
        "distractors": {"value": 0.15},
        "noul-vs-choice": {"value": 0.032},
        "determinism": {"value": 0.0},
    }
}

SUITES = ["heldout-en", "heldout-tr"]


# --- card() tests ---

def test_card_full():
    out = card("en", OURS, BASE, SUITES, GGUF_REPORT, TRAINING, PROBE_OURS, PROBE_BASE)
    assert "Deem-4B" in out
    assert "Try it" in out
    assert "How it was trained" in out
    assert "LoRA rank: 32" in out
    assert "Use and limits" in out
    assert "Citation" in out
    assert 'datasets: ["mertkayacs/jevalt-data"]' in out
    assert "@software{kaya2026jevalt" in out
    assert "Option order" in out
    assert "8.8%" in out  # probe formatted as pct


def test_card_training_details_and_fixes_fold():
    training = {**TRAINING, "method": "LoRA on the bf16 weights, rank 32, alpha 32", "runs": "11 training jobs",
                "compute": "about 2.7 A100 hours", "teachers": ["GLM-5.x", "Kimi K3 (54 rows)"]}
    fixes = "<details>\n<summary><b>How we fixed each problem</b></summary>\n\n- **Hidden instructions.** x\n\n</details>\n"
    out = card("en", OURS, BASE, SUITES, GGUF_REPORT, training, fixes=fixes)
    assert "- Method: LoRA on the bf16 weights, rank 32, alpha 32" in out
    assert "LoRA rank:" not in out
    assert "- Runs: 11 training jobs" in out and "- Compute: about 2.7 A100 hours" in out
    assert "- Writers, labelers and trace writers: GLM-5.x, Kimi K3 (54 rows)" in out
    assert out.index("How we fixed each problem") < out.index("How it was trained") < out.index("## Use and limits")
    assert "</details>\n\n<details>" in out, "a blank line keeps the two folds apart"


def test_card_missing_optional():
    out = card("tr", OURS, BASE, SUITES)
    assert "Karar-4B" in out
    assert "Try it" in out
    assert "How it was trained" not in out
    assert "Probes" not in out
    assert "Use and limits" in out
    assert "Citation" in out
    assert "<summary><b>Türkçe özet</b></summary>" in out


def test_card_examples():
    table = "| Use case | Situation | Question | Answer |\n|---|---|---|---|\n| Support ticket | x | y | **Billing** 94.8% |"
    out = card("en", OURS, BASE, SUITES, examples=table, examples_url="https://example.org/space-examples.json")
    assert "**Billing** 94.8%" in out
    assert "(https://example.org/space-examples.json)" in out
    assert out.index("## Try it") < out.index("## Run it")


def test_card_partial_gguf():
    partial = {"sizes_gb": {"Q4_K_M": 2.7}, "parity": {}}  # no memory key
    out = card("de", OURS, BASE, SUITES, partial)
    assert "Wähler-4B" in out
    assert "peak RAM" not in out  # no RAM line when memory missing


def test_card_probe_only_ours():
    out = card("en", OURS, BASE, SUITES, None, None, PROBE_OURS, None)
    assert "Probes" in out
    assert "-" in out  # baseline column is all dashes


def test_card_no_crash_empty_dicts():
    out = card("en", OURS, BASE, SUITES, {}, {}, {}, {})
    assert "Deem-4B" in out
    assert "Try it" in out


# --- gguf_card() tests ---

def test_gguf_card_full():
    out = gguf_card("en", GGUF_REPORT, MEMORY_REPORT)
    assert "Deem-4B GGUF" in out
    assert "Q4_K_M" in out
    assert "Q5_K_M" in out
    assert "Q8_0" in out
    assert "2.70 GB" in out  # file size from export report
    assert "98.0%" in out  # argmax agreement from parity
    assert "4.64 GB" in out  # peak RAM from memory report (mmap+repack)
    assert "mmap+repack" in out  # load setting
    assert "llama-server" in out
    assert "LM Studio" in out
    assert "jevalt serve" in out


def test_gguf_card_no_memory_report():
    out = gguf_card("tr", GGUF_REPORT, None)
    assert "Karar-4B GGUF" in out
    assert "2.70 GB" in out
    assert "3.40 GB" in out  # peak from export report memory
    assert "no mmap, repack on (jevalt default)" in out  # load setting of the export measurement


def test_gguf_card_missing_inputs():
    partial = {"sizes_gb": {"Q4_K_M": 2.7}}
    out = gguf_card("de", partial, None)
    assert "Wähler-4B GGUF" in out
    assert "Q4_K_M" in out
    assert "Q5_K_M" in out  # still listed
    assert "-" in out  # missing values render as dash


def test_gguf_card_none_reports():
    out = gguf_card("en", None, None)
    assert "Deem-4B GGUF" in out
    assert "Q4_K_M" in out
    assert "-" in out
    assert "llama-server" in out  # usage section still renders


def test_gguf_card_yaml_front_matter():
    out = gguf_card("en", GGUF_REPORT, None)
    assert "base_model" in out
    assert "mertkayacs/Deem-4B" in out
    assert "quantized_by" in out
    assert "gguf" in out
    assert "apache-2.0" in out
