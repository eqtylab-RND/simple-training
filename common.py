"""Shared config, prompt formatting, and label normalization for the PoC.

One place to change the model, the prompt, or how a raw generation gets
mapped back onto the banking77 taxonomy. Both train.py and evaluate.py
import from here so the two stay in lockstep -- if the prompt drifts
between training and evaluation the comparison is meaningless.
"""

from __future__ import annotations

import json
import inspect
import os
import re
from pathlib import Path

import eqty_sdk as sdk

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "out"
RESULTS_DIR = ROOT / "results"

BASE_MODEL = os.environ.get("BASE_MODEL", "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1")
ADAPTER_DIR = OUT_DIR / "adapter"

# Nemotron Nano ships a reasoning on/off toggle driven by the system prompt.
# Classification wants a bare label, not a chain of thought, so pin it off --
# and pin it identically for base and tuned so the comparison stays fair.
SYSTEM_PROMPT = "detailed thinking off"

INSTRUCTION = (
    "Classify the customer message into exactly one intent label.\n"
    "Respond with only the label, nothing else."
)

MAX_SEQ_LEN = 1024


def load_labels() -> list[str]:
    path = DATA_DIR / "labels.json"
    source = sdk.Dataset.from_path(path, name="Banking77 taxonomy file", _store=True)
    labels = json.loads(path.read_text())
    run = sdk.Computation.new(name="Load taxonomy", computation_type="ingest", _store=True)
    run.add_input_cid(sdk.Code.from_object(inspect.getsource(load_labels),
                                         name="load_labels", _store=True).cid)
    run.add_input_cid(source.cid)
    run.add_output_cid(sdk.Dataset.from_object(labels, name="Ordered intent labels",
                                             _store=True).cid)
    run.finalize()
    return labels


def build_user_turn(text: str, labels: list[str] | None) -> str:
    """The user message. `labels` is None in closed-book mode."""
    parts = [INSTRUCTION]
    if labels:
        parts.append("\nLabels:\n" + "\n".join(labels))
    parts.append(f'\nMessage: "{text}"')
    return "\n".join(parts)


def render_prompt(tokenizer, user_turn: str) -> str:
    """Chat-templated prompt, ready for the model to complete."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_turn},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


_NON_LABEL_CHARS = re.compile(r"[^a-z0-9_]+")


def canonical_label(raw: str) -> str:
    """Canonical form of a taxonomy label.

    banking77 ships two labels that are not already in this form:
    `Refund_not_showing_up` (capitalised) and `reverted_card_payment?`
    (trailing question mark). Canonicalizing the taxonomy at prepare time
    puts gold labels and normalized predictions in the same space. Without
    it those two classes score as wrong for every model on every example --
    quietly, and worst for the tuned model, which learns to emit the exact
    gold string and then fails to match it.
    """
    return _NON_LABEL_CHARS.sub("_", raw.strip().lower()).strip("_")


def normalize_prediction(raw: str) -> str:
    """Map a raw generation onto label-space as generously as is still honest.

    Applied identically to base and tuned output. We strip a reasoning block
    if one leaked through, take the first line, drop surrounding punctuation,
    and snap whitespace to underscores. What we deliberately do NOT do is
    fuzzy-match onto the nearest real label -- if the model emitted a
    sentence instead of a label, that is a failure and should score as one.
    """
    text = raw
    if "</think>" in text:
        text = text.split("</think>")[-1]
    text = text.strip().strip("`")
    # First non-empty line only; models that ramble get judged on their opener.
    for line in text.splitlines():
        if line.strip():
            text = line
            break
    text = text.strip().strip("`\"'.,:;!? ")
    return canonical_label(text)
