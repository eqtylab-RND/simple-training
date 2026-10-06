"""Minimal LoRA fine-tune of Nemotron Nano on banking77 intent classification.

Single H100, bf16. Loss is computed on the label tokens only -- the prompt
(including the 77-label list) is masked out. Reads data/train.jsonl +
data/labels.json, writes a LoRA adapter to out/adapter.

    python train.py                      # full run, ~10k examples, 2 epochs
    python train.py --limit 200 --epochs 1   # smoke test
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "out"
ADAPTER_DIR = OUT_DIR / "adapter"

BASE_MODEL = os.environ.get("BASE_MODEL", "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1")
SYSTEM_PROMPT = "detailed thinking off"  # Nemotron: bare label, no chain of thought
INSTRUCTION = (
    "Classify the customer message into exactly one intent label.\n"
    "Respond with only the label, nothing else."
)
MAX_SEQ_LEN = 1024
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build_user_turn(text, labels):
    parts = [INSTRUCTION]
    if labels:
        parts.append("\nLabels:\n" + "\n".join(labels))
    parts.append(f'\nMessage: "{text}"')
    return "\n".join(parts)


def render_prompt(tokenizer, user_turn):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_turn},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def read_jsonl(path):
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def encode(tokenizer, rows, labels, max_len):
    """Tokenize into prompt-masked examples (prompt tokens get label -100)."""
    encoded, truncated = [], 0
    for row in rows:
        prompt = render_prompt(tokenizer, build_user_turn(row["text"], labels))
        completion = row["label"] + tokenizer.eos_token
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(prompt + completion, add_special_tokens=False)["input_ids"]
        n_prompt = len(prompt_ids)
        if len(full_ids) > max_len:
            # Trim from the left so the answer always survives, shift the mask boundary too.
            overflow = len(full_ids) - max_len
            full_ids = full_ids[overflow:]
            n_prompt = max(n_prompt - overflow, 0)
            truncated += 1
        encoded.append({"input_ids": full_ids, "labels": [-100] * n_prompt + full_ids[n_prompt:]})
    if truncated:
        print(f"WARNING: {truncated}/{len(rows)} examples hit max_len={max_len}")
    return encoded


def collate(batch, pad_id):
    width = max(len(item["input_ids"]) for item in batch)
    input_ids, labels, mask = [], [], []
    for item in batch:
        pad = width - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_id] * pad)
        labels.append(item["labels"] + [-100] * pad)
        mask.append([1] * len(item["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(mask),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--limit", type=int, default=0, help="cap training rows (smoke test)")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    set_seed(args.seed)
    labels = json.loads((DATA_DIR / "labels.json").read_text())
    rows = read_jsonl(DATA_DIR / "train.jsonl")
    if args.limit:
        rows = rows[: args.limit]
    print(f"training rows: {len(rows)}  |  labels: {len(labels)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = encode(tokenizer, rows, labels, MAX_SEQ_LEN)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=LORA_TARGETS,
        ),
    )
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(OUT_DIR / "checkpoints"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            logging_steps=10,
            save_strategy="no",
            bf16=True,
            report_to=[],
            seed=args.seed,
        ),
        train_dataset=dataset,
        data_collator=lambda b: collate(b, tokenizer.pad_token_id),
    )
    trainer.train()

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))
    (ADAPTER_DIR / "train_config.json").write_text(
        json.dumps({**vars(args), "n_rows": len(rows)}, indent=2)
    )
    print(f"\nadapter saved to {ADAPTER_DIR}")


if __name__ == "__main__":
    main()
