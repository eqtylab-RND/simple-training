"""LoRA fine-tune of Nemotron Nano on banking77 intent classification.

Single H100, bf16, no quantization needed at 4B. Loss is computed on the
label tokens only -- the prompt (including the 77-label list) is masked out,
so the model is never rewarded for reciting the taxonomy back at us.

    python train.py                     # full run, ~10k examples
    python train.py --limit 200 --epochs 1   # smoke test, a couple of minutes

Writes a LoRA adapter to out/adapter. Evaluation is a separate trigger:
see evaluate.py.
"""

from __future__ import annotations

import argparse
import json

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from common import (
    ADAPTER_DIR,
    BASE_MODEL,
    DATA_DIR,
    MAX_SEQ_LEN,
    OUT_DIR,
    build_user_turn,
    load_labels,
    render_prompt,
)

LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


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
            # Trim from the left so the answer always survives, and shift the
            # mask boundary by the same amount. Truncating both sequences
            # independently would push the boundary past the answer and mask
            # out the entire example -- a silent no-op batch.
            overflow = len(full_ids) - max_len
            full_ids = full_ids[overflow:]
            n_prompt = max(n_prompt - overflow, 0)
            truncated += 1

        target = [-100] * n_prompt + full_ids[n_prompt:]
        encoded.append({"input_ids": full_ids, "labels": target})

    if truncated:
        print(f"WARNING: {truncated}/{len(rows)} examples hit max_len={max_len}; "
              f"raise MAX_SEQ_LEN in common.py")
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
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--no-label-list", action="store_true",
                    help="closed-book: omit the 77 labels from the prompt")
    args = ap.parse_args()

    set_seed(args.seed)
    labels = None if args.no_label_list else load_labels()
    rows = read_jsonl(DATA_DIR / "train.jsonl")
    if args.limit:
        rows = rows[: args.limit]
    print(f"training rows: {len(rows)}  |  label list in prompt: {labels is not None}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = encode(tokenizer, rows, labels, MAX_SEQ_LEN)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.config.use_cache = False
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGETS,
    )
    model = get_peft_model(
        model,
        lora_config,
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
    train_config = {**vars(args), "n_rows": len(rows)}
    (ADAPTER_DIR / "train_config.json").write_text(
        json.dumps(train_config, indent=2)
    )

    print(f"\nadapter saved to {ADAPTER_DIR}")
    print("next: python evaluate.py")


if __name__ == "__main__":
    main()
