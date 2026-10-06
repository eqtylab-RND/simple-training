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
import inspect
import json
import os
from pathlib import Path

import eqty_sdk
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
        rows = [json.loads(line) for line in fh if line.strip()]
    source = eqty_sdk.Code.from_object(
        inspect.getsource(read_jsonl), name="read_jsonl source"
    )
    source_file = eqty_sdk.Document.from_path(
        path, name="Banking77 training examples JSONL"
    )
    row_data = eqty_sdk.Dataset.from_object(
        rows, name="Banking77 training examples"
    )
    (
        eqty_sdk.Computation.new(
            name="Load training examples",
            description="Read the Banking77 training examples from disk.",
            computation_type="ingest",
        )
        .add_input_cid([source.cid, source_file.cid])
        .add_output_cid(row_data.cid)
        .set_computation_cid(source.cid)
        .finalize()
    )
    return rows


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
    source = eqty_sdk.Code.from_object(
        inspect.getsource(encode), name="encode source"
    )
    input_cids = [
        source.cid,
        eqty_sdk.Dataset.from_object(
            rows, name="Training examples supplied for encoding"
        ).cid,
        eqty_sdk.Configuration.from_object(
            {
                "tokenizer": getattr(tokenizer, "name_or_path", type(tokenizer).__name__),
                "max_sequence_length": max_len,
            },
            name="Training encoder configuration",
        ).cid,
    ]
    if labels is not None:
        input_cids.append(
            eqty_sdk.Dataset.from_object(
                labels, name="Labels supplied in training prompts"
            ).cid
        )
    encoded_data = eqty_sdk.Dataset.from_object(
        encoded, name="Prompt-masked training examples"
    )
    (
        eqty_sdk.Computation.new(
            name="Encode training examples",
            description="Build prompts, tokenize completions, and mask prompt tokens from loss.",
            computation_type="transform",
            truncated_examples=truncated,
        )
        .add_input_cid(input_cids)
        .add_output_cid(encoded_data.cid)
        .set_computation_cid(source.cid)
        .finalize()
    )
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

    eqty_dir = Path(
        os.environ.get(
            "EQTY_SDK_DIR",
            Path.home() / ".local" / "state" / "banking77-train-eqty",
        )
    )
    cfg = eqty_sdk.init(
        default_context=eqty_sdk.Context.new("Banking77 model training"),
        custom_dir=eqty_dir,
    ).set_store_all_blobs(True)
    notary_url = os.environ.get("EQTY_NOTARY_URL")
    eqty_sdk.set_active_signer(
        eqty_sdk.Signer.vcomp_notary(
            url=notary_url,
            name="banking77-trainer-notary",
            _load_if_exists=True,
        )
        if notary_url
        else eqty_sdk.Signer.load_or_create(name="banking77-trainer")
    )

    main_source = eqty_sdk.Code.from_object(
        inspect.getsource(main), name="main source"
    )
    set_seed(args.seed)
    labels = None if args.no_label_list else load_labels()
    if labels is not None:
        label_file = eqty_sdk.Document.from_path(
            DATA_DIR / "labels.json", name="Banking77 label taxonomy JSON"
        )
        label_data = eqty_sdk.Dataset.from_object(
            labels, name="Banking77 label taxonomy"
        )
        (
            eqty_sdk.Computation.new(
                name="Load label taxonomy",
                description="Read the Banking77 taxonomy used in training prompts.",
                computation_type="ingest",
            )
            .add_input_cid([main_source.cid, label_file.cid])
            .add_output_cid(label_data.cid)
            .set_computation_cid(main_source.cid)
            .finalize()
        )
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

    base_model_reference = eqty_sdk.Configuration.from_object(
        args.model,
        name="Hugging Face base-model repository identifier",
        representation="repository identifier; weight bytes are not captured",
    )
    lora_configuration = eqty_sdk.Configuration.from_object(
        {
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": LORA_TARGETS,
        },
        name="LoRA adapter configuration",
    )
    configured_model = eqty_sdk.Model.from_object(
        {
            "base_model": args.model,
            "adapter": "LoRA",
            "target_modules": LORA_TARGETS,
            "torch_dtype": "bfloat16",
            "device_map": "cuda",
        },
        name="Configured LoRA training model",
    )
    (
        eqty_sdk.Computation.new(
            name="Configure LoRA model",
            description="Load the base model and attach the trainable LoRA adapter.",
            computation_type="model_call",
        )
        .add_input_cid(
            [main_source.cid, base_model_reference.cid, lora_configuration.cid]
        )
        .add_output_cid(configured_model.cid)
        .set_computation_cid(main_source.cid)
        .finalize()
    )

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

    encoded_data = eqty_sdk.Dataset.from_object(
        dataset, name="Prompt-masked examples supplied for training"
    )
    training_configuration = eqty_sdk.Configuration.from_object(
        {
            **train_config,
            "max_sequence_length": MAX_SEQ_LEN,
            "gradient_checkpointing": False,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.03,
            "bf16": True,
        },
        name="LoRA training configuration",
    )
    adapter_artifact = eqty_sdk.Model.from_path(
        ADAPTER_DIR,
        name="Trained Banking77 LoRA adapter",
        _store=False,
        storage="by-reference",
        storage_reason="model adapter weights are a large external artifact",
        obtain_from="the adapter directory produced by train.py",
    )
    config_artifact = eqty_sdk.Configuration.from_path(
        ADAPTER_DIR / "train_config.json", name="Saved training configuration"
    )
    (
        eqty_sdk.Computation.new(
            name="Train and save LoRA adapter",
            description="Fine-tune the LoRA parameters and write the adapter artifact.",
            computation_type="fit",
            seed=args.seed,
        )
        .add_input_cid(
            [
                main_source.cid,
                encoded_data.cid,
                configured_model.cid,
                training_configuration.cid,
            ]
        )
        .add_output_cid([adapter_artifact.cid, config_artifact.cid])
        .set_computation_cid(main_source.cid)
        .finalize()
    )

    manifest_path = Path(
        os.environ.get(
            "EQTY_MANIFEST_PATH", OUT_DIR / "train.auto.manifest.json"
        )
    )
    cfg.get_default_context().export(manifest_path)

    print(f"\nadapter saved to {ADAPTER_DIR}")
    print("next: python evaluate.py")


if __name__ == "__main__":
    main()
