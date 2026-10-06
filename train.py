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
from importlib.metadata import version
from pathlib import Path

import eqty_sdk as sdk
import torch
from huggingface_hub import try_to_load_from_cache
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from safetensors.torch import save as save_safetensors
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
    source = sdk.Dataset.from_path(path, name="Banking77 training JSONL", _store=True)
    with path.open() as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    run = sdk.Computation.new(name="Load training examples", computation_type="ingest", _store=True)
    run.add_input_cid(sdk.Code.from_object(inspect.getsource(read_jsonl),
                                         name="read_jsonl", _store=True).cid)
    run.add_input_cid(source.cid)
    run.add_output_cid(sdk.Dataset.from_object(rows, name="All training rows", _store=True).cid)
    run.finalize()
    return rows


def encode(tokenizer, rows, labels, max_len):
    """Tokenize into prompt-masked examples (prompt tokens get label -100)."""
    run = sdk.Computation.new(name="Encode prompt-masked training examples",
                              computation_type="transform", _store=True)
    run.add_input_cid(sdk.Code.from_object(inspect.getsource(encode),
                                         name="encode", _store=True).cid)
    run.add_input_cid(sdk.Code.from_path(Path(inspect.getfile(render_prompt)),
                                       name="common.py prompt implementation", _store=True).cid)
    run.add_input_cid(sdk.get_cid_for_bytes(json.dumps(rows).encode("utf-8"), _store=True))
    if labels is not None:
        run.add_input_cid(sdk.get_cid_for_bytes(json.dumps(labels).encode("utf-8"), _store=True))
    run.add_input_cid(sdk.Configuration.from_object({"max_len": max_len},
                                                    name="Sequence length limit", _store=True).cid)
    tokenizer_description = {
        "vocabulary": tokenizer.get_vocab(),
        "backend": (tokenizer.backend_tokenizer.to_str() if tokenizer.is_fast else None),
        "special_tokens": tokenizer.special_tokens_map,
        "chat_template": tokenizer.chat_template,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token": tokenizer.eos_token,
    }
    run.add_input_cid(sdk.get_cid_for_bytes(json.dumps(tokenizer_description).encode("utf-8"), _store=True))
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
    run.add_output_cid(sdk.Dataset.from_object(encoded, name="Prompt-masked token sequences",
                                             _store=True).cid)
    run.finalize()
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
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--no-label-list", action="store_true",
                    help="closed-book: omit the 77 labels from the prompt")
    args = ap.parse_args()

    cfg = sdk.init(default_context=sdk.Context.new("Banking77 LoRA training"),
                   custom_dir=Path(os.environ.get("EQTY_SDK_DIR", ".eqty-training")))
    cfg.set_store_all_blobs(True)
    notary = os.environ.get("EQTY_NOTARY_URL")
    sdk.set_active_signer(
        sdk.Signer.vcomp_notary(url=notary, name="banking77-training-notary", _load_if_exists=True)
        if notary else sdk.Signer.load_or_create(name="banking77-training-local")
    )
    main_code = sdk.Code.from_object(inspect.getsource(main), name="main", _store=True)
    runtime_versions = sdk.Configuration.from_object(
        {package: version(package) for package in
         ("eqty_sdk", "torch", "transformers", "peft", "accelerate", "safetensors", "huggingface_hub")},
        name="Runtime package versions", _store=True)

    set_seed(args.seed)
    labels = None if args.no_label_list else load_labels()
    rows = read_jsonl(DATA_DIR / "train.jsonl")
    selection = sdk.Computation.new(name="Select training examples", computation_type="decide", _store=True)
    selection.add_input_cid(main_code.cid)
    selection.add_input_cid(sdk.get_cid_for_bytes(json.dumps(rows).encode("utf-8"), _store=True))
    selection.add_input_cid(sdk.Configuration.from_object(
        {"limit": args.limit, "no_label_list": args.no_label_list},
        name="Training selection", _store=True).cid)
    if args.limit:
        rows = rows[: args.limit]
    selection.add_output_cid(sdk.Dataset.from_object(rows, name="Selected training rows", _store=True).cid)
    selection.finalize()
    print(f"training rows: {len(rows)}  |  label list in prompt: {labels is not None}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer_description = {
        "vocabulary": tokenizer.get_vocab(),
        "backend": (tokenizer.backend_tokenizer.to_str() if tokenizer.is_fast else None),
        "special_tokens": tokenizer.special_tokens_map,
        "chat_template": tokenizer.chat_template,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token": tokenizer.eos_token,
    }
    tokenizer_asset = sdk.Configuration.from_object(tokenizer_description,
                                                    name="Effective tokenizer", _store=True)
    tokenizer_load = sdk.Computation.new(name="Load and configure tokenizer",
                                         computation_type="ingest", _store=True)
    tokenizer_load.add_input_cid(main_code.cid)
    tokenizer_load.add_input_cid(runtime_versions.cid)
    tokenizer_load.add_input_cid(sdk.Configuration.from_object(
        {"model": args.model, "resolved_commit": tokenizer.init_kwargs.get("_commit_hash")},
        name="Tokenizer source request", _store=True).cid)
    tokenizer_load.add_output_cid(tokenizer_asset.cid)
    tokenizer_load.finalize()

    dataset = encode(tokenizer, rows, labels, MAX_SEQ_LEN)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.config.use_cache = False
    # Hash cached files, not the model ID. Resolve only the revision that loaded.
    base_revision = getattr(model.config, "_commit_hash", None)
    base_inputs = [sdk.Configuration.from_object(
        json.loads(model.config.to_json_string()), name="Loaded base model configuration", _store=True).cid]
    base_files = []
    for filename in ("model.safetensors.index.json", "model.safetensors",
                     "pytorch_model.bin.index.json", "pytorch_model.bin"):
        if Path(args.model).is_dir():
            cached = Path(args.model) / filename
            cached = str(cached) if cached.is_file() else None
        elif base_revision:
            cached = try_to_load_from_cache(args.model, filename, revision=base_revision)
        else:
            cached = None
        if isinstance(cached, str):
            if filename.endswith(".index.json"):
                base_inputs.append(sdk.Configuration.from_path(
                    cached, name="Base weight shard index", _store=True).cid)
                for shard in sorted(set(json.loads(Path(cached).read_text())["weight_map"].values())):
                    shard_path = (str(Path(args.model) / shard) if Path(args.model).is_dir()
                                  else try_to_load_from_cache(args.model, shard, revision=base_revision))
                    if isinstance(shard_path, str) and Path(shard_path).is_file():
                        base_files.append(Path(shard_path))
                    else:
                        raise RuntimeError(f"Loaded base weight shard unavailable for EQTY hashing: {shard}")
            else:
                base_files.append(Path(cached))
            break
    for base_path in base_files:
        base_inputs.append(sdk.Model.from_path(
            base_path, name=f"Base weights {base_path.name}", _store=False,
            storage="by-reference", storage_reason="base model weights are multi-GB artifacts",
            obtain_from=f"{args.model}@{base_revision or 'local'}").cid)
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

    lora_asset = sdk.Configuration.from_object(
        json.loads(json.dumps(lora_config.to_dict(), default=sorted)), name="LoRA configuration", _store=True)
    initial_adapter = sdk.Model.from_cid(
        sdk.get_cid_for_bytes(save_safetensors(get_peft_model_state_dict(model)), _store=False),
        name="Initialized LoRA adapter", storage="by-reference",
        storage_reason="adapter tensor snapshot is a model weight artifact; bytes are omitted")
    setup = sdk.Computation.new(name="Load base model and initialize LoRA", computation_type="ingest",
                                base_weight_files_captured=bool(base_files), _store=True)
    setup.add_input_cid(main_code.cid)
    setup.add_input_cid(runtime_versions.cid)
    setup.add_input_cid(base_inputs)
    setup.add_input_cid(lora_asset.cid)
    setup.add_input_cid(sdk.Configuration.from_object(
        {"model": args.model, "revision": base_revision, "seed": args.seed},
        name="Base model request and initialization seed", _store=True).cid)
    setup.add_output_cid(initial_adapter.cid)
    setup.finalize()

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
            tf32=True,
            report_to=[],
            seed=args.seed,
            # Batch similar-length examples together so collate() pads less.
            group_by_length=True,
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
        ),
        train_dataset=dataset,
        data_collator=lambda b: collate(b, tokenizer.pad_token_id),
    )
    training = sdk.Computation.new(name="Train LoRA adapter", computation_type="fit", _store=True)
    training.add_input_cid(main_code.cid)
    training.add_input_cid(runtime_versions.cid)
    training.add_input_cid(sdk.Code.from_object(inspect.getsource(collate), name="collate", _store=True).cid)
    training.add_input_cid(initial_adapter.cid)
    training.add_input_cid(base_inputs)
    training.add_input_cid(lora_asset.cid)
    training.add_input_cid(tokenizer_asset.cid)
    training.add_input_cid(sdk.get_cid_for_bytes(json.dumps(dataset).encode("utf-8"), _store=True))
    training.add_input_cid(sdk.Configuration.from_object(
        json.loads(trainer.args.to_json_string()), name="Actual Trainer arguments", _store=True).cid)
    train_result = trainer.train()
    trained_adapter = sdk.Model.from_cid(
        sdk.get_cid_for_bytes(save_safetensors(get_peft_model_state_dict(model)), _store=False),
        name="Trained LoRA adapter", storage="by-reference",
        storage_reason="adapter tensor snapshot is a model weight artifact; bytes are omitted",
        obtain_from="out/adapter (serialization may differ from the in-memory snapshot)")
    training.add_output_cid(trained_adapter.cid)
    training.add_output_cid(sdk.BenchmarkResult.from_object(
        {"global_step": train_result.global_step, "training_loss": train_result.training_loss,
         "metrics": train_result.metrics}, name="Training result", _store=True).cid)
    training.finalize()

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    emission = sdk.Computation.new(name="Save adapter, tokenizer and run settings",
                                   computation_type="emit", _store=True)
    emission.add_input_cid(main_code.cid)
    emission.add_input_cid(trained_adapter.cid)
    emission.add_input_cid(tokenizer_asset.cid)
    model.save_pretrained(str(ADAPTER_DIR))
    tokenizer_files = tokenizer.save_pretrained(str(ADAPTER_DIR))
    train_config = {**vars(args), "n_rows": len(rows)}
    (ADAPTER_DIR / "train_config.json").write_text(
        json.dumps(train_config, indent=2)
    )
    emission.add_input_cid(sdk.Configuration.from_object(
        train_config, name="Run arguments and row count", _store=True).cid)
    # Include files from this save only; do not attest stale files in the directory.
    # PEFT's unchanged default save uses safe_serialization=True.
    for filename in ("adapter_model.safetensors", "adapter_config.json"):
        artifact_path = ADAPTER_DIR / filename
        if artifact_path.is_file():
            is_weights = filename != "adapter_config.json"
            if is_weights:
                asset = sdk.Model.from_path(
                    artifact_path, name=f"Saved {filename}", _store=False,
                    storage="by-reference", storage_reason="saved LoRA adapter is a model weight artifact",
                    obtain_from=f"out/adapter/{filename}")
            else:
                asset = sdk.Configuration.from_path(artifact_path, name=filename, _store=True)
            emission.add_output_cid(asset.cid)
    for filename in tokenizer_files:
        if filename is not None and Path(filename).is_file():
            emission.add_output_cid(sdk.Configuration.from_path(
                filename, name=f"Saved tokenizer {Path(filename).name}", _store=True).cid)
    emission.add_output_cid(sdk.Configuration.from_path(
        ADAPTER_DIR / "train_config.json", name="train_config.json", _store=True).cid)
    emission.finalize()
    manifest_path = OUT_DIR / "training.manifest.json"
    cfg.get_default_context().export(manifest_path)
    print(f"EQTY manifest saved to {manifest_path}")

    print(f"\nadapter saved to {ADAPTER_DIR}")
    print("next: python evaluate.py")


if __name__ == "__main__":
    main()
