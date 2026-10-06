# Automatic node selection

Path: `repo/train.py:main()` via `python train.py` with defaults, from loading training inputs through saving the LoRA adapter.

Auto was selected after offering manual selection and receiving no choice.
The existing isolated L1/L2 diagrams and AST-only flow report are the inputs.
The report incorrectly joins prompt helpers and LoraConfig to module-level lines;
source inspection places those inside encode/main. They are covered by those stages.

| CFG boxes | Stage / existing function | Inputs | Outputs | Why |
|---|---|---|---|---|
| D | Load taxonomy / common.load_labels | label file, own source | ordered labels | Ingest |
| E | Load examples / train.read_jsonl | JSONL file, own source | whole row list | Ingest |
| C,F | Select examples / main | whole row list, CLI selection | selected rows | Decide |
| G,H | Load tokenizer / main | model request, resolved tokenizer content | effective tokenizer description | Ingest; records EOS padding branch |
| I–P | Encode / encode | rows, labels, effective tokenizer description, max length, common source | whole encoded dataset | Named transform, includes truncation decisions |
| Q–S | Load base and attach LoRA / main | cached base weight files/config, LoRA config, seed | initial adapter state plus model configuration | Ingest and initialization |
| U–W | Train / main | initial adapter, base/config inputs, encoded rows, actual Trainer arguments, tokenizer, collator source | trained adapter state, Trainer result | Fit |
| Y–AA | Save / main | trained adapter, tokenizer description, CLI args and row count | files returned by save_pretrained, train_config.json | Emit |

Builders belong inline in these existing functions and each includes its function's Code asset. No signatures or return types change. Whole-list Dataset assets avoid automatic list expansion. collate/build_user_turn/render_prompt remain undecorated loop helpers; their code is captured as dependencies. Logging and argument parsing get no separate computations.

Prediction: 8 computations by default, 7 without taxonomy loading; one connected component. Input roots: source files, function/module code, model request, cached base files/config, tokenizer content, CLI selection, seed/LoRA/training configuration. Shape: files → parsed rows/labels → selected rows → encoded dataset → training → saved artifacts, with tokenizer and model branches joining training.

CLI overrides, the Docker entry point and the Kubernetes command use the same instrumentation. No evaluate.py exists, and no evaluation, image build, scheduling, host attestation, or container deployment is instrumented.
