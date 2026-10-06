# Instrumentation prediction and review

Auto selection followed an unanswered offer of manual selection. The source-only
CFG and AST report already existed in this repo; no parent folders were inspected.
No training, imports of target modules, tests, model downloads, or deployments ran.
Static parsing and patch review are not runtime validation.

Command: `python train.py` (requires the existing GPU/model environment).
Manifest: `out/training.manifest.json`, exactly one unmodified SDK export after a
successful save. The existing Kubernetes command uses the same path and writes
under `/app/out`; its existing output-copy command includes the manifest.
SDK: `eqty_sdk==2.4.2`, checked against the published source distribution without
installing/importing it. Existing ML dependencies keep their version ranges;
the runtime records their actual package versions.

Expected computations, in order:
1. Load taxonomy (omitted with `--no-label-list`).
2. Load training examples.
3. Select training examples.
4. Load and configure tokenizer.
5. Encode prompt-masked training examples.
6. Load base model and initialize LoRA.
7. Train LoRA adapter.
8. Save adapter, tokenizer and run settings.

Default: **8 computations, 1 connected component**. With `--no-label-list`:
**7 computations, 1 component**. No node per row, batch or epoch. Data-only
connectivity also joins rows → selection → encoding → training → saving, with
labels, tokenizer and model branches. Shared Code inputs must not be mistaken
for sufficient evidence of data flow.

Root categories: original JSONL/taxonomy files, six Code assets (five functions
and common.py), sequence length, selection settings, tokenizer source request,
base configuration, base weight files and optional shard index, initialization
seed/model request, LoRA settings, Trainer arguments, saved run arguments/row
count, and runtime versions. On the default path with distinct content, expect
17 + W + I roots, where W is the number of captured base weight files and I=1
for a shard index, else 0. Closed-book drops the taxonomy file and its Code root.
Content equality may collapse roots; this is a category prediction, not a
cryptographic uniqueness assertion.

Signer: `EQTY_NOTARY_URL` selects VComp signing under `banking77-training-notary`;
otherwise `banking77-training-local` is loaded/created. A configured notary error
propagates; no local fallback. `EQTY_SDK_DIR` defaults to `.eqty-training/`, ignored
by Git and excluded from the Docker build. It contains private signer keys and
must not be shared with the manifest. Init happens once in main before assets.

## Assumptions and reported gaps

- Existing flow types are AST/catalogue-only. Library behavior is read statically,
  not exercised. JSON rows/labels are plain lists as the actual source shows.
- Loaded base files are resolved only from the loaded model's commit or local
  directory. Standard safetensors/bin single-file and indexed shards are covered.
  Custom loaders, alternate filenames or an unavailable commit can leave no
  base files captured; `base_weight_files_captured=false` reports that gap.
  Hashing a model request is not a commitment to missing weight bytes.
- Base weights, initial adapter snapshot, trained adapter snapshot and saved
  adapter weights are **by reference** with signed storage reasons. Their bytes
  cannot be checked from this manifest alone. The initial adapter snapshot is
  not persisted separately. Retrieving out/adapter does not necessarily reconstruct
  either in-memory safetensors snapshot byte-for-byte, because save_pretrained
  may include different metadata. This is an explicit availability gap.
- Tokenizer description retains fast backend JSON, vocabulary, special tokens,
  chat template and effective padding/EOS. A slow tokenizer's vocabulary is an
  incomplete description of its normalization algorithm; raw tokenizer source
  files are not captured before encoding. Saved tokenizer files are committed
  only at emission. External library implementation code is represented by
  package versions, not a full executable/dependency closure.
- Prompt helpers/constants are captured via common.py; collate's own source is
  a training input. No row/batch helper computations are introduced.
- PEFT's existing default save uses safe_serialization=True and the standard
  adapter_model.safetensors/adapter_config.json names. Tokenizer files are taken
  from its save_pretrained return, ignoring nonexistent optional paths. Other
  generated files (e.g. model card), stale files, and external side effects are
  outside the emission claim. Nonstandard PEFT layouts are not covered.
- Selection with no limit re-emits the same row CID. Multiple producers and a
  content-level self-cycle are expected for that identity selection; do not
  misread it as new data. Code/source names are explicit; no generated Custom
  asset names or path-text substitutes are intentionally used.
- CLI values and actual Trainer arguments preserve their original paths. Their
  blobs can contain absolute output/cache/local model paths; review decoded
  blobs before sharing. Signing does not itself establish TDX/GPU attestation.
- A failed stage is not finalized, and no failure manifest is exported. A prior
  successful manifest can remain at the same destination after a failed rerun.
- Recording adds hashing, CPU serialization and SDK/signing overhead, including
  multi-GB base-file reads. It can fail on unavailable files, unsupported object
  content or signer/storage problems. Training logic, settings, signatures and
  return values otherwise stay unchanged.

## Checks on a subsequently produced manifest

Use the separate eqty-manifest skill for interpretation and verification.
With the skill package paths substituted for `<instrument-skill>` and
`<manifest-skill>`:

```bash
uv run <manifest-skill>/summary.py out/training.manifest.json
python3 <instrument-skill>/check_graph.py out/training.manifest.json --expect-computations 8 --expect-components 1
uv run <manifest-skill>/parse_manifest.py out/training.manifest.json timeline
uv run <manifest-skill>/verify_credentials.py out/training.manifest.json
```

Use `--expect-computations 7` for closed-book mode. Check structure first, then
available blob hashes, then signatures. Missing declared by-reference preimages
remain unverified. Authorize the signer/operator separately from signature checks.

## Patch review checklist

- [ ] Original function definitions own every builder, with their own Code input;
      no adapters, decorated stubs, runners or new instrumented modules.
- [ ] Signatures, return values, masking, truncation, batching and training
      configuration preserve the existing behavior.
- [ ] Lists use one whole-list CID, matching producer and consumer serialization.
- [ ] Selected rows, tokenizer, initialized/trained adapter and saved artifacts
      match observed data flow; no synthetic parameters or edges.
- [ ] Assets are built inline, with explicit names/types; no naming helpers.
- [ ] Runtime dependency is pinned; no missing-SDK fallback or custom decorator.
- [ ] Init precedes signing/registration and stays at startup; keys stay untracked.
- [ ] Notary/local signer names differ and notary failures have no fallback.
- [ ] Every computation has computation_type metadata.
- [ ] All omitted heavy preimages have by-reference reasons and are listed above.
- [ ] Other missing assets, tokenizer limitations, path disclosure, alternate
      base/save layouts, repeated-CID cycles and failed-run behavior are reviewed.
- [ ] Verify stage counts, data connectivity, Code blobs, credential subjects,
      expected issuer/operator, content codecs and available preimages after deployment.
- [ ] Export is one file with no repair or context-embedded twin; no stubbed claim.

No commit or deployment was made. GitHub CLI credentials are invalid; the local
review branch and patch are the handoff instead of a pull request.
