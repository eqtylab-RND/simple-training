# simple-training

Minimal LoRA fine-tune of `nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1` on the
banking77 intent set, on one H100 inside a Kata confidential (TDX) GPU guest.

```bash
# build and push to ACR (make sure you are logged in)
BUILDX_NO_DEFAULT_ATTESTATIONS=1 docker buildx build --platform linux/amd64 --load -t nvidiademoacr.azurecr.io/eqtylab/banking77-training:latest .
docker push nvidiademoacr.azurecr.io/eqtylab/banking77-training:latest

# launch training (inject the real SAS into the __AZURE_BLOB_SAS_URL__ placeholder)
sudo k3s kubectl apply -f training.yaml
sudo k3s kubectl get pod banking77-train -w

# watch until "adapter saved ... / TRAIN_DONE"
sudo k3s kubectl logs -f banking77-train

# copy the adapter out, then clean up
sudo k3s kubectl cp banking77-train:/app/out ./out
sudo k3s kubectl delete pod banking77-train --grace-period=30 --wait
```

Training now exports signed EQTY lineage to `out/training.manifest.json` after
saving the adapter. The existing output-copy command retrieves it with the
adapter. `python train.py` records eight computations; `--no-label-list` records
seven. No training or deployment was run while preparing this patch.

`eqty_sdk==2.4.2` is a required dependency. Set `EQTY_NOTARY_URL` in the training
environment to use a VComp notary; a configured notary failure stops the run.
Otherwise the SDK creates or reuses a local signing key. SDK storage defaults
to `.eqty-training/` (override with `EQTY_SDK_DIR`); this directory holds private
keys and is excluded from Git and the Docker build context.

The manifest includes source, training data, tokenizer configuration and actual
training settings. Weight artifacts are committed by hash with explicit
by-reference declarations. See the
[prediction, limitations and review checklist](eqty-instrumentation/simple-training/simple-training.auto.prediction.md)
for expected structure and subsequent verification commands. The separate
`eqty-manifest` skill can inspect and verify a manifest after deployment.
