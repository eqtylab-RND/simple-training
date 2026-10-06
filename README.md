# simple-training

Minimal LoRA fine-tune of `nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1` on the
banking77 intent set, on one H100 inside a Kata confidential (TDX) GPU guest.

```bash
# build and push to ACR (make sure you are logged in)
BUILDX_NO_DEFAULT_ATTESTATIONS=1 docker buildx build --platform linux/amd64 --load -t nvidiademoacr.azurecr.io/eqtylab/banking77-training:latest .
docker push nvidiademoacr.azurecr.io/eqtylab/banking77-training:latest

# launch training
sudo k3s kubectl apply -f training.yaml
sudo k3s kubectl get pod banking77-train -w

# watch until "adapter saved ... / TRAIN_DONE"
sudo k3s kubectl logs -f banking77-train

# copy the adapter out, then clean up
sudo k3s kubectl cp banking77-train:/app/out ./out
sudo k3s kubectl delete pod banking77-train --grace-period=30 --wait
```
