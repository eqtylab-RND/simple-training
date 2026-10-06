# simple-training

Minimal LoRA fine-tune of `nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1` on the
banking77 intent set, on one H100 inside a Kata GPU guest.

> On B200 rewire the kernel parameters in the `training.yaml` with something like `cgroup_no_v1=all pci=realloc,nocrs,assign-busses nvrc.log=trace agent.secure_storage_integrity=false`

```bash
# build and push to ACR (login first)
docker build --platform linux/amd64 -t nvidiademoacr.azurecr.io/eqtylab/banking77-training:latest .
docker push nvidiademoacr.azurecr.io/eqtylab/banking77-training:latest

# launch training
sudo k3s kubectl apply -f training.yaml
sudo k3s kubectl get pod banking77-train -w

# watch until "adapter saved ... / TRAIN_DONE"
sudo k3s kubectl logs -f banking77-train

# copy the tensors and clean up
sudo k3s kubectl cp banking77-train:/app/out/adapter ./adapter
sudo k3s kubectl delete pod banking77-train --grace-period=30 --wait
```
