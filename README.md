# simple-training

Minimal LoRA fine-tune of `nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1` on the
banking77 intent set, on one GPU inside a Kata Confidential (TDX) GPU guest.

## A. Setting up EQTY tools

### A.1 Get access to training environment

Get `ACR_TOKEN` from EQTY. Note that the token expires in 4 hours.

Allow list your IP address to push to docker repository at `nvidiademoacr.azurecr.io`.

### A.2 Install EQTY Skills in Codex

```bash
codex plugin marketplace add eqtylab/eqty-skills
codex plugin add eqty-skills@eqty-lab
```

### A.3 Install docker

Ensure that you have docker installed locally, so we can build the training harness docker image.

### A.4 Clone this repository

```bash
git clone git@github.com:eqtylab-RND/simple-training.git
cd simple-training
```

## B. Preparing the training harness image

### B.1 Prompt Codex

>Instrument this repo with EQTY SDK. Do not embed data blobs larger than 1MB in the manifest. Do not look at other parent folders.

Install dependencies as needed and if you are asked to choose between these `Auto` or `HITL` modes for the instrumentation, choose `Auto`.

Inspect the instrumentation with `git diff`.

### B.2 Build the training harness docker image

Make sure you have `docker` installed locally. Run this from your `simply-training` folder:

```bash
BUILDX_NO_DEFAULT_ATTESTATIONS=1 docker buildx build --platform linux/amd64 --load -t nvidiademoacr.azurecr.io/eqtylab/banking77-training:latest .
```

### B.3 Push the image to a container repository

Build and push docker image to ACR in the training environment.

```bash
export ACR_TOKEN="{{ ACR_TOKEN }}"
echo "$ACR_TOKEN" | docker login nvidiademoacr.azurecr.io -u 00000000-0000-0000-0000-000000000000 --password-stdin
docker push nvidiademoacr.azurecr.io/eqtylab/banking77-training:latest
```

## C. Launching the training workload

### C.1 Copy training.yaml to the training environment

```bash
scp training.yaml sysadmin@10.96.8.68:~/training.yaml
```

### C.2 Start the training

Connect to the training server:

```bash
ssh sysadmin@10.96.8.68
```

Run `kubectl` to start the workload:

```bash
sudo kubectl apply -f training.yaml
```

### C.3 Monitor the training working in the Verifiable PodVM

```bash
sudo watch 'kubectl get pod banking77-train ; echo "---" ; kubectl describe pod banking77-train | tail ; echo "---" ; kubectl logs banking77-train | tail'
```

You should see an output like this:

```bash
Every 2.0s: k3s kubectl get pod banking77-train ; echo "---" ; kubectl describe pod banking77-train | tail ; echo "---" ; kubectl logs banking77-train     system: Tue Oct  6 17:20:17 2026

NAME              READY   STATUS              RESTARTS   AGE
banking77-train   0/1     ContainerCreating   0          3m9s
---
                             nvidia.com/cc.ready.state=true
Tolerations:                 node.kubernetes.io/not-ready:NoExecute op=Exists for 300s
                             node.kubernetes.io/unreachable:NoExecute op=Exists for 300s
Events:
  Type    Reason     Age    From               Message
  ----    ------     ----   ----               -------
  Normal  Scheduled  3m9s   default-scheduler  Successfully assigned default/banking77-train to system
  Normal  Pulling    2m38s  kubelet            spec.containers{app}: Pulling image "nvidiademoacr.azurecr.io/eqtylab/banking77-training:latest"
  Normal  Pulled     2m32s  kubelet            spec.containers{app}: Successfully pulled image "nvidiademoacr.azurecr.io/eqtylab/banking77-training:latest" in 6.309s (6.309s including wai
ting). Image size: 4289705986 bytes.
  Normal  Created    2m29s  kubelet            spec.containers{app}: Container created
---
Error from server (BadRequest): container "app" in pod "banking77-train" is waiting to start: ContainerCreating
```

Note that the initially, the bottom will show an error until the `banking77-train` workload is in `Running` status, and starts producing logs.

Follow the logs and watch for completion with `TRAIN_DONE`.

### C.4 Clean-up the workload

After completion:

```bash
sudo kubectl delete -f banking77-train --wait
```

### C.5 Download integrity manifest

For this demo, we will download the PodVM output to the training server, then locate the manifest:

```bash
sudo kubectl cp banking77-train:/app/out ./out
ls
```

You should see an output like this:

```bash
adapter  banking77-training.auto.manifest.json  checkpoints
```

Exit the SSH connection to the training environment, then download the `banking77-training.auto.manifest.json` to your local:

```bash
scp sysadmin@10.96.8.68:~/out/banking77-training.auto.manifest.json .
```

Inspect the file in the [Explorer](https://explorer.preview.eqtylab.io).
