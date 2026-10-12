# Qwen CPU interpretability baseline

Project: `pures-511400`. VM: `qwen-research-cpu`, zone: `us-central1-a`.
Machine: `e2-standard-8` (8 vCPUs, 32 GiB RAM), 50 GB balanced boot disk.
The VM was created with a 7,200-second maximum run duration and STOP action.
No GPU, billing upgrade, service account, notebook server, or managed group
is needed. Compute is estimated at $0.268/hour, plus disk and external IPv4;
the persistent disk remains billable against eligible credit while stopped.

## Connect and rerun

```sh
gcloud compute ssh macbook@qwen-research-cpu --project=pures-511400 --zone=us-central1-a
cd ~/qwen-research
.venv/bin/python smoke_test.py
```

Browser SSH may log in as `kezamilindi` instead. The installation is on the VM
under `/home/macbook/qwen-research`, not on the laptop. Use `sudo -iu macbook`
first in that case.

## Custom prompts and NNsight

The separate `.venv-inspect` environment preserves the baseline `.venv`.
It uses NNsight 0.7.0, PyTorch 2.14.1+cpu and Transformers 5.19.0.
Resolved packages are recorded in `requirements.inspect.lock.txt` on the VM.
To rebuild, create the venv, install the CPU torch wheel from
`https://download.pytorch.org/whl/cpu`, then install that lock file.

Flash Linear Attention 0.5.2 is installed separately in `.venv-fla` using its
`[cpu]` extra; import and `pip check` passed. Its Triton kernels do not accelerate
this CPU machine (import warns that Triton is unsupported on this platform).
It is intentionally not enabled in the research environment, which keeps the
reference PyTorch kernels. `requirements.fla.lock.txt` records its packages.

```sh
cd /home/macbook/qwen-research
.venv-inspect/bin/python inspect_model.py --prompt 'What is 2 + 3?' --layers 0,2 --max-new-tokens 32
.venv-inspect/bin/python inspect_model.py --prompt 'Explain gravity briefly.' --system 'Be concise.' --layers all
```

Layer indices are zero-based. `--layers all` captures every language decoder
block's output, not every internal operation or vision-layer activation.
NNsight captures the full `[batch, prompt_token, hidden_feature]` tensors during
one uncached prompt forward pass; generated-token activations are not captured.
Each run creates a unique directory under `runs/`, containing `activations.pt`
and `result.json` with token labels, generation options, response and statistics.
`--print-values` prints all captured values (potentially very large).
`--temperature 0` uses greedy generation; positive values enable sampling.
`--seed` defaults to 0, and `--thinking` enables the thinking chat template.
CPU limits are 512 input tokens and up to 512 generated tokens; no silent truncation.

Inspect a saved run interactively (replace the example directory):

```python
import torch
a = torch.load('runs/RUN_ID/activations.pt', weights_only=True)
print(a.keys())
print(a['layer_0'][0, -1, :])  # all features at final prompt token
```

These are FP32, unquantized, evaluation-mode inference captures without
gradients. An activation value alone does not establish causal importance.

`setup.sh` installs CPU PyTorch and Transformers into a virtual environment.
It records resolved package versions in `requirements.lock.txt`. The first
smoke test pins the Hugging Face commit in `model_revision.txt`; subsequent
runs use that commit. The full multimodal checkpoint is loaded, with text-only
inputs in this initial test. There is no quantization or weight training.

All floating-point parameters are loaded in FP32. Upcasting the published
checkpoint does not recover precision absent in the original weights.
Eager attention, evaluation mode, seed 0 and greedy decoding provide a simple
baseline, not a cross-platform reproducibility guarantee. Thinking is disabled
for the short arithmetic prompt; this is a smoke test, not a quality benchmark.

The test checks finite logits, identical logits across two uncached forward
passes, and finite activations captured from the first decoder layer with a
PyTorch forward hook. Results go to `smoke_test_result.json`. Gradient-based
experiments must run outside `torch.no_grad()` and need additional testing.
Qwen3.5 uses hybrid attention; standard transformer interpretability tools
may need architecture-specific adaptations.

## Stop early

```sh
gcloud compute instances stop qwen-research-cpu --project=pures-511400 --zone=us-central1-a
```

Do not upgrade billing if retaining the non-billable Free Trial is required.
The 2-hour timer limits a run, not total project spending. Save important
research outside the VM before the trial expires.
