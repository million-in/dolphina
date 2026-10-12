# Dolphine

Mechanistic interpretability tools for residual streams, selected neurons and
attention heads. The current checkpoint is Qwen3.5-0.8B; it is not the project name.
Source lives in `research/`. Run local commands from that directory; the VM uses
`~/qwen-research` for compatibility with the existing installation. Cloud resource
names below are operational identifiers, not the project codename.

See [interpretation notes](research/INTERPRETATION.md) for up/down projections,
residual writes, source-reading limitations, and what diagnostic scores mean.
The root `.gitignore` excludes runtime data, model tensors and environments;
only source, documentation and dependency metadata are published.

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

## Offline tensor viewer

```sh
.venv-inspect/bin/python view_activations.py runs/RUN_ID --layer 23 --token 5 --feature 100
.venv-inspect/bin/python view_activations.py runs/RUN_ID --layer 0 --token all --feature 0,7 --csv values.csv
```

This reads existing prompt captures without loading a model. CSV files are never
overwritten. For laptop use, install PyTorch in a local virtual environment and
copy both `activations.pt` and `result.json`; source alone does not include data.

## Generated-token internals

```sh
.venv-inspect/bin/python record_generation.py --prompt 'What is 2 + 3? Answer briefly.' --blocks 0,3,23 --heads 0,1 --neurons 0,17 --max-new-tokens 8 --verify
.venv-inspect/bin/python view_generation.py runs/generation-RUN_ID --position 5 --tensor layer_0.mlp_neurons --feature 17
.venv-inspect/bin/python view_generation.py runs/generation-RUN_ID --position 5 --tensor layer_3.heads --head 1 --feature 0
```

The recorder uses observational PyTorch hooks, not NNsight, for explicit module
input capture; the NNsight prompt workflow remains available unchanged.
It saves embeddings, selected block inputs/outputs, selected MLP intermediate
neurons and selected gated head vectors before the output projection. MLP values
are `activation(gate_proj(x)) * up_proj(x)`, not residual-stream coordinates.
Full-attention heads and DeltaNet value heads have different meanings; neither
capture is an attention-probability matrix. Arbitrary valid block indices can be
selected; head/neuron IDs must exist in every selected block.

Generation is greedy, FP32, uncached prefix replay: slower than cached decoding
but simpler to inspect. The first step records every prompt token; each later
step records only the newest processed token. All emitted tokens, including the
last one/EOS, are processed. `positions` are absolute zero-based token positions;
the last processed token's logits predict `prediction_position`, not itself.
The final step emits nothing. `result.json` includes token labels, response,
original head/neuron IDs, stop reason, and per-step tensor filenames. Each `.pt`
is a tensor dictionary loaded with `weights_only=True`; embedding/block shapes
are `[batch, captured_tokens, hidden]`, neurons `[batch, captured_tokens, selected_neurons]`,
and heads `[batch, captured_tokens, selected_heads, head_dim]`.

Limits: 256 prompt tokens and 128 generated tokens. No silent truncation,
sampling, gradients, interventions, vision-input tracing, or causal claims.
`--verify` compares all emitted IDs with uninstrumented uncached HF generation;
it does not promise equivalence to every cached/optimized backend.
Capturing activity is evidence of a response, not proof that a circuit caused it;
ablation or patching with controls remains a separate experiment.

Qwen3.5-0.8B supports thinking and non-thinking modes: use `--thinking` to enable
the thinking template. Its model card warns about thinking loops, so token caps
remain necessary; a capped run may not reach a final answer. Generated reasoning
text is not a faithful readout of internal computation.
Source: https://huggingface.co/Qwen/Qwen3.5-0.8B

Run tests on the VM: `.venv-inspect/bin/python -m unittest discover -p 'test_*.py' -v`.
Source is retained locally in Git; run data is ignored and must be backed up
separately if needed after VM deletion. The VM environment is used for runtime
validation, and tested source is copied back locally before topic commits.

## Stop early

```sh
gcloud compute instances stop qwen-research-cpu --project=pures-511400 --zone=us-central1-a
```

Do not upgrade billing if retaining the non-billable Free Trial is required.
The 2-hour timer limits a run, not total project spending. Save important
research outside the VM before the trial expires.
