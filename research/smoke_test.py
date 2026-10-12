"""Unquantized FP32 CPU baseline and activation-access smoke test."""
import json
import platform
import time
from pathlib import Path

import torch
import transformers
from huggingface_hub import model_info
from transformers import AutoModelForImageTextToText, AutoTokenizer

MODEL = "Qwen/Qwen3.5-0.8B"
torch.manual_seed(0)
torch.set_num_threads(8)
torch.use_deterministic_algorithms(True)
revision_file = Path("model_revision.txt")
if not revision_file.exists():
    revision_file.write_text(model_info(MODEL).sha + "\n")
revision = revision_file.read_text().strip()
tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=revision)
started = time.monotonic()
model = AutoModelForImageTextToText.from_pretrained(
    MODEL, revision=revision, torch_dtype=torch.float32,
    attn_implementation="eager", trust_remote_code=False,
)
model.eval()
assert all(p.dtype == torch.float32 for p in model.parameters() if p.is_floating_point())
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "What is 2 + 2? Answer briefly."}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
inputs = tokenizer(prompt, return_tensors="pt")
activation = {}


def capture(_module, _args, output):
    tensor = output[0] if isinstance(output, tuple) else output
    activation.update(shape=list(tensor.shape), dtype=str(tensor.dtype),
                      finite=bool(torch.isfinite(tensor).all()))


layer_name, layer = next(
    (name, module) for name, module in model.named_modules()
    if name.endswith("layers.0")
)
handle = layer.register_forward_hook(capture)
with torch.no_grad():
    first = model(**inputs, use_cache=False).logits[:, -1, :].clone()
handle.remove()
with torch.no_grad():
    second = model(**inputs, use_cache=False).logits[:, -1, :]
    generated = model.generate(**inputs, max_new_tokens=32, do_sample=False)
assert torch.isfinite(first).all(), "Nonfinite logits"
assert activation["finite"], "Nonfinite activation"
assert torch.equal(first, second), "Repeated baseline logits differ"
report = {
    "model": MODEL, "revision": revision, "python": platform.python_version(),
    "torch": torch.__version__, "transformers": transformers.__version__,
    "device": str(next(model.parameters()).device), "dtype": "float32",
    "parameters": sum(p.numel() for p in model.parameters()),
    "hook_module": layer_name, "activation": activation,
    "repeat_logits_identical": True, "max_new_tokens": 32,
    "response": tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True),
    "elapsed_seconds": round(time.monotonic() - started, 2),
}
Path("smoke_test_result.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
