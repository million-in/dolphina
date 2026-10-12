"""Custom CPU prompts and prompt-token layer outputs captured with NNsight."""
import argparse
import json
from pathlib import Path
import time

import torch
import nnsight
from nnsight import NNsight
from transformers import AutoModelForImageTextToText, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--system', default=None)
    parser.add_argument('--layers', default='0', help='Comma-separated zero-based indices, or all')
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--temperature', type=float, default=0, help='0 = greedy; positive = sampling')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--thinking', action='store_true')
    parser.add_argument('--print-values', action='store_true', help='Print EVERY captured tensor value (can be huge)')
    parser.add_argument('--output', type=Path, default=None, help='New directory; refuses overwrite')
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 512 or not 0 <= args.temperature <= 5:
        parser.error('max-new-tokens must be 1..512; temperature must be 0..5')
    torch.manual_seed(args.seed)
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    root = Path(__file__).resolve().parent
    revision = (root / 'model_revision.txt').read_text().strip()
    model_id = 'Qwen/Qwen3.5-0.8B'
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    messages = []
    if args.system:
        messages.append({'role': 'system', 'content': args.system})
    messages.append({'role': 'user', 'content': args.prompt})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=args.thinking)
    inputs = tokenizer(text, return_tensors='pt')
    if inputs.input_ids.shape[1] > 512:
        parser.error('CPU safety limit: 512 input tokens including chat template; shorten prompt')
    raw = AutoModelForImageTextToText.from_pretrained(
        model_id, revision=revision, dtype=torch.float32,
        attn_implementation='eager', trust_remote_code=False,
    ).eval()
    assert all(p.dtype == torch.float32 for p in raw.parameters() if p.is_floating_point())
    count = len(raw.model.language_model.layers)
    try:
        indices = list(range(count)) if args.layers == 'all' else sorted(set(map(int, args.layers.split(','))))
    except ValueError:
        parser.error('--layers must be all or comma-separated integers')
    if not indices or any(i < 0 or i >= count for i in indices):
        parser.error(f'layer indices must be 0..{count - 1}')
    output = args.output or root / 'runs' / str(time.time_ns())
    output.mkdir(parents=True, exist_ok=False)
    wrapped = NNsight(raw)
    with torch.no_grad():
        with wrapped.trace(**inputs, use_cache=False):
            captured = nnsight.save([wrapped.model.language_model.layers[i].output for i in indices])
        generation = {'max_new_tokens': args.max_new_tokens, 'do_sample': args.temperature > 0}
        if args.temperature > 0:
            generation['temperature'] = args.temperature
        generated = raw.generate(**inputs, **generation)
    tensors = {}
    summaries = {}
    for index, value in zip(indices, captured):
        tensor = (value[0] if isinstance(value, tuple) else value).detach().cpu().clone()
        assert torch.isfinite(tensor).all(), f'Nonfinite activation at layer {index}'
        name = f'layer_{index}'
        tensors[name] = tensor
        summaries[name] = {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                           'min': tensor.min().item(), 'max': tensor.max().item(),
                           'mean': tensor.mean().item(), 'std': tensor.std().item()}
    torch.save(tensors, output / 'activations.pt')
    result = {'model': model_id, 'revision': revision, 'device': 'cpu', 'dtype': 'float32',
              'torch': torch.__version__, 'nnsight': getattr(nnsight, '__version__', 'unknown'),
              'messages': messages, 'seed': args.seed, 'temperature': args.temperature,
              'thinking': args.thinking, 'max_new_tokens': args.max_new_tokens,
              'capture_scope': 'decoder block outputs on prompt tokens only, before generation',
              'token_ids': inputs.input_ids[0].tolist(),
              'tokens': tokenizer.convert_ids_to_tokens(inputs.input_ids[0].tolist()),
              'response': tokenizer.decode(generated[0, inputs.input_ids.shape[1]:], skip_special_tokens=True),
              'activations': summaries, 'output_directory': str(output)}
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if args.print_values:
        torch.set_printoptions(profile='full')
        for name, tensor in tensors.items():
            print(name, tensor)


if __name__ == '__main__':
    main()
