"""CPU FP32 generation with selected internal activations and explicit token alignment."""
import argparse
import json
from pathlib import Path
import time

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer


def indices(text, size):
    values = sorted(set(int(x) for x in text.split(',')))
    if not values or any(x < 0 or x >= size for x in values):
        raise ValueError(f'indices must be in 0..{size - 1}')
    return values


class Recorder:
    """Capture only requested coordinates; never modify module inputs/outputs."""
    def __init__(self, model, blocks, heads, neurons):
        self.handles = []
        self.values = {}
        self.start = 0
        self.metadata = {}
        lm = model.model.language_model
        specs = [('embeddings', lm.embed_tokens, False, None, None)]
        for i in blocks:
            layer = lm.layers[i]
            mlp_ids = indices(neurons, layer.mlp.down_proj.in_features)
            linear = hasattr(layer, 'linear_attn')
            attn = layer.linear_attn if linear else layer.self_attn
            projection = attn.out_proj if linear else attn.o_proj
            dim = attn.head_v_dim if linear else attn.head_dim
            head_ids = indices(heads, projection.in_features // dim)
            prefix = f'layer_{i}'
            self.metadata[prefix] = {
                'kind': 'linear_attention' if linear else 'full_attention',
                'head_indices': head_ids, 'head_dim': dim, 'neuron_indices': mlp_ids,
                'head_capture': 'gated head vectors before output projection; not attention probabilities',
                'neuron_capture': 'activation(gate_proj(x)) * up_proj(x), before down_proj',
            }
            specs.extend([
                (prefix + '.input', layer, True, None, None),
                (prefix + '.output', layer, False, None, None),
                (prefix + '.mlp_neurons', layer.mlp.down_proj, True, mlp_ids, None),
                (prefix + '.heads', projection, True, head_ids, dim),
            ])
        for name, module, before, selected, dim in specs:
            def capture(value, name=name, selected=selected, dim=dim):
                tensor = value[0] if isinstance(value, tuple) else value
                tensor = tensor[:, self.start:, :]
                if dim is not None:
                    tensor = tensor.reshape(*tensor.shape[:-1], -1, dim)[:, :, selected, :]
                elif selected is not None:
                    tensor = tensor[:, :, selected]
                tensor = tensor.detach().cpu().clone()
                if not torch.isfinite(tensor).all():
                    raise ValueError(f'Nonfinite capture: {name}')
                self.values[name] = tensor
            if before:
                def pre(_module, args, kwargs, capture=capture):
                    capture(args[0] if args else kwargs['hidden_states'])
                self.handles.append(module.register_forward_pre_hook(pre, with_kwargs=True))
            else:
                def post(_module, _args, output, capture=capture):
                    capture(output)
                self.handles.append(module.register_forward_hook(post))

    def close(self):
        for handle in self.handles:
            handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--system')
    parser.add_argument('--blocks', default='0,3')
    parser.add_argument('--heads', default='0', help='Original head indices, valid in each selected block')
    parser.add_argument('--neurons', default='0', help='Original MLP intermediate indices')
    parser.add_argument('--max-new-tokens', type=int, default=32)
    parser.add_argument('--thinking', action='store_true')
    parser.add_argument('--verify', action='store_true', help='Compare IDs against uninstrumented uncached HF generate')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 128:
        parser.error('max-new-tokens must be 1..128')
    torch.set_num_threads(8)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    root = Path(__file__).resolve().parent
    revision = (root / 'model_revision.txt').read_text().strip()
    model_id = 'Qwen/Qwen3.5-0.8B'
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    messages = ([{'role': 'system', 'content': args.system}] if args.system else [])
    messages.append({'role': 'user', 'content': args.prompt})
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=args.thinking)
    inputs = tokenizer(prompt, return_tensors='pt')
    length = inputs.input_ids.shape[1]
    if length > 256:
        parser.error('maximum 256 prompt tokens including template')
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, revision=revision, dtype=torch.float32, attn_implementation='eager',
        trust_remote_code=False).eval()
    try:
        recorder = Recorder(model, indices(args.blocks, len(model.model.language_model.layers)),
                            args.heads, args.neurons)
    except ValueError as exc:
        parser.error(str(exc))
    output = args.output or root / 'runs' / ('generation-' + str(time.time_ns()))
    output.mkdir(parents=True, exist_ok=False)
    ids = inputs.input_ids.clone()
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    rows = []
    finished = False
    try:
        with torch.inference_mode():
            for step in range(args.max_new_tokens + 1):
                recorder.start = 0 if step == 0 else ids.shape[1] - 1
                recorder.values = {}
                logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[:, -1]
                if not torch.isfinite(logits).all():
                    raise ValueError('Nonfinite logits')
                next_id = int(logits.argmax(-1).item())
                filename = f'step_{step:04d}.pt'
                torch.save(recorder.values, output / filename)
                emit = not finished and step < args.max_new_tokens
                positions = list(range(recorder.start, ids.shape[1]))
                rows.append({'file': filename, 'positions': positions,
                             'processed_token_ids': ids[0, recorder.start:].tolist(),
                             'prediction_position': ids.shape[1], 'argmax_next_token_id': next_id,
                             'emitted_token_id': next_id if emit else None})
                if not emit:
                    break
                ids = torch.cat([ids, ids.new_tensor([[next_id]])], dim=1)
                finished = next_id in eos
    finally:
        recorder.close()
    verified = None
    if args.verify:
        with torch.inference_mode():
            baseline = model.generate(**inputs, do_sample=False, use_cache=False,
                                      max_new_tokens=args.max_new_tokens)
        verified = torch.equal(ids, baseline)
        if not verified:
            raise AssertionError('Recorded greedy IDs differ from HF generate; captures left for diagnosis')
    token_ids = ids[0].tolist()
    result = {'schema': 'generation-v1', 'model': model_id, 'revision': revision,
              'torch': torch.__version__, 'dtype': 'float32', 'device': 'cpu',
              'messages': messages, 'thinking': args.thinking, 'max_new_tokens': args.max_new_tokens,
              'prompt_length': length, 'token_ids': token_ids,
              'tokens': tokenizer.convert_ids_to_tokens(token_ids),
              'response': tokenizer.decode(token_ids[length:], skip_special_tokens=True),
              'blocks': recorder.metadata, 'steps': rows, 'baseline_ids_equal': verified,
              'method': 'PyTorch hooks; greedy uncached prefix replay; final emitted token also processed',
              'stop_reason': 'eos' if finished else 'max_new_tokens'}
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'response': result['response'], 'output': str(output),
                      'baseline_ids_equal': verified}, indent=2))


if __name__ == '__main__':
    main()
