"""Weight-space superposition geometry for the pinned Qwen3.5 checkpoint.

Feature-direction sets inspected, following the toy-models notebook's ``W``:

- token embedding rows (tied to the unembedding in this checkpoint), each a
  direction in the residual stream;
- per-block MLP neuron write directions: the columns of ``down_proj.weight``
  (PyTorch orientation ``[hidden, intermediate]``). A neuron writes its gated
  scalar times its column; the scalar can flip the column's sign, and the
  metrics here are sign-insensitive by construction.

Tensors are streamed from safetensors slices in row chunks and upcast to
FP32; the full model is never loaded and no inference runs. Upcasting the
published BF16 weights does not recover precision absent in the checkpoint.

With ``--activations`` the model is also run once per prompt of a small
fixed prompt set (BF16 by default so it fits small machines) and per-neuron
activity is measured: the fraction of token positions whose gate
pre-activation is positive (the gated scalar ``m = SiLU(gate) * up`` can
only be nonzero when its gate is), plus mean ``m**2`` and mean ``|m|``.
The gate-active fraction plays the role of the notebook's feature
probability, so probability-weighted net interference is then reported
per MLP layer. Prompt positions only; frequencies from a handful of
prompts describe that prompt set, not a population of text.

These remain diagnostics, not established features: neurons and tokens are
not features, the embedding gets no net interference (a tiny corpus cannot
estimate vocabulary frequencies), and overlap between write directions
alone does not prove the model uses superposed features. See
INTERPRETATION.md.
"""
import argparse
import json
from pathlib import Path
import time

import torch
from safetensors import safe_open

from record_generation import indices
from superposition import (ToyConfig, feature_geometry_streaming,
                           summarize_geometry, toy_instance_geometry,
                           train_toy)

EMBED = 'model.language_model.embed_tokens.weight'
DOWN = 'model.language_model.layers.{}.mlp.down_proj.weight'


def tensor_files(snapshot):
    """Map tensor name to its safetensors file inside a snapshot directory."""
    snapshot = Path(snapshot)
    index = snapshot / 'model.safetensors.index.json'
    mapping = {}
    if index.exists():
        weight_map = json.loads(index.read_text())['weight_map']
        mapping = {name: snapshot / filename for name, filename in weight_map.items()}
    else:
        for path in sorted(snapshot.glob('*.safetensors')):
            with safe_open(path, framework='pt') as handle:
                for name in handle.keys():
                    mapping[name] = path
    if not mapping:
        raise ValueError(f'no safetensors tensors found in {snapshot}')
    return mapping


def _feature_chunks(path, name, transpose, chunk):
    """Fresh iterator of FP32 ``[rows, hidden]`` feature chunks from a slice."""
    with safe_open(path, framework='pt') as handle:
        view = handle.get_slice(name)
        shape = view.get_shape()
        features = shape[1] if transpose else shape[0]
        for start in range(0, features, chunk):
            block = view[:, start:start + chunk] if transpose else view[start:start + chunk]
            yield block.T.contiguous().float() if transpose else block.float()


def _geometry_for(mapping, name, transpose, n_hidden, chunk, probabilities=None):
    if name not in mapping:
        raise ValueError(f'missing tensor {name}')
    path = mapping[name]
    with safe_open(path, framework='pt') as handle:
        shape = handle.get_slice(name).get_shape()
    if len(shape) != 2 or shape[0 if transpose else 1] != n_hidden:
        raise ValueError(f'{name} has shape {shape}, expected hidden size {n_hidden}')
    if probabilities is not None and len(probabilities) != shape[1 if transpose else 0]:
        raise ValueError(f'probabilities for {name} must match its feature count')

    def make_chunks():
        return _feature_chunks(path, name, transpose, chunk)

    geometry = feature_geometry_streaming(make_chunks, n_hidden, probabilities)
    if not all(torch.isfinite(values).all() for values in geometry.values()):
        raise ValueError(f'nonfinite geometry for {name}')
    return geometry


def _with_full(summary, geometry, full):
    if full:
        summary['per_feature'] = {key: values.tolist()
                                  for key, values in geometry.items()}
    return summary


class NeuronActivity:
    """Observe gated MLP scalars through hooks; never retain full activations.

    Pairs each ``gate_proj`` output with the following ``up_proj`` output of
    the same block and accumulates per-neuron counts over flattened token
    positions: gate-active (gate > 0) count, sum ``m**2`` and sum ``|m|``
    for ``m = SiLU(gate) * up``, all in FP32 chunks.
    """

    def __init__(self, mlps, chunk=8192):
        self.chunk = chunk
        self.handles = []
        self.state = {}
        self.pending = {}
        for layer, mlp in mlps.items():
            width = mlp.gate_proj.out_features
            self.state[layer] = {'positions': 0,
                                 'gate_active': torch.zeros(width),
                                 'sum_m2': torch.zeros(width),
                                 'sum_abs_m': torch.zeros(width)}
            self.handles.append(mlp.gate_proj.register_forward_hook(self._gate(layer)))
            self.handles.append(mlp.up_proj.register_forward_hook(self._up(layer)))

    def _gate(self, layer):
        def hook(_module, _args, output):
            self.pending[layer] = output.detach()
        return hook

    def _up(self, layer):
        def hook(_module, _args, output):
            gate = self.pending.pop(layer).reshape(-1, output.shape[-1])
            up = output.detach().reshape(-1, output.shape[-1])
            state = self.state[layer]
            state['positions'] += len(gate)
            for start in range(0, len(gate), self.chunk):
                g = gate[start:start + self.chunk].float()
                m = torch.nn.functional.silu(g) * up[start:start + self.chunk].float()
                state['gate_active'] += (g > 0).float().sum(0)
                state['sum_m2'] += m.square().sum(0)
                state['sum_abs_m'] += m.abs().sum(0)
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()

    def result(self):
        out = {}
        for layer, state in self.state.items():
            positions = state['positions']
            if positions == 0:
                raise ValueError(f'no activations observed for layer {layer}')
            out[layer] = {'positions': positions,
                          'active_fraction': state['gate_active'] / positions,
                          'mean_square_m': state['sum_m2'] / positions,
                          'mean_abs_m': state['sum_abs_m'] / positions}
        return out


def _quantiles(values):
    q = torch.quantile(values.float(), torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]))
    return {name: float(value) for name, value in
            zip(('min', 'p25', 'median', 'p75', 'max'), q)}


def inspect_weights(snapshot, layers=None, chunk=8192, full=False, activity=None):
    """Superposition summaries for embedding rows and MLP write directions."""
    snapshot = Path(snapshot)
    config = json.loads((snapshot / 'config.json').read_text())
    text = config.get('text_config', config)
    n_hidden = text['hidden_size']
    n_layers = text['num_hidden_layers']
    if layers is None:
        layers = list(range(n_layers))
    if not layers or any(not 0 <= layer < n_layers for layer in layers):
        raise ValueError(f'layers must be in 0..{n_layers - 1}')
    mapping = tensor_files(snapshot)
    embed = _geometry_for(mapping, EMBED, False, n_hidden, chunk)
    result = {
        'embed_tokens': _with_full(summarize_geometry(embed, n_hidden), embed, full),
        'mlp_down_proj_columns': {},
        'checkpoint': {'hidden_size': n_hidden,
                         'intermediate_size': text.get('intermediate_size'),
                         'num_hidden_layers': n_layers,
                         'vocab_size': text.get('vocab_size'),
                         'tie_word_embeddings': text.get('tie_word_embeddings'),
                         'stored_dtype': text.get('dtype')},
    }
    result['embed_tokens']['tensor'] = EMBED
    for layer in layers:
        probabilities = None if activity is None else activity[layer]['active_fraction']
        geometry = _geometry_for(mapping, DOWN.format(layer), True, n_hidden, chunk,
                                 probabilities)
        summary = _with_full(summarize_geometry(geometry, n_hidden), geometry, full)
        summary['tensor'] = DOWN.format(layer)
        if activity is not None:
            stats = activity[layer]
            summary['activation'] = {
                'positions': stats['positions'],
                'active_definition': 'fraction of positions with gate pre-activation > 0',
                'active_fraction': _quantiles(stats['active_fraction']),
                'active_fraction_mean': float(stats['active_fraction'].mean()),
                'mean_square_m': _quantiles(stats['mean_square_m']),
                'mean_abs_m': _quantiles(stats['mean_abs_m'])}
            if full:
                summary['per_feature']['active_fraction'] = \
                    stats['active_fraction'].tolist()
        result['mlp_down_proj_columns'][str(layer)] = summary
    return result


DEFAULT_PROMPTS = [
    'Explain how a rainbow forms.',
    'Explain gravity briefly.',
    'What is 2 + 3? Answer briefly.',
    'Write a haiku about the sea.',
    'Translate "good morning" into French, German and Japanese.',
    'List the planets in order from the Sun.',
    'Write a Python function that reverses a string.',
    'Summarize why the sky is blue in one sentence.',
    'What year did humans first land on the Moon?',
    'Give three tips for staying healthy.',
]


def measure_activity(source, revision, prompts, layers, dtype):
    """Prompt-only forwards, one fixed prompt at a time, frequencies per neuron."""
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    kwargs = {} if revision is None else {'revision': revision}
    tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
    model = AutoModelForImageTextToText.from_pretrained(
        source, dtype=dtype, attn_implementation='eager',
        trust_remote_code=False, **kwargs).cpu().eval()
    lm_layers = model.model.language_model.layers
    activity = NeuronActivity({layer: lm_layers[layer].mlp for layer in layers})
    lengths = []
    try:
        with torch.inference_mode():
            for prompt in prompts:
                text = tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': prompt}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
                inputs = tokenizer(text, return_tensors='pt')
                lengths.append(int(inputs.input_ids.shape[1]))
                if lengths[-1] > 256:
                    raise ValueError('maximum 256 prompt tokens including template')
                model(**inputs, use_cache=False)
    finally:
        activity.close()
    return activity.result(), {'prompts': prompts, 'prompt_token_lengths': lengths,
                                'positions_total': sum(lengths),
                                'dtype': str(dtype).replace('torch.', ''),
                                'positions_are': 'prompt tokens only, chat template included'}


def toy_calibration(steps, n_batch, seed):
    """Notebook intro sweep: 5 features, 2 hidden, sparsity across instances."""
    config = ToyConfig(n_features=5, n_hidden=2, n_instances=10)
    model, record = train_toy(
        config,
        feature_probability=(20 ** -torch.linspace(0, 1, config.n_instances))[:, None],
        importance=(0.9 ** torch.arange(config.n_features))[None, :],
        steps=steps, n_batch=n_batch, seed=seed)
    record['config'] = {'n_features': 5, 'n_hidden': 2, 'n_instances': 10,
                          'importance': '0.9**arange(n_features)',
                          'feature_probability': '20**-linspace(0, 1, n_instances)'}
    record['instances'] = toy_instance_geometry(model)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path,
                        help='Local snapshot directory; default downloads the pinned revision')
    parser.add_argument('--layers', default='all',
                        help='Comma-separated block indices or all (MLP sets only)')
    parser.add_argument('--chunk', type=int, default=8192, help='Feature rows per streamed chunk')
    parser.add_argument('--full', action='store_true',
                        help='Include per-feature arrays (large for the embedding)')
    parser.add_argument('--activations', action='store_true',
                        help='Measure per-neuron activation frequencies on prompts and '
                             'report probability-weighted net interference per MLP layer')
    parser.add_argument('--prompts-file', type=Path,
                        help='One prompt per line; default is the built-in prompt set')
    parser.add_argument('--dtype', choices=['bfloat16', 'float32'], default='bfloat16',
                        help='Inference dtype for the activation pass only (weights stay FP32)')
    parser.add_argument('--toy-calibration', action='store_true',
                        help='Also train the notebook intro toy sweep as a calibration')
    parser.add_argument('--toy-steps', type=int, default=4000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.chunk < 1 or args.toy_steps < 1:
        parser.error('chunk and toy-steps must be positive')
    torch.set_num_threads(8)
    root = Path(__file__).resolve().parent
    revision = (root / 'model_revision.txt').read_text().strip()
    model_id = 'Qwen/Qwen3.5-0.8B'
    snapshot = args.model_dir
    if snapshot is None:
        from huggingface_hub import snapshot_download
        snapshot = Path(snapshot_download(
            model_id, revision=revision,
            allow_patterns=['*.json', '*.safetensors']))
    config = json.loads((snapshot / 'config.json').read_text())
    n_layers = config.get('text_config', config)['num_hidden_layers']
    layers = list(range(n_layers)) if args.layers == 'all' else indices(args.layers, n_layers)
    activity, activation_run = None, None
    if args.activations:
        if args.prompts_file:
            prompts = [line for line in
                       (line.strip() for line in args.prompts_file.read_text().splitlines())
                       if line]
            if not prompts:
                parser.error('prompts file has no prompts')
        else:
            prompts = DEFAULT_PROMPTS
        source = str(snapshot) if args.model_dir else model_id
        activity, activation_run = measure_activity(
            source, None if args.model_dir else revision, prompts, layers,
            getattr(torch, args.dtype))
    result = {'schema': 'superposition-v1', 'model': model_id,
              'revision': revision if args.model_dir is None else None,
              'snapshot': str(snapshot), 'torch': torch.__version__,
              'computation_dtype': 'float32 (upcast from stored weights)',
              'layers': layers, 'caveats': __doc__,
              'activation_run': activation_run,
              **inspect_weights(snapshot, layers, args.chunk, args.full, activity)}
    if args.toy_calibration:
        result['toy_calibration'] = toy_calibration(args.toy_steps, 1024, args.seed)
    output = args.output or root / 'runs' / ('superposition-' + str(time.time_ns()))
    output.mkdir(parents=True, exist_ok=False)
    (output / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    embed = result['embed_tokens']
    layers_summary = {layer: values['dimensionality_sum']
                        for layer, values in result['mlp_down_proj_columns'].items()}
    printed = {'output': str(output),
               'embed_dimensionality_sum': embed['dimensionality_sum'],
               'embed_n_represented': embed['n_represented'],
               'mlp_dimensionality_sums': layers_summary,
               'toy_calibration': args.toy_calibration}
    if activity is not None:
        printed['activation_positions'] = activation_run['positions_total']
        printed['mlp_net_interference_median'] = {
            layer: values['net_interference_represented']['median']
            for layer, values in result['mlp_down_proj_columns'].items()}
        printed['mlp_active_fraction_mean'] = {
            layer: values['activation']['active_fraction_mean']
            for layer, values in result['mlp_down_proj_columns'].items()}
    print(json.dumps(printed, indent=2))


if __name__ == '__main__':
    main()
