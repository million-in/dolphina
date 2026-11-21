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

These are weight geometry, not activation features: neurons and tokens are
not established features, no feature frequencies are measured, so the
notebook's probability-weighted net interference is not reported for Qwen,
and overlap between write directions alone does not prove the model uses
superposed features. See INTERPRETATION.md.
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


def _geometry_for(mapping, name, transpose, n_hidden, chunk):
    if name not in mapping:
        raise ValueError(f'missing tensor {name}')
    path = mapping[name]
    with safe_open(path, framework='pt') as handle:
        shape = handle.get_slice(name).get_shape()
    if len(shape) != 2 or shape[0 if transpose else 1] != n_hidden:
        raise ValueError(f'{name} has shape {shape}, expected hidden size {n_hidden}')

    def make_chunks():
        return _feature_chunks(path, name, transpose, chunk)

    geometry = feature_geometry_streaming(make_chunks, n_hidden)
    if not all(torch.isfinite(values).all() for values in geometry.values()):
        raise ValueError(f'nonfinite geometry for {name}')
    return geometry


def _with_full(summary, geometry, full):
    if full:
        summary['per_feature'] = {key: values.tolist()
                                  for key, values in geometry.items()}
    return summary


def inspect_weights(snapshot, layers=None, chunk=8192, full=False):
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
        geometry = _geometry_for(mapping, DOWN.format(layer), True, n_hidden, chunk)
        summary = _with_full(summarize_geometry(geometry, n_hidden), geometry, full)
        summary['tensor'] = DOWN.format(layer)
        result['mlp_down_proj_columns'][str(layer)] = summary
    return result


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
    result = {'schema': 'superposition-v1', 'model': model_id,
              'revision': revision if args.model_dir is None else None,
              'snapshot': str(snapshot), 'torch': torch.__version__,
              'computation_dtype': 'float32 (upcast from stored weights)',
              'layers': layers, 'caveats': __doc__,
              **inspect_weights(snapshot, layers, args.chunk, args.full)}
    if args.toy_calibration:
        result['toy_calibration'] = toy_calibration(args.toy_steps, 1024, args.seed)
    output = args.output or root / 'runs' / ('superposition-' + str(time.time_ns()))
    output.mkdir(parents=True, exist_ok=False)
    (output / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    embed = result['embed_tokens']
    layers_summary = {layer: values['dimensionality_sum']
                        for layer, values in result['mlp_down_proj_columns'].items()}
    print(json.dumps({'output': str(output),
                      'embed_dimensionality_sum': embed['dimensionality_sum'],
                      'embed_n_represented': embed['n_represented'],
                      'mlp_dimensionality_sums': layers_summary,
                      'toy_calibration': args.toy_calibration}, indent=2))


if __name__ == '__main__':
    main()
