"""Offline viewer for inspect_model.py prompt-token captures.

Example: python research/view_activations.py research/runs/RUN --layer 2 \
    --token all --feature 0,7 --csv values.csv
Indices are zero-based; selectors accept integers, comma-separated integers or
all. Defaults show the first captured layer, token 0 and feature 0. Explicit
all selections can print many rows. No model or tokenizer is downloaded.
"""
import argparse
import csv
import json
from pathlib import Path
import re
import sys
import pickle


def select_indices(selection, available, label):
    """Resolve a selection without silently accepting missing or negative indices."""
    available = list(available)
    if selection is None:
        return available[:1]
    if selection == 'all':
        return available
    try:
        indices = list(dict.fromkeys(int(item) for item in selection.split(',')))
    except ValueError as exc:
        raise ValueError(f'{label}: expected all or comma-separated integers') from exc
    if not indices or any(index not in available for index in indices):
        raise ValueError(f'{label}: selection {selection!r} is not available')
    return indices


def load_capture(path):
    """Load only tensors on CPU and validate the producer's single-batch schema."""
    import torch

    path = Path(path)
    if path.is_dir():
        path = path / 'activations.pt'
    with (path.parent / 'result.json').open(encoding='utf-8') as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise ValueError('result.json must contain an object')
    tokens, token_ids = result.get('tokens'), result.get('token_ids')
    if (not isinstance(tokens, list) or not tokens
            or any(not isinstance(token, str) for token in tokens)
            or not isinstance(token_ids, list) or len(tokens) != len(token_ids)
            or any(type(token_id) is not int or token_id < 0 for token_id in token_ids)):
        raise ValueError('result.json requires matching nonempty tokens and token_ids lists')
    tensors = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(tensors, dict) or not tensors:
        raise ValueError('activations.pt must contain a nonempty layer tensor mapping')
    summaries = result.get('activations')
    if not isinstance(summaries, dict) or summaries.keys() != tensors.keys():
        raise ValueError('result.json activation layers do not match activations.pt')
    for name, tensor in tensors.items():
        if not isinstance(name, str) or re.fullmatch(r'layer_(0|[1-9][0-9]*)', name) is None:
            raise ValueError(f'invalid layer name: {name!r}')
        if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
                or not tensor.is_floating_point() or tensor.ndim != 3
                or tensor.shape[0] != 1 or tensor.shape[1] != len(tokens)
                or tensor.shape[2] < 1):
            raise ValueError(f'{name}: expected floating tensor [1, {len(tokens)}, features]')
        summary = summaries[name]
        if not isinstance(summary, dict) or summary.get('shape') != list(tensor.shape):
            raise ValueError(f'{name}: recorded shape does not match tensor')
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f'{name}: nonfinite activation values')
    return tensors, result


def selected_rows(tensors, result, layer=None, token='0', feature='0'):
    """Validate every selected index before returning an iterator of scalar rows."""
    layers = select_indices(layer, sorted(int(name[6:]) for name in tensors), 'layer')
    tokens = select_indices(token, range(len(result['tokens'])), 'token')
    features = {
        index: select_indices(feature, range(tensors[f'layer_{index}'].shape[2]), 'feature')
        for index in layers
    }
    return (
        (index, position, result['token_ids'][position], result['tokens'][position],
         dimension, tensors[f'layer_{index}'][0, position, dimension].item())
        for index in layers for position in tokens for dimension in features[index]
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', type=Path, help='Run directory or activations.pt path')
    parser.add_argument('--layer', help='Layer index, comma-separated indices, or all; default: first captured')
    parser.add_argument('--token', default='0', help='Prompt token selection; default: 0')
    parser.add_argument('--feature', default='0', help='Hidden feature selection; default: 0')
    parser.add_argument('--csv', type=Path, help='Export selected scalar rows to a NEW CSV file')
    args = parser.parse_args(argv)
    try:
        tensors, result = load_capture(args.input)
        rows = selected_rows(tensors, result, args.layer, args.token, args.feature)
        # Exclusive creation protects captures and any existing exports, including symlinks.
        output = args.csv.open('x', encoding='utf-8', newline='') if args.csv else None
        try:
            writer = csv.writer(output) if output else None
            if writer:
                writer.writerow(('layer', 'token_index', 'token_id', 'token', 'feature', 'value'))
            for row in rows:
                layer, token, token_id, label, feature, value = row
                print(f'layer_{layer} token[{token}] id={token_id} label={label!r} '
                      f'feature[{feature}] = {value!r}')
                if writer:
                    writer.writerow(row)
        finally:
            if output:
                output.close()
    except ImportError as exc:
        parser.error(f'PyTorch is required in the current Python environment: {exc}')
    except (OSError, ValueError, RuntimeError, TypeError, EOFError, pickle.UnpicklingError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == '__main__':
    sys.exit(main())
