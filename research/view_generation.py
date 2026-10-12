"""Read one recorded token's internal tensors without loading model weights."""
import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--position', type=int, required=True, help='Absolute token position, zero-based')
    parser.add_argument('--tensor', default='embeddings', help='For example layer_3.heads or layer_0.mlp_neurons')
    parser.add_argument('--head', type=int, help='Original head ID, not selected-array offset')
    parser.add_argument('--feature', type=int, help='Coordinate; original neuron ID for mlp_neurons')
    args = parser.parse_args()
    result = json.loads((args.run / 'result.json').read_text())
    if result.get('schema') != 'generation-v1':
        parser.error('expected generation-v1 capture')
    if not 0 <= args.position < len(result['token_ids']):
        parser.error('position outside captured tokens')
    step = next(s for s in result['steps'] if args.position in s['positions'])
    filename = step['file']
    if Path(filename).name != filename:
        parser.error('invalid tensor filename')
    values = torch.load(args.run / filename, map_location='cpu', weights_only=True)
    if args.tensor not in values:
        parser.error('available tensors: ' + ', '.join(values))
    value = values[args.tensor][0, step['positions'].index(args.position)]
    block = result['blocks'].get(args.tensor.split('.')[0], {})
    if args.head is not None:
        if not args.tensor.endswith('.heads') or args.head not in block['head_indices']:
            parser.error('head not captured in this tensor')
        value = value[block['head_indices'].index(args.head)]
    if args.feature is not None:
        feature = args.feature
        if args.tensor.endswith('.mlp_neurons'):
            if feature not in block['neuron_indices']:
                parser.error('neuron not captured')
            feature = block['neuron_indices'].index(feature)
        if not 0 <= feature < value.shape[-1]:
            parser.error('feature out of bounds')
        value = value[..., feature]
    print(json.dumps({'position': args.position, 'token': result['tokens'][args.position],
                      'token_id': result['token_ids'][args.position], 'tensor': args.tensor,
                      'head': args.head, 'feature': args.feature, 'values': value.tolist()}, indent=2))


if __name__ == '__main__':
    main()
