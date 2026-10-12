"""Small architecture-faithful hook tests; no downloaded model required."""
import unittest
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from record_generation import Recorder, indices


class Block(nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.down_proj = nn.Linear(4, 4)
        attention = nn.Module()
        if linear:
            attention.head_v_dim = 2
            attention.out_proj = nn.Linear(4, 4)
            self.linear_attn = attention
        else:
            attention.head_dim = 2
            attention.o_proj = nn.Linear(4, 4)
            self.self_attn = attention

    def forward(self, hidden_states):
        x = self.mlp.down_proj(hidden_states)
        if hasattr(self, 'linear_attn'):
            return self.linear_attn.out_proj(x)
        return self.self_attn.o_proj(x)


class RecorderTests(unittest.TestCase):
    def test_selectors(self):
        self.assertEqual(indices('2,0,2', 3), [0, 2])
        for value in ('-1', '3', '', 'foo'):
            with self.assertRaises(ValueError):
                indices(value, 3)

    def test_alignment_and_slices(self):
        lm = SimpleNamespace(embed_tokens=nn.Embedding(8, 4), layers=[Block(True), Block(False)])
        model = SimpleNamespace(model=SimpleNamespace(language_model=lm))
        recorder = Recorder(model, [0, 1], '1', '2')
        recorder.start = 1
        try:
            x = lm.embed_tokens(torch.tensor([[1, 2, 3]]))
            for i, layer in enumerate(lm.layers):
                original = x.clone()
                x = layer(hidden_states=x)
                self.assertTrue(torch.equal(recorder.values[f'layer_{i}.mlp_neurons'], original[:, 1:, [2]]))
                self.assertEqual(recorder.values[f'layer_{i}.heads'].shape, (1, 2, 1, 2))
                self.assertTrue(torch.equal(recorder.values[f'layer_{i}.output'], x[:, 1:]))
            self.assertEqual(recorder.values['embeddings'].shape, (1, 2, 4))
        finally:
            recorder.close()
        self.assertFalse(lm.layers[0]._forward_hooks)
        self.assertFalse(lm.layers[0]._forward_pre_hooks)

    @unittest.skipUnless(os.environ.get('CAPTURE_RUN'), 'set CAPTURE_RUN to a real generation run')
    def test_live_capture(self):
        root = Path(os.environ['CAPTURE_RUN'])
        result = json.loads((root / 'result.json').read_text())
        self.assertTrue(result['baseline_ids_equal'])
        positions = []
        for step in result['steps']:
            positions.extend(step['positions'])
            self.assertEqual(step['processed_token_ids'], [result['token_ids'][p] for p in step['positions']])
            tensors = torch.load(root / step['file'], weights_only=True, map_location='cpu')
            self.assertEqual(len(tensors), 1 + 4 * len(result['blocks']))
            for name, tensor in tensors.items():
                self.assertEqual(tensor.shape[:2], (1, len(step['positions'])))
                self.assertEqual(tensor.dtype, torch.float32)
                self.assertTrue(torch.isfinite(tensor).all())
                block = result['blocks'].get(name.split('.')[0])
                if name.endswith('.heads'):
                    self.assertEqual(tensor.shape[2:], (len(block['head_indices']), block['head_dim']))
                elif name.endswith('.mlp_neurons'):
                    self.assertEqual(tensor.shape[2], len(block['neuron_indices']))
            self.assertEqual(step['prediction_position'], step['positions'][-1] + 1)
            if step['emitted_token_id'] is not None:
                self.assertEqual(step['emitted_token_id'], result['token_ids'][step['prediction_position']])
        self.assertEqual(positions, list(range(len(result['token_ids']))))
        self.assertIsNone(result['steps'][-1]['emitted_token_id'])


if __name__ == '__main__':
    unittest.main()
