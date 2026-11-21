import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from safetensors.torch import save_file

from inspect_superposition import (DOWN, EMBED, NeuronActivity,
                                   inspect_weights, tensor_files)
from superposition import feature_geometry, summarize_geometry


class InspectWeightsTests(unittest.TestCase):
    def make_snapshot(self, tensors, hidden=2, layers=1):
        directory = tempfile.TemporaryDirectory()
        root = Path(directory.name)
        (root / 'config.json').write_text(json.dumps({'text_config': {
            'hidden_size': hidden, 'num_hidden_layers': layers,
            'intermediate_size': 3, 'vocab_size': 5,
            'tie_word_embeddings': True, 'dtype': 'bfloat16'}}))
        save_file(tensors, str(root / 'model.safetensors'), metadata={'format': 'pt'})
        (root / 'model.safetensors.index.json').write_text(json.dumps(
            {'weight_map': {name: 'model.safetensors' for name in tensors}}))
        return directory

    def test_known_weight_geometry(self):
        # Pentagon embedding rows in 2D: every dimensionality is 2/5.
        angles = torch.arange(5) * 2 * torch.pi / 5
        pentagon = torch.stack([angles.cos(), angles.sin()], dim=-1)
        # down_proj stored [hidden, intermediate]; columns are e1, -e1, e2.
        down = torch.tensor([[1.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
        with self.make_snapshot({EMBED: pentagon.bfloat16(),
                                 DOWN.format(0): down.bfloat16()}) as snapshot:
            result = inspect_weights(snapshot, layers=[0], chunk=2, full=True)
        embed = result['embed_tokens']
        self.assertEqual(embed['n_features'], 5)
        # BF16 storage limits precision; the pentagon value is known exactly.
        self.assertAlmostEqual(embed['dimensionality_sum'], 2.0, places=2)
        for value in embed['per_feature']['dimensionality']:
            self.assertAlmostEqual(value, 0.4, places=2)
        mlp = result['mlp_down_proj_columns']['0']
        self.assertEqual(mlp['n_features'], 3)
        for got, want in zip(mlp['per_feature']['dimensionality'], [0.5, 0.5, 1.0]):
            self.assertAlmostEqual(got, want, places=6)
        expected = summarize_geometry(feature_geometry(down.T), 2)
        self.assertAlmostEqual(mlp['dimensionality_sum'],
                               expected['dimensionality_sum'], places=6)

    def test_missing_tensor_and_bad_layer_fail_closed(self):
        tensors = {EMBED: torch.eye(2)}
        with self.make_snapshot(tensors) as snapshot:
            with self.assertRaises(ValueError):
                inspect_weights(snapshot, layers=[0])
            self.assertIn(EMBED, tensor_files(snapshot))
            with self.assertRaises(ValueError):
                inspect_weights(snapshot, layers=[1])
            empty = Path(snapshot) / 'empty'
            empty.mkdir()
            with self.assertRaises(ValueError):
                tensor_files(empty)

    def test_wrong_hidden_shape_is_rejected(self):
        tensors = {EMBED: torch.randn(4, 3), DOWN.format(0): torch.randn(2, 3)}
        with self.make_snapshot(tensors) as snapshot:
            with self.assertRaises(ValueError):
                inspect_weights(snapshot, layers=[0])


class FakeMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 3, bias=False)
        self.up_proj = nn.Linear(4, 3, bias=False)
        self.down_proj = nn.Linear(3, 4, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class NeuronActivityTests(unittest.TestCase):
    def test_counts_match_manual(self):
        torch.manual_seed(5)
        mlp = FakeMlp()
        activity = NeuronActivity({0: mlp}, chunk=2)
        x = torch.randn(2, 5, 4)
        with torch.no_grad():
            mlp(x)
            mlp(x)  # second pass must double, not replace, the positions
        result = activity.result()[0]
        activity.close()
        gate, up = mlp.gate_proj(x), mlp.up_proj(x)
        m = torch.nn.functional.silu(gate) * up
        self.assertEqual(result['positions'], 20)
        torch.testing.assert_close(result['active_fraction'],
                                   (gate > 0).float().mean(dim=(0, 1)))
        torch.testing.assert_close(result['mean_square_m'], m.square().mean(dim=(0, 1)))
        torch.testing.assert_close(result['mean_abs_m'], m.abs().mean(dim=(0, 1)))

    def test_activity_feeds_net_interference(self):
        angles = torch.arange(5) * 2 * torch.pi / 5
        pentagon = torch.stack([angles.cos(), angles.sin()], dim=-1)
        down = torch.tensor([[1.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
        probabilities = torch.tensor([0.25, 1.0, 0.5])
        activity = {0: {'positions': 8, 'active_fraction': probabilities,
                        'mean_square_m': torch.zeros(3), 'mean_abs_m': torch.zeros(3)}}
        helper = InspectWeightsTests()
        with helper.make_snapshot({EMBED: pentagon, DOWN.format(0): down}) as snapshot:
            result = inspect_weights(snapshot, layers=[0], chunk=2, full=True,
                                     activity=activity)
        mlp = result['mlp_down_proj_columns']['0']
        expected = feature_geometry(down.T, probabilities)
        torch.testing.assert_close(
            torch.tensor(mlp['per_feature']['net_interference']),
            expected['net_interference'])
        self.assertIn('net_interference_represented', mlp)
        self.assertEqual(mlp['activation']['active_fraction']['max'], 1.0)
        with helper.make_snapshot({EMBED: pentagon, DOWN.format(0): down}) as snapshot:
            with self.assertRaises(ValueError):
                bad = dict(activity)
                bad[0] = dict(activity[0], active_fraction=torch.zeros(2))
                inspect_weights(snapshot, layers=[0], activity=bad)


if __name__ == '__main__':
    unittest.main()
