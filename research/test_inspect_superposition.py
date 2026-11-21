import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from inspect_superposition import DOWN, EMBED, inspect_weights, tensor_files
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


if __name__ == '__main__':
    unittest.main()
