import unittest

import torch

from superposition import (dense_interference, feature_geometry,
                           feature_geometry_streaming, summarize_geometry)


def notebook_geometry(matrix, probabilities):
    """Dense notebook formulas, written independently of superposition.py.

    Cells ``render_features`` and ``compute_dimensionality`` of
    anthropics/toy-models-of-superposition ``toy_models.ipynb``, for a
    single instance ``W`` of shape [n_features, n_hidden].
    """
    w_norm = matrix / (1e-5 + torch.linalg.norm(matrix, 2, dim=-1, keepdim=True))
    interference = w_norm @ matrix.T
    interference[torch.arange(len(matrix)), torch.arange(len(matrix))] = 0
    norms = torch.linalg.norm(matrix, 2, dim=-1)
    w_unit = matrix / torch.clamp(norms[:, None], 1e-6, float('inf'))
    dim_fracs = norms ** 2 / ((w_unit @ matrix.T) ** 2).sum(-1)
    return {'norms': norms,
            'dimensionality': dim_fracs,
            'polysemanticity': torch.linalg.norm(interference, dim=-1),
            'net_interference': (interference ** 2 * probabilities[None, :]).sum(-1)}


class GeometryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)

    def test_streaming_matches_dense_notebook_formulas(self):
        matrix = torch.randn(37, 11)
        probabilities = torch.rand(37)
        expected = notebook_geometry(matrix, probabilities)
        # Uneven chunks exercise the two-pass accumulation and offsets.
        result = feature_geometry(matrix, probabilities, chunk=8)
        for key in expected:
            torch.testing.assert_close(result[key], expected[key],
                                       rtol=2e-4, atol=2e-5, msg=key)
        streamed = feature_geometry_streaming(
            lambda: iter([matrix[:13], matrix[13:]]), 11, probabilities)
        for key in expected:
            torch.testing.assert_close(streamed[key], result[key], msg=key)
        dense = dense_interference(matrix)
        torch.testing.assert_close(dense.norm(dim=-1), result['polysemanticity'],
                                   rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(dense.diag(), torch.zeros(37))

    def test_known_geometries(self):
        # Orthonormal features: no interference, one dimension each.
        result = feature_geometry(torch.eye(4))
        torch.testing.assert_close(result['dimensionality'], torch.ones(4))
        torch.testing.assert_close(result['polysemanticity'], torch.zeros(4))
        # An antipodal pair shares one dimension: D = 1/2 each.
        pair = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 2.0]])
        result = feature_geometry(pair)
        torch.testing.assert_close(result['dimensionality'],
                                   torch.tensor([0.5, 0.5, 1.0]), atol=1e-6, rtol=0)
        torch.testing.assert_close(result['polysemanticity'],
                                   torch.tensor([1.0, 1.0, 0.0]), atol=1e-6, rtol=0)
        # Five unit features as a pentagon in 2D: sum_j cos^2 = 5/2, D = 2/5.
        angles = torch.arange(5) * 2 * torch.pi / 5
        pentagon = torch.stack([angles.cos(), angles.sin()], dim=-1)
        result = feature_geometry(pentagon)
        torch.testing.assert_close(result['dimensionality'],
                                   torch.full((5,), 0.4), atol=1e-6, rtol=0)
        self.assertAlmostEqual(float(result['dimensionality'].sum()), 2.0, places=5)

    def test_zero_rows_are_unrepresented_not_nan(self):
        matrix = torch.tensor([[0.0, 0.0], [3.0, 0.0], [1.0, 0.0]])
        result = feature_geometry(matrix, torch.tensor([0.5, 0.5, 0.5]))
        for values in result.values():
            self.assertTrue(torch.isfinite(values).all())
            self.assertEqual(float(values[0]), 0.0)
        summary = summarize_geometry(result, 2)
        self.assertEqual(summary['n_represented'], 2)
        self.assertEqual(summary['dimensionality_histogram_0_to_1_10_bins'][0], 0.0)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            feature_geometry(torch.randn(4))
        with self.assertRaises(ValueError):
            feature_geometry(torch.full((3, 2), float('nan')))
        with self.assertRaises(ValueError):
            feature_geometry(torch.randn(3, 2), torch.rand(2))
        with self.assertRaises(ValueError):
            feature_geometry(torch.randn(3, 2), -torch.rand(3))
        with self.assertRaises(ValueError):
            feature_geometry_streaming(lambda: iter([]), 2)
        with self.assertRaises(ValueError):
            feature_geometry_streaming(lambda: iter([torch.randn(2, 3)]), 2)

    def test_summary_threshold_free_quantities(self):
        matrix = torch.eye(4)
        summary = summarize_geometry(feature_geometry(matrix), 4)
        self.assertEqual(summary['dimensionality_sum'], 4.0)
        self.assertEqual(summary['hidden_over_frobenius_sq'], 1.0)
        self.assertEqual(summary['dimensionality_represented']['median'], 1.0)


if __name__ == '__main__':
    unittest.main()
