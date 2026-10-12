import unittest
from types import SimpleNamespace

import torch
from torch import nn

from analyze_model import diagnostics, head_proxies, product_spectrum, Residuals


class OffsetNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.2, -0.5, 0.7, 0.1]))
        self.eps = 1e-6

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps) * (1 + self.weight)


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_lens_and_additive_dla(self):
        norm, unembed = OffsetNorm(), nn.Linear(4, 9)
        x = torch.randn(3, 4)
        target = int(unembed(norm(x[-1])).argmax())
        with torch.no_grad():
            result = diagnostics(list(x), norm, unembed, target)
        self.assertEqual(result['self_target_rank'][-1], 1)
        self.assertLess(result['dla_absolute_error'], 1e-5)
        direction = (1 + norm.weight) * unembed.weight[target] / (x[-1].square().mean() + norm.eps).sqrt()
        self.assertAlmostEqual(result['dla_embedding_then_block_deltas'][1], float(((x[1] - x[0]) @ direction).detach()), places=6)
        expected = torch.distributions.Categorical(logits=unembed(norm(x))).entropy()
        torch.testing.assert_close(torch.tensor(result['entropy_nats']), expected)

    def test_rank_ties(self):
        head = nn.Linear(4, 3, bias=False)
        nn.init.zeros_(head.weight)
        with torch.no_grad():
            result = diagnostics([torch.ones(4)], OffsetNorm(), head, 2)
        self.assertEqual(result['self_target_rank'], [1])
        self.assertAlmostEqual(result['entropy_nats'][0], float(torch.tensor(3.).log()))

    def test_thin_spectrum_matches_dense(self):
        a, b = torch.randn(8, 2), torch.randn(2, 8)
        torch.testing.assert_close(product_spectrum(a, b), torch.linalg.svdvals(a @ b)[:2])

    def test_qwen_interleaved_query_and_gqa_mapping(self):
        attn = SimpleNamespace(head_dim=2, num_key_value_groups=2, scaling=2**-0.5,
                               q_proj=nn.Linear(8, 16, bias=False), k_proj=nn.Linear(8, 4, bias=False),
                               v_proj=nn.Linear(8, 4, bias=False), o_proj=nn.Linear(8, 8, bias=False))
        with torch.no_grad():
            result = head_proxies(attn, 3)
        self.assertEqual(result['kv_head'], 1)
        q, k = attn.q_proj.weight[12:14], attn.k_proj.weight[2:4]
        torch.testing.assert_close(torch.tensor(result['qk_scaled_singular_values']),
                                   torch.linalg.svdvals(q.T @ k * attn.scaling)[:2])
        torch.testing.assert_close(torch.tensor(result['ov_singular_values']),
                                   torch.linalg.svdvals(attn.o_proj.weight[:, 6:8] @ attn.v_proj.weight[2:4])[:2])
        with self.assertRaises(ValueError):
            head_proxies(attn, 4)

    def test_hooks_last_position_and_removal(self):
        lm = SimpleNamespace(embed_tokens=nn.Embedding(5, 4), layers=[nn.Identity()])
        observer = Residuals(lm)
        x = lm.embed_tokens(torch.tensor([[1, 2, 3]]))
        lm.layers[0](x)
        self.assertEqual(len(observer.values), 2)
        torch.testing.assert_close(observer.values[0], x[0, -1])
        observer.close()
        lm.layers[0](x)
        self.assertEqual(len(observer.values), 2)


if __name__ == '__main__':
    unittest.main()
