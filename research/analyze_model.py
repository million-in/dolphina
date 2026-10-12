"""Descriptive CPU FP32 diagnostics, not causal attribution or attention circuits.

Each generated token is its own target at the preceding residual position.
Lens: final RMSNorm + unembedding at every block boundary. DLA: embedding
and additive block deltas, with the observed final RMS denominator frozen.
QK/OV spectra are raw-weight proxies: omit input/QK norms, RoPE, softmax,
attention masks, query-dependent output gates, biases, and all DeltaNet blocks.
"""
import argparse
import json
from pathlib import Path
import time

import torch

from record_generation import indices


def diagnostics(residuals, norm, unembed, target):
    """One position, embedding then every block output; ranks use strict > ties."""
    x = torch.stack(residuals)
    logits = unembed(norm(x)).float()
    logp = logits.log_softmax(-1)
    entropy = -(logp.exp() * logp).sum(-1)
    rank = 1 + (logits > logits[:, target, None]).sum(-1)
    denominator = (x[-1].float().square().mean() + norm.eps).sqrt()
    direction = (1 + norm.weight.float()) * unembed.weight[target].float() / denominator
    pieces = torch.cat((x[:1], x[1:] - x[:-1]))
    contributions = pieces @ direction
    bias = 0.0 if unembed.bias is None else float(unembed.bias[target])
    reconstructed = float(contributions.sum()) + bias
    error = abs(reconstructed - float(logits[-1, target]))
    if not torch.isfinite(logits).all() or not torch.isfinite(contributions).all():
        raise ValueError('nonfinite diagnostics')
    if not torch.allclose(contributions.sum() + bias, logits[-1, target], atol=2e-4, rtol=2e-5):
        raise AssertionError(f'DLA reconstruction error: {error}')
    return {'entropy_nats': entropy.tolist(), 'self_target_rank': rank.tolist(),
            'self_target_logit': logits[:, target].tolist(),
            'dla_embedding_then_block_deltas': contributions.tolist(),
            'dla_unembed_bias': bias, 'dla_reconstructed_logit': reconstructed,
            'dla_absolute_error': error, 'frozen_final_rms_denominator': float(denominator)}


def product_spectrum(left, right):
    """Nonzero singular spectrum of left @ right, without a hidden² matrix.

    Thin QR preserves singular values. Structural trailing zeros are omitted;
    singular vectors are intentionally not saved.
    """
    _, a = torch.linalg.qr(left, mode='reduced')
    _, b = torch.linalg.qr(right.T, mode='reduced')
    return torch.linalg.svdvals(a @ b.T)


def head_proxies(attn, head):
    dim = attn.head_dim
    count = attn.o_proj.in_features // dim
    if not 0 <= head < count:
        raise ValueError('invalid query head')
    kv = head // attn.num_key_value_groups
    # Qwen packs [query, gate] within EACH head, not two global halves.
    q = attn.q_proj.weight.reshape(count, 2 * dim, -1)[head, :dim]
    k = attn.k_proj.weight[kv * dim:(kv + 1) * dim]
    v = attn.v_proj.weight[kv * dim:(kv + 1) * dim]
    o = attn.o_proj.weight[:, head * dim:(head + 1) * dim]
    return {'query_head': head, 'kv_head': kv,
            'qk_scaled_singular_values': product_spectrum(q.T * attn.scaling, k).tolist(),
            'ov_singular_values': product_spectrum(o, v).tolist()}


class Residuals:
    """Observe only the last processed position; do not retain full prefixes."""
    def __init__(self, lm):
        self.values = []
        self.handles = []
        for module in [lm.embed_tokens, *lm.layers]:
            self.handles.append(module.register_forward_hook(self.capture))

    def capture(self, _module, _args, output):
        tensor = output[0] if isinstance(output, tuple) else output
        self.values.append(tensor[0, -1].detach().clone())

    def close(self):
        for handle in self.handles:
            handle.remove()


def main():
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--max-new-tokens', type=int, default=32)
    parser.add_argument('--blocks', default='3', help='Full-attention blocks for weight spectra only')
    parser.add_argument('--heads', default='0')
    parser.add_argument('--verify', action='store_true')
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
    text = tokenizer.apply_chat_template([{'role': 'user', 'content': args.prompt}],
                                        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors='pt')
    length = inputs.input_ids.shape[1]
    if length > 256:
        parser.error('maximum 256 prompt tokens including template')
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, revision=revision, dtype=torch.float32,
        attn_implementation='eager', trust_remote_code=False).cpu().eval()
    if any(p.dtype != torch.float32 or p.device.type != 'cpu' for p in model.parameters()):
        raise AssertionError('model must be CPU FP32')
    lm = model.model.language_model
    unembed = model.get_output_embeddings()
    spectra = {}
    with torch.inference_mode():
        for block in indices(args.blocks, len(lm.layers)):
            layer = lm.layers[block]
            if not hasattr(layer, 'self_attn'):
                parser.error(f'block {block} is not full attention')
            attn = layer.self_attn
            spectra[str(block)] = [head_proxies(attn, h) for h in
                                   indices(args.heads, attn.o_proj.in_features // attn.head_dim)]
    ids = inputs.input_ids.clone()
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    rows = []
    observer = Residuals(lm)
    try:
        with torch.inference_mode():
            for step in range(args.max_new_tokens):
                observer.values = []
                logits = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                               use_cache=False, logits_to_keep=1).logits[0, -1]
                if not torch.isfinite(logits).all():
                    raise ValueError('nonfinite model logits')
                target = int(logits.argmax())
                row = diagnostics(observer.values, lm.norm, unembed, target)
                lens_final = unembed(lm.norm(observer.values[-1]))
                if not torch.allclose(lens_final, logits, atol=2e-4, rtol=2e-5):
                    raise AssertionError('final lens differs from model logits')
                row.update(step=step, residual_position=ids.shape[1] - 1,
                           prediction_position=ids.shape[1], self_target_id=target,
                           final_lens_max_logit_error=float((lens_final - logits).abs().max()))
                rows.append(row)
                ids = torch.cat((ids, ids.new_tensor([[target]])), dim=1)
                if target in eos:
                    break
    finally:
        observer.close()
    verified = None
    if args.verify:
        with torch.inference_mode():
            baseline = model.generate(**inputs, do_sample=False, use_cache=False,
                                      max_new_tokens=args.max_new_tokens)
        verified = torch.equal(ids, baseline)
        if not verified:
            raise AssertionError('instrumented IDs differ from uninstrumented generate')
    output = args.output or root / 'runs' / ('analysis-' + str(time.time_ns()))
    output.mkdir(parents=True, exist_ok=False)
    result = {'schema': 'diagnostics-v1', 'model': model_id, 'revision': revision,
              'torch': torch.__version__, 'dtype': 'float32', 'device': 'cpu',
              'prompt': args.prompt, 'prompt_length': length, 'max_new_tokens': args.max_new_tokens,
              'token_ids': ids[0].tolist(), 'response': tokenizer.decode(ids[0, length:], skip_special_tokens=True),
              'stop_reason': 'eos' if rows[-1]['self_target_id'] in eos else 'max_new_tokens',
              'baseline_ids_equal': verified, 'steps': rows, 'weight_spectra': spectra,
              'residual_labels': ['embedding'] + [f'layer_{i}' for i in range(len(lm.layers))],
              'caveats': __doc__, 'rank_convention': '1 + number of strictly larger logits',
              'dla_convention': 'Signed target-logit decomposition, not probabilities or causal effects; final RMS denominator frozen for every additive term.',
              'spectrum_convention': 'QK = Wq.T @ Wk / sqrt(head_dim); OV = Wo @ Wv; thin QR SVD, structural trailing zeros omitted, no singular vectors; kv_head = query_head // num_key_value_groups.'}
    (output / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(output), 'prompt_tokens': length, 'generated_tokens': len(rows),
                      'baseline_ids_equal': verified,
                      'max_dla_error': max(row['dla_absolute_error'] for row in rows)}, indent=2))


if __name__ == '__main__':
    main()
