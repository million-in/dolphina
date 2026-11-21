"""Superposition geometry from Anthropic's toy-models notebook, as diagnostics.

A feature set is a matrix ``W`` of shape ``[n_features, n_hidden]`` whose rows
are feature directions in a shared hidden space. The notebook measures, per
feature ``i`` with unit direction ``u_i = W_i / ||W_i||``:

- norm ``||W_i||`` (a feature the model does not represent is driven to ~0);
- interference ``u_i . W_j`` for ``j != i``, summarized as polysemanticity
  ``sqrt(sum_{j != i} (u_i . W_j)**2)`` (notebook cell ``render_features``);
- net interference weighted by feature probability,
  ``sum_{j != i} (u_i . W_j)**2 * p_j`` (same cell);
- fractional dimensionality ``D_i = ||W_i||**2 / sum_j (u_i . W_j)**2``
  (notebook cell ``compute_dimensionality``; the denominator includes
  ``j == i``, so ``0 < D_i <= 1`` and the ``D_i`` sum to at most the rank);
- ``n_hidden / ||W||_F**2`` over the whole matrix.

The dense forms need an ``[n_features, n_features]`` Gram matrix. Every
quantity above is also a quadratic form in the ``[n_hidden, n_hidden]``
matrix ``G = W^T W`` (or the probability-weighted ``W^T diag(p) W``):
``sum_j (u_i . W_j)**2 == u_i^T G u_i``, and the ``j != i`` variants subtract
the self term ``(u_i . W_i)**2 == ||W_i||**2``. The streaming implementation
below uses that identity in two passes over row chunks, so a vocabulary-sized
feature set never materializes a features-by-features matrix. Tests pin the
streaming results to the dense notebook formulas.

Zero rows are not represented features. The notebook's clamped division gives
them a NaN dimensionality; here they get exact zeros so results stay JSON-safe.
Geometry alone does not prove a direction is a feature, nor that superposed
directions are used as superposed features at inference time.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def _check(matrix):
    if matrix.ndim != 2:
        raise ValueError('feature matrix must be 2D [n_features, n_hidden]')
    if not matrix.is_floating_point() or not torch.isfinite(matrix).all():
        raise ValueError('feature matrix must be finite floating point')


def dense_interference(matrix):
    """Notebook interference ``u_i . W_j`` with the diagonal zeroed.

    Reference implementation for small matrices and tests; the returned
    tensor is ``[n_features, n_features]``.
    """
    _check(matrix)
    norms = matrix.norm(dim=-1, keepdim=True)
    unit = matrix / norms.clamp_min(1e-12)
    interference = unit @ matrix.T
    interference.fill_diagonal_(0.0)
    return torch.where(norms.squeeze(-1) > 0, interference,
                         torch.zeros_like(interference))


def _geometry_from_grams(matrix_chunks, gram, weighted_gram, probabilities):
    norms, dimensionality, polysemanticity, net = [], [], [], []
    offset = 0
    for chunk in matrix_chunks:
        chunk_norms = chunk.norm(dim=-1)
        unit = chunk / chunk_norms.clamp_min(1e-12).unsqueeze(-1)
        denom = (unit @ gram * unit).sum(-1)
        represented = chunk_norms > 0
        dimensionality.append(torch.where(
            represented, chunk_norms.square() / denom.clamp_min(1e-30),
            torch.zeros_like(chunk_norms)))
        # Notebook polysemanticity excludes j == i; subtract the self term.
        polysemanticity.append(torch.where(
            represented, (denom - chunk_norms.square()).clamp_min(0).sqrt(),
            torch.zeros_like(chunk_norms)))
        if weighted_gram is not None:
            weighted = (unit @ weighted_gram * unit).sum(-1)
            own = probabilities[offset:offset + len(chunk)] * chunk_norms.square()
            net.append(torch.where(
                represented, weighted - own, torch.zeros_like(chunk_norms)))
        norms.append(chunk_norms)
        offset += len(chunk)
    result = {'norms': torch.cat(norms),
              'dimensionality': torch.cat(dimensionality),
              'polysemanticity': torch.cat(polysemanticity)}
    if weighted_gram is not None:
        result['net_interference'] = torch.cat(net)
    return result


def feature_geometry_streaming(make_chunks, n_hidden, probabilities=None):
    """Per-feature geometry in two passes over row chunks of ``W``.

    ``make_chunks()`` must return a fresh iterator of ``[rows, n_hidden]``
    floating chunks covering every feature once, in order. Pass one builds
    ``W^T W`` (and ``W^T diag(p) W`` when per-feature ``probabilities`` are
    given, one value per feature); pass two evaluates the quadratic forms.
    Peak extra memory is the chunk plus two ``[n_hidden, n_hidden]`` matrices.
    """
    gram = None
    weighted_gram = None
    count = 0
    for chunk in make_chunks():
        _check(chunk)
        if chunk.shape[1] != n_hidden:
            raise ValueError('chunk hidden size differs from n_hidden')
        chunk = chunk.float()
        term = chunk.T @ chunk
        gram = term if gram is None else gram + term
        if probabilities is not None:
            p = probabilities[count:count + len(chunk)].float()
            if len(p) != len(chunk):
                raise ValueError('probabilities must have one value per feature')
            weighted = chunk.T @ (chunk * p.unsqueeze(-1))
            weighted_gram = weighted if weighted_gram is None else weighted_gram + weighted
        count += len(chunk)
    if count == 0:
        raise ValueError('feature matrix has no rows')
    if probabilities is not None and len(probabilities) != count:
        raise ValueError('probabilities must have one value per feature')
    return _geometry_from_grams(
        (chunk.float() for chunk in make_chunks()), gram, weighted_gram,
        None if probabilities is None else probabilities.float())


def feature_geometry(matrix, probabilities=None, chunk=8192):
    """Per-feature geometry for an in-memory ``[n_features, n_hidden]`` matrix."""
    _check(matrix)
    if probabilities is not None and (
            probabilities.ndim != 1 or len(probabilities) != len(matrix)
            or not torch.isfinite(probabilities).all()
            or (probabilities < 0).any()):
        raise ValueError('probabilities must be finite, non-negative, one per feature')

    def make_chunks():
        for start in range(0, len(matrix), chunk):
            yield matrix[start:start + chunk]

    return feature_geometry_streaming(make_chunks, matrix.shape[1], probabilities)


def _quantiles(values):
    if len(values) == 0:
        return {name: None for name in ('min', 'p25', 'median', 'p75', 'max')}
    q = torch.quantile(values.float(), torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]))
    return {name: float(value) for name, value in
            zip(('min', 'p25', 'median', 'p75', 'max'), q)}


def summarize_geometry(geometry, n_hidden, represented_threshold=0.1):
    """JSON-safe summary; ``represented`` means row norm above the threshold.

    The threshold is absolute and reported, never hidden: the notebook reads
    representation off norm bar charts, so any cutoff is an analyst choice.
    Threshold-free quantities (dimensionality sum, ``n_hidden / ||W||_F**2``,
    quantiles, dimensionality histogram) are always reported alongside.
    """
    norms = geometry['norms'].float()
    dimensionality = geometry['dimensionality'].float()
    represented = norms > represented_threshold
    histogram = torch.histogram(
        dimensionality[represented], bins=10, range=(0.0, 1.0))[0] \
        if represented.any() else torch.zeros(10)
    summary = {
        'n_features': len(norms),
        'n_hidden': n_hidden,
        'represented_threshold': represented_threshold,
        'n_represented': int(represented.sum()),
        'dimensionality_sum': float(dimensionality.sum()),
        'hidden_over_frobenius_sq':
            float(n_hidden / norms.square().sum()) if norms.square().sum() > 0 else None,
        'norms': _quantiles(norms),
        'dimensionality_represented': _quantiles(dimensionality[represented]),
        'polysemanticity_represented':
            _quantiles(geometry['polysemanticity'].float()[represented]),
        'dimensionality_histogram_0_to_1_10_bins': histogram.tolist(),
    }
    if 'net_interference' in geometry:
        summary['net_interference_represented'] = _quantiles(
            geometry['net_interference'].float()[represented])
    return summary


@dataclass
class ToyConfig:
    """Notebook ``Config``: many instances are trained as one batched model."""
    n_features: int
    n_hidden: int
    n_instances: int


class ToyModel(nn.Module):
    """Tied-weight ReLU autoencoder from the notebook, CPU and seeded.

    Deviations from the Colab original are deliberate and limited: the device
    defaults to CPU instead of CUDA, all randomness flows through an explicit
    ``torch.Generator``, and scalars broadcast exactly as in the notebook.
    ``W`` has shape ``[n_instances, n_features, n_hidden]``; each row
    ``W[i, f]`` is feature ``f``'s direction in instance ``i``'s hidden space.
    """

    def __init__(self, config, feature_probability=None, importance=None,
                 generator=None):
        super().__init__()
        self.config = config
        self.W = nn.Parameter(torch.empty(
            (config.n_instances, config.n_features, config.n_hidden)))
        nn.init.xavier_normal_(self.W, generator=generator)
        self.b_final = nn.Parameter(torch.zeros(
            (config.n_instances, config.n_features)))
        self.feature_probability = torch.ones(()) if feature_probability is None \
            else torch.as_tensor(feature_probability, dtype=torch.float32)
        self.importance = torch.ones(()) if importance is None \
            else torch.as_tensor(importance, dtype=torch.float32)
        for name, value in (('feature_probability', self.feature_probability),
                            ('importance', self.importance)):
            if not torch.isfinite(value).all() or (value < 0).any():
                raise ValueError(f'{name} must be finite and non-negative')
        if (self.feature_probability > 1).any():
            raise ValueError('feature_probability must not exceed 1')

    def forward(self, features):
        hidden = torch.einsum('...if,ifh->...ih', features, self.W)
        out = torch.einsum('...ih,ifh->...if', hidden, self.W)
        return F.relu(out + self.b_final)

    def generate_batch(self, n_batch, generator=None):
        shape = (n_batch, self.config.n_instances, self.config.n_features)
        values = torch.rand(shape, generator=generator)
        active = torch.rand(shape, generator=generator) <= self.feature_probability
        return torch.where(active, values, torch.zeros(()))

    def loss_per_instance(self, batch):
        error = self.importance * (batch.abs() - self(batch)) ** 2
        return error.mean(dim=(0, 2))


def train_toy(config, feature_probability=None, importance=None, steps=4000,
              n_batch=1024, lr=1e-3, seed=0):
    """Notebook ``optimize`` with constant LR, made reproducible.

    Returns ``(model, record)`` where the record holds before/after loss per
    instance on one fixed seeded evaluation batch; noisy training-batch loss
    is not used as evidence of convergence. No LR schedule, rendering or
    progress hooks: the notebook's defaults are constant LR without them.
    """
    if steps < 1 or n_batch < 1 or lr <= 0:
        raise ValueError('steps, n_batch and lr must be positive')
    generator = torch.Generator().manual_seed(seed)
    model = ToyModel(config, feature_probability, importance, generator=generator)
    eval_batch = model.generate_batch(n_batch, generator=torch.Generator().manual_seed(seed + 1))
    with torch.no_grad():
        initial = model.loss_per_instance(eval_batch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss_per_instance(model.generate_batch(n_batch, generator)).sum()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final = model.loss_per_instance(eval_batch)
    if not (torch.isfinite(final).all() and torch.isfinite(model.W).all()):
        raise ValueError('nonfinite toy training result')
    record = {'seed': seed, 'steps': steps, 'n_batch': n_batch, 'lr': lr,
              'eval_loss_per_instance_initial': initial.tolist(),
              'eval_loss_per_instance_final': final.tolist()}
    return model, record


def toy_instance_geometry(model):
    """Geometry summary per trained instance, probabilities included when known."""
    probabilities = model.feature_probability
    summaries = []
    for instance in range(model.config.n_instances):
        p = None
        if probabilities.ndim == 2:
            p = probabilities[instance]
        elif probabilities.ndim == 0:
            p = probabilities.expand(model.config.n_features)
        geometry = feature_geometry(model.W.detach()[instance], p)
        summary = summarize_geometry(geometry, model.config.n_hidden)
        summary['instance'] = instance
        if probabilities.ndim == 2:
            summary['feature_probability_first'] = float(probabilities[instance, 0])
        summaries.append(summary)
    return summaries
