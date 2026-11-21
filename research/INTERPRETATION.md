# Dolphine: interpreting the measurements

## Read/write projections

A residual vector is the shared state at one token position, not a list of
independent concepts. Blocks read normalized versions of that state and add
updates to it. An MLP's up projection reads into a larger intermediate space;
in this checkpoint a separate gate modulates those values:
`m = SiLU(gate_proj(x)) * up_proj(x)`. The down projection writes `down_proj(m)`
back into residual space. A neuron's write is its scalar activation times the
corresponding column of the down-projection weight, not the scalar alone.

The existing recorder saves the intermediate scalars before down projection.
Head vectors likewise need their output-projection columns to describe their
residual writes. Neither a large activation nor a large singular value proves
that a component caused an answer.

## What we take from the sources

[A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/)
motivates residual read/write accounting and separating attention addressing
(QK) from value transport (OV). Its simplified attention-only analysis is not
an exact model of this hybrid gated architecture. The residual-stream, QK/OV,
and MLP discussion were consulted through indexed source excerpts, not a
complete local copy of the article.

[arXiv:2603.06591](https://arxiv.org/pdf/2603.06591) was indexed as a position-zero
attention-sink study. Its indexed abstract motivates comparing first-position
residual norms with other positions. Search results disagree about revisions
and metadata, and the PDF itself has not been verified here: do not treat this
as a full-paper review or evidence that its proposed circuit exists in Qwen3.5.
Norm differences alone are not attention sinks. Establishing a sink requires
actual attention probabilities in full-attention blocks, prompt controls and
interventions; DeltaNet blocks do not have the same softmax attention map.

## Diagnostic boundaries

- Logit lens applies the final RMSNorm and vocabulary unembedding to an
  intermediate residual. This is a diagnostic readout, not the model's actual
  early exit. Entropy is distributional uncertainty, not factual correctness.
- Self-target rank scores the model's own continuation, not a true answer.
  Rank at the final depth is expected to favor greedy-generated targets.
- Direct logit attribution freezes the final token's RMS denominator so
  residual writes can add up in logit space. Re-normalizing each write
  independently is not an additive decomposition. DLA is not an ablation.
- In PyTorch weight orientation, raw head proxies are `Wq.T @ Wk` and
  `Wo @ Wv`. Grouped-query heads share KV weights. These proxies omit RoPE,
  query/key normalization, dynamic gates and attention patterns; their SVD
  spectra describe weight geometry, not the effective input-specific circuit.
  Do not label DeltaNet parameters as ordinary QK/OV attention circuits.

Use these tools to select hypotheses, then compare matched prompts and perform
controlled ablation/patching before making causal claims.

## Superposition geometry

[Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html)
(Elhage et al.) trains a tied-weight ReLU autoencoder `relu(x W^T W + b)` on
synthetic features that are sparse (active with probability `p`) and weighted
by importance. When there are more features than hidden dimensions and the
features are sparse enough, the model does not drop the extra features: it
packs several feature directions into each dimension in near-orthogonal
arrangements (antipodal pairs, pentagons, and other polytopes), accepting
interference that rarely matters because co-activation is rare. The notebook
quantifies this with per-feature norms, interference between directions
(polysemanticity), probability-weighted net interference, and fractional
dimensionality `D_i`, the share of a dimension one feature effectively owns;
the `D_i` sum to at most the number of hidden dimensions actually used.

`inspect_superposition.py` computes the same quantities two ways. The toy
calibration reproduces the notebook's intro sweep under a fixed seed, which
is a ground-truth check on the metrics themselves: dense instances should
keep few near-basis features, sparse instances should represent more
features at fractional dimensionality. The Qwen pass then treats embedding
rows and per-block MLP `down_proj` columns as candidate direction sets.

Boundaries for the Qwen numbers:

- A neuron's write direction is not a feature, and a token embedding row is
  not a feature either. Overlap between write directions is necessary for
  superposition of MLP features, not sufficient evidence for it. Real
  features could be sparser combinations of neurons (the usual sparse
  autoencoder hypothesis) or could live in directions no single column has.
- The toy result is driven by sparsity: superposition pays only when
  co-activation is rare. With `--activations`, per-neuron activity is
  measured as the fraction of prompt-token positions with gate
  pre-activation above zero and used as the notebook's feature probability
  for net interference, alongside mean `m^2` / mean `|m|` of the gated
  scalar. Gate-positive is a definitional choice, not a discovered firing
  threshold; the default corpus is ten short prompts, so frequencies are
  coarse (granularity 1/positions), prompt-set-specific, cover prompt
  positions only, and come from a BF16 pass by default. The embedding set
  still gets no net interference: ten prompts cannot estimate vocabulary
  frequencies. Corpus-scale frequencies are the natural next experiment.
- Dimensionality sums near the hidden size mostly say the direction set
  spans the space at varied norms; read them together with the norm and
  `D_i` distributions, not as a superposition score. A set of random
  directions in high dimensions is also mutually near-orthogonal.
- The gated MLP scalar `SiLU(gate) * up` can flip a column's effective sign;
  these metrics are quadratic and therefore sign-insensitive, which matches
  the toy model's geometry but ignores sign structure in real writes.
- Embedding rows here are also the unembedding (tied weights), so their
  geometry reflects both reading and writing vocabularies.

Treat low-dimensionality, high-polysemanticity concentrations as candidate
places to look with activation-based tools, exactly as with the lens/DLA
diagnostics above: they select hypotheses, they do not confirm circuits.
