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
