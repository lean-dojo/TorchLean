/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.FDeriv.Elementwise
public import NN.Proofs.Autograd.Tape.Nodes
public import NN.Proofs.Autograd.Tape.Nodes.Batched
public import NN.Proofs.Autograd.Tape.Nodes.Shape
public import NN.Proofs.Autograd.Tape.Ops.Attention.MultiHeadSelfAttention
public import NN.Proofs.Autograd.Tape.Ops.Attention.ScaledDotProduct
public import NN.Proofs.Autograd.Tape.Ops.Conv.FDeriv
public import NN.Proofs.Autograd.Tape.Ops.Embedding.GatherRows
public import NN.Proofs.Autograd.Tape.Ops.Norm.BatchNormChannelFirst
public import NN.Proofs.Autograd.Tape.Ops.Norm.LayerNorm
public import NN.Proofs.Autograd.Tape.Ops.Recurrent.ElmanCell
public import NN.Proofs.Autograd.Tape.Ops.Transformer.FeedForward
public import NN.Proofs.Autograd.Tape.Ops.Transformer.PostNorm
public import NN.Proofs.Autograd.Tape.Ops.Transformer.ResidualAttention

/-!
# Autograd Proof Coverage

This module is a curated import surface and roadmap for TorchLean's proved reverse-mode
autograd stack. It does not introduce new theorems; it gathers the pieces users should import when
working with the proof-level layer rather than only executable training.

## Primitive coverage

The smooth, deterministic primitive path has three layers:

1. scalar calculus lemmas in `NN.Proofs.Gradients.Activation`,
2. `OpSpecCorrect` / `OpSpecFDerivCorrect` proofs in `NN.Proofs.Autograd.FDeriv.Elementwise`, and
3. tape-node `NodeFDerivCorrect` wrappers in `NN.Proofs.Autograd.Tape.Nodes`.

Current fully smooth elementwise coverage includes:

* `exp`
* `square`
* `sinh`
* `cosh`
* `tanh`
* `sigmoid`
* `softplus`
* tanh-approximate `gelu`
* `silu`
* `safeLog`, assuming `ε > 0`
* `smoothAbs`, assuming `ε > 0`

Pointwise / condition-carrying coverage includes:

* `relu` away from zero
* `elu` away from zero (global differentiability at zero requires `alpha = 1`)
* `abs` away from zero
* raw `log` / `inv` away from zero
* `sqrt` under its positivity/nonzero hypotheses
* `max` / `min` away from ties

These hypotheses are not bookkeeping noise: they are the mathematical reason we avoid claiming that
nonsmooth primitives have ordinary Fréchet derivatives everywhere. Runtime systems such as PyTorch
choose subgradient conventions at kinks; TorchLean can model those conventions, but the classical
`HasFDerivAt` theorem must state the domain condition explicitly.

## Structured block coverage

The larger block proofs are built by composing the tape-node theorems:

* `DGraph.append`, the reusable graph-composition adapter for globally proof-carrying SSA graphs,
  plus `graphFDerivCorrectAtOfCorrect` for reading such graphs at a point before composing with
  domain-sensitive blocks;
* `DGraph.weakenContext`, the reusable adapter for running a proved graph while carrying extra
  unused inputs such as LayerNorm `gamma`/`beta` through an attention or FFN graph;
* fixed-mask dropout infrastructure: seeded randomness is treated as a stop-gradient mask producer,
  while the differentiable training-mode map is a proved fixed diagonal scaling node;
* last-axis softmax/log-softmax;
* dense/matmul/reduction/broadcast/shape nodes;
* scaled dot-product attention;
* unmasked multi-head self-attention;
* residual multi-head self-attention sublayer `x + MHA(x)`;
* residual Transformer feed-forward sublayers, both one-token/vector-shaped and sequence-shaped
  `X + A₂(GELU(A₁X+b₁))+b₂`;
* arbitrary-context LayerNorm as a whole tape node, so a graph can consume an earlier SSA value plus
  carried `gamma`/`beta` without rebuilding the LayerNorm proof each time;
* a single SSA graph and VJP theorem for the first post-norm Transformer sublayer
  `LayerNorm(x + MHA(x), gamma, beta)`;
* a single SSA graph and VJP theorem for the second post-norm Transformer sublayer
  `LayerNorm(X + FFN(X), gamma, beta)`;
* a theorem-level two-sublayer post-norm Transformer encoder bridge, composing the MHA post-norm
  sublayer into the FFN post-norm sublayer with both LayerNorm domain hypotheses stated explicitly;
* post-norm Transformer boundary `residual_stream ↦ LayerNorm(residual_stream, gamma, beta)`;
* a chain-rule bridge `residualThenPostNorm_hasFDerivAt` showing that any differentiable
  residual-producing map composes correctly with the post-norm LayerNorm graph;
* conv2d FDeriv/backward-dot infrastructure;
* LayerNorm and channel-first BatchNorm-like graphs;
* one-step tanh/Elman RNN cell `h' = tanh(W [x; h] + b)`, plus a two-step composition bridge
  that exposes the chain-rule skeleton used by BPTT;
* finite-index gather-row / embedding lookup adjointness (`gather` VJP is scatter-add).

## Remaining model-level proof work

The runtime/API model zoo is broader than the current end-to-end proof zoo. The reusable pieces
above are intentionally the hard foundations, but the following model-level theorems are still open
work rather than already-proved claims:

* a monolithic SSA backprop theorem that chains the two already-proved post-norm sublayer graphs
  into one executable encoder-block graph. The theorem-level Fréchet-derivative bridge is proved;
  the remaining assembly step is graph-level wiring across the first LayerNorm output into the
  second sequence-FFN input, plus threading the second sublayer's parameters through that composed
  context;
* masked/causal attention blocks used by GPT-style decoders;
* full ViT/GPT encoder or decoder stacks, including embeddings and classifier/language-model heads;
* full recurrent/state-space sequence theorems (`RNN`, `GRU`, `LSTM`,
  Mamba/selective-scan-style recurrences). We cover the one-step tanh/Elman cell and the two-step
  composition bridge; the full sequence theorem is the induction over the unroll plus
  `gather`/`scatter` adjoints;
* FNO spectral-convolution training paths, especially fused FFT/spectral-conv backward rules;
* stochastic training-mode operators beyond dropout. Dropout now has the proof-level fixed-mask
  scaling node; future stochastic layers should follow the same split: sampled object is
  stop-gradient data, differentiable map is proved conditionally on that sampled object;
* executable CUDA/cuBLAS/cuDNN/cuFFT kernels themselves. Those are tested and contract-checked
  against specs, but are not C/CUDA proofs inside Lean.

## Trust boundary

These are source-level mathematical theorems about TorchLean specs and the proof tape. CUDA kernels,
cuBLAS/cuDNN/cuFFT, and compiler backends remain engineering trust boundaries. The intended bridge is:
prove the spec/VJP rule here, then test and contract-check each executable fast path against that
spec.

## References

* PyTorch Autograd documentation: https://pytorch.org/docs/stable/autograd.html
* Baydin et al., "Automatic Differentiation in Machine Learning: a Survey", JMLR 2018.
* Griewank and Walther, *Evaluating Derivatives*, SIAM 2008.
* Vaswani et al., "Attention Is All You Need", NeurIPS 2017.
* Ba et al., "Layer Normalization", arXiv:1607.06450.
* Ioffe and Szegedy, "Batch Normalization", ICML 2015.
-/

@[expose] public section
