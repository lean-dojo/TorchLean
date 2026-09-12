/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Analysis.SpecialFunctions.Pow.NNReal
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
public import Mathlib.Data.Sym.Sym2.Init
import Mathlib.Tactic.NormNum.GCD
-- The roadmap prose below is about these modules, not the constants in them, so `lake shake` sees
-- no use and wants them gone. They stay: this curated surface is what puts the composite-op
-- proofs (attention, transformer blocks, the Elman cell) into a typecheck target.
public import NN.Proofs.Autograd.Tape.Ops.Attention.MaskedMultiHeadSelfAttention
public import NN.Proofs.Autograd.Tape.Ops.Attention.MaskedScaledDotProduct
public import NN.Proofs.Autograd.Tape.Ops.Recurrent.ElmanCell
public import NN.Proofs.Autograd.Tape.Ops.Transformer.DecoderBlock
public import NN.Proofs.Autograd.Tape.Ops.Transformer.EncoderBlock
public import NN.Proofs.Autograd.Tape.Ops.Transformer.FeedForward
public import NN.Proofs.Autograd.Tape.Ops.Transformer.PostNorm
public import NN.Proofs.Autograd.Tape.Ops.Transformer.ResidualAttention

/-!
# Autograd Proof Coverage

This module is a curated import surface and roadmap for TorchLean's proved reverse-mode autograd
library. It does not introduce new theorems; it gathers the pieces users should import when working
with the proof-level layer rather than only executable training.

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
* `safeLog`, assuming $\varepsilon>0$
* `smoothAbs`, assuming $\varepsilon>0$

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
* fixed additive-bias scaled dot-product attention
  $\operatorname{softmax}(cQK^{\mathsf T}+\operatorname{bias})V$;
* fixed additive-bias multi-head attention core over split heads;
* unmasked multi-head self-attention;
* residual multi-head self-attention sublayer $x+\operatorname{MHA}(x)$;
* residual Transformer feed-forward sublayers, both one-token/vector-shaped and sequence-shaped,
  $X+A_2(\operatorname{GELU}(A_1X+b_1))+b_2$;
* arbitrary-context LayerNorm as a whole tape node, so a graph can consume an earlier SSA value plus
  carried `gamma`/`beta` without rebuilding the LayerNorm proof each time;
* a single SSA graph and VJP theorem for the first post-norm Transformer sublayer
  $\operatorname{LayerNorm}(x+\operatorname{MHA}(x),\gamma,\beta)$;
* a single SSA graph and VJP theorem for the second post-norm Transformer sublayer
  $\operatorname{LayerNorm}(X+\operatorname{FFN}(X),\gamma,\beta)$;
* a concrete full post-norm Transformer encoder-block SSA graph and VJP theorem, composing MHA,
  the first LayerNorm, sequence FFN, and the second LayerNorm with both LayerNorm domain hypotheses
  stated explicitly;
* a concrete additive-bias decoder-core SSA graph and VJP theorem, composing biased
  split-head attention, attention projection, the first LayerNorm, sequence FFN, and the second
  LayerNorm with both LayerNorm domain hypotheses stated explicitly;
* a projection-to-residual bridge for GPT-style decoder attention, so differentiable Q/K/V front
  ends and merge/residual packers instantiate the masked decoder block without reopening the
  attention proof;
* a more abstract GPT-style post-norm decoder-block differentiability theorem for outer model
  packers that assemble token projections or other front-end context before the decoder core;
* post-norm Transformer boundary
  $r\mapsto\operatorname{LayerNorm}(r,\gamma,\beta)$;
* a chain-rule bridge `residualThenPostNorm_hasFDerivAt` showing that any differentiable
  residual-producing map composes correctly with the post-norm LayerNorm graph;
* rank-general convolution, including the exact Fréchet derivative and the adjoint input, kernel,
  and bias reverse rules;
* LayerNorm and arbitrary-spatial-rank BatchNorm, including an adjointness theorem for the
  executable input, scale, and bias reverse rules;
* one-step tanh/Elman RNN cell
  $h'=\tanh\!\left(W[x;h]+b\right)$, a two-step composition bridge, and an
  arbitrary-length BPTT chain-rule induction over differentiable recurrent transition builders;
* finite-index gather-row / embedding lookup adjointness (`gather` VJP is scatter-add).

## Runtime link coverage

`NN.Proofs.Autograd.Runtime.Link` connects the executable tape engine to the proved reverse-mode
model. It is precise about which backward variant each theorem covers:

* `backwardDenseFrom_lowerGraphToTape_eq_backpropAllCtx` (`Link.BackwardGraph`) and
  `backwardDenseFrom_lowerGraphToTape_adjoint_fderiv` (`Link.FDeriv`) are about
  `Tape.backwardDenseFrom`, the total sweep that runs every node's VJP;
* `backwardDenseAll_eq_backwardDenseFrom` (`Link.BackwardDense`) shows that the sweep the eager
  trainer executes, `Tape.backwardDenseAll` (which skips nodes that never receive a cotangent and
  zero-fills them), returns exactly the `backwardDenseFrom` result from the one-hot seed array on
  any `ZeroPreserving` tape, errors included;
* `lowerGraphToTape_zeroPreserving` (`Link.BackwardDenseGraph`) discharges that hypothesis for
  every lowered proof-carrying graph, using only the local adjointness law (`node_vjp_full_zero`),
  so `backwardDenseAll_lowerGraphToTape_eq_backpropAllCtx` and, over `ℝ`,
  `backwardDenseAll_lowerGraphToTape_adjoint_fderiv` state the executed backward pass equals the
  proved backpropagation and the adjoint of the Fréchet derivative.

These are theorems about the exact tape model at the stated carrier (`ℚ`, `ℝ`, or any commutative
semiring). The `Float` and CUDA executions of the same algorithm remain approximation and
engineering concerns.

## Remaining model-level proof work

The runtime/API model collection is broader than the current end-to-end proof collection. The
reusable pieces above are intentionally the hard foundations, but the following model-level theorems
are still open work rather than already-proved claims:

* one runtime-layout lowering theorem connecting the concrete encoder-block SSA graph here to each
  executable Transformer example wrapper;
* a concrete SSA graph for a GPT decoder block. The block-level theorem and additive-bias attention
  composition theorem are proved; the remaining lowering step is to instantiate the abstract
  `maskedAttentionPack` with the example decoder's projection/split/merge/residual graph;
* full ViT/GPT encoder or decoder stacks, including embeddings and classifier/language-model heads;
* full recurrent/state-space sequence theorems (`RNN`, `GRU`, `LSTM`,
  Mamba/selective-scan-style recurrences). We cover the one-step tanh/Elman cell and the two-step
  composition bridge; the full sequence theorem is the induction over the unroll plus
  `gather`/`scatter` adjoints;
* FNO spectral-convolution training paths, especially fused FFT/spectral-conv backward rules;
* stochastic training-mode operators beyond dropout. Dropout has the proof-level fixed-mask
  scaling node; future stochastic layers should follow the same split: sampled object is
  stop-gradient data, differentiable map is proved conditionally on that sampled object;
* executable CUDA/cuBLAS/cuDNN/cuFFT kernels themselves. Those are tested and contract-checked
  against specs, but are not C/CUDA proofs inside Lean.

## Trust boundary

These are source-level mathematical theorems about TorchLean specs and the proof tape. CUDA
kernels, cuBLAS/cuDNN/cuFFT, and compiler backends remain engineering trust boundaries. The intended
bridge is: prove the spec/VJP rule here, then test and contract-check each executable fast path
against that spec.

## References

* PyTorch Autograd documentation: https://pytorch.org/docs/stable/autograd.html
* Baydin et al., "Automatic Differentiation in Machine Learning: a Survey", JMLR 2018.
* Griewank and Walther, *Evaluating Derivatives*, SIAM 2008.
* Vaswani et al., "Attention Is All You Need", NeurIPS 2017.
* Ba et al., "Layer Normalization", arXiv:1607.06450.
* Ioffe and Szegedy, "Batch Normalization", ICML 2015.
-/

@[expose] public section
