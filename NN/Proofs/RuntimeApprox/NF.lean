/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.NF.BackwardOps
public import NN.Proofs.RuntimeApprox.NF.Attention
public import NN.Proofs.RuntimeApprox.NF.Convolution
public import NN.Proofs.RuntimeApprox.NF.EndToEnd
public import NN.Proofs.RuntimeApprox.NF.FoldLemmas
public import NN.Proofs.RuntimeApprox.NF.Linalg
public import NN.Proofs.RuntimeApprox.NF.Normalization
public import NN.Proofs.RuntimeApprox.NF.Ops
public import NN.Proofs.RuntimeApprox.NF.Optimizers
public import NN.Proofs.RuntimeApprox.NF.ReductionOps
public import NN.Proofs.RuntimeApprox.NF.ShapeOps
public import NN.Proofs.RuntimeApprox.NF.SoftmaxAxis

/-!
# NF Runtime Approximation Backend

Proof-relevant rounded tensor/operator approximation for `NF`.

`NF` wraps real values and inserts an explicit `neuralRound` step after primitive arithmetic. The
modules collected here prove local bounds for elementwise ops, reductions, shape-only ops, linear
algebra, attention, normalization, optimizers, and graph-level end-to-end execution.

File roles:
- `Ops`: scalar and elementwise tensor bounds, plus primitive `FwdNode` constructors.
- `Linalg`: matrix/vector and matrix/matrix multiplication bounds.
- `ReductionOps`: row/column reductions used by normalization and attention.
- `ShapeOps`: value-preserving tensor rearrangements such as replication/broadcasting.
- `BackwardOps`: VJP bounds and `RevNode` constructors for reverse-mode composition.
- `SoftmaxAxis`: stable last-axis softmax and its rounded VJP.
- `Attention`: scaled-dot-product attention as one composition of the shared operator contracts.
- `Convolution`: ordered forward and backward bounds for arbitrary spatial rank.
- `Normalization`: rank-generic affine-normalization traces with explicit denominator margins.
- `Optimizers`: SGD, momentum-SGD, and AdamW instances of one numerical optimizer contract.
- `EndToEnd`: architecture-independent executable graph bridges, parameter updates, and reports.
- `Utils`: shared list-fold and tensor approximation helpers.

This is the backend we can reason about inside Lean. Hardware CUDA/IEEE execution remains an
implementation trust boundary unless it is connected to this model by a separately proved or
certified semantics.
-/

@[expose] public section
