/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.NF.BackwardOps
public import NN.Proofs.RuntimeApprox.NF.Conv
public import NN.Proofs.RuntimeApprox.NF.ConvBackward
public import NN.Proofs.RuntimeApprox.NF.ConvForward
public import NN.Proofs.RuntimeApprox.NF.EndToEnd
public import NN.Proofs.RuntimeApprox.NF.Linalg
public import NN.Proofs.RuntimeApprox.NF.Ops
public import NN.Proofs.RuntimeApprox.NF.ReductionOps
public import NN.Proofs.RuntimeApprox.NF.ShapeOps
public import NN.Proofs.RuntimeApprox.NF.SoftmaxAxis
public import NN.Proofs.RuntimeApprox.NF.Utils

/-!
# NF Runtime Approximation Backend

Proof-relevant rounded tensor/operator approximation for `NF`.

`NF` wraps real values and inserts an explicit `neural_round` step after primitive arithmetic. The
modules collected here prove local bounds for elementwise ops, reductions, shape-only ops, linear
algebra, Conv2D forward/backward, and graph-level end-to-end execution.

File roles:
- `Ops`: scalar and elementwise tensor bounds, plus primitive `FwdNode` constructors.
- `Linalg`: matrix/vector and matrix/matrix multiplication bounds.
- `ReductionOps`: row/column reductions used by normalization and attention.
- `ShapeOps`: value-preserving tensor rearrangements such as replication/broadcasting.
- `BackwardOps`: VJP bounds and `RevNode` constructors for reverse-mode composition.
- `Conv`, `ConvForward`, `ConvBackward`: Conv2D shared error algebra plus forward/backward bounds.
- `EndToEnd`: bridge from proof-level `RevGraph` theorems to executable `GraphData`.
- `SoftmaxAxis`: documentation separating scalar softmax/logistic bounds from vector-valued
  vector-valued axis softmax bounds.
- `Utils`: shared list-fold and tensor approximation helpers.

This is the backend we can reason about inside Lean. Hardware CUDA/IEEE execution remains an
implementation trust boundary unless it is connected to this model by a separately proved or
certified semantics.
-/

@[expose] public section
