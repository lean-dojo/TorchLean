/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.Models.Mlp

/-!
# MLP Deterministic Initialization

GraphSpec sequential graphs (`Graph` + `>>>`) have a *typed parameter ABI*:
each model comes with an explicit type-level list `ps : List Shape` describing the shapes and the
order of its parameter tensors.

For execution/training examples, we often want deterministic (but nontrivial) initialization rather
than “all zeros”. GraphSpec supports that by letting primitives optionally provide a TorchLean
`LayerDef` (via `Primitive.toLayerDefM?`), whose `initParams` uses occurrence-indexed seeds.

This file proves one concrete bridge theorem: for the 2-layer MLP
`Models.mlp`, the deterministic initialization obtained from GraphSpec is exactly the same typed
parameter list you would get by initializing the two TorchLean `Linear` layers directly with the
expected occurrence-indexed seeds.

That matters because GraphSpec models expose parameters by a typed ABI, not by mutable module
fields. This theorem checks that the ABI order used by GraphSpec initialization agrees with the
runtime layer order used by TorchLean.
-/

@[expose] public section


namespace NN
namespace GraphSpec
namespace Models

open Spec
open Tensor
open NN.Tensor

open Runtime.Autograd.Torch (TList)

/--
Deterministic init for `Models.mlp` is exactly the concatenation of the two TorchLean `Linear`
initializers.

Seed discipline:

- first linear layer uses occurrence index `0`, hence seeds `(0, 1)`,
- second linear layer uses occurrence index `1`, hence seeds `(2, 3)`.
-/
theorem mlp_detInitParams_eq_torchlean_linear_inits
    (inDim hidDim outDim : Nat) :
    LowerToDAG.Graph.detInitParams?
        (mlp (inDim := inDim) (hidDim := hidDim) (outDim := outDim))
    =
    .ok
      (Runtime.Autograd.Torch.Proofs.Autograd.Algebra.TList.append (α := Float)
        (ss₁ := [Shape.Mat hidDim inDim, Shape.Vec hidDim])
        (ss₂ := [Shape.Mat outDim hidDim, Shape.Vec outDim])
        (Runtime.Autograd.TorchLean.NN.linear inDim hidDim (seedW := 0) (seedB := 1)).initParams
        (Runtime.Autograd.TorchLean.NN.linear hidDim outDim (seedW := 2) (seedB := 3)).initParams)
          := by
  -- Unfold the MLP graph and the deterministic-init traversal.
  simp
    [ mlp
    , LowerToDAG.Graph.detInitParams?
    , LowerToDAG.Graph.detInitParamsAux
    , Graph.linear, Graph.relu
    , Primitive.linear, Primitive.relu
    ]
  -- Discharge the “ReLU contributes no params” bookkeeping.
  simp [Runtime.Autograd.Torch.Proofs.Autograd.Algebra.TList.append,
    Runtime.Autograd.TorchLean.NN.relu]

end Models
end GraphSpec
end NN
