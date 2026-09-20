/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.Models.Mlp

/-!
# MLP Deterministic Initialization

GraphSpec chains (`Chain` + `>>>`) have a typed parameter ABI:
each model comes with an explicit type-level list `ps : List Shape` describing the shapes and the
order of its parameter tensors.

For execution/training examples, we often want deterministic (but nontrivial) initialization rather
than “all zeros”. GraphSpec supports that by letting primitives optionally provide a TorchLean
`Layer` (via `Primitive.toLayerM?`), whose `initState` uses occurrence-indexed seeds.

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

open Spec TorchLean
open TorchLean.Tensor

/--
Deterministic init for `Models.mlp` is exactly the concatenation of the two TorchLean `Linear`
initializers.

Seed discipline:

- the first linear weight uses occurrence index `0`,
- the second linear weight uses occurrence index `1`,
- both biases are initialized exactly to zero.
-/
theorem mlp_detInitParams_eq_torchlean_linear_inits
    (inputWidth hiddenWidth outputWidth : Nat) :
    LowerToDAG.Chain.detInitParams?
        (mlp
          (inputWidth := inputWidth)
          (hiddenWidth := hiddenWidth)
          (outputWidth := outputWidth))
    =
    .ok
      (TorchLean.TensorPack.append (α := Float)
        (ss₁ := [[hiddenWidth, inputWidth], [hiddenWidth]])
        (ss₂ := [[outputWidth, hiddenWidth], [outputWidth]])
        (Runtime.Autograd.Model.Layers.linear inputWidth hiddenWidth
          (weightSeed := 0)).initState
        (Runtime.Autograd.Model.Layers.linear hiddenWidth outputWidth
          (weightSeed := 1)).initState)
          := by
  -- Unfold the MLP graph and the deterministic-init traversal.
  simp
    [ mlp
    , LowerToDAG.Chain.detInitParams?
    , LowerToDAG.Chain.detInitParamsFrom
    , Chain.linear, Chain.relu
    , Primitive.linear, Primitive.relu
    ]
  -- Discharge the “ReLU contributes no params” bookkeeping.
  simp [TorchLean.TensorPack.append,
    Runtime.Autograd.Model.Layers.relu]

end Models
end GraphSpec
end NN
