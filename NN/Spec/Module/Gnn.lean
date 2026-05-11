/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Gnn
public import NN.Spec.Module.SpecModule

/-!
# Graph layers as `NNModuleSpec`s

`NN/Spec/Layers/Gnn.lean` defines a small GCN-style layer spec:

`H' = A · H · W + b`

This file wraps that forward spec as an `NNModuleSpec` so it can be composed in `SpecChain`
pipelines and carry simple export metadata.
-/

@[expose] public section


namespace Spec

open Tensor
open ModSpec

variable {α : Type} [Context α]

/-- GCN layer wrapper: `(n, inDim) -> (n, outDim)`. -/
def GCNModuleSpec {n inDim outDim : Nat}
  (layer : GCNLayerSpec n inDim outDim α) :
  NNModuleSpec α (.dim n (.dim inDim .scalar)) (.dim n (.dim outDim .scalar)) :=
{ forward := fun x => gcnLayerSpec (α := α) layer x
  kind := "GCN"
  export_func := {
    toPyTorch := s!"GCNLayer(n={n}, in={inDim}, out={outDim})"
    dimensions := (inDim, outDim)
  } }

end Spec

