/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Normalization
public import NN.Spec.Module.SpecModule

/-!
# Normalization module wrappers

This file wraps selected normalization specs as `NNModuleSpec`s for composition/export.

For `LayerNormModuleSpec`, we require (defaulted) proofs that dimensions are positive. This matches
the spec-level intent: normalization divides by the number of features and uses variance/standard
 deviation, so degenerate "zero-width" cases are excluded when we want clean theorems.

PyTorch mental picture: `nn.LayerNorm(embedDim)` applied at each timestep, with `weight=gamma` and
`bias=beta`.
-/

@[expose] public section


namespace Spec

open Tensor
open ModSpec

variable {α : Type} [Context α]

/-- LayerNorm over the last dimension, wrapped as an `NNModuleSpec`. -/
def LayerNormModuleSpec (seqLen embedDim : Nat)
  (gamma : Tensor α (.dim embedDim .scalar))
  (beta  : Tensor α (.dim embedDim .scalar))
  (h_seq_pos : seqLen > 0)
  (h_embed_pos : embedDim > 0) :
  NNModuleSpec α (.dim seqLen (.dim embedDim .scalar)) (.dim seqLen (.dim embedDim .scalar)) :=
{ forward := fun x =>
    layerNorm (α := α) (seqLen := seqLen) (embedDim := embedDim) x gamma beta h_seq_pos h_embed_pos,
  kind := "LayerNorm",
  export_func := {
    -- Normalizes over the last dimension (feature axis).
    toPyTorch := s!"nn.LayerNorm({embedDim})",
    dimensions := (seqLen, embedDim)
  } }

end Spec
