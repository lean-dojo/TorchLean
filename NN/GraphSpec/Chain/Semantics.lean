/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.Chain.Syntax

/-!
# Pure semantics of sequential GraphSpec chains

The interpreter splits the typed parameter pack at each sequential composition and applies the
two subchains in order. It is intentionally direct, so proofs about chains need not pass through
their SSA/DAG representation.
-/

@[expose] public section

namespace NN
namespace GraphSpec

open Spec TorchLean

namespace Interp

/-- Pure tensor semantics of a sequential `Chain`. -/
def spec
    {ps : List Shape} {σ τ : Shape}
    (g : Chain ps σ τ)
    {α : Type 0} [TorchLean.Storage α] [Context α] :
    TorchLean.TensorPack α ps → TorchLean.Tensor α σ → TorchLean.Tensor α τ :=
  fun params x =>
    match g with
    | .id _ => x
    | .prim p => p.specFwd (α := α) params x
    | .seq (ps₁ := ps₁) (ps₂ := ps₂) g₁ g₂ =>
        let (params₁, params₂) :=
          TorchLean.TensorPack.split (α := α) (ss₁ := ps₁) (ss₂ := ps₂) params
        let y := spec (α := α) g₁ params₁ x
        spec (α := α) g₂ params₂ y

end Interp

end GraphSpec
end NN
