/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Core
public import NN.Spec.Core.TensorReductionShape

/-!
# Tensor Views (API)

Small, shape-preserving tensor views that show up across examples and subsystems.

These helpers are deliberately model-agnostic:
- they are not tied to a particular layer architecture, and
- they avoid pulling in large "example config" modules just to access a view.
-/

@[expose] public section

namespace NN
namespace API

open Spec Tensor

namespace tensor

/-!
## Flattened Prefix View

For quick experiments, it's often useful to treat an arbitrary `source` shape as a flattened
feature vector and keep only the first `takeDim` coordinates. This keeps runnable demos fast
without baking dataset-specific flattening logic into every example.
-/

/--
Flatten each sample in a batch and keep the first `takeDim` entries.

The output is a typed matrix `batch × takeDim`.
-/
def flattenBatchPrefix {α : Type} [Inhabited α]
    (batch takeDim : Nat) {source : Shape}
    (hTake : takeDim ≤ Shape.size source)
    (x : Spec.Tensor α (.dim batch source)) :
    Spec.Tensor α (NN.Tensor.Shape.Mat batch takeDim) :=
  Spec.Tensor.dim (fun bi =>
    let flat := Spec.Tensor.flattenSpec (Spec.getAtSpec x bi)
    Spec.Tensor.dim (fun j =>
      let h : j.val < Shape.size source := Nat.lt_of_lt_of_le j.isLt hTake
      Spec.Tensor.scalar (Spec.Tensor.toScalar (Spec.get flat ⟨j.val, h⟩))))

end tensor

end API
end NN
