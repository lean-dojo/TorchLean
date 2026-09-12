/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Runtime.Link.Core
public import NN.Runtime.Autograd.Engine.Core.Backward

/-!
# Dense Backward Pass on Leaf Tapes

Lowering an empty graph produces a tape that contains only leaf nodes. This file shows that the
runtime dense reverse loop is the identity on such a tape: every leaf's `backward` returns no
contributions, so each step performs its bookkeeping checks and returns the gradient array
unchanged. This is the `nil` case shared by the `Graph` and `GraphData` backward-link theorems.
-/

@[expose] public section

namespace Proofs
namespace Autograd
namespace Algebra

open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace Graph

open Runtime
open Runtime.Autograd

variable {α : Type} [TorchLean.Storage α] {Γ : List Shape}

/-- The node stored at index `n` of a leaf-only tape is the leaf of the `n`-th input tensor. -/
theorem getNode?_addLeaves_empty (x : TorchLean.TensorPack α Γ) {n : Nat}
    (hn : n < Γ.length) :
    (addLeaves (α := α) (t := Tape.empty) (Γ := Γ) x).getNode? n =
      some (leafNodeOfSomeTensor (α := α)
        ((TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ) x)[n]'(by
          simpa [TorchLean.TensorPack.size_toShapeErasedArray] using hn))) := by
  have hnodes :
      (addLeaves (α := α) (t := Tape.empty) (Γ := Γ) x).nodes =
        (TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ) x).map
          (leafNodeOfSomeTensor (α := α)) := by
    simp [nodes_addLeaves, Tape.empty]
  have hidX : n < (TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ) x).size := by
    simpa [TorchLean.TensorPack.size_toShapeErasedArray] using hn
  simp [Tape.getNode?, hnodes, Array.getElem?_map, Array.getElem?_eq_getElem hidX]

/-- One runtime backward step at a leaf leaves the dense gradient array unchanged. -/
theorem backwardDenseFromStep_addLeaves_empty [Add α]
    (x seed : TorchLean.TensorPack α Γ) {n : Nat} (hn : n < Γ.length) :
    Tape.backwardDenseFromStep (t := addLeaves (α := α) (t := Tape.empty) (Γ := Γ) x)
        (TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ) seed) n =
      .ok (TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ) seed) := by
  have hidSeed :
      n < (TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ) seed).size := by
    simpa [TorchLean.TensorPack.size_toShapeErasedArray] using hn
  have hshape :
      ((TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ) seed)[n]'hidSeed).shape =
        ((TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ) x)[n]'(by
          simpa [TorchLean.TensorPack.size_toShapeErasedArray] using hn)).shape := by
    rw [shape_getElem_toShapeErasedArray seed hn, shape_getElem_toShapeErasedArray x hn]
  simp only [Tape.backwardDenseFromStep, getNode?_addLeaves_empty x hn,
    Array.getElem?_eq_getElem hidSeed, result_pure_eq_ok, result_bind_ok]
  simp [leafNodeOfSomeTensor, hshape, result_bind_ok]
  rfl

/-- The runtime backward loop over a leaf-only tape is the identity on the gradient array. -/
theorem backwardDenseFromLoop_addLeaves_empty [Add α]
    (x seed : TorchLean.TensorPack α Γ) :
    ∀ n, n ≤ Γ.length →
      Tape.backwardDenseFromLoop (t := addLeaves (α := α) (t := Tape.empty) (Γ := Γ) x) n
          (TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ) seed) =
        .ok (TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ) seed)
  | 0, _ => rfl
  | n + 1, hn => by
      have hlt : n < Γ.length := hn
      rw [Tape.backwardDenseFromLoop, backwardDenseFromStep_addLeaves_empty x seed hlt,
        result_bind_ok]
      exact backwardDenseFromLoop_addLeaves_empty x seed n (Nat.le_of_lt hlt)

/-- The dense backward pass over a leaf-only tape returns its seed unchanged. -/
theorem backwardDenseFrom_addLeaves_empty [Add α] (x seed : TorchLean.TensorPack α Γ) :
    Tape.backwardDenseFrom (t := addLeaves (α := α) (t := Tape.empty) (Γ := Γ) x)
        (TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ) seed) =
      .ok (TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ) seed) := by
  have hlen : (addLeaves (α := α) (t := Tape.empty) (Γ := Γ) x).nodes.size = Γ.length := by
    simp [size_addLeaves, Tape.empty]
  have hsize :
      (TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ) seed).size =
        (addLeaves (α := α) (t := Tape.empty) (Γ := Γ) x).nodes.size := by
    simp [hlen, TorchLean.TensorPack.size_toShapeErasedArray]
  unfold Tape.backwardDenseFrom
  rw [if_pos hsize, hlen]
  exact backwardDenseFromLoop_addLeaves_empty x seed Γ.length le_rfl

end Graph

end Algebra
end Autograd
end Proofs
