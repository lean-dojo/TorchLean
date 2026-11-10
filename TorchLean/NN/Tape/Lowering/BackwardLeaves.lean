/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Tape.Lowering
public import TorchLean.NN.Tape.Backward

/-!
# Dense Backward Pass on Leaf Tapes

Lowering an empty graph produces a tape that contains only leaf nodes, and on such a tape the
runtime dense reverse loop is the identity. Every leaf's `backward` returns no contributions, so
each step performs its bookkeeping checks and hands back the gradient array unchanged. The result
is the `nil` case of `GraphData.backwardDenseFrom_toTape`, which the `Graph`
version reuses through `Graph.toData`.
-/

@[expose] public section

namespace TorchLean
namespace Tape

open TorchLean.Tensor

variable {α : Type} [Storage α] {Γ : List Shape}

/-- The node stored at index `n` of a leaf-only tape is the leaf of the `n`-th input tensor. -/
theorem getNode?_addLeaves_empty (x : TensorPack α Γ) {n : Nat}
    (hn : n < Γ.length) :
    (addLeaves (α := α) (t := Tape.empty) (Γ := Γ) x).getNode? n =
      some (Node.leaf
        ((TensorPack.erase (α := α) (ss := Γ) x)[n]'(by
          simpa [TensorPack.size_erase] using hn))) := by
  have hnodes :
      (addLeaves (α := α) (t := Tape.empty) (Γ := Γ) x).nodes =
        (TensorPack.erase (α := α) (ss := Γ) x).map Node.leaf := by
    simp [nodes_addLeaves, Tape.empty]
  have hidX : n < (TensorPack.erase (α := α) (ss := Γ) x).size := by
    simpa [TensorPack.size_erase] using hn
  simp [Tape.getNode?, hnodes, Array.getElem?_map, Array.getElem?_eq_getElem hidX]

/-- One runtime backward step at a leaf leaves the dense gradient array unchanged. -/
theorem backwardDenseFromStep_addLeaves_empty [Add α]
    (x seed : TensorPack α Γ) {n : Nat} (hn : n < Γ.length) :
    Tape.backwardDenseFromStep (t := addLeaves (α := α) (t := Tape.empty) (Γ := Γ) x)
        (TensorPack.erase (α := α) (ss := Γ) seed) n =
      .ok (TensorPack.erase (α := α) (ss := Γ) seed) := by
  have hidSeed :
      n < (TensorPack.erase (α := α) (ss := Γ) seed).size := by
    simpa [TensorPack.size_erase] using hn
  have hshape :
      ((TensorPack.erase (α := α) (ss := Γ) seed)[n]'hidSeed).shape =
        ((TensorPack.erase (α := α) (ss := Γ) x)[n]'(by
          simpa [TensorPack.size_erase] using hn)).shape := by
    rw [TensorPack.shape_getElem_erase seed hn, TensorPack.shape_getElem_erase x hn]
  simp only [GradValue.shape_someTensor, GradValue.retag_someTensor, GradValue.check_someTensor,
      Tape.backwardDenseFromStep, getNode?_addLeaves_empty x hn, Array.getElem?_eq_getElem hidSeed,
      Result.pure_eq_ok, Result.bind_ok]
  simp [Node.leaf, hshape, Result.bind_ok]
  rfl

/-- The runtime backward loop over a leaf-only tape is the identity on the gradient array. -/
theorem backwardDenseFromLoop_addLeaves_empty [Add α]
    (x seed : TensorPack α Γ) :
    ∀ n, n ≤ Γ.length →
      Tape.backwardDenseFromLoop (t := addLeaves (α := α) (t := Tape.empty) (Γ := Γ) x) n
          (TensorPack.erase (α := α) (ss := Γ) seed) =
        .ok (TensorPack.erase (α := α) (ss := Γ) seed)
  | 0, _ => rfl
  | n + 1, hn => by
      have hlt : n < Γ.length := hn
      rw [Tape.backwardDenseFromLoop, backwardDenseFromStep_addLeaves_empty x seed hlt,
        Result.bind_ok]
      exact backwardDenseFromLoop_addLeaves_empty x seed n (Nat.le_of_lt hlt)

/-- The dense backward pass over a leaf-only tape returns its seed unchanged. -/
theorem backwardDenseFrom_addLeaves_empty [Add α] (x seed : TensorPack α Γ) :
    Tape.backwardDenseFrom (t := addLeaves (α := α) (t := Tape.empty) (Γ := Γ) x)
        (TensorPack.erase (α := α) (ss := Γ) seed) =
      .ok (TensorPack.erase (α := α) (ss := Γ) seed) := by
  have hlen : (addLeaves (α := α) (t := Tape.empty) (Γ := Γ) x).nodes.size = Γ.length := by
    simp [size_addLeaves, Tape.empty]
  have hsize :
      (TensorPack.erase (α := α) (ss := Γ) seed).size =
        (addLeaves (α := α) (t := Tape.empty) (Γ := Γ) x).nodes.size := by
    simp [hlen, TensorPack.size_erase]
  unfold Tape.backwardDenseFrom
  rw [ite_eq_left hsize, hlen]
  exact backwardDenseFromLoop_addLeaves_empty x seed Γ.length le_rfl

end Tape
end TorchLean
