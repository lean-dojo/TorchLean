/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Tensor.Basic.Core -- shake: keep

/-!
Fold and reduction lemmas for dependent tensors.

This module packages the algebra needed to reason about tensor reductions, finite sums, and
shape-indexed traversals. The finite-sum and associativity results concern exact real arithmetic;
they do not permit reassociating a floating-point reduction.
-/

@[expose] public section

open TorchLean

namespace Spec

open TorchLean TorchLean.Tensor
open scoped BigOperators

/-- Elementwise multiplication is associative (`mulSpec` is pointwise `(*)`). -/
theorem mul_spec_assoc {s : Shape} (a b c : Tensor ℝ s) :
  mulSpec a (mulSpec b c) = mulSpec (mulSpec a b) c := by
  apply TorchLean.Tensor.Internal.Rep.ext
  intro coordinate
  simp [mulSpec, map2Spec, mul_assoc]

/-- Elementwise multiplication is commutative (`mulSpec` is pointwise `(*)`). -/
theorem mul_spec_comm {s : Shape}
  (a b : Tensor ℝ s) : mulSpec a b = mulSpec b a := by
  apply TorchLean.Tensor.Internal.Rep.ext
  intro coordinate
  simp [mulSpec, map2Spec, mul_comm]

/-- Elementwise addition is commutative (`addSpec` is pointwise `(+)`). -/
theorem add_spec_comm {s : Shape}
  (a b : Tensor ℝ s) : addSpec a b = addSpec b a := by
  apply TorchLean.Tensor.Internal.Rep.ext
  intro coordinate
  simp [addSpec, map2Spec, add_comm]

/-! ## Real dot product and fold bridges -/

/--
Real dot product for same-shape tensors, defined as the sum of elementwise products.

This matches the common PyTorch idiom `(a * b).sum()` for same-shape tensors.
Citations:
https://pytorch.org/docs/stable/generated/torch.sum.html

This is the `Spec`-namespace dot product used by real-analysis proofs. The backend-generic
recursive dot product is `Proofs.TensorAlgebra.dot`.
-/
noncomputable def dot {s : Shape} (a b : Tensor ℝ s) : ℝ :=
  sumSpec (mulSpec a b)

/-- Peel one slice from the proof-facing fold starting at `k`, when `k < n`. -/
theorem foldlSpec_go_of_lt {α β : Type} [TorchLean.Storage α]
    (f : β → α → β)
    {n : Nat} {s : Shape} (values : Fin n → Tensor α s) {k : Nat} (acc : β) (hk : k < n) :
    foldlSpec.go f n s values k acc =
      foldlSpec.go f n s values (k + 1) (foldlSpec f acc (values ⟨k, hk⟩)) := by
  rw [foldlSpec.go, foldlSpec.go,
    List.drop_eq_getElem_cons (by simpa using hk)]
  simp

/-- Starting the proof-facing slice fold at or beyond its length returns the accumulator. -/
theorem foldlSpec_go_of_not_lt {α β : Type} [TorchLean.Storage α]
    (f : β → α → β)
    {n : Nat} {s : Shape} (values : Fin n → Tensor α s) {k : Nat} (acc : β) (hk : ¬ k < n) :
    foldlSpec.go f n s values k acc = acc := by
  have hLength : (List.finRange n).length ≤ k := by
    simpa using Nat.le_of_not_gt hk
  rw [foldlSpec.go, List.drop_eq_nil_of_le hLength]
  rfl

/--
Coordinate formula for `matMulSpec` (matrix-matrix multiplication).

This is the coordinate identity `(A @ B)[i,j] = ∑ k, A[i,k] * B[k,j]`, matching the
textbook/PyTorch view of matrix multiplication.

Citations:
https://pytorch.org/docs/stable/generated/torch.matmul.html
-/
theorem get2_mat_mul_spec {m n p : Nat}
  (A : Tensor ℝ [m, n])
  (B : Tensor ℝ [n, p]) (i : Fin m) (j : Fin p) :
  get2 (matMulSpec A B) i j = ∑ k : Fin n, (get2 A i k) * (get2 B k j) := by
  rw [Proofs.TensorAlgebra.get2_eq]
  simp only [Tensor.unstack, Tensor.item, matMulSpec,
    TorchLean.Tensor.Internal.Rep.unstack_apply, TorchLean.Tensor.Internal.Rep.get_ofFn]
  change
    (List.finRange n).foldl
        (fun total k => total + get2 A i k * get2 B k j) 0 =
      ∑ k : Fin n, get2 A i k * get2 B k j
  exact List.finRange_foldl_add_eq_finset_sum _

/-- A packed-buffer sum is the finite sum of the tensor's coordinate observations. -/
theorem sum_spec_eq_coord_sum {s : Shape} (t : Tensor ℝ s) :
    sumSpec t = ∑ coordinate : s.Coord, t coordinate := by
  classical
  have hFlat :
      t.data.toList.sum =
        ∑ index : Fin (TorchLean.Tensor.Internal.Shape.size s.toList), t.getFlat index := by
    have hSize := TorchLean.Tensor.Internal.Rep.data_size t
    calc
      t.data.toList.sum =
          ∑ index : Fin t.data.size, t.data.toList[index.val] := by
        simpa only [Array.length_toList] using
          (Fin.sum_univ_getElem t.data.toList).symm
      _ = ∑ index : Fin (TorchLean.Tensor.Internal.Shape.size s.toList), t.getFlat index := by
        apply Fintype.sum_equiv (finCongr hSize)
        intro index
        rw [Array.getElem_toList]
        simpa using
          (TorchLean.Tensor.Internal.Rep.data_getFlat t (Fin.cast hSize index))
  calc
    sumSpec t = t.data.toList.sum := by
      rw [sumSpec, foldlSpec,
        TorchLean.Tensor.Internal.Rep.foldl_eq_data_foldl, ← Array.foldl_toList,
        ← List.sum_eq_foldl]
    _ = ∑ index : Fin (TorchLean.Tensor.Internal.Shape.size s.toList), t.getFlat index := hFlat
    _ = ∑ coordinate : s.Coord, t coordinate := by
      symm
      simpa [TorchLean.Tensor.Internal.Rep.get, TorchLean.Tensor.Internal.Coord.linearize] using
        ((TorchLean.Tensor.Internal.Coord.equivFin s.toList).sum_comp
          (fun index => t.getFlat index))

/--
Sum over the outer dimension unfolds into a `Finset.univ` sum of inner `sumSpec`.

This is the tensor analogue of `torch.sum` reducing over a leading dimension.
-/
theorem sum_spec_dim {n : Nat} {s : Shape} (t : Tensor ℝ (.dim n s)) :
  sumSpec t = ∑ i : Fin n, sumSpec (get t i) := by
  classical
  rw [sum_spec_eq_coord_sum, Fintype.sum_prod_type]
  apply Finset.sum_congr rfl
  intro i _
  rw [sum_spec_eq_coord_sum]
  apply Finset.sum_congr rfl
  intro coordinate _
  exact (TorchLean.Tensor.Internal.Rep.unstack_apply t i coordinate).symm

/--
The packed real sum-of-products agrees with the generic shape-recursive dot product.

This bridge allows real-analysis proofs to reuse generic dot-product algebra. The equality uses
exact addition; the two groupings need not agree for a rounded scalar model.
-/
theorem dot_eq_tensorAlgebra_dot {s : Shape} (a b : Tensor ℝ s) :
    dot a b = Proofs.TensorAlgebra.dot (α := ℝ) a b := by
  induction s with
  | scalar =>
      rw [dot, sum_spec_eq_coord_sum]
      simp [Proofs.TensorAlgebra.dot, mulSpec, map2Spec, Tensor.item,
        TorchLean.Tensor.Internal.Rep.zipWith_apply]
  | dim n s ih =>
      rw [dot, sum_spec_dim]
      change
        (∑ i : Fin n, sumSpec (get (mulSpec a b) i)) =
          (List.finRange n).foldl
            (fun total i =>
              total + Proofs.TensorAlgebra.dot (a.unstack i) (b.unstack i)) 0
      rw [List.finRange_foldl_add_eq_finset_sum]
      apply Finset.sum_congr rfl
      intro i _
      have hSlice :
          get (mulSpec a b) i = mulSpec (a.unstack i) (b.unstack i) := by
        apply TorchLean.Tensor.Internal.Rep.ext
        intro coordinate
        simp [get, Tensor.unstack, mulSpec, map2Spec]
      rw [hSlice]
      simpa [dot] using ih (a := a.unstack i) (b := b.unstack i)

end Spec
