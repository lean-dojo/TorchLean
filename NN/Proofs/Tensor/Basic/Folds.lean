/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Algebra.BigOperators.Fin
public import Mathlib.Data.Real.Basic
public import NN.Proofs.Tensor.Algebra
public import NN.Spec.Core.TensorReductionShape.Reductions
public import NN.Proofs.Tensor.Basic.Core -- shake: keep

/-!
Fold and reduction lemmas for dependent tensors.

This module packages the algebra needed to reason about tensor reductions, finite sums, and
shape-indexed traversals.
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

/-- Elementwise multiplication of two all-zero tensors is the all-zero tensor. -/
theorem mul_spec_full_zero {s : Shape} :
    mulSpec (Tensor.full s (0 : ℝ)) (Tensor.full s (0 : ℝ)) = Tensor.full s (0 : ℝ) := by
  apply TorchLean.Tensor.Internal.Rep.ext
  intro coordinate
  simp [mulSpec, map2Spec, Tensor.full]

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

/--
One-step unfolding of the internal tail-recursive helper `foldlSpec.go` when the loop
condition holds (`k < n`).

This lemma is a proof tool: it lets proofs *peel one loop step* without using `unfold` directly.
-/
theorem foldlSpec_go_of_lt {α β : Type} [TorchLean.Storage α]
    (f : β → α → β)
    {n : Nat} {s : Shape} (values : Fin n → Tensor α s) {k : Nat} (acc : β) (hk : k < n) :
    foldlSpec.go f n s values k acc =
      foldlSpec.go f n s values (k + 1) (foldlSpec f acc (values ⟨k, hk⟩)) := by
  rw [foldlSpec.go, foldlSpec.go,
    List.drop_eq_getElem_cons (by simpa using hk)]
  simp

/--
One-step unfolding of the internal tail-recursive helper `foldlSpec.go` when the loop
condition fails (`¬ k < n`), i.e. the loop terminates and returns the accumulator.
-/
theorem foldlSpec_go_of_not_lt {α β : Type} [TorchLean.Storage α]
    (f : β → α → β)
    {n : Nat} {s : Shape} (values : Fin n → Tensor α s) {k : Nat} (acc : β) (hk : ¬ k < n) :
    foldlSpec.go f n s values k acc = acc := by
  have hLength : (List.finRange n).length ≤ k := by
    simpa using Nat.le_of_not_gt hk
  rw [foldlSpec.go, List.drop_eq_nil_of_le hLength]
  rfl

/--
Accumulator lemma for `foldlSpec` specialized to addition.

Informally: folding with `(+)` over a tensor adds `sum_spec t` to the initial accumulator.
This is frequently used to move between “fold-style” specs and “sum-style” algebra.
-/
theorem foldlSpec_add_init {s : Shape} (acc : ℝ) (t : Tensor ℝ s) :
    foldlSpec (· + ·) acc t = acc + sumSpec t := by
  change t.foldl (· + ·) acc = acc + t.foldl (· + ·) 0
  rw [TorchLean.Tensor.Internal.Rep.foldl_eq_data_foldl,
    TorchLean.Tensor.Internal.Rep.foldl_eq_data_foldl,
    ← Array.foldl_toList, ← Array.foldl_toList]
  exact List.foldl_add_init _ _ _


-- Rewriting lemma under dot using associativity/commutativity
/-- Reassociate a `dot` over a pointwise product, using commutativity/associativity of `mulSpec`.
  -/
theorem dot_mul_reassoc {s : Shape}
  (dLdy m dx : Tensor ℝ s) :
  dot dLdy (mulSpec m dx) = dot (mulSpec m dLdy) dx := by
  have hAssoc := mul_spec_assoc (a := dLdy) (b := m) (c := dx)
  have hComm := mul_spec_comm (a := dLdy) (b := m)
  -- `mul_spec dLdy (mul_spec m dx) = mul_spec (mul_spec dLdy m) dx`
  -- and `mul_spec (mul_spec dLdy m) dx = mul_spec (mul_spec m dLdy) dx`.
  simp [dot, hAssoc, hComm]

/--
Coordinate formula for `matVecMulSpec`, converted from the spec's `List.finRange` fold to a
`Finset.univ.sum`.

This is the “PyTorch-looking” statement of matvec: each output entry is a dot product of the
corresponding row with the input vector.
-/
theorem getScalar_mat_vec_mul_spec {m n : Nat}
  (A : Tensor ℝ [m, n])
  (v : Tensor ℝ [n]) (i : Fin m) :
  getScalar (matVecMulSpec A v) i = ∑ k : Fin n, (get2 A i k) * (getScalar v k) := by
  -- Reuse the backend-generic lemma from `NN/Proofs/Tensor/Algebra.lean` (instantiated at `ℝ`).
  simpa using
    (Proofs.TensorAlgebra.getScalar_mat_vec_mul_spec (α := ℝ) (A := A) (v := v) (i := i))

/--
Coordinate formula for `matMulSpec` (matrix-matrix multiplication).

This is the standard triple-sum identity: `(A @ B)[i,j] = ∑ k, A[i,k] * B[k,j]`, matching the
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
The real-analysis `Spec.dot` agrees with the backend-generic recursive dot.

`Spec.dot` is defined as `sumSpec (mulSpec a b)`, which is the proof layer version of the
PyTorch idiom `(a * b).sum()`.  `Proofs.TensorAlgebra.dot` is recursive over the tensor shape so it
works over arbitrary semiring-like scalar models.  This bridge lets real proofs reuse generic
algebra instead of repeating finite-sum rearrangements.
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
