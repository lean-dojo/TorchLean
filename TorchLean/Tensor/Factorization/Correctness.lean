/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import Mathlib.Data.List.GetD
public import TorchLean.Tensor.Factorization
public import TorchLean.Tensor.Context.Real
public import TorchLean.Tensor.ShapeChange.Flatten -- shake: keep

/-!
# Cholesky and QR factorization predicates

`IsCholesky` and `IsQR` state the factorization properties using mathlib matrices. The fold-indexing
lemmas track the columns appended by the Cholesky recurrence, and
`cholesky_lower_triangular` shows that the Cholesky recurrence sets above-diagonal entries to
zero without a pivot assumption.

`TorchLean.Tensor.Factorization.Reconstruction` proves Cholesky reconstruction under symmetry and
positive pivots, and QR reconstruction under positive pivots.
`TorchLean.Tensor.Factorization.Orthonormal` supplies QR orthonormality, and
`TorchLean.Tensor.Factorization.Solve` proves that the Cholesky solves return exact solutions. These
are exact real-arithmetic results about the specification bodies. They do not establish
floating-point stability. Compiled code runs array versions of `choleskyColsFn`, `cholSolveFn` and
`solveRidgeFn`, which `@[csimp]` theorems in the spec file prove equal to these definitions for
every scalar type.
-/

@[expose] public section

namespace TorchLean.Factorization

open Matrix
open scoped BigOperators

variable {n : Nat}

/-! ## Specifications

`Factorization.IsCholesky` and `Factorization.IsQR` state each factorization as a predicate on real
matrices. Over `ℝ`, `star = id`, so `conjTranspose = transpose`, and we write both predicates with
`ᵀ`.
-/

/-- `L` is a Cholesky factor of `A`: lower-triangular with `A = L · Lᵀ`. -/
def IsCholesky (A L : Matrix (Fin n) (Fin n) ℝ) : Prop :=
  (∀ i j, i < j → L i j = 0) ∧ A = L * Lᵀ

/-- `(Q, R)` is a QR factorization of `A`: `Q` has orthonormal columns, `R` is upper-triangular,
`A = Q · R`. -/
def IsQR {m k : Nat} (A Q : Matrix (Fin m) (Fin k) ℝ) (R : Matrix (Fin k) (Fin k) ℝ) : Prop :=
  Qᵀ * Q = 1 ∧ (∀ i j, j < i → R i j = 0) ∧ A = Q * R

/-! ### Fold-indexing for the column-building specs

`choleskyColsFn` builds its output with a left fold that appends one column per index. The lemmas
here read off the column produced at a given position, bridging the executable `List.foldl` form to
per-entry reasoning. They are generic over the appended-value function `g`. -/

section FoldSnoc

variable {β : Type*} {ι : Type*}

/-- A left fold that appends one element per input grows the accumulator by `l.length`. -/
theorem length_foldl_snoc (g : List β → ι → β) (l : List ι) (acc : List β) :
    (l.foldl (fun s a => s ++ [g s a]) acc).length = acc.length + l.length := by
  induction l generalizing acc with
  | nil => simp
  | cons a t ih =>
      rw [List.foldl_cons, ih]
      simp only [List.length_append, List.length_cons, List.length_nil]
      grind

/-- A fold that only appends leaves every element already in the accumulator unchanged. -/
theorem getD_foldl_snoc_lt (g : List β → ι → β) (d : β) (l : List ι) (acc : List β)
    (k : Nat) (hk : k < acc.length) :
    (l.foldl (fun s a => s ++ [g s a]) acc).getD k d = acc.getD k d := by
  induction l generalizing acc with
  | nil => simp
  | cons a t ih =>
      rw [List.foldl_cons,
        ih (acc ++ [g acc a]) (by rw [List.length_append]; grind),
        List.getD_append _ _ _ _ hk]

/-- The element at position `k` of the snoc-fold over an arbitrary list `l` is `g` applied to the
fold of the length-`k` prefix and the `k`-th element. -/
theorem getD_foldl_snoc_read (g : List β → ι → β) (d : β) (l : List ι) (k : Nat)
    (hk : k < l.length) :
    (l.foldl (fun s a => s ++ [g s a]) []).getD k d
      = g ((l.take k).foldl (fun s a => s ++ [g s a]) []) (l[k]'hk) := by
  have htake : l.take (k + 1) = l.take k ++ [l[k]'hk] := List.take_succ_eq_append_getElem hk
  have hplen : ((l.take k).foldl (fun s a => s ++ [g s a]) []).length = k := by
    rw [length_foldl_snoc, List.length_nil, List.length_take, Nat.zero_add,
      Nat.min_eq_left (le_of_lt hk)]
  calc
    (l.foldl (fun s a => s ++ [g s a]) []).getD k d
        = ((l.drop (k + 1)).foldl (fun s a => s ++ [g s a])
            ((l.take (k + 1)).foldl (fun s a => s ++ [g s a]) [])).getD k d := by
          conv_lhs => rw [show l = l.take (k + 1) ++ l.drop (k + 1) from
            (List.take_append_drop _ _).symm]
          rw [List.foldl_append]
    _ = ((l.take (k + 1)).foldl (fun s a => s ++ [g s a]) []).getD k d := by
          apply getD_foldl_snoc_lt
          rw [length_foldl_snoc, List.length_nil, List.length_take, Nat.zero_add]
          grind
    _ = g ((l.take k).foldl (fun s a => s ++ [g s a]) []) (l[k]'hk) := by
          rw [htake, List.foldl_append, List.foldl_cons, List.foldl_nil]
          rw [List.getD_append_right _ _ _ _ (le_of_eq hplen), hplen, Nat.sub_self]
          rfl

/-- The element at position `j` of the snoc-fold over `finRange n` is `g` applied to the fold of the
length-`j` prefix and the index `j`. -/
theorem getD_foldl_finRange (g : List β → Fin n → β) (d : β) (j : Fin n) :
    ((List.finRange n).foldl (fun s a => s ++ [g s a]) []).getD j.val d
      = g (((List.finRange n).take j.val).foldl (fun s a => s ++ [g s a]) []) j := by
  have hjlen : j.val < (List.finRange n).length := by
    rw [List.length_finRange]; exact j.isLt
  have hj : (List.finRange n)[j.val]'hjlen = j := by simp [List.getElem_finRange]
  rw [getD_foldl_snoc_read g d (List.finRange n) j.val hjlen, hj]

end FoldSnoc

/-! ### Cholesky factor is lower-triangular

Each column of the executable `choleskyFn` is `0` above the diagonal by construction, so the
factor is lower-triangular for every input matrix, with no pivot hypothesis. -/

/-- Reading an entry of a matrix tensor returns the underlying function value. -/
theorem _root_.TorchLean.Tensor.get2_matrix {m k : Nat} (f : Fin m → Fin k → ℝ) (i : Fin m)
    (j : Fin k) :
    Tensor.get2 (Tensor.matrix f) i j = f i j := by
  simp [Tensor.matrix, Tensor.get2]

/-- The executable Cholesky factor is lower-triangular: entries strictly above the diagonal
vanish. -/
theorem choleskyFn_lower_triangular (A : Fin n → Fin n → ℝ) {i j : Fin n} (hij : i.val < j.val) :
    Factorization.choleskyFn A i j = 0 := by
  unfold Factorization.choleskyFn Factorization.choleskyColsFn
  rw [getD_foldl_finRange]
  rw [ite_eq_left hij]

/-- Tensor-level statement: the Cholesky factor `Tensor.cholesky A` is lower-triangular. -/
theorem _root_.TorchLean.Tensor.cholesky_lower_triangular (A : Tensor ℝ [n, n])
    {i j : Fin n} (hij : i.val < j.val) :
    Tensor.get2 A.cholesky i j = 0 := by
  rw [show A.cholesky =
      Tensor.matrix (Factorization.choleskyFn (Tensor.get2 A)) from rfl,
    Tensor.get2_matrix]
  exact choleskyFn_lower_triangular _ hij

end TorchLean.Factorization
