/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor.Factorizations
public import Mathlib.Data.List.GetD
public import NN.Spec.Core.Context.Real
public import NN.Proofs.Tensor.Basic.Core -- shake: keep

/-!
# Correctness of the exact matrix factorizations (Cholesky and QR)

This file sets up the **formal correctness layer** for the two exact, finite spec-layer
factorizations in `NN.Spec.Core.Tensor.Factorizations` (`choleskySpec`, `qrSpec`): the
factorization predicates over real matrices, and the first structural theorem about the executable
Cholesky factor.

## Architecture (refinement)

* **Specifications** (`IsCholesky`, `IsQR`) are `Prop`s on Mathlib `Matrix (Fin n) (Fin n) ℝ`.
  Mathlib's `Matrix m n α` is *definitionally* `m → n → α`, so the function representation
  `Spec.toMatFn` produced by the executable specs bridges for free.
* **Fold-indexing lemmas** (`length_foldl_snoc`, `getD_foldl_snoc_lt`, `getD_foldl_snoc_read`,
  `getD_foldl_finRange`) read off the column produced at a given position of the left fold that both
  `choleskyColsFn` and `gramSchmidtFn` use to build their output, bridging the executable
  `List.foldl` form to per-entry reasoning. They live here once and are reused by
  `FactorizationsReconstruction`.
* **Structural theorem.** The executable Cholesky factor is lower-triangular
  (`choleskyFn_lower_triangular`, lifted to the tensor level as `choleskySpec_lower_triangular`),
  proved directly from the column fold: the above-diagonal entry is forced to `0` by construction.

## Scope

This file proves only the predicates and the lower-triangularity fact. The exact algebraic
reconstructions, namely `A = L · Lᵀ` for Cholesky (under positive pivots) and `A = Q · R` with
`Qᵀ Q = 1` for Gram–Schmidt (under full column rank), are proved in the companion modules
`NN.Proofs.Tensor.Basic.FactorizationsReconstruction` and
`NN.Proofs.Tensor.Basic.FactorizationsOrthonormal`. Everything here is an exact identity over `ℝ`;
the only hypotheses are the genuine success conditions of the algorithms.
-/

@[expose] public section

namespace Spec.Factorization

open Matrix
open scoped BigOperators

variable {n : Nat}

/-! ## Specifications

The mathematical meaning of each factorization, as a predicate over real matrices. Over `ℝ`,
`star = id` so `conjTranspose = transpose`; we phrase everything with `ᵀ`.
-/

/-- `L` is a Cholesky factor of `A`: lower-triangular with `A = L · Lᵀ`. -/
def IsCholesky (A L : Matrix (Fin n) (Fin n) ℝ) : Prop :=
  (∀ i j, i < j → L i j = 0) ∧ A = L * Lᵀ

/-- `(Q, R)` is a QR factorization of `A`: `Q` has orthonormal columns, `R` is upper-triangular,
`A = Q · R`. -/
def IsQR {m k : Nat} (A Q : Matrix (Fin m) (Fin k) ℝ) (R : Matrix (Fin k) (Fin k) ℝ) : Prop :=
  Qᵀ * Q = 1 ∧ (∀ i j, j < i → R i j = 0) ∧ A = Q * R


/-! ### Fold-indexing for the column-building specs

`choleskyColsFn` and `gramSchmidtFn` build their output with a left fold that appends one column per
index. The lemmas here read off the column produced at a given position, bridging the executable
`List.foldl` form to per-entry reasoning. They are generic over the appended-value function `g` and
are the single home for these snoc-fold read lemmas; `FactorizationsReconstruction` reuses them. -/

section FoldSnoc

variable {β : Type _} {ι : Type _}

/-- A left fold that appends one element per input grows the accumulator by `l.length`. -/
theorem length_foldl_snoc (g : List β → ι → β) (l : List ι) (acc : List β) :
    (l.foldl (fun s a => s ++ [g s a]) acc).length = acc.length + l.length := by
  induction l generalizing acc with
  | nil => simp
  | cons a t ih =>
      rw [List.foldl_cons, ih]
      simp only [List.length_append, List.length_cons, List.length_nil]
      grind

/-- A fold that only appends never changes an index already inside the accumulator. -/
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
  have htake : (List.finRange n).take (j.val + 1)
      = (List.finRange n).take j.val ++ [j] := by
    rw [List.take_succ_eq_append_getElem hjlen]
    congr 1
    simp [List.getElem_finRange]
  have hplen : (((List.finRange n).take j.val).foldl (fun s a => s ++ [g s a]) []).length
      = j.val := by
    rw [length_foldl_snoc, List.length_nil, List.length_take, List.length_finRange, Nat.zero_add,
      Nat.min_eq_left (Nat.le_of_lt j.isLt)]
  calc
    ((List.finRange n).foldl (fun s a => s ++ [g s a]) []).getD j.val d
        = (((List.finRange n).drop (j.val + 1)).foldl (fun s a => s ++ [g s a])
            ((List.finRange n).take (j.val + 1) |>.foldl (fun s a => s ++ [g s a]) [])).getD
              j.val d := by
          conv_lhs => rw [show List.finRange n
            = (List.finRange n).take (j.val + 1) ++ (List.finRange n).drop (j.val + 1) from
            (List.take_append_drop _ _).symm]
          rw [List.foldl_append]
    _ = ((List.finRange n).take (j.val + 1) |>.foldl (fun s a => s ++ [g s a]) []).getD
          j.val d := by
          apply getD_foldl_snoc_lt
          rw [length_foldl_snoc, List.length_nil, List.length_take, List.length_finRange,
            Nat.zero_add]
          grind
    _ = g (((List.finRange n).take j.val).foldl (fun s a => s ++ [g s a]) []) j := by
          rw [htake, List.foldl_append, List.foldl_cons, List.foldl_nil]
          rw [List.getD_append_right _ _ _ _ (le_of_eq hplen), hplen, Nat.sub_self]
          rfl

end FoldSnoc

/-! ### Cholesky factor is lower-triangular

A structural fact about the executable `choleskyFn`, proved directly from the column fold: the entry
above the diagonal is forced to `0` by the construction. -/

/-- Reading an entry of a matrix tensor returns the underlying function value. -/
theorem get2_matrix {m k : Nat} (f : Fin m → Fin k → ℝ) (i : Fin m) (j : Fin k) :
    Spec.get2 (TorchLean.Tensor.matrix f) i j = f i j := by
  simp [TorchLean.Tensor.matrix, Spec.get2]

/-- The executable Cholesky factor is lower-triangular: entries strictly above the diagonal
vanish. -/
theorem choleskyFn_lower_triangular (A : Fin n → Fin n → ℝ) {i j : Fin n} (hij : i.val < j.val) :
    Spec.choleskyFn A i j = 0 := by
  unfold Spec.choleskyFn Spec.choleskyColsFn
  rw [getD_foldl_finRange]
  rw [ite_eq_left hij]

/-- Tensor-level statement: the Cholesky factor `choleskySpec A` is lower-triangular. -/
theorem choleskySpec_lower_triangular (A : TorchLean.Tensor ℝ [n, n])
    {i j : Fin n} (hij : i.val < j.val) :
    Spec.get2 (Spec.choleskySpec A) i j = 0 := by
  rw [show Spec.choleskySpec A =
      TorchLean.Tensor.matrix (Spec.choleskyFn (Spec.toMatFn A)) from rfl,
    get2_matrix]
  exact choleskyFn_lower_triangular _ hij
end Spec.Factorization
