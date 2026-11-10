/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import Mathlib.Data.List.GetD
public import TorchLean.Tensor.Numeric.Real
public import TorchLean.Tensor.Factorization.Correctness

/-!
# Exact reconstruction of the finite factorizations (Cholesky and QR)

Exact algebraic reconstruction for the Cholesky and classical Gram–Schmidt specification
recurrences in `TorchLean.Tensor.Factorization`. Both recurrences build their
factors with finite column folds. The theorems hold in exact real arithmetic and give no
floating-point error bound. Compiled code runs the array-backed `Factorization.choleskyCols` in
place of `choleskyColsFn`, and `choleskyColsFn_eq_choleskyCols` proves the two equal. Positive
pivots are hypotheses of the theorems, not a runtime success check.

## Main results

* `isCholesky_of_pos`: for a symmetric `A : Fin n → Fin n → ℝ` whose executable Cholesky pivots are
  all positive (`0 < choleskyFn A j j`), the factor `L = choleskyFn A` satisfies
  `TorchLean.Factorization.IsCholesky`: it is lower-triangular and `A = L · Lᵀ`.
  `cholesky_reconstruction` is the tensor-level corollary.
* `qr_mul_eq`: for `A : Fin m → Fin n → ℝ` whose executable Gram–Schmidt `R`-pivots are positive
  (`0 < rMat A j j`), the factors `Q` and `R` satisfy `A = Q · R`, with `R` upper-triangular
  (`rMat_upper_triangular`). `qr_reconstruction` is the tensor-level corollary.

## Method

The Cholesky factor is built by a `List.foldl` that snocs one column per index. The core
technical device is `getD_foldl_snoc_read`, a general lemma reading the `j`-th element of such a
fold as the step function applied to the length-`j` prefix. From it, `prefixCols_eq_map`
identifies the prefix with the first `j` columns of the final factor, and `take_map_sum_eq` turns
the left-fold sums of the specification into masked `Finset` partial sums.
The QR fold threads a `GSState` that appends one `R` column at every index and one `Q` column at
each index whose column enters the basis. Each appended value depends only on the `Q` columns
built so far, so the `Q`-list is a single-list fold (`gs_proj_qs`), and `gs_fold_split` with
`rTail_getD` reads the `R`-list. Under positive pivots every column enters in its own slot
(`qsPrefix_pivot`), so `qsPrefix_eq_map` identifies the basis built before `j` with the first `j`
columns of `Q`. The positive-pivot hypotheses also discharge the square-root and division side
conditions.

## Scope

The reconstructions `A = L · Lᵀ` and `A = Q · R` are purely algebraic. The remaining QR property,
orthonormality of the `Q` factor, `Qᵀ Q = 1`, is proved in
`TorchLean.Tensor.Factorization.Orthonormal` by bridging the executable Gram–Schmidt to
Mathlib's `gramSchmidt`, which completes the full `Factorization.IsQR` predicate
(`isQR_of_pos`).
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)

namespace TorchLean.Factorization

open Matrix
open scoped BigOperators

variable {n : Nat}

/-! ## Cholesky: the column-building step

`choleskyColsFn` is a left fold that snocs one column per index. `cholStep` names the appended
column, so the fold-reading lemmas `Factorization.getD_foldl_finRange` and
`Factorization.getD_foldl_snoc_read` of `TorchLean.Tensor.Factorization.Correctness` apply to it. -/

/-- The column appended at index `j` of the Cholesky fold, given the columns `cols` built so far. -/
noncomputable def cholStep (A : Fin n → Fin n → ℝ) (cols : List (Fin n → ℝ)) (j : Fin n) :
    Fin n → ℝ :=
  let sumsq := (cols.map (fun ck => ck j)).foldl (fun s x => s + x * x) 0
  let Ljj := MathFunctions.sqrt (A j j - sumsq)
  fun i =>
    if i.val < j.val then 0
    else if i.val == j.val then Ljj
    else
      let s := (cols.map (fun ck => ck i * ck j)).foldl (fun acc x => acc + x) 0
      (A i j - s) / Ljj

/-- `choleskyColsFn` is the snoc-fold appending `cholStep`. -/
theorem choleskyColsFn_eq (A : Fin n → Fin n → ℝ) :
    Factorization.choleskyColsFn A
      = (List.finRange n).foldl (fun cols j => cols ++ [cholStep A cols j]) [] := rfl

/-- The diagonal value produced by `cholStep`. -/
theorem cholStep_diag (A : Fin n → Fin n → ℝ) (cols : List (Fin n → ℝ)) (j : Fin n) :
    cholStep A cols j j
      = MathFunctions.sqrt
          (A j j - (cols.map (fun ck => ck j)).foldl (fun s x => s + x * x) 0) := by
  simp only [cholStep]
  rw [ite_eq_right (lt_irrefl _), ite_eq_left (beq_self_eq_true _)]

/-- The below-diagonal value produced by `cholStep`. -/
theorem cholStep_offdiag (A : Fin n → Fin n → ℝ) (cols : List (Fin n → ℝ)) {i j : Fin n}
    (hij : j.val < i.val) :
    cholStep A cols j i
      = (A i j - (cols.map (fun ck => ck i * ck j)).foldl (fun acc x => acc + x) 0)
          / MathFunctions.sqrt
              (A j j - (cols.map (fun ck => ck j)).foldl (fun s x => s + x * x) 0) := by
  simp only [cholStep]
  rw [ite_eq_right (by grind), ite_eq_right (by rw [beq_iff_eq]; grind)]

/-- The length-`j` prefix of Cholesky columns built before index `j`. -/
noncomputable def prefixCols (A : Fin n → Fin n → ℝ) (j : Fin n) : List (Fin n → ℝ) :=
  ((List.finRange n).take j.val).foldl (fun cols k => cols ++ [cholStep A cols k]) []

/-- Entry `(i, j)` of the executable Cholesky factor equals `cholStep` evaluated on the prefix. -/
theorem choleskyFn_eq_step (A : Fin n → Fin n → ℝ) (i j : Fin n) :
    Factorization.choleskyFn A i j = cholStep A (prefixCols A j) j i := by
  show (Factorization.choleskyColsFn A).getD j.val (fun _ => 0) i = _
  rw [choleskyColsFn_eq, getD_foldl_finRange (fun cols k => cholStep A cols k) (fun _ => 0) j]
  rfl

/-- The prefix of Cholesky columns is exactly the first `j` columns of the final factor `L`,
each presented as the function `r ↦ L r k`. -/
theorem prefixCols_eq_map (A : Fin n → Fin n → ℝ) (j : Fin n) :
    prefixCols A j
      = ((List.finRange n).take j.val).map (fun k => fun r => Factorization.choleskyFn A r k) := by
  have hjval : ((List.finRange n).take j.val).length = j.val := by
    rw [List.length_take, List.length_finRange, Nat.min_eq_left (le_of_lt j.isLt)]
  apply List.ext_getElem
  · unfold prefixCols
    rw [length_foldl_snoc (fun cols k => cholStep A cols k), List.length_nil, Nat.zero_add,
      List.length_map]
  · intro p h1 h2
    rw [List.length_map, hjval] at h2
    have hpn : p < n := lt_trans h2 j.isLt
    rw [List.getElem_map]
    have hidx :
        ((List.finRange n).take j.val)[p]'(by rw [hjval]; exact h2) = (⟨p, hpn⟩ : Fin n) := by
      rw [List.getElem_take, List.getElem_finRange]; exact Fin.ext rfl
    rw [show (prefixCols A j)[p]'h1 = (prefixCols A j).getD p (fun _ => 0) from
      (List.getD_eq_getElem _ _ h1).symm]
    unfold prefixCols
    rw [getD_foldl_snoc_read (fun cols k => cholStep A cols k) (fun _ => 0)
      ((List.finRange n).take j.val) p (by rw [hjval]; exact h2)]
    rw [List.take_take, Nat.min_eq_left (le_of_lt h2), hidx]
    funext r
    rw [choleskyFn_eq_step]
    rfl

/-! ### List/Finset partial-sum bridges -/

/-- Every element of a `finRange` prefix has index below the cut. -/
theorem mem_take_finRange {m : Nat} {x : Fin n} (hx : x ∈ (List.finRange n).take m) :
    x.val < m := by
  obtain ⟨p, hp, hpx⟩ := List.getElem_of_mem hx
  rw [List.length_take, List.length_finRange] at hp
  rw [List.getElem_take, List.getElem_finRange] at hpx
  subst hpx
  exact lt_of_lt_of_le hp (Nat.min_le_left m n)

/-- Every element of a `finRange` tail has index at least the cut. -/
theorem mem_drop_finRange {m : Nat} {x : Fin n} (hx : x ∈ (List.finRange n).drop m) :
    m ≤ x.val := by
  obtain ⟨p, hp, hpx⟩ := List.getElem_of_mem hx
  rw [List.getElem_drop, List.getElem_finRange] at hpx
  subst hpx
  exact Nat.le_add_right m p

/-- Mapping `f` over a `finRange` prefix and summing equals the masked full sum. -/
theorem take_map_sum_eq (m : Nat) (f : Fin n → ℝ) :
    (((List.finRange n).take m).map f).sum = ∑ k : Fin n, if k.val < m then f k else 0 := by
  rw [Fin.sum_univ_def]
  conv_rhs => rw [show (List.finRange n)
    = (List.finRange n).take m ++ (List.finRange n).drop m from (List.take_append_drop _ _).symm]
  rw [List.map_append, List.sum_append]
  have htake : ((List.finRange n).take m).map (fun k => if k.val < m then f k else 0)
      = ((List.finRange n).take m).map f :=
    List.map_congr_left (fun x hx => ite_eq_left (mem_take_finRange hx))
  have hdrop :
      (((List.finRange n).drop m).map (fun k => if k.val < m then f k else 0)).sum = 0 := by
    rw [List.sum_eq_zero]
    intro y hy
    rw [List.mem_map] at hy
    obtain ⟨x, hx, rfl⟩ := hy
    exact ite_eq_right (by have := mem_drop_finRange hx; grind)
  rw [htake, hdrop, add_zero]

/-- The Cholesky cross-sum equals the masked partial dot product of rows `i` and `j` of `L`. -/
theorem cross_sum_eq (A : Fin n → Fin n → ℝ) (i j : Fin n) :
    ((prefixCols A j).map (fun ck => ck i * ck j)).foldl (fun acc x => acc + x) 0
      = ∑ k : Fin n,
          if k.val < j.val then Factorization.choleskyFn A i k * Factorization.choleskyFn A j k
          else 0 := by
  rw [prefixCols_eq_map, List.map_map, ← List.sum_eq_foldl,
    show ((fun ck : Fin n → ℝ => ck i * ck j) ∘ fun k => fun r => Factorization.choleskyFn A r k)
      = (fun k => Factorization.choleskyFn A i k * Factorization.choleskyFn A j k) from rfl]
  exact take_map_sum_eq j.val
    (fun k => Factorization.choleskyFn A i k * Factorization.choleskyFn A j k)

/-- The Cholesky diagonal sum-of-squares equals the masked partial squared norm of row `j`
of `L`. -/
theorem sumsq_eq (A : Fin n → Fin n → ℝ) (j : Fin n) :
    ((prefixCols A j).map (fun ck => ck j)).foldl (fun s x => s + x * x) 0
      = ∑ k : Fin n,
          if k.val < j.val then Factorization.choleskyFn A j k * Factorization.choleskyFn A j k
          else 0 := by
  rw [prefixCols_eq_map, List.map_map, ← List.foldl_map (f := fun x : ℝ => x * x)
      (g := fun acc x : ℝ => acc + x),
    ← List.sum_eq_foldl, List.map_map,
    show ((fun x : ℝ => x * x) ∘
        ((fun ck : Fin n → ℝ => ck j) ∘ fun k => fun r => Factorization.choleskyFn A r k))
      = (fun k => Factorization.choleskyFn A j k * Factorization.choleskyFn A j k) from rfl]
  exact take_map_sum_eq j.val
    (fun k => Factorization.choleskyFn A j k * Factorization.choleskyFn A j k)

/-! ### Closed-form entries of the executable Cholesky factor -/

/-- The diagonal entry of `L` in closed form: `L[j,j] = √(A[j,j] − Σ_{k<j} L[j,k]²)`. -/
theorem choleskyFn_diag_eq (A : Fin n → Fin n → ℝ) (j : Fin n) :
    Factorization.choleskyFn A j j
      = Real.sqrt (A j j
          - ∑ k, if k.val < j.val then
            Factorization.choleskyFn A j k * Factorization.choleskyFn A j k else 0) := by
  rw [choleskyFn_eq_step, cholStep_diag, sumsq_eq, MathFunctions.sqrt_real]

/-- The below-diagonal entry of `L` in closed form:
`L[i,j] = (A[i,j] − Σ_{k<j} L[i,k]·L[j,k]) / L[j,j]` for `i > j`. -/
theorem choleskyFn_offdiag_eq (A : Fin n → Fin n → ℝ) {i j : Fin n} (hij : j.val < i.val) :
    Factorization.choleskyFn A i j
      = (A i j - ∑ k, if k.val <
        j.val then Factorization.choleskyFn A i k * Factorization.choleskyFn A j k else 0)
          / Factorization.choleskyFn A j j := by
  rw [choleskyFn_eq_step A i j, cholStep_offdiag _ _ hij, cross_sum_eq, sumsq_eq,
    MathFunctions.sqrt_real, ← choleskyFn_diag_eq]

/-! ### Reconstruction `A = L · Lᵀ`

Closed-form entries and positive pivots give reconstruction on and below the diagonal. Symmetry
then gives the remaining entries. -/

/-- If every pivot `L[j,j]` is positive, the `(i, j)` entry of `L · Lᵀ` is `A i j` on and below
the diagonal (`j ≤ i`). -/
theorem choleskyFn_dot_of_le (A : Fin n → Fin n → ℝ)
    (hpos : ∀ j : Fin n, 0 < Factorization.choleskyFn A j j) {i j : Fin n} (hji : j.val ≤ i.val) :
    (∑ k, Factorization.choleskyFn A i k * Factorization.choleskyFn A j k) = A i j := by
  set L := Factorization.choleskyFn A with hL
  have key : ∀ k : Fin n, L i k * L j k
      = (if k.val < j.val then L i k * L j k else 0) + (if k = j then L i j * L j j else 0) := by
    intro k
    rcases lt_trichotomy k.val j.val with h | h | h
    · have hne : k ≠ j := fun hk => by rw [hk] at h; exact lt_irrefl _ h
      rw [ite_eq_left h, ite_eq_right hne, add_zero]
    · have hkj : k = j := Fin.ext h
      rw [ite_eq_right (by grind), ite_eq_left hkj, zero_add, hkj]
    · have hne : k ≠ j := fun hk => by rw [hk] at h; exact lt_irrefl _ h
      rw [ite_eq_right (by grind), ite_eq_right hne, add_zero,
        show L j k = 0 from Factorization.choleskyFn_lower_triangular A h, mul_zero]
  rw [show (∑ k, L i k * L j k)
      = ∑ k, ((if k.val < j.val then L i k * L j k else 0) + (if k = j then L i j * L j j else 0))
      from Finset.sum_congr rfl (fun k _ => key k),
    Finset.sum_add_distrib, Finset.sum_ite_eq' Finset.univ j (fun _ => L i j * L j j)]
  simp only [Finset.mem_univ, ite_true]
  rcases eq_or_lt_of_le hji with heq | hlt
  · have hij' : i = j := Fin.ext heq.symm
    subst hij'
    have hrad : 0 < A i i - (∑ k, if k.val < i.val then L i k * L i k else 0) := by
      have hp := hpos i
      rw [hL, choleskyFn_diag_eq] at hp
      exact Real.sqrt_pos.mp hp
    have hsq : L i i * L i i = A i i - (∑ k, if k.val < i.val then L i k * L i k else 0) := by
      conv_lhs => rw [hL, choleskyFn_diag_eq A i]
      exact Real.mul_self_sqrt hrad.le
    rw [hsq]; ring
  · have hne : L j j ≠ 0 := ne_of_gt (hpos j)
    have hmul : L i j * L j j
        = A i j - (∑ k, if k.val < j.val then L i k * L j k else 0) := by
      rw [hL, choleskyFn_offdiag_eq A hlt, div_mul_eq_mul_div, mul_div_assoc, div_self hne, mul_one]
    rw [hmul]; ring

/-- If `A` is symmetric and every pivot `L[j,j]` is positive, every entry of `L · Lᵀ` is the
corresponding entry of `A`. -/
theorem choleskyFn_dot (A : Fin n → Fin n → ℝ) (hsymm : ∀ i j, A i j = A j i)
    (hpos : ∀ j : Fin n, 0 < Factorization.choleskyFn A j j) (i j : Fin n) :
    (∑ k, Factorization.choleskyFn A i k * Factorization.choleskyFn A j k) = A i j := by
  rcases le_total j.val i.val with h | h
  · exact choleskyFn_dot_of_le A hpos h
  · rw [show (∑ k, Factorization.choleskyFn A i k * Factorization.choleskyFn A j k)
        = ∑ k, Factorization.choleskyFn A j k * Factorization.choleskyFn A i k
        from Finset.sum_congr rfl (fun k _ => mul_comm _ _),
      choleskyFn_dot_of_le A hpos h, hsymm j i]

/-- For a symmetric `A` whose specification Cholesky pivots are all
positive (`0 < L[j,j]`), the factor `L = choleskyFn A` is a
Cholesky factor: lower-triangular with `A = L · Lᵀ`. -/
theorem isCholesky_of_pos (A : Fin n → Fin n → ℝ) (hsymm : ∀ i j, A i j = A j i)
    (hpos : ∀ j : Fin n, 0 < Factorization.choleskyFn A j j) :
    Factorization.IsCholesky (Matrix.of A) (Matrix.of (Factorization.choleskyFn A)) := by
  refine ⟨?_, ?_⟩
  · intro a b hab
    show Factorization.choleskyFn A a b = 0
    exact Factorization.choleskyFn_lower_triangular A (Fin.lt_def.mp hab)
  · ext i j
    rw [Matrix.mul_apply]
    simp only [Matrix.of_apply, Matrix.transpose_apply]
    exact (choleskyFn_dot A hsymm hpos i j).symm

/-- For a symmetric tensor `A` whose `Tensor.cholesky` pivots
are positive, every entry of `A` is reconstructed by `L · Lᵀ`:
`A[i,j] = Σ_k L[i,k] · L[j,k]`, with `L = Tensor.cholesky A`. -/
theorem _root_.TorchLean.Tensor.cholesky_reconstruction (A : Tensor ℝ [n, n])
    (hsymm : ∀ i j, Tensor.get2 A i j = Tensor.get2 A j i)
    (hpos : ∀ j : Fin n, 0 < Tensor.get2 A.cholesky j j) (i j : Fin n) :
    Tensor.get2 A i j = ∑ k, Tensor.get2 A.cholesky i k * Tensor.get2 A.cholesky j k := by
  have hg : ∀ a b, Tensor.get2 A.cholesky a b =
      Factorization.choleskyFn (Tensor.get2 A) a b := by
    intro a b
    rw [show A.cholesky =
        Tensor.matrix (Factorization.choleskyFn (Tensor.get2 A)) from rfl,
      Tensor.get2_matrix]
  simp only [hg]
  show Tensor.get2 A i j = _
  refine (choleskyFn_dot (Tensor.get2 A) (fun a b => hsymm a b) (fun b => ?_) i j).symm
  rw [← hg b b]; exact hpos b

/-! ## QR (classical Gram–Schmidt): exact reconstruction `A = Q · R`

`gramSchmidtFn` threads a `GSState` that appends one `R` column at every index and one `Q` column
at each index whose column enters the basis. The appended values depend only on the `Q`-history
(`st.qs`), never on the `R`-history, so the `Q`-list is itself a single-list fold (`gs_proj_qs`) and
the `R`-list is the `Q`-prefix-indexed tail `rTail`. Under positive pivots, column `j` enters in
slot `j` (`qsPrefix_pivot`). -/

section QR

variable {m : Nat}

open Factorization (GSState)

/-- Column `j` of `A` as a function of the row. -/
noncomputable def gsA (A : Fin m → Fin n → ℝ) (j : Fin n) : Fin m → ℝ := fun i => A i j

/-- The `R` off-diagonal entries `rₖⱼ = qₖ · a` for the columns `qs` built so far. -/
noncomputable def gsRkjs (A : Fin m → Fin n → ℝ) (qs : List (Fin m → ℝ)) (j : Fin n) : List ℝ :=
  qs.map (fun qk => Factorization.dotFn qk (gsA A j))

/-- The orthogonalized (not-yet-normalized) vector `v = a − Σ rₖⱼ qₖ`. -/
noncomputable def gsV (A : Fin m → Fin n → ℝ) (qs : List (Fin m → ℝ)) (j : Fin n) : Fin m → ℝ :=
  Tensor.getScalar <| Tensor.ofFn fun i => gsA A j i
    - (List.zip qs (gsRkjs A qs j)).foldl (fun acc (qk, r) => acc + r * qk i) 0

/-- The diagonal `R` entry `rⱼⱼ = ‖v‖`. -/
noncomputable def gsRjj (A : Fin m → Fin n → ℝ) (qs : List (Fin m → ℝ)) (j : Fin n) : ℝ :=
  Factorization.normFn (gsV A qs j)

/-- Column `j` enters the basis `qs` when `rⱼⱼ > 0` and `qs` has fewer than `min m n` columns. -/
noncomputable def gsAppend (A : Fin m → Fin n → ℝ) (qs : List (Fin m → ℝ)) (j : Fin n) : Bool :=
  Context.gtBool (gsRjj A qs j) 0 && decide (qs.length < min m n)

/-- The `Q` column built at index `j`: `v / rⱼⱼ`. -/
noncomputable def qStep (A : Fin m → Fin n → ℝ) (qs : List (Fin m → ℝ)) (j : Fin n) : Fin m → ℝ :=
  Tensor.getScalar <| Tensor.ofFn fun i => gsV A qs j i / gsRjj A qs j

/-- The basis after index `j`: `qs` followed by `qStep` when column `j` enters, else `qs`. -/
noncomputable def qNext (A : Fin m → Fin n → ℝ) (qs : List (Fin m → ℝ)) (j : Fin n) :
    List (Fin m → ℝ) :=
  if gsAppend A qs j then qs ++ [qStep A qs j] else qs

/-- Column `j` of `R` as a function of the row index `k`, so its value is the entry `R[k, j]`. It is
`rₖⱼ` at the rows `k < qs.length` of the basis `qs`, `rⱼⱼ` at `k = qs.length` when column `j` enters
the basis (`gsAppend`), and `0` otherwise. -/
noncomputable def rStep (A : Fin m → Fin n → ℝ) (qs : List (Fin m → ℝ)) (j : Fin n) : Fin n → ℝ :=
  fun k => if k.val < qs.length then (gsRkjs A qs j).getD k.val 0
    else if k.val == qs.length && gsAppend A qs j then gsRjj A qs j else 0

/-- `gramSchmidtFn` as the dual-list fold applying `qNext` and appending `rStep`. -/
theorem gramSchmidtFn_eq (A : Fin m → Fin n → ℝ) :
    Factorization.gramSchmidtFn A
      = (List.finRange n).foldl
          (fun st j => (⟨qNext A st.qs j, st.rcols ++ [rStep A st.qs j]⟩ : GSState m n ℝ))
          ⟨[], []⟩ := rfl

/-- The `Q`-list projection of the structure fold is the single-list `qNext` fold. -/
theorem gs_proj_qs (A : Fin m → Fin n → ℝ) (l : List (Fin n)) (q0 : List (Fin m → ℝ))
    (r0 : List (Fin n → ℝ)) :
    (l.foldl (fun st j => (⟨qNext A st.qs j, st.rcols ++ [rStep A st.qs j]⟩ : GSState m n ℝ))
        ⟨q0, r0⟩).qs
      = l.foldl (qNext A) q0 := by
  induction l generalizing q0 r0 with
  | nil => rfl
  | cons a t ih => simp only [List.foldl_cons]; exact ih _ _

/-- The `Q` columns built before index `j`. -/
noncomputable def qsPrefix (A : Fin m → Fin n → ℝ) (j : Fin n) : List (Fin m → ℝ) :=
  ((List.finRange n).take j.val).foldl (qNext A) []

/-- The `R`-list tail: the `R` columns produced from `Q`-prefix `q0` over the indices `l`. -/
noncomputable def rTail (A : Fin m → Fin n → ℝ) (q0 : List (Fin m → ℝ)) : List (Fin n) →
    List (Fin n → ℝ)
  | [] => []
  | j :: rest => rStep A q0 j :: rTail A (qNext A q0 j) rest

/-- The structure fold splits into the `qNext` fold (`Q`-list) and the `rTail` (`R`-list). -/
theorem gs_fold_split (A : Fin m → Fin n → ℝ) (l : List (Fin n)) (q0 : List (Fin m → ℝ))
    (r0 : List (Fin n → ℝ)) :
    (l.foldl (fun st j => (⟨qNext A st.qs j, st.rcols ++ [rStep A st.qs j]⟩ : GSState m n ℝ))
        ⟨q0, r0⟩)
      = ⟨l.foldl (qNext A) q0, r0 ++ rTail A q0 l⟩ := by
  induction l generalizing q0 r0 with
  | nil => simp [rTail]
  | cons j rest ih =>
      simp only [List.foldl_cons, rTail]
      rw [ih]
      simp [List.append_assoc]

/-- Reading the `k`-th element of `rTail` recovers `rStep` applied to the `Q`-prefix built over the
first `k` indices. -/
theorem rTail_getD (A : Fin m → Fin n → ℝ) (q0 : List (Fin m → ℝ)) (l : List (Fin n)) (k : Nat)
    (hk : k < l.length) (d : Fin n → ℝ) :
    (rTail A q0 l).getD k d = rStep A ((l.take k).foldl (qNext A) q0) (l[k]'hk) := by
  induction l generalizing q0 k with
  | nil => simp at hk
  | cons j rest ih =>
      cases k with
      | zero => simp [rTail]
      | succ k' =>
          simp only [rTail, List.getD_cons_succ, List.take_succ_cons, List.foldl_cons,
            List.getElem_cons_succ]
          exact ih (qNext A q0 j) k' (by simpa using hk)

/-! ### Growth of the basis -/

/-- One step of the fold keeps the basis built so far as a prefix. -/
theorem qNext_prefix (A : Fin m → Fin n → ℝ) (qs : List (Fin m → ℝ)) (j : Fin n) :
    qs <+: qNext A qs j := by
  unfold qNext
  split
  · exact List.prefix_append qs _
  · exact List.prefix_refl qs

/-- The `qNext` fold over `l` extends `q0` by at most `l.length` columns. -/
theorem foldl_qNext_prefix (A : Fin m → Fin n → ℝ) (l : List (Fin n)) (q0 : List (Fin m → ℝ)) :
    q0 <+: l.foldl (qNext A) q0 ∧ (l.foldl (qNext A) q0).length ≤ q0.length + l.length := by
  induction l generalizing q0 with
  | nil => exact ⟨List.prefix_refl q0, by simp⟩
  | cons j rest ih =>
      obtain ⟨hp, hl⟩ := ih (qNext A q0 j)
      have hj : (qNext A q0 j).length ≤ q0.length + 1 := by
        unfold qNext
        split
        · simp
        · omega
      rw [List.foldl_cons, List.length_cons]
      exact ⟨(qNext_prefix A q0 j).trans hp, by omega⟩

/-- At most `j` columns enter the basis before index `j`. -/
theorem qsPrefix_length_le (A : Fin m → Fin n → ℝ) (j : Fin n) :
    (qsPrefix A j).length ≤ j.val := by
  have h := (foldl_qNext_prefix A ((List.finRange n).take j.val) []).2
  rw [List.length_nil, Nat.zero_add, List.length_take, List.length_finRange,
    Nat.min_eq_left (le_of_lt j.isLt)] at h
  exact h

/-- The `Q`-list of `gramSchmidtFn` is the `qNext` fold over every index. -/
theorem gramSchmidtFn_qs (A : Fin m → Fin n → ℝ) :
    (Factorization.gramSchmidtFn A).qs = (List.finRange n).foldl (qNext A) [] := by
  rw [gramSchmidtFn_eq]
  exact gs_proj_qs A (List.finRange n) [] []

/-- The basis built over the first `b` indices is a prefix of the final `Q`-list. -/
theorem take_foldl_prefix (A : Fin m → Fin n → ℝ) (b : Nat) :
    ((List.finRange n).take b).foldl (qNext A) [] <+: (Factorization.gramSchmidtFn A).qs := by
  have h := (foldl_qNext_prefix A ((List.finRange n).drop b)
    (((List.finRange n).take b).foldl (qNext A) [])).1
  rwa [← List.foldl_append, List.take_append_drop, ← gramSchmidtFn_qs] at h

/-- The basis built over the first `j + 1` indices is `qNext` applied to the `Q`-prefix at `j`. -/
theorem foldl_take_succ (A : Fin m → Fin n → ℝ) (j : Fin n) :
    ((List.finRange n).take (j.val + 1)).foldl (qNext A) [] = qNext A (qsPrefix A j) j := by
  have hjlen : j.val < (List.finRange n).length := by
    rw [List.length_finRange]
    exact j.isLt
  have hj : (List.finRange n)[j.val]'hjlen = j := by simp [List.getElem_finRange]
  rw [List.take_succ_eq_append_getElem hjlen, List.foldl_append, List.foldl_cons, List.foldl_nil,
    hj]
  rfl

/-! ### Entries of the executable `Q` and `R` factors -/

/-- Entry `(i, k)` of the `Q` factor produced by `gramSchmidtFn`. -/
noncomputable def qMat (A : Fin m → Fin n → ℝ) (i : Fin m) (k : Fin n) : ℝ :=
  (Factorization.gramSchmidtFn A).qs.getD k.val (fun _ => 0) i

/-- Entry `(k, j)` of the `R` factor produced by `gramSchmidtFn`. -/
noncomputable def rMat (A : Fin m → Fin n → ℝ) (k j : Fin n) : ℝ :=
  (Factorization.gramSchmidtFn A).rcols.getD j.val (fun _ => 0) k

/-- Column `k` of `Q` as a function of the row. -/
noncomputable def qCol (A : Fin m → Fin n → ℝ) (k : Fin n) : Fin m → ℝ := fun r => qMat A r k

/-- Closed form of an `R` entry: `rStep` evaluated on the `Q`-prefix. -/
theorem rMat_eq (A : Fin m → Fin n → ℝ) (k j : Fin n) :
    rMat A k j = rStep A (qsPrefix A j) j k := by
  have hrc : (Factorization.gramSchmidtFn A).rcols = rTail A [] (List.finRange n) := by
    rw [gramSchmidtFn_eq, gs_fold_split]; simp
  unfold rMat
  rw [hrc, rTail_getD A [] (List.finRange n) j.val (by rw [List.length_finRange]; exact j.isLt)]
  have hk : (List.finRange n)[j.val]'(by rw [List.length_finRange]; exact j.isLt) = j := by
    simp [List.getElem_finRange]
  rw [hk]; rfl

/-- If the pivot `rMat A j j` is positive, then the basis built before `j` has `j` columns and
column `j` enters it with `rⱼⱼ > 0`. -/
theorem qsPrefix_pivot (A : Fin m → Fin n → ℝ) (j : Fin n) (hj : 0 < rMat A j j) :
    (qsPrefix A j).length = j.val ∧ gsAppend A (qsPrefix A j) j = true ∧
      0 < gsRjj A (qsPrefix A j) j := by
  have hle := qsPrefix_length_le A j
  rw [rMat_eq] at hj
  simp only [rStep] at hj
  split_ifs at hj with h1 h2
  · exact absurd h1 (Nat.not_lt.mpr hle)
  · simp only [Bool.and_eq_true, beq_iff_eq] at h2
    exact ⟨h2.1.symm, h2.2, hj⟩
  · exact absurd hj (lt_irrefl 0)

/-- Under a positive pivot at `k`, entry `(i, k)` of `Q` is `qStep` evaluated on the basis built
before `k`. -/
theorem qMat_eq (A : Fin m → Fin n → ℝ) (i : Fin m) (k : Fin n) (hk : 0 < rMat A k k) :
    qMat A i k = qStep A (qsPrefix A k) k i := by
  obtain ⟨hlen, happ, -⟩ := qsPrefix_pivot A k hk
  have hpre := take_foldl_prefix A (k.val + 1)
  rw [foldl_take_succ, qNext, ite_eq_left happ] at hpre
  have hlt : k.val < (qsPrefix A k ++ [qStep A (qsPrefix A k) k]).length := by simp [hlen]
  unfold qMat
  rw [List.getD_eq_getElem _ _ (lt_of_lt_of_le hlt hpre.length_le), ← hpre.getElem hlt,
    List.getElem_concat_length hlen.symm]

/-- The `R` column `rStep A qs j` is zero below the diagonal, at every row `k` with `j < k`, when
the basis `qs` has at most `j` columns. -/
theorem rStep_below_diag_zero (A : Fin m → Fin n → ℝ) (qs : List (Fin m → ℝ)) {j k : Fin n}
    (hlen : qs.length ≤ j.val) (hjk : j.val < k.val) : rStep A qs j k = 0 := by
  have h1 : ¬ k.val < qs.length := by omega
  have h2 : ¬ (k.val == qs.length && gsAppend A qs j) = true := by
    rw [Bool.and_eq_true, beq_iff_eq]
    exact fun h => absurd h.1 (by omega)
  simp only [rStep]
  rw [ite_eq_right h1, ite_eq_right h2]

/-- When the basis `qs` has exactly `j` columns and column `j` enters it, the diagonal `R` entry
is `rⱼⱼ`. -/
theorem rStep_diag (A : Fin m → Fin n → ℝ) (qs : List (Fin m → ℝ)) (j : Fin n)
    (hlen : qs.length = j.val) (happ : gsAppend A qs j = true) :
    rStep A qs j j = gsRjj A qs j := by
  have h1 : ¬ j.val < qs.length := by omega
  have h2 : (j.val == qs.length && gsAppend A qs j) = true := by
    rw [Bool.and_eq_true, beq_iff_eq]
    exact ⟨hlen.symm, happ⟩
  simp only [rStep]
  rw [ite_eq_right h1, ite_eq_left h2]

/-- Entry `i` of the `Q` column built at index `j` is `vᵢ / rⱼⱼ`. -/
theorem qStep_apply (A : Fin m → Fin n → ℝ) (qs : List (Fin m → ℝ)) (j : Fin n) (i : Fin m) :
    qStep A qs j i = gsV A qs j i / gsRjj A qs j := by
  simp only [qStep, Tensor.getScalar_ofFn]

/-! ### The orthogonalization sum as a `Finset` sum -/

/-- The zip-fold defining `v` collapses to a single map-fold over the `Q` columns. -/
theorem cross_fold_eq (qs : List (Fin m → ℝ)) (g : (Fin m → ℝ) → ℝ) (i : Fin m) (a : ℝ) :
    (List.zip qs (qs.map g)).foldl (fun acc (qk, r) => acc + r * qk i) a
      = a + (qs.map (fun qk => g qk * qk i)).sum := by
  induction qs generalizing a with
  | nil => simp
  | cons x xs ih =>
      simp only [List.map_cons, List.zip_cons_cons, List.foldl_cons]
      rw [ih]; simp only [List.sum_cons]; ring

/-- Closed form of `v i`: `A i j` minus the partial projection sum. -/
theorem gsV_eq (A : Fin m → Fin n → ℝ) (qs : List (Fin m → ℝ)) (j : Fin n) (i : Fin m) :
    gsV A qs j i =
      gsA A j i - (qs.map (fun qk => Factorization.dotFn qk (gsA A j) * qk i)).sum := by
  simp only [gsV, gsRkjs, Tensor.getScalar_ofFn]
  rw [cross_fold_eq qs (fun qk => Factorization.dotFn qk (gsA A j)) i 0, zero_add]

/-- Under a positive pivot at `j`, the `Q`-prefix list has length `j`. -/
theorem qsPrefix_length (A : Fin m → Fin n → ℝ) (j : Fin n) (hj : 0 < rMat A j j) :
    (qsPrefix A j).length = j.val :=
  (qsPrefix_pivot A j hj).1

/-- Under a positive pivot at `j`, the `Q`-prefix is exactly the first `j` columns of the final
factor `Q`. -/
theorem qsPrefix_eq_map (A : Fin m → Fin n → ℝ) (j : Fin n) (hj : 0 < rMat A j j) :
    qsPrefix A j = ((List.finRange n).take j.val).map (fun k => qCol A k) := by
  have hjval : ((List.finRange n).take j.val).length = j.val := by
    rw [List.length_take, List.length_finRange, Nat.min_eq_left (le_of_lt j.isLt)]
  have hpre : qsPrefix A j <+: (Factorization.gramSchmidtFn A).qs := take_foldl_prefix A j.val
  apply List.ext_getElem
  · rw [qsPrefix_length A j hj, List.length_map, hjval]
  · intro p h1 h2
    rw [List.length_map, hjval] at h2
    have hpn : p < n := lt_trans h2 j.isLt
    rw [List.getElem_map, hpre.getElem h1]
    have hidx :
        ((List.finRange n).take j.val)[p]'(by rw [hjval]; exact h2) = (⟨p, hpn⟩ : Fin n) := by
      rw [List.getElem_take, List.getElem_finRange]; exact Fin.ext rfl
    rw [hidx]
    funext r
    show _ = (Factorization.gramSchmidtFn A).qs.getD p (fun _ => 0) r
    rw [List.getD_eq_getElem _ _ (lt_of_lt_of_le h1 hpre.length_le)]

/-- `getD` commutes with `dotFn`-mapping when the index is in range. -/
theorem getD_map_dotFn (qs : List (Fin m → ℝ)) (a : Fin m → ℝ) (k : Nat) (hk : k < qs.length) :
    (qs.map (fun qk => Factorization.dotFn qk a)).getD k 0 =
      Factorization.dotFn (qs.getD k (fun _ => 0)) a := by
  rw [List.getD_eq_getElem _ _ (by rw [List.length_map]; exact hk), List.getElem_map,
    List.getD_eq_getElem _ _ hk]

/-- Under a positive pivot at `j`, a `Q`-prefix entry equals the final `Q` column at that
index. -/
theorem qsPrefix_getD (A : Fin m → Fin n → ℝ) {k j : Fin n} (hkj : k.val < j.val)
    (hj : 0 < rMat A j j) :
    (qsPrefix A j).getD k.val (fun _ => 0) = qCol A k := by
  rw [qsPrefix_eq_map A j hj,
    List.getD_eq_getElem _ _ (by rw [List.length_map, List.length_take, List.length_finRange,
      Nat.min_eq_left (le_of_lt j.isLt)]; exact hkj),
    List.getElem_map]
  congr 1
  rw [List.getElem_take, List.getElem_finRange]; exact Fin.ext rfl

/-- Under a positive pivot at `j`, the strictly-above-diagonal `R` entry `R[k, j]`
(`row k < column j`) is the inner product of `Q` column `k` with column `j`. -/
theorem rMat_eq_dotFn_of_lt (A : Fin m → Fin n → ℝ) {k j : Fin n} (hkj : k.val < j.val)
    (hj : 0 < rMat A j j) :
    rMat A k j = Factorization.dotFn (qCol A k) (gsA A j) := by
  have hk : k.val < (qsPrefix A j).length := by rw [qsPrefix_length A j hj]; exact hkj
  rw [rMat_eq]; simp only [rStep]; rw [ite_eq_left hk]; unfold gsRkjs
  rw [getD_map_dotFn (qsPrefix A j) (gsA A j) k.val hk, qsPrefix_getD A hkj hj]

/-- Under a positive pivot at `j`, the projection sum equals the masked partial sum
`Σ_{k<j} R[k,j]·Q[i,k]`. -/
theorem cross_sum_qr (A : Fin m → Fin n → ℝ) (i : Fin m) (j : Fin n) (hj : 0 < rMat A j j) :
    ((qsPrefix A j).map (fun qk => Factorization.dotFn qk (gsA A j) * qk i)).sum
      = ∑ k, if k.val < j.val then rMat A k j * qMat A i k else 0 := by
  rw [qsPrefix_eq_map A j hj]
  rw [List.map_map]
  rw [take_map_sum_eq]
  apply Finset.sum_congr rfl
  intro k _
  by_cases hkj : k.val < j.val
  · rw [ite_eq_left hkj, ite_eq_left hkj]
    show Factorization.dotFn (qCol A k) (gsA A j) * qMat A i k = rMat A k j * qMat A i k
    rw [rMat_eq_dotFn_of_lt A hkj hj]
  · rw [ite_eq_right hkj, ite_eq_right hkj]

/-! ### Exact reconstruction `A = Q · R` -/

/-- `R` is upper-triangular: the entry `R[k, j]` strictly below the diagonal (`column j < row k`)
vanishes. -/
theorem rMat_upper_triangular (A : Fin m → Fin n → ℝ) {k j : Fin n} (hjk : j.val < k.val) :
    rMat A k j = 0 := by
  rw [rMat_eq]; exact rStep_below_diag_zero A (qsPrefix A j) (qsPrefix_length_le A j) hjk

/-- When every specification `R` pivot is positive, `A[i,j] = Σ_k Q[i,k]·R[k,j]`. -/
theorem eq_sum_qMat_mul_rMat (A : Fin m → Fin n → ℝ) (hrank : ∀ j : Fin n, 0 < rMat A j j)
    (i : Fin m) (j : Fin n) :
    A i j = ∑ k, qMat A i k * rMat A k j := by
  have key : ∀ k : Fin n, qMat A i k * rMat A k j
      = (if k.val < j.val then qMat A i k * rMat A k j else 0)
        + (if k = j then qMat A i j * rMat A j j else 0) := by
    intro k
    rcases lt_trichotomy k.val j.val with h | h | h
    · have hne : k ≠ j := fun hk => by rw [hk] at h; exact lt_irrefl _ h
      rw [ite_eq_left h, ite_eq_right hne, add_zero]
    · have hkj : k = j := Fin.ext h
      rw [ite_eq_right (by grind), ite_eq_left hkj, zero_add, hkj]
    · have hne : k ≠ j := fun hk => by rw [hk] at h; exact lt_irrefl _ h
      rw [ite_eq_right (by grind), ite_eq_right hne, add_zero, rMat_upper_triangular A h, mul_zero]
  rw [show (∑ k, qMat A i k * rMat A k j)
      = ∑ k, ((if k.val < j.val then qMat A i k * rMat A k j else 0)
        + (if k = j then qMat A i j * rMat A j j else 0))
      from Finset.sum_congr rfl (fun k _ => key k),
    Finset.sum_add_distrib, Finset.sum_ite_eq' Finset.univ j (fun _ => qMat A i j * rMat A j j)]
  simp only [Finset.mem_univ, ite_true]
  obtain ⟨hlen, happ, hρpos⟩ := qsPrefix_pivot A j (hrank j)
  have hdiag : qMat A i j * rMat A j j = gsV A (qsPrefix A j) j i := by
    rw [qMat_eq A i j (hrank j), qStep_apply, rMat_eq, rStep_diag A _ j hlen happ,
      div_mul_eq_mul_div, mul_div_assoc, div_self (ne_of_gt hρpos), mul_one]
  rw [hdiag, gsV_eq, cross_sum_qr A i j (hrank j),
    show gsA A j i = A i j from rfl,
    show (∑ k, if k.val < j.val then qMat A i k * rMat A k j else 0)
      = (∑ k, if k.val < j.val then rMat A k j * qMat A i k else 0)
      from Finset.sum_congr rfl (fun k _ => by
        by_cases hkj : k.val < j.val
        · rw [ite_eq_left hkj, ite_eq_left hkj, mul_comm]
        · rw [ite_eq_right hkj, ite_eq_right hkj])]
  ring

/-- `A = Q · R` for the specification Gram–Schmidt factors under positive `R` pivots. -/
theorem qr_mul_eq (A : Fin m → Fin n → ℝ) (hrank : ∀ j : Fin n, 0 < rMat A j j) :
    Matrix.of A = Matrix.of (fun i k => qMat A i k) * Matrix.of (fun k j => rMat A k j) := by
  ext i j
  rw [Matrix.mul_apply]
  simp only [Matrix.of_apply]
  exact eq_sum_qMat_mul_rMat A hrank i j

/-- For a tensor `A` whose `Factorization.qr` `R` pivots are positive, every entry of `A` is
reconstructed by `Q · R`: `A[i,j] = Σ_k Q[i,k]·R[k,j]`, with `(Q, R) = Factorization.qr A`. -/
theorem qr_reconstruction (A : Tensor ℝ [m, n])
    (hrank : ∀ j : Fin n, 0 < Tensor.get2 (Factorization.qr A).2 j j) (i : Fin m) (j : Fin n) :
    Tensor.get2 A i j
      = ∑ k, Tensor.get2 (Factorization.qr A).1 i k * Tensor.get2 (Factorization.qr A).2 k j := by
  have hQ : ∀ a b, Tensor.get2 (Factorization.qr A).1 a b = qMat (Tensor.get2 A) a b := by
    intro a b
    simp [Factorization.qr, Tensor.get2_matrix, qMat]
  have hR : ∀ a b, Tensor.get2 (Factorization.qr A).2 a b = rMat (Tensor.get2 A) a b := by
    intro a b
    simp [Factorization.qr, Tensor.get2_matrix, rMat]
  simp only [hQ, hR]
  show Tensor.get2 A i j = _
  exact eq_sum_qMat_mul_rMat (Tensor.get2 A)
    (fun b => by rw [← hR b b]; exact hrank b) i j

end QR

end TorchLean.Factorization
