/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor.Factorizations
public import NN.Proofs.Tensor.Basic.Factorizations
public import Mathlib.Data.List.GetD
public import Mathlib.Algebra.BigOperators.Fin

/-!
# Exact reconstruction of the finite factorizations (Cholesky and QR)

This file proves the *exact* algebraic reconstruction of the finite executable Cholesky and QR
factorizations from [`NN.Spec.Core.Tensor.Factorizations`](../../../Spec/Core/Tensor/Factorizations.lean),
the increment promised in `NN.Proofs.Tensor.Basic.Factorizations`. Unlike the iterative Jacobi/SVD
routines (whose reconstruction is only an a-posteriori residual certificate), Cholesky and Gram–Schmidt
are *finite* constructions, so over `ℝ` they reconstruct their input on the nose under the success
hypotheses.

## Main results

* `isCholesky_of_pos`: for a symmetric `A : Fin n → Fin n → ℝ` whose executable Cholesky pivots are all
  positive (`0 < choleskyFn A j j`, the exact condition under which the algorithm succeeds over `ℝ`),
  the factor `L = choleskyFn A` satisfies the spec `Spec.Factorization.IsCholesky`: lower-triangular
  and `A = L · Lᵀ`. `choleskySpec_reconstruction` is the tensor-level corollary.
* `qr_mul_eq`: for `A : Fin m → Fin n → ℝ` whose executable Gram–Schmidt `R`-pivots are positive
  (`0 < Rmat A j j`, full column rank), the factors `Q = gramSchmidtFn A` and `R` satisfy `A = Q · R`,
  with `R` upper-triangular (`Rmat_upper_triangular`). `qrSpec_reconstruction` is the tensor-level
  corollary.

## Method

Each executable factor is built by a `List.foldl` that snocs one column per index. The core technical
device is `getD_foldl_snoc_read`, a general lemma reading the `j`-th element of such a fold as the step
function applied to the length-`j` prefix. From it, `prefix_eq_map`/`qsPrefix_eq_map` identify the
prefix with the first `j` columns of the final factor, and `take_map_sum_eq` turns the code's
`List.foldl` sums into masked `Finset` partial sums. The QR fold threads a `GSState` that snocs onto
*both* the `Q`-list and the `R`-list at once; `gs_proj_qs` and `gs_fold_split`/`rTail_getD` recover the
single-list read lemmas for each projection (the step depends only on the `Q`-history). The
positive-pivot hypotheses discharge the `√`-radicand and divisor side conditions.

## Scope

This file proves `A = L · Lᵀ` and `A = Q · R` purely algebraically. The remaining QR property —
orthonormality of the `Q` factor, `Qᵀ Q = 1` — is proved in the companion file
[`NN.Proofs.Tensor.Basic.FactorizationsOrthonormal`](FactorizationsOrthonormal.lean) by bridging the
executable Gram–Schmidt to Mathlib's `gramSchmidt`, completing the full `Spec.Factorization.IsQR`
predicate (`isQR_of_pos`).
-/

@[expose] public section

namespace Spec.Factorization.Reconstruction

open Matrix
open scoped BigOperators

variable {n : Nat}

/-! ## List/Finset bridges -/

/-- A left `+`-fold accumulates the list sum. -/
theorem foldl_add_eq_sum (l : List ℝ) (a : ℝ) :
    l.foldl (· + ·) a = a + l.sum := by
  induction l generalizing a with
  | nil => simp
  | cons x t ih => rw [List.foldl_cons, ih, List.sum_cons]; ring

/-- A left `s + x*x`-fold accumulates the sum of squares. -/
theorem foldl_addsq_eq_sum (l : List ℝ) (a : ℝ) :
    l.foldl (fun s x => s + x * x) a = a + (l.map (fun x => x * x)).sum := by
  induction l generalizing a with
  | nil => simp
  | cons x t ih => rw [List.foldl_cons, ih, List.map_cons, List.sum_cons]; ring

/-- A `Fin n` sum is the foldl-sum over `finRange n`. -/
theorem finsum_eq_finRange_sum (h : Fin n → ℝ) :
    ∑ i, h i = ((List.finRange n).map h).sum := by
  rw [← List.sum_toFinset _ (List.nodup_finRange n)]
  · simp [List.toFinset_finRange]

/-! ## General snoc-fold read lemmas -/

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
      omega

/-- A fold that only appends never changes an index already inside the accumulator. -/
theorem getD_foldl_snoc_lt (g : List β → ι → β) (d : β) (l : List ι) (acc : List β)
    (k : Nat) (hk : k < acc.length) :
    (l.foldl (fun s a => s ++ [g s a]) acc).getD k d = acc.getD k d := by
  induction l generalizing acc with
  | nil => simp
  | cons a t ih =>
      rw [List.foldl_cons,
        ih (acc ++ [g acc a]) (by rw [List.length_append]; omega),
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
          omega
    _ = g ((l.take k).foldl (fun s a => s ++ [g s a]) []) (l[k]'hk) := by
          rw [htake, List.foldl_append, List.foldl_cons, List.foldl_nil]
          rw [List.getD_append_right _ _ _ _ (le_of_eq hplen), hplen, Nat.sub_self]
          rfl

end FoldSnoc

/-! ## Cholesky: the column-building step

`choleskyColsFn` is a left fold that snocs one column per index. `cholStep` names the function it
appends, so that the read lemmas above can be specialized to it. -/

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
    Spec.choleskyColsFn A
      = (List.finRange n).foldl (fun cols j => cols ++ [cholStep A cols j]) [] := rfl

/-- The diagonal value produced by `cholStep`. -/
theorem cholStep_diag (A : Fin n → Fin n → ℝ) (cols : List (Fin n → ℝ)) (j : Fin n) :
    cholStep A cols j j
      = MathFunctions.sqrt (A j j - (cols.map (fun ck => ck j)).foldl (fun s x => s + x * x) 0) := by
  simp only [cholStep]
  rw [if_neg (lt_irrefl _), if_pos (beq_self_eq_true _)]

/-- The below-diagonal value produced by `cholStep`. -/
theorem cholStep_offdiag (A : Fin n → Fin n → ℝ) (cols : List (Fin n → ℝ)) {i j : Fin n}
    (hij : j.val < i.val) :
    cholStep A cols j i
      = (A i j - (cols.map (fun ck => ck i * ck j)).foldl (fun acc x => acc + x) 0)
          / MathFunctions.sqrt (A j j - (cols.map (fun ck => ck j)).foldl (fun s x => s + x * x) 0) := by
  simp only [cholStep]
  rw [if_neg (by omega), if_neg (by rw [beq_iff_eq]; omega)]

/-- The length-`j` prefix of Cholesky columns built before index `j`. -/
noncomputable def prefixCols (A : Fin n → Fin n → ℝ) (j : Fin n) : List (Fin n → ℝ) :=
  ((List.finRange n).take j.val).foldl (fun cols k => cols ++ [cholStep A cols k]) []

/-- Entry `(i, j)` of the executable Cholesky factor equals `cholStep` evaluated on the prefix. -/
theorem choleskyFn_eq_step (A : Fin n → Fin n → ℝ) (i j : Fin n) :
    Spec.choleskyFn A i j = cholStep A (prefixCols A j) j i := by
  have hlen : j.val < (List.finRange n).length := by rw [List.length_finRange]; exact j.isLt
  show (Spec.choleskyColsFn A).getD j.val (fun _ => 0) i = _
  rw [choleskyColsFn_eq, getD_foldl_snoc_read (fun cols k => cholStep A cols k) (fun _ => 0)
    (List.finRange n) j.val hlen]
  have hj : (List.finRange n)[j.val]'hlen = j := by simp [List.getElem_finRange]
  rw [hj]
  rfl

/-- The prefix of Cholesky columns is exactly the first `j` columns of the final factor `L`,
each presented as the function `r ↦ L r k`. -/
theorem prefix_eq_map (A : Fin n → Fin n → ℝ) (j : Fin n) :
    prefixCols A j
      = ((List.finRange n).take j.val).map (fun k => fun r => Spec.choleskyFn A r k) := by
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
    have hidx : ((List.finRange n).take j.val)[p]'(by rw [hjval]; exact h2) = (⟨p, hpn⟩ : Fin n) := by
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
  rw [finsum_eq_finRange_sum]
  conv_rhs => rw [show (List.finRange n)
    = (List.finRange n).take m ++ (List.finRange n).drop m from (List.take_append_drop _ _).symm]
  rw [List.map_append, List.sum_append]
  have htake : ((List.finRange n).take m).map (fun k => if k.val < m then f k else 0)
      = ((List.finRange n).take m).map f :=
    List.map_congr_left (fun x hx => if_pos (mem_take_finRange hx))
  have hdrop : (((List.finRange n).drop m).map (fun k => if k.val < m then f k else 0)).sum = 0 := by
    rw [List.sum_eq_zero]
    intro y hy
    rw [List.mem_map] at hy
    obtain ⟨x, hx, rfl⟩ := hy
    exact if_neg (by have := mem_drop_finRange hx; omega)
  rw [htake, hdrop, add_zero]

/-- The Cholesky cross-sum equals the masked partial dot product of rows `i` and `j` of `L`. -/
theorem cross_sum_eq (A : Fin n → Fin n → ℝ) (i j : Fin n) :
    ((prefixCols A j).map (fun ck => ck i * ck j)).foldl (fun acc x => acc + x) 0
      = ∑ k : Fin n, if k.val < j.val then Spec.choleskyFn A i k * Spec.choleskyFn A j k else 0 := by
  rw [prefix_eq_map, List.map_map, foldl_add_eq_sum, zero_add,
    show ((fun ck : Fin n → ℝ => ck i * ck j) ∘ fun k => fun r => Spec.choleskyFn A r k)
      = (fun k => Spec.choleskyFn A i k * Spec.choleskyFn A j k) from rfl]
  exact take_map_sum_eq j.val (fun k => Spec.choleskyFn A i k * Spec.choleskyFn A j k)

/-- The Cholesky diagonal sum-of-squares equals the masked partial squared norm of row `j` of `L`. -/
theorem sumsq_eq (A : Fin n → Fin n → ℝ) (j : Fin n) :
    ((prefixCols A j).map (fun ck => ck j)).foldl (fun s x => s + x * x) 0
      = ∑ k : Fin n, if k.val < j.val then Spec.choleskyFn A j k * Spec.choleskyFn A j k else 0 := by
  rw [prefix_eq_map, List.map_map, foldl_addsq_eq_sum, zero_add, List.map_map,
    show ((fun x : ℝ => x * x) ∘ ((fun ck : Fin n → ℝ => ck j) ∘ fun k => fun r => Spec.choleskyFn A r k))
      = (fun k => Spec.choleskyFn A j k * Spec.choleskyFn A j k) from rfl]
  exact take_map_sum_eq j.val (fun k => Spec.choleskyFn A j k * Spec.choleskyFn A j k)

/-! ### Closed-form entries of the executable Cholesky factor -/

/-- Over `ℝ`, the `Context` square root is `Real.sqrt`. -/
theorem mfsqrt_eq (x : ℝ) : MathFunctions.sqrt x = Real.sqrt x := rfl

/-- The diagonal entry of `L` in closed form: `L[j,j] = √(A[j,j] − Σ_{k<j} L[j,k]²)`. -/
theorem choleskyFn_diag_eq (A : Fin n → Fin n → ℝ) (j : Fin n) :
    Spec.choleskyFn A j j
      = Real.sqrt (A j j
          - ∑ k, if k.val < j.val then Spec.choleskyFn A j k * Spec.choleskyFn A j k else 0) := by
  rw [choleskyFn_eq_step, cholStep_diag, sumsq_eq, mfsqrt_eq]

/-- The below-diagonal entry of `L` in closed form:
`L[i,j] = (A[i,j] − Σ_{k<j} L[i,k]·L[j,k]) / L[j,j]` for `i > j`. -/
theorem choleskyFn_offdiag_eq (A : Fin n → Fin n → ℝ) {i j : Fin n} (hij : j.val < i.val) :
    Spec.choleskyFn A i j
      = (A i j - ∑ k, if k.val < j.val then Spec.choleskyFn A i k * Spec.choleskyFn A j k else 0)
          / Spec.choleskyFn A j j := by
  rw [choleskyFn_eq_step A i j, cholStep_offdiag _ _ hij, cross_sum_eq, sumsq_eq, mfsqrt_eq,
    ← choleskyFn_diag_eq]

/-! ### Reconstruction `A = L · Lᵀ`

The diagonal of the rotated/peeled product is reconstructed using the closed-form entries and the
positive-pivot hypothesis (`0 < L[j,j]`), which is exactly the condition under which the executable
Cholesky succeeds over `ℝ`. -/

/-- Per-entry reconstruction for the lower part (`j ≤ i`): the `(i, j)` entry of `L · Lᵀ` is `A i j`. -/
theorem choleskyFn_dot_eq (A : Fin n → Fin n → ℝ)
    (hpos : ∀ j : Fin n, 0 < Spec.choleskyFn A j j) {i j : Fin n} (hji : j.val ≤ i.val) :
    (∑ k, Spec.choleskyFn A i k * Spec.choleskyFn A j k) = A i j := by
  set L := Spec.choleskyFn A with hL
  have key : ∀ k : Fin n, L i k * L j k
      = (if k.val < j.val then L i k * L j k else 0) + (if k = j then L i j * L j j else 0) := by
    intro k
    rcases lt_trichotomy k.val j.val with h | h | h
    · have hne : k ≠ j := fun hk => by rw [hk] at h; exact lt_irrefl _ h
      rw [if_pos h, if_neg hne, add_zero]
    · have hkj : k = j := Fin.ext h
      rw [if_neg (by omega), if_pos hkj, zero_add, hkj]
    · have hne : k ≠ j := fun hk => by rw [hk] at h; exact lt_irrefl _ h
      rw [if_neg (by omega), if_neg hne, add_zero,
        show L j k = 0 from Spec.Factorization.choleskyFn_lower_triangular A h, mul_zero]
  rw [show (∑ k, L i k * L j k)
      = ∑ k, ((if k.val < j.val then L i k * L j k else 0) + (if k = j then L i j * L j j else 0))
      from Finset.sum_congr rfl (fun k _ => key k),
    Finset.sum_add_distrib, Finset.sum_ite_eq' Finset.univ j (fun _ => L i j * L j j)]
  simp only [Finset.mem_univ, if_true]
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

/-- Per-entry reconstruction for all `(i, j)`, using symmetry of `A`. -/
theorem choleskyFn_dot (A : Fin n → Fin n → ℝ) (hsymm : ∀ i j, A i j = A j i)
    (hpos : ∀ j : Fin n, 0 < Spec.choleskyFn A j j) (i j : Fin n) :
    (∑ k, Spec.choleskyFn A i k * Spec.choleskyFn A j k) = A i j := by
  rcases le_total j.val i.val with h | h
  · exact choleskyFn_dot_eq A hpos h
  · rw [show (∑ k, Spec.choleskyFn A i k * Spec.choleskyFn A j k)
        = ∑ k, Spec.choleskyFn A j k * Spec.choleskyFn A i k
        from Finset.sum_congr rfl (fun k _ => mul_comm _ _),
      choleskyFn_dot_eq A hpos h, hsymm j i]

/-- **Exact Cholesky reconstruction.** For a symmetric `A` whose executable Cholesky pivots are all
positive (`0 < L[j,j]`, the success condition over `ℝ`), the factor `L = choleskyFn A` is a genuine
Cholesky factor: lower-triangular with `A = L · Lᵀ`. -/
theorem isCholesky_of_pos (A : Fin n → Fin n → ℝ) (hsymm : ∀ i j, A i j = A j i)
    (hpos : ∀ j : Fin n, 0 < Spec.choleskyFn A j j) :
    Spec.Factorization.IsCholesky (Matrix.of A) (Matrix.of (Spec.choleskyFn A)) := by
  refine ⟨?_, ?_⟩
  · intro a b hab
    show Spec.choleskyFn A a b = 0
    exact Spec.Factorization.choleskyFn_lower_triangular A (Fin.lt_def.mp hab)
  · ext i j
    rw [Matrix.mul_apply]
    simp only [Matrix.of_apply, Matrix.transpose_apply]
    exact (choleskyFn_dot A hsymm hpos i j).symm

/-- **Tensor-level Cholesky reconstruction.** For a symmetric tensor `A` whose `choleskySpec` pivots
are positive, every entry of `A` is reconstructed by `L · Lᵀ`:
`A[i,j] = Σ_k L[i,k] · L[j,k]`, with `L = choleskySpec A`. -/
theorem choleskySpec_reconstruction (A : Spec.Tensor ℝ (.dim n (.dim n .scalar)))
    (hsymm : ∀ i j, Spec.get2 A i j = Spec.get2 A j i)
    (hpos : ∀ j : Fin n, 0 < Spec.get2 (Spec.choleskySpec A) j j) (i j : Fin n) :
    Spec.get2 A i j
      = ∑ k, Spec.get2 (Spec.choleskySpec A) i k * Spec.get2 (Spec.choleskySpec A) j k := by
  have hg : ∀ a b, Spec.get2 (Spec.choleskySpec A) a b = Spec.choleskyFn (Spec.toMatFn A) a b := by
    intro a b
    rw [show Spec.choleskySpec A = Spec.ofMatFn (Spec.choleskyFn (Spec.toMatFn A)) from rfl,
      Spec.Factorization.get2_ofMatFn]
  simp only [hg]
  show Spec.toMatFn A i j = _
  refine (choleskyFn_dot (Spec.toMatFn A) (fun a b => hsymm a b) (fun b => ?_) i j).symm
  rw [← hg b b]; exact hpos b

/-! ## QR (modified Gram–Schmidt): exact reconstruction `A = Q · R`

`gramSchmidtFn` threads a `GSState` that snocs a column onto *both* the `Q`-list and the `R`-list at
each index. Crucially the appended values depend only on the `Q`-history (`st.qs`), never on the
`R`-history, so the `Q`-list is itself a single-list snoc-fold (`gs_proj_qs`) and the `R`-list is the
`Q`-prefix-indexed tail `rTail`. -/

section QR

variable {m : Nat}

open Spec (GSState)

/-- Column `j` of `A` as a function of the row. -/
noncomputable def gsA (A : Fin m → Fin n → ℝ) (j : Fin n) : Fin m → ℝ := fun i => A i j

/-- The `R` off-diagonal entries `rₖⱼ = qₖ · a` for the columns `qs` built so far. -/
noncomputable def gsRkjs (A : Fin m → Fin n → ℝ) (qs : List (Fin m → ℝ)) (j : Fin n) : List ℝ :=
  qs.map (fun qk => Spec.dotFn qk (gsA A j))

/-- The orthogonalized (not-yet-normalized) vector `v = a − Σ rₖⱼ qₖ`. -/
noncomputable def gsV (A : Fin m → Fin n → ℝ) (qs : List (Fin m → ℝ)) (j : Fin n) : Fin m → ℝ :=
  fun i => gsA A j i
    - (List.zip qs (gsRkjs A qs j)).foldl (fun acc (qk, r) => acc + r * qk i) 0

/-- The diagonal `R` entry `rⱼⱼ = ‖v‖`. -/
noncomputable def gsRjj (A : Fin m → Fin n → ℝ) (qs : List (Fin m → ℝ)) (j : Fin n) : ℝ :=
  Spec.normFn (gsV A qs j)

/-- The `Q` column appended at index `j`: `v / rⱼⱼ` (or `0` when `rⱼⱼ = 0`). -/
noncomputable def qStep (A : Fin m → Fin n → ℝ) (qs : List (Fin m → ℝ)) (j : Fin n) : Fin m → ℝ :=
  fun i => if Context.gtBool (gsRjj A qs j) 0 then gsV A qs j i / gsRjj A qs j else 0

/-- The `R` column appended at index `j`: `rₖⱼ` below the diagonal, `rⱼⱼ` on it, `0` above. -/
noncomputable def rStep (A : Fin m → Fin n → ℝ) (qs : List (Fin m → ℝ)) (j : Fin n) : Fin n → ℝ :=
  fun k => if k.val < j.val then (gsRkjs A qs j).getD k.val 0
    else if k.val == j.val then gsRjj A qs j else 0

/-- `gramSchmidtFn` as the dual-list snoc-fold appending `qStep`/`rStep`. -/
theorem gramSchmidtFn_eq (A : Fin m → Fin n → ℝ) :
    Spec.gramSchmidtFn A
      = (List.finRange n).foldl
          (fun st j => (⟨st.qs ++ [qStep A st.qs j], st.rcols ++ [rStep A st.qs j]⟩ : GSState m n ℝ))
          ⟨[], []⟩ := rfl

/-- The `Q`-list projection of the structure fold is the single-list `qStep` snoc-fold. -/
theorem gs_proj_qs (A : Fin m → Fin n → ℝ) (l : List (Fin n)) (q0 : List (Fin m → ℝ))
    (r0 : List (Fin n → ℝ)) :
    (l.foldl (fun st j => (⟨st.qs ++ [qStep A st.qs j], st.rcols ++ [rStep A st.qs j]⟩ : GSState m n ℝ))
        ⟨q0, r0⟩).qs
      = l.foldl (fun qs j => qs ++ [qStep A qs j]) q0 := by
  induction l generalizing q0 r0 with
  | nil => rfl
  | cons a t ih => simp only [List.foldl_cons]; exact ih _ _

/-- The `Q` columns built before index `j`. -/
noncomputable def qsPrefix (A : Fin m → Fin n → ℝ) (j : Fin n) : List (Fin m → ℝ) :=
  ((List.finRange n).take j.val).foldl (fun qs k => qs ++ [qStep A qs k]) []

/-- The `R`-list tail: the `R` columns produced from `Q`-prefix `q0` over the indices `l`. -/
noncomputable def rTail (A : Fin m → Fin n → ℝ) (q0 : List (Fin m → ℝ)) : List (Fin n) →
    List (Fin n → ℝ)
  | [] => []
  | j :: rest => rStep A q0 j :: rTail A (q0 ++ [qStep A q0 j]) rest

/-- The structure fold splits into the `qStep` snoc-fold (`Q`-list) and the `rTail` (`R`-list). -/
theorem gs_fold_split (A : Fin m → Fin n → ℝ) (l : List (Fin n)) (q0 : List (Fin m → ℝ))
    (r0 : List (Fin n → ℝ)) :
    (l.foldl (fun st j => (⟨st.qs ++ [qStep A st.qs j], st.rcols ++ [rStep A st.qs j]⟩ : GSState m n ℝ))
        ⟨q0, r0⟩)
      = ⟨l.foldl (fun qs j => qs ++ [qStep A qs j]) q0, r0 ++ rTail A q0 l⟩ := by
  induction l generalizing q0 r0 with
  | nil => simp [rTail]
  | cons j rest ih =>
      simp only [List.foldl_cons, rTail]
      rw [ih]
      simp [List.append_assoc]

/-- Reading the `k`-th element of `rTail` recovers `rStep` applied to the length-`k` `Q`-prefix. -/
theorem rTail_getD (A : Fin m → Fin n → ℝ) (q0 : List (Fin m → ℝ)) (l : List (Fin n)) (k : Nat)
    (hk : k < l.length) (d : Fin n → ℝ) :
    (rTail A q0 l).getD k d
      = rStep A ((l.take k).foldl (fun qs j => qs ++ [qStep A qs j]) q0) (l[k]'hk) := by
  induction l generalizing q0 k with
  | nil => simp at hk
  | cons j rest ih =>
      cases k with
      | zero => simp [rTail]
      | succ k' =>
          simp only [rTail, List.getD_cons_succ, List.take_succ_cons, List.foldl_cons,
            List.getElem_cons_succ]
          exact ih (q0 ++ [qStep A q0 j]) k' (by simpa using hk)

/-- Semantics of the `Context` `>` test over `ℝ`. -/
theorem gtBool_true_iff {x y : ℝ} : Context.gtBool x y = true ↔ y < x := by
  unfold Context.gtBool; exact decide_eq_true_iff

/-- A left fold `acc + h x` accumulates the mapped list sum. -/
theorem foldl_addf_eq_sum {β : Type _} (h : β → ℝ) (l : List β) (a : ℝ) :
    l.foldl (fun acc x => acc + h x) a = a + (l.map h).sum := by
  induction l generalizing a with
  | nil => simp
  | cons x t ih => rw [List.foldl_cons, ih, List.map_cons, List.sum_cons]; ring

/-! ### Entries of the executable `Q` and `R` factors -/

/-- Entry `(i, k)` of the `Q` factor produced by `gramSchmidtFn`. -/
noncomputable def Qmat (A : Fin m → Fin n → ℝ) (i : Fin m) (k : Fin n) : ℝ :=
  (Spec.gramSchmidtFn A).qs.getD k.val (fun _ => 0) i

/-- Entry `(k, j)` of the `R` factor produced by `gramSchmidtFn`. -/
noncomputable def Rmat (A : Fin m → Fin n → ℝ) (k j : Fin n) : ℝ :=
  (Spec.gramSchmidtFn A).rcols.getD j.val (fun _ => 0) k

/-- Column `k` of `Q` as a function of the row. -/
noncomputable def Qcol (A : Fin m → Fin n → ℝ) (k : Fin n) : Fin m → ℝ := fun r => Qmat A r k

/-- Closed form of a `Q` entry: `qStep` evaluated on the `Q`-prefix. -/
theorem Qmat_eq (A : Fin m → Fin n → ℝ) (i : Fin m) (k : Fin n) :
    Qmat A i k = qStep A (qsPrefix A k) k i := by
  have hqs : (Spec.gramSchmidtFn A).qs
      = (List.finRange n).foldl (fun qs j => qs ++ [qStep A qs j]) [] := by
    rw [gramSchmidtFn_eq]; exact gs_proj_qs A (List.finRange n) [] []
  unfold Qmat
  rw [hqs, getD_foldl_snoc_read (fun qs j => qStep A qs j) (fun _ => 0) (List.finRange n) k.val
    (by rw [List.length_finRange]; exact k.isLt)]
  have hk : (List.finRange n)[k.val]'(by rw [List.length_finRange]; exact k.isLt) = k := by
    simp [List.getElem_finRange]
  rw [hk]; rfl

/-- Closed form of an `R` entry: `rStep` evaluated on the `Q`-prefix. -/
theorem Rmat_eq (A : Fin m → Fin n → ℝ) (k j : Fin n) :
    Rmat A k j = rStep A (qsPrefix A j) j k := by
  have hrc : (Spec.gramSchmidtFn A).rcols = rTail A [] (List.finRange n) := by
    rw [gramSchmidtFn_eq, gs_fold_split]; simp
  unfold Rmat
  rw [hrc, rTail_getD A [] (List.finRange n) j.val (by rw [List.length_finRange]; exact j.isLt)]
  have hk : (List.finRange n)[j.val]'(by rw [List.length_finRange]; exact j.isLt) = j := by
    simp [List.getElem_finRange]
  rw [hk]; rfl

/-- `R` is upper-triangular: entries strictly below the diagonal vanish. -/
theorem rStep_above (A : Fin m → Fin n → ℝ) (qs : List (Fin m → ℝ)) {j k : Fin n}
    (hjk : j.val < k.val) : rStep A qs j k = 0 := by
  simp only [rStep]; rw [if_neg (by omega), if_neg (by rw [beq_iff_eq]; omega)]

/-- The diagonal `R` entry is `rⱼⱼ`. -/
theorem rStep_diag (A : Fin m → Fin n → ℝ) (qs : List (Fin m → ℝ)) (j : Fin n) :
    rStep A qs j j = gsRjj A qs j := by
  simp only [rStep]; rw [if_neg (lt_irrefl _), if_pos (beq_self_eq_true _)]

/-- The `Q` column when the pivot is positive: `qⱼ = v / rⱼⱼ`. -/
theorem qStep_pos (A : Fin m → Fin n → ℝ) (qs : List (Fin m → ℝ)) (j : Fin n)
    (h : 0 < gsRjj A qs j) (i : Fin m) :
    qStep A qs j i = gsV A qs j i / gsRjj A qs j := by
  simp only [qStep]; rw [if_pos (gtBool_true_iff.mpr h)]

/-! ### The orthogonalization sum as a `Finset` sum -/

set_option linter.unusedSimpArgs false in
/-- The zip-fold defining `v` collapses to a single map-fold over the `Q` columns. -/
theorem cross_fold_eq (qs : List (Fin m → ℝ)) (g : (Fin m → ℝ) → ℝ) (i : Fin m) (a : ℝ) :
    (List.zip qs (qs.map g)).foldl (fun acc (qk, r) => acc + r * qk i) a
      = a + (qs.map (fun qk => g qk * qk i)).sum := by
  induction qs generalizing a with
  | nil => simp
  | cons x xs ih =>
      simp only [List.map_cons, List.zip_cons_cons, List.foldl_cons]
      rw [ih]; simp only [List.map_cons, List.sum_cons]; ring

/-- Closed form of `v i`: `A i j` minus the partial projection sum. -/
theorem gsV_eq (A : Fin m → Fin n → ℝ) (qs : List (Fin m → ℝ)) (j : Fin n) (i : Fin m) :
    gsV A qs j i = gsA A j i - (qs.map (fun qk => Spec.dotFn qk (gsA A j) * qk i)).sum := by
  unfold gsV gsRkjs
  rw [cross_fold_eq qs (fun qk => Spec.dotFn qk (gsA A j)) i 0, zero_add]

/-- Length of the `Q`-prefix list. -/
theorem qsPrefix_length (A : Fin m → Fin n → ℝ) (j : Fin n) : (qsPrefix A j).length = j.val := by
  unfold qsPrefix
  rw [length_foldl_snoc (fun qs k => qStep A qs k), List.length_nil, Nat.zero_add, List.length_take,
    List.length_finRange, Nat.min_eq_left (le_of_lt j.isLt)]

/-- The `Q`-prefix is exactly the first `j` columns of the final factor `Q`. -/
theorem qsPrefix_eq_map (A : Fin m → Fin n → ℝ) (j : Fin n) :
    qsPrefix A j = ((List.finRange n).take j.val).map (fun k => Qcol A k) := by
  have hjval : ((List.finRange n).take j.val).length = j.val := by
    rw [List.length_take, List.length_finRange, Nat.min_eq_left (le_of_lt j.isLt)]
  apply List.ext_getElem
  · unfold qsPrefix
    rw [length_foldl_snoc (fun qs k => qStep A qs k), List.length_nil, Nat.zero_add,
      List.length_map]
  · intro p h1 h2
    rw [List.length_map, hjval] at h2
    have hpn : p < n := lt_trans h2 j.isLt
    rw [List.getElem_map]
    have hidx : ((List.finRange n).take j.val)[p]'(by rw [hjval]; exact h2) = (⟨p, hpn⟩ : Fin n) := by
      rw [List.getElem_take, List.getElem_finRange]; exact Fin.ext rfl
    rw [show (qsPrefix A j)[p]'h1 = (qsPrefix A j).getD p (fun _ => 0) from
      (List.getD_eq_getElem _ _ h1).symm]
    unfold qsPrefix
    rw [getD_foldl_snoc_read (fun qs k => qStep A qs k) (fun _ => 0)
      ((List.finRange n).take j.val) p (by rw [hjval]; exact h2)]
    rw [List.take_take, Nat.min_eq_left (le_of_lt h2), hidx]
    funext r
    rw [show Qcol A (⟨p, hpn⟩ : Fin n) r = Qmat A r ⟨p, hpn⟩ from rfl, Qmat_eq]
    rfl

/-- `getD` commutes with `dotFn`-mapping when the index is in range. -/
theorem getD_map_dotFn (qs : List (Fin m → ℝ)) (a : Fin m → ℝ) (k : Nat) (hk : k < qs.length) :
    (qs.map (fun qk => Spec.dotFn qk a)).getD k 0 = Spec.dotFn (qs.getD k (fun _ => 0)) a := by
  rw [List.getD_eq_getElem _ _ (by rw [List.length_map]; exact hk), List.getElem_map,
    List.getD_eq_getElem _ _ hk]

/-- A `Q`-prefix entry equals the final `Q` column at that index. -/
theorem qsPrefix_getD (A : Fin m → Fin n → ℝ) {k j : Fin n} (hkj : k.val < j.val) :
    (qsPrefix A j).getD k.val (fun _ => 0) = Qcol A k := by
  rw [qsPrefix_eq_map,
    List.getD_eq_getElem _ _ (by rw [List.length_map, List.length_take, List.length_finRange,
      Nat.min_eq_left (le_of_lt j.isLt)]; exact hkj),
    List.getElem_map]
  congr 1
  rw [List.getElem_take, List.getElem_finRange]; exact Fin.ext rfl

/-- The below-diagonal `R` entry is the inner product of the corresponding `Q` column with column `j`. -/
theorem R_below (A : Fin m → Fin n → ℝ) {k j : Fin n} (hkj : k.val < j.val) :
    Rmat A k j = Spec.dotFn (Qcol A k) (gsA A j) := by
  rw [Rmat_eq]; simp only [rStep]; rw [if_pos hkj]; unfold gsRkjs
  rw [getD_map_dotFn (qsPrefix A j) (gsA A j) k.val (by rw [qsPrefix_length]; exact hkj),
    qsPrefix_getD A hkj]

/-- The projection sum equals the masked partial sum `Σ_{k<j} R[k,j]·Q[i,k]`. -/
theorem cross_sum_qr (A : Fin m → Fin n → ℝ) (i : Fin m) (j : Fin n) :
    ((qsPrefix A j).map (fun qk => Spec.dotFn qk (gsA A j) * qk i)).sum
      = ∑ k, if k.val < j.val then Rmat A k j * Qmat A i k else 0 := by
  rw [qsPrefix_eq_map]
  rw [List.map_map]
  rw [take_map_sum_eq]
  apply Finset.sum_congr rfl
  intro k _
  by_cases hkj : k.val < j.val
  · rw [if_pos hkj, if_pos hkj]
    show Spec.dotFn (Qcol A k) (gsA A j) * Qmat A i k = Rmat A k j * Qmat A i k
    rw [R_below A hkj]
  · rw [if_neg hkj, if_neg hkj]

/-! ### Exact reconstruction `A = Q · R` -/

/-- `R` is upper-triangular: entries strictly below the diagonal vanish. -/
theorem Rmat_upper_triangular (A : Fin m → Fin n → ℝ) {k j : Fin n} (hjk : j.val < k.val) :
    Rmat A k j = 0 := by
  rw [Rmat_eq]; exact rStep_above A (qsPrefix A j) hjk

/-- **Per-entry QR reconstruction.** When every `R` pivot is positive (`0 < R[j,j]`, the full
column-rank success condition), `A[i,j] = Σ_k Q[i,k]·R[k,j]`. -/
theorem qr_reconstruction (A : Fin m → Fin n → ℝ) (hrank : ∀ j : Fin n, 0 < Rmat A j j)
    (i : Fin m) (j : Fin n) :
    A i j = ∑ k, Qmat A i k * Rmat A k j := by
  have key : ∀ k : Fin n, Qmat A i k * Rmat A k j
      = (if k.val < j.val then Qmat A i k * Rmat A k j else 0)
        + (if k = j then Qmat A i j * Rmat A j j else 0) := by
    intro k
    rcases lt_trichotomy k.val j.val with h | h | h
    · have hne : k ≠ j := fun hk => by rw [hk] at h; exact lt_irrefl _ h
      rw [if_pos h, if_neg hne, add_zero]
    · have hkj : k = j := Fin.ext h
      rw [if_neg (by omega), if_pos hkj, zero_add, hkj]
    · have hne : k ≠ j := fun hk => by rw [hk] at h; exact lt_irrefl _ h
      rw [if_neg (by omega), if_neg hne, add_zero, Rmat_upper_triangular A h, mul_zero]
  rw [show (∑ k, Qmat A i k * Rmat A k j)
      = ∑ k, ((if k.val < j.val then Qmat A i k * Rmat A k j else 0)
        + (if k = j then Qmat A i j * Rmat A j j else 0))
      from Finset.sum_congr rfl (fun k _ => key k),
    Finset.sum_add_distrib, Finset.sum_ite_eq' Finset.univ j (fun _ => Qmat A i j * Rmat A j j)]
  simp only [Finset.mem_univ, if_true]
  have hρpos : 0 < gsRjj A (qsPrefix A j) j := by
    have h := hrank j; rwa [Rmat_eq, rStep_diag] at h
  have hdiag : Qmat A i j * Rmat A j j = gsV A (qsPrefix A j) j i := by
    rw [Qmat_eq, qStep_pos A (qsPrefix A j) j hρpos,
      show Rmat A j j = gsRjj A (qsPrefix A j) j from by rw [Rmat_eq]; exact rStep_diag _ _ j,
      div_mul_eq_mul_div, mul_div_assoc, div_self (ne_of_gt hρpos), mul_one]
  rw [hdiag, gsV_eq, cross_sum_qr,
    show gsA A j i = A i j from rfl,
    show (∑ k, if k.val < j.val then Qmat A i k * Rmat A k j else 0)
      = (∑ k, if k.val < j.val then Rmat A k j * Qmat A i k else 0)
      from Finset.sum_congr rfl (fun k _ => by
        by_cases hkj : k.val < j.val
        · rw [if_pos hkj, if_pos hkj, mul_comm]
        · rw [if_neg hkj, if_neg hkj])]
  ring

/-- **Matrix-level QR reconstruction.** `A = Q · R` for the executable Gram–Schmidt factors,
under positive `R` pivots (full column rank). -/
theorem qr_mul_eq (A : Fin m → Fin n → ℝ) (hrank : ∀ j : Fin n, 0 < Rmat A j j) :
    Matrix.of A = Matrix.of (fun i k => Qmat A i k) * Matrix.of (fun k j => Rmat A k j) := by
  ext i j
  rw [Matrix.mul_apply]
  simp only [Matrix.of_apply]
  exact qr_reconstruction A hrank i j

/-- **Tensor-level QR reconstruction.** For a tensor `A` whose `qrSpec` `R`-pivots are positive
(full column rank), every entry of `A` is reconstructed by `Q · R`:
`A[i,j] = Σ_k Q[i,k]·R[k,j]`, with `Q = qrQSpec A`, `R = qrRSpec A`. -/
theorem qrSpec_reconstruction (A : Spec.Tensor ℝ (.dim m (.dim n .scalar)))
    (hrank : ∀ j : Fin n, 0 < Spec.get2 (Spec.qrRSpec A) j j) (i : Fin m) (j : Fin n) :
    Spec.get2 A i j
      = ∑ k, Spec.get2 (Spec.qrQSpec A) i k * Spec.get2 (Spec.qrRSpec A) k j := by
  have hQ : ∀ a b, Spec.get2 (Spec.qrQSpec A) a b = Qmat (Spec.toMatFn A) a b := fun _ _ => rfl
  have hR : ∀ a b, Spec.get2 (Spec.qrRSpec A) a b = Rmat (Spec.toMatFn A) a b := fun _ _ => rfl
  simp only [hQ, hR]
  show Spec.toMatFn A i j = _
  exact qr_reconstruction (Spec.toMatFn A) (fun b => by rw [← hR b b]; exact hrank b) i j

end QR

end Spec.Factorization.Reconstruction
