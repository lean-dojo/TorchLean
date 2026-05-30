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
# Exact reconstruction of the finite Cholesky factorization

This file proves the *exact* algebraic reconstruction of the finite executable Cholesky
factorization from [`NN.Spec.Core.Tensor.Factorizations`](../../../Spec/Core/Tensor/Factorizations.lean),
the increment promised in `NN.Proofs.Tensor.Basic.Factorizations`. Unlike the iterative Jacobi/SVD
routines (whose reconstruction is only an a-posteriori residual certificate), Cholesky is a *finite*
construction, so over `ℝ` it reconstructs its input on the nose under the success hypothesis.

## Main result

`isCholesky_of_pos`: for a symmetric `A : Fin n → Fin n → ℝ` whose executable Cholesky pivots are all
positive (`0 < choleskyFn A j j`, the exact condition under which the algorithm succeeds over `ℝ`),
the factor `L = choleskyFn A` satisfies the specification `Spec.Factorization.IsCholesky`:
it is lower-triangular and `A = L · Lᵀ`. `choleskySpec_reconstruction` is the tensor-level corollary.

## Method

The executable factor is built by a `List.foldl` that snocs one column per index. The core technical
device is `getD_foldl_snoc_read`, a general lemma reading the `j`-th element of such a fold as the
step function applied to the length-`j` prefix. From it, `prefix_eq_map` identifies the prefix of
columns with the first `j` columns of the final factor `L`, and `take_map_sum_eq` turns the code's
`List.foldl` sums into masked `Finset` partial sums. The positive-pivot hypothesis discharges the two
side conditions (`√` radicand `> 0` for the diagonal, divisor `≠ 0` for the below-diagonal entries).

## Scope

The QR factorization's exact reconstruction (`A = Q · R` from `gramSchmidtFn`, plus the orthonormality
`Qᵀ Q = 1`) is the remaining finite-fold increment. It needs analogous read lemmas for the
`GSState` *dual-list* structure-fold (the step writes both `qs` and `rcols`), and `Qᵀ Q = 1`
additionally requires the Gram–Schmidt orthogonality invariant, which Mathlib only provides for its
own `gramSchmidt`, not for this executable variant.
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

end Spec.Factorization.Reconstruction
