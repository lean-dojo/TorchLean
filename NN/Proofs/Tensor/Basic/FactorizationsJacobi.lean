/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Tensor.Basic.Factorizations

/-!
# The cyclic Jacobi run is faithful (orthogonality + orthogonal similarity)

The a-posteriori residual certificate in
[`NN.Proofs.Tensor.Basic.Factorizations`](./Factorizations.lean)
(`symEig_reconstruction_residual`, `symEig_frobenius_residual`, `isSymEig_of_diagonal`) is stated
*conditionally*: it assumes the two algebraic premises `Vᵀ V = 1` (the accumulated eigenvector matrix
is orthogonal) and `A = V · Af · Vᵀ` (the rotated matrix is an orthogonal similarity of the input).
Both are *exact, finite, a-priori* facts about the executable `Spec.arrJacobiRun` — they need no
Jacobi convergence theory. This file proves them and thereby discharges the hypotheses, turning the
certificate into an **unconditional** statement about the real solver output.

The development is a refinement bridge from the strict `Array (Array ℝ)` representation the iteration
runs over to Mathlib `Matrix (Fin n) (Fin n) ℝ`:

* `toM` reads an array matrix as a `Matrix`; `toM_matMul`/`toM_tr`/`toM_id` show the array operations
  realise the corresponding matrix operations.
* `givens_orthogonal` is the one genuinely-new piece: each Givens rotation `arrGivens n p q c s` with
  `c² + s² = 1` is an orthogonal matrix (`Jᵀ J = 1`).
* `JacInv` is the loop invariant `Vᵀ V = 1 ∧ A₀ = V · A · Vᵀ`; `jacInv_rotate`/`jacInv_sweep`/
  `jacInv_run` propagate it through one rotation, one sweep, and the whole run.
* `jacobi_orthogonal` and `jacobi_similarity` are the discharged premises for the actual
  `symEigJacobiSpec` output, and `symEigJacobi_*` re-state the residual certificate unconditionally.
-/

@[expose] public section

namespace Spec.Factorization

open Matrix
open scoped BigOperators

variable {n : Nat}

/-! ## Reading array matrices as Mathlib matrices -/

/-- Reading position `i` (in bounds) of `Array.ofFn f` returns `f ⟨i, _⟩`. -/
theorem getD_ofFn {β : Type} (f : Fin n → β) (i : Nat) (hi : i < n) (d : β) :
    (Array.ofFn f).getD i d = f ⟨i, hi⟩ := by
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem (by simpa using hi),
    Option.getD_some, Array.getElem_ofFn]

/-- Reading entry `(i, j)` of a doubly-`ofFn` array matrix returns the underlying function value. -/
theorem arrGet_ofFn₂ (F : Fin n → Fin n → ℝ) (i j : Fin n) :
    Spec.arrGet (Array.ofFn (fun a : Fin n => Array.ofFn (fun b : Fin n => F a b))) i.val j.val
      = F i j := by
  unfold Spec.arrGet
  rw [getD_ofFn (fun a : Fin n => Array.ofFn (fun b : Fin n => F a b)) i.val i.isLt #[],
    getD_ofFn (fun b : Fin n => F ⟨i.val, i.isLt⟩ b) j.val j.isLt 0]

/-- View an `Array (Array ℝ)` as a `Matrix (Fin n) (Fin n) ℝ`. -/
noncomputable def toM (n : Nat) (M : Array (Array ℝ)) : Matrix (Fin n) (Fin n) ℝ :=
  Matrix.of (fun i j => Spec.arrGet M i.val j.val)

@[simp] theorem toM_apply (M : Array (Array ℝ)) (i j : Fin n) :
    toM n M i j = Spec.arrGet M i.val j.val := rfl

/-- The array matrix product realises the matrix product. -/
theorem toM_matMul (X Y : Array (Array ℝ)) :
    toM n (Spec.arrMatMul n X Y) = toM n X * toM n Y := by
  ext i j
  rw [Matrix.mul_apply]
  simp only [toM_apply]
  unfold Spec.arrMatMul
  rw [arrGet_ofFn₂]
  exact Spec.finRange_foldl_add_eq_finset_sum
    (fun k => Spec.arrGet X i.val k.val * Spec.arrGet Y k.val j.val)

/-- The array transpose realises the matrix transpose. -/
theorem toM_tr (X : Array (Array ℝ)) : toM n (Spec.arrTr n X) = (toM n X)ᵀ := by
  ext i j
  rw [Matrix.transpose_apply]
  simp only [toM_apply]
  unfold Spec.arrTr
  rw [arrGet_ofFn₂]

/-- The array identity realises the matrix identity. -/
theorem toM_id : toM n (Spec.arrId n) = 1 := by
  ext i j
  simp only [toM_apply]
  unfold Spec.arrId
  rw [arrGet_ofFn₂]
  by_cases h : i = j
  · subst h; simp
  · rw [Matrix.one_apply_ne h]
    simp [Fin.val_ne_of_ne h]

/-! ## The Givens rotation is orthogonal -/

/-- Entrywise value of the Givens array matrix (boolean conditions). -/
theorem toM_givens_apply (p q : Nat) (c s : ℝ) (a b : Fin n) :
    toM n (Spec.arrGivens n p q c s) a b
      = (if a.val == p && b.val == p then c
         else if a.val == q && b.val == q then c
         else if a.val == p && b.val == q then s
         else if a.val == q && b.val == p then -s
         else if a.val == b.val then 1 else 0) := by
  simp only [toM_apply]
  unfold Spec.arrGivens
  rw [arrGet_ofFn₂]

/-- Entrywise value of the Givens array matrix (propositional conditions). -/
theorem toM_givens_apply' (p q : Nat) (c s : ℝ) (a b : Fin n) :
    toM n (Spec.arrGivens n p q c s) a b
      = (if a.val = p ∧ b.val = p then c
         else if a.val = q ∧ b.val = q then c
         else if a.val = p ∧ b.val = q then s
         else if a.val = q ∧ b.val = p then -s
         else if a.val = b.val then 1 else 0) := by
  rw [toM_givens_apply]
  simp only [Bool.and_eq_true, beq_iff_eq]

/-- Column `p` of the Givens matrix: `c` at row `p`, `-s` at row `q`, `0` elsewhere. -/
theorem givens_col_fp (p q : Nat) (hp : p < n) (hq : q < n) (hpq : p ≠ q) (c s : ℝ) (k : Fin n) :
    toM n (Spec.arrGivens n p q c s) k ⟨p, hp⟩
      = (if k = ⟨p, hp⟩ then c else if k = ⟨q, hq⟩ then -s else 0) := by
  rw [toM_givens_apply']
  by_cases hkp : k.val = p
  · simp [hkp, Fin.ext_iff]
  · by_cases hkq : k.val = q
    · simp [hkq, hpq, Ne.symm hpq, Fin.ext_iff]
    · simp [hkp, hkq, hpq, Fin.ext_iff]

/-- Column `q` of the Givens matrix: `s` at row `p`, `c` at row `q`, `0` elsewhere. -/
theorem givens_col_fq (p q : Nat) (hp : p < n) (hq : q < n) (hpq : p ≠ q) (c s : ℝ) (k : Fin n) :
    toM n (Spec.arrGivens n p q c s) k ⟨q, hq⟩
      = (if k = ⟨p, hp⟩ then s else if k = ⟨q, hq⟩ then c else 0) := by
  rw [toM_givens_apply']
  by_cases hkp : k.val = p
  · simp [hkp, hpq, Ne.symm hpq, Fin.ext_iff]
  · by_cases hkq : k.val = q
    · simp [hkq, Ne.symm hpq, Fin.ext_iff]
    · simp [hkp, hkq, Ne.symm hpq, Fin.ext_iff]

/-- Any other column `o ∉ {p, q}` of the Givens matrix is the `o`-th standard basis vector. -/
theorem givens_col_other (p q : Nat) (c s : ℝ) (o k : Fin n)
    (hop : o.val ≠ p) (hoq : o.val ≠ q) :
    toM n (Spec.arrGivens n p q c s) k o = (if k = o then 1 else 0) := by
  rw [toM_givens_apply']
  by_cases hko : k = o
  · simp [hko, hop, hoq]
  · simp [hop, hoq, hko, Fin.val_ne_of_ne hko]

/-- A sum of products of two indicator functions is the Kronecker delta. -/
private theorem sum_ite_mul_ite (i j : Fin n) :
    ∑ k : Fin n, (if k = i then (1 : ℝ) else 0) * (if k = j then 1 else 0)
      = if i = j then 1 else 0 := by
  by_cases hij : i = j
  · subst hij
    have hterm : ∀ k : Fin n,
        (if k = i then (1 : ℝ) else 0) * (if k = i then 1 else 0) = if k = i then 1 else 0 :=
      fun k => by by_cases hk : k = i <;> simp [hk]
    rw [if_pos rfl, Finset.sum_congr rfl (fun k _ => hterm k), Finset.sum_ite_eq']
    simp
  · rw [if_neg hij]
    refine Finset.sum_eq_zero (fun k _ => ?_)
    by_cases hki : k = i
    · subst hki; simp [hij]
    · rw [if_neg hki, zero_mul]

/-- A sum of products of two functions each supported on `{fp, fq}` (with `fp ≠ fq`). -/
private theorem sum_two_supp (fp fq : Fin n) (hfpq : fp ≠ fq) (x1 y1 x2 y2 : ℝ) :
    ∑ k : Fin n, (if k = fp then x1 else if k = fq then y1 else 0)
                 * (if k = fp then x2 else if k = fq then y2 else 0)
      = x1 * x2 + y1 * y2 := by
  have hterm : ∀ k : Fin n,
      (if k = fp then x1 else if k = fq then y1 else 0)
        * (if k = fp then x2 else if k = fq then y2 else 0)
        = (if k = fp then x1 * x2 else 0) + (if k = fq then y1 * y2 else 0) := by
    intro k
    by_cases hkp : k = fp
    · subst hkp; simp [hfpq]
    · by_cases hkq : k = fq
      · subst hkq; simp [hkp]
      · simp [hkp, hkq]
  rw [Finset.sum_congr rfl (fun k _ => hterm k), Finset.sum_add_distrib,
    Finset.sum_ite_eq', Finset.sum_ite_eq']
  simp

/-- A function supported on `{fp, fq}` times an indicator at `o ∉ {fp, fq}` sums to zero. -/
private theorem sum_two_supp_mul_ite (fp fq o : Fin n) (hop : o ≠ fp) (hoq : o ≠ fq) (x1 y1 : ℝ) :
    ∑ k : Fin n, (if k = fp then x1 else if k = fq then y1 else 0) * (if k = o then (1 : ℝ) else 0)
      = 0 := by
  refine Finset.sum_eq_zero (fun k _ => ?_)
  by_cases hko : k = o
  · subst hko; rw [if_neg hop, if_neg hoq, zero_mul]
  · rw [if_neg hko, mul_zero]

/-- **Givens rotation is orthogonal.** For `c² + s² = 1` and `p ≠ q`, `Jᵀ J = 1`. -/
theorem givens_orthogonal (p q : Nat) (hp : p < n) (hq : q < n) (hpq : p ≠ q) (c s : ℝ)
    (hcs : c ^ 2 + s ^ 2 = 1) :
    (toM n (Spec.arrGivens n p q c s))ᵀ * toM n (Spec.arrGivens n p q c s) = 1 := by
  have hfpq : (⟨p, hp⟩ : Fin n) ≠ ⟨q, hq⟩ := fun h => hpq (Fin.ext_iff.mp h)
  ext i j
  rw [Matrix.mul_apply, Matrix.one_apply]
  simp only [Matrix.transpose_apply]
  by_cases hip : i = ⟨p, hp⟩
  · subst hip
    by_cases hjp : j = ⟨p, hp⟩
    · -- (p, p)
      subst hjp
      rw [Finset.sum_congr rfl (fun k _ => by rw [givens_col_fp p q hp hq hpq c s k]),
        sum_two_supp _ _ hfpq c (-s) c (-s), if_pos rfl]
      nlinarith [hcs]
    · by_cases hjq : j = ⟨q, hq⟩
      · -- (p, q)
        subst hjq
        rw [Finset.sum_congr rfl (fun k _ => by
            rw [givens_col_fp p q hp hq hpq c s k, givens_col_fq p q hp hq hpq c s k]),
          sum_two_supp _ _ hfpq c (-s) s c, if_neg hfpq]
        ring
      · -- (p, other)
        have hjp' : j.val ≠ p := fun h => hjp (Fin.ext h)
        have hjq' : j.val ≠ q := fun h => hjq (Fin.ext h)
        rw [Finset.sum_congr rfl (fun k _ => by
            rw [givens_col_fp p q hp hq hpq c s k, givens_col_other p q c s j k hjp' hjq']),
          sum_two_supp_mul_ite _ _ j hjp hjq c (-s), if_neg (Ne.symm hjp)]
  · by_cases hiq : i = ⟨q, hq⟩
    · subst hiq
      by_cases hjp : j = ⟨p, hp⟩
      · -- (q, p)
        subst hjp
        rw [Finset.sum_congr rfl (fun k _ => by
            rw [givens_col_fq p q hp hq hpq c s k, givens_col_fp p q hp hq hpq c s k]),
          sum_two_supp _ _ hfpq s c c (-s), if_neg (Ne.symm hfpq)]
        ring
      · by_cases hjq : j = ⟨q, hq⟩
        · -- (q, q)
          subst hjq
          rw [Finset.sum_congr rfl (fun k _ => by rw [givens_col_fq p q hp hq hpq c s k]),
            sum_two_supp _ _ hfpq s c s c, if_pos rfl]
          nlinarith [hcs]
        · -- (q, other)
          have hjp' : j.val ≠ p := fun h => hjp (Fin.ext h)
          have hjq' : j.val ≠ q := fun h => hjq (Fin.ext h)
          rw [Finset.sum_congr rfl (fun k _ => by
              rw [givens_col_fq p q hp hq hpq c s k, givens_col_other p q c s j k hjp' hjq']),
            sum_two_supp_mul_ite _ _ j hjp hjq s c, if_neg (Ne.symm hjq)]
    · -- i other
      have hip' : i.val ≠ p := fun h => hip (Fin.ext h)
      have hiq' : i.val ≠ q := fun h => hiq (Fin.ext h)
      by_cases hjp : j = ⟨p, hp⟩
      · -- (other, p)
        subst hjp
        rw [Finset.sum_congr rfl (fun k _ => by
            rw [givens_col_other p q c s i k hip' hiq', givens_col_fp p q hp hq hpq c s k,
              mul_comm]),
          sum_two_supp_mul_ite _ _ i hip hiq c (-s), if_neg hip]
      · by_cases hjq : j = ⟨q, hq⟩
        · -- (other, q)
          subst hjq
          rw [Finset.sum_congr rfl (fun k _ => by
              rw [givens_col_other p q c s i k hip' hiq', givens_col_fq p q hp hq hpq c s k,
                mul_comm]),
            sum_two_supp_mul_ite _ _ i hip hiq s c, if_neg hiq]
        · -- (other, other)
          have hjp' : j.val ≠ p := fun h => hjp (Fin.ext h)
          have hjq' : j.val ≠ q := fun h => hjq (Fin.ext h)
          rw [Finset.sum_congr rfl (fun k _ => by
              rw [givens_col_other p q c s i k hip' hiq', givens_col_other p q c s j k hjp' hjq']),
            sum_ite_mul_ite]

/-- The Golub–Van Loan rotation parameters the implementation uses satisfy `c² + s² = 1` for any
intermediate value `t`: this is `givens_normSq` with `MathFunctions.sqrt = Real.sqrt` and `t·t = t²`. -/
theorem code_givens_normSq (t : ℝ) :
    (1 / MathFunctions.sqrt (1 + t * t)) ^ 2 + (t * (1 / MathFunctions.sqrt (1 + t * t))) ^ 2 = 1 := by
  have h1 : MathFunctions.sqrt (1 + t * t) = Real.sqrt (1 + t ^ 2) := by
    rw [show (1 : ℝ) + t * t = 1 + t ^ 2 from by ring]; rfl
  rw [h1]
  exact givens_normSq t

/-! ## The loop invariant -/

/-- The Jacobi loop invariant relative to the input `A₀`: the running `V` is orthogonal and the
running pair `(A, V)` satisfies the orthogonal-similarity identity `A₀ = V · A · Vᵀ`. -/
def JacInv (A0 : Matrix (Fin n) (Fin n) ℝ) (st : Array (Array ℝ) × Array (Array ℝ)) : Prop :=
  (toM n st.2)ᵀ * toM n st.2 = 1 ∧ A0 = toM n st.2 * toM n st.1 * (toM n st.2)ᵀ

/-- One orthogonal-similarity update by an orthogonal `J` preserves the invariant. -/
theorem jacInv_step {A0 : Matrix (Fin n) (Fin n) ℝ} {A V J : Array (Array ℝ)}
    (hJ : (toM n J)ᵀ * toM n J = 1) (h : JacInv A0 (A, V)) :
    JacInv A0 (Spec.arrMatMul n (Spec.arrTr n J) (Spec.arrMatMul n A J), Spec.arrMatMul n V J) := by
  obtain ⟨hVo, hsim⟩ := h
  simp only [JacInv] at hVo hsim ⊢
  have hJJ : toM n J * (toM n J)ᵀ = 1 := mul_eq_one_comm.mp hJ
  refine ⟨?_, ?_⟩
  · rw [toM_matMul, Matrix.transpose_mul]
    calc (toM n J)ᵀ * (toM n V)ᵀ * (toM n V * toM n J)
        = (toM n J)ᵀ * ((toM n V)ᵀ * toM n V) * toM n J := by
          simp only [Matrix.mul_assoc]
      _ = (toM n J)ᵀ * toM n J := by rw [hVo, Matrix.mul_one]
      _ = 1 := hJ
  · simp only [toM_matMul, toM_tr, Matrix.transpose_mul]
    rw [hsim]
    have e1 : (toM n V * toM n J) * ((toM n J)ᵀ * (toM n A * toM n J)) * ((toM n J)ᵀ * (toM n V)ᵀ)
        = toM n V * (toM n J * (toM n J)ᵀ) * toM n A * (toM n J * (toM n J)ᵀ) * (toM n V)ᵀ := by
      simp only [Matrix.mul_assoc]
    rw [e1, hJJ]
    simp only [Matrix.mul_one, Matrix.mul_assoc]

/-- One Jacobi rotation preserves the invariant (the parameters always give an orthogonal `J`, and
the no-op branch is trivial). -/
theorem jacInv_rotate {A0 : Matrix (Fin n) (Fin n) ℝ} (p q : Nat) (hp : p < n) (hq : q < n)
    (hpq : p ≠ q) {st : Array (Array ℝ) × Array (Array ℝ)} (h : JacInv A0 st) :
    JacInv A0 (Spec.arrJacobiRotate n st.1 st.2 p q) := by
  unfold Spec.arrJacobiRotate
  extract_lets apq
  split
  · exact jacInv_step (givens_orthogonal p q hp hq hpq _ _ (code_givens_normSq _)) h
  · exact h

/-- Every pair produced by `jacobiPairs n` has `p < q < n`. -/
theorem jacobiPairs_spec {pq : Nat × Nat} (h : pq ∈ Spec.jacobiPairs n) :
    pq.1 < pq.2 ∧ pq.2 < n := by
  unfold Spec.jacobiPairs at h
  simp only [List.mem_flatMap, List.mem_filterMap, List.mem_range] at h
  obtain ⟨p, _, q, hq, hcond⟩ := h
  split at hcond
  · rename_i hlt
    simp only [Option.some.injEq] at hcond
    rw [← hcond]
    exact ⟨hlt, hq⟩
  · simp at hcond

/-- One Jacobi sweep preserves the invariant. -/
theorem jacInv_sweep {A0 : Matrix (Fin n) (Fin n) ℝ} {st : Array (Array ℝ) × Array (Array ℝ)}
    (h : JacInv A0 st) : JacInv A0 (Spec.arrJacobiSweep n st) := by
  unfold Spec.arrJacobiSweep
  refine List.foldlRecOn _ _ h ?_
  intro b hb pq hmem
  obtain ⟨hlt, hqn⟩ := jacobiPairs_spec hmem
  exact jacInv_rotate pq.1 pq.2 (Nat.lt_trans hlt hqn) hqn (Nat.ne_of_lt hlt) hb

/-- **The whole Jacobi run preserves the invariant.** Starting from `(A, I)`, after any number of
sweeps the accumulated `V` is orthogonal and `toM A = V · Af · Vᵀ`. -/
theorem jacInv_run (A : Array (Array ℝ)) (sweeps : Nat) :
    JacInv (toM n A) (Spec.arrJacobiRun n A sweeps) := by
  unfold Spec.arrJacobiRun
  refine List.foldlRecOn _ _ ?_ ?_
  · refine ⟨?_, ?_⟩
    · show (toM n (Spec.arrId n))ᵀ * toM n (Spec.arrId n) = 1
      rw [toM_id]; simp
    · show toM n A = toM n (Spec.arrId n) * toM n A * (toM n (Spec.arrId n))ᵀ
      rw [toM_id]; simp
  · intro b hb _ _
    exact jacInv_sweep hb

/-! ## Discharging the residual-certificate hypotheses for the real solver output -/

/-- View of the input tensor `A` as a `Matrix`. -/
noncomputable def inputMat (A : Spec.Tensor ℝ (.dim n (.dim n .scalar))) :
    Matrix (Fin n) (Fin n) ℝ :=
  Matrix.of (Spec.toMatFn A)

/-- The eigenvector matrix `V` produced by the Jacobi run on `A` (columns are the eigenvectors). -/
noncomputable def jacobiV (A : Spec.Tensor ℝ (.dim n (.dim n .scalar))) (sweeps : Nat) :
    Matrix (Fin n) (Fin n) ℝ :=
  toM n (Spec.arrJacobiRun n (Spec.matToArr (Spec.toMatFn A)) sweeps).2

/-- The rotated matrix `Af = Vᵀ A V` produced by the Jacobi run (diagonal in the zero-residual
limit; its diagonal holds the eigenvalues). -/
noncomputable def jacobiAf (A : Spec.Tensor ℝ (.dim n (.dim n .scalar))) (sweeps : Nat) :
    Matrix (Fin n) (Fin n) ℝ :=
  toM n (Spec.arrJacobiRun n (Spec.matToArr (Spec.toMatFn A)) sweeps).1

/-- `toM` of the materialised input function is the input matrix. -/
theorem toM_matToArr (X : Fin n → Fin n → ℝ) : toM n (Spec.matToArr X) = Matrix.of X := by
  ext i j
  simp only [toM_apply, Matrix.of_apply]
  unfold Spec.matToArr
  rw [arrGet_ofFn₂]

/-- **Discharged premise 1 — orthogonality.** The eigenvector matrix the Jacobi solver returns is
orthogonal, with no convergence hypothesis. -/
theorem jacobi_orthogonal (A : Spec.Tensor ℝ (.dim n (.dim n .scalar))) (sweeps : Nat) :
    (jacobiV A sweeps)ᵀ * jacobiV A sweeps = 1 :=
  (jacInv_run (Spec.matToArr (Spec.toMatFn A)) sweeps).1

/-- **Discharged premise 2 — orthogonal similarity.** The input equals `V · Af · Vᵀ` exactly, with no
convergence hypothesis. -/
theorem jacobi_similarity (A : Spec.Tensor ℝ (.dim n (.dim n .scalar))) (sweeps : Nat) :
    inputMat A = jacobiV A sweeps * jacobiAf A sweeps * (jacobiV A sweeps)ᵀ := by
  have h := (jacInv_run (n := n) (Spec.matToArr (Spec.toMatFn A)) sweeps).2
  rw [toM_matToArr] at h
  exact h

/-- **Unconditional residual identity.** Reconstructing with the diagonal of `Af` leaves exactly the
orthogonal conjugation of `Af`'s off-diagonal part — now stated about the real `symEigJacobiSpec`
output rather than under a hypothesis. -/
theorem symEigJacobi_reconstruction_residual (A : Spec.Tensor ℝ (.dim n (.dim n .scalar)))
    (sweeps : Nat) :
    inputMat A
        - jacobiV A sweeps * Matrix.diagonal (fun i => jacobiAf A sweeps i i) * (jacobiV A sweeps)ᵀ
      = jacobiV A sweeps * offDiagonal (jacobiAf A sweeps) * (jacobiV A sweeps)ᵀ :=
  symEig_reconstruction_residual (jacobi_similarity A sweeps)

/-- **Unconditional Frobenius residual certificate.** The squared reconstruction error equals the
squared off-diagonal mass of `Af` — unconditionally for the real solver output. -/
theorem symEigJacobi_frobenius_residual (A : Spec.Tensor ℝ (.dim n (.dim n .scalar)))
    (sweeps : Nat) :
    ((inputMat A
          - jacobiV A sweeps * Matrix.diagonal (fun i => jacobiAf A sweeps i i)
            * (jacobiV A sweeps)ᵀ)ᵀ
        * (inputMat A
          - jacobiV A sweeps * Matrix.diagonal (fun i => jacobiAf A sweeps i i)
            * (jacobiV A sweeps)ᵀ)).trace
      = ((offDiagonal (jacobiAf A sweeps))ᵀ * offDiagonal (jacobiAf A sweeps)).trace :=
  symEig_frobenius_residual (jacobi_orthogonal A sweeps) (jacobi_similarity A sweeps)

/-- **Unconditional correctness in the zero-residual limit.** When the rotated matrix is diagonal,
the solver output is an exact symmetric eigendecomposition of the input — no hypotheses beyond
diagonality. -/
theorem symEigJacobi_isSymEig_of_diagonal (A : Spec.Tensor ℝ (.dim n (.dim n .scalar)))
    (sweeps : Nat)
    (hdiag : jacobiAf A sweeps = Matrix.diagonal (fun i => jacobiAf A sweeps i i)) :
    IsSymEig (inputMat A) (fun i => jacobiAf A sweeps i i) (jacobiV A sweeps) :=
  isSymEig_of_diagonal (jacobi_orthogonal A sweeps) (jacobi_similarity A sweeps) hdiag

/-- The eigenvector matrix read back from the public `symEigJacobiSpec` output is `jacobiV`, so the
theorems above are statements about the actual returned `V`. -/
theorem symEigJacobiSpec_V_eq (A : Spec.Tensor ℝ (.dim n (.dim n .scalar))) (sweeps : Nat) :
    Matrix.of (fun i j => Spec.get2 (Spec.symEigJacobiSpec A sweeps).2 i j) = jacobiV A sweeps :=
  rfl

/-! ## Example: the residual certificate is now unconditional

`symEig_frobenius_residual` and `isSymEig_of_diagonal` used to *take* `Vᵀ V = 1` and
`A = V · Af · Vᵀ` as hypotheses. For the real `symEigJacobiSpec` output those are now theorems
(`jacobi_orthogonal`, `jacobi_similarity`), so the certificate follows from the input and sweep
count alone — no premises to discharge at the call site. -/

/-- The Frobenius residual identity for a `3×3` Jacobi run with `8` sweeps, with no hypotheses. -/
example (A : Spec.Tensor ℝ (.dim 3 (.dim 3 .scalar))) :
    ((inputMat A
          - jacobiV A 8 * Matrix.diagonal (fun i => jacobiAf A 8 i i) * (jacobiV A 8)ᵀ)ᵀ
        * (inputMat A
          - jacobiV A 8 * Matrix.diagonal (fun i => jacobiAf A 8 i i) * (jacobiV A 8)ᵀ)).trace
      = ((offDiagonal (jacobiAf A 8))ᵀ * offDiagonal (jacobiAf A 8)).trace :=
  symEigJacobi_frobenius_residual A 8

/-- In the zero-residual limit the output is a genuine eigendecomposition; the only hypothesis is
diagonality of the rotated matrix — orthogonality and the orthogonal similarity come for free. -/
example (A : Spec.Tensor ℝ (.dim 3 (.dim 3 .scalar)))
    (h : jacobiAf A 8 = Matrix.diagonal (fun i => jacobiAf A 8 i i)) :
    IsSymEig (inputMat A) (fun i => jacobiAf A 8 i i) (jacobiV A 8) :=
  symEigJacobi_isSymEig_of_diagonal A 8 h

end Spec.Factorization
