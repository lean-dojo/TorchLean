/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Tensor.Basic
public import NN.Spec.Layers.Attention

/-!
# Permutation Equivariance of Self-Attention (No Positional Encoding)

Self-attention (without positional information) should be **equivariant** to permutations of the
token axis:

If we reorder the input tokens, the output is reordered in the same way.

This file formalizes that statement for TorchLean’s spec-layer `Spec.selfAttention` over `ℝ`.

The helper reindexing operations below are intentionally proof-local. They describe how this proof
permutes tensor axes, but they are not part of the general `Spec.Tensor` API; reusable tensor
operations should live under `NN.Spec`, while model theorems and their proof scaffolding live here.
-/

open scoped BigOperators

noncomputable section

namespace NN.Proofs.Models.Attention

open Spec
open Spec.Tensor

abbrev Shape := Spec.Shape
abbrev Tensor := Spec.Tensor

/-!
## Token reindexing

Because spec tensors are functions out of `Fin n`, a token permutation is just reindexing the
outer axis.
-/

/-- Reindex the outermost axis of a tensor by a permutation. -/
def reindexOuter {α : Type} {n : Nat} {s : Shape} (σ : Equiv.Perm (Fin n)) :
    Tensor α (.dim n s) → Tensor α (.dim n s)
  | .dim f => .dim (fun i => f (σ i))

@[simp] theorem get_reindexOuter {α : Type} {n : Nat} {s : Shape}
    (σ : Equiv.Perm (Fin n)) (t : Tensor α (.dim n s)) (i : Fin n) :
    Spec.get (reindexOuter (α := α) (n := n) (s := s) σ t) i = Spec.get t (σ i) := by
  cases t with
  | dim _ => rfl

/-- Reindex the *column* axis of a matrix by a permutation. -/
def reindexCols {α : Type} {m n : Nat} (σ : Equiv.Perm (Fin n)) :
    Tensor α (.dim m (.dim n .scalar)) → Tensor α (.dim m (.dim n .scalar))
  | .dim rows =>
      .dim (fun i => reindexOuter (α := α) (n := n) (s := .scalar) σ (rows i))

@[simp] theorem get2_reindexOuter {α : Type} {m n : Nat}
    (σ : Equiv.Perm (Fin m)) (A : Tensor α (.dim m (.dim n .scalar))) (i : Fin m) (j : Fin n) :
    Spec.get2 (reindexOuter (α := α) (n := m) (s := .dim n .scalar) σ A) i j =
      Spec.get2 A (σ i) j := by
  cases A with
  | dim _ => rfl

@[simp] theorem get2_reindexCols {α : Type} {m n : Nat}
    (σ : Equiv.Perm (Fin n)) (A : Tensor α (.dim m (.dim n .scalar))) (i : Fin m) (j : Fin n) :
    Spec.get2 (reindexCols (α := α) (m := m) (n := n) σ A) i j =
      Spec.get2 A i (σ j) := by
  cases A with
  | dim rows =>
      cases hrow : rows i with
      | dim cols =>
          simp [reindexCols, reindexOuter, Spec.get2, Spec.get, Spec.getAtSpec, hrow]

/-- Simultaneously permute rows and columns of an `n×n` matrix by the same permutation. -/
def permMatrix {α : Type} {n : Nat} (σ : Equiv.Perm (Fin n)) (A : Tensor α (.dim n (.dim n .scalar))) :
    Tensor α (.dim n (.dim n .scalar)) :=
  reindexOuter (α := α) (n := n) (s := .dim n .scalar) σ (reindexCols (α := α) (m := n) (n := n) σ A)

@[simp] theorem get2_permMatrix {α : Type} {n : Nat}
    (σ : Equiv.Perm (Fin n)) (A : Tensor α (.dim n (.dim n .scalar))) (i j : Fin n) :
    Spec.get2 (permMatrix (α := α) (n := n) σ A) i j = Spec.get2 A (σ i) (σ j) := by
  simp [permMatrix]

/-!
## Softmax equivariance

TorchLean's spec softmax on vectors is implemented in a stabilized way (`x ↦ exp(x - m) / Σ exp(x - m)`),
but over `ℝ` it agrees with the plain `exp(x) / Σ exp(x)` formula. This lets us prove permutation
equivariance without reasoning about how the stabilizing shift `m` is chosen.
-/

namespace SoftmaxEquivariance

/-- Eliminate a scalar tensor using the same matcher as `Activation.softmax_vec_spec`. -/
private abbrev scalarElim {β : Sort _} (t : Tensor ℝ .scalar) (k : ℝ → β) : β :=
  Activation.softmaxVecSpec.match_1 (motive := fun _ => β) t k

@[simp] private theorem scalarElim_scalar {β : Sort _} (k : ℝ → β) (v : ℝ) :
    scalarElim (β := β) (Spec.Tensor.scalar v) k = k v := rfl

private abbrev scalarVal (t : Tensor ℝ .scalar) : ℝ :=
  scalarElim (β := ℝ) t (fun v => v)

/-- Plain (unstabilized) softmax on a vector tensor. Proof helper. -/
private def softmaxVecPlain {n : Nat} (t : Tensor ℝ (.dim n .scalar)) : Tensor ℝ (.dim n .scalar) :=
  match t with
  | .dim f =>
      let x : Fin n → ℝ := fun i => scalarVal (f i)
      let denom : ℝ := ∑ j : Fin n, Real.exp (x j)
      .dim (fun i => .scalar (Real.exp (x i) / denom))

/-- The stabilized spec `softmax_vec_spec` agrees with `softmaxVecPlain` over `ℝ`. -/
private theorem softmax_vec_spec_eq_plain {n : Nat} (t : Tensor ℝ (.dim (Nat.succ n) .scalar)) :
    Activation.softmaxVecSpec (α := ℝ) (n := Nat.succ n) t = softmaxVecPlain t := by
  classical
  cases t with
  | dim f =>
      let x : Fin (Nat.succ n) → ℝ := fun i => scalarVal (f i)
      -- The internal shift used by the stabilized definition (we do not use any "max" properties).
      let first : ℝ := x ⟨0, Nat.succ_pos n⟩
      -- Define `m` in the same shape as the spec definition (folding over indices with a `match`).
      let m : ℝ :=
        (List.finRange (Nat.succ n)).foldl
          (fun acc i => scalarElim (β := ℝ) (f i) (fun v => max acc v))
          first
      -- Denominators: shifted vs plain.
      let denomPlain : ℝ := ∑ j : Fin (Nat.succ n), Real.exp (x j)
      let denomShift : ℝ := ∑ j : Fin (Nat.succ n), Real.exp (x j - m)
      have hdenomShift :
          denomShift = denomPlain * Real.exp (-m) := by
        -- Rewrite each term `exp(x - m) = exp(x) * exp(-m)` and factor out the constant.
        calc
          denomShift
              = ∑ j : Fin (Nat.succ n), Real.exp (x j) * Real.exp (-m) := by
                  refine Finset.sum_congr rfl ?_
                  intro j _
                  simp [sub_eq_add_neg, Real.exp_add]
          _ = (∑ j : Fin (Nat.succ n), Real.exp (x j)) * Real.exp (-m) := by
                -- `∑ (a_j * c) = (∑ a_j) * c`.
                simpa [denomPlain] using (Finset.sum_mul (s := (Finset.univ : Finset (Fin (Nat.succ n))))
                  (f := fun j => Real.exp (x j)) (a := Real.exp (-m))).symm
      -- Now show coordinatewise equality: the `exp(-m)` factor cancels.
      apply congrArg Spec.Tensor.dim
      funext i
      have hmne : Real.exp (-m) ≠ 0 := Real.exp_ne_zero _
      -- The stabilized output is `exp(x_i - m) / denomShift`.
      -- The plain output is `exp(x_i) / denomPlain`.
      -- Use `mul_div_mul_right` to cancel the shared factor `exp(-m)`.
      have hcancel :
          Real.exp (x i - m) / denomShift = Real.exp (x i) / denomPlain := by
        -- Rewrite numerator and denominator into `(* exp(-m))` form, then cancel.
        calc
          Real.exp (x i - m) / denomShift
              = (Real.exp (x i) * Real.exp (-m)) / (denomPlain * Real.exp (-m)) := by
                  simp [hdenomShift, sub_eq_add_neg, Real.exp_add]
          _ = Real.exp (x i) / denomPlain := by
                simpa [mul_assoc] using (mul_div_mul_right (Real.exp (x i)) denomPlain hmne)
      -- Relate the implementation's list-fold denominator to the `∑` form (`denomShift`).
      have hfold :
          (List.finRange (Nat.succ n)).foldl (fun acc j => acc + Real.exp (x j - m)) 0 = denomShift := by
        dsimp [denomShift]
        simpa using
          (Spec.finRange_foldl_add_eq_finset_sum (n := Nat.succ n) (f := fun j => Real.exp (x j - m)))
      -- Finish by simplifying the scalar combinators and rewriting the denominator fold.
      have hTerm :
          ∀ j : Fin (Nat.succ n),
            scalarElim (β := ℝ)
                (Spec.Tensor.mapSpec MathFunctions.exp
                  (Spec.Tensor.map2Spec (fun a b => a - b) (f j) (Spec.Tensor.scalar m)))
                (fun v => v) =
              Real.exp (x j - m) := by
        intro j
        cases hj : f j with
        | scalar xj =>
            simp [scalarElim, scalarVal, Spec.Tensor.mapSpec, Spec.Tensor.map2Spec, x, hj]
            rfl
      have hfun :
          (fun (acc : ℝ) (j : Fin (Nat.succ n)) =>
              acc +
                scalarElim (β := ℝ)
                  (Spec.Tensor.mapSpec MathFunctions.exp
                    (Spec.Tensor.map2Spec (fun a b => a - b) (f j) (Spec.Tensor.scalar m)))
                  (fun v => v))
            =
          (fun (acc : ℝ) (j : Fin (Nat.succ n)) => acc + Real.exp (x j - m)) := by
        funext acc j
        simp [hTerm j]
      have hden :
          (List.finRange (Nat.succ n)).foldl
              (fun acc j =>
                acc +
                  scalarElim (β := ℝ)
                    (Spec.Tensor.mapSpec MathFunctions.exp
                      (Spec.Tensor.map2Spec (fun a b => a - b) (f j) (Spec.Tensor.scalar m)))
                    (fun v => v))
              0
            =
          denomShift := by
        simpa [hfun] using hfold

      -- Reduce to the real-valued cancellation lemma `hcancel` and lift through `Tensor.scalar`.
      cases hi : f i with
      | scalar xi =>
          have hcancel' : Real.exp (xi - m) / denomShift = Real.exp xi / denomPlain := by
            simpa [x, scalarVal, scalarElim, hi] using hcancel
          have hgoal :
              MathFunctions.exp
                    (xi -
                      List.foldl (fun acc i => scalarElim (β := ℝ) (f i) (fun v => max acc v))
                        (scalarElim (β := ℝ) (f ⟨0, Nat.succ_pos n⟩) (fun v => v))
                        (List.finRange (Nat.succ n))) /
                  List.foldl
                      (fun acc i =>
                        acc +
                          scalarElim (β := ℝ)
                            (Spec.Tensor.mapSpec MathFunctions.exp
                              (Spec.Tensor.map2Spec (fun x1 x2 => x1 - x2) (f i)
                                (Spec.Tensor.scalar
                                  (List.foldl
                                    (fun acc i => scalarElim (β := ℝ) (f i) (fun v => max acc v))
                                    (scalarElim (β := ℝ) (f ⟨0, Nat.succ_pos n⟩) (fun v => v))
                                    (List.finRange (Nat.succ n))))))
                            (fun v => v))
                      (0 : ℝ)
                      (List.finRange (Nat.succ n))
                =
              Real.exp xi /
                ∑ x : Fin (Nat.succ n),
                  Real.exp (scalarElim (β := ℝ) (f x) (fun v => v)) := by
            have htmp := hcancel'
            -- Replace the finite-sum denominator (`denomShift`) with the list-fold denominator used
            -- by the spec definition.
            rw [← hden] at htmp
            simpa [denomPlain, x, m, first, scalarVal, scalarElim] using htmp
          -- Normalized form of `hgoal` matching the shape produced by the `simp`-unfolding below.
          have hgoal' :
              MathFunctions.exp
                    (xi -
                      List.foldl (fun acc i => scalarElim (β := ℝ) (f i) (fun v => max acc v))
                        (scalarElim (β := ℝ) (f 0) (fun v => v))
                        (List.finRange (n + 1))) /
                  List.foldl
                      (fun acc i =>
                        acc +
                          scalarElim (β := ℝ)
                            (Spec.Tensor.mapSpec MathFunctions.exp
                              (Spec.Tensor.map2Spec (fun x1 x2 => x1 - x2) (f i)
                                (Spec.Tensor.scalar
                                  (List.foldl
                                    (fun acc i => scalarElim (β := ℝ) (f i) (fun v => max acc v))
                                    (scalarElim (β := ℝ) (f 0) (fun v => v))
                                    (List.finRange (n + 1))))))
                            (fun v => v))
                      (0 : ℝ)
                      (List.finRange (n + 1))
                =
              Real.exp xi /
                ∑ x : Fin (n + 1),
                  Real.exp (scalarElim (β := ℝ) (f x) (fun v => v)) := by
            simpa [Nat.succ_eq_add_one] using hgoal
          -- Unfold the scalar tensor ops at index `i` and use `hgoal`.
          simp [hi, scalarVal, scalarElim, Spec.replicate, Spec.Tensor.mapSpec, Spec.Tensor.map2Spec,
            Spec.Tensor.subSpec, Spec.Tensor.expSpec]
          -- `simp` has unfolded the tensor combinators, so the goal is a scalar identity.
          simpa [scalarVal, scalarElim] using hgoal'

/-- Plain softmax commutes with reindexing (permuting coordinates). -/
private theorem softmaxVecPlain_reindexOuter {n : Nat} (σ : Equiv.Perm (Fin n))
    (t : Tensor ℝ (.dim n .scalar)) :
    softmaxVecPlain (reindexOuter (α := ℝ) (n := n) (s := .scalar) σ t)
      =
    reindexOuter (α := ℝ) (n := n) (s := .scalar) σ (softmaxVecPlain t) := by
  classical
  cases t with
  | dim f =>
      apply congrArg Spec.Tensor.dim
      funext i
      let x : Fin n → ℝ := fun j => scalarVal (f j)
      have hden :
          (∑ j : Fin n, Real.exp (x (σ j))) = ∑ j : Fin n, Real.exp (x j) := by
        simpa using (Equiv.sum_comp σ (fun j => Real.exp (x j)))
      simp [x, hden, scalarVal, scalarElim]

/-- Spec vector softmax commutes with reindexing (permuting coordinates). -/
theorem softmax_vec_spec_reindexOuter {n : Nat} (σ : Equiv.Perm (Fin (Nat.succ n)))
    (t : Tensor ℝ (.dim (Nat.succ n) .scalar)) :
    Activation.softmaxVecSpec (α := ℝ) (n := Nat.succ n)
        (reindexOuter (α := ℝ) (n := Nat.succ n) (s := .scalar) σ t)
      =
    reindexOuter (α := ℝ) (n := Nat.succ n) (s := .scalar) σ
      (Activation.softmaxVecSpec (α := ℝ) (n := Nat.succ n) t) := by
  -- Reduce to the plain form.
  simpa [softmax_vec_spec_eq_plain] using
    (softmaxVecPlain_reindexOuter (σ := σ) (t := t))

/-- Matrix last-axis softmax commutes with simultaneous row/column permutations (`permMatrix`). -/
theorem softmax_spec_permMatrix {n : Nat} (σ : Equiv.Perm (Fin (Nat.succ n)))
    (A : Spec.Tensor ℝ (.dim (Nat.succ n) (.dim (Nat.succ n) .scalar))) :
    Activation.softmaxSpec (α := ℝ) (s := .dim (Nat.succ n) (.dim (Nat.succ n) .scalar))
        (permMatrix (α := ℝ) (n := Nat.succ n) σ A)
      =
    permMatrix (α := ℝ) (n := Nat.succ n) σ
      (Activation.softmaxSpec (α := ℝ) (s := .dim (Nat.succ n) (.dim (Nat.succ n) .scalar)) A) := by
  cases A with
  | dim rows =>
      apply congrArg Spec.Tensor.dim
      funext i
      -- `softmax_spec` on matrices is rowwise, and `reindexCols` acts within each row.
      simp [Activation.softmaxSpec, softmax_vec_spec_reindexOuter]

end SoftmaxEquivariance

/-!
## Linear algebra: matmul/transpose commute with permutations
-/

/--
Matrix multiplication commutes with independent output-row and output-column reindexing.

This is the most general bookkeeping lemma used below: reindexing rows of the left factor controls
the rows of the product, while reindexing columns of the right factor controls the columns of the
product.
-/
theorem mat_mul_reindexOuter_reindexCols {m n p : Nat}
    (σ : Equiv.Perm (Fin m)) (τ : Equiv.Perm (Fin p))
    (A : Tensor ℝ (.dim m (.dim n .scalar)))
    (B : Tensor ℝ (.dim n (.dim p .scalar))) :
    matMulSpec (reindexOuter (α := ℝ) (n := m) (s := .dim n .scalar) σ A)
        (reindexCols (α := ℝ) (m := n) (n := p) τ B)
      =
    reindexOuter (α := ℝ) (n := m) (s := .dim p .scalar) σ
      (reindexCols (α := ℝ) (m := m) (n := p) τ (matMulSpec A B)) := by
  classical
  apply Spec.matrix_ext
  intro i j
  simp [Spec.get2_mat_mul_spec]

/--
Left-multiplying by a row-permuted matrix only row-permutes the matrix-product output.

This is the projection-layer version of token equivariance: the same learned weight matrix is
applied independently to every token, so changing token order before the projection merely changes
the output token order.
-/
theorem mat_mul_reindexOuter_left {m n p : Nat}
    (σ : Equiv.Perm (Fin m))
    (A : Tensor ℝ (.dim m (.dim n .scalar)))
    (B : Tensor ℝ (.dim n (.dim p .scalar))) :
    matMulSpec (reindexOuter (α := ℝ) (n := m) (s := .dim n .scalar) σ A) B
      =
    reindexOuter (α := ℝ) (n := m) (s := .dim p .scalar) σ (matMulSpec A B) := by
  classical
  apply Spec.matrix_ext
  intro i j
  simp [Spec.get2_mat_mul_spec]

/--
Transposition converts a row permutation into a column permutation.

The attention-score proof uses this to turn `(P Q) (P K)ᵀ` into a simultaneous row/column
permutation of `Q Kᵀ`.
-/
theorem matrix_transpose_reindexOuter {m n : Nat}
    (σ : Equiv.Perm (Fin m)) (A : Tensor ℝ (.dim m (.dim n .scalar))) :
    Spec.Tensor.matrixTransposeSpec
        (reindexOuter (α := ℝ) (n := m) (s := .dim n .scalar) σ A)
      =
    reindexCols (α := ℝ) (m := n) (n := m) σ (Spec.Tensor.matrixTransposeSpec A) := by
  classical
  apply Spec.matrix_ext
  intro i j
  simp [Spec.get2_matrix_transpose_spec]

/-!
## Main theorem: self-attention equivariance
-/

/--
Elementwise scaling commutes with simultaneous row/column reindexing.

Scaled dot-product attention divides all score entries by the same scalar, so this lemma lets the
token-permutation proof move the scale step past the score-matrix conjugation.
-/
theorem scale_spec_permMatrix {n : Nat} (σ : Equiv.Perm (Fin n))
    (A : Tensor ℝ (.dim n (.dim n .scalar))) (c : ℝ) :
    Spec.Tensor.scaleSpec (permMatrix (α := ℝ) (n := n) σ A) c
      =
    permMatrix (α := ℝ) (n := n) σ (Spec.Tensor.scaleSpec A c) := by
  classical
  -- Helper: extract a `scale_spec` entry.
  have get2_scale_spec {m n : Nat}
      (M : Tensor ℝ (.dim m (.dim n .scalar))) (c : ℝ) (i : Fin m) (j : Fin n) :
      get2 (Spec.Tensor.scaleSpec M c) i j = (get2 M i j) * c := by
    cases M with
    | dim rows =>
        cases hrow : rows i with
        | dim cols =>
            cases hcol : cols j with
            | scalar v =>
                simp [Spec.Tensor.scaleSpec, Spec.Tensor.mapSpec, get2, Spec.get, Spec.getAtSpec, hrow, hcol]
  apply Spec.matrix_ext
  intro i j
  -- Both sides reduce to `(A[σ i, σ j]) * c`.
  simp [get2_permMatrix, get2_scale_spec]

/--
Multiplying a simultaneously row/column-permuted attention matrix by a row-permuted value matrix
produces a row-permuted output.

This is the final linear-algebra step in self-attention equivariance: the permutation of the
attention weights and the permutation of the value rows cancel on the internal summation index.
-/
theorem mat_mul_permMatrix_reindexOuter
    {n d : Nat} (σ : Equiv.Perm (Fin n))
    (A : Tensor ℝ (.dim n (.dim n .scalar)))
    (B : Tensor ℝ (.dim n (.dim d .scalar))) :
    matMulSpec (permMatrix (α := ℝ) (n := n) σ A)
        (reindexOuter (α := ℝ) (n := n) (s := .dim d .scalar) σ B)
      =
    reindexOuter (α := ℝ) (n := n) (s := .dim d .scalar) σ
      (matMulSpec A B) := by
  classical
  apply Spec.matrix_ext
  intro i j
  -- Entrywise: reindexing the shared summation index is a change of variables by `σ`.
  simp [Spec.get2_mat_mul_spec]
  simpa using
    (Equiv.sum_comp σ (fun k : Fin n => (get2 A (σ i) k) * (get2 B k j)))

/--
Self-attention without positional encodings is equivariant to any token permutation.

If the input token axis is reordered by `σ`, then `Q`, `K`, and `V` are reordered in the same way,
the score matrix is conjugated by the corresponding permutation matrix, row-wise softmax commutes
with that conjugation, and the final output is exactly the same token reordering of the original
output. This theorem is intentionally stated for the spec-layer block, not for CUDA kernels.
-/
theorem selfAttention_reindexOuter
    {n dModel projDim : Nat}
    (σ : Equiv.Perm (Fin n))
    (x : Tensor ℝ (.dim n (.dim dModel .scalar)))
    (Wq : Tensor ℝ (.dim dModel (.dim projDim .scalar)))
    (Wk : Tensor ℝ (.dim dModel (.dim projDim .scalar)))
    (Wv : Tensor ℝ (.dim dModel (.dim projDim .scalar)))
    (Wo : Tensor ℝ (.dim projDim (.dim dModel .scalar)))
    (h1 : n ≠ 0) :
    Spec.selfAttention (α := ℝ) (n := n) (dModel := dModel) (projDim := projDim)
        (x := reindexOuter (α := ℝ) (n := n) (s := .dim dModel .scalar) σ x)
        (Wq := Wq) (Wk := Wk) (Wv := Wv) (Wo := Wo) h1
      =
    reindexOuter (α := ℝ) (n := n) (s := .dim dModel .scalar) σ
      (Spec.selfAttention (α := ℝ) (n := n) (dModel := dModel) (projDim := projDim)
        (x := x) (Wq := Wq) (Wk := Wk) (Wv := Wv) (Wo := Wo) h1) := by
  classical
  -- Reduce to `n = succ _` using `h1`.
  cases n with
  | zero =>
      cases (h1 rfl)
  | succ n' =>
      -- Abbreviations for the unpermuted intermediates.
      let Q : Tensor ℝ (.dim (Nat.succ n') (.dim projDim .scalar)) := matMulSpec x Wq
      let K : Tensor ℝ (.dim (Nat.succ n') (.dim projDim .scalar)) := matMulSpec x Wk
      let V : Tensor ℝ (.dim (Nat.succ n') (.dim projDim .scalar)) := matMulSpec x Wv

      -- Input projections commute with token permutation.
      have hQ :
          matMulSpec (reindexOuter (α := ℝ) (n := Nat.succ n') (s := .dim dModel .scalar) σ x) Wq
            =
          reindexOuter (α := ℝ) (n := Nat.succ n') (s := .dim projDim .scalar) σ Q := by
        simpa [Q] using
          (mat_mul_reindexOuter_left (σ := σ) (A := x) (B := Wq))
      have hK :
          matMulSpec (reindexOuter (α := ℝ) (n := Nat.succ n') (s := .dim dModel .scalar) σ x) Wk
            =
          reindexOuter (α := ℝ) (n := Nat.succ n') (s := .dim projDim .scalar) σ K := by
        simpa [K] using
          (mat_mul_reindexOuter_left (σ := σ) (A := x) (B := Wk))
      have hV :
          matMulSpec (reindexOuter (α := ℝ) (n := Nat.succ n') (s := .dim dModel .scalar) σ x) Wv
            =
          reindexOuter (α := ℝ) (n := Nat.succ n') (s := .dim projDim .scalar) σ V := by
        simpa [V] using
          (mat_mul_reindexOuter_left (σ := σ) (A := x) (B := Wv))

      -- Build the (unpermuted) attention context for the scaled dot-product block.
      let ctx : Spec.AttentionContext ℝ (Nat.succ n') (Nat.succ n') projDim h1 h1 :=
        { Q := Q, K := K, V := V, bc_sum_to_target := inferInstance, mask := none }
      let ctxσ : Spec.AttentionContext ℝ (Nat.succ n') (Nat.succ n') projDim h1 h1 :=
        { Q := reindexOuter (α := ℝ) (n := Nat.succ n') (s := .dim projDim .scalar) σ Q
          K := reindexOuter (α := ℝ) (n := Nat.succ n') (s := .dim projDim .scalar) σ K
          V := reindexOuter (α := ℝ) (n := Nat.succ n') (s := .dim projDim .scalar) σ V
          bc_sum_to_target := inferInstance
          mask := none }

      -- Equivariance of the scaled dot-product attention block (mask = none).
      have hSDA :
          Spec.scaledDotProductAttention (α := ℝ) (ctx := ctxσ) =
            reindexOuter (α := ℝ) (n := Nat.succ n') (s := .dim projDim .scalar) σ
              (Spec.scaledDotProductAttention (α := ℝ) (ctx := ctx)) := by
        -- Unfold the definition (mask = none) and rewrite each intermediate.
        -- scores: `Q Kᵀ` is conjugated by `permMatrix σ`.
        have hScores :
            matMulSpec
                (reindexOuter (α := ℝ) (n := Nat.succ n') (s := .dim projDim .scalar) σ Q)
                (Spec.Tensor.matrixTransposeSpec
                  (reindexOuter (α := ℝ) (n := Nat.succ n') (s := .dim projDim .scalar) σ K))
              =
            permMatrix (α := ℝ) (n := Nat.succ n') σ
              (matMulSpec Q (Spec.Tensor.matrixTransposeSpec K)) := by
          -- Push `σ` through transpose (rows → cols), then apply the matmul permutation lemma.
          calc
            _ =
              matMulSpec
                (reindexOuter (α := ℝ) (n := Nat.succ n') (s := .dim projDim .scalar) σ Q)
                (reindexCols (α := ℝ) (m := projDim) (n := Nat.succ n') σ
                  (Spec.Tensor.matrixTransposeSpec K)) := by
                    simp [matrix_transpose_reindexOuter]
            _ =
              reindexOuter (α := ℝ) (n := Nat.succ n') (s := .dim (Nat.succ n') .scalar) σ
                (reindexCols (α := ℝ) (m := Nat.succ n') (n := Nat.succ n') σ
                  (matMulSpec Q (Spec.Tensor.matrixTransposeSpec K))) := by
                    simpa using
                      (mat_mul_reindexOuter_reindexCols (σ := σ) (τ := σ) (A := Q)
                        (B := Spec.Tensor.matrixTransposeSpec K))
            _ = _ := rfl

        -- scale commutes with `permMatrix`.
        have hScaledScores :
            Spec.Tensor.scaleSpec
                (permMatrix (α := ℝ) (n := Nat.succ n') σ
                  (matMulSpec Q (Spec.Tensor.matrixTransposeSpec K)))
                (MathFunctions.sqrt (Coe.coe projDim : ℝ))⁻¹
              =
            permMatrix (α := ℝ) (n := Nat.succ n') σ
              (Spec.Tensor.scaleSpec (matMulSpec Q (Spec.Tensor.matrixTransposeSpec K))
                (MathFunctions.sqrt (Coe.coe projDim : ℝ))⁻¹) := by
          simpa using
            (scale_spec_permMatrix (σ := σ)
              (A := matMulSpec Q (Spec.Tensor.matrixTransposeSpec K))
              (c := (MathFunctions.sqrt (Coe.coe projDim : ℝ))⁻¹))

        -- softmax commutes with `permMatrix`.
        have hWeights :
            Activation.softmaxSpec (α := ℝ) (s := .dim (Nat.succ n') (.dim (Nat.succ n') .scalar))
                (permMatrix (α := ℝ) (n := Nat.succ n') σ
                  (Spec.Tensor.scaleSpec (matMulSpec Q (Spec.Tensor.matrixTransposeSpec K))
                    (MathFunctions.sqrt (Coe.coe projDim : ℝ))⁻¹))
              =
            permMatrix (α := ℝ) (n := Nat.succ n') σ
              (Activation.softmaxSpec (α := ℝ) (s := .dim (Nat.succ n') (.dim (Nat.succ n') .scalar))
                (Spec.Tensor.scaleSpec (matMulSpec Q (Spec.Tensor.matrixTransposeSpec K))
                  (MathFunctions.sqrt (Coe.coe projDim : ℝ))⁻¹)) := by
          simpa using
            (SoftmaxEquivariance.softmax_spec_permMatrix (n := n') (σ := σ)
              (A := Spec.Tensor.scaleSpec (matMulSpec Q (Spec.Tensor.matrixTransposeSpec K))
                (MathFunctions.sqrt (Coe.coe projDim : ℝ))⁻¹))

        -- final matmul with `V` turns the conjugation into an outer reindexing.
        have hOut :
            matMulSpec
                (permMatrix (α := ℝ) (n := Nat.succ n') σ
                  (Activation.softmaxSpec (α := ℝ) (s := .dim (Nat.succ n') (.dim (Nat.succ n') .scalar))
                    (Spec.Tensor.scaleSpec (matMulSpec Q (Spec.Tensor.matrixTransposeSpec K))
                      (MathFunctions.sqrt (Coe.coe projDim : ℝ))⁻¹)))
                (reindexOuter (α := ℝ) (n := Nat.succ n') (s := .dim projDim .scalar) σ V)
              =
            reindexOuter (α := ℝ) (n := Nat.succ n') (s := .dim projDim .scalar) σ
              (matMulSpec
                (Activation.softmaxSpec (α := ℝ) (s := .dim (Nat.succ n') (.dim (Nat.succ n') .scalar))
                  (Spec.Tensor.scaleSpec (matMulSpec Q (Spec.Tensor.matrixTransposeSpec K))
                    (MathFunctions.sqrt (Coe.coe projDim : ℝ))⁻¹))
                V) := by
          simpa using
            (mat_mul_permMatrix_reindexOuter (σ := σ)
              (A := Activation.softmaxSpec (α := ℝ) (s := .dim (Nat.succ n') (.dim (Nat.succ n') .scalar))
                (Spec.Tensor.scaleSpec (matMulSpec Q (Spec.Tensor.matrixTransposeSpec K))
                  (MathFunctions.sqrt (Coe.coe projDim : ℝ))⁻¹))
              (B := V))

        -- Put it all together by unfolding `scaledDotProductAttention` (mask = none).
        -- We avoid simp-loops by rewriting the intermediates explicitly.
        simp [Spec.scaledDotProductAttention, ctx, ctxσ]
        rw [hScores]
        rw [hScaledScores]
        rw [hWeights]
        simpa using hOut

      -- Now unfold `selfAttention` and finish via the matmul reindexing lemma.
      calc
        Spec.selfAttention (α := ℝ) (n := Nat.succ n') (dModel := dModel) (projDim := projDim)
            (x := reindexOuter (α := ℝ) (n := Nat.succ n') (s := .dim dModel .scalar) σ x)
            (Wq := Wq) (Wk := Wk) (Wv := Wv) (Wo := Wo) h1
            =
          matMulSpec (Spec.scaledDotProductAttention (α := ℝ) (ctx := ctxσ)) Wo := by
            simp [Spec.selfAttention, Q, K, V, hQ, hK, hV, ctxσ]
        _ =
          matMulSpec
            (reindexOuter (α := ℝ) (n := Nat.succ n') (s := .dim projDim .scalar) σ
              (Spec.scaledDotProductAttention (α := ℝ) (ctx := ctx)))
            Wo := by
              simp [hSDA]
        _ =
          reindexOuter (α := ℝ) (n := Nat.succ n') (s := .dim dModel .scalar) σ
            (matMulSpec (Spec.scaledDotProductAttention (α := ℝ) (ctx := ctx)) Wo) := by
              -- push `σ` through the final projection `Wo`
              exact
                (mat_mul_reindexOuter_left (σ := σ)
                  (A := Spec.scaledDotProductAttention (α := ℝ) (ctx := ctx)) (B := Wo))
        _ =
          reindexOuter (α := ℝ) (n := Nat.succ n') (s := .dim dModel .scalar) σ
            (Spec.selfAttention (α := ℝ) (n := Nat.succ n') (dModel := dModel) (projDim := projDim)
              (x := x) (Wq := Wq) (Wk := Wk) (Wv := Wv) (Wo := Wo) h1) := by
              simp [Spec.selfAttention, Q, K, V, ctx]

end NN.Proofs.Models.Attention
