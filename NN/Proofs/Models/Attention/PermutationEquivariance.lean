/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Analysis.Softmax
public import NN.Spec.Layers.Attention
public import Mathlib.Algebra.Order.Algebra
public import Mathlib.Analysis.SpecialFunctions.Pow.NNReal
public import Mathlib.Data.Sym.Sym2.Init
import Mathlib.Tactic.NormNum.GCD

/-!
# Permutation Equivariance of Self-Attention (No Positional Encoding)

Self-attention (without positional information) should be **equivariant** to permutations of the
token axis:

If we reorder the input tokens, the output is reordered in the same way.

This file formalizes that statement for TorchLean’s spec-layer `Spec.selfAttention` over `ℝ`.

The helper reindexing operations below are intentionally proof-local. They describe how this proof
permutes tensor axes, but they are not part of the general `TorchLean.Tensor` API; reusable tensor
operations should live under `NN.Spec`, while model theorems and their proof scaffolding live here.
-/

open scoped BigOperators

noncomputable section

namespace NN.Proofs.Models.Attention

open _root_.Spec _root_.TorchLean
open TorchLean.Tensor

/-!
## Token reindexing

Because spec tensors are functions out of `Fin n`, a token permutation is just reindexing the
outer axis.
-/

/-- Reindex the outermost axis of a tensor by a permutation. -/
def reindexOuter {α : Type} [TorchLean.Storage α] {n : Nat} {s : Shape} (σ : Equiv.Perm (Fin n)) :
    Tensor α (.dim n s) → Tensor α (.dim n s) :=
  fun tensor => Tensor.dim (fun i => tensor.unstack (σ i))

/-- Reindexing composes with lookup: slice `i` of the permuted tensor is slice `σ i` of the
original.

Spec tensors are functions out of `Fin n`, so a token permutation costs nothing to define and this
lemma is a `rfl`-level fact. That is precisely why the equivariance proofs below stay short. -/
@[simp] theorem get_reindexOuter {α : Type} [TorchLean.Storage α] {n : Nat} {s : Shape}
    (σ : Equiv.Perm (Fin n)) (t : Tensor α (.dim n s)) (i : Fin n) :
    Spec.get (reindexOuter (α := α) (n := n) (s := s) σ t) i = Spec.get t (σ i) := by
  simp [reindexOuter, Spec.get]

/-- Reindex the *column* axis of a matrix by a permutation. -/
def reindexCols {α : Type} [TorchLean.Storage α] {m n : Nat} (σ : Equiv.Perm (Fin n)) :
    Tensor α [m, n] → Tensor α [m, n] :=
  fun matrix =>
    Tensor.dim (fun i =>
      reindexOuter (α := α) (n := n) (s := .scalar) σ (matrix.unstack i))

/-- Matrix form: permuting the outer axis permutes rows. -/
@[simp] theorem get2_reindexOuter {α : Type} [TorchLean.Storage α] {m n : Nat}
    (σ : Equiv.Perm (Fin m)) (A : Tensor α [m, n]) (i : Fin m) (j : Fin n) :
    Spec.get2 (reindexOuter (α := α) (n := m) (s := .dim n .scalar) σ A) i j =
      Spec.get2 A (σ i) j := by
  simp [Spec.get2]

/-- And permuting the inner axis permutes columns. -/
@[simp] theorem get2_reindexCols {α : Type} [TorchLean.Storage α] {m n : Nat}
    (σ : Equiv.Perm (Fin n)) (A : Tensor α [m, n]) (i : Fin m) (j : Fin n) :
    Spec.get2 (reindexCols (α := α) (m := m) (n := n) σ A) i j =
      Spec.get2 A i (σ j) := by
  simp [reindexCols, reindexOuter, Spec.get2, Spec.get, TorchLean.Tensor.getScalar]

/-- Simultaneously permute rows and columns of an `n×n` matrix by the same permutation. -/
def permMatrix {α : Type} [TorchLean.Storage α] {n : Nat} (σ : Equiv.Perm (Fin n))
    (A : Tensor α [n, n]) : Tensor α [n, n] :=
  reindexOuter (α := α) (n := n) (s := .dim n .scalar) σ
    (reindexCols (α := α) (m := n) (n := n) σ A)

/-- Entries of a simultaneously permuted matrix: `(σ i, σ j)` of the original.

This is the form attention scores take under a token permutation, since both the query index and the
key index move with the same `σ`. -/
@[simp] theorem get2_permMatrix {α : Type} [TorchLean.Storage α] {n : Nat}
    (σ : Equiv.Perm (Fin n)) (A : Tensor α [n, n]) (i j : Fin n) :
    Spec.get2 (permMatrix (α := α) (n := n) σ A) i j = Spec.get2 A (σ i) (σ j) := by
  simp [permMatrix]

/-!
## Softmax equivariance

TorchLean's spec softmax on vectors is implemented in a stabilized way
(`x ↦ exp(x - m) / Σ exp(x - m)`), but over `ℝ` it agrees with the plain `exp(x) / Σ exp(x)`
formula (`Proofs.getScalar_softmaxVecSpec_eq_exp_div`). This lets us prove permutation equivariance
without reasoning about how the stabilizing shift `m` is chosen.
-/

namespace SoftmaxEquivariance

/-- Plain (unstabilized) softmax on a vector tensor. Proof helper. -/
private def softmaxVecPlain {n : Nat} (t : Tensor ℝ [n]) : Tensor ℝ [n] :=
  let x : Fin n → ℝ := fun i => t.getScalar i
  let denom : ℝ := ∑ j : Fin n, Real.exp (x j)
  Tensor.dim (fun i => Tensor.scalar (Real.exp (x i) / denom))

/-- The stabilized spec `softmaxVecSpec` agrees with `softmaxVecPlain` over `ℝ`. -/
private theorem softmax_vec_spec_eq_plain {n : Nat} (t : Tensor ℝ [Nat.succ n]) :
    Activation.softmaxVecSpec (α := ℝ) (n := Nat.succ n) t = softmaxVecPlain t := by
  apply TorchLean.Tensor.ext_vector
  intro i
  rw [Proofs.getScalar_softmaxVecSpec_eq_exp_div]
  simp [softmaxVecPlain]

/-- Plain softmax commutes with reindexing (permuting coordinates). -/
private theorem softmaxVecPlain_reindexOuter {n : Nat} (σ : Equiv.Perm (Fin n))
    (t : Tensor ℝ [n]) :
    softmaxVecPlain (reindexOuter (α := ℝ) (n := n) (s := .scalar) σ t)
      =
    reindexOuter (α := ℝ) (n := n) (s := .scalar) σ (softmaxVecPlain t) := by
  classical
  let x : Fin n → ℝ := fun j => t.getScalar j
  have hden :
      (∑ j : Fin n, Real.exp (x (σ j))) = ∑ j : Fin n, Real.exp (x j) := by
    simpa using (Equiv.sum_comp σ (fun j => Real.exp (x j)))
  have hden' :
      (∑ j : Fin n, Real.exp (t.unstack (σ j)).item) =
        ∑ j : Fin n, Real.exp (t.unstack j).item := by
    simpa [x, TorchLean.Tensor.getScalar, Spec.get] using hden
  apply TorchLean.Tensor.ext_vector
  intro i
  simp [softmaxVecPlain, reindexOuter, TorchLean.Tensor.getScalar, Spec.get, hden']

/-- Spec vector softmax commutes with reindexing, including for an empty vector. -/
theorem softmax_vec_spec_reindexOuter {n : Nat} (σ : Equiv.Perm (Fin n))
    (t : Tensor ℝ [n]) :
    Activation.softmaxVecSpec (α := ℝ) (n := n)
        (reindexOuter (α := ℝ) (n := n) (s := .scalar) σ t)
      =
    reindexOuter (α := ℝ) (n := n) (s := .scalar) σ
      (Activation.softmaxVecSpec (α := ℝ) (n := n) t) := by
  cases n with
  | zero => simp [Activation.softmaxVecSpec]
  | succ n =>
      simpa [softmax_vec_spec_eq_plain] using
        (softmaxVecPlain_reindexOuter (σ := σ) (t := t))

/-- Matrix axis-`1` softmax commutes with simultaneous row/column permutations. -/
theorem softmax_spec_permMatrix {n : Nat} (σ : Equiv.Perm (Fin n))
    (A : TorchLean.Tensor ℝ [n, n]) :
    Activation.softmaxSpec (α := ℝ) (s := [n, n]) 1
        (permMatrix (α := ℝ) (n := n) σ A)
      =
    permMatrix (α := ℝ) (n := n) σ
      (Activation.softmaxSpec (α := ℝ) (s := [n, n]) 1 A) := by
  let rows := A.unstack
  rw [← Tensor.dim_unstack A]
  change
    Activation.softmaxSpec 1
        (permMatrix σ (Tensor.dim rows)) =
      permMatrix σ (Activation.softmaxSpec 1 (Tensor.dim rows))
  have hswaps :
      Shape.moveAxisToInnermostSwaps (Shape.rank [n, n]) 1 = [] := by
    rfl
  have hsoftmax (matrix : Tensor ℝ [n, n]) :
      Activation.softmaxSpec (α := ℝ) (s := [n, n]) 1 matrix =
        Activation.Internal.softmaxInnermostSpec matrix := by
    unfold Activation.softmaxSpec
    rw [hswaps]
    simp only [TorchLean.Tensor.permuteByAdjacentSwaps, List.reverse_nil]
    rfl
  rw [hsoftmax, hsoftmax]
  apply Spec.matrix_ext
  intro i j
  rw [get2_permMatrix]
  change
    TorchLean.Tensor.getScalar
        ((Activation.Internal.softmaxInnermostSpec
          (permMatrix σ (Tensor.dim rows))).unstack i) j =
      TorchLean.Tensor.getScalar
        ((Activation.Internal.softmaxInnermostSpec (Tensor.dim rows)).unstack (σ i)) (σ j)
  rw [Activation.unstack_softmaxInnermostSpec_matrix]
  rw [Activation.unstack_softmaxInnermostSpec_matrix]
  have h := congrArg (fun vector : Tensor ℝ [n] => vector.getScalar j)
    (softmax_vec_spec_reindexOuter σ (rows (σ i)))
  simpa [permMatrix, reindexOuter, reindexCols, Spec.get,
    TorchLean.Tensor.getScalar] using h

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
    (A : Tensor ℝ [m, n])
    (B : Tensor ℝ [n, p]) :
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
    (A : Tensor ℝ [m, n])
    (B : Tensor ℝ [n, p]) :
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
    (σ : Equiv.Perm (Fin m)) (A : Tensor ℝ [m, n]) :
    TorchLean.Tensor.swapAdjacentAxes
        (reindexOuter (α := ℝ) (n := m) (s := .dim n .scalar) σ A) 0
      =
    reindexCols (α := ℝ) (m := n) (n := m) σ (TorchLean.Tensor.swapAdjacentAxes A 0) := by
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
    (A : Tensor ℝ [n, n]) (c : ℝ) :
    TorchLean.Tensor.scaleSpec (permMatrix (α := ℝ) (n := n) σ A) c
      =
    permMatrix (α := ℝ) (n := n) σ (TorchLean.Tensor.scaleSpec A c) := by
  classical
  -- Helper: extract a `scale_spec` entry.
  have get2_scale_spec {m n : Nat}
      (M : Tensor ℝ [m, n]) (c : ℝ) (i : Fin m) (j : Fin n) :
      get2 (TorchLean.Tensor.scaleSpec M c) i j = (get2 M i j) * c := by
    simp [TorchLean.Tensor.scaleSpec]
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
    (A : Tensor ℝ [n, n])
    (B : Tensor ℝ [n, d]) :
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
    (x : Tensor ℝ [n, dModel])
    (Wq : Tensor ℝ [dModel, projDim])
    (Wk : Tensor ℝ [dModel, projDim])
    (Wv : Tensor ℝ [dModel, projDim])
    (Wo : Tensor ℝ [projDim, dModel])
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
      let Q : Tensor ℝ [Nat.succ n', projDim] := matMulSpec x Wq
      let K : Tensor ℝ [Nat.succ n', projDim] := matMulSpec x Wk
      let V : Tensor ℝ [Nat.succ n', projDim] := matMulSpec x Wv

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
        { Q := Q, K := K, V := V, mask := none }
      let ctxσ : Spec.AttentionContext ℝ (Nat.succ n') (Nat.succ n') projDim h1 h1 :=
        { Q := reindexOuter (α := ℝ) (n := Nat.succ n') (s := .dim projDim .scalar) σ Q
          K := reindexOuter (α := ℝ) (n := Nat.succ n') (s := .dim projDim .scalar) σ K
          V := reindexOuter (α := ℝ) (n := Nat.succ n') (s := .dim projDim .scalar) σ V
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
                (TorchLean.Tensor.swapAdjacentAxes
                  (reindexOuter (α := ℝ) (n := Nat.succ n') (s := .dim projDim .scalar) σ K) 0)
              =
            permMatrix (α := ℝ) (n := Nat.succ n') σ
              (matMulSpec Q (TorchLean.Tensor.swapAdjacentAxes K 0)) := by
          -- Push `σ` through transpose (rows → cols), then apply the matmul permutation lemma.
          calc
            _ =
              matMulSpec
                (reindexOuter (α := ℝ) (n := Nat.succ n') (s := .dim projDim .scalar) σ Q)
                (reindexCols (α := ℝ) (m := projDim) (n := Nat.succ n') σ
                  (TorchLean.Tensor.swapAdjacentAxes K 0)) := by
                    simp [matrix_transpose_reindexOuter]
            _ =
              reindexOuter (α := ℝ) (n := Nat.succ n') (s := .dim (Nat.succ n') .scalar) σ
                (reindexCols (α := ℝ) (m := Nat.succ n') (n := Nat.succ n') σ
                  (matMulSpec Q (TorchLean.Tensor.swapAdjacentAxes K 0))) := by
                    simpa using
                      (mat_mul_reindexOuter_reindexCols (σ := σ) (τ := σ) (A := Q)
                        (B := TorchLean.Tensor.swapAdjacentAxes K 0))
            _ = _ := rfl

        -- scale commutes with `permMatrix`.
        have hScaledScores :
            TorchLean.Tensor.scaleSpec
                (permMatrix (α := ℝ) (n := Nat.succ n') σ
                  (matMulSpec Q (TorchLean.Tensor.swapAdjacentAxes K 0)))
                (Spec.attentionScaleDenom (α := ℝ) projDim)⁻¹
              =
            permMatrix (α := ℝ) (n := Nat.succ n') σ
              (TorchLean.Tensor.scaleSpec (matMulSpec Q (TorchLean.Tensor.swapAdjacentAxes K 0))
                (Spec.attentionScaleDenom (α := ℝ) projDim)⁻¹) := by
          simpa using
            (scale_spec_permMatrix (σ := σ)
              (A := matMulSpec Q (TorchLean.Tensor.swapAdjacentAxes K 0))
              (c := (Spec.attentionScaleDenom (α := ℝ) projDim)⁻¹))

        -- softmax commutes with `permMatrix`.
        have hWeights :
            Activation.softmaxSpec (α := ℝ) (s := [Nat.succ n', Nat.succ n']) 1
                (permMatrix (α := ℝ) (n := Nat.succ n') σ
                  (TorchLean.Tensor.scaleSpec (matMulSpec Q (TorchLean.Tensor.swapAdjacentAxes K 0))
                    (Spec.attentionScaleDenom (α := ℝ) projDim)⁻¹))
              =
            permMatrix (α := ℝ) (n := Nat.succ n') σ
              (Activation.softmaxSpec (α := ℝ) (s := [Nat.succ n', Nat.succ n']) 1
                (TorchLean.Tensor.scaleSpec (matMulSpec Q (TorchLean.Tensor.swapAdjacentAxes K 0))
                  (Spec.attentionScaleDenom (α := ℝ) projDim)⁻¹)) := by
          simpa using
            (SoftmaxEquivariance.softmax_spec_permMatrix (n := Nat.succ n') (σ := σ)
              (A := TorchLean.Tensor.scaleSpec
                (matMulSpec Q (TorchLean.Tensor.swapAdjacentAxes K 0))
                (Spec.attentionScaleDenom (α := ℝ) projDim)⁻¹))

        -- final matmul with `V` turns the conjugation into an outer reindexing.
        have hOut :
            matMulSpec
                (permMatrix (α := ℝ) (n := Nat.succ n') σ
                  (Activation.softmaxSpec (α := ℝ) (s := [Nat.succ n', Nat.succ n']) 1
                    (TorchLean.Tensor.scaleSpec
                      (matMulSpec Q (TorchLean.Tensor.swapAdjacentAxes K 0))
                      (Spec.attentionScaleDenom (α := ℝ) projDim)⁻¹)))
                (reindexOuter (α := ℝ) (n := Nat.succ n') (s := .dim projDim .scalar) σ V)
              =
            reindexOuter (α := ℝ) (n := Nat.succ n') (s := .dim projDim .scalar) σ
              (matMulSpec
                (Activation.softmaxSpec (α := ℝ) (s := [Nat.succ n', Nat.succ n']) 1
                  (TorchLean.Tensor.scaleSpec (matMulSpec Q (TorchLean.Tensor.swapAdjacentAxes K 0))
                    (Spec.attentionScaleDenom (α := ℝ) projDim)⁻¹))
                V) := by
          simpa using
            (mat_mul_permMatrix_reindexOuter (σ := σ)
              (A := Activation.softmaxSpec (α := ℝ) (s := [Nat.succ n', Nat.succ n']) 1
                (TorchLean.Tensor.scaleSpec (matMulSpec Q (TorchLean.Tensor.swapAdjacentAxes K 0))
                  (Spec.attentionScaleDenom (α := ℝ) projDim)⁻¹))
              (B := V))

        -- Put it all together by unfolding `scaledDotProductAttention` (mask = none).
        -- We avoid simp-loops by rewriting the intermediates explicitly.
        simp only [Spec.scaledDotProductAttention, ctxσ, ctx]
        simp only [one_div]
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
