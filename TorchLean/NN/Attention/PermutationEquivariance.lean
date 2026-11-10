/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Softmax
public import TorchLean.Tensor.Layers.Attention

/-!
# Permutation Equivariance of Self-Attention (No Positional Encoding)

Without positional information, self-attention is equivariant under permutations of the token axis:
reorder the input tokens and the output rows come back reordered the same way.
`selfAttention_reindexOuter` proves it for `Tensor.selfAttention` over `ℝ`.

The reindexing adapters `reindexOuter`, `reindexCols` and `permMatrix` live in the model-proof
namespace, outside the general `TorchLean.Tensor` API, and describe the token permutations the
attention theorems use.
-/

@[expose] public section

open scoped BigOperators

noncomputable section

namespace TorchLean.Attention

open Tensor

/-!
## Token reindexing

The adapters rebuild tensor slices in the order the permutation picks, so along each permuted axis
position `i` of the result holds position `σ i` of the input.
-/

/-- Reindex the outermost axis of a tensor by a permutation. -/
def reindexOuter {α : Type} [Storage α] {n : Nat} {s : Shape} (σ : Equiv.Perm (Fin n)) :
    Tensor α (.dim n s) → Tensor α (.dim n s) :=
  fun tensor => Tensor.dim (fun i => tensor.unstack (σ i))

/-- Slice `i` of the permuted tensor is slice `σ i` of the original. -/
@[simp] theorem unstack_reindexOuter {α : Type} [Storage α] {n : Nat} {s : Shape}
    (σ : Equiv.Perm (Fin n)) (t : Tensor α (.dim n s)) (i : Fin n) :
    Tensor.unstack (reindexOuter (α := α) (n := n) (s := s) σ t) i = Tensor.unstack t (σ i) := by
  simp [reindexOuter]

/-- Reindex the *column* axis of a matrix by a permutation. -/
def reindexCols {α : Type} [Storage α] {m n : Nat} (σ : Equiv.Perm (Fin n)) :
    Tensor α [m, n] → Tensor α [m, n] :=
  fun matrix =>
    Tensor.dim (fun i =>
      reindexOuter (α := α) (n := n) (s := .scalar) σ (matrix.unstack i))

/-- On a matrix, reindexing the outer axis permutes the rows: entry `(i, j)` is entry `(σ i, j)`
of the original. -/
@[simp] theorem get2_reindexOuter {α : Type} [Storage α] {m n : Nat}
    (σ : Equiv.Perm (Fin m)) (A : Tensor α [m, n]) (i : Fin m) (j : Fin n) :
    Tensor.get2 (reindexOuter (α := α) (n := m) (s := .dim n .scalar) σ A) i j =
      Tensor.get2 A (σ i) j := by
  simp [Tensor.get2]

/-- Reindexing the columns of a matrix permutes them: entry `(i, j)` is entry `(i, σ j)` of the
original. -/
@[simp] theorem get2_reindexCols {α : Type} [Storage α] {m n : Nat}
    (σ : Equiv.Perm (Fin n)) (A : Tensor α [m, n]) (i : Fin m) (j : Fin n) :
    Tensor.get2 (reindexCols (α := α) (m := m) (n := n) σ A) i j =
      Tensor.get2 A i (σ j) := by
  simp [reindexCols, reindexOuter, Tensor.get2, Tensor.getScalar]

/-- Simultaneously permute rows and columns of an `n×n` matrix by the same permutation. -/
def permMatrix {α : Type} [Storage α] {n : Nat} (σ : Equiv.Perm (Fin n))
    (A : Tensor α [n, n]) : Tensor α [n, n] :=
  reindexOuter (α := α) (n := n) (s := .dim n .scalar) σ
    (reindexCols (α := α) (m := n) (n := n) σ A)

/-- Entry `(i, j)` of a simultaneously permuted matrix is entry `(σ i, σ j)` of the original.

Attention scores transform this way under a token permutation, because the query index and the
key index move with the same `σ`. -/
@[simp] theorem get2_permMatrix {α : Type} [Storage α] {n : Nat}
    (σ : Equiv.Perm (Fin n)) (A : Tensor α [n, n]) (i j : Fin n) :
    Tensor.get2 (permMatrix (α := α) (n := n) σ A) i j = Tensor.get2 A (σ i) (σ j) := by
  simp [permMatrix]

/-!
## Softmax equivariance

The spec softmax on vectors subtracts the maximum before exponentiating,
`x ↦ exp(x - m) / Σ exp(x - m)`. Over `ℝ` it equals the plain `exp(x) / Σ exp(x)`
(`Softmax.getScalar_vec_eq_exp_div`), so the equivariance proof never needs to know how
the shift `m` is chosen.
-/

namespace SoftmaxEquivariance

/-- Spec vector softmax commutes with reindexing, including for an empty vector. -/
theorem _root_.TorchLean.Softmax.vec_reindexOuter {n : Nat} (σ : Equiv.Perm (Fin n))
    (t : Tensor ℝ [n]) :
    Softmax.vec (α := ℝ) (n := n)
        (reindexOuter (α := ℝ) (n := n) (s := .scalar) σ t)
      =
    reindexOuter (α := ℝ) (n := n) (s := .scalar) σ
      (Softmax.vec (α := ℝ) (n := n) t) := by
  have hget (u : Tensor ℝ [n]) (j : Fin n) :
      (reindexOuter (α := ℝ) (n := n) (s := .scalar) σ u).getScalar j = u.getScalar (σ j) := by
    simp [reindexOuter, Tensor.getScalar]
  apply Tensor.ext_vector
  intro i
  simp only [hget, Softmax.getScalar_vec_eq_exp_div]
  rw [Equiv.sum_comp σ (fun j => Real.exp (t.getScalar j))]

/-- Matrix axis-`1` softmax commutes with simultaneous row/column permutations. -/
theorem _root_.TorchLean.Attention.softmax_permMatrix {n : Nat} (σ : Equiv.Perm (Fin n))
    (A : Tensor ℝ [n, n]) :
    Tensor.softmax (permMatrix (α := ℝ) (n := n) σ A) ⟨1, Nat.one_lt_two⟩
      =
    permMatrix (α := ℝ) (n := n) σ
      (Tensor.softmax (α := ℝ) (s := [n, n]) A ⟨1, Shape.AxisInBounds.proof⟩) := by
  let rows := A.unstack
  rw [← Tensor.dim_unstack A]
  change
    Tensor.softmax (permMatrix σ (Tensor.dim rows)) ⟨1, Shape.AxisInBounds.proof⟩ =
      permMatrix σ (Tensor.softmax (Tensor.dim rows) ⟨1, Shape.AxisInBounds.proof⟩)
  have hsoftmax (matrix : Tensor ℝ [n, n]) :
      Tensor.softmax (α := ℝ) (s := [n, n]) matrix ⟨1, Shape.AxisInBounds.proof⟩ =
        Softmax.innermost matrix :=
    rfl
  rw [hsoftmax, hsoftmax]
  apply Tensor.ext_get2
  intro i j
  rw [get2_permMatrix]
  change
    Tensor.getScalar
        ((Softmax.innermost
          (permMatrix σ (Tensor.dim rows))).unstack i) j =
      Tensor.getScalar
        ((Softmax.innermost (Tensor.dim rows)).unstack (σ i)) (σ j)
  rw [Softmax.unstack_innermost_matrix]
  rw [Softmax.unstack_innermost_matrix]
  have h := congrArg (fun vector : Tensor ℝ [n] => vector.getScalar j)
    (Softmax.vec_reindexOuter σ (rows (σ i)))
  simpa [permMatrix, reindexOuter, reindexCols,
    Tensor.getScalar] using h

end SoftmaxEquivariance

/-!
## Linear algebra: matmul/transpose commute with permutations
-/

/--
Reindexing the rows of the left factor by `σ` and the columns of the right factor by `τ`
reindexes the rows of the product by `σ` and its columns by `τ`.

The permutations are arbitrary, and the score step of the main theorem takes `τ = σ`.
-/
theorem matmul_reindexOuter_reindexCols {m n p : Nat}
    (σ : Equiv.Perm (Fin m)) (τ : Equiv.Perm (Fin p))
    (A : Tensor ℝ [m, n])
    (B : Tensor ℝ [n, p]) :
    Tensor.matmul (reindexOuter (α := ℝ) (n := m) (s := .dim n .scalar) σ A)
        (reindexCols (α := ℝ) (m := n) (n := p) τ B)
      =
    reindexOuter (α := ℝ) (n := m) (s := .dim p .scalar) σ
      (reindexCols (α := ℝ) (m := m) (n := p) τ (Tensor.matmul A B)) := by
  classical
  apply Tensor.ext_get2
  intro i j
  simp [Tensor.get2_matmul_eq_sum]

/--
Permuting the rows of the left factor permutes the rows of the product the same way.

For a projection `x W` this is token equivariance: every token row meets the same weight matrix,
so reordering the tokens before the projection reorders the output rows.
-/
theorem matmul_reindexOuter_left {m n p : Nat}
    (σ : Equiv.Perm (Fin m))
    (A : Tensor ℝ [m, n])
    (B : Tensor ℝ [n, p]) :
    Tensor.matmul (reindexOuter (α := ℝ) (n := m) (s := .dim n .scalar) σ A) B
      =
    reindexOuter (α := ℝ) (n := m) (s := .dim p .scalar) σ (Tensor.matmul A B) := by
  classical
  apply Tensor.ext_get2
  intro i j
  simp [Tensor.get2_matmul_eq_sum]

/--
Transposition converts a row permutation into a column permutation.

The attention-score proof uses this to turn `(P Q) (P K)ᵀ` into a simultaneous row/column
permutation of `Q Kᵀ`.
-/
theorem permute_adjacentSwap_reindexOuter {m n : Nat}
    (σ : Equiv.Perm (Fin m)) (A : Tensor ℝ [m, n]) :
    Tensor.permute (reindexOuter (α := ℝ) (n := m) (s := .dim n .scalar) σ A)
        (Shape.adjacentSwap [m, n] 0) [n, m]
      =
    reindexCols (α := ℝ) (m := n) (n := m) σ (Tensor.permute A
        (Shape.adjacentSwap [m, n] 0) [n, m]) := by
  classical
  apply Tensor.ext_get2
  intro i j
  simp [Tensor.get2_transpose]

/-!
## Main theorem: self-attention equivariance
-/

/--
Elementwise scaling commutes with simultaneous row/column reindexing.

Scaled dot-product attention divides every score by the same scalar, so the main proof can move
the scaling past the conjugation of the score matrix.
-/
theorem scale_permMatrix {n : Nat} (σ : Equiv.Perm (Fin n))
    (A : Tensor ℝ [n, n]) (c : ℝ) :
    Tensor.scale (permMatrix (α := ℝ) (n := n) σ A) c
      =
    permMatrix (α := ℝ) (n := n) σ (Tensor.scale A c) := by
  classical
  have get2_scale {m n : Nat}
      (M : Tensor ℝ [m, n]) (c : ℝ) (i : Fin m) (j : Fin n) :
      get2 (Tensor.scale M c) i j = (get2 M i j) * c := by
    simp [Tensor.scale]
  apply Tensor.ext_get2
  intro i j
  -- Both sides reduce to `(A[σ i, σ j]) * c`.
  simp [get2_permMatrix, get2_scale]

/--
Multiplying a simultaneously permuted attention matrix by a row-permuted value matrix row-permutes
the product.

In the main proof it is the last step inside the attention block: the column permutation of the
weights and the row permutation of the values cancel on the summation index.
-/
theorem matmul_permMatrix_reindexOuter
    {n d : Nat} (σ : Equiv.Perm (Fin n))
    (A : Tensor ℝ [n, n])
    (B : Tensor ℝ [n, d]) :
    Tensor.matmul (permMatrix (α := ℝ) (n := n) σ A)
        (reindexOuter (α := ℝ) (n := n) (s := .dim d .scalar) σ B)
      =
    reindexOuter (α := ℝ) (n := n) (s := .dim d .scalar) σ
      (Tensor.matmul A B) := by
  classical
  apply Tensor.ext_get2
  intro i j
  -- Entrywise: reindexing the shared summation index is a change of variables by `σ`.
  simp [Tensor.get2_matmul_eq_sum]
  simpa using
    (Equiv.sum_comp σ (fun k : Fin n => (get2 A (σ i) k) * (get2 B k j)))

/--
Self-attention without positional encodings is equivariant under every token permutation:
reordering the input rows by `σ` reorders the output rows by `σ`.

The proof follows the permutation through the block. `Q`, `K` and `V` are reordered, the score
matrix is conjugated by the permutation, the row-wise softmax commutes with that conjugation, and
the products with `V` and `Wo` turn it back into a reordering of the rows. The statement is about
the spec block and does not cover the CUDA kernels.
-/
theorem _root_.TorchLean.Tensor.selfAttention_reindexOuter
    {n dModel projDim : Nat}
    (σ : Equiv.Perm (Fin n))
    (x : Tensor ℝ [n, dModel])
    (Wq : Tensor ℝ [dModel, projDim])
    (Wk : Tensor ℝ [dModel, projDim])
    (Wv : Tensor ℝ [dModel, projDim])
    (Wo : Tensor ℝ [projDim, dModel])
    (h1 : n ≠ 0) :
    Tensor.selfAttention (α := ℝ) (n := n) (dModel := dModel) (projDim := projDim)
        (x := reindexOuter (α := ℝ) (n := n) (s := .dim dModel .scalar) σ x)
        (Wq := Wq) (Wk := Wk) (Wv := Wv) (Wo := Wo) h1
      =
    reindexOuter (α := ℝ) (n := n) (s := .dim dModel .scalar) σ
      (Tensor.selfAttention (α := ℝ) (n := n) (dModel := dModel) (projDim := projDim)
        (x := x) (Wq := Wq) (Wk := Wk) (Wv := Wv) (Wo := Wo) h1) := by
  classical
  cases n with
  | zero =>
      cases (h1 rfl)
  | succ n' =>
      let Q : Tensor ℝ [Nat.succ n', projDim] := Tensor.matmul x Wq
      let K : Tensor ℝ [Nat.succ n', projDim] := Tensor.matmul x Wk
      let V : Tensor ℝ [Nat.succ n', projDim] := Tensor.matmul x Wv

      -- Input projections commute with token permutation.
      have hQ :
          Tensor.matmul (reindexOuter (α := ℝ) (n := Nat.succ n') (s := .dim dModel .scalar) σ x) Wq
            =
          reindexOuter (α := ℝ) (n := Nat.succ n') (s := .dim projDim .scalar) σ Q := by
        simpa [Q] using
          (matmul_reindexOuter_left (σ := σ) (A := x) (B := Wq))
      have hK :
          Tensor.matmul (reindexOuter (α := ℝ) (n := Nat.succ n') (s := .dim dModel .scalar) σ x) Wk
            =
          reindexOuter (α := ℝ) (n := Nat.succ n') (s := .dim projDim .scalar) σ K := by
        simpa [K] using
          (matmul_reindexOuter_left (σ := σ) (A := x) (B := Wk))
      have hV :
          Tensor.matmul (reindexOuter (α := ℝ) (n := Nat.succ n') (s := .dim dModel .scalar) σ x) Wv
            =
          reindexOuter (α := ℝ) (n := Nat.succ n') (s := .dim projDim .scalar) σ V := by
        simpa [V] using
          (matmul_reindexOuter_left (σ := σ) (A := x) (B := Wv))

      -- Attention contexts for the original tokens and for the permuted ones.
      let ctx : Inputs ℝ (Nat.succ n') (Nat.succ n') projDim h1 h1 :=
        { query := Q, key := K, value := V, mask := none }
      let ctxσ : Inputs ℝ (Nat.succ n') (Nat.succ n') projDim h1 h1 :=
        { query := reindexOuter (α := ℝ) (n := Nat.succ n') (s := .dim projDim .scalar) σ Q
          key := reindexOuter (α := ℝ) (n := Nat.succ n') (s := .dim projDim .scalar) σ K
          value := reindexOuter (α := ℝ) (n := Nat.succ n') (s := .dim projDim .scalar) σ V
          mask := none }

      -- Equivariance of the scaled dot-product attention block (mask = none).
      have hSDA :
          scaledDotProduct (α := ℝ) (ctx := ctxσ) =
            reindexOuter (α := ℝ) (n := Nat.succ n') (s := .dim projDim .scalar) σ
              (scaledDotProduct (α := ℝ) (ctx := ctx)) := by
        -- Unfold the definition (mask = none) and rewrite each intermediate.
        -- scores: `Q Kᵀ` is conjugated by `permMatrix σ`.
        have hScores :
            Tensor.matmul
                (reindexOuter (α := ℝ) (n := Nat.succ n') (s := .dim projDim .scalar) σ Q)
                (Tensor.permute
                  (reindexOuter (α := ℝ) (n := Nat.succ n') (s := .dim projDim .scalar) σ K)
                  (Shape.adjacentSwap [n'.succ, projDim] 0) [projDim, n'.succ])
              =
            permMatrix (α := ℝ) (n := Nat.succ n') σ
              (Tensor.matmul Q (Tensor.permute K
                  (Shape.adjacentSwap [n'.succ, projDim] 0) [projDim, n'.succ])) := by
          -- Push `σ` through transpose (rows → cols), then apply the matmul permutation lemma.
          calc
            _ =
              Tensor.matmul
                (reindexOuter (α := ℝ) (n := Nat.succ n') (s := .dim projDim .scalar) σ Q)
                (reindexCols (α := ℝ) (m := projDim) (n := Nat.succ n') σ
                  (Tensor.permute K
                      (Shape.adjacentSwap [n'.succ, projDim] 0) [projDim, n'.succ])) := by
                    simp [permute_adjacentSwap_reindexOuter]
            _ =
              reindexOuter (α := ℝ) (n := Nat.succ n') (s := .dim (Nat.succ n') .scalar) σ
                (reindexCols (α := ℝ) (m := Nat.succ n') (n := Nat.succ n') σ
                  (Tensor.matmul Q (Tensor.permute K
                      (Shape.adjacentSwap [n'.succ, projDim] 0) [projDim, n'.succ]))) := by
                    simpa using
                      (matmul_reindexOuter_reindexCols (σ := σ) (τ := σ) (A := Q)
                        (B := Tensor.permute K
                            (Shape.adjacentSwap [n'.succ, projDim] 0) [projDim, n'.succ]))
            _ = _ := rfl

        -- scale commutes with `permMatrix`.
        have hScaledScores :
            Tensor.scale
                (permMatrix (α := ℝ) (n := Nat.succ n') σ
                  (Tensor.matmul Q (Tensor.permute K
                      (Shape.adjacentSwap [n'.succ, projDim] 0) [projDim, n'.succ])))
                (scaleDenom (α := ℝ) projDim)⁻¹
              =
            permMatrix (α := ℝ) (n := Nat.succ n') σ
              (Tensor.scale (Tensor.matmul Q (Tensor.permute K
                  (Shape.adjacentSwap [n'.succ, projDim] 0) [projDim, n'.succ]))
                (scaleDenom (α := ℝ) projDim)⁻¹) := by
          simpa using
            (scale_permMatrix (σ := σ)
              (A := Tensor.matmul Q (Tensor.permute K
                  (Shape.adjacentSwap [n'.succ, projDim] 0) [projDim, n'.succ]))
              (c := (scaleDenom (α := ℝ) projDim)⁻¹))

        -- softmax commutes with `permMatrix`.
        have hWeights :
            Tensor.softmax (α := ℝ) (s := [Nat.succ n',
                Nat.succ n']) (permMatrix (α := ℝ) (n := Nat.succ n') σ
                  (Tensor.scale
                      (Tensor.matmul Q (Tensor.permute K
                          (Shape.adjacentSwap [n'.succ, projDim] 0) [projDim, n'.succ]))
                    (scaleDenom (α := ℝ) projDim)⁻¹)) ⟨1, Shape.AxisInBounds.proof⟩
              =
            permMatrix (α := ℝ) (n := Nat.succ n') σ
              (Tensor.softmax (α := ℝ) (s := [Nat.succ n',
                  Nat.succ n']) (Tensor.scale (Tensor.matmul Q
                      (Tensor.permute K
                          (Shape.adjacentSwap [n'.succ, projDim] 0) [projDim, n'.succ]))
                  (scaleDenom (α := ℝ) projDim)⁻¹) ⟨1, Nat.one_lt_two⟩) := by
          simpa using
            (softmax_permMatrix (n := Nat.succ n') (σ := σ)
              (A := Tensor.scale
                (Tensor.matmul Q (Tensor.permute K
                    (Shape.adjacentSwap [n'.succ, projDim] 0) [projDim, n'.succ]))
                (scaleDenom (α := ℝ) projDim)⁻¹))

        -- final matmul with `V` turns the conjugation into an outer reindexing.
        have hOut :
            Tensor.matmul
                (permMatrix (α := ℝ) (n := Nat.succ n') σ
                  (Tensor.softmax (α := ℝ) (s := [Nat.succ n',
                      Nat.succ n']) (Tensor.scale
                      (Tensor.matmul Q (Tensor.permute K
                          (Shape.adjacentSwap [n'.succ, projDim] 0) [projDim, n'.succ]))
                      (scaleDenom (α := ℝ) projDim)⁻¹) ⟨1, Shape.AxisInBounds.proof⟩))
                (reindexOuter (α := ℝ) (n := Nat.succ n') (s := .dim projDim .scalar) σ V)
              =
            reindexOuter (α := ℝ) (n := Nat.succ n') (s := .dim projDim .scalar) σ
              (Tensor.matmul
                (Tensor.softmax (α := ℝ) (s := [Nat.succ n',
                    Nat.succ n']) (Tensor.scale (Tensor.matmul Q
                        (Tensor.permute K
                            (Shape.adjacentSwap [n'.succ, projDim] 0) [projDim, n'.succ]))
                    (scaleDenom (α := ℝ) projDim)⁻¹) ⟨1, Shape.AxisInBounds.proof⟩)
                V) := by
          simpa using
            (matmul_permMatrix_reindexOuter (σ := σ)
              (A := Tensor.softmax (α := ℝ) (s := [Nat.succ n',
                  Nat.succ n']) (Tensor.scale (Tensor.matmul Q
                      (Tensor.permute K
                          (Shape.adjacentSwap [n'.succ, projDim] 0) [projDim, n'.succ]))
                  (scaleDenom (α := ℝ) projDim)⁻¹) ⟨1, Shape.AxisInBounds.proof⟩)
              (B := V))

        -- Rewriting the intermediates one at a time with `rw` keeps `simp` from looping.
        simp only [scaledDotProduct, ctxσ, ctx]
        simp only [scaledProductsSlice_one Numeric.QuotientArithmetic.supported_real,
          Bool.false_eq_true, ↓reduceIte]
        simp only [one_div]
        rw [hScores]
        rw [hScaledScores]
        rw [hWeights]
        simpa using hOut

      calc
        Tensor.selfAttention (α := ℝ) (n := Nat.succ n') (dModel := dModel) (projDim := projDim)
            (x := reindexOuter (α := ℝ) (n := Nat.succ n') (s := .dim dModel .scalar) σ x)
            (Wq := Wq) (Wk := Wk) (Wv := Wv) (Wo := Wo) h1
            =
          Tensor.matmul (scaledDotProduct (α := ℝ) (ctx := ctxσ)) Wo := by
            simp [Tensor.selfAttention, Q, K, V, hQ, hK, hV, ctxσ]
        _ =
          Tensor.matmul
            (reindexOuter (α := ℝ) (n := Nat.succ n') (s := .dim projDim .scalar) σ
              (scaledDotProduct (α := ℝ) (ctx := ctx)))
            Wo := by
              simp [hSDA]
        _ =
          reindexOuter (α := ℝ) (n := Nat.succ n') (s := .dim dModel .scalar) σ
            (Tensor.matmul (scaledDotProduct (α := ℝ) (ctx := ctx)) Wo) := by
              -- push `σ` through the final projection `Wo`
              exact
                (matmul_reindexOuter_left (σ := σ)
                  (A := scaledDotProduct (α := ℝ) (ctx := ctx)) (B := Wo))
        _ =
          reindexOuter (α := ℝ) (n := Nat.succ n') (s := .dim dModel .scalar) σ
            (Tensor.selfAttention (α := ℝ) (n := Nat.succ n') (dModel := dModel)
              (projDim := projDim)
              (x := x) (Wq := Wq) (Wk := Wk) (Wv := Wv) (Wo := Wo) h1) := by
              simp [Tensor.selfAttention, Q, K, V, ctx]

end TorchLean.Attention
