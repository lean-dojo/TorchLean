/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Attention.HardMask.FDeriv
public import TorchLean.Tensor.Algebra

/-!
# Pairing of the attention JVP and backward specifications

`Tensor.attentionJVP` and `Tensor.attentionBackward` are adjoint in all four tangents at once: the
query, the key, the value and the optional multiplier. The pairing holds over any leading batch
axes and for a fixed optional Boolean mask. Its proof reduces the score product to
`Attention.dot_valuesJVP` through two lemmas about the transposes in the batched scaled
matrix product: a batch slice of a transpose of the two trailing axes is the transpose of the
slice, and that transpose moves across the pairing `dot` as the reverse transpose.
-/

@[expose] public section

open TorchLean.Tensor
open scoped BigOperators
open Rep (dot)

noncomputable section

namespace TorchLean

/-- Slicing a conditional tensor slices the selected branch. -/
private theorem Attention.unstack_ite {β : Type} [Storage β] {k : Nat} {s : Shape} (c : Prop)
    [Decidable c] (x y : Tensor β (.dim k s)) (i : Fin k) :
    (if c then x else y).unstack i = if c then x.unstack i else y.unstack i :=
  apply_ite (Tensor.unstack · i) c x y

/-- A leading slice of an adjacent swap below the leading axis is the swap of the slice. -/
theorem Tensor.unstack_permute_adjacentSwap_succ {β : Type} [Storage β] {k : Nat}
    {s out : Shape} (x : Tensor β (.dim k s)) (depth : Nat)
    (hout : Shape.dim k out =
      Shape.permute (.dim k s) (Shape.adjacentSwap (.dim k s) (depth + 1)))
    (i : Fin k) :
    (permute x (Shape.adjacentSwap (.dim k s) (depth + 1)) (.dim k out) hout).unstack i =
      permute (x.unstack i) (s.adjacentSwap depth) out
        ((List.cons.inj (hout.trans (Shape.permute_adjacentSwap _ _))).2.trans
          (Shape.permute_adjacentSwap s depth).symm) := by
  obtain rfl : out = s.swapAdjacentAtDepth depth :=
    (List.cons.inj (hout.trans (Shape.permute_adjacentSwap _ _))).2
  rw [permute_adjacentSwap_succ, Tensor.unstack_dim]

/-- A leading slice of a batched matrix transpose is the transpose of the slice. -/
theorem Tensor.unstack_permute_batch_transpose {β : Type} [Storage β] {k m n : Nat}
    {batch : Shape} (x : Tensor β ((Shape.dim k batch) ++ [m, n]))
    (hout : (Shape.dim k batch) ++ [n, m] = ((Shape.dim k batch) ++ [m, n]).permute
      (((Shape.dim k batch) ++ [m, n]).adjacentSwap (Shape.dim k batch).rank))
    (i : Fin k) :
    (permute x (((Shape.dim k batch) ++ [m, n]).adjacentSwap (Shape.dim k batch).rank)
        ((Shape.dim k batch) ++ [n, m]) hout).unstack i =
      permute (s := batch ++ [m, n]) (x.unstack i) ((batch ++ [m, n]).adjacentSwap batch.rank)
        (batch ++ [n, m]) :=
  unstack_permute_adjacentSwap_succ x batch.rank hout i

/-- Transposing the trailing matrix axes of a batch moves across the pairing as the reverse
transpose. -/
theorem Tensor.dot_permute_batch_transpose {batch : Shape} {m n : Nat}
    (x : Tensor ℝ (batch ++ [m, n])) (y : Tensor ℝ (batch ++ [n, m])) :
    Rep.dot (permute x (Shape.adjacentSwap (batch ++ [m, n]) batch.rank)
        (batch ++ [n, m])) y =
      Rep.dot x (permute y (Shape.adjacentSwap (batch ++ [n, m]) batch.rank)
        (batch ++ [m, n])) := by
  induction batch with
  | scalar =>
    change Tensor ℝ [m, n] at x
    change Tensor ℝ [n, m] at y
    change Rep.dot (permute x (Shape.adjacentSwap [m, n] 0) [n, m]) y =
      Rep.dot x (permute y (Shape.adjacentSwap [n, m] 0) [m, n])
    rw [← dot_transpose (permute x (Shape.adjacentSwap [m, n] 0) [n, m]) y,
      transpose_transpose]
  | dim k batch ih =>
    rw [dot_eq_sum_unstack, dot_eq_sum_unstack]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [unstack_permute_batch_transpose x _ i, unstack_permute_batch_transpose y _ i]
    exact ih (x.unstack i) (y.unstack i)

/-- The batched product-rule contraction and the batched cotangent contractions are adjoint. -/
theorem Attention.dot_scaledProductsFast_two {batch : Shape} {m n p : Nat}
    (a da : Tensor ℝ (batch ++ [m, n])) (b db : Tensor ℝ (batch ++ [n, p]))
    (gradient : Tensor ℝ (batch ++ [m, p])) (c : ℝ) :
    Rep.dot (scaledProductsFast (fun i : Fin 2 => if i.val == 0 then da else a)
        (fun i => if i.val == 0 then b else db) c) gradient =
      Rep.dot da (scaledProductsFast (fun _ : Fin 1 => gradient)
          (fun _ => permute b (Shape.adjacentSwap (batch ++ [n, p]) batch.rank)
            (batch ++ [p, n])) c) +
        Rep.dot db (scaledProductsFast
          (fun _ : Fin 1 => permute a (Shape.adjacentSwap (batch ++ [m, n]) batch.rank)
            (batch ++ [n, m])) (fun _ => gradient) c) := by
  induction batch with
  | scalar =>
    simp only [scaledProductsFast,
      scaledProductsSlice_two Numeric.QuotientArithmetic.supported_real,
      scaledProductsSlice_one Numeric.QuotientArithmetic.supported_real,
      Bool.false_eq_true, ↓reduceIte]
    change Tensor ℝ [m, n] at a da
    change Tensor ℝ [n, p] at b db
    change Tensor ℝ [m, p] at gradient
    change Rep.dot (Tensor.scale (Tensor.matmul da b + Tensor.matmul a db) c) gradient =
      Rep.dot da (Tensor.scale (Tensor.matmul gradient
          (permute b (Shape.adjacentSwap [n, p] 0) [p, n])) c) +
        Rep.dot db (Tensor.scale (Tensor.matmul
          (permute a (Shape.adjacentSwap [m, n] 0) [n, m]) gradient) c)
    rw [Tensor.dot_scale_left, show Rep.dot (Tensor.matmul da b + Tensor.matmul a db) gradient =
        _ from Tensor.dot_add_left _ _ _, Tensor.dot_matmul_right_adjoint,
      Tensor.dot_matmul_left_adjoint, Tensor.dot_scale_right, Tensor.dot_scale_right,
      add_mul]
  | dim k batch ih =>
    simp only [scaledProductsFast]
    rw [Tensor.dot_eq_sum_unstack, Tensor.dot_eq_sum_unstack,
      Tensor.dot_eq_sum_unstack, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun i _ => ?_
    simp only [Tensor.unstack_dim]
    rw [unstack_permute_batch_transpose b _ i, unstack_permute_batch_transpose a _ i]
    simp only [List.cons_append, unstack_ite]
    exact ih (a.unstack i) (da.unstack i) (b.unstack i) (db.unstack i) (gradient.unstack i)

/-- At identity broadcasts the scaled product JVP and VJP are adjoint over any batch. -/
theorem Attention.dot_scaledMatmulJVP_refl {batch : Shape} {m n p : Nat}
    (a da : Tensor ℝ (batch ++ [m, n])) (b db : Tensor ℝ (batch ++ [n, p]))
    (gradient : Tensor ℝ (batch ++ [m, p])) (c : ℝ) :
    Rep.dot (scaledMatmulJVP (Shape.CanBroadcastTo.refl batch)
        (Shape.CanBroadcastTo.refl batch) a da b db c) gradient =
      Rep.dot da (scaledMatmulBackward (Shape.CanBroadcastTo.refl batch)
          (Shape.CanBroadcastTo.refl batch) a b gradient c).1 +
        Rep.dot db (scaledMatmulBackward (Shape.CanBroadcastTo.refl batch)
          (Shape.CanBroadcastTo.refl batch) a b gradient c).2 := by
  simp only [scaledMatmulJVP, scaledMatmulBackward,
    Tensor.broadcast_self, reduceBroadcast_self]
  rw [scaledProductsFast_eq]
  exact dot_scaledProductsFast_two a da b db gradient c

/-- Reading the multiplier tangent and its cotangent only when a multiplier is present gives the
same pairing on either side, and both sides are zero without a multiplier. -/
private theorem Attention.dot_elim_zeros {s : Shape} (weights dWeights : Option (Tensor ℝ s))
    (x : Tensor ℝ s) :
    Rep.dot (weights.elim (Tensor.zeros s) fun _ => dWeights.getD (Tensor.zeros s)) x =
      Rep.dot (dWeights.getD (Tensor.zeros s)) (weights.elim (Tensor.zeros s) fun _ => x) := by
  cases weights with
  | none => rw [Option.elim_none, Option.elim_none, Tensor.dot_full_zero_left,
      Tensor.dot_full_zero_right]
  | some w => rfl

/-- `Tensor.attentionJVP` and `Tensor.attentionBackward` are adjoint over any leading batch axes in
the query, key, value and multiplier tangents, for a fixed optional Boolean mask. -/
theorem Tensor.dot_attentionJVP {batch : Shape} {rows keys depth width : Nat}
    (query dQuery : Tensor ℝ (batch ++ [rows, depth]))
    (key dKey : Tensor ℝ (batch ++ [keys, depth]))
    (value dValue : Tensor ℝ (batch ++ [keys, width])) (scale : ℝ)
    (mask : Option (Tensor Bool [rows, keys]))
    (weights dWeights : Option (Tensor ℝ (batch ++ [rows, keys])))
    (gradient : Tensor ℝ (batch ++ [rows, width])) :
    let backward := Tensor.attentionBackward query key value scale mask weights gradient
    Rep.dot (Tensor.attentionJVP query key value scale mask weights dQuery dKey dValue dWeights)
        gradient =
      Rep.dot dQuery backward.1 + (Rep.dot dKey backward.2.1 +
        (Rep.dot dValue backward.2.2.1 +
          Rep.dot (dWeights.getD (Tensor.zeros _)) backward.2.2.2)) := by
  intro backward
  refine (Attention.dot_valuesJVP
    (Attention.scaledMatmul (Shape.CanBroadcastTo.refl batch)
      (Shape.CanBroadcastTo.refl batch) query (permute key
        (Shape.adjacentSwap (batch ++ [keys, depth]) batch.rank) (batch ++ [depth, keys]))
      scale)
    (Attention.scaledMatmulJVP (Shape.CanBroadcastTo.refl batch)
      (Shape.CanBroadcastTo.refl batch) query dQuery
      (permute key (Shape.adjacentSwap (batch ++ [keys, depth]) batch.rank)
        (batch ++ [depth, keys]))
      (permute dKey (Shape.adjacentSwap (batch ++ [keys, depth]) batch.rank)
        (batch ++ [depth, keys])) scale)
    (weights.getD (Tensor.full _ 1))
    (weights.elim (Tensor.zeros _) fun _ => dWeights.getD (Tensor.zeros _))
    value dValue gradient mask).trans ?_
  rw [Attention.dot_scaledMatmulJVP_refl, dot_permute_batch_transpose,
    Attention.dot_elim_zeros, add_assoc]
  rfl

end TorchLean
