/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Layers.Attention
public import TorchLean.Tensor.Reduce.Order

/-!
# Scaled matrix products in a reduction order

`Attention.scaledProductIn order a b scale` is the one-group `Attention.scaledProductsSlice`,
`scale * (a * b)` with the scale applied after the sum, in the order `order`. On a supported
carrier an entry folds the checked products `(value, flags)` of its row and column with
`order.fold` from `(0, 0)` and `Attention.checkedAdd`. That step adds the values and joins the
flags of both operands with those of the addition, and with one more when a nonzero right operand
is absorbed: in the left fold the right operand is the incoming product. When a flag of the fold
or of the scaling is set, the entry is replayed exactly and rounded once by
`Attention.replayProducts?`, as in the left fold, since the replay does not depend on the order. On
an unsupported carrier it is `Tensor.scale (Tensor.matmulIn order a b) scale`.
`Attention.scaledProductsIn` takes it to every coordinate of a shared batch prefix and
`Attention.scaledMatmulIn` broadcasts the batch prefixes first, as `Attention.scaledMatmul` does.

- `scaledProductIn_leftFold`, `scaledProductsIn_leftFold`, `scaledMatmulIn_leftFold`: in the left
  fold they are `scaledProductsSlice`, `scaledProducts` and `scaledMatmul` on one group, at every
  scalar.
- `scaledMatmulIn_real`: over `ℝ`, in an order that covers the inner dimension, `scaledMatmulIn`
  is `scaledMatmul`.
-/

@[expose] public section

namespace TorchLean.Attention

open Tensor

variable {α : Type} [Storage α] [Context α]

/-- The checked sum of two partial results `(value, flags)` of a scaled contraction: the sum of the
values, with the flags of both operands and of the addition, and one more when the right value is
nonzero and the sum is the left value. -/
def checkedAdd (l r : α × UInt8) : α × UInt8 :=
  let (total, flags) := l
  let (value, valueFlags) := r
  let (next, sumFlags) := (inferInstance : Numeric.QuotientArithmetic α).addWithFlags total value
  (next, flags ||| valueFlags ||| sumFlags ||| (value != 0 && next == total).toUInt8)

/-- `scale * (a * b)` in the order `order`, the scale after the sum. On a supported carrier each
entry folds the checked products of its row and column with `checkedAdd` and is replayed by
`replayProducts?` when a flag is set; otherwise it is `Tensor.scale (matmulIn order a b) scale`. -/
def scaledProductIn (order : ReduceOrder) {m n p : Nat} (a : Tensor α [m, n])
    (b : Tensor α [n, p]) (scale : α) : Tensor α [m, p] :=
  let ops := (inferInstance : Numeric.QuotientArithmetic α)
  if !ops.supported then Tensor.scale (matmulIn order a b) scale
  else
    Tensor.Rep.ofFn fun coordinate =>
      let (total, flags) := order.fold n checkedAdd (0, 0) fun index =>
        ops.mulWithFlags (get2 a coordinate.1 index) (get2 b index coordinate.2.1)
      let (ordinary, scaleFlags) := ops.mulWithFlags total scale
      if flags ||| scaleFlags == 0 then ordinary else
        (replayProducts? (fun _ : Fin 1 => a) (fun _ => b) scale ordinary coordinate.1
          coordinate.2.1).getD ordinary

/-- `scaledProductIn` at each coordinate of a shared batch prefix. -/
def scaledProductsIn (order : ReduceOrder) {batch : Shape} {m n p : Nat}
    (a : Tensor α (batch ++ [m, n])) (b : Tensor α (batch ++ [n, p])) (scale : α) :
    Tensor α (batch ++ [m, p]) :=
  match batch with
  | .scalar => scaledProductIn order a b scale
  | .dim _ _ => Tensor.dim fun index =>
      scaledProductsIn order (Tensor.unstack a index) (Tensor.unstack b index) scale

/-- `scaledMatmul` in the order `order`: both batch prefixes broadcast to `batch`, then
`scaledProductsIn`. -/
def scaledMatmulIn (order : ReduceOrder) {batchA batchB batch : Shape} {m n p : Nat}
    (broadcastA : Shape.CanBroadcastTo batchA batch)
    (broadcastB : Shape.CanBroadcastTo batchB batch)
    (left : Tensor α (batchA ++ [m, n])) (right : Tensor α (batchB ++ [n, p])) (scale : α) :
    Tensor α (batch ++ [m, p]) :=
  have hA := Shape.CanBroadcastTo.concat_right [m, n] broadcastA
  have hB := Shape.CanBroadcastTo.concat_right [n, p] broadcastB
  scaledProductsIn order (Tensor.broadcast left _ hA.dims hA.broadcastDims)
    (Tensor.broadcast right _ hB.dims hB.broadcastDims) scale

theorem scaledProductIn_leftFold {m n p : Nat} (a : Tensor α [m, n]) (b : Tensor α [n, p])
    (scale : α) :
    scaledProductIn .leftFold a b scale =
      scaledProductsSlice (fun _ : Fin 1 => a) (fun _ => b) scale := by
  unfold scaledProductIn scaledProductsSlice
  by_cases hs : (inferInstance : Numeric.QuotientArithmetic α).supported
  · simp only [hs, Bool.not_true, Bool.false_eq_true, ite_false, ReduceOrder.fold_leftFold,
      checkedAdd, UInt8.or_zero, List.finRange_zero, List.foldl_nil]
  · simp only [hs, Bool.not_false, ite_true, Bool.false_eq_true, ite_false, matmulIn_leftFold,
      List.finRange_zero, List.foldl_nil]

theorem scaledProductsIn_leftFold {batch : Shape} {m n p : Nat} (a : Tensor α (batch ++ [m, n]))
    (b : Tensor α (batch ++ [n, p])) (scale : α) :
    scaledProductsIn .leftFold a b scale =
      scaledProducts (fun _ : Fin 1 => a) (fun _ => b) scale := by
  rw [scaledProductsFast_eq]
  induction batch with
  | scalar => exact scaledProductIn_leftFold a b scale
  | dim k rest ih => simp only [scaledProductsIn, scaledProductsFast, ih]

theorem scaledMatmulIn_leftFold {batchA batchB batch : Shape} {m n p : Nat}
    (broadcastA : Shape.CanBroadcastTo batchA batch)
    (broadcastB : Shape.CanBroadcastTo batchB batch)
    (left : Tensor α (batchA ++ [m, n])) (right : Tensor α (batchB ++ [n, p])) (scale : α) :
    scaledMatmulIn .leftFold broadcastA broadcastB left right scale =
      scaledMatmul broadcastA broadcastB left right scale :=
  scaledProductsIn_leftFold _ _ _

private theorem scaledProductsIn_real {order : ReduceOrder} {batch : Shape} {m n p : Nat}
    (a : Tensor ℝ (batch ++ [m, n])) (b : Tensor ℝ (batch ++ [n, p])) (scale : ℝ)
    (hcov : order.Covers n) :
    scaledProductsIn order a b scale = scaledProductsIn .leftFold a b scale := by
  induction batch with
  | scalar =>
      have hm : matmulIn order a b = matmulIn .leftFold a b := by
        rw [matmulIn_leftFold]
        exact contractIn_eq_contract a b _ _ _ _ _ (hcov.mono (Nat.le_of_eq (Shape.size_vector n)))
      simp only [scaledProductsIn, scaledProductIn, Numeric.QuotientArithmetic.supported_real,
        Bool.not_false, ite_true, hm]
  | dim k rest ih => simp only [scaledProductsIn, ih]

/-- Over `ℝ`, `scaledMatmulIn` in an order that covers the inner dimension is `scaledMatmul`. -/
theorem scaledMatmulIn_real {order : ReduceOrder} {batchA batchB batch : Shape} {m n p : Nat}
    (broadcastA : Shape.CanBroadcastTo batchA batch)
    (broadcastB : Shape.CanBroadcastTo batchB batch)
    (left : Tensor ℝ (batchA ++ [m, n])) (right : Tensor ℝ (batchB ++ [n, p])) (scale : ℝ)
    (hcov : order.Covers n) :
    scaledMatmulIn order broadcastA broadcastB left right scale =
      scaledMatmul broadcastA broadcastB left right scale :=
  (scaledProductsIn_real _ _ scale hcov).trans (scaledMatmulIn_leftFold _ _ _ _ scale)

end TorchLean.Attention
