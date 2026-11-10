/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.NN.Op.Attention
public import TorchLean.Verify.RuntimeApprox.OnlineSoftmax.Real

/-!
# The derivative of attention at every schedule

Over the reals, attention by the online schedule is attention by the two-pass schedule
(`Tensor.attention_online_eq`). So at every schedule, attention without a multiplier is the value
product with the multiplier one of the hard-masked weights of the scaled score product
(`Tensor.attention_none_real`), and its stored backward is the backward of that composite
(`Tensor.attentionBackward_none_real`). With the `@[fun_prop]` rules of
`TorchLean.NN.Op.Attention`, `op_deriv` proves that the stored backward of `Op.attention` is the
adjoint of the Fréchet derivative of the real forward, jointly in the query, the key and the
value, at every mask, schedule and scale (`Op.attention_prod_backward_eq_adjoint`).
-/

@[expose] public section

open TorchLean.Shape TorchLean.Tensor

namespace TorchLean

namespace Tensor

variable {batch : Shape} {rows keys depth width : Nat}

/-- Over the reals, attention without a multiplier is, at every schedule, the value product with
the multiplier one of the hard-masked weights of the scaled score product. -/
@[op_deriv_simps]
theorem attention_none_real (query : Tensor ℝ (batch ++ [rows, depth]))
    (key : Tensor ℝ (batch ++ [keys, depth])) (value : Tensor ℝ (batch ++ [keys, width]))
    (scale : ℝ) (mask : Option (Tensor Bool [rows, keys])) (schedule : Softmax.Schedule) :
    attention query key value scale mask none schedule =
      Attention.values (Attention.weights (Attention.scaledMatmul
        (Shape.CanBroadcastTo.refl batch) (Shape.CanBroadcastTo.refl batch) query
        (permute key (Shape.adjacentSwap (batch ++ [keys, depth]) batch.rank)
          (batch ++ [depth, keys])) scale) mask) value (some (full _ 1)) := by
  cases schedule with
  | twoPass => exact Attention.values_none_eq _ _
  | online blocks =>
    rw [attention_online_eq]
    exact Attention.values_none_eq _ _

/-- Over the reals, the attention backward without a multiplier applies `Attention.valuesBackward`
with the multiplier one, then `Attention.scaledMatmulBackward` and the key transpose, and gives
the multiplier the cotangent zero. -/
@[op_deriv_simps]
theorem attentionBackward_none_real (query : Tensor ℝ (batch ++ [rows, depth]))
    (key : Tensor ℝ (batch ++ [keys, depth])) (value : Tensor ℝ (batch ++ [keys, width]))
    (scale : ℝ) (mask : Option (Tensor Bool [rows, keys]))
    (gradient : Tensor ℝ (batch ++ [rows, width])) :
    attentionBackward query key value scale mask none gradient =
      ((Attention.scaledMatmulBackward (Shape.CanBroadcastTo.refl batch)
          (Shape.CanBroadcastTo.refl batch) query
          (permute key (Shape.adjacentSwap (batch ++ [keys, depth]) batch.rank)
            (batch ++ [depth, keys]))
          (Attention.valuesBackward (Attention.weights (Attention.scaledMatmul
            (Shape.CanBroadcastTo.refl batch) (Shape.CanBroadcastTo.refl batch) query
            (permute key (Shape.adjacentSwap (batch ++ [keys, depth]) batch.rank)
              (batch ++ [depth, keys])) scale) mask) value (full _ 1) gradient).1 scale).1,
        permute (Attention.scaledMatmulBackward (Shape.CanBroadcastTo.refl batch)
          (Shape.CanBroadcastTo.refl batch) query
          (permute key (Shape.adjacentSwap (batch ++ [keys, depth]) batch.rank)
            (batch ++ [depth, keys]))
          (Attention.valuesBackward (Attention.weights (Attention.scaledMatmul
            (Shape.CanBroadcastTo.refl batch) (Shape.CanBroadcastTo.refl batch) query
            (permute key (Shape.adjacentSwap (batch ++ [keys, depth]) batch.rank)
              (batch ++ [depth, keys])) scale) mask) value (full _ 1) gradient).1 scale).2
          (Shape.adjacentSwap (batch ++ [depth, keys]) batch.rank) (batch ++ [keys, depth]),
        (Attention.valuesBackward (Attention.weights (Attention.scaledMatmul
          (Shape.CanBroadcastTo.refl batch) (Shape.CanBroadcastTo.refl batch) query
          (permute key (Shape.adjacentSwap (batch ++ [keys, depth]) batch.rank)
            (batch ++ [depth, keys])) scale) mask) value (full _ 1) gradient).2.1,
        zeros _) :=
  rfl

end Tensor

namespace Op

/-- At every leading shape, mask, schedule and frozen scale, attention is differentiable jointly
in the query, the key and the value, and its stored backward is the adjoint of the derivative.
A row without an allowed key has constant zero weights. -/
@[op_cert TorchLean.Op.attention derivative 0 1 2]
theorem attention_prod_backward_eq_adjoint {batch : Shape} {rows keys depth width : Nat}
    (mask : Option (Tensor Bool [rows, keys])) (schedule : Softmax.Schedule)
    (q : Tensor ℝ (batch ++ [rows, depth])) (k : Tensor ℝ (batch ++ [keys, depth]))
    (v : Tensor ℝ (batch ++ [keys, width])) (scale : Tensor ℝ [])
    (δ : Tensor ℝ (batch ++ [rows, width])) :
    DifferentiableAt ℝ (fun p : WithLp 2 (Tensor ℝ (batch ++ [rows, depth]) ×
          WithLp 2 (Tensor ℝ (batch ++ [keys, depth]) × Tensor ℝ (batch ++ [keys, width]))) =>
        (((Op.attention (batch := batch) (depth := depth) (width := width) mask schedule).rule?
          (α := ℝ) |>.get rfl).forward (p.fst, p.snd.fst, p.snd.snd, scale, ())).1)
      (WithLp.toLp 2 (q, WithLp.toLp 2 (k, v))) ∧
    WithLp.toLp 2
        ((((Op.attention (batch := batch) (depth := depth) (width := width) mask schedule).rule?
          (α := ℝ) |>.get rfl).backward (q, k, v, scale, ()) (δ, ())).1,
        WithLp.toLp 2
          ((((Op.attention (batch := batch) (depth := depth) (width := width) mask
            schedule).rule? (α := ℝ) |>.get rfl).backward (q, k, v, scale, ()) (δ, ())).2.1,
          (((Op.attention (batch := batch) (depth := depth) (width := width) mask
            schedule).rule? (α := ℝ) |>.get rfl).backward (q, k, v, scale, ()) (δ, ())).2.2.1)) =
      (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ (batch ++ [rows, depth]) ×
          WithLp 2 (Tensor ℝ (batch ++ [keys, depth]) × Tensor ℝ (batch ++ [keys, width]))) =>
        (((Op.attention (batch := batch) (depth := depth) (width := width) mask schedule).rule?
          (α := ℝ) |>.get rfl).forward (p.fst, p.snd.fst, p.snd.snd, scale, ())).1)
        (WithLp.toLp 2 (q, WithLp.toLp 2 (k, v)))).adjoint δ := by
  op_deriv

end Op

end TorchLean
