/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.NN.Op.Layout
public import TorchLean.NN.Tactic.OpDeriv

/-!
# `op_deriv` through a concatenation, and the certificate of `Op.cat`

The two theorems are composites that `op_deriv` proves through the `fun_prop` rule of the
concatenation of a family of maps, at two parts and at any number of parts, with the backward
pass written as the composed pullback. The examples after them instantiate the derivative
certificate of `Op.cat` at the sizes `[2, 3]` along the axis of `[5]` and at a variable list of
sizes along the first axis of a shape `[_, 4]`.
-/

@[expose] public section

namespace TorchLean.Test.OpDerivCat

open TorchLean Tensor

variable {s : Shape}

/-- Concatenating two copies of `y * y` along `dim` has the backward pass `B i * y + B i * y`
summed over the parts `i`, for `B` the parts of `catBackward`. -/
theorem catMul_backward (dim : Fin s.rank) (n : Nat) (out : Shape)
    (hout : out = s.set dim (∑ _ : Fin 2, n)) (y : Tensor ℝ (s.set dim n))
    (δ : Tensor ℝ out) :
    DifferentiableAt ℝ (fun z : Tensor ℝ (s.set dim n) =>
        Tensor.cat dim (fun _ : Fin 2 => n) (fun _ => z * z) out hout) y ∧
      ∑ i, (catBackward dim (fun _ : Fin 2 => n) out hout δ i * y +
          catBackward dim (fun _ : Fin 2 => n) out hout δ i * y) =
        (fderiv ℝ (fun z : Tensor ℝ (s.set dim n) =>
          Tensor.cat dim (fun _ : Fin 2 => n) (fun _ => z * z) out hout) y).adjoint δ := by
  op_deriv

/-- Concatenating the multiples `c i • y` along `dim`, for `k` parts of size `n`, has the backward
pass that sums `c i` times part `i` of `catBackward` over the parts `i`. -/
theorem catSmul_backward (dim : Fin s.rank) (n k : Nat) (c : Fin k → ℝ) (out : Shape)
    (hout : out = s.set dim (∑ _ : Fin k, n)) (y : Tensor ℝ (s.set dim n))
    (δ : Tensor ℝ out) :
    DifferentiableAt ℝ (fun z : Tensor ℝ (s.set dim n) =>
        Tensor.cat dim (fun _ : Fin k => n) (fun i => c i • z) out hout) y ∧
      ∑ i, c i • catBackward dim (fun _ : Fin k => n) out hout δ i =
        (fderiv ℝ (fun z : Tensor ℝ (s.set dim n) =>
          Tensor.cat dim (fun _ : Fin k => n) (fun i => c i • z) out hout) y).adjoint δ := by
  op_deriv

/-- The certificate of `Op.cat` for the sizes `[2, 3]` along the axis of `[5]`. -/
example (xs : Values ℝ [.value [2], .value [3]]) (δ : Tensor ℝ [5]) :=
  cat_backward_eq_adjoint (s := [5]) (0 : Fin 1) [2, 3] xs δ

/-- The certificate of `Op.cat` for any list of sizes along the first axis of a shape `[_, 4]`. -/
example (sizes : List Nat) (xs : Values ℝ (sizes.map fun n => .value [n, 4]))
    (δ : Tensor ℝ [sizes.sum, 4]) :=
  cat_backward_eq_adjoint (s := [7, 4]) (0 : Fin 2) sizes xs δ

end TorchLean.Test.OpDerivCat
