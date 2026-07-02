/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/


module

public import NN.Runtime.Autograd.Engine.Core.Neural
public import NN.Spec.Layers.Loss

/-!
# Core Tape Activations and Losses

This file implements activation and loss tape nodes for the backend-independent autograd engine.
Each node records the spec-layer forward value and a backward closure that computes the corresponding
VJP contribution.
-/

@[expose] public section

namespace Runtime
namespace Autograd

open Spec
open Tensor

namespace Tape

/--
Elementwise logistic sigmoid activation.

 This builds a tape node whose forward pass is `Activation.sigmoid_spec`, and whose backward pass
 multiplies the upstream gradient by `Activation.sigmoid_deriv_spec` (i.e. `σ(x) * (1 - σ(x))`,
 pointwise).

 PyTorch comparison: `torch.sigmoid` / `torch.nn.functional.sigmoid`.
 Reference: https://pytorch.org/docs/stable/generated/torch.sigmoid.html
 -/
def sigmoid {α : Type} [Context α] [DecidableEq Shape]
  {s : Shape} (t : Tape α) (xId : Nat) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t) (s:=s) xId
  let y := Activation.sigmoidSpec (α:=α) x
  let node : Node α :=
    { name := some "sigmoid"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        let dsig := Activation.sigmoidDerivSpec (α := α) x
        pure [(xId, AnyTensor.mk (mulSpec dsig dLdy))]
    }
  pure (t.addNode node)

/--
 Elementwise hyperbolic tangent activation.

 Forward uses `Activation.tanh_spec`; backward uses `Activation.tanh_deriv_spec` (pointwise
 derivative, usually `1 - tanh(x)^2`).

 PyTorch comparison: `torch.tanh`.
 Reference: https://pytorch.org/docs/stable/generated/torch.tanh.html
 -/
def tanh {α : Type} [Context α] [DecidableEq Shape]
  {s : Shape} (t : Tape α) (xId : Nat) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t) (s:=s) xId
  let y := Activation.tanhSpec (α:=α) x
  let node : Node α :=
    { name := some "tanh"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        let dtanh := Activation.tanhDerivSpec (α := α) x
        pure [(xId, AnyTensor.mk (mulSpec dtanh dLdy))]
    }
  pure (t.addNode node)

/--
 Softmax along the last axis (recursing over outer dimensions).

 This matches `Activation.softmax_spec` (which applies softmax to the final dimension and recurses
 over earlier dimensions). The backward pass uses the standard Jacobian-vector product implemented
 by `Activation.softmax_backward_spec`, avoiding materializing an `n×n` Jacobian per slice.

 PyTorch comparison: `torch.softmax(x, dim=-1)`.
 Reference: https://pytorch.org/docs/stable/generated/torch.softmax.html
 -/
def softmax {α : Type} [Context α] [DecidableEq Shape]
  {s : Shape} (t : Tape α) (xId : Nat) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t) (s:=s) xId
  let y := Activation.softmaxSpec (α:=α) x
  let node : Node α :=
    { name := some "softmax"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        let dx := Activation.softmaxBackwardSpec (α := α) (s := s) x dLdy
        pure [(xId, AnyTensor.mk dx)]
    }
  pure (t.addNode node)

/--
Stable log-softmax along the last axis.

Unlike `log (softmax x)`, this uses `Activation.logSoftmaxSpec`, i.e. the max-shifted
`x - max(x) - log(sum(exp(x - max(x))))` formulation.  That matches the numerical contract of
`torch.nn.functional.log_softmax` and is the right primitive for cross-entropy on logits.
-/
def logSoftmax {α : Type} [Context α] [DecidableEq Shape]
  {s : Shape} (t : Tape α) (xId : Nat) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t) (s:=s) xId
  let y := Activation.logSoftmaxSpec (α:=α) x
  let node : Node α :=
    { name := some "log_softmax"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        let dx := Activation.logSoftmaxBackwardSpec (α := α) (s := s) y dLdy
        pure [(xId, AnyTensor.mk dx)]
    }
  pure (t.addNode node)

/--
 Elementwise softplus activation.

 Forward uses `Activation.softplus_spec`; backward uses `Activation.softplus_deriv_spec`.

 PyTorch comparison: `torch.nn.functional.softplus`.
 Reference: https://pytorch.org/docs/stable/generated/torch.nn.functional.softplus.html
 -/
def softplus {α : Type} [Context α] [DecidableEq Shape]
  {s : Shape} (t : Tape α) (xId : Nat) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t) (s:=s) xId
  let y := Activation.softplusSpec (α:=α) x
  let node : Node α :=
    { name := some "softplus"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        let dsoft := Activation.softplusDerivSpec (α := α) x
        pure [(xId, AnyTensor.mk (mulSpec dsoft dLdy))]
    }
  pure (t.addNode node)

/--
 Elementwise exponential.

 Forward uses `exp_spec`; backward multiplies by `exp(x)` (pointwise), i.e. `d/dx exp(x) = exp(x)`.

 PyTorch comparison: `torch.exp`.
 Reference: https://pytorch.org/docs/stable/generated/torch.exp.html
 -/
def exp {α : Type} [Context α] [DecidableEq Shape]
  {s : Shape} (t : Tape α) (xId : Nat) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t) (s:=s) xId
  let y := expSpec (α:=α) x
  let node : Node α :=
    { name := some "exp"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        pure [(xId, AnyTensor.mk (mulSpec (expSpec (α := α) x) dLdy))]
    }
  pure (t.addNode node)

/--
 Elementwise natural logarithm.

 Forward uses `log_spec`; backward multiplies by `1/x` (pointwise), i.e. `d/dx log(x) = 1/x`
 (on its mathematical domain; this runtime does not model NaNs/Infs explicitly).

 PyTorch comparison: `torch.log`.
 Reference: https://pytorch.org/docs/stable/generated/torch.log.html
 -/
def log {α : Type} [Context α] [DecidableEq Shape]
  {s : Shape} (t : Tape α) (xId : Nat) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t) (s:=s) xId
  -- `log` is only defined on positive inputs (and `d/dx log(x) = 1/x` blows up as `x → 0⁺`).
  -- Rather than implicitly relying on backend NaN/Inf behavior, we make the precondition explicit
  -- and ask users to opt into `safe_log` when they want epsilon protection.
  if !(allSpec (α := α) (s := s) (fun v => decide (v > (0 : α))) x) then
    throw "autograd: log: input contains values <= 0 (or NaN); use `safe_log` if you want epsilon protection"
  let y := logSpec (α:=α) x
  let node : Node α :=
    { name := some "log"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        pure [(xId, AnyTensor.mk (mulSpec (invSpec (α := α) x) dLdy))]
    }
  pure (t.addNode node)

/--
 Elementwise reciprocal `x ↦ 1/x`.

 Backward implements `d/dx (x⁻¹) = -(x⁻¹)²` (pointwise).

 PyTorch comparison: `torch.reciprocal`.
 Reference: https://pytorch.org/docs/stable/generated/torch.reciprocal.html
 -/
def inv {α : Type} [Context α] [DecidableEq Shape]
  {s : Shape} (t : Tape α) (xId : Nat) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t) (s:=s) xId
  let y := invSpec (α := α) x
  let node : Node α :=
    { name := some "inv"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        -- d/dx (x⁻¹) = -(x⁻¹)²
        let invx := invSpec (α := α) x
        let invx2 := mulSpec invx invx
        let dx := scaleSpec (α := α) (s := s) (mulSpec dLdy invx2) (-1 : α)
        pure [(xId, AnyTensor.mk dx)]
    }
  pure (t.addNode node)

/--
 Elementwise "safe log" that protects against `log(0)` by adding a small `ε` internally.

 This uses `Activation.safe_log_spec` and `Activation.safe_log_deriv_spec`. The exact behavior is
 controlled by the spec-layer definition; conceptually it is similar to `log(x + ε)` used in
 numerically-stable losses.

 PyTorch comparison: commonly written as `torch.log(x + eps)` in user code (there is no single
 dedicated `torch.safe_log` primitive).
 -/
def safeLog {α : Type} [Context α] [DecidableEq Shape]
  {s : Shape} (t : Tape α) (xId : Nat) (ε : α := Numbers.epsilon) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t) (s:=s) xId
  let y := Activation.safeLogSpec (α:=α) x ε
  let node : Node α :=
    { name := some "safe_log"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        let dlog := Activation.safeLogDerivSpec (α := α) x ε
        pure [(xId, AnyTensor.mk (mulSpec dlog dLdy))]
    }
  pure (t.addNode node)

/--
 Reduce-sum over all entries, producing a scalar node.

 Backward replicates the upstream scalar gradient to every entry of the input tensor (i.e.
 `d/dx Σ_i x_i = 1` per coordinate).

 PyTorch comparison: `torch.sum(x)` with `dim=None`.
 Reference: https://pytorch.org/docs/stable/generated/torch.sum.html
 -/
def sum {α : Type} [Add α] [Zero α] [DecidableEq Shape]
  {s : Shape} (t : Tape α) (xId : Nat) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t) (s:=s) xId
  let y : Tensor α Shape.scalar := Tensor.scalar (sumSpec (α:=α) x)
  let node : Node α :=
    { name := some "sum"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := Shape.scalar) dLdyAny
        pure [(xId, AnyTensor.mk (replicate (α := α) (s := s) dLdy))]
    }
  pure (t.addNode node)

/--
 Mean-squared error (MSE) scalar loss with `"mean"` reduction over all entries.

 `mse_spec_basic` is the scalar loss `(Σ_i (yhat_i - target_i)^2) / N`, where
 `N = Spec.meanDenom s`.
 This matches the default reduction of `torch.nn.functional.mse_loss(..., reduction="mean")`.

 For nonempty shapes, `Spec.meanDenom s = Shape.size s`. For empty shapes, TorchLean's totalized
 spec uses denominator `1`; the empty sum and empty gradient then contribute zero.
 -/
def mseSpecBasic {α : Type} [Add α] [Sub α] [Mul α] [Div α] [Zero α] [Coe Nat α]
  {s : Shape} (predicted target : Tensor α s) : α :=
  let diff := subSpec predicted target
  let squared := mulSpec diff diff
  let sum := sumSpec (α:=α) (s:=s) squared
  sum / (Spec.meanDenom s : α)

/--
 Gradient of `mse_spec_basic` with respect to `predicted` (same shape as the inputs).

 If `mse = (Σ_i (yhat_i - target_i)^2) / N`, with `N = Spec.meanDenom s`, then:
 `∂mse/∂yhat = (2/N) * (yhat - target)`.
 -/
def mseDerivSpecBasic {α : Type} [Add α] [Sub α] [Mul α] [Div α] [Zero α] [One α] [Coe Nat α]
  {s : Shape} (predicted target : Tensor α s) : Tensor α s :=
  let diff := subSpec predicted target
  let two : α := (1 : α) + 1
  scaleSpec (α:=α) (s:=s) diff (two / (Spec.meanDenom s : α))

/--
 Tape node for MSE loss with `"mean"` reduction.

 The forward value is a scalar. The backward pass returns gradients for both inputs:
 `dL/dyhat` from `mse_deriv_spec_basic`, and `dL/dtarget = - dL/dyhat`.

 PyTorch comparison: `torch.nn.functional.mse_loss`.
 Reference: https://pytorch.org/docs/stable/generated/torch.nn.functional.mse_loss.html
 -/
def mseLoss {α : Type}
  [Add α] [Sub α] [Mul α] [Div α] [Zero α] [One α] [Coe Nat α] [DecidableEq Shape]
  {s : Shape} (t : Tape α) (yhatId targetId : Nat) : Result (Tape α × Nat) := do
  let yhat ← requireValue (α:=α) (t:=t) (s:=s) yhatId
  let target ← requireValue (α:=α) (t:=t) (s:=s) targetId
  let y : Tensor α Shape.scalar := Tensor.scalar (mseSpecBasic (α:=α) (s:=s) yhat target)
  let node : Node α :=
    { name := some "mse_loss"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [yhatId, targetId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := Shape.scalar) dLdyAny
        let g : α := Tensor.toScalar dLdy
        let dYhat :=
          scaleSpec (α:=α) (s:=s) (mseDerivSpecBasic (α:=α) (s:=s) yhat target) g
        let dTarget : Tensor α s := subSpec (fill (0 : α) s) dYhat
        pure [(yhatId, AnyTensor.mk dYhat), (targetId, AnyTensor.mk dTarget)]
    }
  pure (t.addNode node)
