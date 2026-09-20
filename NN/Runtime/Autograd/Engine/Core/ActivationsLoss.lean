/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/


module

public import NN.Runtime.Autograd.Engine.Core.Base
public import NN.Spec.Layers.Loss

/-!
# Core Tape Activations and Losses

This file implements activation and loss tape nodes for the backend-independent autograd engine.
Each node records the spec-layer forward value and a backward closure that computes the
corresponding VJP contribution.
-/

@[expose] public section

namespace Runtime
namespace Autograd

open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace Tape

/--
Elementwise logistic sigmoid activation.

 This builds a tape node whose forward pass is `Activation.sigmoidSpec`, and whose backward pass
 multiplies the upstream gradient by `Activation.sigmoidDerivSpec` (i.e. `σ(x) * (1 - σ(x))`,
 pointwise).

 PyTorch comparison: `torch.sigmoid` / `torch.nn.functional.sigmoid`.
 Reference: https://pytorch.org/docs/stable/generated/torch.sigmoid.html
 -/
def sigmoid {α : Type} [TorchLean.Storage α] [Context α]
  {s : Shape} (t : Tape α) (xId : Nat) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t) (s:=s) xId
  let y := Activation.sigmoidSpec (α:=α) x
  let node : Node α :=
    { name := some "sigmoid"
      value := Spec.SomeTensor.ofTensor y
      requiresGrad := true
      parents := #[xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        let dsig := Activation.sigmoidDerivSpec (α := α) x
        pure #[(xId, Spec.SomeTensor.ofTensor (mulSpec dsig dLdy))]
    }
  pure (t.addNode node)

/--
 Elementwise hyperbolic tangent activation.

 Forward uses `Activation.tanhSpec`; backward uses `Activation.tanhDerivSpec` (pointwise
 derivative, usually `1 - tanh(x)^2`).

 PyTorch comparison: `torch.tanh`.
 Reference: https://pytorch.org/docs/stable/generated/torch.tanh.html
 -/
def tanh {α : Type} [TorchLean.Storage α] [Context α]
  {s : Shape} (t : Tape α) (xId : Nat) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t) (s:=s) xId
  let y := Activation.tanhSpec (α:=α) x
  let node : Node α :=
    { name := some "tanh"
      value := Spec.SomeTensor.ofTensor y
      requiresGrad := true
      parents := #[xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        let dtanh := Activation.tanhDerivSpec (α := α) x
        pure #[(xId, Spec.SomeTensor.ofTensor (mulSpec dtanh dLdy))]
    }
  pure (t.addNode node)

/--
Elementwise tanh-approximate GELU.

The tape records GELU as one semantic operation. Its backward closure uses the derivative proved in
`NN.Proofs.Gradients.Activation`; runtime backends may fuse the corresponding pointwise work
without changing this tape-level rule.
-/
def gelu {α : Type} [TorchLean.Storage α] [Context α]
  {s : Shape} (t : Tape α) (xId : Nat) : Result (Tape α × Nat) := do
  let x ← requireValue (α := α) (t := t) (s := s) xId
  let y := Activation.geluSpec (α := α) x
  let node : Node α :=
    { name := some "gelu"
      value := Spec.SomeTensor.ofTensor y
      requiresGrad := true
      parents := #[xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        let dgelu := Activation.geluDerivSpec (α := α) x
        pure #[(xId, Spec.SomeTensor.ofTensor (mulSpec dgelu dLdy))]
    }
  pure (t.addNode node)

/--
 Softmax along the last axis (recursing over outer dimensions).

 This is the tape primitive behind the general axis API after it moves the selected axis to the
 innermost position. Its backward pass avoids materializing an `n×n` Jacobian per slice.

 PyTorch comparison: `torch.softmax(x, dim=-1)`.
 Reference: https://pytorch.org/docs/stable/generated/torch.softmax.html
 -/
def softmaxLast {α : Type} [TorchLean.Storage α] [Context α]
  {s : Shape} (t : Tape α) (xId : Nat) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t) (s:=s) xId
  let y := Activation.Internal.softmaxInnermostSpec (α := α) x
  let node : Node α :=
    { name := some "softmax"
      value := Spec.SomeTensor.ofTensor y
      requiresGrad := true
      parents := #[xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        let dx := Activation.Internal.softmaxInnermostBackwardSpec (α := α) (s := s) x dLdy
        pure #[(xId, Spec.SomeTensor.ofTensor dx)]
    }
  pure (t.addNode node)

/--
Stable log-softmax along the last axis.

Unlike `log (softmax x)`, this uses the max-shifted
`x - max(x) - log(sum(exp(x - max(x))))` formulation.  That matches the numerical contract of
`torch.nn.functional.log_softmax` and is the right primitive for cross-entropy on logits.
-/
def logSoftmaxLast {α : Type} [TorchLean.Storage α] [Context α]
  {s : Shape} (t : Tape α) (xId : Nat) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t) (s:=s) xId
  let y := Activation.Internal.logSoftmaxInnermostSpec (α := α) x
  let node : Node α :=
    { name := some "log_softmax"
      value := Spec.SomeTensor.ofTensor y
      requiresGrad := true
      parents := #[xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        let dx := Activation.Internal.logSoftmaxInnermostBackwardSpec
          (α := α) (s := s) y dLdy
        pure #[(xId, Spec.SomeTensor.ofTensor dx)]
    }
  pure (t.addNode node)

/--
 Elementwise softplus activation.

 Forward uses `Activation.softplusSpec`; backward uses `Activation.softplusDerivSpec`.

 PyTorch comparison: `torch.nn.functional.softplus`.
 Reference: https://pytorch.org/docs/stable/generated/torch.nn.functional.softplus.html
 -/
def softplus {α : Type} [TorchLean.Storage α] [Context α]
  {s : Shape} (t : Tape α) (xId : Nat) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t) (s:=s) xId
  let y := Activation.softplusSpec (α:=α) x
  let node : Node α :=
    { name := some "softplus"
      value := Spec.SomeTensor.ofTensor y
      requiresGrad := true
      parents := #[xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        let dsoft := Activation.softplusDerivSpec (α := α) x
        pure #[(xId, Spec.SomeTensor.ofTensor (mulSpec dsoft dLdy))]
    }
  pure (t.addNode node)

/--
 Elementwise exponential.

 Forward uses `expSpec`; backward multiplies by `exp(x)` (pointwise), i.e. `d/dx exp(x) = exp(x)`.

 PyTorch comparison: `torch.exp`.
 Reference: https://pytorch.org/docs/stable/generated/torch.exp.html
 -/
def exp {α : Type} [TorchLean.Storage α] [Context α]
  {s : Shape} (t : Tape α) (xId : Nat) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t) (s:=s) xId
  let y := expSpec (α:=α) x
  let node : Node α :=
    { name := some "exp"
      value := Spec.SomeTensor.ofTensor y
      requiresGrad := true
      parents := #[xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        pure #[(xId, Spec.SomeTensor.ofTensor (mulSpec (expSpec (α := α) x) dLdy))]
    }
  pure (t.addNode node)

/--
 Elementwise natural logarithm.

 Forward uses `logSpec`; backward multiplies by `1/x` (pointwise), i.e. `d/dx log(x) = 1/x`
 (on its mathematical domain; this runtime does not model NaNs/Infs explicitly).

 PyTorch comparison: `torch.log`.
 Reference: https://pytorch.org/docs/stable/generated/torch.log.html
 -/
def log {α : Type} [TorchLean.Storage α] [Context α]
  {s : Shape} (t : Tape α) (xId : Nat) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t) (s:=s) xId
  -- `log` is only defined on positive inputs (and `d/dx log(x) = 1/x` blows up as `x → 0⁺`).
  -- Rather than implicitly relying on backend NaN/Inf behavior, we make the precondition explicit
  -- and ask users to opt into `safe_log` when they want epsilon protection.
  if !(allSpec (α := α) (s := s) (fun v => decide (v > (0 : α))) x) then
    throw "autograd: log: input contains values <= 0 (or NaN); \
      use `safe_log` if you want epsilon protection"
  let y := logSpec (α:=α) x
  let node : Node α :=
    { name := some "log"
      value := Spec.SomeTensor.ofTensor y
      requiresGrad := true
      parents := #[xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        pure #[(xId, Spec.SomeTensor.ofTensor (mulSpec (invSpec (α := α) x) dLdy))]
    }
  pure (t.addNode node)

/--
 Elementwise reciprocal `x ↦ 1/x`.

 Backward implements `d/dx (x⁻¹) = -(x⁻¹)²` (pointwise).

 PyTorch comparison: `torch.reciprocal`.
 Reference: https://pytorch.org/docs/stable/generated/torch.reciprocal.html
 -/
def inv {α : Type} [TorchLean.Storage α] [Context α]
  {s : Shape} (t : Tape α) (xId : Nat) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t) (s:=s) xId
  let y := invSpec (α := α) x
  let node : Node α :=
    { name := some "inv"
      value := Spec.SomeTensor.ofTensor y
      requiresGrad := true
      parents := #[xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        -- d/dx (x⁻¹) = -(x⁻¹)²
        let invx := invSpec (α := α) x
        let invx2 := mulSpec invx invx
        let dx := scaleSpec (α := α) (s := s) (mulSpec dLdy invx2) (-1 : α)
        pure #[(xId, Spec.SomeTensor.ofTensor dx)]
    }
  pure (t.addNode node)

/--
 Elementwise "safe log" that protects against `log(0)` by adding a small `ε` internally.

 This uses `Activation.safeLogSpec` and `Activation.safeLogDerivSpec`. The exact behavior is
 controlled by the spec-layer definition; conceptually it is similar to `log(x + ε)` used in
 numerically-stable losses.

 PyTorch comparison: commonly written as `torch.log(x + eps)` in user code (there is no single
 dedicated `torch.safe_log` primitive).
 -/
def safeLog {α : Type} [TorchLean.Storage α] [Context α]
  {s : Shape} (t : Tape α) (xId : Nat) (ε : α := Context.defaultEpsilon) :
    Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t) (s:=s) xId
  let y := Activation.safeLogSpec (α:=α) x ε
  let node : Node α :=
    { name := some "safe_log"
      value := Spec.SomeTensor.ofTensor y
      requiresGrad := true
      parents := #[xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        let dlog := Activation.safeLogDerivSpec (α := α) x ε
        pure #[(xId, Spec.SomeTensor.ofTensor (mulSpec dlog dLdy))]
    }
  pure (t.addNode node)

/--
 Reduce-sum over all entries, producing a scalar node.

 Backward replicates the upstream scalar gradient to every entry of the input tensor (i.e.
 `d/dx Σ_i x_i = 1` per coordinate).

 PyTorch comparison: `torch.sum(x)` with `dim=None`.
 Reference: https://pytorch.org/docs/stable/generated/torch.sum.html
 -/
def sum {α : Type} [TorchLean.Storage α] [Add α] [Zero α]
  {s : Shape} (t : Tape α) (xId : Nat) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t) (s:=s) xId
  let y : Tensor α .scalar := Tensor.scalar (sumSpec (α:=α) x)
  let node : Node α :=
    { name := some "sum"
      value := Spec.SomeTensor.ofTensor y
      requiresGrad := true
      parents := #[xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := Shape.scalar) dLdyAny
        pure #[(xId, Spec.SomeTensor.ofTensor (replicate (α := α) (shape := s) dLdy))]
    }
  pure (t.addNode node)

/--
 Tape node for MSE loss with `"mean"` reduction.

 The forward value is a scalar. The backward pass returns gradients for both inputs:
 `dL/dyhat` from `Spec.mseDerivSpec`, and `dL/dtarget = -dL/dyhat`.

 PyTorch comparison: `torch.nn.functional.mse_loss`.
 Reference: https://pytorch.org/docs/stable/generated/torch.nn.functional.mse_loss.html
 -/
def mseLoss {α : Type} [TorchLean.Storage α]
  [Add α] [Sub α] [Mul α] [Div α] [Zero α] [One α] [NatCast α]
  {s : Shape} (t : Tape α) (yhatId targetId : Nat) : Result (Tape α × Nat) := do
  let yhat ← requireValue (α:=α) (t:=t) (s:=s) yhatId
  let target ← requireValue (α:=α) (t:=t) (s:=s) targetId
  let y : Tensor α .scalar := Tensor.scalar (Spec.mseSpec (α := α) yhat target)
  let node : Node α :=
    { name := some "mse_loss"
      value := Spec.SomeTensor.ofTensor y
      requiresGrad := true
      parents := #[yhatId, targetId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := Shape.scalar) dLdyAny
        let g : α := Tensor.item dLdy
        let dYhat :=
          scaleSpec (α := α) (s := s) (Spec.mseDerivSpec (α := α) yhat target) g
        let dTarget : Tensor α s := subSpec (Tensor.full s (0 : α)) dYhat
        pure #[(yhatId, Spec.SomeTensor.ofTensor dYhat),
          (targetId, Spec.SomeTensor.ofTensor dTarget)]
    }
  pure (t.addNode node)
