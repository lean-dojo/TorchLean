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

/-- Record weighted logits cross-entropy and both pullbacks on the existing tape. -/
def crossEntropy {α : Type} [TorchLean.Storage α] [Context α] {s : Shape}
    (axis : Nat) [Shape.AxisInBounds axis s]
    (t : Tape (Spec.SomeTensor α)) (logitsId targetId : Nat) (mean : Bool := true) :
    Result (Tape (Spec.SomeTensor α) × Nat) :=
  binary (σ₁ := s) (σ₂ := s) (τ := Shape.scalar) t "cross_entropy" logitsId targetId
    (fun logits target => Tensor.scalar (Spec.crossEntropyLogitsSpec axis logits target mean))
    (fun logits target seed =>
      (scaleSpec (Spec.crossEntropyLogitsDerivSpec axis logits target mean) seed.item,
        Spec.CrossEntropy.targetGradient axis logits seed.item mean))

/-- Cross-entropy from a fixed teacher's logits. The student uses the cancellation-resistant
shared gradient; the teacher is a forward dependency, not a differentiable parent. -/
def distill {α : Type} [TorchLean.Storage α] [Context α] {s : Shape}
    (axis : Nat) [Shape.AxisInBounds axis s]
    (t : Tape (Spec.SomeTensor α)) (studentId teacherId : Nat) :
    Result (Tape (Spec.SomeTensor α) × Nat) := do
  let student ← requireValue (α := α) (t := t) (s := s) studentId
  let teacher ← requireValue (α := α) (t := t) (s := s) teacherId
  let node : Node (Spec.SomeTensor α) :=
    { name := some "distill"
      value := Spec.SomeTensor.ofTensor (Tensor.scalar (Spec.distillSpec axis student teacher))
      requiresGrad := (t.getNode? studentId).any (·.requiresGrad)
      parents := #[studentId, teacherId]
      backward := fun seed => do
        let seed ← requireGrad (α := α) (τ := Shape.scalar) seed
        let gradient := scaleSpec (Spec.distillDerivSpec axis student teacher) seed.item
        pure #[(studentId, Spec.SomeTensor.ofTensor gradient)] }
  pure (t.addNode node)

/--
Elementwise logistic sigmoid activation.

 This builds a tape node whose forward pass is `Activation.sigmoidSpec`, and whose backward pass
 applies `Activation.sigmoidBackwardSpec`. It uses the preactivation and retains the incoming
 gradient when repairing a local slope that would round to zero on its own.

 PyTorch comparison: `torch.sigmoid` / `torch.nn.functional.sigmoid`.
 Reference: https://pytorch.org/docs/stable/generated/torch.sigmoid.html
 -/
@[inline] def sigmoid {α : Type} [TorchLean.Storage α] [Context α]
  {s : Shape} (t : Tape (Spec.SomeTensor α)) (xId : Nat) : Result (Tape (Spec.SomeTensor α)
      × Nat) :=
  unary (α := α) (t := t) (σ := s) (τ := s) "sigmoid" xId
    (forward := fun x => Activation.sigmoidSpec (α := α) x)
    (backward := Activation.sigmoidBackwardSpec)

/--
 Elementwise hyperbolic tangent activation.

 Forward uses `Activation.tanhSpec`; backward uses `Activation.tanhBackwardSpec`, retaining the
 incoming gradient through the local slope's range repair.

 PyTorch comparison: `torch.tanh`.
 Reference: https://pytorch.org/docs/stable/generated/torch.tanh.html
 -/
@[inline] def tanh {α : Type} [TorchLean.Storage α] [Context α]
  {s : Shape} (t : Tape (Spec.SomeTensor α)) (xId : Nat) : Result (Tape (Spec.SomeTensor α)
      × Nat) :=
  unary (α := α) (t := t) (σ := s) (τ := s) "tanh" xId
    (forward := fun x => Activation.tanhSpec (α := α) x)
    (backward := Activation.tanhBackwardSpec)

/-- Elementwise SiLU, using the shared split-exponential value and derivative specifications. -/
@[inline] def silu {α : Type} [TorchLean.Storage α] [Context α]
    {s : Shape} (t : Tape (Spec.SomeTensor α)) (xId : Nat) :
    Result (Tape (Spec.SomeTensor α) × Nat) :=
  unary (α := α) (t := t) (σ := s) (τ := s) "silu" xId
    (forward := Activation.swishSpec)
    (backward := fun x seed => mulSpec (Activation.swishDerivSpec x) seed)

/--
Elementwise tanh-approximate GELU.

The tape records GELU as one semantic operation. Its backward closure uses the derivative proved in
`NN.Proofs.Gradients.Activation`; runtime backends may fuse the corresponding pointwise work
without changing this tape-level rule.
-/
@[inline] def gelu {α : Type} [TorchLean.Storage α] [Context α]
  {s : Shape} (t : Tape (Spec.SomeTensor α)) (xId : Nat) : Result (Tape (Spec.SomeTensor α)
      × Nat) :=
  unary (α := α) (t := t) (σ := s) (τ := s) "gelu" xId
    (forward := fun x => Activation.geluSpec (α := α) x)
    (backward := fun x dLdy =>
      let dgelu := Activation.geluDerivSpec (α := α) x
      mulSpec dgelu dLdy)

/--
 Softmax along the last axis (recursing over outer dimensions).

 This is the tape primitive behind the general axis API after it moves the selected axis to the
 innermost position. Its backward pass avoids materializing an `n×n` Jacobian per slice.

 PyTorch comparison: `torch.softmax(x, dim=-1)`.
 Reference: https://pytorch.org/docs/stable/generated/torch.softmax.html
 -/
@[inline] def softmaxLast {α : Type} [TorchLean.Storage α] [Context α]
  {s : Shape} (t : Tape (Spec.SomeTensor α)) (xId : Nat) : Result (Tape (Spec.SomeTensor α)
      × Nat) :=
  unary (α := α) (t := t) (σ := s) (τ := s) "softmax" xId
    (forward := fun x => Activation.Internal.softmaxInnermostSpec (α := α) x)
    (backward := fun x dLdy =>
      Activation.Internal.softmaxInnermostBackwardSpec (α := α) (s := s) x dLdy)

/--
Stable log-softmax along the last axis.

The shared specification separates one maximal exponential from the sum and uses a compensated
`log1p` for the remaining tail. Backward retains the same tail through `expm1`, rather than
subtracting a rounded unit probability from one. This is the primitive used by cross-entropy.
-/
@[inline] def logSoftmaxLast {α : Type} [TorchLean.Storage α] [Context α]
  {s : Shape} (t : Tape (Spec.SomeTensor α)) (xId : Nat) : Result (Tape (Spec.SomeTensor α) × Nat)
      := do
  let x ← requireValue (α:=α) (t:=t) (s:=s) xId
  let y := Activation.Internal.logSoftmaxInnermostSpec (α := α) x
  let node : Node (Spec.SomeTensor α) :=
    { name := some "log_softmax"
      value := Spec.SomeTensor.ofTensor y
      requiresGrad := (t.getNode? xId).any (·.requiresGrad)
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
@[inline] def softplus {α : Type} [TorchLean.Storage α] [Context α]
  {s : Shape} (t : Tape (Spec.SomeTensor α)) (xId : Nat) : Result (Tape (Spec.SomeTensor α)
      × Nat) :=
  unary (α := α) (t := t) (σ := s) (τ := s) "softplus" xId
    (forward := fun x => Activation.softplusSpec (α := α) x)
    (backward := fun x dLdy =>
      let dsoft := Activation.softplusDerivSpec (α := α) x
      mulSpec dsoft dLdy)

/--
 Elementwise exponential.

 Forward uses `expSpec`; backward multiplies by `exp(x)` (pointwise), i.e. `d/dx exp(x) = exp(x)`.

 PyTorch comparison: `torch.exp`.
 Reference: https://pytorch.org/docs/stable/generated/torch.exp.html
 -/
@[inline] def exp {α : Type} [TorchLean.Storage α] [Context α]
  {s : Shape} (t : Tape (Spec.SomeTensor α)) (xId : Nat) : Result (Tape (Spec.SomeTensor α)
      × Nat) :=
  unary (α := α) (t := t) (σ := s) (τ := s) "exp" xId
    (forward := fun x => expSpec (α := α) x)
    (backward := fun x dLdy => mulSpec (expSpec (α := α) x) dLdy)

/--
 Elementwise natural logarithm.

 Forward uses `logSpec`; backward divides the incoming gradient by `x` (pointwise)
 (on its mathematical domain; this runtime does not model NaNs/Infs explicitly).

 PyTorch comparison: `torch.log`.
 Reference: https://pytorch.org/docs/stable/generated/torch.log.html
 -/
@[inline] def log {α : Type} [TorchLean.Storage α] [Context α]
  {s : Shape} (t : Tape (Spec.SomeTensor α)) (xId : Nat) : Result (Tape (Spec.SomeTensor α) × Nat)
      := do
  let x ← requireValue (α:=α) (t:=t) (s:=s) xId
  -- `log` is only defined on positive inputs (and `d/dx log(x) = 1/x` blows up as `x → 0⁺`).
  -- Rather than implicitly relying on backend NaN/Inf behavior, we make the precondition explicit
  -- and point users to `safe_log`, the total `log(softplus(x) + eps)`.
  if !(allSpec (α := α) (s := s) (fun v => decide (v > (0 : α))) x) then
    throw "autograd: log: input contains values <= 0 (or NaN); \
      `safe_log` computes log(softplus(x) + eps) and accepts every input"
  let y := logSpec (α:=α) x
  let node : Node (Spec.SomeTensor α) :=
    { name := some "log"
      value := Spec.SomeTensor.ofTensor y
      requiresGrad := (t.getNode? xId).any (·.requiresGrad)
      parents := #[xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        pure #[(xId, Spec.SomeTensor.ofTensor (divSpec dLdy x))]
    }
  pure (t.addNode node)

/--
 Elementwise reciprocal `x ↦ 1/x`.

 Backward implements `d/dx (x⁻¹) = -(x⁻¹)²` (pointwise).

 PyTorch comparison: `torch.reciprocal`.
 Reference: https://pytorch.org/docs/stable/generated/torch.reciprocal.html
 -/
@[inline] def inv {α : Type} [TorchLean.Storage α] [Context α]
  {s : Shape} (t : Tape (Spec.SomeTensor α)) (xId : Nat) : Result (Tape (Spec.SomeTensor α)
      × Nat) :=
  unary (α := α) (t := t) (σ := s) (τ := s) "inv" xId
    (forward := fun x => invSpec (α := α) x)
    (backward := fun x dLdy =>
      -- Successive divisions avoid overflowing a reciprocal or squaring a large denominator.
      divSpec (divSpec (negSpec dLdy) x) x)

/--
 Elementwise `log(softplus(x) + ε)`.

 The forward value is `Activation.safeLogSpec` and the backward factor is
 `Activation.safeLogDerivSpec`, which is `sigmoid(x) / (softplus(x) + ε)`. Over the reals with
 positive `ε`, the argument stays positive and the value approaches `log ε` as `x` tends to
 negative infinity. Executable arithmetic uses the chosen scalar representation; this operation
 does not validate `ε` or promise finite results for every input.

 PyTorch comparison: `torch.log(torch.nn.functional.softplus(x) + eps)`; PyTorch has no single
 `safe_log` primitive.
 -/
@[inline] def safeLog {α : Type} [TorchLean.Storage α] [Context α]
  {s : Shape} (t : Tape (Spec.SomeTensor α)) (xId : Nat) (ε : α := Context.defaultEpsilon) :
    Result (Tape (Spec.SomeTensor α) × Nat) :=
  unary (α := α) (t := t) (σ := s) (τ := s) "safe_log" xId
    (forward := fun x => Activation.safeLogSpec (α := α) x ε)
    (backward := fun x dLdy =>
      let dlog := Activation.safeLogDerivSpec (α := α) x ε
      mulSpec dlog dLdy)

/--
 Reduce-sum over all entries, producing a scalar node.

 Backward replicates the upstream scalar gradient to every entry of the input tensor (i.e.
 `d/dx Σ_i x_i = 1` per coordinate).

 PyTorch comparison: `torch.sum(x)` with `dim=None`.
 Reference: https://pytorch.org/docs/stable/generated/torch.sum.html
 -/
@[inline] def sum {α : Type} [TorchLean.Storage α] [Add α] [Zero α]
  {s : Shape} (t : Tape (Spec.SomeTensor α)) (xId : Nat) : Result (Tape (Spec.SomeTensor α)
      × Nat) :=
  unary (α := α) (t := t) (σ := s) (τ := .scalar) "sum" xId
    (forward := fun x => Tensor.scalar (sumSpec (α := α) x))
    (backward := fun _x dLdy => replicate (α := α) (shape := s) dLdy)

/--
 Tape node for MSE loss with `"mean"` reduction.

 The forward value is a scalar. The backward pass returns gradients for both inputs:
 `dL/dyhat` from `Spec.mseDerivSpec`, and `dL/dtarget = -dL/dyhat`.

 PyTorch comparison: `torch.nn.functional.mse_loss`.
 Reference: https://pytorch.org/docs/stable/generated/torch.nn.functional.mse_loss.html
 -/
@[inline] def mseLoss {α : Type} [TorchLean.Storage α]
  [Add α] [Sub α] [Mul α] [Div α] [Zero α] [One α] [NatCast α]
  {s : Shape} (t : Tape (Spec.SomeTensor α)) (yhatId targetId : Nat) : Result (Tape
      (Spec.SomeTensor α) × Nat) :=
  binary (α := α) (t := t) (σ₁ := s) (σ₂ := s) (τ := Shape.scalar)
    "mse_loss" yhatId targetId
    (forward := fun yhat target => Tensor.scalar (Spec.mseSpec (α := α) yhat target))
    (backward := fun yhat target dLdy =>
      let dYhat := scaleSpec (Spec.mseDerivSpec (α := α) yhat target) (Tensor.item dLdy)
      (dYhat, subSpec (Tensor.full s (0 : α)) dYhat))
