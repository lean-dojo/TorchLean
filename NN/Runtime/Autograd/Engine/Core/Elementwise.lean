/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/


module

public import NN.Runtime.Autograd.Engine.Core.Base
public import NN.Spec.Autograd.Ops

/-!
Elementwise eager-engine operations.

This file contains scalar-lifted tensor nodes and their runtime/autograd implementation, including
arithmetic, comparisons, activations, and loss-adjacent pointwise operations.
-/

@[expose] public section

namespace Runtime
namespace Autograd

open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace Tape

/-- Elementwise addition. PyTorch: `torch.add` / `+`. -/
@[inline] def add {α : Type} [TorchLean.Storage α] [Add α] {s : Shape}
  (t : Tape α) (aId bId : Nat) : Result (Tape α × Nat) := do
  let a ← requireValue (α:=α) (t:=t) (s:=s) aId
  let b ← requireValue (α:=α) (t:=t) (s:=s) bId
  let y := addSpec a b
  let node : Node α :=
    { name := some "add"
      value := Spec.SomeTensor.ofTensor y
      requiresGrad :=
        (t.getNode? aId).any (·.requiresGrad) ||
        (t.getNode? bId).any (·.requiresGrad)
      parents := #[aId, bId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        pure #[(aId, Spec.SomeTensor.ofTensor dLdy), (bId, Spec.SomeTensor.ofTensor dLdy)]
    }
  pure (t.addNode node)

/-- Elementwise subtraction. PyTorch: `torch.sub` / `-`. -/
@[inline] def sub {α : Type} [TorchLean.Storage α] [Sub α] [Zero α] {s : Shape}
  (t : Tape α) (aId bId : Nat) : Result (Tape α × Nat) := do
  let a ← requireValue (α:=α) (t:=t) (s:=s) aId
  let b ← requireValue (α:=α) (t:=t) (s:=s) bId
  let y := subSpec a b
  let node : Node α :=
    { name := some "sub"
      value := Spec.SomeTensor.ofTensor y
      requiresGrad :=
        (t.getNode? aId).any (·.requiresGrad) ||
        (t.getNode? bId).any (·.requiresGrad)
      parents := #[aId, bId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        let neg_dLdy : Tensor α s := subSpec (Tensor.full s (0 : α)) dLdy
        pure #[(aId, Spec.SomeTensor.ofTensor dLdy), (bId, Spec.SomeTensor.ofTensor neg_dLdy)]
    }
  pure (t.addNode node)

/-- Elementwise multiplication. PyTorch: `torch.mul` / `*`. -/
@[inline] def mul {α : Type} [TorchLean.Storage α] [Mul α] {s : Shape}
  (t : Tape α) (aId bId : Nat) : Result (Tape α × Nat) := do
  let a ← requireValue (α:=α) (t:=t) (s:=s) aId
  let b ← requireValue (α:=α) (t:=t) (s:=s) bId
  let y := mulSpec a b
  let node : Node α :=
    { name := some "mul"
      value := Spec.SomeTensor.ofTensor y
      requiresGrad :=
        (t.getNode? aId).any (·.requiresGrad) ||
        (t.getNode? bId).any (·.requiresGrad)
      parents := #[aId, bId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        let da : Tensor α s := mulSpec dLdy b
        let db : Tensor α s := mulSpec dLdy a
        pure #[(aId, Spec.SomeTensor.ofTensor da), (bId, Spec.SomeTensor.ofTensor db)]
    }
  pure (t.addNode node)

/-- Elementwise division. PyTorch: `torch.div` / `/`. Backward is the ordinary quotient
rule, valid for nonzero denominators: `∂(a/b)/∂a = 1/b`, `∂(a/b)/∂b = −a/b²` (mirrors the
CUDA `div` node; negation subtracts from `Tensor.full s 0` as `sub` does, so no `Neg α` is
required).

Domain: real calculus does not define the derivative of `a/b` at `b = 0`, so this backward is
the genuine quotient rule only where `b ≠ 0`. The carrier's `/` (and hence `divSpec`) may
totalize or be backend-dependent at `b = 0`, but no real-valued gradient is implied there.

Requires `[TorchLean.Storage α] [Context α]` like the sibling `abs`/`sqrt`/`exp` nodes (its
`divSpec` forward rides the carrier's `/`). -/
@[inline] def div {α : Type} [TorchLean.Storage α] [Context α] {s : Shape}
  (t : Tape α) (aId bId : Nat) : Result (Tape α × Nat) := do
  let a ← requireValue (α:=α) (t:=t) (s:=s) aId
  let b ← requireValue (α:=α) (t:=t) (s:=s) bId
  let y := divSpec a b
  let node : Node α :=
    { name := some "div"
      value := Spec.SomeTensor.ofTensor y
      requiresGrad :=
        (t.getNode? aId).any (·.requiresGrad) ||
        (t.getNode? bId).any (·.requiresGrad)
      parents := #[aId, bId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        let da : Tensor α s := divSpec dLdy b
        -- `(a / b) / b` rather than `a / (b * b)`: `b * b` overflows or underflows for
        -- moderate `|b|` in floating point even when `a / b²` is representable.
        let dLdyA : Tensor α s := mulSpec dLdy (divSpec (divSpec a b) b)
        let db : Tensor α s := subSpec (Tensor.full s (0 : α)) dLdyA
        pure #[(aId, Spec.SomeTensor.ofTensor da), (bId, Spec.SomeTensor.ofTensor db)]
    }
  pure (t.addNode node)

/-- Multiply a tensor by a scalar constant. PyTorch: `x * c` for Python scalar `c`. -/
@[inline] def scale {α : Type} [TorchLean.Storage α] [Mul α] {s : Shape}
  (t : Tape α) (xId : Nat) (c : α) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t) (s:=s) xId
  let y := scaleSpec x c
  let node : Node α :=
    { name := some "scale"
      value := Spec.SomeTensor.ofTensor y
      requiresGrad := (t.getNode? xId).any (·.requiresGrad)
      parents := #[xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        pure #[(xId, Spec.SomeTensor.ofTensor (scaleSpec dLdy c))]
    }
  pure (t.addNode node)

/--
Elementwise absolute value.

Backward uses the sign function (`signSpec`) as a subgradient at `0`.
PyTorch comparison: `torch.abs`.
-/
@[inline] def abs {α : Type} [TorchLean.Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {s : Shape} (t : Tape α) (xId : Nat) : Result (Tape α × Nat) :=
  unary (α := α) (t := t) (σ := s) (τ := s)
    "abs" xId
    (forward := fun x => absSpec (α := α) (s := s) x)
    (backward := fun x dLdy =>
      let dabs : Tensor α s := signSpec (α := α) (s := s) x
      mulSpec dabs dLdy)

/--
Elementwise square root.

Backward uses `1 / (2 * sqrt(x))` for `x > 0` and `0` otherwise (totalized).
PyTorch comparison: `torch.sqrt`.
-/
@[inline] def sqrt {α : Type} [TorchLean.Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {s : Shape} (t : Tape α) (xId : Nat) : Result (Tape α × Nat) :=
  unary (α := α) (t := t) (σ := s) (τ := s)
    "sqrt" xId
    (forward := fun x => sqrtSpec (α := α) (s := s) x)
    (backward := fun x dLdy =>
      let dsqrt : Tensor α s :=
        mapSpec (α := α) (s := s) (fun v =>
          if v > 0 then
            (1 : α) / (((2 : Nat) : α) * MathFunctions.sqrt v)
          else
            (0 : α)) x
      mulSpec dsqrt dLdy)

/--
Elementwise clamp to `[minVal, maxVal]`.

Backward multiplies by an indicator of the open interval `(minVal, maxVal)` (zero at boundaries).
PyTorch comparison: `torch.clamp`.
-/
@[inline] def clamp {α : Type} [TorchLean.Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {s : Shape} (t : Tape α) (xId : Nat) (minVal maxVal : α) : Result (Tape α × Nat) :=
  unary (α := α) (t := t) (σ := s) (τ := s)
    "clamp" xId
    (forward := fun x => clampSpec (α := α) (s := s) x minVal maxVal)
    (backward := fun x dLdy =>
      let dclamp : Tensor α s :=
        mapSpec (α := α) (s := s) (fun v =>
          if v > minVal ∧ maxVal > v then (1 : α) else (0 : α)) x
      mulSpec dclamp dLdy)

/--
Elementwise maximum.

Tie-breaking: when `a = b`, the upstream gradient is split evenly (`0.5`) between both inputs.
PyTorch comparison: `torch.maximum`.
-/
@[inline] def max {α : Type} [TorchLean.Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {s : Shape} (t : Tape α) (aId bId : Nat) : Result (Tape α × Nat) := do
  let a ← requireValue (α:=α) (t:=t) (s:=s) aId
  let b ← requireValue (α:=α) (t:=t) (s:=s) bId
  let y := maxSpec (α := α) (s := s) a b
  let node : Node α :=
    { name := some "max"
      value := Spec.SomeTensor.ofTensor y
      requiresGrad :=
        (t.getNode? aId).any (·.requiresGrad) ||
        (t.getNode? bId).any (·.requiresGrad)
      parents := #[aId, bId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        pure #[
          (aId, Spec.SomeTensor.ofTensor ((Spec.maxOp b).backward a dLdy)),
          (bId, Spec.SomeTensor.ofTensor ((Spec.maxOp a).backward b dLdy))
        ]
    }
  pure (t.addNode node)

/--
Elementwise minimum.

Tie-breaking: when `a = b`, the upstream gradient is split evenly (`0.5`) between both inputs.
PyTorch comparison: `torch.minimum`.
-/
@[inline] def min {α : Type} [TorchLean.Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {s : Shape} (t : Tape α) (aId bId : Nat) : Result (Tape α × Nat) := do
  let a ← requireValue (α:=α) (t:=t) (s:=s) aId
  let b ← requireValue (α:=α) (t:=t) (s:=s) bId
  let y := minSpec (α := α) (s := s) a b
  let node : Node α :=
    { name := some "min"
      value := Spec.SomeTensor.ofTensor y
      requiresGrad :=
        (t.getNode? aId).any (·.requiresGrad) ||
        (t.getNode? bId).any (·.requiresGrad)
      parents := #[aId, bId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        pure #[
          (aId, Spec.SomeTensor.ofTensor ((Spec.minOp b).backward a dLdy)),
          (bId, Spec.SomeTensor.ofTensor ((Spec.minOp a).backward b dLdy))
        ]
    }
  pure (t.addNode node)

/--
Record elementwise sine with the VJP from `Spec.sinOp`.

The tape retains the input for `cos(x) * dLdy`, so the backward pass uses the same angle as
the forward pass even when different angles produce the same sine value.
-/
@[inline] def sin {α : Type} [TorchLean.Storage α] [Context α]
    {s : Shape} (t : Tape α) (xId : Nat) : Result (Tape α × Nat) :=
  unary (α := α) (t := t) (σ := s) (τ := s) "sin" xId
    (forward := (Spec.sinOp (α := α) (s := s)).forward)
    (backward := (Spec.sinOp (α := α) (s := s)).backward)

/-- Record elementwise cosine with the VJP `-sin(x) * dLdy` from `Spec.cosOp`. -/
@[inline] def cos {α : Type} [TorchLean.Storage α] [Context α]
    {s : Shape} (t : Tape α) (xId : Nat) : Result (Tape α × Nat) :=
  unary (α := α) (t := t) (σ := s) (τ := s) "cos" xId
    (forward := (Spec.cosOp (α := α) (s := s)).forward)
    (backward := (Spec.cosOp (α := α) (s := s)).backward)

/--
Elementwise ReLU.

PyTorch comparison: `torch.relu(x)` / `torch.nn.functional.relu(x)`.
-/
@[inline] def relu {α : Type} [TorchLean.Storage α]
  [Mul α] [Zero α] [Max α] [BEq α] [One α] [LT α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {s : Shape} (t : Tape α) (xId : Nat) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t) (s:=s) xId
  let y := Activation.reluSpec (α:=α) x
  let node : Node α :=
    { name := some "relu"
      value := Spec.SomeTensor.ofTensor y
      requiresGrad := (t.getNode? xId).any (·.requiresGrad)
      parents := #[xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        let drelu := Activation.reluDerivSpec (α:=α) x
        pure #[(xId, Spec.SomeTensor.ofTensor (mulSpec drelu dLdy))]
    }
  pure (t.addNode node)
