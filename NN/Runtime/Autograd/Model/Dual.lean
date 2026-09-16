/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tensor.Pack

/-!
# Dual

Dual numbers for forward-mode differentiation.

This runtime-oriented `Dual α` scalar can be used to compute:
- Jacobian-vector products by running a function once with dual inputs, and
- Hessian-vector products by running reverse-mode over dual scalars (forward-over-reverse).

The rules follow the primal branch. Absolute value and guarded square root select zero tangent
at zero; minimum and maximum split ties equally. ReLU separately selects zero at its kink, and
tensor clamp selects zero tangent at either endpoint. These choices agree with the recorded graph
rules.

`detach` removes the tangent and recursively clears differentiation metadata from the primal,
so nested dual computations respect the same boundary. The power rule uses
$d(x^y)=x^y(y'\log x+yx'/x)$ and requires the logarithm and quotient to be defined. Mathematical
derivative theorems state their domain assumptions in the separate `fderiv` proof layer.
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace Model

open Spec TorchLean
open TorchLean TorchLean.Tensor

/--
A dual number $x+\varepsilon\,dx$ used for forward-mode automatic differentiation.

`re` is the primal value and `du` is the tangent (directional derivative). When you run a
computation once on dual inputs, the resulting `du` component computes a Jacobian-vector product.
-/
structure Dual (α : Type) where
  /-- Primal part: the value the ordinary computation would have produced. -/
  re : α
  /-- Tangent part, the coefficient of `ε`. Truncating at `ε² = 0` is what makes the arithmetic
  rules below exactly the chain rule. -/
  du : α
deriving Repr

namespace Dual

/-- Project the primal component of a dual number. -/
@[simp] def primal {α : Type} (x : Dual α) : α := x.re
/-- Project the tangent component of a dual number. -/
@[simp] def tangent {α : Type} (x : Dual α) : α := x.du

/-- Embed a primal value as a dual number with zero tangent. -/
def ofPrimal {α : Type} [Zero α] (x : α) : Dual α := ⟨x, 0⟩
/-- Convenience constructor using the `(primal, tangent)` order. -/
def mk' {α : Type} (x dx : α) : Dual α := ⟨x, dx⟩

end Dual

namespace Dual

/-- `Dual` is inhabited by `default` primal value with zero tangent. -/
instance {α : Type} [Inhabited α] [Zero α] : Inhabited (Dual α) :=
  ⟨⟨default, 0⟩⟩

/-- Zero dual number: $0+\varepsilon\cdot0$. -/
instance {α : Type} [Zero α] : Zero (Dual α) where
  zero := ⟨0, 0⟩
/-- One dual number: $1+\varepsilon\cdot0$. -/
instance {α : Type} [One α] [Zero α] : One (Dual α) where
  one := ⟨1, 0⟩
/-- Negation is componentwise:
$-(x+\varepsilon\,dx)=-x+\varepsilon(-dx)$. -/
instance {α : Type} [Neg α] : Neg (Dual α) where
  neg x := ⟨-x.re, -x.du⟩

/-- Addition is componentwise, so tangents add linearly. -/
instance {α : Type} [Add α] : Add (Dual α) where
  add x y := ⟨x.re + y.re, x.du + y.du⟩
/-- Subtraction is componentwise, so tangents subtract linearly. -/
instance {α : Type} [Sub α] : Sub (Dual α) where
  sub x y := ⟨x.re - y.re, x.du - y.du⟩

/-- Multiplication uses the product rule: $(xy)'=x'y+xy'$. -/
instance {α : Type} [Mul α] [Add α] : Mul (Dual α) :=
  ⟨fun x y => ⟨x.re * y.re, x.du * y.re + x.re * y.du⟩⟩

/-- Evaluate a complete quotient layer while retaining all enclosed range flags. -/
@[inline] def checkedDiv {α : Type}
    (ops : TorchLean.Numeric.QuotientArithmetic α) (x y : Dual α) : Dual α × UInt8 :=
  let (re, a) := ops.divWithFlags x.re y.re
  let (denom, b) := ops.mulWithFlags y.re y.re
  let (left, c) := ops.mulWithFlags x.du y.re
  let (right, d) := ops.mulWithFlags x.re y.du
  let (numerator, e) := ops.subWithFlags left right
  let (du, f) := ops.divWithFlags numerator denom
  (⟨re, du⟩, a ||| b ||| c ||| d ||| e ||| f)

/-- Lift checked arithmetic through both coefficients without discarding inner range flags. -/
@[inline] instance {α : Type} [lower : TorchLean.Numeric.QuotientArithmetic α] :
    TorchLean.Numeric.QuotientArithmetic (Dual α) where
  supported := lower.supported
  depth := lower.depth + 1
  addWithFlags x y :=
    let (re, a) := lower.addWithFlags x.re y.re
    let (du, b) := lower.addWithFlags x.du y.du
    (⟨re, du⟩, a ||| b)
  subWithFlags x y :=
    let (re, a) := lower.subWithFlags x.re y.re
    let (du, b) := lower.subWithFlags x.du y.du
    (⟨re, du⟩, a ||| b)
  mulWithFlags x y :=
    let (re, a) := lower.mulWithFlags x.re y.re
    let (left, b) := lower.mulWithFlags x.du y.re
    let (right, c) := lower.mulWithFlags x.re y.du
    let (du, d) := lower.addWithFlags left right
    (⟨re, du⟩, a ||| b ||| c ||| d)
  divWithFlags x y :=
    checkedDiv lower x y
  encode x := do
    let re ← lower.encode x.re
    let du ← lower.encode x.du
    return .dual re du
  decode | .dual re du => do
    let re ← lower.decode re
    let du ← lower.decode du
    return ⟨re, du⟩
  copyPrimal native exact := ⟨lower.copyPrimal native.re exact.re, exact.du⟩

/--
Division uses the quotient rule $(x/y)'=(x'y-xy')/y^2$ at every nesting depth.

For native binary32/64 coefficient trees, ordinary operations retain this evaluation order.
If any enclosed operation exposes a nonfinite or subnormal value, including a nonzero product
or quotient rounded to zero, we replay the complete quotient over exact rational coefficients.
Every final coefficient is then rounded independently to its native format. A representable
mixed derivative can therefore survive an intermediate overflow or underflow at an inner depth.
Replay requires all input coefficients finite and the denominator's scalar primal nonzero.
Otherwise the native quotient-rule result is retained. An unrepresentable exact coefficient
still rounds to infinity or zero; coefficients are never clamped or deleted.

The scope is one quotient of the given coefficient trees. Rounding before entry is irreversible;
ordinary rounding and cancellation outside a detected range event retain their native semantics.
Carriers without checked coefficient arithmetic retain the previous operational rearrangement.
-/
@[inline] instance {α : Type} [Div α] [Mul α] [Sub α] [Add α] [BEq α] [Zero α]
    [TorchLean.Numeric.QuotientArithmetic α] : Div (Dual α) :=
  ⟨fun x y =>
    let ops : TorchLean.Numeric.QuotientArithmetic (Dual α) := inferInstance
    if ops.supported then
      let (result, replay) :=
        @TorchLean.Numeric.QuotientArithmetic.divChecked _ ops x y
      if replay then
        match TorchLean.Numeric.QuotientArithmetic.replay? ops.encode ops.decode x y with
        | some exact => ops.copyPrimal result exact
        | none => result
      else result
    else
      let quotient := x.re / y.re
      let denom := y.re * y.re
      let tangent := (x.du * y.re - x.re * y.du) / denom
      let needsFallback := denom == 0 || !((denom - denom) == 0) ||
        !((tangent - tangent) == 0)
      let finiteInputs := (x.re - x.re) == 0 && (x.du - x.du) == 0 &&
        (y.re - y.re) == 0 && (y.du - y.du) == 0
      let quotientRoundedAway := quotient == 0 && !(x.re == 0)
      let derivative :=
        if needsFallback && finiteInputs && (quotient - quotient) == 0 &&
            !quotientRoundedAway then
          (x.du - quotient * y.du) / y.re
        else tangent
      ⟨quotient, derivative⟩⟩

/-- Boolean equality compares primals only (tangents are treated as metadata). -/
instance {α : Type} [BEq α] : BEq (Dual α) where
  beq x y := x.re == y.re
/-- Strict order compares primals only. -/
instance {α : Type} [LT α] : LT (Dual α) where
  lt x y := x.re < y.re
/-- Non-strict order compares primals only. -/
instance {α : Type} [LE α] : LE (Dual α) where
  le x y := x.re ≤ y.re

/--
Take the larger primal and its tangent. Equal primals split the tangent equally between the two
arguments, matching the graph's elementwise maximum rule. Each tangent is scaled before the sum:
adding two large finite tangents first can overflow even when their average is representable.
-/
instance {α : Type} [TorchLean.Storage α] [Context α] : Max (Dual α) where
  max x y :=
    if x.re > y.re then x
    else if y.re > x.re then y
    else
      let half : α := 1 / 2
      ⟨Max.max x.re y.re, half * x.du + half * y.du⟩

/-- Elementwise minimum uses the same equal split at a tie as the graph's minimum rule. -/
instance {α : Type} [TorchLean.Storage α] [Context α] : Min (Dual α) where
  min x y :=
    if x.re < y.re then x
    else if y.re < x.re then y
    else
      let half : α := 1 / 2
      ⟨Min.min x.re y.re, half * x.du + half * y.du⟩

/-- Coerce naturals into dual numbers with zero tangent. -/
instance {α : Type} [TorchLean.Storage α] [Context α] : NatCast (Dual α) where
  natCast n := ⟨(n : α), 0⟩

/-- Forward-mode chain rule implementations for `MathFunctions` over dual numbers. -/
instance {α : Type} [TorchLean.Storage α] [Context α] : MathFunctions (Dual α) where
  exp x :=
    let ex := MathFunctions.exp x.re
    ⟨ex, ex * x.du⟩
  tanh x :=
    let th := MathFunctions.tanh x.re
    -- d/dx tanh = 1 - tanh^2
    ⟨th, (1 - th * th) * x.du⟩
  cosh x :=
    let ch := MathFunctions.cosh x.re
    ⟨ch, MathFunctions.sinh x.re * x.du⟩
  sinh x :=
    let sh := MathFunctions.sinh x.re
    ⟨sh, MathFunctions.cosh x.re * x.du⟩
  sqrt x :=
    let r := MathFunctions.sqrt x.re
    ⟨r, if x.re > 0 then x.du / (((2 : Nat) : α) * r) else 0⟩
  abs x :=
    ⟨MathFunctions.abs x.re,
      if x.re > 0 then x.du else if x.re < 0 then -x.du else 0⟩
  log x :=
    ⟨MathFunctions.log x.re, x.du / x.re⟩
  pi := ⟨MathFunctions.pi, 0⟩
  cos x :=
    ⟨MathFunctions.cos x.re, -(MathFunctions.sin x.re) * x.du⟩
  sin x :=
    ⟨MathFunctions.sin x.re, (MathFunctions.cos x.re) * x.du⟩

/--
Power rule for `x^y` over dual numbers.

We use the standard identity $d(x^y)=x^y(y'\log x+yx'/x)$, which is mathematically
justified only on domains where the right-hand side is defined.
-/
instance {α : Type} [TorchLean.Storage α] [Context α] : Pow (Dual α) (Dual α) where
  pow x y :=
    -- d (x^y) = x^y * (y' * log x + y * x'/x)
    let r : α := x.re ^ y.re
    let dr : α := r * (y.du * MathFunctions.log x.re + y.re * (x.du / x.re))
    ⟨r, dr⟩

/-- Lift a scalar `Context` to dual numbers by deciding comparisons on primals. -/
instance {α : Type} [TorchLean.Storage α] [Context α] : Context (Dual α) where
  defaultEpsilon := ⟨Context.defaultEpsilon (α := α), 0⟩
  decidableGT := fun x y => (Context.decidableGT) x.re y.re
  stopGradient? := some fun x => ⟨Context.stopGradient x.re, 0⟩
  ratCast value := ⟨(value : α), 0⟩

end Dual

namespace DualTensor

/-- Map a tensor of primals to a tensor of duals with zero tangents. -/
def ofPrimal {α : Type} [TorchLean.Storage α]
    [TorchLean.Storage (Dual α)] [Zero α] :
    ∀ {s : Shape}, Tensor α s → Tensor (Dual α) s :=
  fun {_s} t => TorchLean.Tensor.map (fun a => Dual.ofPrimal a) t

/-- Lift a tensor pack to dual numbers with zero tangents. -/
def ofPrimalPack {α : Type} [TorchLean.Storage α]
    [TorchLean.Storage (Dual α)] [Zero α] :
    {ss : List Shape} → TorchLean.TensorPack α ss →
      TorchLean.TensorPack (Dual α) ss
  | [], .nil => .nil
  | _ :: ss, .cons x xs => .cons (ofPrimal (s := _) x) (ofPrimalPack (ss := ss) xs)

/--
Combine a primal tensor and a tangent tensor into a dual tensor.

This is the tensor-level analogue of `Dual.mk'`.
-/
def withTangents {α : Type} [TorchLean.Storage α]
    [TorchLean.Storage (Dual α)] [Context α]
    {s : Shape} (primal tangent : Tensor α s) : Tensor (Dual α) s :=
  TorchLean.Tensor.map2Spec Dual.mk' primal tangent

/-- Apply `withTangents` pointwise to a tensor pack. -/
def withTangentsPack {α : Type} [TorchLean.Storage α]
    [TorchLean.Storage (Dual α)] [Context α] :
    {ss : List Shape} →
      TorchLean.TensorPack α ss →
      TorchLean.TensorPack α ss →
      TorchLean.TensorPack (Dual α) ss
  | [], .nil, .nil => .nil
  | _ :: ss, .cons x xs, .cons dx dxs =>
      .cons (withTangents (s := _) x dx) (withTangentsPack (ss := ss) xs dxs)

/-- Project the tangent part of a dual tensor. -/
def tangent {α : Type} [TorchLean.Storage α]
    [TorchLean.Storage (Dual α)] {s : Shape}
    (tensor : Tensor (Dual α) s) : Tensor α s :=
  TorchLean.Tensor.map Dual.tangent tensor

/-- Extract every tangent in a tensor pack. -/
def tangentPack {α : Type} [TorchLean.Storage α]
    [TorchLean.Storage (Dual α)] :
    {ss : List Shape} → TorchLean.TensorPack (Dual α) ss →
      TorchLean.TensorPack α ss
  | [], .nil => .nil
  | _ :: ss, .cons x xs => .cons (tangent (s := _) x) (tangentPack (ss := ss) xs)

end DualTensor

end Model
end Autograd
end Runtime
